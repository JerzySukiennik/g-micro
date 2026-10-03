"""G-Weird Live on the Mac: draw from words, then redraw whatever was painted over.

A masked model, not an autoregressive one. It fills the whole 24x24 token grid in
twelve parallel rounds instead of 576 sequential passes, and because it already
works by hiding tokens and filling them back in, "redraw this region" is the
same operation as "draw this picture" with part of the grid left visible.

**What the editor does and does not do, measured on the 50000-step checkpoint:**
tokens outside the painted cells come back unchanged (the sampler checks it), the
region is refilled plausibly under the original caption, but a NEW caption mostly
does not change what is drawn there — the context wins. Training did not yet
include contiguous holes. This module does not hide that; the page says so.

Loaded by file path through weird.py's loader, not by importing G-Weird's `model`
package: three projects in this runtime ship one, and a leak of it once broke
every photo edit for the life of the process.
"""

import base64
import io
import json
import math
from pathlib import Path

from runtime.weird import G_WEIRD, _load_module, _load_tokenizer, usable

# Weights only, 248 MB. The training checkpoint is 745 MB because it carries the
# optimiser state, which is useful for resuming on Kaggle and dead weight here.
CKPT = G_WEIRD / "run-live/maskgit.pt"
VQVAE = G_WEIRD / "run11/vqvae.pt"            # the 192px tokenizer, grid 24x24
TOKENIZER = G_WEIRD / "data/text-12.json"

GRID = 24
CELL = 8                                       # pixels per token: 192 / 24
RES = GRID * CELL
CELLS = GRID * GRID
MASK_HEX_LEN = CELLS // 4

# The original MaskGIT settings. 0.7 / 24 rounds / guidance 2 were tuned in
# September on a sampler whose Gumbel noise was NaN for every input, and gave
# paler pictures than these on the same checkpoint.
STEPS = 12
SCALE = 4.0
TEMP = 1.0


class Stopped(Exception):
    """Raised from the per-round callback when the stop button was pressed."""


def parse_mask(hex_str):
    """144 hex characters -> 576 booleans, row-major, first cell = high bit.

    The page writes the same layout: cell k lives in nibble k // 4, and cell 4j
    is that nibble's most significant bit. One bit per grid cell, so what the
    person sees painted is exactly what is redrawn, with no hidden widening.
    """
    # Zdanie dla czlowieka, nie opis formatu: strona sama sklada maske, wiec to
    # zobaczy tylko ktos, kto wyslal zadanie recznie albo ze starej wersji strony,
    # i dla niego jedyna uzyteczna rada to zaznaczyc obszar od nowa.
    bad = "Zaznaczenie jest nieprawidłowe. Zaznacz obszar jeszcze raz."
    s = (hex_str or "").strip().lower()
    if len(s) != MASK_HEX_LEN:
        raise ValueError(bad)
    try:
        nibbles = [int(c, 16) for c in s]
    except ValueError:
        raise ValueError(bad) from None
    bits = []
    for n in nibbles:
        bits.extend(bool((n >> (3 - b)) & 1) for b in range(4))
    return bits


def parse_job_text(text):
    """The page sends plain text to draw, JSON to edit: {"p": prompt, "m": mask, "s": seed}.

    The mask rides inside `text` because the database rules allow exactly the
    fields model, text, image, at, history and cancel on a job, and a new field
    would mean redeploying rules that gate every other model on the site.
    """
    t = (text or "").strip()
    if not t.startswith("{"):
        return {"prompt": t, "mask": None, "seed": None}
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        return {"prompt": t, "mask": None, "seed": None}
    return {"prompt": str(obj.get("p", "")).strip(),
            "mask": obj.get("m") or None,
            "seed": obj.get("s")}


def decode_png(data_url_or_bytes):
    from PIL import Image
    if isinstance(data_url_or_bytes, (bytes, bytearray)):
        raw = bytes(data_url_or_bytes)
    else:
        s = data_url_or_bytes or ""
        raw = base64.b64decode(s.split(",", 1)[1] if "," in s else s)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def to_data_url(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class LiveModel:
    """Lazily loaded: constructing this only stats paths."""

    def __init__(self):
        self.net = None
        self.cfg = None
        self.vq = None
        self.tok = None
        self.mg = None
        self.step = None

    def available(self):
        return usable(CKPT) and usable(VQVAE) and TOKENIZER.is_file()

    def load(self):
        if self.net is not None:
            return {"step": self.step}
        if not self.available():
            raise RuntimeError("brak wag G-Weird Live")
        import torch
        self.mg = _load_module("model/maskgit.py", "_gweird_maskgit")
        vqmod = _load_module("model/vqvae.py", "_gweird_vqvae")

        ck = torch.load(str(CKPT), map_location="cpu", weights_only=False)
        cfg = self.mg.MaskGITConfig(**{k: v for k, v in ck["cfg"].items()
                                       if k in self.mg.MaskGITConfig.__dataclass_fields__})
        if cfg.image_len != CELLS:
            raise RuntimeError(f"checkpoint ma siatke {cfg.image_len}, strona zaklada {CELLS}")
        net = self.mg.MaskGIT(cfg)
        net.load_state_dict(ck["model"])
        net.eval()

        vk = torch.load(str(VQVAE), map_location="cpu", weights_only=False)
        vq = vqmod.VQVAE(**vk["arch"])
        vq.load_state_dict(vk["model"], strict=False)
        vq.eval()

        self.cfg, self.net, self.vq = cfg, net, vq
        self.tok = _load_tokenizer(TOKENIZER)
        self.step = ck.get("step", 0)
        return {"step": self.step}

    # ------------------------------------------------------------- helpers --

    def _text_rows(self, prompt):
        import torch
        cfg = self.cfg
        ids = self.tok.encode(prompt).ids[:cfg.text_len]
        row = [cfg.text_token(i) for i in ids] + [cfg.PAD] * (cfg.text_len - len(ids))
        return torch.tensor([row], dtype=torch.long)

    def _decode(self, codes):
        """(1, 576) codebook ids -> PIL image at 192px."""
        import numpy as np
        import torch
        from PIL import Image
        with torch.no_grad():
            img = self.vq.decode(codes.view(-1, GRID, GRID))
        arr = ((img.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).numpy()[0]
        return Image.fromarray(arr)

    def _round_cb(self, on_progress, should_stop):
        def cb(done, total):
            if should_stop is not None and should_stop():
                raise Stopped()
            if on_progress is not None:
                on_progress(done / total)
        return cb

    # ---------------------------------------------------------------- draw --

    def draw(self, prompt, seed=None, on_progress=None, should_stop=None):
        """One picture from words alone, as a PNG data URL."""
        import os
        import torch
        self.load()
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big")
        torch.manual_seed(int(seed))
        codes = self.mg.generate(self.net, self._text_rows(prompt), self.cfg,
                                 steps=STEPS, scale=SCALE, temp=TEMP,
                                 on_round=self._round_cb(on_progress, should_stop))
        return to_data_url(self._decode(codes))

    # ---------------------------------------------------------------- edit --

    def edit(self, image, mask_hex, prompt, seed=None, on_progress=None,
             should_stop=None):
        """Redraw the painted cells of `image`, keep every other pixel as it was.

        The caption describes the WHOLE picture as it should look afterwards, not
        just the new part: that is how the model was trained, and it is how the
        measured behaviour on the 50000-step checkpoint was obtained.
        """
        import os
        import numpy as np
        import torch
        from PIL import Image, ImageFilter
        self.load()
        cells = parse_mask(mask_hex)
        n_cells = sum(cells)
        if n_cells == 0:
            raise ValueError("Nic nie zaznaczono.")
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big")
        torch.manual_seed(int(seed))

        src = image if hasattr(image, "resize") else decode_png(image)
        if src.size != (RES, RES):
            src = src.resize((RES, RES), Image.LANCZOS)
        x = torch.from_numpy(np.asarray(src)).permute(2, 0, 1)[None].float() / 127.5 - 1.0
        with torch.no_grad():
            idx = self.vq.encode(x)                                 # (1, 24, 24)
        lo = self.cfg.image_token(0)
        init = idx.view(1, -1) + lo
        hole = torch.tensor(cells, dtype=torch.bool)
        init[:, hole] = self.cfg.MASK

        codes = self.mg.generate(self.net, self._text_rows(prompt), self.cfg,
                                 steps=STEPS, scale=SCALE, temp=TEMP, init=init,
                                 on_round=self._round_cb(on_progress, should_stop))
        # The guarantee the whole feature rests on, checked rather than assumed.
        if not bool((codes[:, ~hole] == idx.view(1, -1)[:, ~hole]).all()):
            raise RuntimeError("sampler zmienil tokeny poza zaznaczeniem")
        painted = self._decode(codes)

        # Paste the original pixels back everywhere outside the painted cells, so
        # the rest of the picture is not softened by a second trip through the
        # tokenizer. Blended over a thin band INSIDE the region to hide the seam,
        # and multiplied by the cell mask so nothing outside it can change.
        grid = np.array(cells, dtype=np.uint8).reshape(GRID, GRID) * 255
        up = Image.fromarray(grid).resize((RES, RES), Image.NEAREST)
        soft = up.filter(ImageFilter.MinFilter(5)).filter(ImageFilter.GaussianBlur(1.5))
        alpha = (np.asarray(soft, dtype=np.float32) / 255.0) \
            * (np.asarray(up, dtype=np.float32) / 255.0)
        a = alpha[..., None]
        out = (np.asarray(painted, dtype=np.float32) * a
               + np.asarray(src, dtype=np.float32) * (1.0 - a))
        return to_data_url(Image.fromarray(out.round().clip(0, 255).astype(np.uint8)))
