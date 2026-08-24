"""G-Weird on the Mac: type a prompt, get a picture.

Generation is ~7 s on CPU for one 256x256 image, so this never needed a GPU and
never needed the browser to download 300 MB of weights.

**The sampling loop is duplicated here rather than imported from G-Weird's
train/sample.py, on purpose.** That module puts the project root on sys.path and
does `from model.transformer import ...`, and three projects in this runtime ship
a `model/` package. The last time a G-Doodle import leaked its `data` package
into sys.modules, every photo edit failed with "No module named 'model.unet'"
for the rest of the process's life — see the comment in doodle.py. Thirty lines
copied is a smaller price than re-opening that class of bug, and the copy is
pinned to a frozen checkpoint that will not drift.
"""

import base64
import io
import math
from pathlib import Path

G_WEIRD = Path.home() / "Downloads/Claude/Projects/AIe/G-Weird"

CKPT = G_WEIRD / "run/gweird.pt"
TOKENIZER = G_WEIRD / "run/text.json"

# Both versions share ONE transformer and one codebook — they differ only in the
# decoder that turns codes into pixels. That is exactly why swapping is safe:
# the code ids mean the same thing to both, so 0.9 and 1 are the same picture
# rendered two ways rather than two models.
#
#   0.9  the original VQ-VAE decoder: soft, oil-painted, but coherent.
#   1    8000 steps of adversarial fine-tuning with the encoder and codebook
#        frozen. Reconstruction error 10.3/255 against 15.7, and detail where
#        there was smear — at the cost of a fine crackle over everything.
#
# Which looks better is a matter of taste, which is why both stay reachable.
DECODERS = {
    "g-weird": G_WEIRD / "run/vqvae.pt",
    "g-weird-1": G_WEIRD / "run/decoder.pt",
}
VQVAE = DECODERS["g-weird"]

MIN_CKPT_BYTES = 10_000_000


def usable(path) -> bool:
    try:
        return path.is_file() and path.stat().st_size >= MIN_CKPT_BYTES
    except OSError:
        return False


def _load_module(rel, name):
    """Import a G-Weird module by file path, under a private name.

    No sys.path manipulation at all: the two modules loaded here (the
    transformer and the VQ-VAE) import nothing from their own project, so they
    do not need it — and not touching sys.path is what keeps G-Images and
    G-Doodle working in the same process.
    """
    import importlib.util
    import sys as _sys

    if name in _sys.modules:
        return _sys.modules[name]
    before = set(_sys.modules)
    path_before = list(_sys.path)
    try:
        spec = importlib.util.spec_from_file_location(name, G_WEIRD / rel)
        mod = importlib.util.module_from_spec(spec)
        _sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        for key in set(_sys.modules) - before - {name}:
            origin = getattr(_sys.modules[key], "__file__", "") or ""
            if str(G_WEIRD) in origin:
                del _sys.modules[key]
        _sys.path[:] = path_before


def _load_tokenizer(path):
    """Newer `tokenizers` writes merges as pairs; older installs want "a b"
    strings and fail with an opaque ModelWrapper error. Convert on the fly so
    one checkpoint works on either version."""
    import json
    from tokenizers import Tokenizer
    try:
        return Tokenizer.from_file(str(path))
    except Exception:
        d = json.load(open(path))
        m = d.get("model", {}).get("merges")
        if not m or isinstance(m[0], str):
            raise
        d["model"]["merges"] = [" ".join(p) for p in m]
        tmp = str(path) + ".compat.json"
        json.dump(d, open(tmp, "w"))
        return Tokenizer.from_file(tmp)


class WeirdModel:
    """Lazily loaded: constructing this only stats a path."""

    def __init__(self):
        self.model = None
        self.decoders = {}
        self.tok = None
        self.cfg = None
        self.step = None

    def available(self, version="g-weird") -> bool:
        dec = DECODERS.get(version)
        return bool(dec) and usable(CKPT) and usable(dec) and TOKENIZER.is_file()

    def load(self, version="g-weird"):
        """Decoders are cached per version: switching back and forth in the UI
        should not re-read 200 MB from disk every time."""
        if self.model is not None and version in self.decoders:
            return {"step": self.step}
        if not self.available(version):
            raise RuntimeError(f"brak wag G-Weird dla wersji {version}")

        import torch
        tr = _load_module("model/transformer.py", "_gweird_transformer")
        vq = _load_module("model/vqvae.py", "_gweird_vqvae")

        if version not in self.decoders:
            vk = torch.load(str(DECODERS[version]), map_location="cpu",
                            weights_only=False)
            dec = vq.VQVAE(**vk["arch"])
            dec.load_state_dict(vk["model"])
            dec.eval()
            self.decoders[version] = dec

        if self.model is not None:
            return {"step": self.step}

        ck = torch.load(str(CKPT), map_location="cpu", weights_only=False)
        saved = ck.get("cfg") or ck.get("arch") or {}
        cfg = tr.WeirdConfig(**{k: v for k, v in saved.items()
                                if k in tr.WeirdConfig.__dataclass_fields__})
        net = tr.WeirdGPT(cfg)
        net.load_state_dict(ck["model"])
        net.eval()

        self.model, self.cfg = net, cfg
        self.kv = tr
        self.tok = _load_tokenizer(TOKENIZER)
        self.step = ck.get("step", 0)
        return {"step": self.step}

    def paint(self, prompt, scale=4.0, temp=1.0, top_k=100, seed=None,
              should_stop=None, version="g-weird"):
        """One image, returned as a PNG data URL.

        Classifier-free guidance: the prompt runs beside a blank one in the same
        batch and the logits are extrapolated away from the unconditional
        answer. The vocabulary mask matters as much — text, image and special
        tokens share one 16388-entry vocabulary, so without it the model can
        emit a word in the middle of a picture, which it does happily.

        `should_stop` is polled between tokens and makes paint() return None.
        Without it a cancel only removed the job from the queue while this loop
        kept running to the end, so the Mac stayed busy for a picture nobody
        would ever see — and with a queue in front of it, five cancels meant
        thirty-five seconds of work owed to nothing.
        """
        import os
        import numpy as np
        import torch
        import torch.nn.functional as F
        from PIL import Image

        gpt_base = _load_module("model/gpt_base.py", "_gweird_gpt_base")
        cfg = self.cfg
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big")
        torch.manual_seed(seed)

        ids = self.tok.encode(prompt).ids[:cfg.text_len]
        row = ([cfg.text_token(i) for i in ids]
               + [cfg.PAD] * (cfg.text_len - len(ids)) + [cfg.BOS_IMG])
        blank = [cfg.PAD] * cfg.text_len + [cfg.BOS_IMG]
        prefix = torch.tensor([row, blank], dtype=torch.long)

        lo, hi = cfg.image_token(0), cfg.image_token(cfg.n_image - 1)
        kv = gpt_base.KVCache(cfg.n_layer)
        with torch.no_grad():
            logits = self.model(prefix, kv=kv, pos=0)[:, -1]
            out = []
            for stepno in range(cfg.image_len):
                # Between tokens, not mid-matmul: one token is ~25 ms, which is
                # a fine granularity to notice a cancel at.
                if should_stop is not None and should_stop():
                    return None
                cond, uncond = logits[:1], logits[1:]
                g = uncond + scale * (cond - uncond)
                g[:, :lo] = -math.inf
                g[:, hi + 1:] = -math.inf
                g = g / max(temp, 1e-5)
                if top_k:
                    kth = g.topk(min(top_k, hi - lo + 1), dim=-1).values[:, -1:]
                    g = g.masked_fill(g < kth, -math.inf)
                nxt = torch.multinomial(F.softmax(g, dim=-1), 1)
                out.append(nxt)
                if stepno == cfg.image_len - 1:
                    break
                # Both branches continue with the SAME token: letting them
                # diverge would make the unconditional branch describe a
                # different picture and the guidance term meaningless.
                logits = self.model(torch.cat([nxt, nxt], dim=0), kv=kv,
                                    pos=cfg.text_len + 1 + stepno)[:, -1]
            codes = torch.cat(out, dim=1) - lo
            grid = int(round(math.sqrt(cfg.image_len)))
            img = self.decoders[version].decode(codes.view(-1, grid, grid))

        arr = ((img.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).numpy()[0]
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
