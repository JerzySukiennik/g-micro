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

# Kazda wersja to teraz KOMPLET: transformer, dekoder i tokenizer tekstu.
#
# Do 1 wlacznie wystarczal jeden slownik i jeden transformer, bo 0.9 i 1 roznily
# sie WYLACZNIE dekoderem — te same id kodow, ten sam obraz narysowany dwa razy.
# 1.1 lamie to zalozenie w kazdym punkcie: ma wlasny tokenizer obrazu (576
# tokenow zamiast 256, wiec id 4711 znaczy tam co innego), wlasny tokenizer
# tekstu (ze starego slownika przetrwalo 3193 z 8192 tokenow) i wlasny
# transformer. Trzymanie tego w jednej wspolnej sciezce dawaloby obrazki
# zlozone z czesci dwoch roznych modeli — bez bledu, po prostu smieci.
#
#   0.9  pierwotny dekoder VQ-VAE: miekki, malowany, ale spojny.
#   1    8000 krokow douczania adwersarialnego przy zamrozonym enkoderze.
#        Blad rekonstrukcji 10,3/255 wobec 15,7, detal tam gdzie byla maz,
#        kosztem drobnego trzasku na calosci.
#   1.1  nowy tokenizer (24x24 zamiast 16x16 — cztery razy gestsza siatka) i
#        transformer trenowany od zera na 2 357 878 parach zamiast 1 780 125.
VERSIONS = {
    "g-weird": {
        "ckpt": G_WEIRD / "run/gweird.pt",
        "vqvae": G_WEIRD / "run/vqvae.pt",
        "text": G_WEIRD / "run/text.json",
    },
    "g-weird-1": {
        "ckpt": G_WEIRD / "run/gweird.pt",
        "vqvae": G_WEIRD / "run/decoder.pt",
        "text": G_WEIRD / "run/text.json",
    },
    "g-weird-11": {
        "ckpt": G_WEIRD / "run11/gweird.pt",
        "vqvae": G_WEIRD / "run11/vqvae.pt",
        "text": G_WEIRD / "run11/text.json",
    },
}

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
        # Jeden komplet na wersje. 0.9 i 1 dziela plik transformera, wiec torch
        # wczyta go dwa razy — 745 MB nadmiaru, gdyby ktos przelaczal miedzy
        # nimi. Warte tego: alternatywa to wspoldzielony cache po sciezce, ktory
        # musi wiedziec, ze cfg dwoch wersji jest zgodne, a to zalozenie wlasnie
        # przestalo byc prawdziwe przy 1.1.
        self.bundles = {}

    def available(self, version="g-weird") -> bool:
        v = VERSIONS.get(version)
        return bool(v) and usable(v["ckpt"]) and usable(v["vqvae"]) \
            and v["text"].is_file()

    def load(self, version="g-weird"):
        """Komplety sa cache'owane per wersja: przelaczanie w UI nie moze
        czytac setek megabajtow z dysku za kazdym razem."""
        if version in self.bundles:
            return {"step": self.bundles[version]["step"]}
        if not self.available(version):
            raise RuntimeError(f"brak wag G-Weird dla wersji {version}")

        import torch
        tr = _load_module("model/transformer.py", "_gweird_transformer")
        vq = _load_module("model/vqvae.py", "_gweird_vqvae")
        v = VERSIONS[version]

        vk = torch.load(str(v["vqvae"]), map_location="cpu", weights_only=False)
        dec = vq.VQVAE(**vk["arch"])
        # strict=False: checkpoint tokenizera 1.1 nosi bufory EMA codebooka,
        # ktorych sam dekoder nie potrzebuje.
        dec.load_state_dict(vk["model"], strict=False)
        dec.eval()

        ck = torch.load(str(v["ckpt"]), map_location="cpu", weights_only=False)
        saved = ck.get("cfg") or ck.get("arch") or {}
        cfg = tr.WeirdConfig(**{k: val for k, val in saved.items()
                                if k in tr.WeirdConfig.__dataclass_fields__})
        net = tr.WeirdGPT(cfg)
        net.load_state_dict(ck["model"])
        net.eval()

        # Siatka dekodera musi zgadzac sie z dlugoscia obrazu transformera.
        # Gdyby ktos wskazal w VERSIONS dekoder z innego tokenizera, obrazek
        # wyszedlby bez bledu i bez sensu — wiec sprawdzamy tu, raz, przy
        # wczytaniu.
        grid = int(round(math.sqrt(cfg.image_len)))
        if grid * grid != cfg.image_len:
            raise RuntimeError(f"{version}: {cfg.image_len} tokenow nie jest "
                               f"kwadratem")
        down = 2 ** len(vk["arch"]["mults"])
        if vk["arch"].get("n_codes", 8192) != cfg.n_image:
            raise RuntimeError(f"{version}: codebook {vk['arch']['n_codes']} "
                               f"wobec {cfg.n_image} w transformerze")

        self.bundles[version] = {
            "model": net, "cfg": cfg, "dec": dec, "grid": grid,
            "tok": _load_tokenizer(v["text"]), "step": ck.get("step", 0),
            "res": grid * down,
        }
        return {"step": self.bundles[version]["step"]}

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
        # Wczytanie na zadanie, a nie ufanie wolajacemu: paint(), ktory poprosil
        # o wersje jeszcze nie wczytana, konczyl sie KeyError-em docierajacym do
        # przegladarki jako "coz sie stalo" bez wskazania czego. Zdarzylo sie na
        # zywo — serwer podawal wersje do load(), ale nie do paint().
        if version not in self.bundles:
            self.load(version)
        b = self.bundles[version]
        cfg = b["cfg"]
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big")
        torch.manual_seed(seed)

        ids = b["tok"].encode(prompt).ids[:cfg.text_len]
        row = ([cfg.text_token(i) for i in ids]
               + [cfg.PAD] * (cfg.text_len - len(ids)) + [cfg.BOS_IMG])
        blank = [cfg.PAD] * cfg.text_len + [cfg.BOS_IMG]
        prefix = torch.tensor([row, blank], dtype=torch.long)

        lo, hi = cfg.image_token(0), cfg.image_token(cfg.n_image - 1)
        kv = gpt_base.KVCache(cfg.n_layer)
        with torch.no_grad():
            logits = b["model"](prefix, kv=kv, pos=0)[:, -1]
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
                logits = b["model"](torch.cat([nxt, nxt], dim=0), kv=kv,
                                    pos=cfg.text_len + 1 + stepno)[:, -1]
            codes = torch.cat(out, dim=1) - lo
            img = b["dec"].decode(codes.view(-1, b["grid"], b["grid"]))

        arr = ((img.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).numpy()[0]
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
