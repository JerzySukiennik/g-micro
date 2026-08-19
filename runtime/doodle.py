"""G-Doodle on the Mac: type a word, watch strokes arrive.

It was built to run in the browser — 17 MB of int8 ONNX, no server, working
while the Mac sleeps. That was its one advantage over G-Micro and G-Images, and
it is given up here deliberately, because in practice the browser path failed
twice over: the progress bar read "17.3 / 14 MB" (Pages reports the gzipped
length while the stream decompresses larger), and building the ONNX session
froze the tab, since a 17 MB model loads on the main thread.

Moving it here costs the offline story and buys a page that does not freeze and
a first drawing that starts immediately instead of after a 17 MB download.

The live drawing survives the move. `model.generate` is a generator, so strokes
are decoded and sent as they are sampled rather than at the end — the same
"watch it draw" the panel had, with the sampling happening 20 cm away instead of
in the tab.
"""

import sys
from pathlib import Path

G_DOODLE = Path.home() / "Downloads/Claude/Projects/AIe/G-Doodle"

CKPTS = [G_DOODLE / "checkpoints/g-doodle.pt"]

# A checkpoint whose download died leaves a small file that still "exists".
MIN_CKPT_BYTES = 10_000_000

MAX_TOKENS = 460


def usable(path) -> bool:
    try:
        return path.is_file() and path.stat().st_size >= MIN_CKPT_BYTES
    except OSError:
        return False


def _load_module(rel, name, needs_path=False):
    """Import a G-Doodle module by path, under a name of our own.

    Both projects ship a `model/` and a `runtime/` package, so a plain
    `sys.path.insert` plus `from runtime.sample import load` resolves to
    G-Micro's own runtime and fails — or worse, silently shadows one project's
    package with the other's depending on import order. Loading by file path
    under a private name removes the collision instead of racing it.

    `needs_path` is for modules with their own intra-project imports
    (prompt.py reads data.categories); those need G-Doodle on sys.path while
    they execute, and it is appended rather than inserted so it cannot shadow
    G-Micro's packages for anyone else.
    """
    import importlib.util
    import sys as _sys

    if name in _sys.modules:
        return _sys.modules[name]

    # Snapshot the whole path, not just our own addition: prompt.py inserts
    # G-Doodle's root itself so it can reach data.categories, and that entry
    # outlived the import. With it in place, G-Images' `from model.unet import
    # UNet` resolved `model` to G-DOODLE's model package — which has gpt.py and
    # tokenizer.py but no unet.py — so every photo edit failed with
    # "No module named 'model.unet'" until the bridge was restarted.
    path_before = list(_sys.path)
    if needs_path and str(G_DOODLE) not in _sys.path:
        _sys.path.append(str(G_DOODLE))

    # Everything the module imported for itself gets unregistered afterwards.
    # prompt.py pulls in `data.categories`, which binds sys.modules['data'] to
    # G-Doodle's package — and G-Images has a `data` package of its own, so it
    # then failed to load with "No module named 'model.unet'" for the rest of the
    # process's life. Restoring sys.modules is what makes these two projects
    # coexist; the private module name alone was never enough, because the
    # imports a module performs are not under that name.
    before = set(_sys.modules)
    try:
        spec = importlib.util.spec_from_file_location(name, G_DOODLE / rel)
        mod = importlib.util.module_from_spec(spec)
        _sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        for key in set(_sys.modules) - before - {name}:
            origin = getattr(_sys.modules[key], "__file__", "") or ""
            if str(G_DOODLE) in origin:
                del _sys.modules[key]
        _sys.path[:] = path_before


class DoodleModel:
    """Lazily loaded, like ImageModel: constructing this only stats a path."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.step = None

    def available(self) -> bool:
        return any(usable(p) for p in CKPTS)

    def load(self):
        if self.model is not None:
            return {"step": self.step}

        path = next((p for p in CKPTS if usable(p)), None)
        if path is None:
            raise RuntimeError("nie znaleziono checkpointu G-Doodle")

        import torch
        gpt = _load_module("model/gpt.py", "_gdoodle_gpt")
        tok = _load_module("model/tokenizer.py", "_gdoodle_tokenizer")

        ck = torch.load(str(path), map_location="cpu", weights_only=False)
        cfg = gpt.DoodleConfig(**ck["config"])
        net = gpt.DoodleGPT(cfg)
        net.load_state_dict(ck["model"])
        net.eval()

        self.model = net
        self.tokenizer = tok.StrokeTokenizer(ck["categories"])
        self.tok_mod = tok
        self.step = ck.get("step", 0)
        return {"step": self.step}

    def resolve(self, text):
        """Free text -> category, with the honesty rules the project insists on."""
        prompt = _load_module("runtime/prompt.py", "_gdoodle_prompt",
                              needs_path=True)
        return prompt.resolve(text)

    def draw(self, category, on_strokes=None, temperature=0.9, top_p=0.9):
        """Sample one drawing, calling on_strokes(strokes, done) as it goes.

        Decoding the whole prefix on every token is O(n^2) in principle, but n is
        at most 460 short integers and the decode is a Python loop over them —
        microseconds against the ~40 ms the model spends sampling each token. The
        simpler code is worth more here than the saved arithmetic.
        """
        import torch
        BOS, EOS = self.tok_mod.BOS, self.tok_mod.EOS

        ids = [BOS, self.tokenizer.category_token(category)]
        idx = torch.tensor([ids], dtype=torch.long)
        out = list(ids)
        budget = min(MAX_TOKENS, self.model.config.block_size - len(ids) - 1)

        for token in self.model.generate(idx, budget, temperature=temperature,
                                         top_p=top_p, stop_token=EOS):
            out.append(token)
            if on_strokes is not None:
                on_strokes(self.tokenizer.decode(out), False)

        strokes = self.tokenizer.decode(out)
        if on_strokes is not None:
            on_strokes(strokes, True)
        return strokes


def strokes_to_json(strokes):
    """Absolute points, rounded to whole pixels.

    The canvas draws at 256 px, so a tenth of a pixel is invisible and costs
    three characters in every point of every flush — and these go through a
    realtime database on every update, not once at the end.
    """
    return [[[int(round(float(x))), int(round(float(y)))] for x, y in s]
            for s in strokes if len(s) > 1]
