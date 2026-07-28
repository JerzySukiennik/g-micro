"""G-Images inside the chat app: attach a photo, say what to change.

The model lives in a sibling project (AIe/G-Images) and is loaded lazily —
only when someone actually picks it — so the chat app still starts in under
four seconds and does not hold a second model in memory for people who never
touch photos.

**It does not read sentences.** Conditioning is a discrete edit type with its
own learned tokens, because measurement on the text-conditioned version showed
it had learned to ignore the prompt almost entirely (cos(e_cond, e_null) =
1.000). G-Images' own inference script says mapping a sentence onto a type is
"an inference-time concern and deliberately deferred" — this module is where
that deferral gets paid, in Polish, with a keyword table.

If nothing matches, the app says so and lists what the model actually knows,
rather than applying whatever type happened to score highest. A photo edited in
a way nobody asked for is worse than an honest "nie rozumiem".
"""

import sys
import unicodedata
from pathlib import Path

import numpy as np
import torch

G_IMAGES = Path.home() / "Downloads/Claude/Projects/AIe/G-Images"

# Checkpoints newest first.
CKPTS = [
    G_IMAGES / "kaggle-run/output-v4/run/ckpt.pt",
    G_IMAGES / "kaggle-run/output-v3/run/ckpt.pt",
    G_IMAGES / "kaggle-run/output-train/run/ckpt.pt",
]

RES = 128           # the resolution every checkpoint so far was trained at
STEPS = 60          # edit_photo.py: "Don't judge quality below ~50"
GUIDANCE = 3.0

# The 24 types the v3 taxonomy defined, in order. The v4 taxonomy only appends
# (verified against git), so truncating the current name list to whatever the
# checkpoint's embedding table is wide gives correct names — but only as long
# as that stays true. This snapshot turns a future reordering from silently
# mislabelled edits into a loud failure at load time.
V3_TYPES = [
    "null", "black_and_white", "sepia_vintage", "brighter", "darker",
    "more_colorful", "less_colorful", "blur_background", "sharper", "warmer",
    "cooler", "inverted", "high_contrast", "low_contrast", "painting",
    "rain_storm", "snow_winter", "cartoon_anime", "desert", "fire_lava",
    "night", "sunset_sunrise", "space_scifi", "drawing_sketch",
]

# Polish phrasings → edit type. Matched on folded text: lowercase, no
# diacritics, punctuation flattened to spaces. That last part is not cosmetic —
# "czarno-białe" and "czarno biale" are the same request and both have to land,
# the same lesson G-Micro's identity examples already paid for once.
PHRASES = {
    "black_and_white": ["czarno bial", "czarnobial", "monochrom", "bez kolor",
                        "odcieniach szarosci", "black and white"],
    "sepia_vintage": ["sepia", "sepii", "vintage", "stare zdjecie", "retro"],
    "brighter": ["jasniej", "rozjasn", "za ciemne"],
    "darker": ["ciemniej", "przyciemn", "za jasne"],
    "more_colorful": ["bardziej kolorow", "nasyc", "zywsze kolor", "mocniejsze kolor"],
    "less_colorful": ["mniej kolorow", "wyblak", "stonuj", "slabsze kolor"],
    "blur_background": ["rozmyj tlo", "rozmyte tlo", "bokeh", "rozmycie tla"],
    "sharper": ["wyostrz", "ostrzej", "ostrosc"],
    "warmer": ["cieplej", "cieple barwy", "ociepl"],
    "cooler": ["chlodniej", "zimniej", "chlodne barwy", "ochlodz"],
    "inverted": ["negatyw", "odwroc kolor", "inwersj"],
    "high_contrast": ["wiekszy kontrast", "mocniejszy kontrast", "kontrastow"],
    "low_contrast": ["mniejszy kontrast", "plaski kontrast"],
    "painting": ["obraz olejny", "jak obraz", "malars", "malowid", "obraz"],
    "rain_storm": ["deszcz", "burz", "ulew"],
    "snow_winter": ["snieg", "zasniez", "sniezn"],
    "cartoon_anime": ["kreskowk", "anime", "komiks", "bajk", "kreskowe"],
    "desert": ["pustyni", "pustynn"],
    "fire_lava": ["ogien", "lawa", "plomien", "ognie"],
    "night": ["noc", "po zmroku", "ciemna pora"],
    "sunset_sunrise": ["zachod slonca", "wschod slonca", "zachod", "wschod"],
    "space_scifi": ["kosmos", "kosmiczn", "sci fi", "science fiction"],
    "drawing_sketch": ["szkic", "rysunek", "olowk", "narysow"],
}

# What to call each type when talking to a person.
LABELS = {
    "black_and_white": "czarno-białe", "sepia_vintage": "sepia",
    "brighter": "jaśniej", "darker": "ciemniej",
    "more_colorful": "mocniejsze kolory", "less_colorful": "słabsze kolory",
    "blur_background": "rozmyte tło", "sharper": "wyostrzenie",
    "warmer": "cieplejsze barwy", "cooler": "chłodniejsze barwy",
    "inverted": "negatyw", "high_contrast": "większy kontrast",
    "low_contrast": "mniejszy kontrast", "painting": "obraz olejny",
    "rain_storm": "deszcz", "snow_winter": "śnieg",
    "cartoon_anime": "kreskówka", "desert": "pustynia",
    "fire_lava": "ogień", "night": "noc",
    "sunset_sunrise": "zachód słońca", "space_scifi": "kosmos",
    "drawing_sketch": "szkic ołówkiem",
}


def _fold(s: str) -> str:
    s = s.lower().replace("ł", "l")
    s = "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))
    return "".join(c if c.isalnum() else " " for c in s)


class ImageModel:
    """Lazily loaded G-Images, with the edit-type vocabulary read from the
    checkpoint rather than from the sibling project's source.

    The two drifted: the newest local checkpoint knows 24 types while the code
    in G-Images now declares 64, so building the network from the source
    constant fails outright with a shape mismatch. Taking the count from the
    weights means whichever checkpoint exists defines the architecture — which
    is the only version that can possibly be right.
    """

    def __init__(self):
        self.net = None
        self.schedule = None
        self.type_names = []
        self.info = None

    def available(self) -> bool:
        return any(p.exists() for p in CKPTS)

    def load(self):
        if self.net is not None:
            return self.info
        path = next((p for p in CKPTS if p.exists()), None)
        if path is None:
            raise RuntimeError("nie znaleziono checkpointu G-Images")

        sys.path.insert(0, str(G_IMAGES))
        from model.unet import UNet
        from model.scheduler import DiffusionSchedule
        from data.edit_types import TYPE_NAMES

        ck = torch.load(path, map_location="cpu", weights_only=False)
        state = ck["ema"] if "ema" in ck else ck["model"]
        n_types = int(state["type_tokens.emb.weight"].shape[0])

        names = list(TYPE_NAMES)[:n_types]
        if n_types >= len(V3_TYPES) and names[:len(V3_TYPES)] != V3_TYPES:
            raise RuntimeError(
                "kolejność typów w G-Images się zmieniła — indeksy w checkpointie "
                "nie odpowiadają już nazwom, trzeba zmapować je ręcznie")

        net = UNet(n_types=n_types)
        net.load_state_dict({k: v.to(net.state_dict()[k].dtype) for k, v in state.items()})
        net.eval()

        self.net = net
        self.schedule = DiffusionSchedule(schedule="cosine", prediction="v", device="cpu")
        self.type_names = names
        self.info = {"step": ck.get("step"), "n_types": n_types,
                     "checkpoint": path.parent.parent.name,
                     "params": sum(p.numel() for p in net.parameters())}
        return self.info

    def match(self, text: str):
        """Sentence → edit type, or None. Longest matching phrase wins, so
        "mniej kolorowe" cannot be swallowed by another type's "kolorow"."""
        folded = _fold(text)
        best = None
        for name, phrases in PHRASES.items():
            if name not in self.type_names:
                continue                   # this checkpoint never learned it
            for p in phrases:
                if _fold(p).strip() in folded and (best is None or len(p) > best[1]):
                    best = (name, len(p))
        return best[0] if best else None

    def known_labels(self):
        """Human-facing names of the types this checkpoint can actually do."""
        return [LABELS[n] for n in PHRASES if n in self.type_names and n in LABELS]

    @torch.no_grad()
    def edit(self, image: np.ndarray, type_name: str, steps=STEPS, guidance=GUIDANCE,
             on_progress=None):
        """Apply one edit type. Returns a uint8 HxWx3 array at RES.

        Takes ~36 s on this Intel CPU: classifier-free guidance costs two
        forward passes per step, and it is not optional — at guidance 1.0 the
        raw prediction barely responds to conditioning at all.

        `on_progress` receives a real fraction, counted from actual forward
        passes through a proxy around the network. The sampler has no callback
        of its own and adding one would mean editing a file that live Kaggle
        runs still pull; counting calls from outside measures the same thing
        without touching it.
        """
        self.load()
        from data.edit_types import NULL_TYPE

        idx = self.type_names.index(type_name)
        src = torch.from_numpy(np.asarray(image, dtype=np.float32) / 127.5 - 1.0)
        before = src.permute(2, 0, 1)[None]

        base, net = self.net, self.net
        if on_progress:
            total = steps * 2       # e_full and e_image; no zeroed-image branch
            seen = 0

            def counting(*a, **kw):
                nonlocal seen
                out = base(*a, **kw)
                seen += 1
                on_progress(min(1.0, seen / total))
                return out
            net = counting

        out = self.schedule.ddim_sample(
            net, before, torch.tensor([idx]), steps=steps, device="cpu",
            text_uncond=torch.tensor([NULL_TYPE]), guidance=guidance)
        arr = ((out[0].clamp(-1, 1) + 1) * 127.5).byte().numpy()
        return arr.transpose(1, 2, 0)
