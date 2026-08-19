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

# The image models this Mac can actually serve, by the name the browser asks for.
#
# There is more than one entry because there is more than one architecture. The
# 22.4M generation and the 70.5M generation are different networks, not different
# training lengths: loading one into the other raises a wall of "size mismatch
# for up_attns.3.norm.weight: torch.Size([64]) vs [128]" and every edit fails.
# Treating them as a fallback chain cost a live debugging session once; keeping
# each version with the shape it was trained at is what makes both loadable.
#
# `arch` was not guessed. It is read back off the weights: in_conv gives
# base_channels and the down-path conv widths give the multipliers, which is how
# 64/(1,2,4,4) was recovered for the 22M run. Verified: 0 missing keys, 0
# unexpected, 22.4M parameters.
VERSIONS = {
    "g-images": {
        "name": "G-Images",
        "desc": "edycja zdjęć",
        "ckpts": [
            G_IMAGES / "kaggle-run/out-vb/run/ckpt.pt",
            G_IMAGES / "kaggle-run/out-v12/run/ckpt.pt",
            G_IMAGES / "kaggle-run/out-v10/run/ckpt.pt",
        ],
        "arch": {"base_channels": 128, "channel_mults": (1, 2, 3, 4)},
    },
    "g-image-2-1": {
        "name": "G-Image 2.1",
        "desc": "najnowszy, 98M",
        "ckpts": [
            G_IMAGES / "kaggle-run/out-21-68k/run/ckpt.pt",
        ],
        "arch": {"base_channels": 152, "channel_mults": (1, 2, 3, 4)},
    },
    "g-image-1": {
        "name": "G-Image 1",
        "desc": "pierwsza wersja, 22M",
        "ckpts": [
            G_IMAGES / "kaggle-run/out-v9/run/ckpt.pt",
            G_IMAGES / "kaggle-run/out-v8/run/ckpt.pt",
        ],
        "arch": {"base_channels": 64, "channel_mults": (1, 2, 4, 4)},
    },
}

DEFAULT_VERSION = "g-images"

# Kept so nothing that imported the old name breaks.
CKPTS = VERSIONS[DEFAULT_VERSION]["ckpts"]

# A checkpoint whose download died leaves a zero-byte file that still "exists",
# and picking it by name alone would look exactly like a missing model — there
# is one such file in kaggle-run right now (out-final). Torch needs a few
# hundred megabytes here, so anything trivially small is a corpse.
MIN_CKPT_BYTES = 10_000_000


def usable(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size >= MIN_CKPT_BYTES
    except OSError:
        return False

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

    def __init__(self, version=None):
        # Which of VERSIONS this instance serves. One object per version, so a
        # loaded 70.5M network is never handed a 22M checkpoint's weights.
        self.version = version or DEFAULT_VERSION
        self.net = None
        self.schedule = None
        self.type_names = []
        self.info = None

    def available(self) -> bool:
        return any(usable(p) for p in VERSIONS[self.version]["ckpts"])

    def load(self):
        if self.net is not None:
            return self.info
        spec = VERSIONS[self.version]
        path = next((p for p in spec["ckpts"] if usable(p)), None)
        if path is None:
            raise RuntimeError(f"nie znaleziono checkpointu dla {spec['name']}")

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

        # The architecture belongs to the version, not to this file's defaults.
        net = UNet(n_types=n_types, **spec["arch"])
        net.load_state_dict({k: v.to(net.state_dict()[k].dtype) for k, v in state.items()})
        net.eval()

        self.net = net
        self.schedule = DiffusionSchedule(schedule="cosine", prediction="v", device="cpu")
        self.type_names = names
        self.info = {"step": ck.get("step"), "n_types": n_types,
                     "version": self.version,
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
