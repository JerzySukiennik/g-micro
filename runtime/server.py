"""
G-Micro inference backend — a WebSocket server wrapping the model.

Electron's main process spawns this on launch and kills it on quit. It is
also a complete, standalone WebSocket server: run it directly and point any
WS client (including a plain browser tab) at ws://localhost:PORT to test the
whole live protocol without Electron in the loop at all — that is how this
was built and verified.

Deliberately does not import from model/gpt.py by copying or modifying it —
that file is pulled fresh by every Kaggle training cycle still in flight
against this repo, and any change here must never risk that run.

Protocol (JSON over one text WS message per line):

  client -> server
    {"type": "chat", "text": str, "temperature": float, "top_k": int,
     "rag": bool,                                       # optional, default False
     "model": "g-micro" | "g-images",                   # optional, default g-micro
     "image": "data:image/png;base64,..."}              # required by g-images
    {"type": "stop"}

  server -> client
    {"type": "wake", "layer": int, "of": int}        # -1 = embeddings done
    {"type": "ready", "step": int|str, "params": int,
     "models": [{"id": str, "name": str, "available": bool}, ...]}
    {"type": "step", "token": str, "done": bool,
     "neurons": [[[idx,val], ...] x n_layer],          # top-64 per layer
     "probs": {"top": [[token_str, prob, token_id], ...],
               "entropy_norm": float, "label": str}}
    {"type": "image_progress", "p": float}            # g-images, real pass count
    {"type": "image_result", "image": str, "label": str}
    {"type": "error", "message": str}
"""

import asyncio
import base64
import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import websockets
from tokenizers import Tokenizer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from model.gpt import GPT, GPTConfig, Capture  # noqa: E402
from runtime.rag import retrieve  # noqa: E402
from runtime.images import RES, ImageModel  # noqa: E402

PORT = 8899
TOKENIZER_PATH = REPO / "data" / "tokenizer-v2.json"
FALLBACK_TOKENIZER_PATH = REPO / "data" / "tokenizer.json"
TOP_K_NEURONS = 64          # per layer, per UI-SPEC.md "Dane z modelu"
WAKE_STEP_DELAY = 0.05      # perceptual pacing between wake events — see load()

# How many past exchanges to show the model.
#
# Measured 2026-07-28 with bench/multiturn_eval.py over 18 single-turn checks
# and 5 follow-ups. Every SFT example was a single turn, so history is out of
# distribution and the cost is real and monotonic:
#
#     history      ordinary questions      follow-ups
#     0 turns              89%                 40%
#     1 turn               89%   (no loss)     80%
#     2 turns              83%   (-6 pts)
#     4 turns              78%   (-11 pts)
#
# One exchange buys the entire follow-up gain for nothing. Four would pay
# eleven points on *every* message for an ability already fully present at one.
# Raise this only with a model trained on multi-turn data, and re-run the
# benchmark when doing so.
HISTORY_TURNS = 1

# Leave room for the reply inside the model's 1024-token window.
PROMPT_BUDGET = 640

U, A, EOT, CTX = "<|user|>", "<|assistant|>", "<|endoftext|>", "<|context|>"


def find_checkpoint() -> Path:
    """Always the newest checkpoint available, per UI-SPEC.md: "Checkpointy:
    Zawsze najnowszy, bez przełączania". Prefers a finished pretraining run,
    falls back to whatever the training pipeline has produced so far — the
    app must still be usable while the model is mid-training.
    """
    candidates = [
        REPO / "checkpoints" / "sft" / "best.pt",
        REPO / "checkpoints" / "run1" / "best.pt",
        REPO / "Niepotrzebne" / "kaggle-orchestration" / "output" / "run" / "best.pt",
        REPO / "Niepotrzebne" / "kaggle-orchestration" / "output" / "run" / "ckpt.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "no checkpoint found — looked in: " + ", ".join(str(c) for c in candidates))


def decode_data_url(data_url: str) -> np.ndarray:
    """data: URL -> square uint8 RGB array at the resolution G-Images expects.

    Cropped to a centre square before resizing rather than squashed: a
    stretched face is a worse answer than a cropped one, and every training
    pair the model ever saw was square.
    """
    from PIL import Image, ImageOps

    payload = data_url.split(",", 1)[-1]
    img = Image.open(io.BytesIO(base64.b64decode(payload)))
    # Phone photos carry their rotation in EXIF; without this a portrait shot
    # comes back lying on its side.
    img = ImageOps.exif_transpose(img).convert("RGB")
    side = min(img.size)
    left, top = (img.width - side) // 2, (img.height - side) // 2
    img = img.crop((left, top, left + side, top + side)).resize((RES, RES), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def encode_data_url(arr: np.ndarray, scale: int = 3) -> str:
    from PIL import Image

    img = Image.fromarray(arr)
    # The model works at 128px. Upscaling with NEAREST keeps the result honest —
    # a smooth interpolation would suggest detail the model never produced.
    img = img.resize((RES * scale, RES * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def model_list(images_available: bool = None):
    """What the picker may offer.

    Every image version this machine has a checkpoint for gets its own entry
    under its own id, because the browser picks a version and has to be told
    truthfully whether that exact one can answer. Listing only the newest made
    every other version look asleep forever: the page was waiting for a name the
    Mac never published.

    Entries stay in the list when their checkpoint is missing, marked
    unavailable — a model that silently disappears from the menu looks like a
    bug, while a greyed-out entry explains itself.

    A plain function, not a method, because the bridge publishes this list in
    every heartbeat and must be able to do so without a loaded model: asking a
    Backend would defeat the whole point of dropping the weights while idle. The
    argument is ignored and kept only so older callers still work.
    """
    from runtime.images import VERSIONS, ImageModel

    models = [{"id": "g-micro", "name": "G-Micro", "desc": "rozmowa",
               "available": True}]
    for wire, spec in VERSIONS.items():
        models.append({
            "id": wire,
            "name": spec["name"],
            "desc": spec["desc"],
            "available": ImageModel(wire).available(),
            "needs_image": True,
        })
    from runtime.doodle import DoodleModel
    models.append({
        "id": "g-doodle",
        "name": "G-Doodle",
        "desc": "rysuje kreska po kresce",
        "available": DoodleModel().available(),
        "needs_text": True,
    })
    return models


class DoodleBackend:
    """G-Doodle behind the same socket as everything else.

    On a worker thread for the same reason as images: sampling is ~45 ms per
    token for a few hundred tokens, and holding the event loop for five seconds
    would freeze the stop button along with the socket.
    """

    def __init__(self):
        self.model = None

    def _get(self):
        from runtime.doodle import DoodleModel
        if self.model is None:
            self.model = DoodleModel()
        return self.model

    @property
    def available(self):
        return self._get().available()

    async def run(self, send, text, stop_event):
        from runtime.doodle import strokes_to_json
        model = self._get()
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, model.load)
        except Exception as e:
            await self._say(send, f"Nie udało mi się wczytać G-Doodle: {e}")
            return

        hit = model.resolve(text or "")
        if not hit.get("category"):
            alts = ", ".join(hit.get("alternatives") or [])
            extra = f" Najbliższe, co znam: {alts}." if alts else ""
            await self._say(
                send,
                "Nie znam tego. G-Doodle rysuje 345 rzeczy i tylko je - "
                f"wolny tekst dopasowuję do tej listy.{extra}")
            return

        label = hit.get("label") or hit["category"]
        await self._say(send, f"Rysuję: {label}.", done=False)

        def on_strokes(strokes, done):
            if stop_event.is_set():
                raise _Stopped()
            asyncio.run_coroutine_threadsafe(
                send({"type": "doodle", "strokes": strokes_to_json(strokes),
                      "label": label, "done": done}), loop)

        try:
            await loop.run_in_executor(
                None, lambda: model.draw(hit["category"], on_strokes=on_strokes))
        except _Stopped:
            await self._say(send, "Zatrzymane.")
            return
        except Exception as e:
            await self._say(send, f"Coś poszło nie tak przy rysowaniu: {e}")
            return

        await self._say(send, "")

    async def _say(self, send, text, done=True):
        if text:
            await send({"type": "step", "token": text, "done": False,
                        "neurons": [], "probs": _NO_PROBS})
        if done:
            await send({"type": "step", "token": "", "done": True,
                        "neurons": [], "probs": _NO_PROBS})


class ImageBackend:
    """G-Images as a second model behind the same socket.

    Runs on a worker thread: one edit is ~36 s of solid CPU, and doing it on
    the event loop would freeze the socket — including the stop button, which
    is the one thing a person definitely wants during a 36-second wait.
    """

    def __init__(self):
        # One loaded network per version. Sharing a single slot would mean the
        # 22M and 70.5M checkpoints evicting each other on every switch, and a
        # switch is one click in the picker.
        self.models = {}

    def for_version(self, version):
        from runtime.images import VERSIONS, DEFAULT_VERSION
        if version not in VERSIONS:
            version = DEFAULT_VERSION
        if version not in self.models:
            self.models[version] = ImageModel(version)
        return self.models[version]

    @property
    def model(self):
        """The default version, for callers that predate the version argument."""
        return self.for_version(None)

    @property
    def available(self):
        return self.model.available()

    async def run(self, send, text, image_url, stop_event, version=None):
        if not image_url:
            await self._say(send, "Dodaj zdjęcie - bez niego nie mam czego zmienić. "
                                  "Kliknij spinacz obok pola tekstowego.")
            return

        model = self.for_version(version)
        loop = asyncio.get_running_loop()
        try:
            info = await loop.run_in_executor(None, model.load)
        except Exception as e:
            await self._say(send, f"Nie udało mi się wczytać G-Images: {e}")
            return

        edit_type = model.match(text)
        if edit_type is None:
            labels = ", ".join(model.known_labels())
            await self._say(
                send,
                "Nie rozumiem tej zmiany. G-Images nie czyta zdań - zna skończoną "
                f"listę {len(model.known_labels())} przeróbek i to wszystko, "
                f"czego się nauczył:\n\n{labels}.\n\n"
                "Napisz coś z tej listy, np. „zrób to czarno-białe”.")
            return

        from runtime.images import LABELS
        label = LABELS.get(edit_type, edit_type)
        # Measured on this Mac from the preview run on 2026-08-19, not guessed:
        # the 70.5M network takes about five minutes per edit and the 22.4M one
        # about two. "Two minutes" was the old blanket answer and it was wrong by
        # more than double for the model people actually land on.
        minutes = "pięć" if model.version == "g-images" else "dwie"
        await self._say(send, f"Robię: {label}. To potrwa jakieś {minutes} minut(y) - "
                              f"model liczy na procesorze.", done=False)
        await send({"type": "image_progress", "p": 0.0})

        arr = decode_data_url(image_url)
        last = [0.0]

        def progress(p):
            if stop_event.is_set():
                raise _Stopped()
            # One message per percent is plenty; per forward pass would be 120
            # sends for a bar that only has so many pixels.
            if p - last[0] >= 0.01 or p >= 1.0:
                last[0] = p
                asyncio.run_coroutine_threadsafe(
                    send({"type": "image_progress", "p": p}), loop)

        try:
            out = await loop.run_in_executor(
                None, lambda: model.edit(arr, edit_type, on_progress=progress))
        except _Stopped:
            await self._say(send, "Zatrzymane.")
            return
        except Exception as e:
            await self._say(send, f"Coś poszło nie tak przy edycji: {e}")
            return

        await send({"type": "image_result", "image": encode_data_url(out),
                    "label": label, "step": info.get("step")})
        await self._say(send, "")

    async def _say(self, send, text, done=True):
        """Deliver a whole sentence through the same streaming shape the chat
        model uses, so the renderer has exactly one way to append a reply."""
        if text:
            await send({"type": "step", "token": text, "done": False,
                        "neurons": [], "probs": _NO_PROBS})
        if done:
            await send({"type": "step", "token": "", "done": True,
                        "neurons": [], "probs": _NO_PROBS})


class _Stopped(Exception):
    pass


_NO_PROBS = {"top": [], "entropy_norm": 0, "label": "confident"}


class Backend:
    def __init__(self):
        self.cfg = GPTConfig()
        self.model = GPT(self.cfg)
        self.model.eval()
        tok_path = TOKENIZER_PATH if TOKENIZER_PATH.exists() else FALLBACK_TOKENIZER_PATH
        self.tok = Tokenizer.from_file(str(tok_path))
        self.ckpt_step = "?"
        self.ready = False
        # Constructed, not loaded: this only stats a file path. The 22M-param
        # network stays on disk until someone picks it.
        self.images = ImageBackend()
        self.doodle = DoodleBackend()

    async def load(self, send):
        """Load weights and drive the Wake sequence.

        The load itself (one state_dict call) is a fraction of a second —
        too fast to see, and splitting it artificially would mean fabricating
        timing rather than reporting it. Instead: load for real first, then
        walk the now-loaded blocks in order and report each one, with a
        small fixed pacing delay so a human can actually perceive twelve
        distinct steps. What is reported (that block i is loaded, and its
        real parameter norm) is always true; only the reveal is paced.
        """
        path = find_checkpoint()
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.ckpt_step = ck.get("step", "?")

        await send({"type": "wake", "layer": -1, "of": self.cfg.n_layer})
        await asyncio.sleep(WAKE_STEP_DELAY)
        for i, block in enumerate(self.model.blocks):
            norm = sum(p.data.norm().item() ** 2 for p in block.parameters()) ** 0.5
            await send({"type": "wake", "layer": i, "of": self.cfg.n_layer, "norm": norm})
            await asyncio.sleep(WAKE_STEP_DELAY)

        self.ready = True
        await send({"type": "ready", "step": self.ckpt_step,
                    "params": self.model.num_params(), "checkpoint": path.name,
                    "models": self.model_list()})

    def model_list(self):
        return model_list(self.images.available)

    def _reduce_neurons(self, capture: Capture):
        """Full 12 layers, top-64 FFN activations per layer — see
        UI-SPEC.md "Dane z modelu": the grid shows all 24,576 neurons, but
        only the top-64-per-layer strongest firings are worth transmitting
        every token; that keeps each message small enough to stay smooth at
        15-28 tok/s instead of shipping all 2048 x 12 every step.
        """
        out = []
        for layer_gate in capture.ffn_gate:
            last = layer_gate[-1]                      # (2048,) — newest token
            vals, idx = last.abs().topk(min(TOP_K_NEURONS, last.numel()))
            out.append([[int(i), float(v)] for i, v in zip(idx.tolist(), vals.tolist())])
        return out

    def _probs_summary(self, probs: torch.Tensor, top_k_used: int):
        top = torch.topk(probs, min(10, probs.numel()))
        # id included alongside the decoded text: several distinct token ids
        # can decode to the same surface string (whitespace variants,
        # special tokens), so the frontend needs a real unique key to track
        # a specific candidate's rank across steps rather than matching on
        # text and occasionally conflating two different tokens.
        pairs = [[self.tok.decode([int(i)], skip_special_tokens=False), float(p), int(i)]
                 for p, i in zip(top.values.tolist(), top.indices.tolist())]
        # Entropy over the (already top-k-filtered) distribution, normalised
        # against the max possible entropy for that filter width — 0 means
        # fully confident, 1 means as spread out as the filter allows.
        nz = probs[probs > 0]
        entropy = float(-(nz * nz.log()).sum())
        max_entropy = math.log(max(top_k_used, 2))
        entropy_norm = max(0.0, min(1.0, entropy / max_entropy)) if max_entropy > 0 else 0.0
        label = ("confident" if entropy_norm < 0.33
                else "wavering" if entropy_norm < 0.66 else "guessing")
        return {"top": pairs, "entropy_norm": entropy_norm, "label": label}

    def build_prompt(self, text, history, context_prefix=""):
        """Lay out the conversation the way the model was trained to read it.

        Each turn is the same <|user|>/<|assistant|> pair the SFT data used, so
        the format is familiar even though the *sequence* of several turns is
        not: every training example was a single exchange. Feeding history is
        therefore out of distribution, and how far it can be pushed before
        answers degrade is a measured question, not an assumed one — see
        bench/multiturn_eval.py and HISTORY_TURNS below.

        History is trimmed from the oldest end so the newest exchange always
        survives, and the whole prompt is capped well under the model's 1024
        token window to leave room for the reply.
        """
        parts = []
        for turn in history:
            q, a = turn.get("user", ""), turn.get("assistant", "")
            if q and a:
                parts.append(f"{U}\n{q}\n{A}\n{a}{EOT}\n")
        parts.append(f"{U}\n{text}\n{A}\n")

        # Drop whole exchanges, oldest first, until the prompt fits. Cutting
        # mid-turn would leave a question with no answer, which is exactly the
        # pattern that teaches a model to invent questions.
        while len(parts) > 1:
            candidate = context_prefix + "".join(parts)
            if len(self.tok.encode(candidate).ids) <= PROMPT_BUDGET:
                break
            parts.pop(0)
        return context_prefix + "".join(parts)

    async def chat(self, send, text, temperature, top_k, stop_event, rag=False,
                   history=None):
        context_prefix = ""
        if rag:
            hit = retrieve(text)
            if hit:
                # <|context|> was reserved before pretraining but never
                # used in a single real training example until the
                # 2026-07-25 SFT pass added synthetic context+question+
                # answer examples specifically to teach this association
                # (see data/build_sft.py's CONTEXT_QA_* pipeline). Wrapping
                # retrieved text in it here only works with a checkpoint
                # trained on that data — on an older checkpoint this is no
                # better than the plain-text version it replaced.
                context_prefix = f"{CTX}\n{hit['text']}\n\n"
                # Tell the UI which document is speaking. Retrieval used to be
                # completely invisible, so a bad hit was indistinguishable
                # from the model losing its mind — confirmed 2026-07-27, when
                # an unrelated film article turned "Jak masz na imię?" into
                # "Argentyńskie" with nothing on screen to explain it.
                await send({"type": "context", "source": hit.get("source", "?"),
                            "text": hit["text"][:400]})
        history = (history or [])[-HISTORY_TURNS:]
        prompt = self.build_prompt(text, history, context_prefix)
        ids = self.tok.encode(prompt).ids
        idx = torch.tensor([ids])
        # The visualisation panel this fed was removed on 2026-07-28, so the
        # per-token copies of attention and activations off to the CPU are now
        # pure cost. Measurement could not resolve the saving on this hardware
        # (the run-to-run spread swamped it), but dead work cannot be faster.
        capture = Capture(enabled=False)

        for token_id, probs in self.model.generate(
                idx, max_new_tokens=400, temperature=temperature,
                top_k=top_k, capture=capture, use_cache=True):
            if stop_event.is_set():
                break

            is_eot = self.tok.token_to_id(EOT) == token_id
            text_piece = "" if is_eot else self.tok.decode([token_id], skip_special_tokens=False)

            await send({
                "type": "step",
                "token": text_piece,
                "done": is_eot,
                "neurons": self._reduce_neurons(capture),
                "probs": self._probs_summary(probs, top_k),
            })
            if is_eot:
                return
        await send({"type": "step", "token": "", "done": True,
                    "neurons": [], "probs": {"top": [], "entropy_norm": 0, "label": "confident"}})


async def handler(ws):
    backend = ws_state.setdefault("backend", None)
    if backend is None:
        backend = Backend()
        ws_state["backend"] = backend

    async def send(obj):
        await ws.send(json.dumps(obj))

    if not backend.ready:
        try:
            await backend.load(send)
        except Exception as e:
            await send({"type": "error", "message": f"failed to load model: {e}"})
            return
    else:
        await send({"type": "ready", "step": backend.ckpt_step,
                    "params": backend.model.num_params(),
                    "models": backend.model_list()})

    stop_event = asyncio.Event()
    pending = set()

    try:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if msg.get("type") == "chat":
                stop_event.clear()
                if msg.get("model") == "g-images":
                    coro = backend.images.run(
                        send, msg.get("text", ""), msg.get("image") or "", stop_event)
                else:
                    coro = backend.chat(
                        send, msg.get("text", ""),
                        float(msg.get("temperature", 0.8)),
                        int(msg.get("top_k", 50)),
                        stop_event, rag=bool(msg.get("rag", False)),
                        history=msg.get("history") or [])
                task = asyncio.create_task(coro)
                pending.add(task)
                task.add_done_callback(pending.discard)
            elif msg.get("type") == "stop":
                stop_event.set()
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        for t in pending:
            t.cancel()


ws_state = {}  # single-process, single-model: shared across reconnects


async def main():
    print(f"G-Micro backend listening on ws://localhost:{PORT}", flush=True)
    async with websockets.serve(handler, "localhost", PORT):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
