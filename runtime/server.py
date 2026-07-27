"""
MicroG inference backend — a WebSocket server wrapping the model.

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
     "rag": bool}                                       # optional, default False
    {"type": "stop"}

  server -> client
    {"type": "wake", "layer": int, "of": int}        # -1 = embeddings done
    {"type": "ready", "step": int|str, "params": int}
    {"type": "step", "token": str, "done": bool,
     "neurons": [[[idx,val], ...] x n_layer],          # top-64 per layer
     "probs": {"top": [[token_str, prob, token_id], ...],
               "entropy_norm": float, "label": str}}
    {"type": "error", "message": str}
"""

import asyncio
import json
import math
import sys
import time
from pathlib import Path

import torch
import websockets
from tokenizers import Tokenizer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from model.gpt import GPT, GPTConfig, Capture  # noqa: E402
from runtime.rag import retrieve  # noqa: E402

PORT = 8899
TOKENIZER_PATH = REPO / "data" / "tokenizer-v2.json"
FALLBACK_TOKENIZER_PATH = REPO / "data" / "tokenizer.json"
TOP_K_NEURONS = 64          # per layer, per UI-SPEC.md "Dane z modelu"
WAKE_STEP_DELAY = 0.05      # perceptual pacing between wake events — see load()

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


class Backend:
    def __init__(self):
        self.cfg = GPTConfig()
        self.model = GPT(self.cfg)
        self.model.eval()
        tok_path = TOKENIZER_PATH if TOKENIZER_PATH.exists() else FALLBACK_TOKENIZER_PATH
        self.tok = Tokenizer.from_file(str(tok_path))
        self.ckpt_step = "?"
        self.ready = False

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
                    "params": self.model.num_params(), "checkpoint": path.name})

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

    async def chat(self, send, text, temperature, top_k, stop_event, rag=False):
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
        prompt = f"{context_prefix}{U}\n{text}\n{A}\n"
        ids = self.tok.encode(prompt).ids
        idx = torch.tensor([ids])
        capture = Capture(enabled=True)

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
                    "params": backend.model.num_params()})

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
                task = asyncio.create_task(backend.chat(
                    send, msg.get("text", ""),
                    float(msg.get("temperature", 0.8)),
                    int(msg.get("top_k", 50)),
                    stop_event, rag=bool(msg.get("rag", False))))
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
    print(f"MicroG backend listening on ws://localhost:{PORT}", flush=True)
    async with websockets.serve(handler, "localhost", PORT):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
