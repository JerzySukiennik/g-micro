"""
Build the instruction-tuning dataset.

Pretraining taught MicroG what Polish looks like. It did not teach it that a
question is followed by an answer — in web text a question is usually followed
by more questions. This stage fixes that, and only that.

Two things make it different from pretraining:

  Chat template. Turns are wrapped in the control tokens reserved in the
  tokenizer before pretraining started, so no embedding resize is needed.

  Loss masking. The model is scored only on the assistant's reply. Training on
  the user's turn as well would teach it to invent questions, which is exactly
  the behaviour we are trying to remove.

Output: pl_sft_tokens.bin (uint16) and pl_sft_mask.bin (uint8), same length.
Mask 1 = a position the loss is computed on.
"""

import argparse
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

SOURCES = [
    # (repo, weight of trust). The cleaned set is the backbone; the second adds
    # coverage at the cost of translation artefacts, which clean() handles.
    "saillab/alpaca-polish-cleaned",
    "Lajonbot/alpaca-dolly-chrisociepa-instruction-only-polish",
]

U, A, EOT, CTX = "<|user|>", "<|assistant|>", "<|endoftext|>", "<|context|>"

# Neither Alpaca corpus contains a single example teaching the model its own
# name — confirmed live, 2026-07-22: asked "jak masz na imię?" the SFT model
# answered "Tak, mam na imię." with no name at all, because nothing in its
# training data ever associated one with it. Hand-written and repeated many
# times so a 110M model actually memorises the pattern instead of it being one
# drop in ~90k unrelated examples — this is exactly the kind of short, fixed
# association a small model CAN reliably learn, unlike open-ended world facts.
IDENTITY_EXAMPLES = [
    ("Jak masz na imię?", "Nazywam się MicroG."),
    ("Jak się nazywasz?", "Nazywam się MicroG."),
    ("Kim jesteś?", "Jestem MicroG, mały model językowy stworzony od zera przez Jurka."),
    ("Czym jesteś?", "Jestem modelem językowym o nazwie MicroG. Mam około 110 milionów parametrów."),
    ("Powiedz mi kim jesteś.", "Jestem MicroG — model językowy wytrenowany od podstaw przez Jurka."),
    ("Kto cię stworzył?", "Zostałem stworzony od zera przez Jurka, w ramach projektu Gzowo AI."),
    ("Ile masz parametrów?", "Mam około 110 milionów parametrów."),
    ("Jesteś duży czy mały?", "Jestem małym modelem — mam tylko około 110 milionów parametrów."),
    ("Czy jesteś ChatGPT?", "Nie, nie jestem ChatGPT. Nazywam się MicroG i jestem osobnym, dużo mniejszym modelem."),
    ("Kto cię wytrenował?", "Wytrenował mnie Jurek, od zera, na własną rękę."),
    ("Jak masz na imię", "Nazywam się MicroG."),
    ("Przedstaw się.", "Jestem MicroG, model językowy stworzony od zera przez Jurka."),
    ("Opowiedz coś o sobie.", "Nazywam się MicroG. Jestem małym modelem językowym, wytrenowanym od zera przez Jurka."),
    ("Skąd pochodzisz?", "Zostałem stworzony od zera przez Jurka, w ramach projektu Gzowo AI."),
    ("Czy masz imię?", "Tak, nazywam się MicroG."),
    ("Jak brzmi twoja nazwa?", "Nazywam się MicroG."),
    ("Czy jesteś Claude?", "Nie, nie jestem Claude. Nazywam się MicroG i jestem osobnym, dużo mniejszym modelem."),
    ("Czy jesteś Gemini?", "Nie, nie jestem Gemini. Nazywam się MicroG i jestem osobnym, dużo mniejszym modelem."),
    ("Czy jesteś człowiekiem?", "Nie, jestem modelem językowym o nazwie MicroG."),
]
IDENTITY_REPEATS = 60


def strip_diacritics(text: str) -> str:
    """Polish text with the accents removed — 'imię' -> 'imie'.

    NFD decomposition splits an accented letter into base + combining mark, so
    dropping the marks leaves the bare letter. 'ł' is the one Polish letter
    that has no decomposition (the stroke is part of the glyph, not a combining
    mark), so it needs mapping by hand or it would survive untouched.
    """
    text = text.replace("ł", "l").replace("Ł", "L")
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if not unicodedata.combining(c))


def identity_pairs():
    """IDENTITY_EXAMPLES plus an accent-free variant of every question.

    Confirmed live, 2026-07-27: after the first identity fix landed, "Jak masz
    na imię?" answered "Nazywam się MicroG." with p=0.92 on the first token —
    solidly memorised — while "Jak masz na imie?" (same question typed without
    the ogonek) answered "Mam na imie." and "Kim jestes?" wandered off into
    unrelated chatter. BPE sees 'imię' and 'imie' as different token
    sequences, and 16 strings x60 repeats memorises *those sequences*, not the
    concept behind them, so there is nothing to generalise from. People type
    Polish without diacritics constantly, so both spellings have to be taught.

    Only the question is stripped, never the answer: sloppy input should still
    get a correctly-written Polish reply back.
    """
    seen = set()
    for instruction, output in IDENTITY_EXAMPLES:
        for variant in (instruction, strip_diacritics(instruction)):
            if variant not in seen:
                seen.add(variant)
                yield variant, output

# Confirmed live, 2026-07-25: even with the right Wikipedia/vault snippet
# handed to the model as plain text right before the question (see
# runtime/rag.py), the answer still ignored it and confabulated something
# unrelated. The identity fix worked because it's a closed, memorisable set
# repeated many times; grounding-in-context is a *skill* (read the preceding
# text, then answer from it), and nothing in either Alpaca corpus ever
# demonstrates that skill — SFT so far only ever taught "answer the
# question", never "answer the question using the text above it". Unlike the
# identity examples, this needs breadth (many different contexts), not
# repetition of a few — the model needs to generalise the *habit*, not
# memorise specific facts.
CONTEXT_QA_REPO = "clarin-pl/poquad"
CONTEXT_QA_FILE = "poquad-dev.json"  # dev split alone has ~5.7k answerable QAs
CONTEXT_QA_SAMPLE = 1200


def load_context_qa_examples(sample_size=CONTEXT_QA_SAMPLE, seed=0):
    """PoQuAD ships as raw SQuAD-format JSON with a loader script the
    `datasets` library no longer supports running ("Dataset scripts are no
    longer supported") — download and parse the JSON directly instead of
    going through `load_dataset`."""
    path = hf_hub_download(CONTEXT_QA_REPO, CONTEXT_QA_FILE, repo_type="dataset",
                            token=read_token())
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    triples = []
    for article in data["data"]:
        for para in article["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                if qa["is_impossible"] or not qa["answers"]:
                    continue
                ans = qa["answers"][0]
                # generative_answer reads like a real reply ("Warszawa jest
                # stolicą Polski"); the raw extracted span ("Polski") reads
                # like a search-engine result. We want the former — this is
                # meant to teach conversational grounding, not extraction.
                answer = ans.get("generative_answer") or ans.get("text")
                if not answer:
                    continue
                triples.append((context, qa["question"], answer))

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(triples))[:sample_size]
    return [triples[i] for i in idx]


def format_context_example(context, question, answer, max_context_chars=800):
    """Same (prompt, reply) shape as format_example, with the source text
    wrapped in the tokenizer's reserved <|context|> token in front of the
    <|user|> turn. That token was reserved before pretraining started but
    never actually used in a single real training example until now — this
    IS the training pass that finally gives it a meaning, so use it properly
    rather than perpetuating the plain-text workaround runtime/rag.py used
    only because nothing had taught the model this token yet. Once this SFT
    pass lands, update rag.py's injection to match (wrap retrieved text in
    <|context|>...) instead of bare prepended prose."""
    context, question, answer = clean(context), clean(question), clean(answer)
    if len(context) < 20 or len(question) < 4 or len(answer) < 2:
        return None
    context = context[:max_context_chars]
    return f"{CTX}\n{context}\n\n{U}\n{question}\n{A}\n", answer + EOT


def read_token(env=Path(".env")):
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip()
    return None


def clean(s) -> str:
    """Undo the damage machine translation left in these corpora.

    Several rows arrive with their quoting baked into the value rather than
    around it — the field literally contains `'Oceń to zdanie'`. Left alone the
    model learns to sprinkle stray quotes through its answers.
    """
    if s is None:
        return ""
    s = str(s).strip()
    if s.lower() in ("nan", "none", "null"):
        return ""
    # strip one layer of wrapping quotes, repeatedly
    while len(s) > 1 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    s = s.replace("\\n", "\n").replace(" ", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def format_example(instruction, inp, output):
    """One conversation turn, split into (prompt, reply) or None.

    The two halves are returned separately and tokenised separately. Slicing a
    single encoded string by character offset would be guessing: BPE merges can
    straddle any boundary, so the token count of a prefix is not guaranteed to
    equal the length of that prefix inside the whole. Here the split point sits
    immediately after a special token, which the tokenizer never merges across,
    so encoding the halves apart is both safe and exact.
    """
    instruction, inp, output = clean(instruction), clean(inp), clean(output)
    if len(instruction) < 4 or len(output) < 2:
        return None
    user = f"{instruction}\n{inp}" if inp else instruction
    return f"{U}\n{user}\n{A}\n", output + EOT


def build(tokenizer_path: Path, out_prefix: Path, max_len: int):
    tok = Tokenizer.from_file(str(tokenizer_path))
    for t in (U, A, EOT, CTX):
        assert tok.token_to_id(t) is not None, f"{t} missing from tokenizer"

    token_buf, mask_buf = [], []
    kept = dropped = 0
    seen = set()

    for name in SOURCES:
        ds = load_dataset(name, split="train", token=read_token())
        print(f"{name}: {len(ds):,} rows", flush=True)
        for row in ds:
            made = format_example(row.get("instruction"), row.get("input"),
                                  row.get("output"))
            if made is None:
                dropped += 1
                continue
            prompt, reply = made

            # Exact-duplicate drop: the two corpora overlap heavily, and a
            # small model will happily memorise anything it sees twice.
            key = hash(prompt + reply)
            if key in seen:
                dropped += 1
                continue
            seen.add(key)

            p_ids = tok.encode(prompt).ids
            r_ids = tok.encode(reply).ids
            ids = p_ids + r_ids
            if len(ids) > max_len:
                dropped += 1
                continue
            # Score the reply only. Training on the user's turn would teach the
            # model to invent questions — the habit we are here to remove.
            mask = np.zeros(len(ids), dtype=np.uint8)
            mask[len(p_ids):] = 1

            token_buf.append(np.asarray(ids, dtype=np.uint16))
            mask_buf.append(mask)
            kept += 1
            if kept % 10000 == 0:
                print(f"  {kept:,} kept", flush=True)

    # Identity examples bypass the exact-duplicate filter above on purpose —
    # deliberate repetition is the whole point here, not something to dedup
    # away like an accidental corpus overlap.
    identity_kept = 0
    identity_written = 0
    for instruction, output in identity_pairs():
        identity_written += 1
        made = format_example(instruction, None, output)
        if made is None:
            continue
        prompt, reply = made
        p_ids = tok.encode(prompt).ids
        r_ids = tok.encode(reply).ids
        ids = p_ids + r_ids
        if len(ids) > max_len:
            continue
        mask = np.zeros(len(ids), dtype=np.uint8)
        mask[len(p_ids):] = 1
        for _ in range(IDENTITY_REPEATS):
            token_buf.append(np.asarray(ids, dtype=np.uint16))
            mask_buf.append(mask)
            identity_kept += 1
    kept += identity_kept
    print(f"identity examples: {len(IDENTITY_EXAMPLES)} written "
          f"({identity_written} incl. accent-free variants) x{IDENTITY_REPEATS} "
          f"= {identity_kept:,} kept", flush=True)

    # Context-grounded QA: teaches the *habit* of reading the <|context|>
    # block and answering from it, which nothing else in this build ever
    # demonstrates. Breadth matters here, not repetition — sampled once each
    # from PoQuAD rather than repeated like the identity examples, since the
    # goal is generalising a skill across many different texts, not
    # memorising a fixed handful of facts.
    context_kept = context_dropped = 0
    for context, question, answer in load_context_qa_examples():
        made = format_context_example(context, question, answer)
        if made is None:
            context_dropped += 1
            continue
        prompt, reply = made
        p_ids = tok.encode(prompt).ids
        r_ids = tok.encode(reply).ids
        ids = p_ids + r_ids
        if len(ids) > max_len:
            context_dropped += 1
            continue
        mask = np.zeros(len(ids), dtype=np.uint8)
        mask[len(p_ids):] = 1
        token_buf.append(np.asarray(ids, dtype=np.uint16))
        mask_buf.append(mask)
        context_kept += 1
    kept += context_kept
    print(f"context-QA examples: {context_kept:,} kept, {context_dropped:,} dropped",
          flush=True)

    # finetune.py's val split is just "the last val_frac of examples by
    # position here" (SFTData: offsets[:cut] / offsets[cut:]) — with identity
    # examples appended after the whole main corpus, every single one of them
    # landed past the cut and the model was trained on literally zero of
    # them, only ever evaluated against them. Confirmed by direct
    # calculation before this fix: 89,495 main + 960 identity, 2% val cut
    # sits at index 88,645 — entirely before the identity block starts at
    # 89,495. Shuffling before the split fixes this generally, not just for
    # the current counts.
    rng = np.random.default_rng(0)
    order = rng.permutation(len(token_buf))
    token_buf = [token_buf[i] for i in order]
    mask_buf = [mask_buf[i] for i in order]

    tokens = np.concatenate(token_buf)
    masks = np.concatenate(mask_buf)
    assert len(tokens) == len(masks)

    # Where each example begins in the concatenated stream. Training samples
    # windows from these offsets rather than from anywhere, so a window never
    # opens midway through a reply — the model would otherwise be asked to
    # continue an answer whose question it never saw.
    offsets = np.zeros(len(token_buf), dtype=np.int64)
    np.cumsum([len(a) for a in token_buf[:-1]], out=offsets[1:])

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    tokens.tofile(f"{out_prefix}_tokens.bin")
    masks.tofile(f"{out_prefix}_mask.bin")
    offsets.tofile(f"{out_prefix}_offsets.bin")

    print(f"\nkept {kept:,} examples, dropped {dropped:,}")
    print(f"{len(tokens)/1e6:.1f}M tokens, {masks.mean()*100:.1f}% trained on")
    print(f"-> {out_prefix}_tokens.bin / _mask.bin")

    print("\n--- sample (special tokens shown) ---")
    print(repr(tok.decode([int(t) for t in token_buf[0]],
                          skip_special_tokens=False))[:420])
    print(f"(loss on {int(mask_buf[0].sum())} of {mask_buf[0].size} tokens)")

    # Alignment audit across the whole set, not one lucky example: the token
    # immediately before the first scored position must be the newline that
    # follows <|assistant|>. If this drifts, the model is trained to predict
    # the wrong half of the conversation and nothing else will reveal it.
    a_id, bad = tok.token_to_id(A), 0
    for t_arr, m_arr in zip(token_buf[:5000], mask_buf[:5000]):
        first = int(np.argmax(m_arr))
        if first < 2 or int(t_arr[first - 2]) != a_id:
            bad += 1
    print(f"alignment check: {bad} of {min(5000, len(token_buf))} examples misaligned")
    assert bad == 0, "loss mask does not start right after <|assistant|>"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer-v2.json"))
    ap.add_argument("--out-prefix", type=Path, default=Path("data/pl_sft"))
    ap.add_argument("--max-len", type=int, default=768)
    args = ap.parse_args()
    build(args.tokenizer, args.out_prefix, args.max_len)
    sys.stdout.flush()
    os._exit(0)   # see fetch_corpus.py — datasets can abort at interpreter exit
