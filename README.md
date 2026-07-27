<!-- One image, no <picture>: the icon carries its own dark plate, so it reads
     correctly in both GitHub themes. A bare transparent mark does not — white
     on white is how this ended up as a blank square the first time. -->
<img src="Design/g-micro-icon.svg" alt="G-Micro" width="150">

# G-Micro

**100M parameters. 100% Gzowo.**

A ~110M parameter Polish language model, written and trained from scratch — tokenizer, architecture, training loop, inference. No fine-tuning of someone else's weights, no API behind it.

The point is not to compete with large models. It is to own every layer, and to be able to look inside one while it runs.

## What it is

| | |
|---|---|
| Parameters | 109,529,856 |
| Architecture | 12 layers × 12 heads × 768 dim, 1024 context |
| Tokenizer | byte-level BPE, 32k vocab, trained on Polish |
| Corpus | Wikipedia PL + FineWeb-2 `pol_Latn`, 2.0B tokens |
| Runs on | CPU, offline — an Intel MacBook Pro is the target |

The architecture is GPT-2 sized but modernised: RoPE instead of learned positional embeddings, RMSNorm instead of LayerNorm, SwiGLU instead of a GELU MLP, no biases, tied embeddings. Essentially Llama in miniature.

## Honest expectations

At 110M trained from scratch, this is GPT-2-class (2019), not ChatGPT. It writes grammatical Polish with correct inflection — Polish morphology is hard, and a model this size handles it. It is **not** reliable on facts and will confabulate. That is a ceiling of the size, not a bug in the implementation, and it is why retrieval over a local note vault does the factual work while the model does the language.

## Why a custom tokenizer

GPT-2's tokenizer was fitted to English. On Polish it shatters words into fragments and splits every diacritic into two broken bytes:

```
ours :  Naj | mniej | szy | wspólny | mianow | nik          6 tokens
GPT-2:  N | aj | mn | ie | js | zy | w | sp | ó | l | ny | m | ian | own …   15 tokens
```

26–60% fewer tokens across ordinary Polish sentences, which at a 1024-token context means roughly twice as much text fits.

## Measured, not assumed

Every claim here has a number behind it.

| | |
|---|---|
| Loss at random init | 10.55 vs ln(32000) = 10.37 ✓ |
| KV-cache vs none, 900-token context | **8.95× faster** (1.4 → 12.1 tok/s) |
| KV-cache correctness | max logit delta 1.79e-06, identical sampled sequences |
| Characters per token (Polish) | 4.37 |

Without a KV-cache, generation cost grows with the square of the reply length and collapses from 14.6 to 1.4 tok/s as context fills. That is the difference between usable and not.

## Layout

```
model/gpt.py           the model — RoPE, RMSNorm, SwiGLU, KV-cache, Capture hooks
data/train_tokenizer.py  Polish BPE
data/fetch_corpus.py     corpus download (Wikipedia + FineWeb-2)
data/pack_data.py        text → uint16 token binary
train/train.py           AdamW, cosine schedule, resumable checkpoints
Design/                  logo sources and built icon assets
WALKTHROUGH.md           how every layer works, and why it is that way
UI-SPEC.md               specification for the desktop app
```

`Capture` in `model/gpt.py` records attention, layer activations and FFN firings during a forward pass — disabled by default so training pays nothing for it. It is what the desktop app's live network view reads from.

## Status

Model, tokenizer, KV-cache and training loop are written and verified. Corpus is downloaded. Pretraining and the desktop app are next.

## Licence

MIT.
