# MicroG — Handoff

Rewritten 2026-07-27 (the previous version was three days stale and cost the
next session twenty minutes rediscovering work that was already done — keep
this one current).

## What this project is

A ~109.5M parameter Polish language model trained **entirely from scratch** by
Jurek (13, vibecoder, GitHub `JerzySukiennik`) on free Kaggle GPU quota, and
run offline afterwards on his Intel MacBook Pro (i9, no CUDA/MLX). Slogan:
"100M parameters. 100% Gzowo." Dual goal: a working model **and** Jurek
understanding every layer (`WALKTHROUGH.md`).

Repo: `github.com/JerzySukiennik/microg`. Local:
`/Users/jurek/Downloads/Claude/Projects/AIe/MicroG`.

## Current state

**Pretraining: done.** Step 4060, perplexity 21.96. Best checkpoint saved at
step 4000 (val 3.0862) — that is the SFT base and it is correct.

**SFT: four improvement rounds done 2026-07-27. Round D is deployed** at
`checkpoints/sft/best.pt` (step 900, val 2.2689).

**Nothing is running.** Both Kaggle kernels report COMPLETE.

### Scores — `python bench/chat_eval.py`

| metric | round A | round B | round C | **round D** |
|---|---|---|---|---|
| identity (diacritics / plain) | 100/100 | 100/100 | 100/100 | **100/100** |
| no identity leak | 92% | 92% | 100% | **100%** |
| grounding — numbers | 38% | 46% | 53% | **65%** |
| grounding — names | 57% | 71% | 42%¹ | **75%** |
| grounding — long contexts | — | — | 11% | **44%** |
| list following / correct | 70/29% | 80/43% | 90/71% | **90/71%** |
| OVERALL | 78.7% | 82.6% | 87.9% | **89.8%** |

¹ round C's name score fell when long-context cases were added to the suite —
the task got harder, not the model worse.

Backups: `checkpoints/sft-roundA`, `-roundB`, `-roundC` and two `-prev`
copies. Each is 2.4GB and the directory totals 20GB — **safe to prune down to
roundC + the deployed one** (~12GB back). Nobody has done it yet.

## What each round changed, and why it worked

- **A** — context QA 1,200 → 30,000 examples, source switched from PoQuAD's
  `dev` split to `train` (46k answerable QAs, never previously used). Loss is
  computed on replies only, so 1,200 short answers were ~0.2% of all trained
  tokens. Fixed **names**, left numbers untouched.
- **B** — numeric answers oversampled to 45%; `IDENTITY_REPEATS` 60 → 40.
  Marginal.
- **C** — 1,680 synthetic "list N things" rows on categories **disjoint from
  the benchmark's** (vegetables, months, trades vs the suite's fruit, animals,
  planets), so the score measures generalisation rather than memorisation. It
  generalised, killed the identity leak, and unexpectedly lifted numbers too.
- **D** — 9,000 **copy drills**: invented towns, scholars and companies
  described through nine-plus facts of the same kind, so six competing years
  or five competing counts sit in every context and only exact retrieval gives
  the exact answer. Values are random per example and every answer appears
  verbatim in its context, so nothing is memorisable. This is what took long
  context grounding from 11% to 44%.

Three hypotheses were **measured and rejected** before round D, and re-testing
them would waste quota: more epochs (round C's val went 2.2960 → 2.2875 →
2.3398, already rising), longer training contexts (PoQuAD's median is 776
characters, 78% over 600), and rephrased-versus-verbatim answers (74.5% of
`generative_answer` values already appear literally in the context).

## Known ceiling

Multi-digit numbers. Failures are consistently off-by-one or digit-corrupted:
1417→1418, 1963→1964, 8 431→8 321. The tokenizer splits `1417` into
`['14','17']` so the model copies the first token and misses the second, and
years like `1963`/`1964` are single adjacent tokens with near-identical
embeddings. Round B tripled numeric training data for +8 points, so this is
representation precision at 110M, not a data shortage. **Do not spend another
round on it.** Improving factual numbers means either a bigger model or
extracting the value in the app rather than asking the model to reproduce it.

## The benchmark lies more often than the model does

Five times on 2026-07-27 a "model failure" was a measurement bug, and once it
caused a wrong rollback. Read raw answers in
`Niepotrzebne/chat_eval_last.json` before believing any score.

1. Sampling hides regressions — the identity leak measured 0/6 at temperature
   0.7 and 1/12 greedy. The suite is greedy on purpose. Do not "improve" it by
   sampling.
2. A regex counted "1) Niebieski 2) Żółty 3) Zielony" as one item.
3. Five cases per category put a whole verdict inside its own noise; round B
   was rolled back as a regression and re-measuring reversed it.
4. Counting list items without reading them scored "Ziemia, Europa,
   Antarktyda" as a perfect list of planets.
5. Vocabulary gaps read as model errors — "wróbel" missed because the stem
   "wróbl" was listed, "jaguar" and "jaskółka" simply absent.

If a score moves by one case, it has not moved.

## Desktop app (`app/` + `runtime/`)

Rebuilt 2026-07-27 as a plain chatbot at Jurek's request: one centred reading
column, Polish copy, a welcome screen with four suggestions, no settings.
Sampling is fixed (temperature 0.7, top_k 40, RAG off) in `GENERATION` in
`app/renderer/app.js`. The neurons/probabilities panel still exists and is one
shortcut away (⌥⌘D) — **do not delete it**, it is half the point of the
project; Jurek was told it was kept and can still ask for it to go.

`runtime/server.py` is the WebSocket backend on port 8899; Electron spawns and
kills it. Launch with `npm start` in `app/`. `/Applications/MicroG.app` is a
thin launcher stub, not a packaged build — deliberate.

### RAG — fixed, and now genuinely useful

It was badly broken and made the model look insane: with RAG on, "Jak masz na
imię?" answered **"Argentyńskie"** because `opensearch` matched the *phrasing
of the question* to a Wikipedia article titled "Jak mam na imię" — a 2012
Argentine film — and the model faithfully answered from it. Every guard
(STOPWORDS, MIN_SCORE) lived in `search_vault()` and never touched the
Wikipedia path.

Now: full-text `list=search`, identity questions and greetings never retrieve,
and a relevance check compares folded five-character prefixes because Polish
inflects everything. The UI shows `kontekst: wikipedia:X` above any answer
built from a document — retrieval used to be invisible, which is what made a
bad hit indistinguishable from a broken model.

## Kaggle

Account `jerzysukiennik` (do not create a second one to dodge quota).
Orchestration lives in `Niepotrzebne/kaggle-orchestration/` (gitignored, local
only). To run another SFT round:

```bash
cd "/Users/jurek/Downloads/Claude/Projects/AIe/MicroG/Niepotrzebne/kaggle-orchestration"
/Users/jurek/Downloads/Claude/Projects/AIe/MicroG/.venv/bin/python run_finetune.py
```

One-shot, bounded, safe to leave. ~50-70 minutes. **The kernel does
`git clone` from GitHub, so push `data/build_sft.py` before running it** or
the GPU grinds the old data.

Traps, all previously paid for in GPU hours:

- `/kaggle/input/` mount depth is not fixed — always `glob("**/…",
  recursive=True)`. A fixed-depth glob silently found nothing and restarted
  pretraining from a stale checkpoint.
- `kernel/train_kernel.py` (deployed) and `kaggle/02-train.py` (repo mirror)
  drift silently. Same for `kernel-finetune/finetune_kernel.py` vs
  `kaggle/03-finetune.py`.
- Kernel status has more states than four: `CANCEL_REQUESTED` (still running)
  and `CANCEL_ACKNOWLEDGED` (can hang 20+ minutes after it is really done).
  Parse the exact `KernelWorkerStatus.<STATE>` token.
- `torch.load(..., weights_only=False)` is required — checkpoints store `Path`
  objects.
- `kaggle kernels output` only serves the *latest* version. Harvest before
  pushing the next one.
- The `kaggle` CLI shim has a stale shebang from before the project moved into
  `AIe/`. Use `.venv/bin/python -m kaggle`.

## Working with Jurek on this

He is 13 and a vibecoder — he does not touch code or the terminal, and he is
genuinely interested in *why*, not just whether it works. Explain the
mechanism. When something looks odd in the output, dig for a real cause before
reassuring him: on 2026-07-27 every "huh, that's odd" turned out to be a real
bug — the diacritics gap, the RAG disaster, five benchmark defects. He gives
direct feedback and dislikes over-engineered answers to questions he did not
ask.

## Open

- Playtest round D live (RAG is worth trying again now).
- Prune ~12GB of redundant checkpoint backups.
- Optional: show the source span in the app next to a grounded answer, so
  numbers the model garbles are still visible to the reader. Cheaper and more
  reliable than another training round.
