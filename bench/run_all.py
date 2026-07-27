"""
G-Micro benchmark suite — run everything against a checkpoint, produce a
markdown report and a JSON file with the raw numbers.

Usage:
    python bench/run_all.py <checkpoint.pt> [--out results/step2000.md]

Every benchmark here is deliberately scaled to what a 110M from-scratch base
model can meaningfully be judged on: perplexity and top-k accuracy against
held-out Polish text, a hand-built probe for the one thing this project set
out to prove (correct Polish inflection), and offline CPU throughput — the
product's actual promise. Modern instruction-following benchmarks (MMLU,
HumanEval, ...) are not here on purpose: at this scale they would measure
floor noise, not model quality.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_model, load_tokenizer  # noqa: E402
import perplexity, topk, inflection_probe, speed  # noqa: E402
from inflection_pairs import PAIRS as INFLECTION_PAIRS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--skip-speed", action="store_true",
                    help="speed.py takes ~1-2 min on CPU; skip for a quick pass")
    args = ap.parse_args()

    print(f"loading {args.checkpoint} ...")
    model, step, best_val = load_model(args.checkpoint)
    tok = load_tokenizer()

    print("perplexity ...")
    ppl = perplexity.run(model)
    print(f"  loss {ppl['loss']:.4f}  perplexity {ppl['perplexity']:.2f}")

    print("top-k accuracy ...")
    tk = topk.run(model)
    print("  " + "  ".join(f"{k} {v*100:.1f}%" for k, v in tk.items() if k != "tokens"))

    print(f"inflection probe ({len(INFLECTION_PAIRS)} pairs) ...")
    infl = inflection_probe.run(model, tok)
    print(f"  overall {infl['overall_accuracy']*100:.1f}%")

    spd = None
    if not args.skip_speed:
        print("speed (cached vs uncached) ...")
        spd = speed.run(model)
        print(f"  cached {spd['cached_tok_s']:.1f} tok/s  "
              f"uncached {spd['uncached_tok_s']:.1f} tok/s")

    report = {
        "checkpoint": str(args.checkpoint),
        "step": step,
        "training_best_val": best_val,
        "perplexity": ppl,
        "topk": tk,
        "inflection": {"overall_accuracy": infl["overall_accuracy"],
                       "per_category": infl["per_category"],
                       "n_pairs": infl["n_pairs"]},
        "speed": spd,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    md = render_markdown(report)
    print("\n" + md)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md)
        args.out.with_suffix(".json").write_text(json.dumps(report, indent=2))
        print(f"\nwritten -> {args.out}  /  {args.out.with_suffix('.json')}")


def render_markdown(r):
    lines = [
        f"# G-Micro benchmark — step {r['step']}",
        f"_{r['timestamp']}_",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Perplexity (held-out) | {r['perplexity']['perplexity']:.2f} |",
        f"| Top-1 next-token accuracy | {r['topk'].get('top1_accuracy', 0)*100:.1f}% |",
        f"| Top-5 next-token accuracy | {r['topk'].get('top5_accuracy', 0)*100:.1f}% |",
        f"| Inflection probe ({r['inflection']['n_pairs']} pairs) | {r['inflection']['overall_accuracy']*100:.1f}% |",
    ]
    if r["speed"]:
        s = r["speed"]
        speedup = s["cached_tok_s"] / s["uncached_tok_s"] if s["uncached_tok_s"] else float("nan")
        lines.append(f"| Speed, KV-cache (CPU, ctx {s['prompt_len']}) | {s['cached_tok_s']:.1f} tok/s |")
        lines.append(f"| Speed, no cache (same context) | {s['uncached_tok_s']:.1f} tok/s |")
        lines.append(f"| KV-cache speedup | {speedup:.1f}x |")
    lines += ["", "## Inflection probe by category", "", "| Category | Accuracy |", "|---|---|"]
    for cat, acc in r["inflection"]["per_category"].items():
        lines.append(f"| {cat} | {acc*100:.1f}% |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
