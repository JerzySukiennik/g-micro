"""
Inflection probe: does G-Micro prefer grammatically correct Polish over a
minimally-broken variant of the same sentence?

Scores each pair in inflection_pairs.PAIRS by total log-likelihood under
teacher forcing and checks whether the grammatical sentence wins. Reports
overall accuracy and a per-category breakdown, since "does it get genitive
right" and "does it get verb-person agreement right" are different claims
that a single number would hide.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import score_sentence  # noqa: E402
from inflection_pairs import PAIRS  # noqa: E402


def run(model, tok, verbose=False):
    results = []
    for pair in PAIRS:
        s_correct, _ = score_sentence(model, tok, pair["correct"])
        s_incorrect, _ = score_sentence(model, tok, pair["incorrect"])
        won = s_correct > s_incorrect
        results.append({**pair, "log_correct": s_correct,
                        "log_incorrect": s_incorrect, "correct_won": won})
        if verbose:
            mark = "✓" if won else "✗"
            print(f"  {mark} [{pair['category']}] {pair['correct']}")

    overall = sum(r["correct_won"] for r in results) / len(results)

    by_cat = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r["correct_won"])
    per_category = {c: sum(v) / len(v) for c, v in sorted(by_cat.items())}

    return {"overall_accuracy": overall, "per_category": per_category,
            "n_pairs": len(results), "results": results}


if __name__ == "__main__":
    from common import load_model, load_tokenizer
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "../Niepotrzebne/kaggle-orchestration/output/run/best.pt"
    model, step, _ = load_model(ckpt)
    tok = load_tokenizer()
    out = run(model, tok, verbose=True)
    print(f"\nstep {step} — overall: {out['overall_accuracy']*100:.1f}% "
          f"({sum(r['correct_won'] for r in out['results'])}/{out['n_pairs']})")
    for cat, acc in out["per_category"].items():
        print(f"  {cat:22} {acc*100:5.1f}%")
