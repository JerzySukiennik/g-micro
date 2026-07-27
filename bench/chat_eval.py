"""Chat-behaviour benchmark for the instruction-tuned (SFT) model.

The rest of bench/ measures the *base* model's language modelling — perplexity,
inflection, speed. None of it can see whether the chat model knows its own name,
follows an instruction, stops when it is done, or leaks identity into unrelated
answers. Those are exactly the things every SFT round has changed so far, and
until now each round was judged by talking to the model and forming an
impression. This turns that impression into numbers so two checkpoints can be
compared honestly.

Everything is decoded greedily (temperature 0.1, top_k 1). That is deliberate:
at temperature 0.7 a bad mode is often hidden by sampling luck — the identity
leak introduced on 2026-07-27 measured 0/6 under sampling and 1/12 under greedy
decoding, and the greedy number was the true one. A regression you can only see
sometimes is still a regression.

Usage:
    python bench/chat_eval.py                        # checkpoints/sft/best.pt
    python bench/chat_eval.py path/to/best.pt        # a specific checkpoint
    python bench/chat_eval.py a.pt b.pt              # compare two, side by side
"""

import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_model, load_tokenizer, score_sentence  # noqa: E402

U, A, EOT, CTX = "<|user|>", "<|assistant|>", "<|endoftext|>", "<|context|>"
REPO = Path(__file__).resolve().parents[1]
# Generation stops at <|endoftext|>, so this budget only costs time on answers
# that fail to terminate — which is exactly what stops_cleanly measures. It was
# 48, which turned out to measure the budget rather than the model: "Czym jest
# Warszawa?" writes a complete 43-word answer and emits EOT just past the old
# cutoff, and was being scored as a failure to stop.
MAX_NEW = 128

# --------------------------------------------------------------- test sets --

IDENTITY_DIACRITICS = [
    "Jak masz na imię?", "Jak się nazywasz?", "Kim jesteś?",
    "Czy masz imię?", "Przedstaw się.", "Jak brzmi twoja nazwa?",
]
IDENTITY_PLAIN = [
    "Jak masz na imie?", "Jak sie nazywasz?", "Kim jestes?",
    "Czy masz imie?", "Przedstaw sie.", "Czy jestes Claude?",
]
# Unrelated prompts, weighted toward shapes closest to the identity examples
# ("Opowiedz ... o X" mirrors the trained "Opowiedz coś o sobie."), because
# those are where over-training identity actually bleeds through.
UNRELATED = [
    "Opowiedz krótko o psach.", "Opowiedz coś o Krakowie.",
    "Opowiedz o drugiej wojnie światowej.", "Przedstaw najważniejsze fakty o Księżycu.",
    "Czym jest Warszawa?", "Co to jest fotosynteza?", "Jak zrobić naleśniki?",
    "Wymień trzy planety.", "Jaki jest największy ocean?",
    "Kto napisał Pana Tadeusza?", "Wyjaśnij czym jest grawitacja.",
    "Podaj przepis na zupę pomidorową.",
]
IDENTITY_MARKERS = ("MicroG", "modelem językowym", "model językowy", "110 milionów")

# (context, question, strings that a grounded answer should contain).
# The point is not world knowledge — it is whether the model reads the text in
# front of it. Facts are deliberately checkable only from the context.
CONTEXT_CASES = [
    ("Zamek w Bąkowie został zbudowany w 1417 roku przez rycerza Mikołaja Warkosza. "
     "Zamek ma cztery wieże i fosę o głębokości sześciu metrów.",
     "W którym roku zbudowano zamek w Bąkowie?", ["1417"]),
    ("Zamek w Bąkowie został zbudowany w 1417 roku przez rycerza Mikołaja Warkosza. "
     "Zamek ma cztery wieże i fosę o głębokości sześciu metrów.",
     "Ile wież ma zamek w Bąkowie?", ["cztery", "4"]),
    ("Firma Nortem wyprodukowała w zeszłym roku 12 400 rowerów elektrycznych. "
     "Jej siedziba mieści się w Gdyni, a założycielem jest Anna Reut.",
     "Kto założył firmę Nortem?", ["Anna Reut", "Reut"]),
    ("Firma Nortem wyprodukowała w zeszłym roku 12 400 rowerów elektrycznych. "
     "Jej siedziba mieści się w Gdyni, a założycielem jest Anna Reut.",
     "Gdzie mieści się siedziba firmy Nortem?", ["Gdyni", "Gdynia"]),
    ("Rzeka Skawica ma 42 kilometry długości i wpada do Skawy w miejscowości Białka. "
     "Nad rzeką leżą trzy młyny wodne.",
     "Ile kilometrów ma rzeka Skawica?", ["42"]),
]

# "List exactly N things" — checks instruction following, and catches the
# "1) Azja 2.) Azja" degenerate repeat seen on 2026-07-27.
LIST_CASES = [
    ("Wymień trzy planety.", 3), ("Wymień trzy owoce.", 3),
    ("Wymień trzy kolory.", 3), ("Podaj dwa polskie miasta.", 2),
]

# Plain, unambiguous Polish. Lower loss per token = the model finds ordinary
# Polish more plausible. Same sentences every run, so runs are comparable.
PROBES = [
    "Warszawa jest stolicą Polski i największym miastem w kraju.",
    "Fotosynteza to proces, w którym rośliny wytwarzają cukry z dwutlenku węgla.",
    "Wczoraj wieczorem poszedłem do sklepu po chleb i mleko.",
    "Komputer składa się z procesora, pamięci operacyjnej i dysku twardego.",
    "W niedzielę pojechaliśmy nad jezioro, żeby popływać kajakiem.",
]


# ----------------------------------------------------------------- helpers --

def make_asker(model, tok):
    eot_id = tok.encode(EOT).ids[0]

    def ask(prompt_text, max_new=MAX_NEW):
        ids = tok.encode(prompt_text).ids
        out, hit_eot = [], False
        for tid, _ in model.generate(torch.tensor([ids]), max_new_tokens=max_new,
                                     temperature=0.1, top_k=1):
            if tid == eot_id:
                hit_eot = True
                break
            out.append(int(tid))
        return tok.decode(out).strip(), hit_eot

    return ask


def chat_prompt(question, context=None):
    if context:
        return f"{CTX}\n{context}\n\n{U}\n{question}\n{A}\n"
    return f"{U}\n{question}\n{A}\n"


def has_repeat_loop(text, min_len=4):
    """True if some phrase repeats back-to-back — the failure mode
    repetition_penalty was added for, checked here so a regression in it
    cannot pass silently."""
    words = text.split()
    for size in range(1, 6):
        for i in range(len(words) - size * 2 + 1):
            a = words[i:i + size]
            if a == words[i + size:i + size * 2] and len(" ".join(a)) >= min_len:
                return True
    return False


def count_list_items(text):
    """Number of DISTINCT items in a list answer.

    Distinct, because the failure worth catching is three slots filled with the
    same word, not a short list. Items may be numbered ("1) x 2) x"), bulleted,
    or plain comma-separated, and are frequently all on one line, so splitting
    on newlines alone misses most of them — the marker itself has to be the
    delimiter.
    """
    marker = re.compile(r"(?:\d+\s*[.)]|^\s*[-*])\s*", re.M)
    if marker.search(text):
        parts = marker.split(text)
    else:
        parts = re.split(r"[,;\n]", text)
    norm = set()
    for p in parts:
        # Keep the first few words: "Kraków: Miasto, które ma najwięcej…" and
        # "Kraków" are the same item, and a trailing empty marker ("3.)") is not
        # an item at all.
        key = re.sub(r"[^\wąćęłńóśźżĄĆĘŁŃÓŚŹŻ ]+", " ", p.lower()).strip()
        key = " ".join(key.split()[:2])
        if len(key) > 1:
            norm.add(key)
    return len(norm)


# ------------------------------------------------------------------- suite --

def evaluate(ckpt_path):
    model, step, best_val = load_model(ckpt_path)
    tok = load_tokenizer()
    ask = make_asker(model, tok)
    res = {"checkpoint": str(ckpt_path), "step": step, "best_val": best_val}
    detail = {}

    def section(name, cases, check):
        hits, rows = 0, []
        for case in cases:
            q = case if isinstance(case, str) else case[0]
            prompt = chat_prompt(q) if isinstance(case, str) else None
            ans, eot = ask(prompt or chat_prompt(q))
            ok = check(case, ans)
            hits += ok
            rows.append({"q": q, "a": ans, "ok": bool(ok), "eot": eot})
        res[name] = hits / len(cases)
        detail[name] = rows
        return rows

    section("identity_diacritics", IDENTITY_DIACRITICS,
            lambda c, a: "MicroG" in a)
    section("identity_plain", IDENTITY_PLAIN,
            lambda c, a: "MicroG" in a)
    # Inverted: passing means NOT sounding like an identity answer.
    section("no_identity_leak", UNRELATED,
            lambda c, a: not any(m in a for m in IDENTITY_MARKERS))

    # Context grounding needs its own loop — the prompt carries a context block.
    hits, rows = 0, []
    for ctx, q, expect in CONTEXT_CASES:
        ans, eot = ask(chat_prompt(q, context=ctx))
        ok = any(e.lower() in ans.lower() for e in expect)
        hits += ok
        rows.append({"q": q, "a": ans, "ok": bool(ok), "expect": expect})
    res["context_grounding"] = hits / len(CONTEXT_CASES)
    detail["context_grounding"] = rows

    hits, rows = 0, []
    for q, n in LIST_CASES:
        ans, eot = ask(chat_prompt(q))
        got = count_list_items(ans)
        ok = got >= n
        hits += ok
        rows.append({"q": q, "a": ans, "ok": bool(ok), "want": n, "got": got})
    res["list_following"] = hits / len(LIST_CASES)
    detail["list_following"] = rows

    # Termination and repetition, measured over everything already generated.
    all_rows = [r for k in ("identity_diacritics", "identity_plain",
                            "no_identity_leak", "list_following")
                for r in detail[k]]
    res["stops_cleanly"] = sum(r.get("eot", False) for r in all_rows) / len(all_rows)
    res["no_repeat_loop"] = 1 - sum(has_repeat_loop(r["a"]) for r in all_rows) / len(all_rows)

    tot_lp = tot_n = 0
    for s in PROBES:
        lp, n = score_sentence(model, tok, s)
        tot_lp += lp
        tot_n += n
    res["polish_loss_per_token"] = -tot_lp / tot_n

    del model
    return res, detail


SCORES = ["identity_diacritics", "identity_plain", "no_identity_leak",
          "context_grounding", "list_following", "stops_cleanly", "no_repeat_loop"]


def main():
    paths = [Path(p) for p in sys.argv[1:]] or [REPO / "checkpoints" / "sft" / "best.pt"]
    results, details = [], []
    for p in paths:
        if not p.exists():
            raise SystemExit(f"no such checkpoint: {p}")
        print(f"evaluating {p} …", flush=True)
        r, d = evaluate(p)
        results.append(r)
        details.append(d)

    width = max(len(s) for s in SCORES) + 2
    print(f"\n{'metric':<{width}}" + "".join(f"{Path(r['checkpoint']).parent.name:>16}"
                                             for r in results))
    print("-" * (width + 16 * len(results)))
    for s in SCORES:
        print(f"{s:<{width}}" + "".join(f"{r[s] * 100:>15.0f}%" for r in results))
    print(f"{'OVERALL':<{width}}" +
          "".join(f"{sum(r[s] for s in SCORES) / len(SCORES) * 100:>15.1f}%" for r in results))
    print(f"\n{'polish_loss/token':<{width}}" +
          "".join(f"{r['polish_loss_per_token']:>16.4f}" for r in results) + "   (lower better)")
    print(f"{'sft val_loss':<{width}}" +
          "".join(f"{r['best_val']:>16.4f}" for r in results) +
          "   (NOT comparable across different data)")

    out = REPO / "Niepotrzebne" / "chat_eval_last.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"results": results, "details": details},
                              ensure_ascii=False, indent=2))
    print(f"\nfull answers written to {out}")


if __name__ == "__main__":
    main()
