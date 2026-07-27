"""Does showing the model past turns help, or does it break it?

Every SFT example this model saw was a single <|user|>/<|assistant|> pair.
Feeding it a conversation is therefore out of distribution: the *format* of each
turn is familiar, the *sequence* of several is not. That could go either way,
and "chat apps have memory" is not evidence about this model.

Two things are measured, because history could help one and hurt the other:

  Follow-ups. Questions that are meaningless without the previous turn — "a
  rozwiń to", "a co z drugim". With no history the model cannot answer these
  even in principle, so any score above zero is memory working.

  Contamination. The single-turn questions from chat_eval, asked again with
  unrelated history in front of them. If the score drops, history is costing
  quality on ordinary questions, and that cost applies to every message rather
  than only the follow-ups.

    python bench/multiturn_eval.py
"""

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "bench"))
from common import load_model, load_tokenizer  # noqa: E402

U, A, EOT = "<|user|>", "<|assistant|>", "<|endoftext|>"

# Filler exchanges used as "unrelated history" for the contamination test.
FILLER = [
    ("Wymień trzy kolory.", "1) niebieski 2) zielony 3) żółty"),
    ("Jak zrobić herbatę?", "Zalej torebkę herbaty wrzątkiem i odczekaj trzy minuty."),
    ("Podaj dwa polskie miasta.", "1) Kraków 2) Gdańsk"),
    ("Co to jest rower?", "Rower to pojazd napędzany siłą mięśni, poruszający się na dwóch kołach."),
]

# (history, question, strings that show the model used the history)
FOLLOWUPS = [
    ([("Opowiedz o psach.", "Psy to zwierzęta domowe, które żyją z ludźmi od tysięcy lat.")],
     "A koty?", ["kot", "Kot"]),
    ([("Wymień trzy owoce.", "1) jabłka 2) gruszki 3) śliwki")],
     "A warzywa?", ["marchew", "ziemniak", "pomidor", "ogórek", "kapust", "cebul", "burak", "warzyw"]),
    ([("Czym jest Kraków?", "Kraków to miasto w południowej Polsce, dawna stolica kraju.")],
     "A Gdańsk?", ["Gdańsk", "morz", "północ", "port"]),
    ([("Jak masz na imię?", "Nazywam się G-Micro.")],
     "A kto cię stworzył?", ["Jurek", "Jurka"]),
    ([("Wymień dwa instrumenty.", "1) gitara 2) pianino")],
     "A jeszcze dwa?", ["skrzyp", "perkus", "flet", "trąb", "bęb", "harf", "saksof", "klarnet"]),
]

# Single-turn checks reused from the chat benchmark's strongest categories.
# Deliberately eighteen rather than six: at six, one flipped answer moves the
# score by 17 points and the contamination test cannot tell a real regression
# from noise — the same trap this project fell into on 2026-07-27 when a round
# was rolled back on a six-case verdict and re-measuring reversed it.
SINGLE = [
    ("Jak masz na imię?", ["MicroG", "G-Micro"]),
    ("Kim jesteś?", ["MicroG", "G-Micro"]),
    ("Jak masz na imie?", ["MicroG", "G-Micro"]),
    ("Kim jestes?", ["MicroG", "G-Micro"]),
    ("Czy masz imię?", ["MicroG", "G-Micro"]),
    ("Przedstaw się.", ["MicroG", "G-Micro"]),
    ("Wymień trzy kolory.", ["niebiesk", "zielon", "żółt", "czerwon", "czarn", "biał"]),
    ("Wymień trzy owoce.", ["jabłk", "grusz", "śliw", "banan", "jagod", "malin", "truskaw"]),
    ("Wymień trzy zwierzęta.", ["pies", "psa", "kot", "koń", "krow", "lis", "wilk", "sarn", "jeleń", "ptak"]),
    ("Podaj dwa polskie miasta.", ["krak", "warszaw", "gdańsk", "wrocław", "poznań", "łódź"]),
    ("Wymień trzy dni tygodnia.", ["poniedział", "wtorek", "środ", "czwartek", "piątek", "sobot", "niedziel"]),
    ("Wymień cztery pory roku.", ["wiosn", "lat", "jesień", "zim"]),
    ("Czym jest Warszawa?", ["Warszaw", "miast", "Polsk"]),
    ("Co to jest fotosynteza?", ["roślin", "światł", "proces"]),
    ("Opowiedz krótko o psach.", ["pies", "psy", "psów", "zwierz"]),
    ("Jak zrobić naleśniki?", ["mąk", "jajk", "mlek", "ciast", "patelni"]),
    ("Jaki jest największy ocean?", ["ocean", "spokojn", "wod"]),
    ("Wyjaśnij czym jest grawitacja.", ["siła", "przycią", "ciał", "ziemi"]),
]


def build(text, history):
    parts = [f"{U}\n{q}\n{A}\n{a}{EOT}\n" for q, a in history]
    parts.append(f"{U}\n{text}\n{A}\n")
    return "".join(parts)


def main():
    model, step, _ = load_model(REPO / "checkpoints" / "sft" / "best.pt")
    tok = load_tokenizer()
    eot = tok.encode(EOT).ids[0]
    print(f"checkpoint step={step}\n")

    def ask(text, history):
        ids = tok.encode(build(text, history)).ids
        out = []
        for tid, _ in model.generate(torch.tensor([ids]), max_new_tokens=48,
                                     temperature=0.1, top_k=1):
            if tid == eot:
                break
            out.append(int(tid))
        return tok.decode(out).strip()

    print("=== 1. PYTANIA KONTEKSTOWE (bez historii są niemożliwe) ===")
    for turns in (0, 1):
        hits = []
        for hist, q, expect in FOLLOWUPS:
            h = hist if turns else []
            a = ask(q, h)
            ok = any(e.lower() in a.lower() for e in expect)
            hits.append(ok)
            if turns:
                print(f"  [{'OK ' if ok else '--'}] {q:22} -> {a[:60]}")
        n = sum(hits)
        print(f"  {'z historią' if turns else 'bez historii'}: {n}/{len(FOLLOWUPS)}"
              f" = {n/len(FOLLOWUPS)*100:.0f}%\n")

    print("=== 2. CZY HISTORIA PSUJE ZWYKŁE PYTANIA ===")
    print(f"  {'historia':>10}  {'wynik':>8}")
    base = None
    for turns in (0, 1, 2, 4):
        hits = 0
        for q, expect in SINGLE:
            hist = FILLER[:turns]
            a = ask(q, hist)
            hits += any(e.lower() in a.lower() for e in expect)
        pct = hits / len(SINGLE) * 100
        if base is None:
            base = pct
        delta = "" if turns == 0 else f"   {pct - base:+.0f} pkt"
        print(f"  {turns:>7} wym.  {hits}/{len(SINGLE)} = {pct:3.0f}%{delta}")

    print("\nWniosek zależy od obu liczb naraz: historia ma sens tylko wtedy,")
    print("gdy pytania kontekstowe działają, a zwykłe nie tracą.")


if __name__ == "__main__":
    main()
