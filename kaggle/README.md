# Trening G-Micro na Kaggle — instrukcja

Kaggle daje **30 h GPU tygodniowo za darmo**, w tym T4 ×2. To wystarczy na cały trening. Sesja jest ucinana po 12 h i może paść wcześniej bez ostrzeżenia, więc wszystko poniżej jest zbudowane pod wznawianie.

Czego potrzebujesz: konta na kaggle.com z **zweryfikowanym numerem telefonu** (bez weryfikacji nie ma GPU ani internetu w notebookach).

---

## Dlaczego nie wysyłamy danych z domu

Spakowany korpus to ~3,8 GB. Przy twoim łączu wysyłka trwałaby kilka godzin. Kaggle ma szybki internet, więc **pobieramy i pakujemy dane na Kaggle**, raz, i zapisujemy jako Dataset. Potem każda sesja treningowa startuje z gotowych binariów w kilka sekund.

---

## Krok 1 — przygotowanie danych (raz, ~1 h)

1. **New Notebook**
2. Prawy panel → **Accelerator: None**, **Internet: On**
3. *(opcjonalnie, przyspiesza pobieranie)* **Add-ons → Secrets → Add secret**, nazwa `HF_TOKEN`, wartość = twój token z HuggingFace
4. Wklej zawartość [`01-prep.py`](01-prep.py) do komórki i uruchom
5. Po zakończeniu: **Save Version → Save & Run All**, poczekaj aż się wykona
6. Na stronie wersji: **Output → New Dataset**, nazwij **`microg-data`**

Powstaną `pl_train.bin`, `pl_val.bin` i `tokenizer.json`.

## Krok 2 — trening (~10 h na T4 ×2)

1. **New Notebook**
2. **Accelerator: GPU T4 x2**, **Internet: On**, **Persistence: Variables and Files**
3. **Add Input** → twój dataset `microg-data`
4. Wklej [`02-train.py`](02-train.py) i uruchom
5. Gdy sesja się skończy (albo padnie): **Save Version**, potem **Output → New Dataset**, nazwij **`microg-ckpt`**

## Krok 3 — wznowienie (jeśli 12 h nie starczyło)

Nowy notebook jak w kroku 2, ale **Add Input** dwa razy: `microg-data` **oraz** `microg-ckpt`. Skrypt sam wykryje checkpoint i podejmie od ostatniego kroku — razem z momentami Adama, więc bez skoku loss.

Po każdej sesji nadpisuj `microg-ckpt` nowym outputem (**Output → Update Dataset**).

---

## Czego się spodziewać

| | |
|---|---|
| Tokenów na krok | 491 520 (16 × 30 × 1024) |
| Kroków | 4060 → 2,0 mld tokenów |
| Czas | ~10 h na T4 ×2, ~17 h na jednym |
| Checkpoint co | 100 kroków (~13 min pracy w ryzyku) |
| Loss na starcie | **10,4** — jeśli widzisz mniej, coś jest zepsute |
| Loss na końcu | orientacyjnie 3,3–3,8 (perplexity ~30–45) |

**Loss startowy to najważniejsza liczba w pierwszej minucie.** Musi wynosić ≈ `ln(32000)` = 10,37, bo model nie wie jeszcze nic i zgaduje równomiernie ze słownika. Wartość *niższa* oznacza wyciek targetów, nie sukces — wyłącz i zgłoś.

## Jak sprawdzić, czy idzie dobrze

Po ~300 krokach loss powinien być w okolicach 6, po 1000 poniżej 5. Jeśli stoi na 10,4 przez kilkaset kroków — learning rate jest za niski albo dane są zepsute. Jeśli skacze do `nan` — za wysoki (zbij `--lr` na `3e-4`).

## Jak coś nie działa

- **`CUDA out of memory`** → zbij `BATCH` do 8 i podnieś `ACCUM` do 60 (ta sama liczba tokenów na krok)
- **DataParallel się sypie** → dopisz `"--single-gpu"` do listy `cmd` w `02-train.py`; trening potrwa ~17 h zamiast 10, ale jest prostszy
- **Notebook nie widzi danych** → sprawdź, czy dataset jest faktycznie dodany w **Add Input**, nie tylko utworzony
