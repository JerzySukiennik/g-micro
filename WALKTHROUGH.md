# Jak działa G-Micro — walkthrough

Przewodnik po `model/gpt.py`, warstwa po warstwie. Nie streszczenie — wyjaśnienie *dlaczego* każda część jest taka, a nie inna.

---

## 0. Co model w ogóle robi

Cały model językowy robi **jedną rzecz**: dostaje ciąg tokenów i zgaduje, jaki token będzie następny. Nic więcej. Rozmowa, tłumaczenie, pisanie kodu — to wszystko jest tą samą operacją powtarzaną w kółko, gdzie wynik dokleja się na koniec wejścia i model liczy od nowa.

To jest ważne, bo demistyfikuje całość: model nie „myśli o odpowiedzi", tylko liczy rozkład prawdopodobieństwa nad 32 000 tokenami i losuje z niego jeden.

Wyjście modelu to wektor 32 000 liczb — po jednej na każdy token w słowniku. Po `softmax` sumują się do 1, więc to prawdziwe prawdopodobieństwa. „Model jest pewny" znaczy: jedna liczba blisko 1, reszta blisko 0.

---

## 1. Tokeny — model nie widzi liter

Sieć neuronowa umie mnożyć liczby, nie czytać tekst. Więc najpierw tekst → liczby.

Naiwne pomysły i czemu są złe:
- **litera = token** → ciągi są koszmarnie długie, model marnuje pojemność na uczenie się ortografii
- **słowo = token** → słownik nieskończony; polska fleksja („kot, kota, kotu, kotem…") to osobne wpisy, a nowego słowa nie da się zapisać w ogóle

**BPE** jest pomiędzy. Start: 256 bajtów (pokrywa absolutnie każdy tekst). Potem powtarzaj: znajdź najczęstszą sąsiadującą parę symboli i scal ją w nowy symbol. Po 32 000 scaleń częste słowa są jednym tokenem, rzadkie rozpadają się na kawałki, a nic nie jest niezapisywalne.

Dlatego trenujemy **własny** tokenizer na polskim. Tokenizer GPT-2 uczył się na angielskim, więc polską morfologię tnie na sieczkę — a każdy zmarnowany token to zmarnowany kontekst i zmarnowana moc. Przy 109M nas na to nie stać.

---

## 2. Embedding — token staje się kierunkiem w przestrzeni

```python
self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
```

Tablica 32000 × 768. Token nr 5123 to po prostu wiersz 5123 — wektor 768 liczb.

Te wektory **nie są zaprogramowane, tylko wyuczone**. Na starcie to losowy szum. W trakcie treningu tokeny używane w podobnych kontekstach dryfują ku sobie, aż geometria zaczyna kodować znaczenie: „król" i „królowa" lądują blisko, a kierunek między nimi jest mniej więcej tym samym kierunkiem, co między „mężczyzna" i „kobieta". Nikt tego nie zaprojektował — to wypada z zadania przewidywania następnego tokenu.

768 to **wymiar modelu** (`n_embd`). Wszystko wewnątrz płynie w tej szerokości.

---

## 3. Residual stream — najważniejszy mental model

Spójrz na `Block.forward`:

```python
x = x + self.attn(self.norm_attn(x), cos, sin, capture)
x = x + self.ffn(self.norm_ffn(x), capture)
```

Zwróć uwagę: **`x` nigdy nie jest nadpisywane.** Każda warstwa tylko *dodaje* do niego poprawkę.

Wyobraź sobie `x` jako taśmę biegnącą przez cały model. Każda z 12 warstw czyta taśmę, wymyśla poprawkę i ją dopisuje. Warstwa 1 dokłada coś prostego („to rzeczownik"), warstwa 8 coś abstrakcyjnego („to podmiot pytania"). Nic nie jest kasowane, tylko wzbogacane.

Dwie ważne konsekwencje:

1. **Gradient ma autostradę.** Przy uczeniu sygnał błędu wraca do wcześniejszych warstw przez dodawanie, nie przez 12 mnożeń. Bez residuali głębokie sieci po prostu się nie uczyły — to jest ten wynalazek (ResNet, 2015), który odblokował głębokość.
2. **Możesz podglądać taśmę na dowolnej głębokości** i zobaczyć sensowną, coraz bogatszą reprezentację. Dokładnie to zbiera `capture.activations` i to zobaczysz w panelu — jak reprezentacja tokenu dojrzewa warstwa po warstwie.

---

## 4. Attention — skąd token bierze kontekst

Sam token nic nie znaczy. „Zamek" to budowla, suwak albo mechanizm w drzwiach — zależy od sąsiadów. Attention to mechanizm, którym token **pobiera informację od innych tokenów**.

Każdy token produkuje trzy wektory:

| wektor | pytanie | rola |
|---|---|---|
| **Q** (query) | „czego szukam?" | co ten token chce wiedzieć |
| **K** (key) | „co oferuję?" | po czym inne tokeny mogą go znaleźć |
| **V** (value) | „co przekazuję?" | treść, którą odda, jeśli zostanie wybrany |

Algorytm:
1. Policz `Q · K` dla każdej pary tokenów → tabela dopasowań
2. Podziel przez `√head_dim` — bez tego przy 64 wymiarach iloczyny robią się ogromne, softmax nasyca się do 0/1 i gradienty umierają
3. Zamaskuj przyszłość (`tril`) — token nie może patrzeć w przód
4. `softmax` po wierszach → każdy wiersz sumuje się do 1
5. Weź średnią ważoną wektorów **V** tymi wagami

Czyli: *„policz, jak bardzo mnie obchodzi każdy wcześniejszy token, i weź ich ważoną średnią"*.

**Maska przyczynowa to serce sprawy.** Bez niej model przy przewidywaniu tokenu 5 widziałby token 5 — dostałby 100% trafności i nie nauczyłby się niczego. To dokładnie ten błąd, który złapaliśmy w smoke teście (loss 8.94 zamiast 10.37).

**Wiele głów** (`n_head=12`): zamiast jednego attention na 768 wymiarach robimy 12 równoległych po 64. Każda głowa może śledzić inny typ zależności — jedna zgodność podmiotu z orzeczeniem, inna dopasowanie nawiasów, inna odległe powiązania tematyczne. Na końcu wyniki się sklejają. To dlatego `capture.attn` ma kształt `(12, T, T)` — jedna mapa na głowę, i w panelu będą wyglądać zauważalnie różnie.

### Dwie ścieżki w kodzie

```python
if capture is not None and capture.enabled:
    att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
    ...
else:
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

To **ta sama matematyka**. Wersja fused (`scaled_dot_product_attention`) nigdy nie tworzy pełnej macierzy T×T w pamięci — liczy ją kawałkami i od razu zużywa. Szybciej i drastycznie mniej RAM-u.

Ale skoro macierz nigdy nie powstaje, **nie da się jej narysować**. Dlatego przy włączonym `capture` schodzimy na wolniejszą, jawną ścieżkę. To świadomy kompromis: trening i normalne generowanie są szybkie, a kosztu jawnej wersji nie płacimy nigdy poza momentem, gdy panel faktycznie chce podejrzeć jeden krok.

---

## 5. RoPE — skąd model wie, co jest gdzie

Problem: attention liczy średnią ważoną i **kompletnie nie widzi kolejności**. „Pies gryzie człowieka" i „człowieka gryzie pies" to dla samego attention to samo. Trzeba wstrzyknąć pozycję.

GPT-2 miał drugą tablicę embeddingów — jeden wektor na pozycję — i dodawał go do tokenu. Działa, ale ma wady: pozycja 500 i 501 uczą się niezależnie (model musi *odkryć*, że są sąsiadami), a poza wyuczoną długość nie ma jak wyjść.

**RoPE robi to inaczej: nie dodaje, tylko obraca.** Wektory Q i K są obracane o kąt proporcjonalny do pozycji.

Sztuczka jest w tym, że attention liczy **iloczyn skalarny** Q·K, a iloczyn skalarny dwóch obróconych wektorów zależy tylko od **różnicy** kątów. Obróć oba o ten sam kąt — wynik bez zmian.

Efekt: model automatycznie widzi „ten token jest 3 miejsca wcześniej", a nie „ten jest na pozycji 47, a tamten na 44". **Pozycja względna wypada z matematyki obrotu, zamiast być uczona.** Stąd `head_dim` dzieli się na pary i każda para obraca się z inną częstotliwością — szybkie łapią bliskich sąsiadów, wolne dalekie zależności.

---

## 6. RMSNorm — trzymanie liczb w ryzach

Bez normalizacji liczby w głębokiej sieci rozjeżdżają się wykładniczo i trening wybucha.

LayerNorm: odejmij średnią, podziel przez odchylenie. RMSNorm pomija odejmowanie średniej — okazało się, że nic nie wnosi. Zostaje: **podziel przez własną „długość", potem przeskaluj wyuczonym parametrem**.

Ważny szczegół — **pre-normalizacja**: normalizujemy *przed* podwarstwą (`self.attn(self.norm_attn(x))`), nie po. Dzięki temu residual stream `x` biegnie przez cały model nigdy nietknięty normalizacją — autostrada gradientu z punktu 3 zostaje nienaruszona. Oryginalny Transformer robił post-norm i wymagał starannego rozgrzewania learning rate, żeby nie eksplodować.

---

## 7. SwiGLU — gdzie model trzyma wiedzę

Attention **przenosi** informację między tokenami. Warstwa FFN ją **przetwarza** — osobno dla każdego tokenu.

Klasyczny MLP: `W2(gelu(W1 x))`. SwiGLU rozdziela pierwszą projekcję na dwie:

```python
gate = F.silu(self.w_gate(x))      # ile z tego przepuścić
hidden = gate * self.w_up(x)        # kandydaci na wartości
return self.w_down(hidden)
```

Jedna gałąź produkuje wartości, druga **bramkę**, która decyduje, ile z każdej wartości przeżyje. Mnożenie jest sednem: sieć może **warunkowo wyciszać** cechy, zamiast zawsze stosować tę samą nieliniowość.

Tu siedzi większość parametrów i większość wiedzy faktograficznej modelu. `capture.ffn_gate` (kształt `T × 2048`) to najbliższy odpowiednik „neuronów, które się odpaliły" — i to jest ten widok w panelu, który wygląda najbardziej jak żywy mózg.

---

## 8. Wyjście i weight tying

```python
self.lm_head.weight = self.tok_emb.weight
```

To nie kopia. To **ten sam tensor**. Tablica zamieniająca token → wektor jest jednocześnie tablicą zamieniającą wektor → prawdopodobieństwa tokenów.

Uzasadnienie: „który to token" i „jaki token będzie następny" chcą tej samej geometrii słownika. Oszczędza ~25M parametrów (jedna piąta modelu!) i przy tej skali **poprawia** wyniki, bo każdy wiersz embeddingu dostaje gradient z dwóch stron.

Drobna optymalizacja w `forward`: przy generowaniu liczymy `lm_head` tylko dla **ostatniej** pozycji. Interesuje nas następny token, więc projektowanie wszystkich T pozycji na 32 000 wymiarów byłoby czystym marnotrawstwem.

---

## 9. Uczenie w jednym akapicie

Podajemy tekst, model przewiduje każdy następny token, mierzymy `cross_entropy` między przewidywaniem a prawdą. Loss to „jak bardzo byłeś zaskoczony". Backprop liczy, jak każdy z 109M parametrów przyczynił się do zaskoczenia, i przesuwa każdy odrobinę w stronę mniejszego. Powtórz kilkaset tysięcy razy.

**Test poprawności wart zapamiętania:** na starcie loss musi wyjść ≈ `ln(vocab_size)` = 10,37. Model nie wie nic, więc zgaduje równomiernie z 32 000 opcji. Wynik **niższy oznacza wyciek**, nie sukces — u nas 8,94 zdemaskowało niezsunięte targety. Ten test kosztuje sekundę i ratuje przed trenowaniem zepsutego pipeline'u przez wiele godzin.

---

## 10. Gdzie mieszka te 109M

| część | parametry | udział |
|---|---|---|
| embedding tokenów (32000 × 768, współdzielony z wyjściem) | 24,6M | 22% |
| 12 × attention (QKV + projekcja) | 28,3M | 26% |
| 12 × SwiGLU | 56,6M | 52% |
| RMSNormy | ~0,02M | ~0% |
| **razem** | **109,5M** | |

Ponad połowa modelu to warstwy feed-forward. Attention przyciąga całą uwagę w wyjaśnieniach, ale **większość pojemności to FFN** — tam jest przechowywana wiedza.

---

## Co dalej

- **KV-cache** — teraz `generate()` przelicza cały kontekst przy każdym tokenie. Zamiast tego zapamiętamy K i V wcześniejszych tokenów, bo one się nie zmieniają. Największy pojedynczy zysk prędkości.
- **Kwantyzacja** — fp32 → int8. Wąskim gardłem na CPU jest przepustowość pamięci, nie liczenie, więc mniejsze wagi = szybciej.
- **Trening** — tokenizer → pretraining na polskim → instruction-tuning.
- **RAG + UI** — model odpowiada za język, vault za fakty.
