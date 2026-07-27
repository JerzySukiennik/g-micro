# G-Micro — specyfikacja interfejsu

Ustalone w wywiadzie `/pytania` 2026-07-20 (41 pytań, skille `frontend-design` + `apple-design`).
Status: **zatwierdzone, czeka na komendę „Buduj UI"**. Nic z tego nie jest domysłem — każda pozycja to decyzja Jurka albo jego wyraźna delegacja.

---

## Tożsamość

**G-Micro.** Nazwa małym monospace'em w rogu, obok liczby parametrów i tok/s. Apka nie krzyczy nazwą — pokazuje stan.

Język interfejsu: **angielski** (spójne z konwencją kod/commity EN, terminy ML brzmią naturalnie, krótsze etykiety lepiej leżą w minimalistycznym układzie).

---

## Powłoka i architektura

- **Electron**, macOS, okno **bez ramki** — trzy kropki systemowe na półprzezroczystym pasku, całe okno należy do treści.
- Klik w ikonę → **Electron sam startuje proces Pythona** → łączy się z nim po **WebSockecie** na localhost.
- Zamknięcie okna → proces Pythona dostaje sygnał i ginie. **Plus zabezpieczenie przed sierotą** na wypadek twardego ubicia Electrona (PID file / watchdog). Wymóg Jurka: zero procesów w tle po zamknięciu, zero terminala.
- ONNX odrzucony świadomie: konwersja zabiłaby hooki `Capture`, czyli sedno projektu.

**Podział danych:** inference w procesie Pythona na CPU, wizualizacja w rendererze Electrona na GPU. Różne procesy, różne układy — nie rywalizują o zasoby (istotne na Intel i9 bez Apple Silicon).

---

## Układ

```
┌─────────────────────────────────────────────────────┐
│ ● ● ●                                    G-Micro  109.5M │
├──────────────────────┬──────────────────────────────┤
│                      │  [ Neurons ] [ Probabilities ]│
│   rozmowa  (40%)     │                              │
│                      │      panel sieci (60%)       │
│                      │                              │
│  ┌────────────────┐  │                              │
│  │ input (glass)  │  │                              │
└──┴────────────────┴──┴──────────────────────────────┘
```

- Historia rozmów: **wysuwana z lewej**, domyślnie schowana, przykrywa czat półprzezroczystą warstwą szkła. Nie zjada stałego pasa (ważne przy 40/60).
- Zakładki panelu przełączane **Tab**.

---

## Estetyka

**Kierunek: Apple jako przyrząd precyzyjny.** Nie Gemini, nie aurora gradienty.

- **Wyłącznie ciemny motyw.** Świecące neurony na czerni to sedno — na białym tle światło nie świeci, tylko brudzi. Jeden motyw = każdy piksel dopracowany.
- **Zero barwy w interfejsie.** Skala szarości i biel. Jedyny wyjątek: **stłumiona czerwień na błędy**. Uzasadnienie: skoro biel niesie znaczenie „aktywacja", każda inna barwa musi znaczyć coś równie ważnego. Kolor jako alarm, nigdy jako ozdoba.
- **Typografia:** Inter (lub systemowy) do interfejsu i rozmowy, **mono do wszystkich liczb** — prawdopodobieństwa, tok/s, numery warstw — żeby nie skakały przy zmianie. Tracking zależny od rozmiaru (ujemny na dużym, ~0 na tekście), leading odwrotnie do rozmiaru.
- **Materiały:** szkło z `backdrop-filter` na warstwach unoszących się nad treścią (input, sidebar, pasek okna). Treść przewija się pod spodem, nie pod nieprzezroczystym pasem. Nigdy nie stackować jasnego szkła na jasnym.

---

## Zakładka 1 — Neurons (główna)

- **Wszystkie 24 576 neuronów** (12 warstw × 2048), siatka ~157×157.
- **Ułożone kolejno warstwa po warstwie, bez ramek i podpisów.** Wygląda jak jedna gęsta plansza, ale po chwili widać, że dół żyje innym rytmem niż góra. Efekt „mózgu" zachowany, struktura nie zgubiona.
- **Punkt z miękką poświatą.** Nieaktywny — ledwo widoczny ciemny punkt; aktywny — świeci z halo. Jasność = siła aktywacji. **Czysta biel, zero barwy.**
- **Aktualizacja live 1:1** z generowaniem (15-28 Hz). Bez wygładzania między tokenami — decyzja Jurka, wierność ponad płynność. To i tak jest blisko klatkażu, więc nie strobuje.
- **Hover:** etykieta `layer 7 · neuron 1432`.
- **Klik:** przypina neuron, rysuje jego aktywację w czasie na mini-wykresie. Zamienia obraz w narzędzie — można znaleźć neuron reagujący na konkretne słowo.
- **Stan spoczynku:** ostatni **prawdziwy** stan aktywacji delikatnie faluje. Ważne: nie wymyślony szum — realne dane, tylko modulowane. Ruch bez kłamstwa.

Render: canvas 2D lub WebGL (24k punktów to nic dla Radeona).

---

## Zakładka 2 — Probabilities

- **Top-10 kandydatów** na następny token, każdy ze słupkiem i wartością procentową monospace'em.
- Przy zmianie kolejności słupki **przejeżdżają sprężyście** na nowe miejsca, nie przeskakują. Widać walkę kandydatów o pierwsze miejsce.
- Pod spodem **pewność modelu**: entropia rozkładu przetłumaczona na skalę `confident → wavering → guessing`, jako liczba + pasek. Najbardziej pouczający element — widać, że model jest pewny przy końcówkach fleksyjnych i gubi się przy faktach.

---

## Rozmowa

- **Bez dymków.** Wiadomości Jurka przygaszone i wcięte, odpowiedzi modelu pełną bielą i normalną szerokością. Czysty strumień jak transkrypcja.
- **Tekst token po tokenie** — czasem po pół słowa, bo tak działa tokenizer. Szczere wobec natury modelu i spójne z resztą apki, która pokazuje prawdę.
- Każdy nowy token **rozbłyska jaśniej i gaśnie do bieli** — wiąże tekst z pulsem sieci obok.
- **Input:** rośnie z tekstem (jedna linia → więcej), Enter wysyła, Shift+Enter łamie linię. Na szkle, przyklejony do dołu, tekst przewija się pod spodem.

---

## Signature — „Wake"

**Przy starcie 24 tysiące neuronów zapalają się warstwa po warstwie**, w rytmie realnego ładowania checkpointu (1,2 GB, kilka sekund).

To nie jest atrapa ładowania — to jest postęp wczytywania wag. Zamienia czekanie w najlepszy moment apki i od razu tłumaczy, na co patrzysz. Widoczne przy każdym uruchomieniu.

Po nim: **czysty ekran i kursor.** Zero powitań, zero podpowiedzi startowych, zero informacji o modelu. Pusty ekran jest zaproszeniem, nie brakiem.

---

## Zachowanie

| Rzecz | Decyzja |
|---|---|
| RAG | **Niewidoczny** (przełącznik w ustawieniach — patrz Ryzyka) |
| Temperatura / top-k | Schowane pod jednym przyciskiem |
| Checkpointy | Zawsze najnowszy, bez przełączania |
| Historia | Zapisywana lokalnie, panel wysuwany |
| Błąd backendu | Tekst zostaje, pod nim jedna linia stłumioną czerwienią + `Retry`. Apka próbuje wstać sama **raz** w tle, zanim powie |
| Skróty | Cmd+N nowa, Cmd+K historia, Cmd+, ustawienia, Tab zakładki, Esc przerywa. Bez podpowiedzi na ekranie |

---

## Ruch

**Bogata warstwa mikroanimacji** (decyzja Jurka wbrew mojej rekomendacji oszczędności — ale konfliktu o CPU nie ma, patrz Architektura).

Wszystko na **sprężynach**, nigdy na `transition` o stałym czasie:
- domyślnie `damping 1.0`, `response 0.3–0.4` (krytycznie tłumione, bez odbicia)
- odbicie (`damping ~0.8`) **tylko** tam, gdzie gest niósł pęd
- animacja zawsze **od wartości bieżącej**, nie docelowej (przerwanie bez skoku)
- reakcja na `pointerdown`, nie na `click`
- `prefers-reduced-motion` → cross-fade zamiast sprężyn, bez overshootu
- `prefers-reduced-transparency` → szkło robi się matowe

---

## Dane z modelu

**Pełne 12 warstw, ale zredukowane:** attention uśredniony po głowach, **top-64 neurony FFN zamiast 2048** w transmisji (siatka pokazuje wszystkie, ale przez rzadką aktualizację). Cel: ~50 KB na token, płynne przy 15-28 tok/s.

Pełna rozdzielczość (~5 MB/token) odrzucona — zadławiłaby generowanie.

---

## Świadome ryzyka (nie przeoczenia)

1. **Niewidoczny RAG.** Przy modelu 109M, który **będzie mylił fakty**, brak widocznych źródeł oznacza, że nie odróżnisz „przeczytał z notatki" od „zmyślił". Dlatego: przełącznik w ustawieniach, domyślnie wyłączony.
2. **Oddech w spoczynku.** Pokazywanie ruchu tam, gdzie nie ma obliczeń, łamie zasadę „ruch tylko tam, gdzie coś się dzieje". Złagodzone: falowanie robione z ostatniego prawdziwego stanu aktywacji, nie z wymyślonego szumu.
3. **DesignPass.dev** — Jurek podał jako źródło komponentów, ale to **płatny marketplace**; nie kopiujemy ich kodu. Budujemy własne komponenty w tym samym języku (sprężyny, liquid glass, magnetyczny hover), co i tak pokrywa skill `apple-design`.

---

## Zakres pierwszej wersji

**Wszystko naraz, jeden spójny build.** Electron + backend + obie zakładki + historia + ustawienia + Wake. Rozbicie na etapy dałoby pozszywaną zbieraninę — design doklejany na końcu zawsze widać.

Budowane na **niewytrenowanych wagach** — wizualizacja jest w pełni prawdziwa od pierwszej minuty (`Capture` zbiera realne dane niezależnie od tego, czy wagi są mądre). Model gada bzdury do czasu końca treningu; podmiana checkpointu to jedna ścieżka w configu.
