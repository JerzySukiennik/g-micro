# Dlaczego reguły wyglądają tak, jak wyglądają

Realtime Database nie przyjmuje komentarzy w `database.rules.json` (poza korzeniem
dozwolony jest tylko klucz `rules`), więc uzasadnienie mieszka tutaj.

## Cel: prywatność per urządzenie, bez logowania

Każda przeglądarka losuje sobie 26-znakowy identyfikator przy pierwszym wejściu
(`web/app.js`, `resolveClient`) i trzyma go we własnym `localStorage`. To jest cała jej
tożsamość — zastępuje konto. Wszystko, co wysyła, żyje pod `open/<client>/`.

Izolacja stoi na dwóch nogach naraz:

1. **Nie da się wylistować drzewa.** `.read` na `open` przysługuje wyłącznie uid Maca,
   więc żadna przeglądarka nie dowie się, jakie identyfikatory istnieją.
2. **Nie da się ich zgadnąć.** Identyfikator to 128 bitów z generatora kryptograficznego,
   który nigdy nie opuszcza przeglądarki.

Dlatego `open/$client/out` ma `.read: true` — to nie jest dziura. Odczyt wymaga znajomości
ścieżki, a ścieżka *jest* sekretem. Zapis do `out` ma już wyłącznie Mac, więc nikt nie
podrzuci komuś fałszywej odpowiedzi.

## Po co Macowi konto

Asymetria musi skądś pochodzić. Most ma widzieć zadania **wszystkich** urządzeń (jeden
strumień SSE obsługuje wszystkich), a przeglądarka ma widzieć **tylko swoje**. Gdyby most
nie był uwierzytelniony, byłby zwykłym anonimowym klientem — a wtedy cokolwiek pozwalałoby
mu znaleźć zadania, pozwalałoby na to każdemu.

Dlatego most loguje się raz anonimowo (`runtime/bridge.py`, `MacIdentity`) i reguły znają
go po uid. Konto żyje dzięki refresh tokenowi w `~/.g-micro/identity.json` (chmod 600),
więc uid jest stały.

**Uwaga operacyjna:** utrata `identity.json` = nowy uid = reguły trzeba wdrożyć ponownie
z nową wartością. Bieżący uid wypisuje `python runtime/bridge.py --print-uid`.

## Obecność osobno

`status/mac` leży poza drzewem klientów i jest czytelne dla wszystkich, bo urządzenie musi
wiedzieć, czy Mac nie śpi, **zanim** cokolwiek u siebie zapisze. Zapis do niego ma tylko
Mac, więc nikt nie udaje, że jest online.

## Limity

Walidacja tnie rozmiary przy wejściu: tekst ≤ 2000 znaków, zdjęcie ≤ 400 000, nieznane
pola odrzucone. To ochrona przed zapchaniem bazy, nie przed odczytem.
