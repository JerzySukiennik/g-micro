"""
Hand-written minimal pairs for the G-Micro inflection probe.

Each pair is identical except for one grammatical choice: a case ending,
a verb-person agreement, or an adjective-noun gender match. `correct` is
the grammatical Polish sentence; `incorrect` swaps in the wrong form
(usually the nominative, or the wrong person/gender) in the same slot.

This is the same technique BLiMP uses for English acceptability judgments:
score both sentences under the model via teacher-forced log-likelihood and
check that the grammatical one scores higher. It answers a narrower, more
checkable question than perplexity alone — not "how surprised was the
model," but "does it specifically prefer correct case government."
"""

PAIRS = [
    # --- dopełniacz (genitive) after prepositions: do, z, od, bez, dla, u, obok, koło ---
    dict(category="dopełniacz", correct="Idę do sklepu po chleb.", incorrect="Idę do sklep po chleb."),
    dict(category="dopełniacz", correct="Wracam ze szkoły o trzeciej.", incorrect="Wracam ze szkoła o trzeciej."),
    dict(category="dopełniacz", correct="To jest prezent dla mamy.", incorrect="To jest prezent dla mama."),
    dict(category="dopełniacz", correct="Siedzę obok okna.", incorrect="Siedzę obok okno."),
    dict(category="dopełniacz", correct="Mieszkam u babci przez lato.", incorrect="Mieszkam u babcia przez lato."),
    dict(category="dopełniacz", correct="Zostałem bez pieniędzy.", incorrect="Zostałem bez pieniądze."),
    dict(category="dopełniacz", correct="Wyszedł z domu bardzo wcześnie.", incorrect="Wyszedł z dom bardzo wcześnie."),
    dict(category="dopełniacz", correct="To jest córka naszego sąsiada.", incorrect="To jest córka nasz sąsiad."),

    # --- celownik (dative) after verbs / dzięki, ku ---
    dict(category="celownik", correct="Dziękuję nauczycielowi za pomoc.", incorrect="Dziękuję nauczyciel za pomoc."),
    dict(category="celownik", correct="Pomagam mamie w kuchni.", incorrect="Pomagam mama w kuchni."),
    dict(category="celownik", correct="Dzięki tobie zdałem egzamin.", incorrect="Dzięki ty zdałem egzamin."),
    dict(category="celownik", correct="Daję psu jedzenie każdego ranka.", incorrect="Daję pies jedzenie każdego ranka."),
    dict(category="celownik", correct="Powiedziałem koledze o wszystkim.", incorrect="Powiedziałem kolega o wszystkim."),
    dict(category="celownik", correct="Ufam swojemu bratu.", incorrect="Ufam swój brat."),

    # --- biernik (accusative) after transitive verbs ---
    dict(category="biernik", correct="Widzę psa na podwórku.", incorrect="Widzę pies na podwórku."),
    dict(category="biernik", correct="Lubię tę książkę.", incorrect="Lubię ta książka."),
    dict(category="biernik", correct="Czytam ciekawą powieść.", incorrect="Czytam ciekawa powieść."),
    dict(category="biernik", correct="Widzę mojego kolegę na przystanku.", incorrect="Widzę mój kolega na przystanku."),
    dict(category="biernik", correct="Znam tę dziewczynę od dziecka.", incorrect="Znam ta dziewczyna od dziecka."),
    dict(category="biernik", correct="Kupiłem nową sukienkę na wesele.", incorrect="Kupiłem nowa sukienka na wesele."),

    # --- narzędnik (instrumental) after z, jestem/zostanę ---
    dict(category="narzędnik", correct="Jadę autobusem do pracy.", incorrect="Jadę autobus do pracy."),
    dict(category="narzędnik", correct="Piszę długopisem w zeszycie.", incorrect="Piszę długopis w zeszycie."),
    dict(category="narzędnik", correct="Chcę zostać lekarzem.", incorrect="Chcę zostać lekarz."),
    dict(category="narzędnik", correct="Rozmawiam z przyjacielem przez telefon.", incorrect="Rozmawiam z przyjaciel przez telefon."),
    dict(category="narzędnik", correct="Interesuję się muzyką klasyczną.", incorrect="Interesuję się muzyka klasyczna."),
    dict(category="narzędnik", correct="Jest bardzo dumna swoim synem.", incorrect="Jest bardzo dumna swój syn."),

    # --- miejscownik (locative) after w, na, o ---
    dict(category="miejscownik", correct="Mieszkam w Warszawie od roku.", incorrect="Mieszkam w Warszawa od roku."),
    dict(category="miejscownik", correct="Myślę o wakacjach codziennie.", incorrect="Myślę o wakacje codziennie."),
    dict(category="miejscownik", correct="Siedzę na krześle przy stole.", incorrect="Siedzę na krzesło przy stole."),
    dict(category="miejscownik", correct="Pracuję w dużej szkole.", incorrect="Pracuję w duża szkoła."),
    dict(category="miejscownik", correct="Rozmawialiśmy o nowym filmie.", incorrect="Rozmawialiśmy o nowy film."),

    # --- zgodność liczby: podmiot-czasownik ---
    dict(category="liczba_czasownika", correct="Dzieci bawią się w parku.", incorrect="Dzieci bawi się w parku."),
    dict(category="liczba_czasownika", correct="Chłopcy grają dzisiaj w piłkę.", incorrect="Chłopcy gra dzisiaj w piłkę."),
    dict(category="liczba_czasownika", correct="Kot śpi teraz na kanapie.", incorrect="Kot śpią teraz na kanapie."),
    dict(category="liczba_czasownika", correct="Moi rodzice pracują w mieście.", incorrect="Moi rodzice pracuje w mieście."),

    # --- odmiana czasownika przez osoby ---
    dict(category="osoba_czasownika", correct="Ja idę teraz do domu.", incorrect="Ja idzie teraz do domu."),
    dict(category="osoba_czasownika", correct="Ty jesteś moim najlepszym przyjacielem.", incorrect="Ty jest moim najlepszym przyjacielem."),
    dict(category="osoba_czasownika", correct="My mamy dzisiaj dużo pracy.", incorrect="My ma dzisiaj dużo pracy."),
    dict(category="osoba_czasownika", correct="Oni czytają razem te same książki.", incorrect="Oni czyta razem te same książki."),
    dict(category="osoba_czasownika", correct="Ona pisze list do babci.", incorrect="Ona piszę list do babci."),

    # --- zgodność rodzaju: przymiotnik-rzeczownik ---
    dict(category="rodzaj_przymiotnika", correct="To jest bardzo dobry pies.", incorrect="To jest bardzo dobra pies."),
    dict(category="rodzaj_przymiotnika", correct="Ona ma piękną nową suknię.", incorrect="Ona ma piękny nowy suknię."),
    dict(category="rodzaj_przymiotnika", correct="To jest bardzo duże miasto.", incorrect="To jest bardzo duży miasto."),
    dict(category="rodzaj_przymiotnika", correct="Mam nowy czerwony samochód.", incorrect="Mam nowa czerwona samochód."),
    dict(category="rodzaj_przymiotnika", correct="Kupiłem sobie małego czarnego kota.", incorrect="Kupiłem sobie mała czarna kota."),
    dict(category="rodzaj_przymiotnika", correct="To była naprawdę ciekawa historia.", incorrect="To był naprawdę ciekawy historia."),

    # --- liczba mnoga rzeczownika po liczebnikach ---
    dict(category="liczba_mnoga", correct="Widzę na podwórku dwa psy.", incorrect="Widzę na podwórku dwa pies."),
    dict(category="liczba_mnoga", correct="Mam trzech starszych braci.", incorrect="Mam trzech starszy brat."),
    dict(category="liczba_mnoga", correct="W koszyku leżały cztery jabłka.", incorrect="W koszyku leżały cztery jabłko."),
]

assert 40 <= len(PAIRS) <= 60, f"pair count drifted out of the intended 40-60 range: {len(PAIRS)}"
