# Yagi 6 prvků, 10 m — rozpis a rozměry

Ráhno 7,50 m (0,71 λ) · prvky AL trubka 16×1,5 mm · optimalizováno 28,0–28,8 MHz
Modelováno v AntOpt (metoda momentů), ověřeno proti NEC-2.

## Rozměry

Vzdálenost se měří od reflektoru podél ráhna, délka je **celková** délka prvku.

| Prvek | Poloha [mm] | Délka [mm] | Poloviny [mm] | Délka při průchodu ráhnem* [mm] |
|---|---:|---:|---:|---:|
| Reflektor | 0 | 5288 | 2 × 2644 | 5303 |
| Zářič | 1050 | 4969 | 2 × 2485 | 4984 |
| D1 | 2079 | 4916 | 2 × 2458 | 4931 |
| D2 | 3946 | 4660 | 2 × 2330 | 4675 |
| D3 | 5156 | 4782 | 2 × 2391 | 4797 |
| D4 | 7500 | 4667 | 2 × 2333 | 4682 |

\* Korekce +15 mm platí, jsou-li prvky **protažené ráhnem a vodivě s ním spojené**.
Při izolovaném uchycení na destičce nad ráhnem platí délky ve sloupci „Délka“.

Spotřeba trubky 16×1,5: **29,3 m** (+ přesahy na středové spoje).

## Napájení — vlásenka (hairpin)

Zářič je záměrně zkrácený, takže je kapacitní (28,2 − j24,8 Ω na 28,4 MHz).

- vodiče vlásenky Ø 10 mm, rozteč 60 mm → Z₀ vedení 297 Ω
- délka vlásenky **317 mm**, na konci zkratovaná
- zářič musí být **dělený a izolovaný od ráhna**
- za vlásenku patří balun 1:1

Vlásenku vyveď pár cm od zářiče a při ladění zkracuj/prodlužuj zkratovací
příčku — praktický rozsah je zhruba 260–380 mm.

## Vypočtené parametry

| f [MHz] | Zisk [dBi] | F/B [dB] | PSV (50 Ω) |
|---|---:|---:|---:|
| 28,0 | 10,45 | 22,9 | 1,32 |
| 28,2 | 10,54 | 31,2 | 1,10 |
| 28,4 | 10,60 | 27,0 | 1,00 |
| 28,6 | 10,61 | 22,9 | 1,08 |
| 28,8 | 10,55 | 22,6 | 1,62 |

(volný prostor; zisk v dBd = uvedená hodnota − 2,15)

Nad průměrnou zemí ve výšce 10 m: **15,5 dBi** v elevaci **14°**,
šířka svazku 24° vodorovně / 14° svisle.
