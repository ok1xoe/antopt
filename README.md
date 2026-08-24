# AntOpt — modelování a optimalizace antén

Desktopová aplikace pro modelování drátových antén metodou momentů a pro
automatickou optimalizaci geometrie. Obdoba MMANA-GAL, ale s vlastním
solverem napsaným od nuly a s otevřeným zdrojovým kódem.

---

## Instalace na macOS

Potřebuješ Python 3.10+ **s Tkinter**.

> **Pozor:** `pip install tkinter` nikdy nebude fungovat. Tkinter není balíček
> z PyPI — je součástí standardní knihovny, ale potřebuje C modul `_tkinter`
> přeložený proti Tcl/Tk. Homebrew ho dodává jako samostatnou formuli,
> instalátor z python.org ho má rovnou v sobě.

**Python z Homebrew** — doplň Tk k té verzi, kterou používáš:

```bash
brew install python-tk@3.14      # číslo musí sedět na tvůj python3 -V
```

Existující virtuální prostředí není potřeba vytvářet znovu — bere standardní
knihovnu ze základního Pythonu.

**Nebo Python z [python.org](https://www.python.org/downloads/macos/)**, který
má Tcl/Tk zabudovaný a tenhle problém vůbec nemá.

Pak už jen:

```bash
cd antopt
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt      # numpy, scipy, matplotlib
python3 run_antopt.py
```

Ověření, že Tkinter je k dispozici:

```bash
python3 -c "import tkinter; print(tkinter.TkVersion)"
```

`run_antopt.py` si prostředí zkontroluje sám a když něco chybí, vypíše
přesný příkaz.

### Linux

```bash
sudo apt install python3-tk        # Debian/Ubuntu
sudo dnf install python3-tkinter   # Fedora
```

---

## Co program umí

**Model**

* libovolně orientované tenké dráty ve 3D, spoje více drátů v jednom uzlu
* volný prostor / dokonalá zem / reálná zem (ε_r, σ, přednastavené typy)
* ztráty vodiče podle materiálu (měď, hliník, mosaz, ocel, …)
* soustředné zátěže R+jX i sériové RLC
* více zdrojů s fází — fázovaná pole
* uzemněné vertikály (drát dosahující na z = 0) + ruční ztráty zemního systému

**Výpočet**

* vstupní impedance, PSV, činitel odrazu
* vyzařovací diagramy ve stylu MMANA — vodorovný plný kruh, svislý půlkruh
  přes celou rovinu (dopředu i dozadu) s obzorem, prstence po 10 dB,
  volitelná polarizace (celkem / svislá / vodorovná / všechny),
  značky −3 dB, směr maxima, měřicí kurzor po kliknutí
* zisk v dBi/dBd, F/B, poměr před/stranou
* elevace a azimut maxima, šířky svazků v −3 dB, účinnost z integrálu diagramu
* kmitočtové rozmítání se šířkou pásma pod PSV 2
* rozložení proudu barevně na 3D náhledu geometrie

**Optimalizace**

* volitelné parametry: délka prvku, rozteč, výška, poloměr, jednotlivá souřadnice
* cílová funkce váží zisk, F/B, PSV a impedanci — i na několika kmitočtech naráz
* genetický algoritmus + doladění Nelder-Meadem, běží na pozadí, lze zastavit

**Přizpůsobení**

* návrh vlásenky (hairpin / beta match) — potřebná reaktance, Z₀ dvoulinky,
  délka pahýlu, průběh PSV po přizpůsobení
* automatické doladění délky zářiče na podmínku `R² + X² = Z₀·R`

**Soubory**

* vlastní projekt `.json`
* import a export NEC (`.nec`) a MMANA (`.maa`, `.mma`)

---

## Jak to počítá

Elektricko-polní integrální rovnice (EFIE) ve smíšeném potenciálovém tvaru,
Galerkinovo testování po částech lineárními (trojúhelníkovými) bázemi na
segmentech, redukované jádro:

```
Z_mn = jωμ/4π ∫∫ f_m·f_n G dl'dl  +  1/(j4πωε) ∫∫ f'_m f'_n G dl'dl
G(R) = e^{-jkR}/R,   R = √(|r−r'|² + a²)
```

Singularita se odděluje jako `G = (e^{-jkR}−1)/R + 1/R`. První člen je hladký
a integruje se Gaussovou kvadraturou; druhý má vnitřní integrál v uzavřeném
tvaru (přes `arcsinh`), takže vlastní členy matice jsou přesné a nezávisí na
jemnosti kvadratury.

Uzly: volný konec → nulový proud; spoj dvou segmentů → jedna báze; uzel se
stupněm M → M−1 bází (Kirchhoff); uzel na zemi → M bází, proud smí odtéct
do země. Zem se řeší obrazovou teorií: obraz báze `f` je `−M(f)`, kde `M`
zrcadlí z → −z. Pro dokonalou zem je to přesné.

---

## Ověření proti NEC-2

Testy v `tests/test_validation.py` porovnávají výsledky s NEC-2 (přes PyNEC)
a s analytickými hodnotami. Naměřeno:

| Případ | AntOpt | NEC-2 | rozdíl R | rozdíl X |
|---|---|---|---|---|
| Dipól λ/2, a/λ=1e-03 | 86.19 +46.23j | 86.15 +49.00j | 0.05 % | 2.77 Ω |
| Dipól λ/2, a/λ=5e-04 | 83.57 +45.86j | 83.46 +47.63j | 0.13 % | 1.77 Ω |
| Dipól λ/2, a/λ=1e-04 | 80.27 +45.00j | 80.11 +45.65j | 0.20 % | 0.65 Ω |
| Dipól λ/2, a/λ=1e-05 | 78.05 +44.20j | 77.93 +44.51j | 0.15 % | 0.31 Ω |
| Yagi 3 prvky, 14,1 MHz | 26.18 −18.54j | 26.34 −17.83j | 0.62 % | 0.71 Ω |
| Yagi 3 prvky — zisk | 7.17 dBi | 7.18 dBi | 0.01 dB | — |
| Vertikál λ/4 nad dok. zemí | 40.74 +22.52j | 40.52 +23.31j | 0.55 % | 0.79 Ω |

Další ověřené hodnoty: zisk půlvlnného dipólu 2,15 dBi; monopol nad dokonalou
zemí má přesně poloviční impedanci než odpovídající dipól a zisk 5,15 dBi;
integrál vyzařovacího diagramu bezeztrátové antény dá účinnost 1,000;
impedanční matice je symetrická na 10⁻¹⁰ (reciprocita).

Spuštění testů:

```bash
pip install pytest PyNEC     # PyNEC je volitelný, bez něj se NEC testy přeskočí
python3 -m pytest tests/ -q
```

---

## Kde být opatrný

Tohle jsou skutečná omezení modelu, ne formality:

* **Reálná zem v impedanci je jen přibližná.** Matice se počítá s obrazem nad
  dokonalou zemí (stejně jako MININEC, ze kterého vychází i MMANA); Fresnelovy
  koeficienty se uplatní až ve vyzařování. U vodorovných antén nad zemí to
  na zisk sedí do ~0,2 dB, na impedanci je odchylka jednotky ohmů. **U vertikálu
  napájeného proti zemi je impedance nepoužitelná** — zadej ztráty zemního
  systému ručně (pole „Ztráty zemn. systému“) nebo model ověř jinde.
* **Zakopané radiály neumí** nikdo z této třídy programů, tenhle taky ne.
* **Velmi tlusté dráty** (poloměr nad ~λ/200) — reaktance se začne rozcházet,
  odpor zůstává dobrý. Pro trubky 25 mm na 20 m je odchylka pod 1 Ω.
* **Segment musí být delší než 4× poloměr drátu.** Program to hlídá a upozorní.
* Není tu vodivá plocha, válec, dielektrikum ani NEC-4 „ground screen“.

---

## Struktura

```
antopt/
  model.py      datové struktury (drát, zdroj, zátěž, zem, model)
  mesh.py       segmentace a stavba bázových funkcí
  solver.py     jádro MoM — impedanční matice, buzení, proudy
  farfield.py   dálné pole, zisk, F/B, Fresnelovy koeficienty
  analysis.py   rozmítání kmitočtu, šířka pásma
  optimize.py   parametry, cílová funkce, GA + Nelder-Mead
  match.py      návrh vlásenky (hairpin), ladění zářiče
  fileio.py     import/export NEC a MMANA
  examples.py   vestavěné příklady
  gui.py        Tkinter aplikace
run_antopt.py   spouštěč
tests/          ověřovací sada
```

Solver se dá používat i bez GUI:

```python
from antopt.examples import yagi3_20m
from antopt.analysis import analyse

r = analyse(yagi3_20m())
print(r.zin, r.swr, r.gain_dbi, r.fb_db)
```

---

## Vestavěné příklady

Yagi 6 prvků 10 m (ráhno 7,5 m, AL 16×1,5, s vlásenkou) · Dipól 20 m ·
Inverted V 40 m · Yagi 3 prvky 20 m · Yagi 5 prvků 2 m · Vertikál 40 m ·
Delta loop 20 m

Kompletní návrh 6prvkové Yagi na 10 m včetně rozpisu rozměrů je ve složce
`navrhy/yagi6-10m/`.
