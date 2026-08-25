# AntOpt — modelování a optimalizace antén

Desktopová aplikace pro modelování drátových antén metodou momentů a pro
automatickou optimalizaci geometrie. Obdoba MMANA-GAL, s otevřeným zdrojovým
kódem a **dvěma zaměnitelnými výpočetními jádry**: vlastním solverem napsaným
od nuly a původním **NEC-2**.

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

## Spustitelná aplikace

Aby se dal program spouštět poklepáním a bez instalovaného Pythonu, sbalí se
PyInstallerem do jednoho balíku. **Sestavuje se vždy na tom systému, pro který
má výsledek být** — PyInstaller neumí křížový překlad a na Macu vznikne
aplikace jen pro ten procesor, na kterém se sestavila (Apple Silicon, nebo
Intel).

**macOS** — ve Finderu poklepat na `build/build_macos.command`.
(Kdyby se místo spuštění otevřel v editoru: `chmod +x build/build_macos.command`.)

**Windows** — poklepat na `build\build_windows.bat`.

**Linux** — `./build/build_linux.sh`

Skript si sám vytvoří oddělené prostředí, doinstaluje PyInstaller a Pillow,
vykreslí ikonu, sestaví aplikaci a nakonec ji **spustí na kontrolu**. Trvá to
minutu až dvě, výsledek je v `dist/`:

| systém | výsledek | velikost |
|---|---|---|
| macOS | `dist/AntOpt.app` | ~200 MB |
| Windows | `dist/AntOpt/AntOpt.exe` | ~200 MB |
| Linux | `dist/AntOpt/AntOpt` | ~200 MB |

Velikost je daná numpy, scipy a matplotlibem uvnitř. Na Windows a Linuxu se
musí přenášet **celá složka** `dist/AntOpt`, ne jen spustitelný soubor;
`AntOpt.app` na macOS je balík a přenáší se celý sám.

### Kontrola sestavené aplikace

```bash
dist/AntOpt.app/Contents/MacOS/AntOpt --selftest     # macOS
dist\AntOpt\AntOpt.exe --selftest                    # Windows
```

Ověří import scipy.optimize, matplotlibu s Tk a PIL.ImageTk, spočítá zkušební
model a proběhne krátkou optimalizaci včetně doladění. Tyhle věci se importují
až za běhu, takže bez téhle kontroly by chybějící kus vyplaval třeba až
uprostřed dlouhé optimalizace.

### Předání někomu dalšímu

Aplikace není podepsaná vývojářským certifikátem, takže ji macOS na cizím
počítači napoprvé nepustí. Řešení je klepnout na ni pravým tlačítkem a dát
**Otevřít**, případně:

```bash
xattr -dr com.apple.quarantine /cesta/AntOpt.app
```

---

## Výpočetní jádra

Jádro se přepíná v záložce Výpočet a platí pro analýzu, rozmítání i optimalizaci.

| | vlastní | NEC-2 |
|---|---|---|
| závislosti | žádné | `pip install PyNEC` |
| rychlost analýzy | ~0,2 s | ~0,3 s |
| volný prostor | ✅ shoda do 0,2 % | ✅ |
| dokonalá zem | ✅ shoda do 1 % | ✅ |
| **reálná zem — impedance** | aproximace (obraz nad dokonalou zemí) | **Sommerfeldova zem** |
| dráty blízko země | odchylka roste | spolehlivé |
| licence | součást projektu | GPL (načítá se volitelně) |

Pravidlo: **volný prostor a dokonalá zem — obě jádra dají totéž**, ber to
rychlejší. **Reálná zem a cokoliv blízko země — ber NEC-2.**

Kde ani jedno jádro nestačí: **vertikál napájený přímo ze země nad reálnou
zemí.** Vlastní jádro ho počítá jako nad dokonalou zemí (ignoruje ztráty),
NEC-2 dá u drátu dotýkajícího se země s reálnou zemí nesmysl (200 Ω místo
36 Ω) — to je známé omezení NEC-2, na které je potřeba NEC-4. Buď zvedni
základnu a dej radiály, nebo počítej s dokonalou zemí a ztráty zemního
systému zadej ručně.

### Proč zrovna NEC-2 a ne 4nec2 nebo EZNEC

**4nec2 a EZNEC nejsou solvery, ale uzavřená okna nad NEC-2** (EZNEC Pro nad
NEC-4, který je licencovaný a exportně omezený) — knihovna, na kterou by
šlo linkovat, neexistuje. **MININEC 3** je veřejný, ale je to starší a méně
přesný předchůdce NEC-2, ze kterého vychází právě MMANA. Z těch možností je
NEC-2 jediné jádro, které má smysl a jde použít.

---

## Co program umí

**Tvorba modelu**

* průvodci: dipól, inverted V, vertikál, Yagi (2–8 prvků), quad, delta loop,
  fázované pole, long wire
* úpravy geometrie: posun, rotace, zrcadlení, změna měřítka
* **přeladění celé antény na jiný kmitočet** jedním příkazem
* **stoh / řada** až 64 kopií s volitelným fázováním
* **zúžené (teleskopické) prvky** — prvek z několika trubek různého průměru;
  pracuje se s ním **jako s jedním prvkem**: dá se kdykoli znovu otevřít
  a přestavět, délka i rozteč se mění vcelku a segmenty jsou v celém prvku
  stejně dlouhé
* zadání drátu polárně (délka, azimut, zenit)
* editor prvků — pracuje s celými prvky včetně zúžení, umí i vysunout
  samotnou koncovou trubku (tak se prvek ladí doopravdy)

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
* **hledání rezonance**, činitel jakosti Q
* F/B klasicky nebo **v zadním výseku** (60/90/120/180°) jako v MMANA
* **historie výpočtů** s exportem do CSV
* **3D vyzařovací diagram** s vykreslenou anténou
* rozložení proudu barevně na 3D náhledu geometrie

**Optimalizace**

* 19 druhů proměnných: délky a polohy drátů i celých prvků, poloměry, výška,
  azimut a zenit, hodnoty zátěží (R, X, L, C), napětí a fáze zdrojů, kmitočet
* **svázané proměnné** — `#2`, `0.95*#2`, `#2-0.15`; drží symetrii a poměry
* volitelný **krok** proměnné (kvantizace na reálné rozměry)
* cíle: zisk, F/B, F/S, PSV, R, X, **úhel vyzařování**, **proud v uzlu** —
  i na několika kmitočtech naráz
* genetický algoritmus + doladění Nelder-Meadem, běží na pozadí, lze zastavit

**Přizpůsobení a VF výpočty**

* návrh vlásenky (hairpin / beta match) + doladění délky zářiče na podmínku
  `R² + X² = Z₀·R`
* **LC článek** — obě topologie, obě znaménka, ověřená řešení
* **pahýl** (zkratovaný i otevřený) a **vložená transformační sekce**
* **napáječ** — transformace impedance, ztráty zvýšené odrazem, 14 typů
  kabelů s činitelem zkrácení
* reaktance L a C, rezonance LC, návrh vzduchové cívky

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

Se stejným `pip install PyNEC` se v aplikaci zpřístupní i **jádro NEC-2**.

---

## Shoda s MMANA-GAL

Co z MMANA je hotové a co ne — bez přikrášlování:

| Funkce MMANA | AntOpt |
|---|---|
| Tabulka drátů, zdrojů, zátěží | ✅ |
| Zdroje s fází, zátěže R+jX i RLC | ✅ |
| Zem: volný prostor / dokonalá / reálná | ✅ |
| Vyzařovací diagramy, polarizace H/V/celkem | ✅ |
| Měřicí vektor v diagramu | ✅ |
| 3D diagram s anténou | ✅ |
| Plots: Z, PSV, zisk/F-B přes kmitočet | ✅ |
| Resonance — hledání rezonančního kmitočtu | ✅ |
| F/B v zadním výseku | ✅ |
| Wire Scale — přeladění na jiný kmitočet | ✅ |
| Make Stack | ✅ |
| Taper Wire Set — zúžené prvky | ✅ |
| Move / rotace / zrcadlení / měřítko | ✅ |
| Wire definition — polární zadání | ✅ |
| Element editor | ✅ |
| Průvodci tvorbou antén | ✅ |
| Optimalizace: cíle jX, PSV, zisk, F/B, elevace, proud | ✅ |
| Optimalizace: svázané proměnné (association) | ✅ |
| Optimalizace: krok proměnné (pitch) | ✅ |
| Optimalizace: All elements | ✅ |
| Tools: rezonance, cívka, LC match, stub match, line match | ✅ |
| Hairpin match | ✅ |
| Historie výpočtů | ✅ |
| Import/export MMANA .maa a NEC | ✅ |
| **Grafický editor drátů myší** (XY/XZ/YZ pohledy) | ❌ zatím ne |
| **Režim λ** (rozměry ve vlnových délkách) | ❌ |
| **Zúžená segmentace** (DM1/DM2/SC/EC) | ❌ jen rovnoměrná |
| **Search & Replace souřadnic** | ❌ |
| **Log optimalizace se 128 kroky a ručním výběrem** | ❌ |
| **Překryv diagramů více antén** (.mab porovnání) | ❌ historie je jen tabulka |
| **Tisk** | ❌ jen uložení obrázku |
| Sommerfeldova zem (MMANA ji taky nemá) | ✅ s jádrem NEC-2 |

---

## Kde být opatrný

Tohle jsou skutečná omezení modelu, ne formality:

* **Reálná zem v impedanci je u vlastního jádra jen přibližná.** Matice se
  počítá s obrazem nad dokonalou zemí (stejně jako MININEC, ze kterého vychází
  i MMANA); Fresnelovy koeficienty se uplatní až ve vyzařování. U vodorovných
  antén nad zemí to na zisk sedí do ~0,2 dB, na impedanci je odchylka jednotky
  ohmů. **Přepni na jádro NEC-2** — má Sommerfeldovu zem a tenhle problém nemá.
* **Vertikál napájený přímo ze země nad reálnou zemí neumí ani jedno jádro** —
  viz tabulka jader výše.
* **Zakopané radiály neumí** nikdo z této třídy programů, tenhle taky ne.
* **Velmi tlusté dráty** (poloměr nad ~λ/200) — reaktance se začne rozcházet,
  odpor zůstává dobrý. Pro trubky 25 mm na 20 m je odchylka pod 1 Ω.
* **Segment musí být delší než 4× poloměr drátu.** Program to hlídá a upozorní.
* **Zúžený prvek je aproximace, ale slušná.** Skok průměru neumí tenkodrátový
  model přesně žádný program této třídy. Porovnání s publikovanými hodnotami
  (W4RNL, „Tapering to Perfection“, dipóly 14 MHz — referencí je korigovaná
  hodnota, která sedí s MININECem):

  | případ | reference | vlastní jádro | NEC-2 nekorig. |
  |---|---:|---:|---:|
  | celistvý 1,0″ | 71,8 −0,6j | 71,6 −1,9j | 72,0 +0,4j |
  | skok daleko od středu | 72,0 +0,4j | 71,5 −3,9j | 73,7 +6,5j |
  | skok blízko středu | 71,8 −0,5j | 73,0 −2,3j | 72,6 +6,8j |

  Vlastní jádro se mýlí spíš do kapacity, NEC-2 do indukce a o kus víc —
  **u zúžených prvků je vlastní jádro bližší pravdě než NEC-2.** Přesnou
  odpověď dá až Leesonova korekce (náhradní prvek), ta tu zatím není.
* **Zúžení prvků anténu rozladí** — prvek z trubek 25/20/16 je elektricky
  kratší než stejně dlouhý prvek z 16 mm. Po poskládání prvků se musí
  optimalizace pustit znovu; není to chyba výpočtu.
* **Činitel jakosti Q** je lokální derivace impedance. U antén záměrně
  zploštělých přes celé pásmo je nestabilní — program to pozná a místo jedné
  hodnoty ukáže rozsah. Skutečná šířka pásma se pozná jen z rozmítání.
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
  optimize.py   parametry, cílová funkce, GA + Nelder-Mead, svázané proměnné
  geometry_ops.py  posun/rotace/zrcadlení/měřítko, přeladění, stoh, zúžené prvky
  wizards.py    průvodci tvorbou antén
  match.py      návrh vlásenky (hairpin), ladění zářiče
  hfcalc.py     rezonance, cívky, LC článek, pahýly, vedení, napáječe
  engines.py    výměnná jádra (vlastní MoM / NEC-2 přes PyNEC)
  fileio.py     import/export NEC a MMANA
  examples.py   vestavěné příklady
  dialogs.py    dialogy GUI
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
