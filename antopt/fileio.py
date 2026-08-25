"""Import a export: NEC (.nec) a MMANA (.maa / .mma).

MMANA formát je popsán volně dostupnou specifikací (OpenNEC).  Import je
"best effort" — nepodporované prvky (zúžená segmentace, paralelní zátěže,
Sommerfeld-Norton zem) se mapují na nejbližší podporovanou variantu a hlásí
se ve varováních.
"""
from __future__ import annotations

import math
import re
from typing import List, Optional, Tuple

from .model import Model, Wire, Source, Load, Ground, _ranges as _ranges_txt


# ==========================================================================
#  NEC
# ==========================================================================
def to_nec(model: Model) -> str:
    lines = [f"CM {model.name}", "CM vygenerovano AntOpt", "CE"]
    for i, w in enumerate(model.wires, start=1):
        lines.append(
            f"GW {i} {w.nseg} {w.x1:.6f} {w.y1:.6f} {w.z1:.6f} "
            f"{w.x2:.6f} {w.y2:.6f} {w.z2:.6f} {w.radius:.6f}"
        )
    lines.append("GE 1" if model.ground.kind != "free" else "GE 0")
    if model.ground.kind == "perfect":
        lines.append("GN 1")
    elif model.ground.kind == "real":
        lines.append(f"GN 2 0 0 0 {model.ground.eps_r:g} {model.ground.sigma:g}")
    sigma = model.conductivity()
    if sigma > 0:
        lines.append(f"LD 5 0 0 0 {sigma:g}")
    for ld in model.loads:
        seg = _seg_index(model, ld.wire, ld.pos)
        if ld.kind == "RX":
            lines.append(f"LD 4 {ld.wire + 1} {seg} {seg} {ld.r:g} {ld.x:g}")
        else:
            lines.append(f"LD 0 {ld.wire + 1} {seg} {seg} {ld.r:g} "
                         f"{ld.l_uh * 1e-6:g} {ld.c_pf * 1e-12:g}")
    for s in model.sources:
        seg = _seg_index(model, s.wire, s.pos)
        vr = s.voltage * math.cos(math.radians(s.phase))
        vi = s.voltage * math.sin(math.radians(s.phase))
        lines.append(f"EX 0 {s.wire + 1} {seg} 0 {vr:.6f} {vi:.6f}")
    lines.append(f"FR 0 1 0 0 {model.freq_mhz:g} 0")
    lines.append("RP 0 91 361 1000 0 0 1 1")
    lines.append("EN")
    return "\n".join(lines) + "\n"


def _seg_index(model: Model, wire: int, pos: float) -> int:
    n = max(1, model.wires[wire].nseg)
    return max(1, min(n, int(round(pos * n + 0.5))))


def _pos_from_seg(seg: int, n: int) -> float:
    """Střed segmentu -> relativní poloha, se srovnáním na střed a konce drátu.

    NEC budí střed segmentu, takže při sudém počtu segmentů neumí trefit
    přesný střed drátu. Poloha do půl segmentu od středu (nebo od konce) se
    proto zarovná — jinak by se napájecí bod při každém průchodu formátem
    posouval.
    """
    n = max(1, int(n))
    pos = (seg - 0.5) / n
    tol = 0.5 / n + 1e-9
    for anchor in (0.0, 0.5, 1.0):
        if abs(pos - anchor) <= tol:
            return anchor
    return pos


def from_nec(text: str) -> Tuple[Model, List[str]]:
    warn: List[str] = []
    m = Model(name="import NEC")
    ground_kind = "free"
    eps, sig = 13.0, 0.005
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        tag = line[:2].upper()
        rest = line[2:].replace(",", " ").split()
        try:
            if tag == "CM":
                if m.name == "import NEC" and line[2:].strip():
                    m.name = line[2:].strip()
            elif tag == "GW":
                _, ns, x1, y1, z1, x2, y2, z2, r = rest[:9]
                m.wires.append(Wire(float(x1), float(y1), float(z1),
                                    float(x2), float(y2), float(z2),
                                    float(r), max(1, int(float(ns)))))
            elif tag == "GN":
                t = int(float(rest[0]))
                if t == 1:
                    ground_kind = "perfect"
                elif t in (0, 2, -1):
                    ground_kind = "real"
                    if len(rest) >= 6:
                        eps, sig = float(rest[4]), float(rest[5])
                    if t == -1:
                        warn.append("GN -1 (bez země) mapováno na reálnou zem.")
            elif tag == "GE":
                if rest and int(float(rest[0])) == 0:
                    ground_kind = "free"
            elif tag == "EX":
                typ = int(float(rest[0]))
                if typ != 0:
                    warn.append(f"EX typ {typ} není podporován, beru jako napěťový zdroj.")
                wi = int(float(rest[1])) - 1
                seg = int(float(rest[2]))
                vr = float(rest[4]) if len(rest) > 4 else 1.0
                vi = float(rest[5]) if len(rest) > 5 else 0.0
                v = complex(vr, vi)
                n = m.wires[wi].nseg if 0 <= wi < len(m.wires) else 1
                m.sources.append(Source(wi, _pos_from_seg(seg, n), abs(v),
                                        math.degrees(math.atan2(vi, vr))))
            elif tag == "LD":
                t = int(float(rest[0]))
                if t == 5:
                    m.material = "měď"
                    continue
                wi = int(float(rest[1])) - 1
                seg = int(float(rest[2]))
                n = m.wires[wi].nseg if 0 <= wi < len(m.wires) else 1
                pos = _pos_from_seg(seg, n)
                if t == 4:
                    m.loads.append(Load(wi, pos, "RX", float(rest[4]), float(rest[5])))
                elif t == 0:
                    r_ = float(rest[4]); l_ = float(rest[5]) if len(rest) > 5 else 0.0
                    c_ = float(rest[6]) if len(rest) > 6 else 0.0
                    m.loads.append(Load(wi, pos, "RLC", r_, 0.0, l_ * 1e6, c_ * 1e12))
                else:
                    warn.append(f"LD typ {t} přeskočen.")
            elif tag == "FR":
                if len(rest) >= 5:
                    m.freq_mhz = float(rest[4])
        except (ValueError, IndexError) as e:
            warn.append(f"Nepřečteno: {line}  ({e})")
    m.ground = Ground(ground_kind, eps, sig)
    if not m.wires:
        raise ValueError("Soubor neobsahuje žádnou GW kartu.")
    return m, warn


# ==========================================================================
#  MMANA .maa
# ==========================================================================
_NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def _nums(line: str) -> List[float]:
    return [float(x) for x in re.findall(_NUM, line)]


MAA_TAPER_END = 99999.9        # „tahle trubka jde až na konec prvku“


def _maa_taper_table(lines: List[str]) -> dict:
    """Rozpisy teleskopických prvků: {klíč: [(délka, poloměr), …]}.

    Řádek má tvar ``klíč, 0, délka, poloměr, délka, poloměr, …, 99999.9, poloměr``.
    Klíč je záporné číslo, kterým se prvek odkazuje ze sloupce poloměru.
    """
    out: dict = {}
    for s in lines:
        s = s.strip()
        if not s or s[0] in "*#$":
            continue
        v = _nums(s)
        if not any(abs(x - MAA_TAPER_END) < 1.0 for x in v):
            continue
        # na jednom řádku může být i víc rozpisů za sebou — každý začíná
        # svým záporným klíčem
        starts = [k for k, x in enumerate(v) if -0.5 < x < 0]
        for si, k0 in enumerate(starts):
            k1 = starts[si + 1] if si + 1 < len(starts) else len(v)
            chunk = v[k0:k1]
            if len(chunk) < 4:
                continue
            rest = chunk[2:] if abs(chunk[1]) < 1e-12 else chunk[1:]
            pairs = []
            for k in range(0, len(rest) - 1, 2):
                ln, rad = rest[k], rest[k + 1]
                if rad <= 0 or ln <= 0:
                    break
                pairs.append((ln, rad))
                if abs(ln - MAA_TAPER_END) < 1.0:
                    break
            if pairs and abs(pairs[-1][0] - MAA_TAPER_END) < 1.0:
                out[round(chunk[0], 6)] = pairs
    return out


def _taper_key(radius: float, defs: dict) -> Optional[float]:
    if radius >= 0 or not defs:
        return None
    k = round(radius, 6)
    if k in defs:
        return k
    for cand in defs:
        if abs(cand - radius) < 1e-7:
            return cand
    return None


def _taper_sections(pairs: List[Tuple[float, float]], half: float):
    """Rozpis trubek -> sekce od středu ven pro ``geometry_ops.taper_element``.

    První trubka je středová, její délka platí přes celý střed (na jednu
    stranu tedy polovina). Další už jsou na každou stranu. Poslední
    (99999.9) dopne prvek až na konec.
    """
    from .geometry_ops import TaperSection
    secs, used = [], 0.0
    for k, (ln, rad) in enumerate(pairs):
        if abs(ln - MAA_TAPER_END) < 1.0 or used >= half - 1e-6:
            secs.append(TaperSection(max(half - used, 1e-3), rad))
            used = half
            break
        step = (ln / 2.0) if k == 0 else ln
        step = min(step, half - used)
        if step <= 1e-6:
            continue
        secs.append(TaperSection(step, rad))
        used += step
    if used < half - 1e-6:
        secs.append(TaperSection(half - used, pairs[-1][1]))
    return secs


def from_maa(text: str) -> Tuple[Model, List[str]]:
    warn: List[str] = []
    lines = [l.rstrip() for l in text.splitlines()]
    i = 0

    def skip_blank():
        nonlocal i
        while i < len(lines) and not lines[i].strip():
            i += 1

    # --- hlavička: název, kmitočet, počet drátů
    #
    # Hlaviček je v praxi několik podob — komentář, název antény, nebo obojí.
    # Název přitom bývá plný číslic („OK1M-14-21-28_9el“), takže se z něj
    # nesmí stát kmitočet. Trojice (kmitočet, počet drátů, dráty) se proto
    # bere jako celek a ověřuje se: za počtem drátů musí opravdu následovat
    # tolik řádků s osmi čísly. Když to nesedí, zkusí se další kandidát.
    def _payload(idx: int, strict: bool = True) -> Optional[Tuple[float, int, int]]:
        """(kmitočet, počet drátů, index prvního drátu) pro kandidáta na řádku idx."""
        v = _nums(lines[idx])
        if not v or not (0.001 <= v[0] <= 300000.0):
            return None
        j = idx + 1
        while j < len(lines):
            s = lines[j].strip()
            if not s or s.startswith("*") or s.startswith("#"):
                j += 1
                continue
            w = _nums(s)
            if not w:
                return None
            n = int(w[0])
            if not (1 <= n <= 5000) or w[0] != n:
                return None
            k, seen = j + 1, 0
            while k < len(lines) and seen < n:
                t = lines[k].strip()
                k += 1
                if not t or t.startswith("*") or t.startswith("#"):
                    continue
                if len(_nums(t)) < 8:
                    return None
                seen += 1
            if seen != n:
                return None
            # Přednost dostane kandidát, za jehož posledním drátem už další
            # řádek s osmi čísly nenásleduje — jinak by šlo o špatně přečtený
            # počet drátů. Když takový kandidát není, bere se i tenhle
            # (dráty se stejně čtou greedy a nesoulad se ohlásí).
            while strict and k < len(lines):
                t = lines[k].strip()
                k += 1
                if not t or t.startswith("*") or t.startswith("#"):
                    break
                if len(_nums(t)) >= 8:
                    return None
                break
            return v[0], n, j + 1
        return None

    name = ""
    head: Optional[Tuple[float, int, int]] = None
    for strict in (True, False):
        name = ""
        for idx in range(len(lines)):
            s = lines[idx].strip()
            if not s:
                continue
            core = s.strip("*").strip()
            if not core:
                continue
            if _looks_numeric(core):
                head = _payload(idx, strict)
                if head:
                    break
            else:
                # název antény je poslední textový řádek před kmitočtem,
                # předchozí bývá jen hlavička souboru („MMANA-GAL antenna file“)
                name = core
        if head:
            break
    if head is None:
        raise ValueError("Hlavička souboru nedává smysl — nenašel jsem "
                         "kmitočet a počet drátů.")
    freq, n_wires, i = head

    model = Model(name=name or "import MMANA", freq_mhz=freq)

    # --- předehra: tabulka teleskopických prvků
    #
    # Zúžený prvek MMANA nepíše jako několik drátů. Do sloupce poloměru dá
    # ZÁPORNÝ KLÍČ (-0.001, -0.002, …) a rozpis trubek uloží zvlášť:
    #
    #     -0.001, 0, 2.4, 0.015, 1.2, 0.0125, 1.2, 0.01, 99999.9, 0.008
    #      klíč   ?  délka,poloměr  …  99999.9 = „až na konec“
    #
    # První trubka je středová a její délka platí přes celý střed, další
    # už jsou na každou stranu. Bez téhle tabulky se z prvku z trubek
    # 30/25/20/16 stal drát o poloměru 1 mm — anténa pak vyšla o 1 dB hůř
    # a s úplně jinou impedancí.
    taper_defs = _maa_taper_table(lines)
    taper_hits: List[Tuple[int, float]] = []
    maybe_height = [None]

    # Dráty se čtou, dokud řádky vypadají jako dráty — ne jen deklarovaný počet.
    # Kdyby se počet přečetl špatně, zbytek geometrie by se tiše zahodil.
    read = 0
    n_taper = n_negr = n_negr_mm = 0
    thin: List[int] = []
    while i < len(lines):
        s = lines[i].strip()
        if not s or s.startswith("#"):
            i += 1
            continue
        if s.startswith("*"):
            if read >= n_wires:
                break
            i += 1
            continue
        v = _nums(s)
        if len(v) < 8:
            if read >= n_wires:
                break
            i += 1
            continue
        i += 1
        x1, y1, z1, x2, y2, z2, r, ns = v[:8]
        nseg = int(ns)
        if nseg <= 0:
            nseg = 0            # doplní se automaticky
            if int(ns) < 0:
                n_taper += 1
        key = _taper_key(r, taper_defs)
        if key is not None:
            taper_hits.append((read, key))
            r = taper_defs[key][-1][1]        # prozatím poloměr koncové trubky
        elif r < 0:
            # Záporný poloměr je v .maa značka „hodnota je v jiné jednotce“.
            # Která to je, se pozná z velikosti: poloměr vodiče přes 0,5 m
            # neexistuje, takže velké číslo jsou milimetry a malé metry.
            # Slepé dělení tisícem udělalo z trubky 25 mm drát 0,0125 mm,
            # což anténu zabilo — ztráty vodiče sežraly 80 % výkonu.
            if abs(r) >= 0.5:
                r = abs(r) / 1000.0
                n_negr_mm += 1
            else:
                r = abs(r)
                n_negr += 1
        if r < 2.5e-5:                      # tenčí než 0,05 mm v průměru
            thin.append(read + 1)
        model.wires.append(Wire(x1, y1, z1, x2, y2, z2, max(r, 1e-6), max(nseg, 0)))
        read += 1
    if read != n_wires:
        warn.append(f"Hlavička hlásí {n_wires} drátů, v souboru jich je {read} "
                    f"— načteno všech {read}.")

    # další sekce
    section = None
    pending = 0
    while i < len(lines):
        s = lines[i].strip()
        i += 1
        if not s:
            continue
        low = s.lower()
        if s[0] in "*$":
            # MMANA odděluje sekce hvězdičkami i dolary
            # (***Wires***, $$$Taper wire set$$$)
            if "taper" in low:
                section, pending = "taper", None
            elif "source" in low:
                section, pending = "src", None
            elif "load" in low:
                section, pending = "load", None
            elif "g/h/m" in low or "azel" in low:
                section, pending = "ground", None
            elif "segment" in low:
                section, pending = "seg", None
            else:
                section = None
            continue
        if s.startswith("#"):
            continue
        if section == "taper":
            continue                       # rozpisy trubek se čtou zvlášť předem

        if section == "src":
            if pending is None:
                pending = int(_nums(s)[0]) if _nums(s) else 0
                continue
            if pending > 0:
                _parse_maa_source(model, s, warn)
                pending -= 1
        elif section == "load":
            if pending is None:
                pending = int(_nums(s)[0]) if _nums(s) else 0
                continue
            if pending > 0:
                _parse_maa_load(model, s, warn)
                pending -= 1
        elif section == "ground":
            v = _nums(s)
            if v:
                t = int(v[0])
                if t == 0:
                    model.ground = Ground("free")
                elif t == 1:
                    model.ground = Ground("perfect")
                else:
                    sigma = v[1] / 1000.0 if len(v) > 1 else 0.005
                    eps = v[2] if len(v) > 2 and 1.0 <= v[2] <= 90.0 else 13.0
                    model.ground = Ground("real", eps, max(sigma, 1e-5))
                    # Sekce se jmenuje G/H/M/R — druhé pole je podle názvu
                    # výška, podle jiné dokumentace vodivost. Rozhodne se
                    # až podle geometrie (viz níž): model položený v z = 0
                    # se zapnutou zemí nedává smysl, takže tam jde o výšku.
                    if len(v) > 1:
                        maybe_height[0] = float(v[1])
                if len(v) > 3 and v[3] in (50.0, 75.0, 112.0, 200.0, 300.0, 450.0, 600.0):
                    model.z0 = v[3]
                if t == -1:
                    warn.append("Sommerfeld-Norton zem nahrazena aproximací reálné země.")
                section = None

    if not model.sources:
        warn.append("Soubor neobsahoval zdroj — vložen do středu prvního drátu.")
        model.sources.append(Source(0, 0.5, 1.0))

    # --- rozbalit teleskopické prvky (až teď, ať se zdroje přečíslují správně)
    if taper_hits:
        from .geometry_ops import taper_element
        popis = []
        for wi, key in sorted(taper_hits, reverse=True):
            if wi >= len(model.wires):
                continue
            half = model.wires[wi].length / 2.0
            secs = _taper_sections(taper_defs[key], half)
            try:
                taper_element(model, wi, secs)
            except ValueError as e:
                warn.append(f"Drát {wi + 1}: zúžený prvek se nepodařilo "
                            f"sestavit ({e}).")
                continue
            popis.append("Ø " + "/".join(
                f"{s.radius * 2000:g}" for s in secs) + " mm")
        if popis:
            uniq = sorted(set(popis))
            warn.append(f"{len(popis)}× teleskopický prvek sestaven z rozpisu "
                        f"trubek v souboru: " + "; ".join(uniq) + ".")

    if n_negr_mm:
        warn.append(f"{n_negr_mm}× poloměr zapsaný záporně, hodnota v milimetrech "
                    f"— přepočteno na metry.")
    if n_negr:
        warn.append(f"{n_negr}× poloměr zapsaný záporně, hodnota už v metrech "
                    f"— vzato jak je.")
    if model.wires:
        d = sorted({round(w.radius * 2000, 2) for w in model.wires})
        rng = f"{d[0]:g} mm" if len(d) == 1 else f"{d[0]:g} až {d[-1]:g} mm"
        warn.append(f"ZKONTROLUJ PRŮMĚRY VODIČŮ: {rng}. "
                    f"Musí sedět na skutečné trubky — špatná jednotka průměru "
                    f"anténu zabije (ztráty vodiče), ale nic jiného to nepozná.")
    if thin:
        warn.append(f"Dráty {_ranges_txt(thin)} jsou tenčí než 0,05 mm. To není "
                    f"reálný vodič — skoro všechen výkon se v něm spálí. "
                    f"Skoro jistě je špatně jednotka poloměru v souboru.")
    if n_taper:
        warn.append(
            f"{n_taper}× nerovnoměrné dělení drátu na segmenty (MMANA -1/-2/-3) "
            f"nahrazeno rovnoměrným.\n"
            f"POZOR, tohle NENÍ o zúžených prvcích: týká se to jen toho, na kolik "
            f"dílků se drát krájí pro výpočet, ne průměrů trubek. Prvek složený "
            f"z trubek různého průměru se importuje se všemi průměry.\n"
            f"MMANA krájí segmenty ke koncům drátu nakrátko, aby vystačila "
            f"s menším počtem. Tenhle program krájí rovnoměrně, ale hustěji "
            f"(45 segmentů na vlnovou délku), takže vyjde totéž.")

    # segmenty doplň jen tam, kde je soubor nezadal
    explicit = {i: w.nseg for i, w in enumerate(model.wires) if w.nseg > 0}
    if len(explicit) < len(model.wires):
        model.auto_segment()
        for i, n in explicit.items():
            model.wires[i].nseg = n
    if not model.wires:
        raise ValueError("Soubor neobsahuje žádné dráty.")

    h = maybe_height[0]
    if h is not None and model.lies_on_ground() and 0.1 <= h <= 200.0:
        # druhé pole je výška, ne vodivost — vodivost se vrátí na výchozí
        setattr(model, "_maa_height", h)
        model.ground = Ground("real", model.ground.eps_r, 0.005)
    else:
        h = None
    if model.lies_on_ground():
        if h:
            warn.append(
                f"Celá anténa ({len(model.wires)} drátů) leží v rovině z = 0 — "
                f"MMANA takové modely kreslí kolem počátku a výšku drží zvlášť. "
                f"V sekci G/H/M/R je hodnota {h:g}; jestli je to výška nad zemí, "
                f"model se o ni musí zvednout, jinak je zkratovaný do země.")
        else:
            warn.append(
            f"Celá anténa ({len(model.wires)} drátů) leží v rovině z = 0 a zem je "
            f"zapnutá — takhle je zkratovaná do země a výsledky nedávají smysl. "
            f"MMANA takové modely kreslí kolem počátku a výšku zadává zvlášť; "
            f"zvedni model na skutečnou výšku (Úpravy → Posun, dz) nebo přepni "
            f"na volný prostor.")
    return model, warn


def _designator(model: Model, token: str) -> Tuple[int, float]:
    """'w2c', 'w3b1', 'W1E-2' -> (index drátu, relativní poloha)."""
    t = token.strip().upper().lstrip("VW")
    mm = re.match(r"(\d+)\s*([CBE])\s*([-+]?\d+)?", t)
    if not mm:
        raise ValueError(f"Neznámý zápis polohy: {token}")
    wi = int(mm.group(1)) - 1
    where = mm.group(2)
    off = int(mm.group(3)) if mm.group(3) else 0
    n = model.wires[wi].nseg if 0 <= wi < len(model.wires) else 11
    n = max(n, 1)
    base = {"C": n / 2.0, "B": 0.0, "E": float(n)}[where]
    return wi, min(1.0, max(0.0, (base + off) / n))


def _parse_maa_source(model: Model, line: str, warn: List[str]) -> None:
    parts = [p.strip() for p in line.replace("\t", ",").split(",") if p.strip()]
    if not parts:
        return
    try:
        wi, pos = _designator(model, parts[0])
    except ValueError as e:
        warn.append(str(e))
        return
    phase = float(parts[1]) if len(parts) > 1 else 0.0
    mag = float(parts[2]) if len(parts) > 2 else 1.0
    model.sources.append(Source(wi, pos, mag, phase))


def _parse_maa_load(model: Model, line: str, warn: List[str]) -> None:
    parts = [p.strip() for p in line.replace("\t", ",").split(",") if p.strip()]
    if not parts:
        return
    try:
        if re.match(r"^[VW]?\d+\s*[CBE]", parts[0].upper()):
            wi, pos = _designator(model, parts[0])
            vals = [float(x) for x in parts[1:]]
        else:
            v = _nums(line)
            wi = int(v[0]) - 1
            n = max(1, model.wires[wi].nseg) if 0 <= wi < len(model.wires) else 11
            pos = min(1.0, max(0.0, v[1] / n))
            vals = v[2:]
    except (ValueError, IndexError) as e:
        warn.append(f"Zátěž nepřečtena: {line} ({e})")
        return
    r = vals[0] if len(vals) > 0 else 0.0
    x = vals[1] if len(vals) > 1 else 0.0
    l_uh = vals[2] if len(vals) > 2 else 0.0
    c_pf = vals[3] if len(vals) > 3 else 0.0
    if l_uh or c_pf:
        model.loads.append(Load(wi, pos, "RLC", r, 0.0, l_uh, c_pf))
    else:
        model.loads.append(Load(wi, pos, "RX", r, x))


def to_maa(model: Model) -> str:
    out = [model.name or "Antenna", f"{model.freq_mhz:g}", "***Wires***",
           str(len(model.wires))]
    for w in model.wires:
        out.append(f"{w.x1:.5f}, {w.y1:.5f}, {w.z1:.5f}, "
                   f"{w.x2:.5f}, {w.y2:.5f}, {w.z2:.5f}, "
                   f"{w.radius:.5f}, {w.nseg}")
    out.append("***Source***")
    out.append(str(len(model.sources)))
    for s in model.sources:
        out.append(f"{_maa_tag(model, s.wire, s.pos)}, {s.phase:g}, {s.voltage:g}")
    out.append("***Load***")
    out.append(str(len(model.loads)))
    for l in model.loads:
        n = max(1, model.wires[l.wire].nseg)
        seg = max(1, min(n, int(round(l.pos * n + 0.5))))
        out.append(f"{l.wire + 1}, {seg}, {l.r:g}, {l.x:g}, {l.l_uh:g}, {l.c_pf:g}")
    gtype = {"free": 0, "perfect": 1, "real": 2}[model.ground.kind]
    out.append("***G/H/M/R/AzEl/X***")
    out.append(f"{gtype}, {model.ground.sigma * 1000:g}, {model.ground.eps_r:g}, "
               f"{model.z0:g}, 0, 0, 0")
    return "\n".join(out) + "\n"


def _maa_tag(model: Model, wire: int, pos: float) -> str:
    """Označení polohy na drátu v zápisu MMANA (w1b, w2c, w3c+1, w1e)."""
    n = max(1, model.wires[wire].nseg)
    if pos <= 1e-9:
        return f"w{wire + 1}b"
    if pos >= 1.0 - 1e-9:
        return f"w{wire + 1}e"
    off = int(round(pos * n - n / 2.0))
    return f"w{wire + 1}c" + (f"{off:+d}" if off else "")


def _looks_numeric(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    try:
        float(s.split(",")[0].split()[0])
        return True
    except (ValueError, IndexError):
        return False
