"""Import a export: NEC (.nec) a MMANA (.maa / .mma).

MMANA formát je popsán volně dostupnou specifikací (OpenNEC).  Import je
"best effort" — nepodporované prvky (zúžená segmentace, paralelní zátěže,
Sommerfeld-Norton zem) se mapují na nejbližší podporovanou variantu a hlásí
se ve varováních.
"""
from __future__ import annotations

import math
import re
from typing import List, Tuple

from .model import Model, Wire, Source, Load, Ground


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


def from_maa(text: str) -> Tuple[Model, List[str]]:
    warn: List[str] = []
    lines = [l.rstrip() for l in text.splitlines()]
    i = 0

    def skip_blank():
        nonlocal i
        while i < len(lines) and not lines[i].strip():
            i += 1

    name = ""
    skip_blank()
    if i < len(lines) and not _looks_numeric(lines[i]):
        name = lines[i].strip().lstrip("*").strip()
        i += 1
    skip_blank()
    while i < len(lines) and lines[i].strip() in ("*", "**", "***"):
        i += 1
    skip_blank()

    freq = 14.1
    if i < len(lines):
        n = _nums(lines[i])
        if n:
            freq = n[0]
            i += 1

    model = Model(name=name or "import MMANA", freq_mhz=freq)

    # počet drátů: buď samostatně, nebo za hlavičkou ***Wires***
    n_wires = None
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        if s.startswith("*"):
            i += 1
            continue
        nums = _nums(s)
        if nums:
            n_wires = int(nums[0])
            i += 1
            break
        i += 1
    if n_wires is None:
        raise ValueError("V souboru není počet drátů.")

    read = 0
    while i < len(lines) and read < n_wires:
        s = lines[i].strip()
        i += 1
        if not s or s.startswith("*") or s.startswith("#"):
            continue
        v = _nums(s)
        if len(v) < 8:
            continue
        x1, y1, z1, x2, y2, z2, r, ns = v[:8]
        nseg = int(ns)
        if nseg <= 0:
            nseg = 0            # doplní se automaticky
            if int(ns) < 0:
                warn.append("Zúžená segmentace (-1/-2/-3) nahrazena rovnoměrnou.")
        if r < 0:
            warn.append("Záporný poloměr (poloměr v mm) přepočten.")
            r = abs(r) / 1000.0
        model.wires.append(Wire(x1, y1, z1, x2, y2, z2, max(r, 1e-5), max(nseg, 0)))
        read += 1

    # další sekce
    section = None
    pending = 0
    while i < len(lines):
        s = lines[i].strip()
        i += 1
        if not s:
            continue
        low = s.lower()
        if s.startswith("*"):
            if "source" in low:
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
                if len(v) > 3 and v[3] in (50.0, 75.0, 112.0, 200.0, 300.0, 450.0, 600.0):
                    model.z0 = v[3]
                if t == -1:
                    warn.append("Sommerfeld-Norton zem nahrazena aproximací reálné země.")
                section = None

    if not model.sources:
        warn.append("Soubor neobsahoval zdroj — vložen do středu prvního drátu.")
        model.sources.append(Source(0, 0.5, 1.0))
    for w in model.wires:
        if w.nseg <= 0:
            w.nseg = 0
    if any(w.nseg <= 0 for w in model.wires):
        model.auto_segment()
    if not model.wires:
        raise ValueError("Soubor neobsahuje žádné dráty.")
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
