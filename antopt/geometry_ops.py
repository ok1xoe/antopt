"""Operace nad geometrií — obdoba nabídky Edit v MMANA.

Posun, rotace, zrcadlení, škálování, přeladění na jiný kmitočet, stohování,
zúžené (teleskopické) prvky, polární zadání drátu a pohled po prvcích.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .model import Model, Wire, Source, Load, C0

AXES = {"x": 0, "y": 1, "z": 2}


def _idx(sel: Optional[Iterable[int]], n: int) -> List[int]:
    return list(range(n)) if sel is None else [i for i in sel if 0 <= i < n]


def _set(w: Wire, a: np.ndarray, b: np.ndarray) -> None:
    w.x1, w.y1, w.z1 = (float(v) for v in a)
    w.x2, w.y2, w.z2 = (float(v) for v in b)


# --------------------------------------------------------------------------
def move(model: Model, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0,
         wires: Optional[Iterable[int]] = None) -> None:
    """Posune vybrané dráty o zadaný vektor."""
    d = np.array([dx, dy, dz], dtype=float)
    for i in _idx(wires, len(model.wires)):
        w = model.wires[i]
        _set(w, w.a + d, w.b + d)


def rotate(model: Model, angle_deg: float, axis: str = "z",
           center: Optional[Sequence[float]] = None,
           wires: Optional[Iterable[int]] = None) -> None:
    """Otočí vybrané dráty kolem osy procházející bodem ``center``."""
    ax = AXES[axis.lower()]
    c = np.zeros(3) if center is None else np.asarray(center, dtype=float)
    t = math.radians(angle_deg)
    ct, st = math.cos(t), math.sin(t)
    R = np.eye(3)
    i, j = [k for k in range(3) if k != ax]
    R[i, i] = ct; R[i, j] = -st
    R[j, i] = st; R[j, j] = ct
    for k in _idx(wires, len(model.wires)):
        w = model.wires[k]
        _set(w, R @ (w.a - c) + c, R @ (w.b - c) + c)


def mirror(model: Model, plane: str = "x", coord: float = 0.0,
           wires: Optional[Iterable[int]] = None, copy: bool = False) -> None:
    """Zrcadlí dráty podle roviny kolmé na osu ``plane``.

    ``copy=True`` původní dráty zachová a přidá zrcadlené.
    """
    ax = AXES[plane.lower()]
    sel = _idx(wires, len(model.wires))
    new: List[Wire] = []
    for k in sel:
        w = model.wires[k]
        a, b = w.a.copy(), w.b.copy()
        a[ax] = 2 * coord - a[ax]
        b[ax] = 2 * coord - b[ax]
        if copy:
            new.append(Wire(*a, *b, w.radius, w.nseg, w.name))
        else:
            _set(w, a, b)
    model.wires.extend(new)


def scale(model: Model, factor: float, center: Optional[Sequence[float]] = None,
          scale_radius: bool = True, wires: Optional[Iterable[int]] = None) -> None:
    """Zvětší/zmenší geometrii daným poměrem."""
    c = np.zeros(3) if center is None else np.asarray(center, dtype=float)
    for k in _idx(wires, len(model.wires)):
        w = model.wires[k]
        _set(w, c + (w.a - c) * factor, c + (w.b - c) * factor)
        if scale_radius:
            w.radius = max(1e-6, w.radius * factor)


def rescale_to_frequency(model: Model, new_freq_mhz: float,
                         scale_radius: bool = True,
                         keep_height: bool = False) -> float:
    """Přeladí celou anténu na jiný kmitočet (Edit → Wire Scale v MMANA).

    Vrací použitý poměr. ``keep_height=True`` nechá výšku nad zemí beze změny.
    """
    if new_freq_mhz <= 0 or model.freq_mhz <= 0:
        raise ValueError("Kmitočet musí být kladný.")
    f = model.freq_mhz / new_freq_mhz
    heights = [(w.z1, w.z2) for w in model.wires] if keep_height else None
    scale(model, f, center=np.zeros(3), scale_radius=scale_radius)
    if heights is not None:
        for w, (z1, z2) in zip(model.wires, heights):
            w.z1, w.z2 = z1, z2
    model.freq_mhz = new_freq_mhz
    return f


# --------------------------------------------------------------------------
def make_stack(model: Model, nx: int = 1, nz: int = 2,
               dx: float = 0.0, dz: float = 0.0,
               feed_all: bool = True, phase_step: float = 0.0) -> None:
    """Vytvoří stoh / řadu kopií antény (Edit → Make Stack).

    ``nx`` kopií vedle sebe s roztečí ``dx`` [m], ``nz`` nad sebou s roztečí
    ``dz`` [m].  Sestava se vycentruje kolem původní polohy ve směru Y a
    naskládá vzhůru ve směru Z.
    """
    nx, nz = max(1, int(nx)), max(1, int(nz))
    if nx * nz <= 1:
        return
    if nx * nz > 64:
        raise ValueError("Nejvýš 64 kopií (jako v MMANA).")
    base_w = [Wire(w.x1, w.y1, w.z1, w.x2, w.y2, w.z2, w.radius, w.nseg, w.name)
              for w in model.wires]
    base_s = [Source(s.wire, s.pos, s.voltage, s.phase) for s in model.sources]
    base_l = [Load(l.wire, l.pos, l.kind, l.r, l.x, l.l_uh, l.c_pf, l.parallel)
              for l in model.loads]
    n_w = len(base_w)

    model.wires, model.sources, model.loads = [], [], []
    y0 = -(nx - 1) * dx / 2.0
    k = 0
    for iz in range(nz):
        for ix in range(nx):
            off = np.array([0.0, y0 + ix * dx, iz * dz])
            for w in base_w:
                model.wires.append(Wire(*(w.a + off), *(w.b + off),
                                        w.radius, w.nseg, w.name))
            if feed_all or k == 0:
                for s in base_s:
                    model.sources.append(Source(s.wire + k * n_w, s.pos,
                                                s.voltage, s.phase + k * phase_step))
            for l in base_l:
                model.loads.append(Load(l.wire + k * n_w, l.pos, l.kind,
                                        l.r, l.x, l.l_uh, l.c_pf, l.parallel))
            k += 1


# --------------------------------------------------------------------------
@dataclass
class TaperSection:
    """Jedna sekce zúženého prvku, měřeno od středu ven."""
    length: float        # délka sekce [m] (na jedné polovině)
    radius: float        # poloměr trubky [m]


def taper_element(model: Model, wire: int, sections: Sequence[TaperSection],
                  seg_per_section: int = 0) -> List[int]:
    """Nahradí jeden drát symetrickým zúženým prvkem (Edit → Taper Wire Set).

    Sekce se zadávají od středu ven; celková délka prvku je 2× jejich součet.
    Vrací indexy nově vzniklých drátů.
    """
    if not sections:
        raise ValueError("Zadej aspoň jednu sekci.")
    w = model.wires[wire]
    c = w.center()
    L = w.length
    if L <= 0:
        raise ValueError("Drát má nulovou délku.")
    u = (w.b - w.a) / L
    lam = model.wavelength

    total = sum(s.length for s in sections)
    if total <= 0:
        raise ValueError("Součet délek sekcí musí být kladný.")

    # nové dráty: nejdřív záporná polovina od středu ven, pak kladná
    new: List[Wire] = []
    spans: List[Tuple[float, float]] = []      # rozsah podél u, měřeno od středu
    for sign in (-1.0, +1.0):
        d = 0.0
        for s in sections:
            t0, t1 = sign * d, sign * (d + s.length)
            n = seg_per_section or max(2, int(round(20 * s.length / lam)) + 1)
            n = max(2, min(n, 51))
            a, b = c + u * t0, c + u * t1
            if sign < 0:
                a, b = b, a                     # vždy od vnějšku ke středu / dál ven
                t0, t1 = t1, t0
            new.append(Wire(*a, *b, s.radius, n))
            spans.append((t0, t1))
            d += s.length

    def remap_point(pos: float) -> Tuple[int, float]:
        """Bod na původním drátu -> (index v `new`, relativní poloha)."""
        t = (pos - 0.5) * L                     # vzdálenost od středu podél u
        for k, (t0, t1) in enumerate(spans):
            lo, hi = min(t0, t1), max(t0, t1)
            if lo - 1e-9 <= t <= hi + 1e-9:
                span = t1 - t0
                return k, 0.5 if span == 0 else float(np.clip((t - t0) / span, 0, 1))
        k = 0 if t < 0 else len(new) - 1
        t0, t1 = spans[k]
        span = t1 - t0
        return k, 0.5 if span == 0 else float(np.clip((t - t0) / span, 0, 1))

    shift = len(new) - 1
    for s in model.sources:
        if s.wire == wire:
            k, p = remap_point(s.pos)
            s.wire, s.pos = wire + k, p
        elif s.wire > wire:
            s.wire += shift
    for ld in model.loads:
        if ld.wire == wire:
            k, p = remap_point(ld.pos)
            ld.wire, ld.pos = wire + k, p
        elif ld.wire > wire:
            ld.wire += shift

    model.wires[wire:wire + 1] = new
    return list(range(wire, wire + len(new)))


def element_center_wire(model: Model, idx: Sequence[int]) -> Tuple[int, float]:
    """Najde drát a relativní polohu odpovídající středu zúženého prvku."""
    if not idx:
        raise ValueError("Prázdný prvek.")
    pts = []
    for i in idx:
        w = model.wires[i]
        pts += [w.a, w.b]
    pts = np.array(pts)
    c = pts.mean(axis=0)
    best, bestd, bestpos = idx[0], float("inf"), 0.5
    for i in idx:
        w = model.wires[i]
        for pos, p in ((0.0, w.a), (1.0, w.b)):
            d = float(np.linalg.norm(p - c))
            if d < bestd:
                best, bestd, bestpos = i, d, pos
    return best, bestpos


# --------------------------------------------------------------------------
def polar_to_wire(origin: Sequence[float], length: float,
                  azimuth_deg: float, zenith_deg: float,
                  radius: float = 0.001, nseg: int = 21) -> Wire:
    """Drát zadaný polárně (Edit → Wire definition v MMANA).

    ``azimuth`` se měří od osy X k ose Y, ``zenith`` od osy Z dolů
    (0° = svisle vzhůru, 90° = vodorovně).
    """
    a = np.asarray(origin, dtype=float)
    az, ze = math.radians(azimuth_deg), math.radians(zenith_deg)
    d = np.array([math.sin(ze) * math.cos(az),
                  math.sin(ze) * math.sin(az),
                  math.cos(ze)]) * length
    return Wire(*a, *(a + d), radius, nseg)


def wire_to_polar(w: Wire) -> Tuple[float, float, float]:
    """(délka, azimut [°], zenit [°]) drátu."""
    d = w.b - w.a
    L = float(np.linalg.norm(d))
    if L <= 0:
        return 0.0, 0.0, 90.0
    az = math.degrees(math.atan2(d[1], d[0]))
    ze = math.degrees(math.acos(max(-1.0, min(1.0, d[2] / L))))
    return L, az, ze


# --------------------------------------------------------------------------
@dataclass
class Element:
    """Skupina kolineárních propojených drátů = jeden prvek antény."""
    wires: List[int]
    center: np.ndarray
    direction: np.ndarray
    length: float

    @property
    def position_x(self) -> float:
        return float(self.center[0])

    def radii(self, model: Model) -> List[float]:
        return [model.wires[i].radius for i in self.wires]

    def describe(self, model: Model) -> str:
        rs = sorted({round(model.wires[i].radius * 2000, 1) for i in self.wires})
        if len(rs) == 1:
            return f"Ø {rs[0]:g} mm"
        return "Ø " + "/".join(f"{r:g}" for r in rs) + " mm"


def find_elements(model: Model, tol: float = 1e-6) -> List[Element]:
    """Seskupí dráty do prvků (kolineární a spojené konci)."""
    n = len(model.wires)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    lam = model.wavelength
    eps = max(tol, lam * 1e-7)
    for i in range(n):
        wi = model.wires[i]
        if wi.length <= 0:
            continue
        ui = (wi.b - wi.a) / wi.length
        for j in range(i + 1, n):
            wj = model.wires[j]
            if wj.length <= 0:
                continue
            uj = (wj.b - wj.a) / wj.length
            if abs(abs(float(np.dot(ui, uj))) - 1.0) > 1e-6:
                continue
            touch = any(np.linalg.norm(p - q) < eps
                        for p in (wi.a, wi.b) for q in (wj.a, wj.b))
            if touch:
                union(i, j)

    groups: dict = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    out: List[Element] = []
    for _, idx in sorted(groups.items()):
        pts = np.array([p for i in idx for p in (model.wires[i].a, model.wires[i].b)])
        w0 = model.wires[idx[0]]
        u = (w0.b - w0.a) / max(w0.length, 1e-12)
        t = pts @ u
        length = float(t.max() - t.min())
        center = pts[np.argmin(t)] + u * (length / 2.0)
        out.append(Element(sorted(idx), center, u, length))
    out.sort(key=lambda e: (e.position_x, e.center[1], e.center[2]))
    return out


def set_element_length(model: Model, el: Element, new_length: float) -> None:
    """Změní celkovou délku prvku, poměry sekcí zůstanou."""
    if el.length <= 0 or new_length <= 0:
        return
    f = new_length / el.length
    c = el.center
    for i in el.wires:
        w = model.wires[i]
        _set(w, c + (w.a - c) * f, c + (w.b - c) * f)


def set_element_position(model: Model, el: Element, axis: str, value: float) -> None:
    """Posune celý prvek tak, aby jeho střed ležel na zadané souřadnici."""
    ax = AXES[axis.lower()]
    d = np.zeros(3)
    d[ax] = value - el.center[ax]
    for i in el.wires:
        w = model.wires[i]
        _set(w, w.a + d, w.b + d)
    el.center = el.center + d
