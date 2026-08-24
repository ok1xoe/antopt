"""Průvodci tvorbou antén — obdoba „Make new antenna“ v MMANA.

Každý průvodce vrátí hotový model, který se dá rovnou počítat a optimalizovat.
Rozměry vycházejí z osvědčených poměrů; přesné doladění je pak na optimalizátoru.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .model import Model, Wire, Source, Load, Ground, C0


def _lam(f_mhz: float) -> float:
    return C0 / (f_mhz * 1e6)


def _finish(m: Model, ground: str, material: str) -> Model:
    m.ground = Ground.from_name(ground)
    m.material = material
    m.auto_segment()
    return m


# --------------------------------------------------------------------------
def dipole(freq_mhz: float = 14.1, height: float = 12.0,
           radius_mm: float = 1.0, ground: str = "průměrná",
           material: str = "měď", length_factor: float = 0.478) -> Model:
    lam = _lam(freq_mhz)
    L = length_factor * lam
    m = Model(name=f"Dipól {freq_mhz:g} MHz", freq_mhz=freq_mhz)
    m.wires = [Wire(-L / 2, 0, height, L / 2, 0, height, radius_mm / 2000.0, 21)]
    m.sources = [Source(0, 0.5, 1.0)]
    return _finish(m, ground, material)


def inverted_v(freq_mhz: float = 7.05, apex: float = 12.0,
               droop_deg: float = 45.0, radius_mm: float = 1.5,
               ground: str = "průměrná", material: str = "měď",
               length_factor: float = 0.480) -> Model:
    lam = _lam(freq_mhz)
    arm = length_factor * lam / 2.0
    t = math.radians(droop_deg)
    dy, dz = arm * math.cos(t), arm * math.sin(t)
    m = Model(name=f"Inverted V {freq_mhz:g} MHz", freq_mhz=freq_mhz)
    m.wires = [Wire(0, 0, apex, 0, -dy, apex - dz, radius_mm / 2000.0, 21),
               Wire(0, 0, apex, 0, dy, apex - dz, radius_mm / 2000.0, 21)]
    m.sources = [Source(0, 0.0, 1.0)]
    return _finish(m, ground, material)


def vertical(freq_mhz: float = 7.05, radius_mm: float = 15.0,
             radials: int = 4, elevated: float = 0.0,
             ground: str = "průměrná", material: str = "hliník",
             ground_loss: float = 12.0, length_factor: float = 0.242) -> Model:
    lam = _lam(freq_mhz)
    h = length_factor * lam
    m = Model(name=f"Vertikál {freq_mhz:g} MHz", freq_mhz=freq_mhz)
    z0 = max(elevated, 0.0)
    m.wires = [Wire(0, 0, z0, 0, 0, z0 + h, radius_mm / 2000.0, 21)]
    if elevated > 0 and radials > 0:
        rl = 0.25 * lam
        for k in range(radials):
            a = 2 * math.pi * k / radials
            m.wires.append(Wire(0, 0, z0, rl * math.cos(a), rl * math.sin(a), z0,
                                0.001, 15))
    m.sources = [Source(0, 0.0, 1.0)]
    m.ground_loss_r = 0.0 if elevated > 0 else ground_loss
    return _finish(m, ground, material)


def yagi(freq_mhz: float = 14.15, elements: int = 3, boom: Optional[float] = None,
         height: float = 12.0, radius_mm: float = 12.0,
         ground: str = "průměrná", material: str = "hliník") -> Model:
    """Yagi s rozumným výchozím rozložením — pak nech doladit optimalizátorem."""
    n = max(2, int(elements))
    lam = _lam(freq_mhz)
    # rozteče v λ, osvědčené poměry pro dané počty prvků
    if boom is None:
        boom = {2: 0.13, 3: 0.30, 4: 0.45, 5: 0.60, 6: 0.72,
                7: 0.90, 8: 1.05}.get(n, 0.15 * n) * lam
    spacings = _yagi_spacings(n, boom / lam)
    lens = _yagi_lengths(n)
    m = Model(name=f"Yagi {n} prvků {freq_mhz:g} MHz", freq_mhz=freq_mhz)
    xs = np.cumsum([0.0] + spacings) * lam
    xs = xs - xs[1]                       # zářič do počátku
    a = radius_mm / 2000.0
    for x, Lf in zip(xs, lens):
        L = Lf * lam
        m.wires.append(Wire(float(x), -L / 2, height, float(x), L / 2, height, a, 17))
    m.sources = [Source(1, 0.5, 1.0)]
    return _finish(m, ground, material)


def _yagi_spacings(n: int, boom_lam: float) -> List[float]:
    """Rozteče mezi sousedními prvky v λ (reflektor→zářič→direktory)."""
    if n == 2:
        return [boom_lam]
    base = [0.14, 0.11] + [0.16 + 0.02 * i for i in range(n - 3)]
    base = base[:n - 1]
    s = sum(base)
    return [b * boom_lam / s for b in base]


def _yagi_lengths(n: int) -> List[float]:
    """Délky prvků v λ."""
    out = [0.495, 0.472]                       # reflektor, zářič
    for i in range(n - 2):
        out.append(0.462 - 0.006 * i)
    return out[:n]


def quad(freq_mhz: float = 14.1, elements: int = 2, height: float = 12.0,
         radius_mm: float = 1.5, ground: str = "průměrná",
         material: str = "měď") -> Model:
    """Čtvercová smyčka (cubical quad), buzená ve spodní straně."""
    lam = _lam(freq_mhz)
    n = max(1, int(elements))
    m = Model(name=f"Quad {n} prvků {freq_mhz:g} MHz", freq_mhz=freq_mhz)
    a = radius_mm / 2000.0
    per = [1.065, 1.030, 1.010, 1.000][:n] + [0.995] * max(0, n - 4)
    xs = [0.0] + [0.15 * lam * (i + 1) for i in range(n - 1)]
    if n > 1:
        xs = [-0.17 * lam] + [0.0] + [0.15 * lam * (i + 1) for i in range(n - 2)]
        per = [1.090, 1.052] + [1.022 - 0.006 * i for i in range(n - 2)]
    for x, pf in zip(xs, per):
        side = pf * lam / 4.0
        h = side / 2.0
        z0, z1 = height - h, height + h
        m.wires += [
            Wire(x, -h, z0, x, h, z0, a, 13),
            Wire(x, h, z0, x, h, z1, a, 13),
            Wire(x, h, z1, x, -h, z1, a, 13),
            Wire(x, -h, z1, x, -h, z0, a, 13),
        ]
    m.sources = [Source(0 if n == 1 else 4, 0.5, 1.0)]
    return _finish(m, ground, material)


def delta_loop(freq_mhz: float = 14.1, apex: float = 14.0,
               radius_mm: float = 1.5, ground: str = "průměrná",
               material: str = "měď", feed: str = "spodní střed") -> Model:
    lam = _lam(freq_mhz)
    side = 1.077 * lam / 3.0
    h = side * math.sqrt(3) / 2.0
    base_z = apex - h
    m = Model(name=f"Delta loop {freq_mhz:g} MHz", freq_mhz=freq_mhz)
    a = radius_mm / 2000.0
    m.wires = [
        Wire(0, -side / 2, base_z, 0, 0, apex, a, 17),
        Wire(0, 0, apex, 0, side / 2, base_z, a, 17),
        Wire(0, side / 2, base_z, 0, -side / 2, base_z, a, 17),
    ]
    m.sources = [Source(2, 0.5, 1.0)]
    return _finish(m, ground, material)


def phased_array(freq_mhz: float = 7.05, count: int = 2,
                 spacing_lam: float = 0.25, phase_step: float = -90.0,
                 height: float = 15.0, radius_mm: float = 1.0,
                 ground: str = "průměrná", material: str = "měď") -> Model:
    """Fázované pole svislých dipólů/vertikálů — klasický kardioidní systém."""
    lam = _lam(freq_mhz)
    n = max(2, int(count))
    d = spacing_lam * lam
    L = 0.478 * lam
    m = Model(name=f"Fázované pole {n}× {freq_mhz:g} MHz", freq_mhz=freq_mhz)
    a = radius_mm / 2000.0
    for k in range(n):
        x = k * d
        m.wires.append(Wire(x, 0, height - L / 2, x, 0, height + L / 2, a, 21))
        m.sources.append(Source(k, 0.5, 1.0, k * phase_step))
    return _finish(m, ground, material)


def long_wire(freq_mhz: float = 7.05, waves: float = 2.0, height: float = 12.0,
              radius_mm: float = 1.0, ground: str = "průměrná",
              material: str = "měď") -> Model:
    lam = _lam(freq_mhz)
    L = waves * lam
    m = Model(name=f"Long wire {waves:g} λ {freq_mhz:g} MHz", freq_mhz=freq_mhz)
    m.wires = [Wire(0, 0, height, L, 0, height, radius_mm / 2000.0, 41)]
    m.sources = [Source(0, 0.0, 1.0)]
    return _finish(m, ground, material)


# --------------------------------------------------------------------------
@dataclass
class WizardField:
    key: str
    label: str
    default: float | str
    kind: str = "f"          # 'f' číslo, 'i' celé, 'ground', 'material', 'text'


@dataclass
class Wizard:
    name: str
    fn: Callable[..., Model]
    fields: List[WizardField]


_COMMON = [WizardField("ground", "Zem", "průměrná", "ground"),
           WizardField("material", "Materiál", "měď", "material")]

WIZARDS: List[Wizard] = [
    Wizard("Dipól", dipole, [
        WizardField("freq_mhz", "Kmitočet [MHz]", 14.1),
        WizardField("height", "Výška [m]", 12.0),
        WizardField("radius_mm", "Průměr vodiče [mm]", 2.0),
        WizardField("length_factor", "Délka [λ]", 0.478)] + _COMMON),
    Wizard("Inverted V", inverted_v, [
        WizardField("freq_mhz", "Kmitočet [MHz]", 7.05),
        WizardField("apex", "Výška vrcholu [m]", 12.0),
        WizardField("droop_deg", "Sklon ramen [°]", 45.0),
        WizardField("radius_mm", "Průměr vodiče [mm]", 2.0),
        WizardField("length_factor", "Délka [λ]", 0.480)] + _COMMON),
    Wizard("Vertikál", vertical, [
        WizardField("freq_mhz", "Kmitočet [MHz]", 7.05),
        WizardField("radius_mm", "Průměr trubky [mm]", 30.0),
        WizardField("elevated", "Zvednutá základna [m] (0 = na zemi)", 0.0),
        WizardField("radials", "Počet radiálů (jen zvednutý)", 4, "i"),
        WizardField("ground_loss", "Ztráty zemního systému [Ω]", 12.0),
        WizardField("length_factor", "Výška [λ]", 0.242)] + _COMMON),
    Wizard("Yagi", yagi, [
        WizardField("freq_mhz", "Kmitočet [MHz]", 14.15),
        WizardField("elements", "Počet prvků", 3, "i"),
        WizardField("boom", "Délka ráhna [m] (0 = automaticky)", 0.0),
        WizardField("height", "Výška [m]", 12.0),
        WizardField("radius_mm", "Průměr prvků [mm]", 25.0)] + _COMMON),
    Wizard("Quad", quad, [
        WizardField("freq_mhz", "Kmitočet [MHz]", 14.1),
        WizardField("elements", "Počet prvků", 2, "i"),
        WizardField("height", "Výška [m]", 12.0),
        WizardField("radius_mm", "Průměr vodiče [mm]", 3.0)] + _COMMON),
    Wizard("Delta loop", delta_loop, [
        WizardField("freq_mhz", "Kmitočet [MHz]", 14.1),
        WizardField("apex", "Výška vrcholu [m]", 14.0),
        WizardField("radius_mm", "Průměr vodiče [mm]", 3.0)] + _COMMON),
    Wizard("Fázované pole", phased_array, [
        WizardField("freq_mhz", "Kmitočet [MHz]", 7.05),
        WizardField("count", "Počet prvků", 2, "i"),
        WizardField("spacing_lam", "Rozteč [λ]", 0.25),
        WizardField("phase_step", "Fázový krok [°]", -90.0),
        WizardField("height", "Výška středu [m]", 15.0),
        WizardField("radius_mm", "Průměr vodiče [mm]", 2.0)] + _COMMON),
    Wizard("Long wire", long_wire, [
        WizardField("freq_mhz", "Kmitočet [MHz]", 7.05),
        WizardField("waves", "Délka [λ]", 2.0),
        WizardField("height", "Výška [m]", 12.0),
        WizardField("radius_mm", "Průměr vodiče [mm]", 2.0)] + _COMMON),
]

WIZARD_BY_NAME: Dict[str, Wizard] = {w.name: w for w in WIZARDS}
