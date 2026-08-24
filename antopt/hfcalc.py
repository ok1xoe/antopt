"""Pomocné VF výpočty — obdoba Tools → HF components v MMANA.

Rezonance, návrh cívky, LC článek, pahýl, transformační vedení, koaxiální
napáječ se ztrátami.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .model import C0
from .solver import swr_from_z


# ==========================================================================
#  koaxiální a dvoulinková vedení
# ==========================================================================
@dataclass
class Line:
    name: str
    z0: float            # vlnová impedance [Ω]
    vf: float            # činitel zkrácení
    k1: float = 0.0      # ztráty: a = k1*sqrt(f) + k2*f  [dB/100 m, f v MHz]
    k2: float = 0.0

    def loss_db_per_100m(self, freq_mhz: float) -> float:
        return self.k1 * math.sqrt(freq_mhz) + self.k2 * freq_mhz

    def wavelength(self, freq_mhz: float) -> float:
        return C0 / (freq_mhz * 1e6) * self.vf


# k1/k2 podle katalogových hodnot útlumu; k2 zahrnuje ztráty v dielektriku
LINES: List[Line] = [
    Line("RG-58 C/U", 50.0, 0.66, 1.32, 0.0175),
    Line("RG-213 U", 50.0, 0.66, 0.63, 0.0074),
    Line("RG-8X", 50.0, 0.78, 0.98, 0.0110),
    Line("RG-174", 50.0, 0.66, 3.10, 0.0200),
    Line("RG-59", 75.0, 0.66, 1.10, 0.0110),
    Line("RG-11", 75.0, 0.66, 0.60, 0.0060),
    Line("H-155", 50.0, 0.79, 0.72, 0.0080),
    Line("H-1000", 50.0, 0.83, 0.42, 0.0043),
    Line("Aircell 7", 50.0, 0.83, 0.60, 0.0055),
    Line("Ecoflex 10", 50.0, 0.85, 0.36, 0.0035),
    Line("LMR-400", 50.0, 0.85, 0.39, 0.0037),
    Line("Dvoulinka 300 Ω", 300.0, 0.82, 0.15, 0.0015),
    Line("Žebříček 450 Ω", 450.0, 0.91, 0.07, 0.0007),
    Line("Otevřené vedení 600 Ω", 600.0, 0.95, 0.05, 0.0004),
]

LINE_BY_NAME = {l.name: l for l in LINES}


# ==========================================================================
#  rezonance a cívky
# ==========================================================================
def reactance_l(l_uh: float, freq_mhz: float) -> float:
    return 2 * math.pi * freq_mhz * 1e6 * l_uh * 1e-6


def reactance_c(c_pf: float, freq_mhz: float) -> float:
    if c_pf <= 0:
        return float("-inf")
    return -1.0 / (2 * math.pi * freq_mhz * 1e6 * c_pf * 1e-12)


def l_for_reactance(x_ohm: float, freq_mhz: float) -> float:
    """Indukčnost [µH] dávající reaktanci +X na daném kmitočtu."""
    return x_ohm / (2 * math.pi * freq_mhz * 1e6) * 1e6


def c_for_reactance(x_ohm: float, freq_mhz: float) -> float:
    """Kapacita [pF] dávající reaktanci −|X| na daném kmitočtu."""
    x = abs(x_ohm)
    if x <= 0:
        return float("inf")
    return 1.0 / (2 * math.pi * freq_mhz * 1e6 * x) * 1e12


def resonant_frequency(l_uh: float, c_pf: float) -> float:
    """Rezonanční kmitočet LC obvodu [MHz]."""
    if l_uh <= 0 or c_pf <= 0:
        return float("nan")
    return 1.0 / (2 * math.pi * math.sqrt(l_uh * 1e-6 * c_pf * 1e-12)) / 1e6


def coil_inductance(diameter_mm: float, length_mm: float, turns: float) -> float:
    """Indukčnost jednovrstvé vzduchové cívky [µH] — Wheelerův vzorec."""
    d, l = diameter_mm / 25.4, length_mm / 25.4        # v palcích
    if d <= 0 or l <= 0 or turns <= 0:
        return float("nan")
    return (d * d * turns * turns) / (18.0 * d + 40.0 * l)


def coil_turns(target_uh: float, diameter_mm: float, length_mm: float) -> float:
    """Kolik závitů je potřeba pro danou indukčnost."""
    d, l = diameter_mm / 25.4, length_mm / 25.4
    if d <= 0 or l <= 0 or target_uh <= 0:
        return float("nan")
    return math.sqrt(target_uh * (18.0 * d + 40.0 * l)) / d


def coil_length_for_turns(turns: float, wire_dia_mm: float,
                          spacing_mm: float = 0.0) -> float:
    return turns * (wire_dia_mm + spacing_mm)


# ==========================================================================
#  LC přizpůsobovací článek
# ==========================================================================
@dataclass
class LcMatch:
    topology: str          # popis zapojení
    x_series: float        # reaktance sériového prvku [Ω]
    x_shunt: float         # reaktance paralelního prvku [Ω]
    series_l_uh: Optional[float]
    series_c_pf: Optional[float]
    shunt_l_uh: Optional[float]
    shunt_c_pf: Optional[float]
    q: float

    def report(self, freq_mhz: float) -> str:
        def part(x, l, c):
            if x > 0:
                return f"cívka {l:.3f} µH  (+{x:.1f} Ω)"
            return f"kondenzátor {c:.1f} pF  ({x:.1f} Ω)"
        return "\n".join([
            f"{self.topology}",
            f"  sériově:    {part(self.x_series, self.series_l_uh, self.series_c_pf)}",
            f"  paralelně:  {part(self.x_shunt, self.shunt_l_uh, self.shunt_c_pf)}",
            f"  provozní Q: {self.q:.2f}",
        ])


def lc_match(z_load: complex, z0: float = 50.0, freq_mhz: float = 14.0
             ) -> List[LcMatch]:
    """Návrh L-článku pro přizpůsobení zátěže na Z₀. Vrací obě řešení."""
    rl, xl = z_load.real, z_load.imag
    if rl <= 0:
        return []
    out: List[LcMatch] = []

    def make(topology, xs, xsh):
        return LcMatch(
            topology=topology, x_series=xs, x_shunt=xsh,
            series_l_uh=l_for_reactance(xs, freq_mhz) if xs > 0 else None,
            series_c_pf=c_for_reactance(xs, freq_mhz) if xs < 0 else None,
            shunt_l_uh=l_for_reactance(xsh, freq_mhz) if xsh > 0 else None,
            shunt_c_pf=c_for_reactance(xsh, freq_mhz) if xsh < 0 else None,
            q=math.sqrt(max(abs(max(rl, z0) / min(rl, z0) - 1.0), 0.0)),
        )

    if rl < z0:
        # paralelní prvek u generátoru, sériový u zátěže
        q = math.sqrt(z0 / rl - 1.0)
        for sgn in (+1.0, -1.0):
            xs = sgn * q * rl - xl
            xsh = -sgn * z0 / q
            out.append(make("paralelní prvek na straně 50 Ω, sériový k anténě",
                            xs, xsh))
    else:
        # paralelní prvek u zátěže
        # nejdřív sériově vykompenzuj X, pak transformuj R
        q = math.sqrt(max(rl / z0 - 1.0, 0.0))
        if q == 0:
            return out
        for sgn in (+1.0, -1.0):
            xsh = sgn * rl * (1.0 + (xl / rl) ** 2) / (q + xl / rl) if (q + xl / rl) != 0 else sgn * rl / q
            # paralelní prvek přes zátěž, pak sériový dorovná zbytek
            y = 1.0 / z_load + 1.0 / (1j * xsh)
            z_after = 1.0 / y
            xs = -z_after.imag
            if abs(z_after.real - z0) / z0 < 0.05:
                out.append(make("paralelní prvek na straně antény, sériový k 50 Ω",
                                xs, xsh))
    return out


# ==========================================================================
#  vedení: transformace, pahýly
# ==========================================================================
def transform_along_line(z_load: complex, z0_line: float, length_m: float,
                         freq_mhz: float, vf: float = 1.0,
                         loss_db_100m: float = 0.0) -> complex:
    """Impedance na vstupu vedení délky ``length_m`` zakončeného ``z_load``."""
    lam = C0 / (freq_mhz * 1e6) * vf
    beta = 2 * math.pi / lam
    alpha = (loss_db_100m / 100.0) / 8.685889638  # Np/m
    gl = complex(alpha, beta) * length_m
    t = np.tanh(gl)
    if np.isinf(abs(z_load)):
        return z0_line / t
    return z0_line * (z_load + z0_line * t) / (z0_line + z_load * t)


def stub_reactance(z0_line: float, length_m: float, freq_mhz: float,
                   vf: float = 1.0, shorted: bool = True) -> float:
    """Reaktance pahýlu (zkratovaného nebo otevřeného)."""
    lam = C0 / (freq_mhz * 1e6) * vf
    b = 2 * math.pi * length_m / lam
    t = math.tan(b)
    if shorted:
        return z0_line * t
    return -z0_line / t if t != 0 else float("inf")


def stub_length_for_reactance(x: float, z0_line: float, freq_mhz: float,
                              vf: float = 1.0, shorted: bool = True) -> float:
    """Nejkratší kladná délka pahýlu dávající zadanou reaktanci [m]."""
    lam = C0 / (freq_mhz * 1e6) * vf
    if shorted:
        ang = math.atan2(x, z0_line)
    else:
        ang = math.atan2(-z0_line, x)
    if ang < 0:
        ang += math.pi
    return lam * ang / (2 * math.pi)


@dataclass
class StubSolution:
    distance_m: float        # vzdálenost pahýlu od antény
    stub_len_m: float
    shorted: bool
    z_at_point: complex

    def report(self, vf: float) -> str:
        kind = "zkratovaný" if self.shorted else "otevřený"
        return (f"  pahýl {kind}: vzdálenost od napájecího bodu "
                f"{self.distance_m * 1000:.0f} mm, délka {self.stub_len_m * 1000:.0f} mm "
                f"(vf {vf:g})")


def single_stub_match(z_load: complex, z0: float = 50.0, freq_mhz: float = 14.0,
                      vf_main: float = 0.66, vf_stub: float = 0.66,
                      z0_stub: Optional[float] = None,
                      shorted: bool = True, n: int = 4000
                      ) -> List[StubSolution]:
    """Přizpůsobení jedním pahýlem — najde vzdálenost i délku pahýlu."""
    z0_stub = z0_stub or z0
    lam = C0 / (freq_mhz * 1e6) * vf_main
    out: List[StubSolution] = []
    ds = np.linspace(0, lam / 2, n)
    prev = None
    for d in ds:
        z = transform_along_line(z_load, z0, float(d), freq_mhz, vf_main)
        y = 1.0 / z
        g = y.real * z0
        f = g - 1.0
        if prev is not None and prev[1] * f <= 0 and abs(f) < 1.0:
            d0 = prev[0] + (0.0 - prev[1]) * (d - prev[0]) / (f - prev[1]) if f != prev[1] else d
            z0p = transform_along_line(z_load, z0, float(d0), freq_mhz, vf_main)
            b = (1.0 / z0p).imag
            x_needed = -1.0 / b if b != 0 else float("inf")
            if math.isfinite(x_needed):
                ls = stub_length_for_reactance(x_needed, z0_stub, freq_mhz,
                                               vf_stub, shorted)
                out.append(StubSolution(float(d0), ls, shorted, z0p))
        prev = (float(d), f)
        if len(out) >= 2:
            break
    return out


def series_section_match(z_load: complex, z0: float = 50.0,
                         z0_section: float = 75.0, freq_mhz: float = 14.0,
                         vf_main: float = 0.66, vf_section: float = 0.66,
                         n: int = 3000) -> List[Tuple[float, float, complex]]:
    """Transformace vloženou sekcí jiné impedance (Line match v MMANA).

    Vrací [(vzdálenost od antény [m], délka sekce [m], výsledná Z)].
    """
    lam1 = C0 / (freq_mhz * 1e6) * vf_main
    lam2 = C0 / (freq_mhz * 1e6) * vf_section
    best: List[Tuple[float, float, complex, float]] = []
    for d in np.linspace(0, lam1 / 2, 120):
        z1 = transform_along_line(z_load, z0, float(d), freq_mhz, vf_main)
        for ls in np.linspace(0, lam2 / 2, 120):
            z2 = transform_along_line(z1, z0_section, float(ls), freq_mhz, vf_section)
            s = swr_from_z(z2, z0)
            best.append((float(d), float(ls), z2, s))
    best.sort(key=lambda t: t[3])
    out = []
    for d, ls, z2, s in best:
        if s > 1.05:
            break
        if any(abs(d - o[0]) < lam1 * 0.02 for o in out):
            continue
        out.append((d, ls, z2))
        if len(out) >= 2:
            break
    if not out and best:
        d, ls, z2, s = best[0]
        out.append((d, ls, z2))
    return out


def coax_feed(z_load: complex, line: Line, length_m: float, freq_mhz: float,
              z0_ref: float = 50.0):
    """Co uvidí vysílač na konci napáječe: (Z, PSV u antény, PSV u TX, ztráta dB)."""
    a100 = line.loss_db_per_100m(freq_mhz)
    z_in = transform_along_line(z_load, line.z0, length_m, freq_mhz,
                                line.vf, a100)
    swr_ant = swr_from_z(z_load, line.z0)
    matched_loss = a100 * length_m / 100.0
    g = abs((z_load - line.z0) / (z_load + line.z0))
    g = min(g, 0.999)
    extra = 0.0
    if g > 0:
        a = 10 ** (matched_loss / 10.0)
        num = a * a - g * g
        den = a * (1 - g * g)
        if num > 0 and den > 0:
            extra = 10 * math.log10(num / den) - matched_loss
    total = matched_loss + max(extra, 0.0)
    return z_in, swr_ant, swr_from_z(z_in, z0_ref), total
