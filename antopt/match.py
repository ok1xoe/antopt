"""Návrh přizpůsobení napájecího bodu.

Yagi optimalizovaná na zisk má typicky vstupní odpor 15–30 Ω. Na 50 Ω se
nejčastěji přizpůsobuje **vlásenkou (hairpin / beta match)**: zářič se zkrátí
pod rezonanci, takže je kapacitní, a paralelně k napájecímu bodu se připojí
zkratovaný dvoulinkový pahýl působící jako indukčnost.

Podmínka přizpůsobení sériové kombinace R − jX na Z₀:

    R² + X² = Z₀ · R          (odtud plyne potřebné X zářiče)
    X_vlásenky = Z₀ · R / |X|  (paralelní indukční reaktance)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .model import Model, C0
from .solver import solve, swr_from_z


# --------------------------------------------------------------------------
def required_reactance(r: float, z0: float = 50.0) -> Optional[float]:
    """Jak kapacitní musí být zářič, aby šel přizpůsobit vlásenkou na Z₀."""
    disc = z0 * r - r * r
    if disc <= 0:
        return None                      # R už je >= Z0, vlásenka nepomůže
    return -math.sqrt(disc)              # záporné = kapacitní


def hairpin_reactance(z: complex, z0: float = 50.0) -> Optional[float]:
    """Potřebná indukční reaktance vlásenky pro danou vstupní impedanci."""
    r, x = z.real, z.imag
    if x >= 0 or r >= z0:
        return None
    return -(r * r + x * x) / x


def line_impedance(spacing_mm: float, diameter_mm: float) -> float:
    """Vlnová impedance dvoulinky (vzduch)."""
    ratio = spacing_mm / diameter_mm
    if ratio <= 1.0:
        return float("nan")
    return 119.9 * math.acosh(ratio)


def hairpin_length(x_l: float, z_h: float, freq_mhz: float) -> float:
    """Délka zkratovaného pahýlu [m] pro danou indukční reaktanci."""
    lam = C0 / (freq_mhz * 1e6)
    return lam / (2 * math.pi) * math.atan(x_l / z_h)


def matched_impedance(z_ant: complex, z_h: float, length_m: float,
                      freq_mhz: float) -> complex:
    """Impedance po připojení vlásenky (paralelně k napájecímu bodu)."""
    lam = C0 / (freq_mhz * 1e6)
    x_l = z_h * math.tan(2 * math.pi * length_m / lam)
    if abs(x_l) < 1e-9:
        return z_ant
    y = 1.0 / z_ant + 1.0 / (1j * x_l)
    return 1.0 / y


@dataclass
class Hairpin:
    freq_mhz: float
    z_ant: complex
    x_l: float
    z_line: float
    length_m: float
    spacing_mm: float
    diameter_mm: float

    def report(self) -> str:
        return (
            f"Vlásenka (hairpin) pro {self.freq_mhz:.3f} MHz\n"
            f"  impedance zářiče     {self.z_ant.real:.1f} {self.z_ant.imag:+.1f} j Ω\n"
            f"  potřebná reaktance   +{self.x_l:.1f} Ω\n"
            f"  vodiče Ø {self.diameter_mm:.0f} mm, rozteč {self.spacing_mm:.0f} mm "
            f"→ Z₀ vedení {self.z_line:.0f} Ω\n"
            f"  délka vlásenky       {self.length_m * 1000:.0f} mm "
            f"(zkratovaná na konci)\n"
            f"  zářič musí být dělený a izolovaný od ráhna; za vlásenku patří balun 1:1."
        )


def design_hairpin(z_ant: complex, freq_mhz: float, z0: float = 50.0,
                   spacing_mm: float = 60.0, diameter_mm: float = 10.0
                   ) -> Optional[Hairpin]:
    x_l = hairpin_reactance(z_ant, z0)
    if x_l is None:
        return None
    z_h = line_impedance(spacing_mm, diameter_mm)
    length = hairpin_length(x_l, z_h, freq_mhz)
    return Hairpin(freq_mhz, z_ant, x_l, z_h, length, spacing_mm, diameter_mm)


# --------------------------------------------------------------------------
def tune_driven_for_hairpin(model: Model, wire: int, z0: float = 50.0,
                            span: float = 0.12, tol: float = 1e-4
                            ) -> Tuple[Model, complex]:
    """Zkrátí zářič tak, aby platilo R² + X² = Z₀·R (vlásenka pak sedí přesně).

    Vrací nový model a jeho vstupní impedanci.
    """
    from .optimize import Parameter, apply_param, read_param

    base = model.copy()
    p = Parameter("delka", [wire], 0.0, 1.0)
    l0 = read_param(base, p)

    def residual(scale: float) -> float:
        m = base.copy()
        apply_param(m, p, l0 * scale)
        z = solve(m).zin
        return z.real ** 2 + z.imag ** 2 - z0 * z.real

    # Reziduum je záporné jen v úzkém okolí rezonance a kladné na obě strany.
    # Vlásenka potřebuje KAPACITNÍ zářič, takže hledáme kořen směrem ke kratšímu.
    hi, f_hi = 1.0, residual(1.0)
    if f_hi > 0:
        # zářič už je dost kapacitní (nebo induktivní) – posuň se k rezonanci
        hi = 1.0
        for s in np.arange(1.0, 1.0 + span + 1e-9, 0.005):
            f = residual(float(s))
            if f < 0:
                hi, f_hi = float(s), f
                break
        else:
            raise ValueError("Zářič není v okolí rezonance – vlásenku nelze navrhnout.")
    lo, f_lo = hi, f_hi
    for s in np.arange(hi - 0.004, hi - span - 1e-9, -0.004):
        f = residual(float(s))
        if f > 0:
            lo, f_lo = float(s), f
            break
    else:
        raise ValueError(f"Zářič nelze naladit na podmínku vlásenky "
                         f"v rozsahu −{span * 100:.0f} % délky.")
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f_mid = residual(mid)
        if abs(f_mid) < tol:
            break
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    out = base.copy()
    apply_param(out, p, l0 * mid)
    return out, solve(out).zin


def swr_with_hairpin(model: Model, hp: Hairpin, freqs_mhz, z0: float = 50.0):
    """PSV na 50 Ω po připojení vlásenky, přes zadané kmitočty."""
    work = model.copy()
    out = []
    for f in np.atleast_1d(freqs_mhz):
        work.freq_mhz = float(f)
        z_ant = solve(work).zin
        z = matched_impedance(z_ant, hp.z_line, hp.length_m, float(f))
        out.append((float(f), z, swr_from_z(z, z0)))
    return out
