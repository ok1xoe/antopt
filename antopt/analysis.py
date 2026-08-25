"""Vyšší vrstva nad solverem: kmitočtové rozmítání a souhrnné parametry."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np

from .model import Model
from .mesh import build_mesh
from .solver import solve, swr_from_z, Solution
from .farfield import performance, Performance


@dataclass
class Result:
    freq_mhz: float
    zin: complex
    swr: float
    gain_dbi: float
    fb_db: float
    fs_db: float
    elevation_deg: float
    azimuth_deg: float
    beam_h_deg: float
    beam_v_deg: float
    efficiency: float
    solution: Optional[Solution] = None
    perf: Optional[Performance] = None
    fb_mmana_db: float = float("nan")
    engine: str = "vlastní"
    backend: Optional[dict] = None


def analyse(model: Model, n_th: int = 91, n_ph: int = 181,
            keep_solution: bool = True, fb_sector_deg: float = 0.0,
            engine: Optional[str] = None) -> Result:
    """Spočítá model zvoleným jádrem (None = výchozí, viz ``engines``)."""
    from . import engines
    return engines.get(engine).analyse(model, n_th=n_th, n_ph=n_ph,
                                       keep_solution=keep_solution,
                                       fb_sector_deg=fb_sector_deg)


def _analyse_own(model: Model, n_th: int = 91, n_ph: int = 181,
                 keep_solution: bool = True, fb_sector_deg: float = 0.0) -> Result:
    sol = solve(model)
    p = performance(sol, n_th=n_th, n_ph=n_ph, fb_sector_deg=fb_sector_deg)
    return Result(
        freq_mhz=model.freq_mhz,
        zin=sol.zin,
        swr=swr_from_z(sol.zin, model.z0),
        gain_dbi=p.gain_dbi,
        fb_db=p.fb_db,
        fs_db=p.fs_db,
        fb_mmana_db=getattr(p, "fb_mmana_db", float("nan")),
        elevation_deg=p.elevation_deg,
        azimuth_deg=p.max_phi_deg,
        beam_h_deg=p.beam_h_deg,
        beam_v_deg=p.beam_v_deg,
        efficiency=p.efficiency,
        solution=sol if keep_solution else None,
        perf=p if keep_solution else None,
        engine="vlastní",
    )


def sweep(model: Model, f_start: float, f_stop: float, n: int = 21,
          full: bool = False, progress: Optional[Callable[[int, int], bool]] = None,
          engine: Optional[str] = None) -> List[Result]:
    """Rozmítání kmitočtu. ``full=True`` počítá i zisk a F/B (pomalejší).

    ``progress(i, n)`` může vrátit False pro přerušení.
    """
    freqs = np.linspace(f_start, f_stop, n)
    out: List[Result] = []
    work = model.copy()
    for i, f in enumerate(freqs):
        work.freq_mhz = float(f)
        if full:
            r = analyse(work, n_th=46, n_ph=91, keep_solution=False, engine=engine)
        elif engine and engine != "vlastní":
            from . import engines
            rr = engines.get(engine).analyse(work, n_th=13, n_ph=25,
                                             keep_solution=False)
            r = Result(freq_mhz=float(f), zin=rr.zin, swr=rr.swr,
                       gain_dbi=float("nan"), fb_db=float("nan"),
                       fs_db=float("nan"), elevation_deg=float("nan"),
                       azimuth_deg=float("nan"), beam_h_deg=float("nan"),
                       beam_v_deg=float("nan"), efficiency=float("nan"),
                       engine=engine)
        else:
            sol = solve(work)
            r = Result(freq_mhz=float(f), zin=sol.zin,
                       swr=swr_from_z(sol.zin, work.z0),
                       gain_dbi=float("nan"), fb_db=float("nan"),
                       fs_db=float("nan"), elevation_deg=float("nan"),
                       azimuth_deg=float("nan"), beam_h_deg=float("nan"),
                       beam_v_deg=float("nan"), efficiency=float("nan"))
        out.append(r)
        if progress is not None and progress(i + 1, len(freqs)) is False:
            break
    return out


def bandwidth(results: Sequence[Result], swr_limit: float = 2.0):
    """Šířka pásma pod daným PSV. Vrací (f_lo, f_hi, šířka v kHz) nebo None."""
    f = np.array([r.freq_mhz for r in results])
    s = np.array([r.swr for r in results])
    below = s <= swr_limit
    if not np.any(below):
        return None
    imin = int(np.argmin(s))
    lo = imin
    while lo > 0 and below[lo - 1]:
        lo -= 1
    hi = imin
    while hi < len(s) - 1 and below[hi + 1]:
        hi += 1

    def interp(i0, i1):
        if i0 == i1:
            return f[i0]
        return float(np.interp(swr_limit, [s[i0], s[i1]], [f[i0], f[i1]]))

    f_lo = interp(lo, lo - 1) if lo > 0 else f[0]
    f_hi = interp(hi, hi + 1) if hi < len(s) - 1 else f[-1]
    return f_lo, f_hi, (f_hi - f_lo) * 1000.0


def _zin(model: Model, engine: Optional[str] = None) -> complex:
    from . import engines
    if engine and engine != "vlastní":
        return engines.get(engine).analyse(model, n_th=7, n_ph=13,
                                           keep_solution=False).zin
    return solve(model).zin


def find_resonance(model: Model, f_lo: Optional[float] = None,
                   f_hi: Optional[float] = None, tol: float = 1e-5,
                   engine: Optional[str] = None):
    """Najde nejbližší kmitočet, kde je reaktance nulová (Plots → Resonance).

    Vrací (f [MHz], Z) nebo None, když v rozsahu rezonance není.
    """
    f0 = model.freq_mhz
    f_lo = f_lo if f_lo else f0 * 0.80
    f_hi = f_hi if f_hi else f0 * 1.20
    work = model.copy()

    def x_at(f: float) -> float:
        work.freq_mhz = f
        return _zin(work, engine).imag

    # hrubé prohledání od středu ven
    fs = np.concatenate([np.linspace(f0, f_hi, 60), np.linspace(f0, f_lo, 60)[1:]])
    prev_f, prev_x = None, None
    brackets = []
    for arm in (np.linspace(f0, f_hi, 60), np.linspace(f0, f_lo, 60)):
        prev_f, prev_x = None, None
        for f in arm:
            x = x_at(float(f))
            if prev_x is not None and prev_x * x <= 0:
                brackets.append((prev_f, prev_x, float(f), x))
                break
            prev_f, prev_x = float(f), x
    if not brackets:
        return None
    brackets.sort(key=lambda b: abs(0.5 * (b[0] + b[2]) - f0))
    a, xa, b, xb = brackets[0]
    for _ in range(60):
        m = 0.5 * (a + b)
        xm = x_at(m)
        if abs(xm) < 1e-6 or (b - a) < tol:
            break
        if xa * xm <= 0:
            b, xb = m, xm
        else:
            a, xa = m, xm
    fr = 0.5 * (a + b)
    work.freq_mhz = fr
    return fr, _zin(work, engine)


def q_factor(model: Model, delta_rel: float = 0.01,
             engine: Optional[str] = None) -> float:
    """Činitel jakosti napájecího bodu podle Yaghjiana–Besta:

        Q = (ω₀ / 2R₀) · |dR/dω + j(dX/dω + |X₀|/ω₀)|

    Bere v úvahu i změnu odporu, takže nezkolabuje tam, kde má reaktance
    inflexi — na rozdíl od prostého (ω/2R)·dX/dω. Pro skutečnou šířku pásma
    použij rozmítání kmitočtu, Q je jen rychlý odhad.
    """
    work = model.copy()
    f0 = model.freq_mhz
    work.freq_mhz = f0 * (1 - delta_rel); z1 = _zin(work, engine)
    work.freq_mhz = f0 * (1 + delta_rel); z2 = _zin(work, engine)
    work.freq_mhz = f0; z0 = _zin(work, engine)
    if z0.real <= 0:
        return float("nan")
    df = 2 * delta_rel * f0
    drdf = (z2.real - z1.real) / df
    dxdf = (z2.imag - z1.imag) / df
    zp = complex(drdf, dxdf + abs(z0.imag) / f0)
    return float(f0 / (2 * z0.real) * abs(zp))


def q_estimate(model: Model, engine: Optional[str] = None):
    """Q se dvěma šířkami kroku. Vrací (Q, spolehlivé?, Q_min, Q_max).

    U antén záměrně zploštělých přes celé pásmo je lokální derivace impedance
    malá a Q silně závisí na kroku — pak je jediné poctivé číslo šířka pásma
    z rozmítání, ne Q.
    """
    qs = []
    for d in (0.005, 0.02):
        try:
            qs.append(q_factor(model, d, engine))
        except Exception:
            pass
    qs = [q for q in qs if np.isfinite(q)]
    if not qs:
        return float("nan"), False, float("nan"), float("nan")
    lo, hi = min(qs), max(qs)
    ok = hi <= 1.5 * max(lo, 1e-9)
    return (0.5 * (lo + hi) if not ok else qs[0]), ok, lo, hi
