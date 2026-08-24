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


def analyse(model: Model, n_th: int = 91, n_ph: int = 181,
            keep_solution: bool = True) -> Result:
    sol = solve(model)
    p = performance(sol, n_th=n_th, n_ph=n_ph)
    return Result(
        freq_mhz=model.freq_mhz,
        zin=sol.zin,
        swr=swr_from_z(sol.zin, model.z0),
        gain_dbi=p.gain_dbi,
        fb_db=p.fb_db,
        fs_db=p.fs_db,
        elevation_deg=p.elevation_deg,
        azimuth_deg=p.max_phi_deg,
        beam_h_deg=p.beam_h_deg,
        beam_v_deg=p.beam_v_deg,
        efficiency=p.efficiency,
        solution=sol if keep_solution else None,
        perf=p if keep_solution else None,
    )


def sweep(model: Model, f_start: float, f_stop: float, n: int = 21,
          full: bool = False, progress: Optional[Callable[[int, int], bool]] = None
          ) -> List[Result]:
    """Rozmítání kmitočtu. ``full=True`` počítá i zisk a F/B (pomalejší).

    ``progress(i, n)`` může vrátit False pro přerušení.
    """
    freqs = np.linspace(f_start, f_stop, n)
    out: List[Result] = []
    work = model.copy()
    for i, f in enumerate(freqs):
        work.freq_mhz = float(f)
        if full:
            r = analyse(work, n_th=46, n_ph=91, keep_solution=False)
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
