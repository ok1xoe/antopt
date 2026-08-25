"""Drobné sdílené pomůcky (aby na nich nezávisel jen GUI modul)."""
from __future__ import annotations
import numpy as np


def beamwidth_of(cut: np.ndarray, angles_deg: np.ndarray, gmax: float) -> float:
    """Šířka hlavního laloku v −3 dB podél zadaného řezu."""
    above = cut >= gmax - 3.0
    if not np.any(above):
        return float("nan")
    imax = int(np.argmax(cut))
    lo = imax
    while lo - 1 >= 0 and above[lo - 1]:
        lo -= 1
    hi = imax
    while hi + 1 < len(cut) and above[hi + 1]:
        hi += 1
    return float(abs(angles_deg[hi] - angles_deg[lo]))
