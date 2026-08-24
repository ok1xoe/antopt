"""Výpočet dálného pole, zisku, vyzařovacích diagramů a účinnosti."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .model import Model, MU0, C0, ETA0, EPS0
from .solver import Solution


def _phase_integrals(phi: np.ndarray):
    """∫_0^1 e^{jφt}dt a ∫_0^1 t e^{jφt}dt, numericky stabilní i pro φ→0."""
    small = np.abs(phi) < 1e-6
    jphi = np.empty(phi.shape, dtype=complex)
    jphi.real = 0.0
    jphi.imag = phi
    e = np.exp(jphi)
    den = np.where(small, 1.0, jphi)
    E0 = (e - 1.0) / den
    E1 = (e - E0) / den
    if small.any():
        E0[small] = 1.0 + jphi[small] / 2.0
        E1[small] = 0.5 + jphi[small] / 3.0
    return E0, E1


def fresnel(eps_c: complex, sin_psi: np.ndarray):
    """Fresnelovy koeficienty pro odraz od poloprostoru (ψ = elevace)."""
    cos2 = 1.0 - sin_psi ** 2
    sq = np.sqrt(eps_c - cos2)
    rv = (eps_c * sin_psi - sq) / (eps_c * sin_psi + sq)
    rh = (sin_psi - sq) / (sin_psi + sq)
    return rv, rh


def _radiation_vector(sol: Solution, dirs: np.ndarray, mirror: bool = False):
    """Vyzařovací vektor N(r̂) pro zadané směry (Nd,3). Vrací (Nd,3) komplex."""
    mesh = sol.mesh
    k = 2 * math.pi * sol.freq_hz / C0
    A = mesh.a
    U = mesh.u
    alpha = sol.seg_alpha
    beta = sol.seg_beta
    sign = 1.0
    if mirror:
        M = np.array([1.0, 1.0, -1.0])
        A = A * M
        U = U * M
        sign = -1.0                     # obraz báze f je -M(f)

    ru = dirs @ U.T                     # (Nd, Ns)
    phi = k * ru * mesh.length[None, :]
    E0, E1 = _phase_integrals(phi)
    base = np.exp(1j * k * (dirs @ A.T))
    amp = base * (alpha[None, :] * E0 + beta[None, :] * E1) * mesh.length[None, :]
    return sign * (amp @ U)             # (Nd,3)


@dataclass
class Pattern:
    theta: np.ndarray            # (Nd,) [rad] od zenitu
    phi: np.ndarray              # (Nd,) [rad]
    e_theta: np.ndarray          # (Nd,) komplex — svislá polarizace
    e_phi: np.ndarray            # (Nd,) komplex — vodorovná polarizace
    gain_dbi: np.ndarray         # (Nd,) celkový zisk
    power_in: float = 1.0

    @property
    def elevation_deg(self) -> np.ndarray:
        return 90.0 - np.degrees(self.theta)

    def _gain_of(self, e: np.ndarray) -> np.ndarray:
        s = np.abs(e) ** 2 / (2 * ETA0)
        if self.power_in <= 0:
            return np.full(e.shape, -300.0)
        with np.errstate(divide="ignore"):
            return 10 * np.log10(np.maximum(4 * math.pi * s / self.power_in, 1e-30))

    @property
    def gain_v_dbi(self) -> np.ndarray:
        """Zisk svisle polarizované složky (E_theta)."""
        return self._gain_of(self.e_theta)

    @property
    def gain_h_dbi(self) -> np.ndarray:
        """Zisk vodorovně polarizované složky (E_phi)."""
        return self._gain_of(self.e_phi)

    def component(self, which: str) -> np.ndarray:
        which = (which or "total").lower()
        if which.startswith("v"):
            return self.gain_v_dbi
        if which.startswith("h"):
            return self.gain_h_dbi
        return self.gain_dbi


def far_field(sol: Solution, theta: np.ndarray, phi: np.ndarray) -> Pattern:
    """Dálné pole pro pole úhlů theta/phi (v radiánech, stejné délky)."""
    model = sol.model
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    theta, phi = np.broadcast_arrays(theta, phi)
    shape = theta.shape
    th = theta.ravel()
    ph = phi.ravel()

    st, ct = np.sin(th), np.cos(th)
    sp, cp = np.sin(ph), np.cos(ph)
    rhat = np.stack([st * cp, st * sp, ct], axis=1)
    that = np.stack([ct * cp, ct * sp, -st], axis=1)
    phat = np.stack([-sp, cp, np.zeros_like(sp)], axis=1)

    N = _radiation_vector(sol, rhat, mirror=False)
    Nt = np.sum(N * that, axis=1)
    Np_ = np.sum(N * phat, axis=1)

    if model.ground.kind != "free":
        Ni = _radiation_vector(sol, rhat, mirror=True)
        Nti = np.sum(Ni * that, axis=1)
        Npi = np.sum(Ni * phat, axis=1)
        if model.ground.kind == "perfect":
            Nt = Nt + Nti
            Np_ = Np_ + Npi
        else:
            sin_psi = np.cos(th)                      # elevace nad obzorem
            eps_c = model.ground.eps_complex(sol.freq_hz)
            rv, rh = fresnel(eps_c, np.clip(sin_psi, 0.0, 1.0))
            Nt = Nt + rv * Nti
            Np_ = Np_ - rh * Npi
        below = th > math.pi / 2 + 1e-12
        Nt = np.where(below, 0.0, Nt)
        Np_ = np.where(below, 0.0, Np_)

    omega = 2 * math.pi * sol.freq_hz
    fac = omega * MU0 / (4 * math.pi)
    et = -1j * fac * Nt
    ep = -1j * fac * Np_

    s = (np.abs(et) ** 2 + np.abs(ep) ** 2) / (2 * ETA0)
    if sol.power_in > 0:
        g = 4 * math.pi * s / sol.power_in
    else:
        g = np.zeros_like(s)
    with np.errstate(divide="ignore"):
        gdb = 10 * np.log10(np.maximum(g, 1e-30))

    return Pattern(theta=th.reshape(shape), phi=ph.reshape(shape),
                   e_theta=et.reshape(shape), e_phi=ep.reshape(shape),
                   gain_dbi=gdb.reshape(shape), power_in=sol.power_in)


# --------------------------------------------------------------------------
@dataclass
class Performance:
    gain_dbi: float
    gain_dbd: float
    fb_db: float
    fs_db: float
    max_theta_deg: float
    max_phi_deg: float
    elevation_deg: float
    beam_h_deg: float
    beam_v_deg: float
    efficiency: float
    radiated_power: float


def _hemisphere_grid(ground: bool, n_th: int = 91, n_ph: int = 181):
    th_max = 89.99 if ground else 179.9
    th = np.radians(np.linspace(0.1, th_max, n_th))
    ph = np.radians(np.linspace(0.0, 360.0, n_ph))
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    return TH, PH


def performance(sol: Solution, n_th: int = 91, n_ph: int = 181,
                fb_sector_deg: float = 0.0) -> Performance:
    """Zisk, F/B, směr maxima, šířky svazků, účinnost.

    ``fb_sector_deg`` > 0 počítá F/B jako v MMANA — proti nejsilnějšímu
    laloku v zadaném zadním výseku (MMANA má ve výchozím stavu 120°).
    Nula znamená klasické porovnání s přesně opačným směrem.
    """
    grounded = sol.model.ground.kind != "free"
    TH, PH = _hemisphere_grid(grounded, n_th, n_ph)
    pat = far_field(sol, TH, PH)
    g = pat.gain_dbi

    i = np.unravel_index(np.argmax(g), g.shape)
    gmax = float(g[i])
    th0 = float(TH[i])
    ph0 = float(PH[i])

    # zpětný lalok
    dphi_all = np.abs(((PH - ph0 + math.pi) % (2 * math.pi)) - math.pi)
    if fb_sector_deg and fb_sector_deg > 0:
        half = math.radians(fb_sector_deg) / 2.0
        rear = dphi_all >= (math.pi - half)
        gb = float(np.max(g[rear])) if np.any(rear) else float("-inf")
    else:
        ph_back = (ph0 + math.pi) % (2 * math.pi)
        gb = float(far_field(sol, np.array([th0]), np.array([ph_back])).gain_dbi[0])
    fb = gmax - gb

    # nejhorší (největší) postranní/zadní lalok mimo ±60° od maxima
    mask = dphi_all > math.radians(60.0)
    fs = gmax - float(np.max(g[mask])) if np.any(mask) else float("nan")

    # šířky svazků v -3 dB
    beam_h = _beamwidth(g[i[0], :], np.degrees(PH[i[0], :]), gmax)
    beam_v = _beamwidth(g[:, i[1]], np.degrees(TH[:, i[1]]), gmax)

    # vyzářený výkon integrací
    sin_th = np.sin(TH)
    lin = 10 ** (g / 10.0)
    dth = TH[1, 0] - TH[0, 0] if TH.shape[0] > 1 else 0.0
    dph = PH[0, 1] - PH[0, 0] if PH.shape[1] > 1 else 0.0
    integral = float(np.sum(lin * sin_th) * dth * dph)
    eff = integral / (4 * math.pi)

    return Performance(
        gain_dbi=gmax,
        gain_dbd=gmax - 2.15,
        fb_db=fb,
        fs_db=fs,
        max_theta_deg=math.degrees(th0),
        max_phi_deg=math.degrees(ph0),
        elevation_deg=90.0 - math.degrees(th0),
        beam_h_deg=beam_h,
        beam_v_deg=beam_v,
        efficiency=eff,
        radiated_power=eff * sol.power_in,
    )


def _beamwidth(cut: np.ndarray, angles_deg: np.ndarray, gmax: float) -> float:
    above = cut >= gmax - 3.0
    if not np.any(above):
        return float("nan")
    idx = np.where(above)[0]
    # spojitý úsek kolem maxima
    imax = int(np.argmax(cut))
    lo = imax
    while lo - 1 >= 0 and above[lo - 1]:
        lo -= 1
    hi = imax
    while hi + 1 < len(cut) and above[hi + 1]:
        hi += 1
    return float(abs(angles_deg[hi] - angles_deg[lo]))


def cut_azimuth(sol: Solution, elevation_deg: float, n: int = 361):
    """Vodorovný řez. Vrací (azimut [°], Pattern) — MMANA styl, plných 360°."""
    ph = np.radians(np.linspace(0, 360, n))
    th = np.full(n, math.radians(np.clip(90.0 - elevation_deg, 0.05, 179.95)))
    return np.degrees(ph), far_field(sol, th, ph)


def cut_vertical(sol: Solution, azimuth_deg: float, n: int = 361):
    """Řez svislou rovinou procházející zadaným azimutem — jako v MMANA.

    Se zemí se vrací úhly 0…180° (0 = obzor vpředu, 90 = zenit,
    180 = obzor vzadu), ve volném prostoru 0…360°.
    """
    grounded = sol.model.ground.kind != "free"
    span = 180.0 if grounded else 360.0
    ang = np.linspace(0.0, span, n)
    ar = np.radians(ang)
    # směr: vodorovná složka cos(a) podél azimutu, svislá sin(a)
    horiz = np.cos(ar)
    vert = np.sin(ar)
    th = np.arccos(np.clip(vert, -1.0, 1.0))
    th = np.clip(th, math.radians(0.05), math.radians(179.95))
    ph = np.where(horiz >= 0, math.radians(azimuth_deg),
                  math.radians(azimuth_deg + 180.0))
    return ang, far_field(sol, th, ph)


def azimuth_cut(sol: Solution, elevation_deg: float, n: int = 361) -> Tuple[np.ndarray, np.ndarray]:
    ph = np.radians(np.linspace(0, 360, n))
    th = np.full(n, math.radians(90.0 - elevation_deg))
    pat = far_field(sol, th, ph)
    return np.degrees(ph), pat.gain_dbi


def elevation_cut(sol: Solution, azimuth_deg: float, n: int = 361) -> Tuple[np.ndarray, np.ndarray]:
    """Řez svislou rovinou. Vrací (elevace [°], zisk [dBi]).

    Se zemí 0..90°, ve volném prostoru 0..360° (nad 180° jde o opačný azimut).
    """
    grounded = sol.model.ground.kind != "free"
    if grounded:
        el = np.linspace(0.0, 90.0, n)
        th = np.radians(np.clip(90.0 - el, 0.05, 89.99))
        ph = np.full(n, math.radians(azimuth_deg))
    else:
        el = np.linspace(0.0, 360.0, n)
        er = np.radians(el)
        th = np.arccos(np.clip(np.sin(er), -1.0, 1.0))
        ph = np.where(np.cos(er) >= 0,
                      math.radians(azimuth_deg),
                      math.radians(azimuth_deg + 180.0))
    pat = far_field(sol, th, ph)
    return el, pat.gain_dbi
