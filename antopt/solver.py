"""Jádro metody momentů (MoM) pro tenké dráty.

Formulace: elektricko-polní integrální rovnice (EFIE) ve smíšeném potenciálovém
tvaru, Galerkinovo testování po částech lineárními (trojúhelníkovými) bázemi,
redukované jádro (přičtený poloměr drátu).

    Z_mn = jωμ/4π ∫∫ f_m·f_n G dl'dl  +  1/(j4πωε) ∫∫ f'_m f'_n G dl'dl
    G(R) = e^{-jkR}/R,  R = sqrt(|r-r'|² + a²)

Singularita se odděluje:  G = (e^{-jkR}-1)/R + 1/R.
První člen je hladký a integruje se Gaussovou kvadraturou, druhý má vnitřní
integrál v uzavřeném tvaru (přes arcsinh), takže self-členy jsou přesné.

Zem se řeší zrcadlením (obrazová teorie).  Obraz báze f je -M(f), kde M zrcadlí
z -> -z; pro dokonalou zem je to přesné, pro reálnou zem se používá jako
aproximace pro impedanci a Fresnelovy koeficienty pro vyzařování.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np

from .model import Model, MU0, EPS0, C0, ETA0
from .mesh import Mesh, build_mesh, node_for_position


# --------------------------------------------------------------------------
# integrály přes dvojice segmentů
# --------------------------------------------------------------------------
@lru_cache(maxsize=16)
def _gauss(n: int):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


def _moment_integrals(oa, ou, oL, sa, su, sL, srad, k,
                      q_out=8, q_in=6, block=24):
    """Momentové integrály mezi dvojicemi segmentů.

    Vrací S00, S10, S01, S11 tvaru (Nm, Nn), kde

        S_ab = L_m L_n ∫_0^1 ∫_0^1 t^a t'^b G(R(t,t')) dt' dt
    """
    Nm = oa.shape[0]
    Nn = sa.shape[0]

    to, wo = _gauss(q_out)
    ti, wi = _gauss(q_in)

    S00 = np.zeros((Nm, Nn), dtype=complex)
    S10 = np.zeros((Nm, Nn), dtype=complex)
    S01 = np.zeros((Nm, Nn), dtype=complex)
    S11 = np.zeros((Nm, Nn), dtype=complex)

    a2 = (srad ** 2)[None, None, :]                      # (1,1,Nn)
    # zdrojové body pro hladkou část: (Nn, q_in, 3)
    Qs = sa[:, None, :] + ti[None, :, None] * (sL[:, None, None] * su[:, None, :])

    for m0 in range(0, Nm, block):
        m1 = min(Nm, m0 + block)
        Bm = m1 - m0
        # pozorovací body (Bm, q_out, 3)
        P = oa[m0:m1, None, :] + to[None, :, None] * (oL[m0:m1, None, None] * ou[m0:m1, None, :])

        # ---------- hladká část  (e^{-jkR}-1)/R ----------
        diff = P[:, :, None, None, :] - Qs[None, None, :, :, :]      # (Bm,qo,Nn,qi,3)
        R2 = np.einsum("mongc,mongc->mong", diff, diff) + a2[..., None]
        R = np.sqrt(R2)
        Gs = (np.exp(-1j * k * R) - 1.0) / R                          # (Bm,qo,Nn,qi)
        del diff, R2, R

        # integrace přes vnitřní proměnnou
        A0 = np.einsum("monj,j->mon", Gs, wi)
        A1 = np.einsum("monj,j->mon", Gs, wi * ti)
        del Gs

        # ---------- singulární část  1/R, vnitřní integrál analyticky ----------
        w = P[:, :, None, :] - sa[None, None, :, :]                   # (Bm,qo,Nn,3)
        s0 = np.sum(w * su[None, None, :, :], axis=-1)                # (Bm,qo,Nn)
        wn2 = np.einsum("monc,monc->mon", w, w)
        p2 = wn2 - s0 ** 2 + a2
        p2 = np.maximum(p2, 1e-30)
        p = np.sqrt(p2)
        Lc = sL[None, None, :]
        R0 = np.sqrt(s0 ** 2 + p2)
        RL = np.sqrt((Lc - s0) ** 2 + p2)
        I0 = np.arcsinh((Lc - s0) / p) - np.arcsinh(-s0 / p)
        I1 = (RL - R0) + s0 * I0
        B0 = I0 / Lc                       # ∫_0^1 dt'/R
        B1 = I1 / (Lc ** 2)                # ∫_0^1 t' dt'/R
        del w, s0, wn2, p2, p, R0, RL, I0, I1

        T0 = A0 + B0
        T1 = A1 + B1
        scale = oL[m0:m1, None] * sL[None, :]
        S00[m0:m1] = np.einsum("mon,o->mn", T0, wo) * scale
        S10[m0:m1] = np.einsum("mon,o->mn", T0, wo * to) * scale
        S01[m0:m1] = np.einsum("mon,o->mn", T1, wo) * scale
        S11[m0:m1] = np.einsum("mon,o->mn", T1, wo * to) * scale

    return S00, S10, S01, S11


def _assemble(mesh: Mesh, S, u_src, sign, omega):
    """Sestaví příspěvek k impedanční matici z momentových integrálů."""
    S00, S10, S01, S11 = S
    Gtt = S11
    Gt1 = S10 - S11
    G1t = S01 - S11
    G11 = S00 - S10 - S01 + S11

    kv = 1j * omega * MU0 / (4 * math.pi)
    ks = 1.0 / (1j * 4 * math.pi * omega * EPS0)

    nb = mesh.nbasis
    Z = np.zeros((nb, nb), dtype=complex)
    div = mesh.divergence_coef()

    for kk in range(2):
        sm = mesh.bs_seg[:, kk]
        em = mesh.bs_end[:, kk]
        cm = mesh.bs_coef[:, kk]
        if not np.any(cm):
            continue
        for ll in range(2):
            sn = mesh.bs_seg[:, ll]
            en = mesh.bs_end[:, ll]
            cn = mesh.bs_coef[:, ll]
            if not np.any(cn):
                continue
            ix = np.ix_(sm, sn)
            mm = (em == 1)[:, None]
            nn = (en == 1)[None, :]
            Gsel = np.where(mm & nn, Gtt[ix],
                    np.where(mm & ~nn, Gt1[ix],
                     np.where(~mm & nn, G1t[ix], G11[ix])))
            D = mesh.u[sm] @ u_src[sn].T
            amp = cm[:, None] * cn[None, :]
            Z += kv * D * amp * Gsel
            Z += ks * (div[:, kk][:, None] * div[:, ll][None, :]) * S00[ix]
    return sign * Z


def _wire_loss(mesh: Mesh, model: Model) -> np.ndarray:
    """Rozložený odpor drátu (povrchová impedance) jako přídavek k Z."""
    sigma = model.conductivity()
    nb = mesh.nbasis
    Z = np.zeros((nb, nb), dtype=complex)
    if sigma <= 0:
        return Z
    f = model.freq_hz
    Rs = math.sqrt(math.pi * f * MU0 / sigma)          # povrchový odpor [Ω/□]
    zp = Rs / (2 * math.pi * mesh.radius)              # Ω/m
    zp = zp * (1 + 1j)                                 # skinefekt: R + jX(=R)
    for kk in range(2):
        sm = mesh.bs_seg[:, kk]
        em = mesh.bs_end[:, kk]
        cm = mesh.bs_coef[:, kk]
        for ll in range(2):
            sn = mesh.bs_seg[:, ll]
            en = mesh.bs_end[:, ll]
            cn = mesh.bs_coef[:, ll]
            same = (sm[:, None] == sn[None, :])
            act = (cm != 0)[:, None] & (cn != 0)[None, :]
            equal_shape = (em[:, None] == en[None, :])
            fac = np.where(equal_shape, 1.0 / 3.0, 1.0 / 6.0)
            contrib = (zp * mesh.length)[sm][:, None] * fac * (cm[:, None] * cn[None, :])
            Z += np.where(same & act, contrib, 0.0)
    return Z


@dataclass
class Solution:
    model: Model
    mesh: Mesh
    currents: np.ndarray          # (Nb,) proudy bází [A]
    seg_alpha: np.ndarray         # (Ns,) proud na začátku segmentu
    seg_beta: np.ndarray          # (Ns,) lineární člen
    zin: complex
    feed_current: complex
    power_in: float
    source_nodes: list
    freq_hz: float

    @property
    def swr(self) -> float:
        return swr_from_z(self.zin, self.model.z0)

    def segment_current(self, n: int = 2) -> np.ndarray:
        """Proud ve `n` bodech na každém segmentu (pro grafy)."""
        t = np.linspace(0, 1, n)
        return self.seg_alpha[:, None] + self.seg_beta[:, None] * t[None, :]


def swr_from_z(z: complex, z0: float = 50.0) -> float:
    if z0 <= 0:
        return float("inf")
    g = abs((z - z0) / (z + z0))
    g = min(g, 0.999999)
    return (1 + g) / (1 - g)


def solve(model: Model, mesh: Optional[Mesh] = None,
          q_out: int = 6, q_in: int = 4, accurate: bool = False) -> Solution:
    """Vyřeší model na jeho kmitočtu a vrátí proudy a vstupní impedanci."""
    if accurate:
        q_out, q_in = 10, 8
    if mesh is None:
        mesh = build_mesh(model)

    f = model.freq_hz
    omega = 2 * math.pi * f
    k = omega / C0

    S = _moment_integrals(mesh.a, mesh.u, mesh.length,
                          mesh.a, mesh.u, mesh.length, mesh.radius,
                          k, q_out=q_out, q_in=q_in)
    Z = _assemble(mesh, S, mesh.u, +1.0, omega)

    if model.ground.kind != "free":
        Mm = np.array([1.0, 1.0, -1.0])
        a_img = mesh.a * Mm
        u_img = mesh.u * Mm
        S_img = _moment_integrals(mesh.a, mesh.u, mesh.length,
                                  a_img, u_img, mesh.length, mesh.radius,
                                  k, q_out=q_out, q_in=q_in)
        Z += _assemble(mesh, S_img, u_img, -1.0, omega)

    Z += _wire_loss(mesh, model)

    # soustředné zátěže
    for ld in model.loads:
        node = node_for_position(mesh, model, ld.wire, ld.pos)
        bi = mesh.basis_at_node(node)
        if bi >= 0:
            Z[bi, bi] += ld.impedance(f)

    # buzení
    V = np.zeros(mesh.nbasis, dtype=complex)
    src_nodes = []
    for src in model.sources:
        node = node_for_position(mesh, model, src.wire, src.pos)
        bi = mesh.basis_at_node(node)
        if bi < 0:
            raise ValueError("Zdroj není na použitelném uzlu.")
        V[bi] += src.phasor()
        src_nodes.append((bi, src.phasor()))
        # ztráty zemního systému u uzemněného napájení
        if (model.ground_loss_r > 0 and model.ground.kind != "free"
                and abs(mesh.points[node, 2]) < 1e-6):
            Z[bi, bi] += model.ground_loss_r

    I = np.linalg.solve(Z, V)

    # proud na segmentech: I(t) = alpha + beta t
    alpha = np.zeros(mesh.nseg, dtype=complex)
    beta = np.zeros(mesh.nseg, dtype=complex)
    for kk in range(2):
        seg = mesh.bs_seg[:, kk]
        end = mesh.bs_end[:, kk]
        coef = mesh.bs_coef[:, kk]
        amp = I * coef
        up = end == 1
        np.add.at(beta, seg[up], amp[up])
        np.add.at(alpha, seg[~up], amp[~up])
        np.add.at(beta, seg[~up], -amp[~up])

    p_in = 0.0
    for bi, v in src_nodes:
        p_in += 0.5 * float(np.real(v * np.conj(I[bi])))
    bi0, v0 = src_nodes[0]
    ifeed = I[bi0]
    zin = v0 / ifeed if ifeed != 0 else complex(float("inf"), 0)

    return Solution(model=model, mesh=mesh, currents=I,
                    seg_alpha=alpha, seg_beta=beta,
                    zin=zin, feed_current=ifeed, power_in=p_in,
                    source_nodes=[b for b, _ in src_nodes], freq_hz=f)
