"""Ověření solveru proti analytickým výsledkům a proti NEC-2 (pokud je PyNEC).

Spuštění:   pytest -q          (nebo  python3 tests/test_validation.py)
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from antopt.model import Model, Wire, Source, Ground, C0
from antopt.solver import solve, swr_from_z
from antopt.farfield import performance
from antopt.analysis import analyse
from antopt.fileio import to_nec, from_nec, to_maa, from_maa
from antopt.examples import EXAMPLES

try:
    from PyNEC import nec_context
    HAVE_NEC = True
except ImportError:
    HAVE_NEC = False

F = 300.0
LAM = C0 / (F * 1e6)


def dipole(L_lam=0.5, a_lam=5e-4, nseg=41, ground="free", h_lam=0.0):
    m = Model(freq_mhz=F, material="dokonalý vodič", ground=Ground(ground))
    L = L_lam * LAM
    h = h_lam * LAM
    m.wires = [Wire(-L / 2, 0, h, L / 2, 0, h, radius=a_lam * LAM, nseg=nseg)]
    m.sources = [Source(0, 0.5, 1.0)]
    return m


# ---------------------------------------------------------------- analytika
def test_halfwave_dipole_gain():
    """Zisk půlvlnného dipólu ve volném prostoru = 2,15 dBi."""
    p = performance(solve(dipole(0.4784)), n_th=181, n_ph=181)
    assert abs(p.gain_dbi - 2.15) < 0.1


def test_dipole_efficiency_integral():
    """Integrál vyzařovacího diagramu musí dát účinnost 1 (bezeztrátový vodič)."""
    p = performance(solve(dipole(0.4784)), n_th=181, n_ph=361)
    assert abs(p.efficiency - 1.0) < 0.02


def test_monopole_is_half_of_dipole():
    """Monopol nad dokonalou zemí má přesně poloviční impedanci než dipól."""
    zd = solve(dipole(0.5, a_lam=1e-4, nseg=61)).zin
    m = Model(freq_mhz=F, material="dokonalý vodič", ground=Ground("perfect"))
    m.wires = [Wire(0, 0, 0.0, 0, 0, 0.25 * LAM, radius=1e-4 * LAM, nseg=31)]
    m.sources = [Source(0, 0.0, 1.0)]
    zm = solve(m).zin
    assert abs(zm - zd / 2) / abs(zd / 2) < 0.02


def test_monopole_gain_3db_over_dipole():
    """Monopol nad dokonalou zemí: 2,15 + 3 = 5,15 dBi."""
    m = Model(freq_mhz=F, material="dokonalý vodič", ground=Ground("perfect"))
    m.wires = [Wire(0, 0, 0.0, 0, 0, 0.239 * LAM, radius=1e-4 * LAM, nseg=31)]
    m.sources = [Source(0, 0.0, 1.0)]
    p = performance(solve(m), n_th=181, n_ph=181)
    assert abs(p.gain_dbi - 5.15) < 0.15


def test_convergence_with_segments():
    """Zjemňování sítě musí konvergovat, ne divergovat."""
    zs = [solve(dipole(0.5, nseg=n)).zin for n in (11, 21, 41, 81)]
    diffs = [abs(zs[i + 1] - zs[i]) for i in range(3)]
    assert diffs[0] > diffs[1] > diffs[2]
    # zbytková změna mezi 41 a 81 segmenty pod 1 % z |Z|
    assert diffs[-1] / abs(zs[-1]) < 0.01


def test_resonant_length():
    """Rezonanční délka tenkého dipólu leží mezi 0,47 a 0,49 λ."""
    lo, hi = 0.44, 0.52
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        x = solve(dipole(mid, nseg=41)).zin.imag
        if x < 0:
            lo = mid
        else:
            hi = mid
    assert 0.470 < mid < 0.490


def test_dipole_over_perfect_ground_gain():
    """Dipól λ/2 nad dokonalou zemí: maximum v 30° elevaci."""
    p = performance(solve(dipole(0.4784, ground="perfect", h_lam=0.5)),
                    n_th=181, n_ph=181)
    assert abs(p.elevation_deg - 30.0) < 1.5
    assert 8.0 < p.gain_dbi < 8.8


def test_reciprocity_matrix_symmetry():
    """Impedanční matice musí být symetrická (reciprocita)."""
    from antopt.mesh import build_mesh
    from antopt.solver import _moment_integrals, _assemble
    m = dipole(0.5, nseg=21)
    mesh = build_mesh(m)
    k = 2 * math.pi * m.freq_hz / C0
    S = _moment_integrals(mesh.a, mesh.u, mesh.length,
                          mesh.a, mesh.u, mesh.length, mesh.radius, k)
    Z = _assemble(mesh, S, mesh.u, 1.0, 2 * math.pi * m.freq_hz)
    assert np.max(np.abs(Z - Z.T)) / np.max(np.abs(Z)) < 1e-10


# ---------------------------------------------------------------- vs NEC-2
def _nec_dipole(L_lam, a_lam, nseg, ground=None, h_lam=0.0):
    ctx = nec_context()
    g = ctx.get_geometry()
    L = L_lam * LAM
    h = h_lam * LAM
    g.wire(1, nseg, -L / 2, 0, h, L / 2, 0, h, a_lam * LAM, 1, 1)
    ctx.geometry_complete(0 if ground is None else 1)
    if ground == "perfect":
        ctx.gn_card(1, 0, 0, 0, 0, 0, 0, 0)
    ctx.ex_card(0, 1, (nseg + 1) // 2, 0, 1.0, 0, 0, 0, 0, 0)
    ctx.fr_card(0, 1, F, 0)
    ctx.xq_card(0)
    return ctx.get_input_parameters(0).get_impedance()[0]


@pytest.mark.skipif(not HAVE_NEC, reason="PyNEC není nainstalován")
@pytest.mark.parametrize("a_lam", [1e-3, 5e-4, 1e-4, 1e-5])
def test_vs_nec_dipole_resistance(a_lam):
    """Odpor musí souhlasit s NEC-2 do 1 %."""
    mine = solve(dipole(0.5, a_lam=a_lam, nseg=61), accurate=True).zin
    ref = _nec_dipole(0.5, a_lam, 61)
    assert abs(mine.real - ref.real) / ref.real < 0.01


@pytest.mark.skipif(not HAVE_NEC, reason="PyNEC není nainstalován")
@pytest.mark.parametrize("a_lam", [5e-4, 1e-4, 1e-5])
def test_vs_nec_dipole_reactance(a_lam):
    """Reaktance musí souhlasit s NEC-2 do 2 Ω (rozdíl modelu jádra)."""
    mine = solve(dipole(0.5, a_lam=a_lam, nseg=61), accurate=True).zin
    ref = _nec_dipole(0.5, a_lam, 61)
    assert abs(mine.imag - ref.imag) < 2.0


@pytest.mark.skipif(not HAVE_NEC, reason="PyNEC není nainstalován")
def test_vs_nec_grounded_monopole():
    """Uzemněný vertikál nad dokonalou zemí: shoda s NEC do 2 %."""
    m = Model(freq_mhz=14.1, material="dokonalý vodič", ground=Ground("perfect"))
    lam = C0 / 14.1e6
    m.wires = [Wire(0, 0, 0, 0, 0, 0.25 * lam, radius=0.005, nseg=21)]
    m.sources = [Source(0, 0.0, 1.0)]
    mine = solve(m, accurate=True).zin
    ctx = nec_context()
    ctx.get_geometry().wire(1, 21, 0, 0, 0, 0, 0, 0.25 * lam, 0.005, 1, 1)
    ctx.geometry_complete(1)
    ctx.gn_card(1, 0, 0, 0, 0, 0, 0, 0)
    ctx.ex_card(0, 1, 1, 0, 1.0, 0, 0, 0, 0, 0)
    ctx.fr_card(0, 1, 14.1, 0)
    ctx.xq_card(0)
    ref = ctx.get_input_parameters(0).get_impedance()[0]
    assert abs(mine - ref) / abs(ref) < 0.02


@pytest.mark.skipif(not HAVE_NEC, reason="PyNEC není nainstalován")
def test_vs_nec_yagi_gain():
    """Zisk 3prvkové Yagi musí souhlasit s NEC do 0,1 dB."""
    els = [(-1.60, 10.60), (0.0, 10.10), (2.20, 9.60)]
    a, f, n = 0.0125, 14.1, 21
    m = Model(freq_mhz=f, material="hliník", ground=Ground("free"))
    m.wires = [Wire(x, -L / 2, 0, x, L / 2, 0, radius=a, nseg=n) for x, L in els]
    m.sources = [Source(1, 0.5, 1.0)]
    p = performance(solve(m, accurate=True), n_th=181, n_ph=181)

    ctx = nec_context()
    g = ctx.get_geometry()
    for i, (x, L) in enumerate(els):
        g.wire(i + 1, n, x, -L / 2, 0, x, L / 2, 0, a, 1, 1)
    ctx.geometry_complete(0)
    ctx.ex_card(0, 2, (n + 1) // 2, 0, 1.0, 0, 0, 0, 0, 0)
    ctx.fr_card(0, 1, f, 0)
    ctx.rp_card(0, 181, 361, 0, 5, 0, 0, 0, 0, 1.0, 1.0, 0, 0)
    ref_gain = ctx.get_radiation_pattern(0).get_gain().max()
    ref_z = ctx.get_input_parameters(0).get_impedance()[0]
    assert abs(p.gain_dbi - ref_gain) < 0.1
    solz = solve(m, accurate=True).zin
    assert abs(solz.real - ref_z.real) / ref_z.real < 0.02


# ---------------------------------------------------------------- soubory
def test_nec_roundtrip():
    m = EXAMPLES["Yagi 3 prvky 20 m"]()
    m2, warn = from_nec(to_nec(m))
    a, b = analyse(m), analyse(m2)
    assert abs(a.zin - b.zin) < 0.05
    assert abs(a.gain_dbi - b.gain_dbi) < 0.02


def test_maa_roundtrip():
    m = EXAMPLES["Yagi 3 prvky 20 m"]()
    m2, warn = from_maa(to_maa(m))
    a, b = analyse(m), analyse(m2)
    assert abs(a.zin - b.zin) < 0.05


def test_maa_parses_real_world_layout():
    text = (
        "Test dipol\n"
        "14.1\n"
        "***Wires***\n"
        "1\n"
        "-5.05, 0, 12, 5.05, 0, 12, 0.001, 21\n"
        "***Source***\n"
        "1\n"
        "w1c, 0, 1.0\n"
        "***Load***\n"
        "0\n"
        "***G/H/M/R/AzEl/X***\n"
        "2, 5, 13, 50, 0, 0, 0\n"
    )
    m, warn = from_maa(text)
    assert len(m.wires) == 1
    assert m.freq_mhz == pytest.approx(14.1)
    assert m.ground.kind == "real"
    assert m.ground.sigma == pytest.approx(0.005)
    assert m.sources[0].pos == pytest.approx(0.5)


def test_all_examples_solve():
    for name, fn in EXAMPLES.items():
        r = analyse(fn())
        assert np.isfinite(r.zin.real) and np.isfinite(r.gain_dbi)
        assert r.zin.real > 0, name


def test_optimizer_improves_cost():
    from antopt.optimize import Parameter, Objective, optimize, evaluate
    m = EXAMPLES["Yagi 3 prvky 20 m"]()
    obj = Objective(w_gain=1.0, w_fb=0.5, w_swr=2.0, n_th=19, n_ph=37)
    before = evaluate(m, obj).cost
    params = [Parameter("delka", [0], 10.2, 11.2), Parameter("delka", [2], 9.2, 10.2)]
    res = optimize(m, params, obj, pop_size=8, generations=5, polish=False, seed=1)
    assert res.cost < before


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
