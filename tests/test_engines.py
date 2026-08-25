"""Testy výměnných jader: vlastní MoM vs NEC-2."""
import math, os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from antopt.model import Model, Wire, Source, Ground, C0
from antopt.analysis import analyse, find_resonance
from antopt.examples import EXAMPLES
from antopt import engines
from antopt.optimize import Objective, evaluate, Parameter, optimize

HAVE_NEC = engines.get("NEC-2").is_available() if "NEC-2" in [e.name for e in engines.all_engines()] else False
nec_only = pytest.mark.skipif(not HAVE_NEC, reason="PyNEC není nainstalován")


def test_own_engine_always_available():
    assert "vlastní" in engines.available_engines()


def test_unknown_engine_raises():
    with pytest.raises(ValueError):
        engines.get("EZNEC")


@nec_only
def test_free_space_dipole_agrees():
    """Ve volném prostoru se jádra musí shodnout v odporu do 2 %."""
    f = 300.0
    lam = C0 / (f * 1e6)
    m = Model(freq_mhz=f, material="dokonalý vodič", ground=Ground("free"))
    L = 0.478 * lam
    m.wires = [Wire(-L / 2, 0, 0, L / 2, 0, 0, radius=5e-4 * lam, nseg=40)]
    m.sources = [Source(0, 0.5, 1.0)]
    a = analyse(m, engine="vlastní")
    b = analyse(m, engine="NEC-2")
    assert abs(a.zin.real - b.zin.real) / b.zin.real < 0.02
    assert abs(a.gain_dbi - b.gain_dbi) < 0.15


@nec_only
def test_perfect_ground_agrees():
    """Nad dokonalou zemí se jádra musí shodnout — tam neaproximuje ani jedno."""
    m = EXAMPLES["Yagi 3 prvky 20 m"]()
    m.ground = Ground("perfect")
    a = analyse(m, engine="vlastní")
    b = analyse(m, engine="NEC-2")
    assert abs(a.zin.real - b.zin.real) / b.zin.real < 0.10
    assert abs(a.gain_dbi - b.gain_dbi) < 0.3


@nec_only
def test_elevated_ground_plane_junction_feed():
    """Vertikál se 4 radiály — napájení ve větvení musí dát ~34 Ω, ne 70."""
    lam = C0 / 7.1e6
    m = Model(freq_mhz=7.1, material="hliník", ground=Ground("perfect"))
    m.wires = [Wire(0, 0, 3.0, 0, 0, 3.0 + 0.25 * lam, radius=0.015, nseg=20)]
    for k in range(4):
        ang = 2 * math.pi * k / 4
        m.wires.append(Wire(0, 0, 3.0, 0.25 * lam * math.cos(ang),
                            0.25 * lam * math.sin(ang), 3.0, 0.002, 12))
    m.sources = [Source(0, 0.0, 1.0)]
    a = analyse(m, engine="vlastní")
    b = analyse(m, engine="NEC-2")
    assert 25 < a.zin.real < 45, f"vlastní: {a.zin}"
    assert abs(a.zin.real - b.zin.real) / b.zin.real < 0.08


@nec_only
def test_cuts_agree():
    m = EXAMPLES["Yagi 3 prvky 20 m"]()
    outs = {}
    for name in ("vlastní", "NEC-2"):
        r = analyse(m, engine=name)
        ang, pat = engines.get(name).cut_azimuth(r, r.elevation_deg, 73)
        outs[name] = (ang, pat, r)
    a_ang, a_pat, _ = outs["vlastní"]
    b_ang, b_pat, _ = outs["NEC-2"]
    assert np.allclose(a_ang, b_ang)
    i0 = int(np.argmin(np.abs(a_ang)))
    assert abs(a_pat.gain_dbi[i0] - b_pat.gain_dbi[i0]) < 0.3
    # vodorovně polarizovaná Yagi: H složka musí být dominantní v obou jádrech
    for p in (a_pat, b_pat):
        assert p.gain_h_dbi[i0] > p.gain_v_dbi[i0] + 10


@nec_only
def test_vertical_cut_shape():
    m = EXAMPLES["Yagi 3 prvky 20 m"]()
    r = analyse(m, engine="NEC-2")
    ang, pat = engines.get("NEC-2").cut_vertical(r, 0.0, 91)
    assert ang[0] == 0.0 and ang[-1] == pytest.approx(180.0)
    assert np.all(np.isfinite(pat.gain_dbi))


@nec_only
def test_resonance_engines_close():
    m = EXAMPLES["Dipól 20 m"]()
    fa, _ = find_resonance(m, engine="vlastní")
    fb, _ = find_resonance(m, engine="NEC-2")
    assert abs(fa - fb) / fb < 0.01


@nec_only
def test_optimizer_runs_on_nec():
    m = EXAMPLES["Yagi 3 prvky 20 m"]()
    obj = Objective(w_gain=1.0, w_fb=0.0, w_swr=0.0, n_th=13, n_ph=25,
                    engine="NEC-2")
    before = evaluate(m, obj)
    assert before.detail and np.isfinite(before.cost)
    ps = [Parameter("delka", [0], 10.3, 11.0)]
    r = optimize(m, ps, obj, pop_size=5, generations=2, polish=False, seed=1)
    assert r.cost <= before.cost + 1e-9


@nec_only
def test_engine_recorded_in_result():
    m = EXAMPLES["Dipól 20 m"]()
    assert analyse(m, engine="NEC-2").engine == "NEC-2"
    assert analyse(m, engine="vlastní").engine == "vlastní"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
