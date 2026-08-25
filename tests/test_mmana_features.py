"""Testy funkcí převzatých z MMANA — geometrie, průvodci, VF kalkulátory."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from antopt.model import Model, Wire, Source, Ground, C0
from antopt.solver import solve, swr_from_z
from antopt.analysis import analyse, find_resonance, q_factor, q_estimate
from antopt.examples import EXAMPLES
from antopt import geometry_ops as go
from antopt import hfcalc as hf
from antopt import wizards as wz
from antopt.optimize import (Parameter, Objective, optimize, evaluate,
                             expand_values, free_params, read_param, apply_param)


# ---------------------------------------------------------------- geometrie
def test_move_and_rotate_roundtrip():
    m = EXAMPLES["Yagi 3 prvky 20 m"]()
    before = np.array([[*w.a, *w.b] for w in m.wires])
    go.move(m, 3.0, -2.0, 1.5)
    go.rotate(m, 37.0, "z", (1.0, 1.0, 0.0))
    go.rotate(m, -37.0, "z", (1.0, 1.0, 0.0))
    go.move(m, -3.0, 2.0, -1.5)
    after = np.array([[*w.a, *w.b] for w in m.wires])
    assert np.allclose(before, after, atol=1e-9)


def test_rescale_preserves_impedance():
    """Přeladěná anténa musí mít na novém kmitočtu stejnou impedanci."""
    m = EXAMPLES["Dipól 20 m"]()
    m.material = "dokonalý vodič"
    z1 = solve(m).zin
    go.rescale_to_frequency(m, 7.05, scale_radius=True, keep_height=False)
    z2 = solve(m).zin
    assert abs(z2 - z1) / abs(z1) < 0.01
    assert m.freq_mhz == pytest.approx(7.05)


def test_mirror_copy_doubles_wires():
    m = EXAMPLES["Dipól 20 m"]()
    n = len(m.wires)
    go.mirror(m, "x", 5.0, copy=True)
    assert len(m.wires) == 2 * n
    assert m.wires[n].x1 == pytest.approx(10.0 - m.wires[0].x1)


def test_stack_increases_gain():
    m = EXAMPLES["Yagi 3 prvky 20 m"]()
    g1 = analyse(m).gain_dbi
    go.make_stack(m, nx=1, nz=2, dz=m.wavelength / 2)
    r = analyse(m)
    assert len(m.sources) == 2
    assert 1.5 < r.gain_dbi - g1 < 3.5


def test_stack_limit():
    m = EXAMPLES["Dipól 20 m"]()
    with pytest.raises(ValueError):
        go.make_stack(m, nx=9, nz=9)


def test_taper_element_keeps_source_at_centre():
    m = Model(freq_mhz=28.4, material="hliník", ground=Ground("free"))
    m.wires = [Wire(0, -2.5, 0, 0, 2.5, 0, radius=0.008, nseg=21)]
    m.sources = [Source(0, 0.5, 1.0)]
    idx = go.taper_element(m, 0, [go.TaperSection(1.5, 0.0125),
                                  go.TaperSection(1.0, 0.008)])
    assert len(idx) == 4
    r = analyse(m)
    assert np.isfinite(r.zin.real) and r.zin.real > 0
    # napájecí bod zůstal ve středu prvku
    s = m.sources[0]
    w = m.wires[s.wire]
    p = w.a + s.pos * (w.b - w.a)
    assert np.allclose(p, [0, 0, 0], atol=1e-9)


def test_taper_preserves_total_length():
    m = Model(freq_mhz=28.4, ground=Ground("free"))
    m.wires = [Wire(0, -2.5, 0, 0, 2.5, 0, radius=0.008, nseg=21)]
    m.sources = [Source(0, 0.5, 1.0)]
    go.taper_element(m, 0, [go.TaperSection(1.5, 0.0125), go.TaperSection(1.0, 0.008)])
    els = go.find_elements(m)
    assert len(els) == 1
    assert els[0].length == pytest.approx(5.0, abs=1e-6)


def test_find_elements_groups_yagi():
    m = EXAMPLES["Yagi 3 prvky 20 m"]()
    els = go.find_elements(m)
    assert len(els) == 3
    assert [round(e.position_x, 2) for e in els] == [-1.6, 0.0, 2.2]


def test_polar_roundtrip():
    w = go.polar_to_wire([1, 2, 3], 7.5, 33.0, 71.0)
    L, az, ze = go.wire_to_polar(w)
    assert L == pytest.approx(7.5)
    assert az == pytest.approx(33.0)
    assert ze == pytest.approx(71.0)


def test_element_length_change():
    m = EXAMPLES["Yagi 3 prvky 20 m"]()
    el = go.find_elements(m)[0]
    go.set_element_length(m, el, 11.0)
    assert go.find_elements(m)[0].length == pytest.approx(11.0)


# ---------------------------------------------------------------- průvodci
@pytest.mark.parametrize("wiz", wz.WIZARDS, ids=[w.name for w in wz.WIZARDS])
def test_wizards_produce_valid_models(wiz):
    kw = {f.key: f.default for f in wiz.fields}
    if wiz.name == "Yagi":
        kw["boom"] = None
    m = wiz.fn(**kw)
    assert m.wires and m.sources
    r = analyse(m)
    assert np.isfinite(r.zin.real) and r.zin.real > 0
    assert np.isfinite(r.gain_dbi)


def test_yagi_wizard_gain_grows_with_elements():
    gains = [analyse(wz.yagi(28.4, n, height=12.0)).gain_dbi for n in (2, 4, 6)]
    assert gains[0] < gains[1] < gains[2]


def test_loops_near_resonance():
    for m in (wz.delta_loop(), wz.quad(elements=1)):
        z = solve(m).zin
        assert abs(z.imag) < 0.35 * z.real, f"{m.name}: X={z.imag:.1f}, R={z.real:.1f}"


# ------------------------------------------------------------- VF kalkulátory
def test_reactance_roundtrip():
    f = 14.1
    assert hf.reactance_l(hf.l_for_reactance(75.0, f), f) == pytest.approx(75.0)
    assert hf.reactance_c(hf.c_for_reactance(75.0, f), f) == pytest.approx(-75.0)


def test_coil_roundtrip():
    n = hf.coil_turns(2.0, 30.0, 50.0)
    assert hf.coil_inductance(30.0, 50.0, n) == pytest.approx(2.0, rel=1e-6)


def test_lc_resonance():
    assert hf.resonant_frequency(1.0, 1000.0) == pytest.approx(5.0329, rel=1e-3)


@pytest.mark.parametrize("z", [complex(25, -10), complex(12, 0), complex(20, 30),
                               complex(120, -40), complex(200, 60)])
def test_lc_match_reaches_50_ohm(z):
    f, z0 = 14.1, 50.0
    sols = hf.lc_match(z, z0, f)
    assert sols, f"pro {z} nenalezeno řešení"
    for s in sols:
        assert swr_from_z(s.input_impedance(z), z0) < 1.05, f"{z}: {s.topology}"


def test_stub_reactance_roundtrip():
    f, z0, vf = 14.1, 50.0, 0.66
    for x in (30.0, 120.0, -80.0):
        for shorted in (True, False):
            L = hf.stub_length_for_reactance(x, z0, f, vf, shorted)
            assert L >= 0
            got = hf.stub_reactance(z0, L, f, vf, shorted)
            assert got == pytest.approx(x, rel=1e-6, abs=1e-6)


def test_single_stub_match_gives_50_ohm():
    f, z0, vf = 14.1, 50.0, 0.66
    z = complex(25, -30)
    sols = hf.single_stub_match(z, z0, f, vf, vf, z0, shorted=True)
    assert sols
    for s in sols:
        zl = hf.transform_along_line(z, z0, s.distance_m, f, vf)
        xs = hf.stub_reactance(z0, s.stub_len_m, f, vf, True)
        zin = 1.0 / (1.0 / zl + 1.0 / (1j * xs))
        assert swr_from_z(zin, z0) < 1.1


def test_quarter_wave_transformer():
    """λ/4 vedení musí transformovat R na Z₀²/R."""
    f, vf = 14.1, 1.0
    lam = C0 / (f * 1e6)
    z = hf.transform_along_line(complex(12.5, 0), 50.0, lam / 4, f, vf)
    assert z.real == pytest.approx(200.0, rel=1e-3)
    assert abs(z.imag) < 1e-6


def test_half_wave_line_repeats_impedance():
    f, vf = 14.1, 0.66
    lam = C0 / (f * 1e6) * vf
    z0 = complex(30, -40)
    z = hf.transform_along_line(z0, 50.0, lam / 2, f, vf)
    assert abs(z - z0) < 1e-6


def test_coax_loss_grows_with_swr():
    line = hf.LINE_BY_NAME["RG-58 C/U"]
    _, _, _, l_matched = hf.coax_feed(complex(50, 0), line, 30.0, 14.1)
    _, _, _, l_mismatched = hf.coax_feed(complex(10, -40), line, 30.0, 14.1)
    assert l_mismatched > l_matched > 0


# ------------------------------------------------------------- analýza
def test_find_resonance_gives_zero_reactance():
    m = EXAMPLES["Dipól 20 m"]()
    res = find_resonance(m)
    assert res is not None
    f, z = res
    assert abs(z.imag) < 0.05
    work = m.copy(); work.freq_mhz = f
    assert abs(solve(work).zin.imag) < 0.05


def test_q_of_dipole_matches_bandwidth():
    """Q dipólu musí odpovídat naměřené šířce pásma do 20 %."""
    from antopt.analysis import sweep, bandwidth
    m = EXAMPLES["Dipól 20 m"]()
    res = find_resonance(m)
    m.freq_mhz = res[0]
    m.z0 = res[1].real
    q = q_factor(m)
    rs = sweep(m, m.freq_mhz * 0.95, m.freq_mhz * 1.05, 61)
    bw = bandwidth(rs, 2.0)
    assert bw is not None
    est = m.freq_mhz / q * (2 - 1) / math.sqrt(2) * 1000.0
    assert abs(est - bw[2]) / bw[2] < 0.20


def test_q_is_stable_for_dipole():
    q, ok, lo, hi = q_estimate(EXAMPLES["Dipól 20 m"]())
    assert ok and 10 < q < 20


def test_fb_sector_is_never_better_than_direct():
    m = EXAMPLES["Yagi 3 prvky 20 m"]()
    direct = analyse(m).fb_db
    sector = analyse(m, fb_sector_deg=120).fb_db
    assert sector <= direct + 1e-6


# ------------------------------------------------------------- optimalizace
def test_linked_variables():
    ps = [Parameter("delka", [0], 10, 11),
          Parameter("delka", [1], 9, 11, link=0, link_factor=0.95),
          Parameter("delka", [2], 9, 11, link=0, link_offset=-0.5)]
    assert free_params(ps) == [0]
    vals = expand_values(ps, [10.0])
    assert vals == pytest.approx([10.0, 9.5, 9.5])


def test_optimizer_respects_link():
    m = EXAMPLES["Yagi 3 prvky 20 m"]()
    obj = Objective(w_gain=1.0, w_fb=0.3, w_swr=1.0, n_th=13, n_ph=25)
    ps = [Parameter("delka", [0], 10.2, 11.0),
          Parameter("delka", [2], 9.0, 10.5, link=0, link_factor=0.90)]
    r = optimize(m, ps, obj, pop_size=6, generations=3, polish=False, seed=1)
    assert r.values[1] == pytest.approx(r.values[0] * 0.90)


def test_step_quantisation():
    m = EXAMPLES["Yagi 3 prvky 20 m"]()
    obj = Objective(w_gain=1.0, w_fb=0.0, w_swr=0.0, n_th=13, n_ph=25)
    ps = [Parameter("delka", [0], 10.0, 11.0, step=0.05)]
    r = optimize(m, ps, obj, pop_size=6, generations=2, polish=False, seed=3)
    assert abs(r.values[0] / 0.05 - round(r.values[0] / 0.05)) < 1e-6


def test_element_parameter_handles_taper():
    """Parametr 'prvek_delka' musí měnit i zúžený prvek jako celek."""
    m = Model(freq_mhz=28.4, ground=Ground("free"))
    m.wires = [Wire(0, -2.5, 0, 0, 2.5, 0, radius=0.008, nseg=15)]
    m.sources = [Source(0, 0.5, 1.0)]
    go.taper_element(m, 0, [go.TaperSection(1.5, 0.0125), go.TaperSection(1.0, 0.008)])
    p = Parameter("prvek_delka", list(range(len(m.wires))), 4.0, 6.0)
    assert read_param(m, p) == pytest.approx(5.0)
    apply_param(m, p, 4.6)
    assert go.find_elements(m)[0].length == pytest.approx(4.6)


def test_new_parameter_kinds_roundtrip():
    m = EXAMPLES["Yagi 3 prvky 20 m"]()
    m.loads.append(__import__("antopt.model", fromlist=["Load"]).Load(0, 0.5, "RX", 10, 5))
    checks = [
        (Parameter("kmitocet", [], 13, 15), 14.6),
        (Parameter("zatez_r", [0], 0, 100), 33.0),
        (Parameter("zatez_x", [0], -100, 100), -22.0),
        (Parameter("zdroj_u", [0], 0.1, 10), 2.5),
        (Parameter("zdroj_faze", [0], -180, 180), 45.0),
        (Parameter("azimut", [0], -90, 90), 12.0),
        (Parameter("vyska_vse", [], 5, 30), 18.0),
    ]
    for p, val in checks:
        apply_param(m, p, val)
        assert read_param(m, p) == pytest.approx(val, abs=1e-6), p.kind


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# --------------------------------------------------------------------------
# průběh optimalizace se hlásí i během doladění
# --------------------------------------------------------------------------
def test_optimalizace_hlasi_prubeh_i_pri_doladeni():
    """Doladění Nelder-Meadem je nejdelší fáze. Dřív o sobě nedávalo vědět
    a program vypadal zaseklý — ukazatel stál na poslední generaci."""
    from antopt import examples
    from antopt.optimize import Objective, optimize, suggest_parameters

    m = examples.yagi3_20m()
    m.auto_segment(per_wavelength=20.0, min_seg=6)
    ps = suggest_parameters(m)[:2]
    seen = []

    def cb(g, total, best, txt):
        seen.append((g, total, txt))
        return True

    optimize(m, ps, Objective(), pop_size=6, generations=2,
             polish=True, progress=cb)

    faze = lambda p: [t for _, _, t in seen if t.startswith(p)]
    assert faze("výchozí populace"), "start se nehlásí"
    assert len(faze("generace")) == 2
    assert faze("doladění"), "doladění se nehlásí — právě tam to vypadalo zaseklé"
    # ukazatel nesmí přetéct ani se vracet
    tot = seen[-1][1]
    assert all(g <= tot for g, _, _ in seen)
    assert seen[-1][0] == tot


def test_optimalizaci_lze_prerusit_v_doladeni():
    from antopt import examples
    from antopt.optimize import Objective, optimize, suggest_parameters

    m = examples.yagi3_20m()
    m.auto_segment(per_wavelength=20.0, min_seg=6)
    ps = suggest_parameters(m)[:2]
    calls = []

    def cb(g, total, best, txt):
        calls.append(txt)
        return not txt.startswith("doladění")     # stop hned na začátku doladění

    res = optimize(m, ps, Objective(), pop_size=6, generations=2,
                   polish=True, progress=cb)
    assert res.model is not None and res.history


def test_jednotky_promennych():
    """Rozdíly délek se ukazují v milimetrech, úhly ve stupních."""
    from antopt.optimize import param_unit
    assert param_unit("prvek_delka") == ("m", "mm", 1000.0, 4)
    assert param_unit("delka_konec")[1] == "mm"
    assert param_unit("azimut") == ("°", "°", 1.0, 2)
    assert param_unit("zatez_l")[0] == "µH"
    assert param_unit("neznamy")[1] == "mm"          # rozumný výchozí
