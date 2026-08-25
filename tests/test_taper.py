"""Skládání prvků ze sekcí (zúžené / teleskopické prvky).

Kromě čistě geometrických kontrol je tady i porovnání s publikovanými
referenčními hodnotami pro zúžené dipóly (L. B. Cebik, W4RNL: „Tapering to
Perfection“ — tabulka zúžených dipólů na 14 MHz).  Referenční hodnota je
tam „opravená“ (substituční prvek podle Leesona), která se shoduje
s MININECem; nekorigovaný NEC-2 se od ní liší výrazně víc.
"""
from __future__ import annotations

import numpy as np
import pytest

from antopt import analysis, examples
from antopt import geometry_ops as go
from antopt import optimize as opt
from antopt.model import Ground, Model, Source, Wire

T = go.TaperSection
IN = 0.0254


def _yagi_tapered(n_sec: bool = True) -> Model:
    """6prvková Yagi se všemi prvky poskládanými z trubek 25/20/16."""
    m = examples.yagi6_10m()
    for _ in range(6):
        plain = [e for e in go.find_elements(m) if len(e.wires) == 1]
        if not plain:
            break
        el = plain[0]
        half = el.length / 2.0
        go.taper_element(m, el.wires, [T(half * 0.40, 0.0125),
                                       T(half * 0.30, 0.010),
                                       T(half * 0.30, 0.008)])
    return m


# --------------------------------------------------------------------------
# geometrie
# --------------------------------------------------------------------------
def test_taper_zachova_delku_a_polohu_prvku():
    base = examples.yagi6_10m()
    before = [(e.position_x, e.length) for e in go.find_elements(base)]
    m = _yagi_tapered()
    after = [(e.position_x, e.length) for e in go.find_elements(m)]
    assert len(after) == 6
    for (x0, l0), (x1, l1) in zip(before, after):
        assert x1 == pytest.approx(x0, abs=1e-9)
        assert l1 == pytest.approx(l0, abs=1e-9)


def test_taper_vsech_prvku_nechá_zdroj_na_zarici():
    m = _yagi_tapered()
    assert len(m.sources) == 1
    src_wire = m.sources[0].wire
    driven = [e for e in go.find_elements(m) if abs(e.position_x) < 1e-9][0]
    assert src_wire in driven.wires, "zdroj utekl z zářiče na jiný prvek"
    # a sedí na středu prvku
    w = m.wires[src_wire]
    p = w.a + m.sources[0].pos * (w.b - w.a)
    assert np.linalg.norm(p - driven.center) < 1e-6


def test_taper_zachova_pocet_prvku_i_pri_opakovani():
    """Zúžený prvek jde znovu otevřít a přestavět, ne rozbít na kusy."""
    m = _yagi_tapered()
    el = [e for e in go.find_elements(m) if abs(e.position_x) < 1e-9][0]
    n_before = len(m.wires)
    go.taper_element(m, el.wires, [T(1.2, 0.0125), T(0.8, 0.010), T(0.6, 0.008)])
    assert len(go.find_elements(m)) == 6
    assert len(m.wires) == n_before          # 6 sekcí za 6 sekcí
    el = [e for e in go.find_elements(m) if abs(e.position_x) < 1e-9][0]
    assert el.length == pytest.approx(5.2, abs=1e-9)
    assert m.sources[0].wire in el.wires


def test_element_sections_je_protejsek_taperu():
    m = _yagi_tapered()
    for el in go.find_elements(m):
        secs = go.element_sections(m, el.wires)
        assert len(secs) == 3
        assert 2 * sum(s.length for s in secs) == pytest.approx(el.length, abs=1e-9)
        assert [round(s.radius, 6) for s in secs] == [0.0125, 0.010, 0.008]


def test_taper_vybraneho_drátu_vezme_cely_prvek():
    m = _yagi_tapered()
    el = go.find_elements(m)[3]
    inner = el.wires[2]                       # prostřední trubka, ne krajní
    got = go.element_wires(m, inner)
    assert got == list(el.wires)


def test_segmenty_jsou_v_prvku_stejne_dlouhe():
    m = _yagi_tapered()
    for el in go.find_elements(m):
        lens = [m.wires[i].length / m.wires[i].nseg for i in el.wires]
        assert max(lens) / min(lens) < 1.15, "segmenty přes skok průměru se moc liší"


def test_auto_segment_nerozhodi_zuzeny_prvek():
    m = _yagi_tapered()
    m.auto_segment()
    for el in go.find_elements(m):
        lens = [m.wires[i].length / m.wires[i].nseg for i in el.wires]
        assert max(lens) / min(lens) < 1.35


def test_set_element_tip_meni_jen_koncovou_trubku():
    m = _yagi_tapered()
    el = go.find_elements(m)[0]
    secs0 = go.element_sections(m, el.wires)
    go.set_element_tip(m, el, secs0[-1].length + 0.10)
    el = go.find_elements(m)[0]
    secs1 = go.element_sections(m, el.wires)
    assert [round(s.length, 6) for s in secs1[:-1]] == \
           [round(s.length, 6) for s in secs0[:-1]]
    assert secs1[-1].length == pytest.approx(secs0[-1].length + 0.10, abs=1e-9)


def test_taper_precisluje_ostatni_draty():
    """Zúžení jednoho prvku nesmí posunout zdroj na sousední prvek."""
    m = examples.yagi6_10m()
    src = (m.sources[0].wire, m.sources[0].pos)
    go.taper_element(m, 0, [T(1.3, 0.0125), T(1.3444, 0.008)])   # reflektor
    assert len(go.find_elements(m)) == 6
    driven = [e for e in go.find_elements(m) if abs(e.position_x) < 1e-9][0]
    assert m.sources[0].wire in driven.wires
    assert m.sources[0].pos == pytest.approx(src[1])


# --------------------------------------------------------------------------
# optimalizace
# --------------------------------------------------------------------------
def test_navrh_promennych_bere_zuzeny_prvek_jako_celek():
    m = _yagi_tapered()
    ps = opt.suggest_parameters(m)
    delky = [p for p in ps if p.kind in ("delka", "delka_konec", "prvek_delka")]
    assert len(delky) == 6, "u 6prvkové Yagi má být 6 délek, ne jedna na trubku"
    assert all(p.kind == "prvek_delka" for p in delky)
    assert all(len(p.wires) == 6 for p in delky)
    posuny = [p for p in ps if p.kind in ("posun_x", "prvek_x")]
    assert len(posuny) == 5 and all(p.kind == "prvek_x" for p in posuny)


def test_zmena_delky_prvku_nerozpoji_sekce():
    m = _yagi_tapered()
    ps = [p for p in opt.suggest_parameters(m) if p.kind == "prvek_delka"]
    opt.apply_param(m, ps[0], 5.60)
    els = go.find_elements(m)
    assert len(els) == 6
    assert els[0].length == pytest.approx(5.60, abs=1e-9)
    assert len(els[0].wires) == 6           # nerozpadl se


def test_parametr_hrot():
    m = _yagi_tapered()
    el = go.find_elements(m)[0]
    p = opt.Parameter("prvek_hrot", list(el.wires), 0.5, 1.2)
    cur = opt.read_param(m, p)
    opt.apply_param(m, p, cur + 0.08)
    assert opt.read_param(m, p) == pytest.approx(cur + 0.08, abs=1e-9)
    assert len(go.find_elements(m)) == 6


def test_nezuzeny_model_ma_navrh_beze_zmeny():
    m = examples.yagi6_10m()
    ps = opt.suggest_parameters(m)
    delky = [p for p in ps if p.kind in ("delka", "delka_konec")]
    assert len(delky) == 6


# --------------------------------------------------------------------------
# fyzika: porovnání s publikovanými hodnotami (W4RNL)
# --------------------------------------------------------------------------
def _dipole(secs, freq=14.0):
    tot = sum(s.length for s in secs)
    m = Model(name="ref", freq_mhz=freq, material="hliník")
    m.wires = [Wire(0, -tot, 0, 0, tot, 0, secs[-1].radius, 20)]
    m.sources = [Source(0, 0.5, 1.0)]
    m.ground = Ground("free")
    if len(secs) > 1:
        go.taper_element(m, 0, secs, seg_per_wavelength=60.0)
    else:
        m.auto_segment(per_wavelength=60.0)
    return m


@pytest.mark.parametrize("name,secs,r_ref,x_ref", [
    # W4RNL, „Tapering to Perfection“, dipóly 14 MHz, hodnoty po korekci
    ("celistvý 1,0″ / 402,5″", [T(402.5 / 2 * IN, 0.5 * IN)], 71.8, -0.6),
    ("skok daleko od středu", [T(150 * IN, 0.5 * IN), T(54 * IN, 0.375 * IN)],
     72.0, 0.4),
    ("skok blízko středu", [T(50 * IN, 0.5 * IN), T(154 * IN, 0.375 * IN)],
     71.8, -0.5),
])
def test_zuzeny_dipol_proti_publikovanym_hodnotam(name, secs, r_ref, x_ref):
    z = analysis.analyse(_dipole(secs)).zin
    assert z.real == pytest.approx(r_ref, abs=2.0), name
    # ±5 Ω je poctivá mez: nekorigovaný NEC-2 se na týchž případech mýlí o 5–7 Ω
    assert z.imag == pytest.approx(x_ref, abs=5.0), name


def test_zuzeni_posouva_rezonanci_spravnym_smerem():
    """Prvek se silnějším středem je elektricky KRATŠÍ než stejně dlouhý
    celistvý prvek — u obou průměrů. To je ten skutečný efekt zúžení,
    který nekorigovaný NEC-2 zachytí jen zčásti."""
    L = 408 * IN
    x_uni = []
    for d in (1.0, 0.75):
        m = _dipole([T(L / 2, d * IN / 2)])
        x_uni.append(analysis.analyse(m).zin.imag)
    x_tap = analysis.analyse(
        _dipole([T(150 * IN, 0.5 * IN), T(54 * IN, 0.375 * IN)])).zin.imag
    assert x_tap < min(x_uni) - 5.0
