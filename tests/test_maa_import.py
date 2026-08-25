"""Import MMANA .maa — hlavička, jednotky, hlášky.

Hlavička .maa je v praxi rozmanitá: někdy komentář, někdy název antény,
někdy obojí. Název přitom bývá plný číslic („OK1M-14-21-28_9el“) a nesmí
se z něj stát kmitočet — na tom se to dřív lámalo úplně tiše.
"""
from __future__ import annotations

import pytest

from antopt import fileio
from antopt import geometry_ops as go
from antopt.model import collapse_messages

ELS = [(-2.0, 5.30), (-0.9, 5.05), (0.6, 4.90), (1.9, 3.55), (2.8, 3.40),
       (3.9, 3.30), (5.2, 2.60), (6.1, 2.50), (7.0, 2.45)]


def _maa(header, radius="-12.5", seg="-1", z=0.0, ground="2, 5.0, 0, 50, 0, 0, 0"):
    lines = list(header) + [str(len(ELS))]
    for x, L in ELS:
        lines.append(f"{x:.4f}, {-L / 2:.4f}, {z}, {x:.4f}, {L / 2:.4f}, {z}, "
                     f"{radius}, {seg}")
    lines += ["***Source***", "1", "w2c, 0.0, 1.0",
              "***Load***", "0",
              "***G/H/M/R/AzEl/X***", ground, "***End***"]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# hlavička
# --------------------------------------------------------------------------
@pytest.mark.parametrize("header", [
    ["*** OK1M-14-21-28_9el_v0.3_20m_15m_10m ***", "14.150"],
    ["*** MMANA-GAL antenna file ***", "OK1M-14-21-28_9el_v0.3", "14.150"],
    ["OK1M 9el 20/15/10", "14.150"],
    ["*** komentář ***", "", "9el tribander v0.3", "", "14.150"],
    ["14.150"],
])
def test_kmitocet_se_nebere_z_nazvu(header):
    m, _ = fileio.from_maa(_maa(header))
    assert m.freq_mhz == pytest.approx(14.150)
    assert len(m.wires) == len(ELS)


def test_nazev_s_cislicemi_prezije():
    m, _ = fileio.from_maa(
        _maa(["*** MMANA ***", "OK1M-14-21-28_9el_v0.3", "14.150"]))
    assert "OK1M" in m.name
    assert m.freq_mhz == pytest.approx(14.150)


def test_rozbita_hlavicka_je_chyba_ne_nesmysl():
    with pytest.raises(ValueError):
        fileio.from_maa("*** nic ***\nnějaký text\njiný text\n")


# --------------------------------------------------------------------------
# jednotky a segmentace
# --------------------------------------------------------------------------
def test_zaporny_polomer_je_v_mm():
    m, warn = fileio.from_maa(_maa(["14.150"], radius="-12.5"))
    assert all(w.radius == pytest.approx(0.0125) for w in m.wires)
    assert sum("v milimetrech" in w for w in warn) == 1, "hláška má být jedna, ne devět"


def test_zuzena_segmentace_da_jednu_hlasku():
    m, warn = fileio.from_maa(_maa(["14.150"], seg="-1"))
    taper = [w for w in warn if "na segmenty" in w]
    assert len(taper) == 1 and "9×" in taper[0]
    assert all(w.nseg >= 6 for w in m.wires)


def test_zadane_segmenty_se_neprepisou():
    """Když soubor u části drátů segmenty určí, musí zůstat."""
    txt = _maa(["14.150"], seg="-1").splitlines()
    txt[2] = txt[2].rsplit(",", 1)[0] + ", 17"      # první drát dostane 17
    m, _ = fileio.from_maa("\n".join(txt))
    assert m.wires[0].nseg == 17
    assert m.wires[1].nseg != 17                     # ostatní doplněny automaticky


# --------------------------------------------------------------------------
# model položený na zemi
# --------------------------------------------------------------------------
def test_anteny_v_rovine_z0_se_zemi_se_pozna():
    m, warn = fileio.from_maa(_maa(["14.150"], z=0.0))
    assert m.ground.kind == "real"
    assert m.lies_on_ground()
    assert any("z = 0" in w for w in warn)
    msgs = m.validate()
    assert len(msgs) == 1, "jedna souhrnná hláška, ne devět stejných"
    assert "9 drátů" in msgs[0]


def test_po_zvednuti_je_model_v_poradku():
    m, _ = fileio.from_maa(_maa(["14.150"], z=0.0))
    go.move(m, dz=12.0)
    assert not m.lies_on_ground()
    assert m.validate() == []


def test_volny_prostor_v_rovine_z0_nevadi():
    m, warn = fileio.from_maa(_maa(["14.150"], z=0.0, ground="0, 0, 0, 50, 0, 0, 0"))
    assert m.ground.kind == "free"
    assert not m.lies_on_ground()
    assert not any("z = 0" in w for w in warn)


# --------------------------------------------------------------------------
# slučování hlášek
# --------------------------------------------------------------------------
def test_collapse_messages():
    msgs = [f"Drát {i}: leží přesně na zemi." for i in (1, 2, 3, 7, 9)]
    out = collapse_messages(msgs)
    assert out == ["Dráty 1–3, 7, 9: leží přesně na zemi."]


def test_collapse_nespojuje_ruzne_hlasky():
    out = collapse_messages(["Drát 1: nulová délka.", "Drát 2: pod zemí (z < 0).",
                             "Model nemá žádný zdroj."])
    assert len(out) == 3


# --------------------------------------------------------------------------
# zúžené PRVKY (mechanika) vs zúžená SEGMENTACE (výpočet) — dvě různé věci
# --------------------------------------------------------------------------
def _maa_tapered_elements(n_wires_decl=None, seg="-1"):
    """3prvková Yagi, každý prvek ze tří trubek 25/20/16 na každé polovině."""
    sec = [(0.9939, -25.0), (0.7454, -20.0), (0.7454, -16.0)]
    w = []
    for x in (-1.05, 0.0, 1.03):
        for sign in (-1, +1):
            t = 0.0
            for L, d in sec:
                a, b = (t, t + L) if sign > 0 else (-(t + L), -t)
                w.append(f"{x}, {a:.4f}, 12.0, {x}, {b:.4f}, 12.0, {d / 2:.1f}, {seg}")
                t += L
    head = ["*** MMANA-GAL antenna file ***", "OK1M 3el zúžené prvky", "28.400"]
    return "\n".join(head + [str(n_wires_decl or len(w))] + w +
                     ["***Source***", "1", "w7b, 0.0, 1.0", "***Load***", "0",
                      "***G/H/M/R/AzEl/X***", "2, 5.0, 0, 50, 0, 0, 0", "***End***"])


def test_zuzene_prvky_se_importuji_se_vsemi_prumery():
    """Zúžená segmentace se nahrazuje, ale průměry trubek zůstávají."""
    m, warn = fileio.from_maa(_maa_tapered_elements())
    assert len(m.wires) == 18
    els = go.find_elements(m)
    assert len(els) == 3
    for el in els:
        secs = go.element_sections(m, el.wires)
        assert [round(s.radius * 2000) for s in secs] == [25, 20, 16]
        assert el.length == pytest.approx(2 * (0.9939 + 0.7454 + 0.7454), abs=1e-6)
    assert any("NENÍ o zúžených prvcích" in w for w in warn)


def test_spatny_pocet_dratu_se_nezahodi_tise():
    """Kdyby se počet drátů přečetl špatně, nesmí se zbytek geometrie ztratit."""
    m, warn = fileio.from_maa(_maa_tapered_elements(n_wires_decl=9))
    assert len(m.wires) == 18, "polovina prvků by se tiše zahodila"
    assert len(go.find_elements(m)) == 3


# --------------------------------------------------------------------------
# jednotka poloměru — tichý zabiják antény
# --------------------------------------------------------------------------
def _maa_3el(rad):
    els = [(-2.05, 10.60), (0.0, 10.10), (2.20, 9.60)]
    L = ["*** MMANA ***", "OK1M 3el 20m", "14.200", str(len(els))]
    for x, ln in els:
        L.append(f"{x}, {-ln / 2:.4f}, 12.0, {x}, {ln / 2:.4f}, 12.0, {rad}, 0")
    L += ["***Source***", "1", "w2c, 0.0, 1.0", "***Load***", "0",
          "***G/H/M/R/AzEl/X***", "2, 5.0, 0, 50, 0, 0, 0", "***End***"]
    return "\n".join(L)


def test_zaporny_polomer_v_metrech_se_nedeli_tisicem():
    """-0.0125 je trubka Ø 25 mm, ne drát 0,0125 mm.

    Dělení tisícem z ní udělalo vodič tenčí než vlas — anténa pak ztratila
    80 % výkonu ve ztrátách a zisk spadl o 9 dB, aniž by to cokoli hlásilo.
    """
    m, _ = fileio.from_maa(_maa_3el("-0.0125"))
    assert m.wires[0].radius == pytest.approx(0.0125)


def test_zaporny_polomer_v_mm_se_prepocte():
    m, _ = fileio.from_maa(_maa_3el("-25.0"))
    assert m.wires[0].radius == pytest.approx(0.025)


def test_prumery_se_vzdy_ohlasi_ke_kontrole():
    _, warn = fileio.from_maa(_maa_3el("0.0125"))
    assert any("ZKONTROLUJ PRŮMĚRY" in w and "25 mm" in w for w in warn)


def test_nesmyslne_tenky_vodic_se_ohlasi():
    m, warn = fileio.from_maa(_maa_3el("0.00001"))
    assert any("tenčí než 0,05 mm" in w for w in warn)
    assert any("není reálný vodič" in msg for msg in m.validate())


def test_spravny_polomer_da_spravnou_ucinnost():
    from antopt import analysis
    m, _ = fileio.from_maa(_maa_3el("-0.0125"))
    r = analysis.analyse(m)
    assert r.efficiency > 0.7, "3prvková Yagi z trubek nemá ztrácet výkon"
    assert r.gain_dbi > 11.0
    assert r.fb_db > 12.0
