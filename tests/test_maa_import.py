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


# --------------------------------------------------------------------------
# F/B: dvě různé definice, obě správně
# --------------------------------------------------------------------------
def test_fb_jako_mmana_je_prumer_zadni_oblasti():
    """MMANA hlásí „F/B; Rear: Azim. 120 deg, Elev. 60 deg“ — to není zisk
    v přesně opačném směru, ale střední výkon v celé zadní oblasti.
    Na tribanderu OK1M dá první definice 6,1 dB a druhá 11,6 dB;
    autorova MMANA hlásí 11,23 dB."""
    from antopt.model import Model, Wire, Source, Ground
    from antopt import analysis
    W = [(-2.75, 5.50, 0.001, 24), (0.0, 5.16, 0.001, 22), (-0.26, 3.485, 0.002, 16),
         (0.22, 2.53, 0.008, 12), (-2.20, 3.575, 0.002, 16), (-1.53, 2.552, 0.008, 12),
         (1.198, 3.33, 0.002, 14), (0.693, 2.441, 0.008, 10), (2.44, 2.348, 0.008, 10)]
    m = Model(name="tribander", freq_mhz=14.2, material="hliník")
    m.wires = [Wire(x, -y, 12.0, x, y, 12.0, r, n) for x, y, r, n in W]
    m.sources = [Source(1, 0.5, 1.0)]
    m.ground = Ground.from_name("průměrná")
    r = analysis.analyse(m)
    assert r.fb_mmana_db > r.fb_db + 3.0
    assert r.fb_mmana_db == pytest.approx(11.2, abs=1.5)   # MMANA: 11,23 dB
    assert r.elevation_deg == pytest.approx(23.7, abs=1.0)  # MMANA: 23,7°
    assert r.efficiency > 0.7


def test_neznamy_typ_zeme_tise_nevypne_zem():
    from antopt.model import Ground
    assert Ground.from_name("dobrá").kind == "real"        # zkrácený název projde
    with pytest.raises(ValueError):
        Ground.from_name("nesmysl")


# --------------------------------------------------------------------------
# teleskopické prvky zadané rozpisem trubek (MMANA je nepíše jako dráty)
# --------------------------------------------------------------------------
OK1M = [(-2.7500, 11.0000, "-0.001", 24), (0.0000, 10.3200, "-0.001", 22),
        (-2.2000, 7.1500, "-0.002", 16), (-0.2600, 6.9700, "-0.002", 16),
        (1.1980, 6.6600, "-0.002", 14), (-1.5300, 5.1040, "0.008", 12),
        (0.2200, 5.0600, "0.008", 12), (0.6930, 4.8820, "0.008", 10),
        (2.4400, 4.6960, "0.008", 10)]
TAPER = ["-0.001, 0, 2.4, 0.015, 1.2, 0.0125, 1.2, 0.01, 99999.9, 0.008",
         "-0.002, 0, 2.4, 0.01, 99999.9, 0.008",
         "-0.003, 0, 99999.9, 0.015"]


def _maa_ok1m(taper=True):
    L = ["*** MMANA-GAL antenna file ***", "OK1M-14-21-28_9el_v0.3", "14.200", "9"]
    for x, ln, r, n in OK1M:
        L.append(f"{x:.4f}, {-ln / 2:.4f}, 12.0, {x:.4f}, {ln / 2:.4f}, 12.0, {r}, {n}")
    L += ["***Source***", "1", "w2c, 0.0, 1.0", "***Load***", "0"]
    if taper:
        L += ["***Segmentation***"] + TAPER
    L += ["***G/H/M/R/AzEl/X***", "2, 5.0, 0, 50, 0, 0, 0", "***End***"]
    return "\n".join(L)


def test_rozpis_trubek_sestavi_teleskopicke_prvky():
    m, warn = fileio.from_maa(_maa_ok1m())
    els = go.find_elements(m)
    assert len(els) == 9, "prvků musí zůstat devět"
    by_x = {round(e.position_x, 4): e for e in els}
    refl = by_x[-2.7500]                       # 20 m: 30/25/20/16
    secs = go.element_sections(m, refl.wires)
    assert [round(s.radius * 2000) for s in secs] == [30, 25, 20, 16]
    # první trubka je 2,4 m PŘES STŘED, tedy 1,2 m na stranu
    assert secs[0].length == pytest.approx(1.2, abs=1e-6)
    assert secs[1].length == pytest.approx(1.2, abs=1e-6)
    assert 2 * sum(s.length for s in secs) == pytest.approx(11.0, abs=1e-6)
    d15 = go.element_sections(m, by_x[-2.2000].wires)          # 15 m: 20/16
    assert [round(s.radius * 2000) for s in d15] == [20, 16]
    d10 = go.element_sections(m, by_x[-1.5300].wires)          # 10 m: celý 16
    assert [round(s.radius * 2000) for s in d10] == [16]
    assert any("teleskopický prvek" in w for w in warn)


def test_zdroj_zustane_na_zarici_i_po_rozbaleni():
    m, _ = fileio.from_maa(_maa_ok1m())
    driven = [e for e in go.find_elements(m) if abs(e.position_x) < 1e-9][0]
    assert m.sources[0].wire in driven.wires


def test_rozpis_trubek_trefi_hodnoty_autora():
    """Bez rozpisu vyšla anténa o 1 dB hůř a s jinou impedancí.

    Autor v MMANA: Z = 49,2 + 2,6j, PSV 1,1, zisk 11,4 dBi, elevace 23,7°.
    """
    from antopt import analysis
    m, _ = fileio.from_maa(_maa_ok1m())
    r = analysis.analyse(m)
    assert r.zin.real == pytest.approx(49.2, abs=3.0)
    assert r.zin.imag == pytest.approx(2.6, abs=3.0)
    assert r.gain_dbi == pytest.approx(11.4, abs=0.4)
    assert r.swr == pytest.approx(1.1, abs=0.15)
    assert r.elevation_deg == pytest.approx(23.7, abs=1.0)


def test_bez_rozpisu_vyjde_antena_znatelne_jinak():
    """Kontrola, že to není náhoda — bez tabulky trubek se výsledek liší."""
    from antopt import analysis
    a = analysis.analyse(fileio.from_maa(_maa_ok1m(taper=True))[0])
    b = analysis.analyse(fileio.from_maa(_maa_ok1m(taper=False))[0])
    assert abs(a.zin - b.zin) > 8.0
    assert a.gain_dbi - b.gain_dbi > 0.5
