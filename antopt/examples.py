"""Vestavěné ukázkové modely."""
from __future__ import annotations

from .model import Model, Wire, Source, Load, Ground, C0


def dipole_20m() -> Model:
    m = Model(name="Dipól 20 m ve výšce 12 m", freq_mhz=14.1, material="měď")
    L = 10.05
    m.wires = [Wire(-L / 2, 0, 12.0, L / 2, 0, 12.0, radius=0.001, nseg=31)]
    m.sources = [Source(0, 0.5, 1.0)]
    m.ground = Ground.from_name("průměrná")
    return m


def inverted_v_40m() -> Model:
    m = Model(name="Inverted V 40 m, vrchol 14 m", freq_mhz=7.05, material="měď")
    m.wires = [
        Wire(0, 0, 14.0, 0, -9.6, 7.0, radius=0.00075, nseg=31),
        Wire(0, 0, 14.0, 0, 9.6, 7.0, radius=0.00075, nseg=31),
    ]
    m.sources = [Source(0, 0.0, 1.0)]
    m.ground = Ground.from_name("průměrná")
    return m


def yagi3_20m() -> Model:
    m = Model(name="Yagi 3 prvky, 20 m, ráhno 3,8 m", freq_mhz=14.150,
              material="hliník")
    a = 0.0125
    m.wires = [
        Wire(-1.60, -5.30, 12.0, -1.60, 5.30, 12.0, radius=a, nseg=21),   # reflektor
        Wire(0.00, -5.05, 12.0, 0.00, 5.05, 12.0, radius=a, nseg=21),    # zářič
        Wire(2.20, -4.80, 12.0, 2.20, 4.80, 12.0, radius=a, nseg=21),    # direktor
    ]
    m.sources = [Source(1, 0.5, 1.0)]
    m.ground = Ground.from_name("průměrná")
    return m


def yagi5_2m() -> Model:
    m = Model(name="Yagi 5 prvků, 2 m", freq_mhz=145.0, material="hliník")
    a = 0.004
    els = [(-0.30, 1.030), (0.0, 0.980), (0.28, 0.940),
           (0.72, 0.928), (1.24, 0.918)]
    m.wires = [Wire(x, -L / 2, 5.0, x, L / 2, 5.0, radius=a, nseg=15)
               for x, L in els]
    m.sources = [Source(1, 0.5, 1.0)]
    m.ground = Ground("free")
    return m


def vertical_40m() -> Model:
    m = Model(name="Vertikál 1/4 vlny 40 m se zemním systémem", freq_mhz=7.1,
              material="hliník")
    m.wires = [Wire(0, 0, 0.0, 0, 0, 10.2, radius=0.015, nseg=21)]
    m.sources = [Source(0, 0.0, 1.0)]
    m.ground = Ground.from_name("průměrná")
    m.ground_loss_r = 12.0
    return m


def loop_20m() -> Model:
    m = Model(name="Delta loop 20 m", freq_mhz=14.1, material="měď")
    s = 7.15
    h = s * 0.866
    top = 12.0
    m.wires = [
        Wire(0, -s / 2, top - h, 0, 0, top, radius=0.001, nseg=21),
        Wire(0, 0, top, 0, s / 2, top - h, radius=0.001, nseg=21),
        Wire(0, s / 2, top - h, 0, -s / 2, top - h, radius=0.001, nseg=21),
    ]
    m.sources = [Source(2, 0.5, 1.0)]
    m.ground = Ground.from_name("průměrná")
    return m


def yagi6_10m() -> Model:
    """6prvková Yagi na 10 m, ráhno 7,5 m, AL trubka 16×1,5, zářič pro vlásenku.

    Optimalizováno na 28,0–28,8 MHz. Zářič je záměrně zkrácený (kapacitní),
    aby se dal přizpůsobit vlásenkou na 50 Ω — bez ní je PSV vysoké.
    """
    m = Model(name="Yagi 6 prvků 10 m (ráhno 7,5 m)", freq_mhz=28.4,
              material="hliník")
    a = 0.008                       # AL trubka 16×1,5
    els = [(-1.0500, 5.2880),       # reflektor
           (0.0000, 4.9694),        # zářič (zkrácený pro vlásenku)
           (1.0294, 4.9158),        # D1
           (2.8957, 4.6600),        # D2
           (4.1061, 4.7819),        # D3
           (6.4500, 4.6667)]        # D4
    m.wires = [Wire(x, -L / 2, 10.0, x, L / 2, 10.0, radius=a, nseg=17)
               for x, L in els]
    m.sources = [Source(1, 0.5, 1.0)]
    m.ground = Ground.from_name("průměrná")
    return m


EXAMPLES = {
    "Yagi 6 prvků 10 m": yagi6_10m,
    "Dipól 20 m": dipole_20m,
    "Inverted V 40 m": inverted_v_40m,
    "Yagi 3 prvky 20 m": yagi3_20m,
    "Yagi 5 prvků 2 m": yagi5_2m,
    "Vertikál 40 m": vertical_40m,
    "Delta loop 20 m": loop_20m,
}
