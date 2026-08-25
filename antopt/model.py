"""Datové struktury popisující anténní model.

Souřadnice v metrech, X = dopředu (boom), Y = do stran, Z = výška nad zemí.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import numpy as np

C0 = 299_792_458.0
MU0 = 4.0e-7 * math.pi
EPS0 = 1.0 / (MU0 * C0 * C0)
ETA0 = math.sqrt(MU0 / EPS0)

# vodivost běžných materiálů [S/m]
MATERIALS = {
    "dokonalý vodič": 0.0,          # 0 == bez ztrát
    "měď": 5.8e7,
    "hliník": 3.54e7,
    "mosaz": 1.5e7,
    "ocel": 1.0e7,
    "nerez": 1.45e6,
    "pozinkovaný drát": 8.0e6,
}

# typy země: (epsilon_r, sigma [S/m])
GROUND_TYPES = {
    "volný prostor": None,
    "dokonalá zem": "perfect",
    "velmi dobrá (bažina)": (30.0, 0.01),
    "dobrá (pastvina)": (13.0, 0.005),
    "průměrná": (13.0, 0.005),
    "středně dobrá (les)": (13.0, 0.006),
    "špatná (písek, skála)": (5.0, 0.001),
    "velmi špatná (město)": (3.0, 0.001),
    "mořská voda": (81.0, 5.0),
    "sladká voda": (80.0, 0.001),
}


@dataclass
class Wire:
    """Přímý drátový segment mezi body a -> b."""
    x1: float
    y1: float
    z1: float
    x2: float
    y2: float
    z2: float
    radius: float = 0.001
    nseg: int = 11
    name: str = ""

    @property
    def a(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.z1], dtype=float)

    @property
    def b(self) -> np.ndarray:
        return np.array([self.x2, self.y2, self.z2], dtype=float)

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.b - self.a))

    def center(self) -> np.ndarray:
        return 0.5 * (self.a + self.b)


@dataclass
class Source:
    """Napěťový zdroj (delta-gap) v uzlu drátu.

    ``wire`` je index drátu, ``pos`` relativní poloha 0..1 podél drátu.
    Zdroj se přichytí k nejbližšímu vnitřnímu uzlu sítě.
    """
    wire: int
    pos: float = 0.5
    voltage: float = 1.0      # amplituda [V]
    phase: float = 0.0        # fáze [°]

    def phasor(self) -> complex:
        return self.voltage * np.exp(1j * math.radians(self.phase))


@dataclass
class Load:
    """Soustředná zátěž v uzlu.

    kind = 'RX'  -> R + jX [ohm] (X platí na aktuálním kmitočtu)
    kind = 'RLC' -> sériové R [ohm], L [uH], C [pF]
    """
    wire: int
    pos: float = 0.5
    kind: str = "RX"
    r: float = 0.0
    x: float = 0.0
    l_uh: float = 0.0
    c_pf: float = 0.0
    parallel: bool = False

    def impedance(self, freq_hz: float) -> complex:
        if self.kind == "RX":
            return complex(self.r, self.x)
        w = 2 * math.pi * freq_hz
        zl = 1j * w * self.l_uh * 1e-6 if self.l_uh else 0.0
        if self.c_pf:
            zc = 1.0 / (1j * w * self.c_pf * 1e-12)
        else:
            zc = None
        if self.parallel:
            y = 0.0 + 0.0j
            if self.r:
                y += 1.0 / self.r
            if zl:
                y += 1.0 / zl
            if zc is not None:
                y += 1.0 / zc
            return 1.0 / y if y != 0 else 0.0 + 0.0j
        z = complex(self.r, 0.0)
        if zl:
            z += zl
        if zc is not None:
            z += zc
        return z


@dataclass
class Ground:
    kind: str = "free"          # 'free' | 'perfect' | 'real'
    eps_r: float = 13.0
    sigma: float = 0.005

    @classmethod
    def from_name(cls, name: str) -> "Ground":
        spec = GROUND_TYPES.get(name)
        if spec is None:
            return cls("free")
        if spec == "perfect":
            return cls("perfect")
        return cls("real", spec[0], spec[1])

    def eps_complex(self, freq_hz: float) -> complex:
        return complex(self.eps_r, 0.0) - 1j * self.sigma / (2 * math.pi * freq_hz * EPS0)


@dataclass
class Model:
    name: str = "Anténa"
    freq_mhz: float = 14.1
    wires: List[Wire] = field(default_factory=list)
    sources: List[Source] = field(default_factory=list)
    loads: List[Load] = field(default_factory=list)
    ground: Ground = field(default_factory=Ground)
    material: str = "měď"
    z0: float = 50.0
    ground_loss_r: float = 0.0   # sériový odpor zemního systému u uzemněných zdrojů [Ω]

    # ---------------------------------------------------------------- utils
    @property
    def freq_hz(self) -> float:
        return self.freq_mhz * 1e6

    @property
    def wavelength(self) -> float:
        return C0 / self.freq_hz

    def conductivity(self) -> float:
        return MATERIALS.get(self.material, 0.0)

    def copy(self) -> "Model":
        return Model.from_dict(json.loads(json.dumps(self.to_dict())))

    def bounds(self):
        pts = []
        for w in self.wires:
            pts.append(w.a)
            pts.append(w.b)
        if not pts:
            return np.zeros(3), np.ones(3)
        p = np.array(pts)
        return p.min(axis=0), p.max(axis=0)

    def auto_segment(self, per_wavelength: float = 45.0, min_seg: int = 6,
                     max_seg: int = 150, force_even: bool = True) -> None:
        """Nastaví počet segmentů podle délky drátu vůči vlnové délce.

        45 segmentů na vlnovou délku je bezpečné minimum — s klasickým
        pravidlem 20/λ se u lomených antén rozchází reaktance o jednotky ohmů.

        Sudý počet je pro vlastní jádro lepší: bázové funkce sedí na uzlech,
        takže při sudém počtu leží uzel přesně ve středu drátu a napájení
        uprostřed je symetrické. (NEC to má naopak — tam se budí střed
        segmentu, takže mu vyhovuje lichý počet; převod si hlídá sám.)

        **Zúžené (teleskopické) prvky** se segmentují jako celek: všechny
        sekce prvku dostanou stejně dlouhé segmenty. Kdyby se na každou
        sekci pustilo ``min_seg`` zvlášť, měla by krátká koncová trubka
        segmenty několikrát kratší než střed — a právě sousedství různě
        dlouhých segmentů na skoku průměru dělá u zúžených prvků
        nejvíc škody.
        """
        from .geometry_ops import find_elements

        lam = self.wavelength
        target = lam / max(1.0, per_wavelength)
        try:
            elements = find_elements(self)
        except Exception:
            elements = []
        grouped = set()
        for el in elements:
            if len(el.wires) < 2:
                continue
            grouped.update(el.wires)
            # společná délka segmentu pro celý prvek
            n_tot = max(min_seg, int(round(el.length / target)))
            n_tot = min(max_seg, n_tot)
            seg = el.length / max(1, n_tot)
            for i in el.wires:
                w = self.wires[i]
                w.nseg = max(1, min(max_seg, int(round(w.length / seg))))
        for i, w in enumerate(self.wires):
            if i in grouped:
                continue
            n = max(min_seg, int(round(per_wavelength * w.length / lam)))
            n = min(max_seg, n)
            if force_even and n % 2:
                n += 1
            w.nseg = n

    def validate(self) -> List[str]:
        msgs = []
        lam = self.wavelength
        for i, w in enumerate(self.wires):
            if w.length <= 0:
                msgs.append(f"Drát {i + 1} má nulovou délku.")
                continue
            seg_len = w.length / max(w.nseg, 1)
            if w.radius <= 0:
                msgs.append(f"Drát {i + 1}: poloměr musí být > 0.")
            elif seg_len < 4 * w.radius:
                msgs.append(
                    f"Drát {i + 1}: segment ({seg_len * 1000:.1f} mm) je kratší než 4x poloměr "
                    f"— sniž počet segmentů nebo poloměr."
                )
            if seg_len > lam / 8:
                msgs.append(
                    f"Drát {i + 1}: segment je delší než λ/8 — zvyš počet segmentů."
                )
            if self.ground.kind != "free":
                if min(w.z1, w.z2) < -1e-9:
                    msgs.append(f"Drát {i + 1} je pod zemí (z < 0).")
                if abs(w.z1) < 1e-9 and abs(w.z2) < 1e-9:
                    msgs.append(
                        f"Drát {i + 1} leží přesně na zemi — se zemí se zkratuje. "
                        f"Zvedni ho aspoň pár mm."
                    )
        if not self.sources:
            msgs.append("Model nemá žádný zdroj.")
        return msgs

    # ------------------------------------------------------------- (de)ser.
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "freq_mhz": self.freq_mhz,
            "wires": [asdict(w) for w in self.wires],
            "sources": [asdict(s) for s in self.sources],
            "loads": [asdict(l) for l in self.loads],
            "ground": asdict(self.ground),
            "material": self.material,
            "z0": self.z0,
            "ground_loss_r": self.ground_loss_r,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Model":
        return cls(
            name=d.get("name", "Anténa"),
            freq_mhz=d.get("freq_mhz", 14.1),
            wires=[Wire(**w) for w in d.get("wires", [])],
            sources=[Source(**s) for s in d.get("sources", [])],
            loads=[Load(**l) for l in d.get("loads", [])],
            ground=Ground(**d.get("ground", {})),
            material=d.get("material", "měď"),
            z0=d.get("z0", 50.0),
            ground_loss_r=d.get("ground_loss_r", 0.0),
        )

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Model":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
