"""Výměnná výpočetní jádra.

AntOpt umí počítat dvěma jádry se stejným rozhraním:

* **vlastní** — metoda momentů napsaná v tomhle projektu (``solver.py``).
  Nepotřebuje nic navíc, běží všude, rychlá.
* **NEC-2** — původní kód z Lawrence Livermore, přes balíček ``PyNEC``.
  Umí navíc **Sommerfeldovu zem**, takže dává správnou impedanci i tam,
  kde vlastní jádro jen aproximuje: dráty blízko země, ramena visící
  k zemi, vertikály napájené proti zemi.

Poznámka k tomu, co jde a nejde zabudovat: **4nec2 a EZNEC nejsou solvery,
ale uzavřená okna nad NEC-2** (EZNEC Pro nad NEC-4, který je licencovaný a
exportně omezený) — knihovna, na kterou by šlo linkovat, neexistuje.
**MININEC 3** je sice veřejný, ale je to starší a méně přesný předchůdce
NEC-2, ze kterého vychází MMANA. Z těch čtyř je tedy NEC-2 jediné jádro,
které má smysl a jde použít.

PyNEC je pod GPL. Je proto zapojený jako **volitelný** doplněk načítaný za
běhu — AntOpt samotný na něm nezávisí.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .model import Model, C0, ETA0
from .solver import solve, swr_from_z, Solution


# ==========================================================================
#  společný tvar výsledku
# ==========================================================================
@dataclass
class PatternData:
    """Vyzařovací diagram nezávislý na jádru (stejné rozhraní jako Pattern)."""
    theta: np.ndarray
    phi: np.ndarray
    gain_dbi: np.ndarray
    gain_v: np.ndarray
    gain_h: np.ndarray

    @property
    def gain_v_dbi(self) -> np.ndarray:
        return self.gain_v

    @property
    def gain_h_dbi(self) -> np.ndarray:
        return self.gain_h

    def component(self, which: str) -> np.ndarray:
        which = (which or "total").lower()
        if which.startswith("v"):
            return self.gain_v
        if which.startswith("h"):
            return self.gain_h
        return self.gain_dbi


@dataclass
class Geometry:
    """Body segmentů a velikost proudu — pro barevný náhled."""
    a: np.ndarray
    b: np.ndarray
    magnitude: np.ndarray


# ==========================================================================
#  rozhraní jádra
# ==========================================================================
class Engine:
    name = "?"
    description = ""

    @staticmethod
    def is_available() -> bool:
        return True

    def analyse(self, model: Model, n_th: int = 91, n_ph: int = 181,
                keep_solution: bool = True, fb_sector_deg: float = 0.0):
        raise NotImplementedError

    def cut_azimuth(self, result, elevation_deg: float, n: int = 361):
        raise NotImplementedError

    def cut_vertical(self, result, azimuth_deg: float, n: int = 361):
        raise NotImplementedError

    def geometry(self, result) -> Optional[Geometry]:
        return None


# --------------------------------------------------------------------------
class OwnEngine(Engine):
    name = "vlastní"
    description = ("Metoda momentů z tohoto projektu. Bez dalších závislostí. "
                   "Reálná zem se v impedanci aproximuje.")

    def analyse(self, model, n_th=91, n_ph=181, keep_solution=True,
                fb_sector_deg=0.0):
        from .analysis import _analyse_own
        return _analyse_own(model, n_th, n_ph, keep_solution, fb_sector_deg)

    def cut_azimuth(self, result, elevation_deg, n=361):
        from .farfield import cut_azimuth as f
        ang, pat = f(result.solution, elevation_deg, n)
        return ang, PatternData(pat.theta, pat.phi, pat.gain_dbi,
                                pat.gain_v_dbi, pat.gain_h_dbi)

    def cut_vertical(self, result, azimuth_deg, n=361):
        from .farfield import cut_vertical as f
        ang, pat = f(result.solution, azimuth_deg, n)
        return ang, PatternData(pat.theta, pat.phi, pat.gain_dbi,
                                pat.gain_v_dbi, pat.gain_h_dbi)

    def geometry(self, result):
        sol = getattr(result, "solution", None)
        if sol is None:
            return None
        mag = np.abs(sol.seg_alpha + 0.5 * sol.seg_beta)
        return Geometry(sol.mesh.a, sol.mesh.b, mag)


# ==========================================================================
#  NEC-2
# ==========================================================================
def _pynec():
    from PyNEC import nec_context
    return nec_context


class Nec2Engine(Engine):
    name = "NEC-2"
    description = ("Původní NEC-2 přes PyNEC. Umí Sommerfeldovu zem — "
                   "spolehlivější u drátů blízko země a vertikálů. "
                   "Nutno doinstalovat: pip install PyNEC")

    @staticmethod
    def is_available() -> bool:
        try:
            import PyNEC  # noqa: F401
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    @staticmethod
    def _seg_of(model: Model, wire: int, pos: float) -> int:
        """Segment NEC odpovídající relativní poloze na drátu.

        NEC budí střed segmentu, AntOpt uzel — u konců drátu se proto bere
        krajní segment, jinak ten, do kterého poloha spadá.
        """
        n = max(1, int(model.wires[wire].nseg))
        if pos <= 1e-9:
            return 1
        if pos >= 1.0 - 1e-9:
            return n
        return int(min(n, max(1, math.floor(pos * n) + 1)))

    def _context(self, model: Model, freq_mhz: Optional[float] = None):
        nec_context = _pynec()
        c = nec_context()
        g = c.get_geometry()
        for i, w in enumerate(model.wires, start=1):
            g.wire(i, max(1, int(w.nseg)), w.x1, w.y1, w.z1,
                   w.x2, w.y2, w.z2, max(w.radius, 1e-6), 1.0, 1.0)
        grounded = model.ground.kind != "free"
        c.geometry_complete(1 if grounded else 0)
        if model.ground.kind == "perfect":
            c.gn_card(1, 0, 0, 0, 0, 0, 0, 0)
        elif model.ground.kind == "real":
            # typ 2 = Sommerfeld-Norton, přesná zem
            c.gn_card(2, 0, model.ground.eps_r, model.ground.sigma, 0, 0, 0, 0)

        sigma = model.conductivity()
        if sigma > 0:
            c.ld_card(5, 0, 0, 0, sigma, 0.0, 0.0)
        for ld in model.loads:
            seg = self._seg_of(model, ld.wire, ld.pos)
            if ld.kind == "RX":
                c.ld_card(4, ld.wire + 1, seg, seg, ld.r, ld.x, 0.0)
            else:
                c.ld_card(0, ld.wire + 1, seg, seg, ld.r,
                          ld.l_uh * 1e-6, ld.c_pf * 1e-12)
        if model.ground_loss_r and grounded:
            for s in model.sources:
                w = model.wires[s.wire]
                zs = w.z1 if s.pos < 0.5 else w.z2
                if abs(zs) < 1e-6:
                    seg = self._seg_of(model, s.wire, s.pos)
                    c.ld_card(4, s.wire + 1, seg, seg, model.ground_loss_r, 0.0, 0.0)

        for s in model.sources:
            seg = self._seg_of(model, s.wire, s.pos)
            vr = s.voltage * math.cos(math.radians(s.phase))
            vi = s.voltage * math.sin(math.radians(s.phase))
            c.ex_card(0, s.wire + 1, seg, 0, vr, vi, 0, 0, 0, 0)
        c.fr_card(0, 1, float(freq_mhz or model.freq_mhz), 0)
        return c

    @staticmethod
    def _impedance(c, executed: bool = False) -> complex:
        """Impedance napájecího bodu. NEC ji vydá až po spuštění výpočtu."""
        if not executed:
            c.xq_card(0)
        ip = c.get_input_parameters(0)
        if ip is None:
            raise RuntimeError("NEC nevrátil parametry napájení — "
                               "zkontroluj zdroj a geometrii.")
        return complex(ip.get_impedance()[0])

    def _grid(self, c, th0: float, dth: float, nth: int,
              ph0: float, dph: float, nph: int, idx: int = 0):
        """RP karta -> (gain_total, gain_V, gain_H) tvaru (nth, nph) v dBi."""
        # výstupní formát 1 = svislá/vodorovná složka (0 by dal hlavní/vedlejší osu)
        c.rp_card(0, nth, nph, 1, 5, 0, 0, th0, ph0, dth, dph, 0, 0)
        rp = c.get_radiation_pattern(idx)
        shape = (nth, nph)

        def clean(arr):
            a = np.asarray(arr, dtype=float).reshape(shape)
            return np.where(a < -900.0, -300.0, a)   # NEC značí "nic" jako -999.99

        return clean(rp.get_gain()), clean(rp.get_gain_vert()), clean(rp.get_gain_horiz())

    # ------------------------------------------------------------------
    def analyse(self, model, n_th=91, n_ph=181, keep_solution=True,
                fb_sector_deg=0.0):
        from .analysis import Result
        grounded = model.ground.kind != "free"
        c = self._context(model)
        th_max = 89.9 if grounded else 179.9
        dth = (th_max - 0.1) / max(1, n_th - 1)
        dph = 360.0 / max(1, n_ph - 1)
        g, gv, gh = self._grid(c, 0.1, dth, n_th, 0.0, dph, n_ph)
        zin = self._impedance(c, executed=True)

        TH = np.radians(0.1 + dth * np.arange(n_th))[:, None] * np.ones((1, n_ph))
        PH = np.radians(dph * np.arange(n_ph))[None, :] * np.ones((n_th, 1))

        i = np.unravel_index(np.argmax(g), g.shape)
        gmax = float(g[i])
        th0 = float(TH[i])
        ph0 = float(PH[i])

        dphi = np.abs(((PH - ph0 + math.pi) % (2 * math.pi)) - math.pi)
        if fb_sector_deg and fb_sector_deg > 0:
            half = math.radians(fb_sector_deg) / 2.0
            rear = dphi >= (math.pi - half)
            gb = float(np.max(g[rear])) if np.any(rear) else float("-inf")
        else:
            back = (ph0 + math.pi) % (2 * math.pi)
            d = np.abs(((PH[i[0], :] - back + math.pi) % (2 * math.pi)) - math.pi)
            gb = float(g[i[0], int(np.argmin(d))])
        fb = gmax - gb
        mask = dphi > math.radians(60.0)
        fs = gmax - float(np.max(g[mask])) if np.any(mask) else float("nan")

        from .gui_helpers import beamwidth_of
        beam_h = beamwidth_of(g[i[0], :], np.degrees(PH[i[0], :]), gmax)
        beam_v = beamwidth_of(g[:, i[1]], np.degrees(TH[:, i[1]]), gmax)

        lin = 10 ** (g / 10.0)
        eff = float(np.sum(lin * np.sin(TH)) * (np.radians(dth)) *
                    (np.radians(dph)) / (4 * math.pi))

        return Result(
            freq_mhz=model.freq_mhz, zin=zin,
            swr=swr_from_z(zin, model.z0),
            gain_dbi=gmax, fb_db=fb, fs_db=fs,
            elevation_deg=90.0 - math.degrees(th0),
            azimuth_deg=math.degrees(ph0),
            beam_h_deg=beam_h, beam_v_deg=beam_v,
            efficiency=eff,
            solution=None, perf=None,
            engine=self.name,
            backend={"model": model.copy()} if keep_solution else None,
        )

    # ------------------------------------------------------------------
    def _model_of(self, result) -> Model:
        b = getattr(result, "backend", None)
        if not b or "model" not in b:
            raise ValueError("Výsledek NEC-2 neobsahuje model pro řezy.")
        return b["model"]

    def cut_azimuth(self, result, elevation_deg, n=361):
        model = self._model_of(result)
        c = self._context(model)
        th = float(np.clip(90.0 - elevation_deg, 0.05, 179.95))
        dph = 360.0 / max(1, n - 1)
        g, gv, gh = self._grid(c, th, 0.0, 1, 0.0, dph, n)
        ang = dph * np.arange(n)
        return ang, PatternData(np.radians(np.full(n, th)), np.radians(ang),
                                g[0], gv[0], gh[0])

    def cut_vertical(self, result, azimuth_deg, n=361):
        model = self._model_of(result)
        grounded = model.ground.kind != "free"
        c = self._context(model)
        span = 180.0 if grounded else 360.0
        ang = np.linspace(0.0, span, n)
        ar = np.radians(ang)
        th = np.degrees(np.arccos(np.clip(np.sin(ar), -1.0, 1.0)))
        front = np.cos(ar) >= 0
        # NEC umí jen pravidelnou mřížku -> dvě roviny (az, az+180)
        nt = n
        th_lo, th_hi = 0.05, 89.95 if grounded else 179.95
        dth = (th_hi - th_lo) / (nt - 1)
        g2, gv2, gh2 = self._grid(c, th_lo, dth, nt, azimuth_deg, 180.0, 2)
        grid_th = th_lo + dth * np.arange(nt)

        def pick(arr):
            out = np.empty(n)
            for k in range(n):
                col = 0 if front[k] else 1
                out[k] = np.interp(np.clip(th[k], th_lo, th_hi), grid_th, arr[:, col])
            return out

        ph = np.where(front, math.radians(azimuth_deg),
                      math.radians(azimuth_deg + 180.0))
        return ang, PatternData(np.radians(th), ph, pick(g2), pick(gv2), pick(gh2))

    def geometry(self, result):
        model = self._model_of(result)
        c = self._context(model)
        c.xq_card(0)
        sc = c.get_structure_currents(0)
        cur = np.abs(np.asarray(sc.get_current()))
        a_list, b_list = [], []
        for w in model.wires:
            nseg = max(1, int(w.nseg))
            for k in range(nseg):
                a_list.append(w.a + (w.b - w.a) * (k / nseg))
                b_list.append(w.a + (w.b - w.a) * ((k + 1) / nseg))
        a = np.array(a_list)
        b = np.array(b_list)
        if len(cur) != len(a):
            cur = np.resize(cur, len(a))
        return Geometry(a, b, cur)


# ==========================================================================
_ENGINES: Dict[str, Engine] = {}


def register(engine: Engine) -> None:
    _ENGINES[engine.name] = engine


register(OwnEngine())
register(Nec2Engine())

_default = "vlastní"


def available_engines() -> List[str]:
    return [n for n, e in _ENGINES.items() if e.is_available()]


def all_engines() -> List[Engine]:
    return list(_ENGINES.values())


def get(name: Optional[str] = None) -> Engine:
    n = name or _default
    e = _ENGINES.get(n)
    if e is None:
        raise ValueError(f"Neznámé jádro: {n}")
    if not e.is_available():
        raise RuntimeError(
            f"Jádro {n} není k dispozici. {e.description}")
    return e


def set_default(name: str) -> None:
    global _default
    if name not in _ENGINES:
        raise ValueError(f"Neznámé jádro: {name}")
    _default = name


def default_name() -> str:
    return _default
