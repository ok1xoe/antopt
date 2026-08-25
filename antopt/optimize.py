"""Optimalizace geometrie antény.

Parametry se definují jako "co se smí měnit" (délka prvku, rozteč, výška,
poloměr, konkrétní souřadnice).  Cílová funkce váží zisk, F/B, PSV a impedanci
na jednom nebo více kmitočtech.  Hledání: reálně kódovaný genetický algoritmus
s následným doladěním Nelder-Meadem.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .model import Model, Wire
from .solver import solve, swr_from_z
from .farfield import performance

PARAM_KINDS = {
    "delka": "Délka drátu (symetricky kolem středu) [m]",
    "delka_konec": "Délka drátu s pevným koncem (drží spoje) [m]",
    "posun_x": "Poloha drátu podél X (rozteč) [m]",
    "posun_y": "Poloha drátu podél Y [m]",
    "vyska": "Výška (posun drátu v Z) [m]",
    "polomer": "Poloměr vodiče [m]",
    "souradnice": "Jedna souřadnice jednoho konce [m]",
    "prvek_delka": "Délka celého prvku i se zúžením [m]",
    "prvek_hrot": "Vysunutí koncové trubky prvku [m]",
    "prvek_x": "Poloha celého prvku podél X [m]",
    "prvek_y": "Poloha celého prvku podél Y [m]",
    "prvek_z": "Poloha celého prvku podél Z [m]",
    "azimut": "Azimut drátu [°]",
    "zenit": "Zenitový úhel drátu [°]",
    "zatez_r": "Zátěž — R [Ω]",
    "zatez_x": "Zátěž — X [Ω]",
    "zatez_l": "Zátěž — L [µH]",
    "zatez_c": "Zátěž — C [pF]",
    "zdroj_u": "Zdroj — napětí [V]",
    "zdroj_faze": "Zdroj — fáze [°]",
    "vyska_vse": "Výška celé antény [m]",
}


@dataclass
class Parameter:
    kind: str
    wires: List[int]
    lo: float
    hi: float
    endpoint: int = 0          # jen pro 'souradnice': 0 = první bod, 1 = druhý
    axis: int = 0              # jen pro 'souradnice': 0=x,1=y,2=z
    label: str = ""
    step: float = 0.0          # nejmenší krok (0 = spojitě)
    link: Optional[int] = None     # index svázaného parametru
    link_factor: float = 1.0       # hodnota = factor * odkaz + offset
    link_offset: float = 0.0

    def describe(self) -> str:
        if self.label:
            return self.label
        w = "+".join(str(i + 1) for i in self.wires)
        return f"{PARAM_KINDS.get(self.kind, self.kind)} – {w}"

    def linked_to(self) -> str:
        if self.link is None:
            return ""
        f, o = self.link_factor, self.link_offset
        txt = f"#{self.link + 1}"
        if f != 1.0:
            txt = f"{f:g}×{txt}"
        if o:
            txt += f" {o:+g}"
        return txt


def expand_values(params: Sequence[Parameter], free_values: Sequence[float]) -> List[float]:
    """Doplní hodnoty svázaných parametrů z jejich odkazů."""
    vals: List[Optional[float]] = [None] * len(params)
    it = iter(free_values)
    for i, p in enumerate(params):
        if p.link is None:
            vals[i] = float(next(it))
    for _ in range(len(params)):
        done = True
        for i, p in enumerate(params):
            if vals[i] is None:
                src = vals[p.link] if p.link is not None and 0 <= p.link < len(params) else None
                if src is None:
                    done = False
                else:
                    vals[i] = p.link_factor * src + p.link_offset
        if done:
            break
    return [0.0 if v is None else float(v) for v in vals]


def free_params(params: Sequence[Parameter]) -> List[int]:
    return [i for i, p in enumerate(params) if p.link is None]


def _element_of(model: Model, idx: Sequence[int]):
    from .geometry_ops import find_elements
    want = set(idx)
    for el in find_elements(model):
        if want & set(el.wires):
            return el
    return None


def read_param(model: Model, p: Parameter) -> float:
    from .geometry_ops import wire_to_polar
    if p.kind == "kmitocet":
        return model.freq_mhz
    if p.kind == "vyska_vse":
        lo, hi = model.bounds()
        return float((lo[2] + hi[2]) / 2.0)
    if p.kind.startswith("zatez_"):
        ld = model.loads[p.wires[0]]
        return {"zatez_r": ld.r, "zatez_x": ld.x,
                "zatez_l": ld.l_uh, "zatez_c": ld.c_pf}[p.kind]
    if p.kind.startswith("zdroj_"):
        s_ = model.sources[p.wires[0]]
        return s_.voltage if p.kind == "zdroj_u" else s_.phase
    if p.kind.startswith("prvek_"):
        el = _element_of(model, p.wires)
        if el is None:
            return 0.0
        if p.kind == "prvek_delka":
            return el.length
        if p.kind == "prvek_hrot":
            from .geometry_ops import element_tip_length
            return element_tip_length(model, el)
        return float(el.center[{"prvek_x": 0, "prvek_y": 1, "prvek_z": 2}[p.kind]])
    w = model.wires[p.wires[0]]
    if p.kind == "azimut":
        return wire_to_polar(w)[1]
    if p.kind == "zenit":
        return wire_to_polar(w)[2]
    if p.kind in ("delka", "delka_konec"):
        return w.length
    if p.kind == "posun_x":
        return float(w.center()[0])
    if p.kind == "posun_y":
        return float(w.center()[1])
    if p.kind == "vyska":
        return float(w.center()[2])
    if p.kind == "polomer":
        return w.radius
    if p.kind == "souradnice":
        return [w.x1, w.y1, w.z1][p.axis] if p.endpoint == 0 else [w.x2, w.y2, w.z2][p.axis]
    raise ValueError(p.kind)


def apply_param(model: Model, p: Parameter, value: float) -> None:
    from .geometry_ops import (set_element_length, set_element_position,
                               wire_to_polar, polar_to_wire, move)
    if p.kind == "kmitocet":
        model.freq_mhz = max(0.001, float(value))
        return
    if p.kind == "vyska_vse":
        cur = read_param(model, p)
        move(model, dz=value - cur)
        return
    if p.kind.startswith("zatez_"):
        for li in p.wires:
            if 0 <= li < len(model.loads):
                ld = model.loads[li]
                setattr(ld, {"zatez_r": "r", "zatez_x": "x",
                             "zatez_l": "l_uh", "zatez_c": "c_pf"}[p.kind],
                        float(value))
        return
    if p.kind.startswith("zdroj_"):
        for si in p.wires:
            if 0 <= si < len(model.sources):
                s_ = model.sources[si]
                if p.kind == "zdroj_u":
                    s_.voltage = float(value)
                else:
                    s_.phase = float(value)
        return
    if p.kind.startswith("prvek_"):
        el = _element_of(model, p.wires)
        if el is None:
            return
        if p.kind == "prvek_delka":
            set_element_length(model, el, max(1e-6, float(value)))
        elif p.kind == "prvek_hrot":
            from .geometry_ops import set_element_tip
            set_element_tip(model, el, max(1e-4, float(value)))
        else:
            set_element_position(model, el,
                                 {"prvek_x": "x", "prvek_y": "y", "prvek_z": "z"}[p.kind],
                                 float(value))
        return
    if p.kind in ("azimut", "zenit"):
        for wi in p.wires:
            w = model.wires[wi]
            L, az, ze = wire_to_polar(w)
            if p.kind == "azimut":
                az = float(value)
            else:
                ze = float(value)
            nw = polar_to_wire(w.a, L, az, ze, w.radius, w.nseg)
            w.x2, w.y2, w.z2 = nw.x2, nw.y2, nw.z2
        return
    if p.kind == "delka_konec":
        for wi in p.wires:
            w = model.wires[wi]
            L = w.length
            if L <= 0:
                continue
            anchor = w.a if p.endpoint == 0 else w.b
            far = w.b if p.endpoint == 0 else w.a
            u = (far - anchor) / L
            new = anchor + u * max(1e-6, float(value))
            if p.endpoint == 0:
                w.x2, w.y2, w.z2 = (float(v) for v in new)
            else:
                w.x1, w.y1, w.z1 = (float(v) for v in new)
        return
    for wi in p.wires:
        w = model.wires[wi]
        if p.kind == "delka":
            c = w.center()
            L = w.length
            if L <= 0:
                continue
            u = (w.b - w.a) / L
            a = c - u * value / 2.0
            b = c + u * value / 2.0
            w.x1, w.y1, w.z1 = map(float, a)
            w.x2, w.y2, w.z2 = map(float, b)
        elif p.kind in ("posun_x", "posun_y", "vyska"):
            ax = {"posun_x": 0, "posun_y": 1, "vyska": 2}[p.kind]
            c = w.center()[ax]
            d = value - c
            if ax == 0:
                w.x1 += d; w.x2 += d
            elif ax == 1:
                w.y1 += d; w.y2 += d
            else:
                w.z1 += d; w.z2 += d
        elif p.kind == "polomer":
            w.radius = max(1e-5, value)
        elif p.kind == "souradnice":
            names = [("x1", "y1", "z1"), ("x2", "y2", "z2")][p.endpoint]
            setattr(w, names[p.axis], float(value))


def build_candidate(base: Model, params: Sequence[Parameter],
                    values: Sequence[float]) -> Model:
    m = base.copy()
    for p, v in zip(params, values):
        apply_param(m, p, float(v))
    return m


# --------------------------------------------------------------------------
@dataclass
class Objective:
    """Váhy cílové funkce. Vyšší váha = důležitější."""
    freqs_mhz: List[float] = field(default_factory=list)   # prázdné = kmitočet modelu
    w_gain: float = 1.0
    w_fb: float = 0.5
    w_fs: float = 0.0
    w_swr: float = 2.0
    w_r: float = 0.0
    w_x: float = 0.0
    w_elev: float = 0.0         # kladné = tlač úhel vyzařování dolů
    w_current: float = 0.0      # kladné = maximalizuj proud v zadaném bodě
    current_at: Optional[tuple] = None   # (drát, poloha 0..1)
    target_swr: float = 1.5
    target_r: float = 50.0
    target_x: float = 0.0
    fb_cap: float = 25.0        # nad tímhle F/B už se nevyplácí honit další dB
    fb_sector_deg: float = 0.0  # >0 = F/B v zadním výseku jako v MMANA
    engine: Optional[str] = None  # None = výchozí jádro
    elevation_deg: Optional[float] = None    # None = maximum přes celý poloprostor
    n_th: int = 37
    n_ph: int = 73

    def freq_list(self, model: Model) -> List[float]:
        return list(self.freqs_mhz) if self.freqs_mhz else [model.freq_mhz]


@dataclass
class Evaluation:
    cost: float
    detail: List[dict]

    def summary(self) -> str:
        parts = []
        for d in self.detail:
            txt = (f"{d['f']:.3f} MHz: G={d['gain']:.2f} dBi  F/B={d['fb']:.1f} dB  "
                   f"PSV={d['swr']:.2f}  Z={d['r']:.1f}{d['x']:+.1f}j")
            if np.isfinite(d.get("elev", float("nan"))):
                txt += f"  el={d['elev']:.1f}°"
            parts.append(txt)
        return "\n".join(parts)


def evaluate(model: Model, obj: Objective) -> Evaluation:
    cost = 0.0
    detail = []
    work = model.copy()
    use_own = obj.engine in (None, "vlastní")
    for f in obj.freq_list(model):
        work.freq_mhz = f
        sol = None
        try:
            if use_own:
                sol = solve(work)
                z = sol.zin
            else:
                from . import engines
                res = engines.get(obj.engine).analyse(
                    work, n_th=obj.n_th, n_ph=obj.n_ph, keep_solution=False,
                    fb_sector_deg=obj.fb_sector_deg)
                z = res.zin
        except Exception:
            return Evaluation(1e6, [])
        if not np.isfinite(z.real) or not np.isfinite(z.imag):
            return Evaluation(1e6, [])
        s = swr_from_z(z, work.z0)
        need_pattern = (obj.w_gain != 0 or obj.w_fb != 0 or obj.w_fs != 0
                        or obj.w_elev != 0)
        elev = float("nan")
        if not use_own:
            gain, fb, fs = res.gain_dbi, res.fb_db, res.fs_db
            elev = res.elevation_deg
        elif need_pattern:
            p = performance(sol, n_th=obj.n_th, n_ph=obj.n_ph,
                            fb_sector_deg=obj.fb_sector_deg)
            gain, fb, fs = p.gain_dbi, p.fb_db, p.fs_db
            elev = p.elevation_deg
            if obj.elevation_deg is not None:
                from .farfield import far_field
                th = np.radians(90.0 - obj.elevation_deg)
                ph = np.radians(np.linspace(0, 360, obj.n_ph))
                pat = far_field(sol, np.full_like(ph, th), ph)
                gain = float(np.max(pat.gain_dbi))
        else:
            gain = fb = fs = 0.0

        cur = float("nan")
        if obj.w_current and obj.current_at and sol is not None:
            try:
                from .mesh import node_for_position
                wi, pos = obj.current_at
                node = node_for_position(sol.mesh, work, int(wi), float(pos))
                bi = sol.mesh.basis_at_node(node)
                cur = float(abs(sol.currents[bi])) if bi >= 0 else float("nan")
            except Exception:
                cur = float("nan")

        c = 0.0
        c -= obj.w_gain * gain
        c -= obj.w_fb * min(fb, obj.fb_cap)
        c -= obj.w_fs * min(fs, obj.fb_cap) if np.isfinite(fs) else 0.0
        if obj.w_swr:
            over = max(0.0, s - obj.target_swr)
            c += obj.w_swr * (over ** 2) * 4.0
        if obj.w_r:
            c += obj.w_r * abs(z.real - obj.target_r) / 10.0
        if obj.w_x:
            c += obj.w_x * abs(z.imag - obj.target_x) / 10.0
        if obj.w_elev and np.isfinite(elev):
            c += obj.w_elev * elev / 10.0
        if obj.w_current and np.isfinite(cur):
            c -= obj.w_current * 20.0 * math.log10(max(cur, 1e-12))
        cost += c
        detail.append({"f": f, "gain": gain, "fb": fb, "fs": fs,
                       "swr": s, "r": z.real, "x": z.imag,
                       "elev": elev, "cur": cur})
    return Evaluation(cost / max(1, len(detail)), detail)


# --------------------------------------------------------------------------
@dataclass
class OptResult:
    model: Model
    values: List[float]
    cost: float
    history: List[float]
    evaluation: Evaluation
    iterations: int


def optimize(base: Model, params: Sequence[Parameter], obj: Objective,
             pop_size: int = 24, generations: int = 30,
             polish: bool = True, seed: int = 0,
             progress: Optional[Callable[[int, int, float, str], bool]] = None
             ) -> OptResult:
    """Genetický algoritmus + Nelder-Mead.

    ``progress(gen, total, best_cost, text)`` může vrátit False pro přerušení.
    """
    rng = np.random.default_rng(seed)
    fidx = free_params(params)
    n = len(fidx)
    if not params or n == 0:
        vals = expand_values(params, [])
        m = build_candidate(base, params, vals) if params else base
        ev = evaluate(m, obj)
        return OptResult(m, vals, ev.cost, [ev.cost], ev, 0)

    lo = np.array([params[i].lo for i in fidx], dtype=float)
    hi = np.array([params[i].hi for i in fidx], dtype=float)
    x0 = np.array([np.clip(read_param(base, params[i]), params[i].lo, params[i].hi)
                   for i in fidx])
    steps = np.array([params[i].step for i in fidx], dtype=float)

    def quantise(x: np.ndarray) -> np.ndarray:
        out = np.array(x, dtype=float)
        m = steps > 0
        if np.any(m):
            out[m] = np.round(out[m] / steps[m]) * steps[m]
        return np.clip(out, lo, hi)

    cache: dict = {}

    def cost_of(x: np.ndarray) -> float:
        x = quantise(x)
        key = tuple(np.round(x, 8))
        if key in cache:
            return cache[key]
        m = build_candidate(base, params, expand_values(params, x))
        c = evaluate(m, obj).cost
        cache[key] = c
        return c

    # počáteční populace: výchozí bod + latinský hyperkrychlový vzorek
    pop = np.empty((pop_size, n))
    pop[0] = x0
    for j in range(n):
        vals = (np.arange(pop_size - 1) + rng.random(pop_size - 1)) / (pop_size - 1)
        rng.shuffle(vals)
        pop[1:, j] = lo[j] + vals * (hi[j] - lo[j])
    fit = np.array([cost_of(p) for p in pop])

    history = [float(fit.min())]
    span = hi - lo
    total = generations

    for gen in range(generations):
        order = np.argsort(fit)
        pop, fit = pop[order], fit[order]
        elite = max(2, pop_size // 8)
        newpop = [pop[i].copy() for i in range(elite)]
        sigma = 0.25 * span * (1.0 - 0.85 * gen / max(1, generations - 1))
        while len(newpop) < pop_size:
            # turnajový výběr
            cand = rng.integers(0, pop_size, 4)
            pa = pop[cand[np.argmin(fit[cand])]]
            cand = rng.integers(0, pop_size, 4)
            pb = pop[cand[np.argmin(fit[cand])]]
            # BLX-alfa křížení
            alpha = 0.4
            cmin = np.minimum(pa, pb)
            cmax = np.maximum(pa, pb)
            d = cmax - cmin
            child = rng.uniform(cmin - alpha * d, cmax + alpha * d)
            # mutace
            mask = rng.random(n) < 0.3
            child = child + mask * rng.normal(0, sigma)
            newpop.append(np.clip(child, lo, hi))
        pop = np.array(newpop)
        fit = np.array([cost_of(p) for p in pop])
        best = float(fit.min())
        history.append(best)
        if progress is not None:
            txt = f"generace {gen + 1}/{generations}, nejlepší cena {best:.3f}"
            if progress(gen + 1, total, best, txt) is False:
                break

    ib = int(np.argmin(fit))
    xbest = quantise(pop[ib].copy())
    iters = len(cache)

    if polish:
        from scipy.optimize import minimize
        step = 0.03 * span
        res = minimize(lambda x: cost_of(np.clip(x, lo, hi)), xbest,
                       method="Nelder-Mead",
                       options={"xatol": 1e-5, "fatol": 1e-4,
                                "maxiter": 60 * n, "initial_simplex": None})
        if res.fun < fit[ib]:
            xbest = quantise(res.x)
        history.append(float(min(res.fun, fit[ib])))
        iters = len(cache)
        if progress is not None:
            progress(total, total, float(history[-1]), "doladění hotovo")

    full = expand_values(params, xbest)
    best_model = build_candidate(base, params, full)
    ev = evaluate(best_model, obj)
    return OptResult(model=best_model, values=[float(v) for v in full],
                     cost=ev.cost, history=history, evaluation=ev,
                     iterations=iters)


# --------------------------------------------------------------------------
def _shared_endpoint(model: Model, i: int, tol: float) -> Optional[int]:
    """Vrátí konec drátu i (0/1), kterým se dotýká jiného drátu, jinak None."""
    wi = model.wires[i]
    for k, pt in ((0, wi.a), (1, wi.b)):
        for j, wj in enumerate(model.wires):
            if j == i:
                continue
            if min(np.linalg.norm(pt - wj.a), np.linalg.norm(pt - wj.b)) < tol:
                return k
    return None


def suggest_parameters(model: Model, span_pct: float = 12.0) -> List[Parameter]:
    """Rozumná výchozí sada parametrů: délky drátů + rozteče prvků.

    Dráty, které se někde stýkají, dostanou délku s **pevným spojem**, aby
    optimalizace geometrii nerozpojila. Stejně dlouhá ramena vycházející
    ze stejného bodu se automaticky sváží, takže zůstanou symetrická.

    **Zúžený (teleskopický) prvek se bere jako jeden celek.** Šestisekční
    prvek Yagi tedy dostane jednu délku, ne šest — jinak by optimalizace
    posouvala jednotlivé trubky nezávisle a prvek by se rozpadl.
    """
    from .geometry_ops import find_elements

    out: List[Parameter] = []
    tol = max(1e-9, model.wavelength * 1e-6)
    lam = model.wavelength

    try:
        elements = find_elements(model)
    except Exception:
        elements = []
    composed = {}                      # index drátu -> prvek, pokud je složený
    for el in elements:
        if len(el.wires) >= 2:
            for i in el.wires:
                composed[i] = el

    # --- složené prvky: jedna délka a jedna poloha na celý prvek
    el_x: List[Tuple[float, int]] = []
    seen = set()
    for el in elements:
        if len(el.wires) < 2 or el.wires[0] in seen:
            continue
        seen.update(el.wires)
        L = el.length
        n_sec = len(el.wires) // 2
        out.append(Parameter("prvek_delka", list(el.wires),
                             L * (1 - span_pct / 100), L * (1 + span_pct / 100),
                             label=f"Délka prvku na x={el.position_x:+.3f} m"
                                   f" ({n_sec} sekce)"))
        el_x.append((el.position_x, len(out) - 1))

    # --- jednotlivé dráty (nezúžené prvky)
    anchors: List[Optional[int]] = [None] * len(model.wires)
    idx_of_wire: dict = {}
    for i, w in enumerate(model.wires):
        if i in composed:
            continue
        L = w.length
        anc = _shared_endpoint(model, i, tol)
        anchors[i] = anc
        idx_of_wire[i] = len(out)
        if anc is None:
            out.append(Parameter("delka", [i], L * (1 - span_pct / 100),
                                 L * (1 + span_pct / 100),
                                 label=f"Délka drátu {i + 1}"))
        else:
            out.append(Parameter("delka_konec", [i], L * (1 - span_pct / 100),
                                 L * (1 + span_pct / 100), endpoint=anc,
                                 label=f"Délka drátu {i + 1} (spoj drží)"))

    # svázat stejně dlouhá ramena vycházející ze společného bodu
    for i in idx_of_wire:
        pi_idx = idx_of_wire[i]
        if anchors[i] is None or out[pi_idx].link is not None:
            continue
        pi = model.wires[i].a if anchors[i] == 0 else model.wires[i].b
        for j in idx_of_wire:
            if j <= i:
                continue
            pj_idx = idx_of_wire[j]
            if anchors[j] is None or out[pj_idx].link is not None:
                continue
            pj = model.wires[j].a if anchors[j] == 0 else model.wires[j].b
            if (np.linalg.norm(pi - pj) < tol
                    and abs(model.wires[i].length - model.wires[j].length) < tol * 1e3):
                out[pj_idx].link = pi_idx
                out[pj_idx].label = f"Délka drátu {j + 1} (= drát {i + 1})"

    # --- rozteče podél ráhna
    spots: List[Tuple[float, str, List[int]]] = [
        (x, "prvek_x", list(out[k].wires)) for x, k in el_x]
    spots += [(float(model.wires[i].center()[0]), "posun_x", [i])
              for i in idx_of_wire]
    if len({round(x, 6) for x, _, _ in spots}) > 1:
        spots.sort(key=lambda s: s[0])
        xs = [s[0] for s in spots]
        for rank, (x, kind, wires) in enumerate(spots):
            if abs(x) < 1e-9:
                continue                      # zářič drží počátek
            gaps = []
            if rank > 0:
                gaps.append(abs(x - xs[rank - 1]))
            if rank < len(xs) - 1:
                gaps.append(abs(xs[rank + 1] - x))
            span = min(0.06 * lam, 0.45 * min(gaps)) if gaps else 0.06 * lam
            name = (f"Poloha X prvku na x={x:+.3f} m" if kind == "prvek_x"
                    else f"Poloha X drátu {wires[0] + 1}")
            out.append(Parameter(kind, wires, x - span, x + span, label=name))
    return out
