"""Optimalizace geometrie antény.

Parametry se definují jako "co se smí měnit" (délka prvku, rozteč, výška,
poloměr, konkrétní souřadnice).  Cílová funkce váží zisk, F/B, PSV a impedanci
na jednom nebo více kmitočtech.  Hledání: reálně kódovaný genetický algoritmus
s následným doladěním Nelder-Meadem.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Callable, List, Optional, Sequence

import numpy as np

from .model import Model, Wire
from .solver import solve, swr_from_z
from .farfield import performance

PARAM_KINDS = {
    "delka": "Délka prvku (symetricky kolem středu) [m]",
    "posun_x": "Poloha prvku podél X (rozteč) [m]",
    "posun_y": "Poloha prvku podél Y [m]",
    "vyska": "Výška (posun celého drátu v Z) [m]",
    "polomer": "Poloměr vodiče [m]",
    "souradnice": "Jedna souřadnice jednoho konce [m]",
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

    def describe(self) -> str:
        if self.label:
            return self.label
        w = "+".join(str(i + 1) for i in self.wires)
        return f"{PARAM_KINDS.get(self.kind, self.kind)} – drát {w}"


def read_param(model: Model, p: Parameter) -> float:
    w = model.wires[p.wires[0]]
    if p.kind == "delka":
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
    target_swr: float = 1.5
    target_r: float = 50.0
    target_x: float = 0.0
    fb_cap: float = 25.0        # nad tímhle F/B už se nevyplácí honit další dB
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
            parts.append(
                f"{d['f']:.3f} MHz: G={d['gain']:.2f} dBi  F/B={d['fb']:.1f} dB  "
                f"PSV={d['swr']:.2f}  Z={d['r']:.1f}{d['x']:+.1f}j"
            )
        return "\n".join(parts)


def evaluate(model: Model, obj: Objective) -> Evaluation:
    cost = 0.0
    detail = []
    work = model.copy()
    for f in obj.freq_list(model):
        work.freq_mhz = f
        try:
            sol = solve(work)
        except Exception:
            return Evaluation(1e6, [])
        z = sol.zin
        if not np.isfinite(z.real) or not np.isfinite(z.imag):
            return Evaluation(1e6, [])
        s = swr_from_z(z, work.z0)
        need_pattern = obj.w_gain != 0 or obj.w_fb != 0 or obj.w_fs != 0
        if need_pattern:
            p = performance(sol, n_th=obj.n_th, n_ph=obj.n_ph)
            gain, fb, fs = p.gain_dbi, p.fb_db, p.fs_db
            if obj.elevation_deg is not None:
                from .farfield import far_field
                th = np.radians(90.0 - obj.elevation_deg)
                ph = np.radians(np.linspace(0, 360, obj.n_ph))
                pat = far_field(sol, np.full_like(ph, th), ph)
                gain = float(np.max(pat.gain_dbi))
        else:
            gain = fb = fs = 0.0

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
        cost += c
        detail.append({"f": f, "gain": gain, "fb": fb, "fs": fs,
                       "swr": s, "r": z.real, "x": z.imag})
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
    n = len(params)
    if n == 0:
        ev = evaluate(base, obj)
        return OptResult(base, [], ev.cost, [ev.cost], ev, 0)

    lo = np.array([p.lo for p in params], dtype=float)
    hi = np.array([p.hi for p in params], dtype=float)
    x0 = np.array([np.clip(read_param(base, p), p.lo, p.hi) for p in params])

    cache: dict = {}

    def cost_of(x: np.ndarray) -> float:
        key = tuple(np.round(x, 6))
        if key in cache:
            return cache[key]
        m = build_candidate(base, params, x)
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
    xbest = pop[ib].copy()
    iters = len(cache)

    if polish:
        from scipy.optimize import minimize
        step = 0.03 * span
        res = minimize(lambda x: cost_of(np.clip(x, lo, hi)), xbest,
                       method="Nelder-Mead",
                       options={"xatol": 1e-5, "fatol": 1e-4,
                                "maxiter": 60 * n, "initial_simplex": None})
        if res.fun < fit[ib]:
            xbest = np.clip(res.x, lo, hi)
        history.append(float(min(res.fun, fit[ib])))
        iters = len(cache)
        if progress is not None:
            progress(total, total, float(history[-1]), "doladění hotovo")

    best_model = build_candidate(base, params, xbest)
    ev = evaluate(best_model, obj)
    return OptResult(model=best_model, values=[float(v) for v in xbest],
                     cost=ev.cost, history=history, evaluation=ev,
                     iterations=iters)


# --------------------------------------------------------------------------
def suggest_parameters(model: Model, span_pct: float = 12.0) -> List[Parameter]:
    """Rozumná výchozí sada parametrů: délky všech prvků + rozteče (mimo první)."""
    out: List[Parameter] = []
    for i, w in enumerate(model.wires):
        L = w.length
        out.append(Parameter("delka", [i], L * (1 - span_pct / 100),
                             L * (1 + span_pct / 100),
                             label=f"Délka drátu {i + 1}"))
    xs = np.array([float(w.center()[0]) for w in model.wires])
    if len(np.unique(np.round(xs, 6))) > 1:
        lam = model.wavelength
        order = np.argsort(xs)
        for rank, i in enumerate(order):
            x = xs[i]
            if abs(x) < 1e-9:
                continue                      # zářič drží počátek
            # nesmí přeskočit sousední prvek
            gaps = []
            if rank > 0:
                gaps.append(abs(x - xs[order[rank - 1]]))
            if rank < len(order) - 1:
                gaps.append(abs(xs[order[rank + 1]] - x))
            span = min(0.06 * lam, 0.45 * min(gaps)) if gaps else 0.06 * lam
            out.append(Parameter("posun_x", [int(i)], x - span, x + span,
                                 label=f"Poloha X drátu {i + 1}"))
    return out
