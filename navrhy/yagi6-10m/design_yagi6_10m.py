"""Návrh 6prvkové Yagi na 10 m, ráhno 7,5 m, úsek 28,0-28,8 MHz."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from antopt.model import Model, Wire, Source, Ground, C0
from antopt.solver import solve
from antopt.analysis import analyse, sweep, bandwidth
from antopt.optimize import Parameter, Objective, optimize, evaluate

FC = 28.4
LAM = C0/(FC*1e6)
RAD = 0.008          # AL trubka 16x1.5 -> polomer 8 mm
NSEG = 17
BOOM = 7.5

# vychozi geometrie: zaric v x=0, reflektor vzadu, direktory dopredu
# (x, delka)
START = [
    (-1.05, 5.28),   # reflektor
    ( 0.00, 5.02),   # zaric
    ( 0.85, 4.90),   # D1
    ( 2.32, 4.82),   # D2
    ( 4.32, 4.75),   # D3
    ( 6.45, 4.66),   # D4
]
NAMES = ["Reflektor","Zářič","D1","D2","D3","D4"]

def build(els, f=FC, ground=None, h=0.0, z0=50.0):
    m = Model(name="Yagi 6 prvků 10 m", freq_mhz=f, material="hliník", z0=z0)
    m.wires = [Wire(x, -L/2, h, x, L/2, h, radius=RAD, nseg=NSEG) for x,L in els]
    m.sources = [Source(1, 0.5, 1.0)]
    m.ground = Ground("free") if ground is None else ground
    return m

def els_of(model):
    return [(float(w.center()[0]), w.length) for w in model.wires]

print(f"lambda = {LAM:.3f} m,  rahno = {BOOM} m = {BOOM/LAM:.3f} lambda")
m0 = build(START)
r0 = analyse(m0)
print(f"VYCHOZI (volny prostor, {FC} MHz): Z={r0.zin.real:.1f}{r0.zin.imag:+.1f}j  "
      f"G={r0.gain_dbi:.2f} dBi  F/B={r0.fb_db:.1f} dB  F/S={r0.fs_db:.1f} dB")

# ---- optimalizace: 6 delek + 3 polohy direktoru (R a D4 drzi rahno 7,5 m)
# meze delek se prekryvaji jen mirne -> vynuti klesajici taper smerem k D4
LEN_BOUNDS = [(5.15, 5.60), (4.90, 5.20), (4.74, 5.02),
              (4.66, 4.94), (4.58, 4.86), (4.46, 4.78)]
POS_BOUNDS = {2: (0.62, 1.28), 3: (1.75, 2.95), 4: (3.40, 5.00)}
params = []
for i,(x,L) in enumerate(START):
    lo,hi = LEN_BOUNDS[i]
    params.append(Parameter("delka", [i], lo, hi, label=f"délka {NAMES[i]}"))
for i,xr in POS_BOUNDS.items():
    params.append(Parameter("posun_x", [i], xr[0], xr[1], label=f"poloha {NAMES[i]}"))

obj = Objective(freqs_mhz=[28.0, 28.4, 28.8],
                w_gain=1.0, w_fb=0.22, fb_cap=22.0,
                w_swr=2.0, target_swr=1.4,
                n_th=31, n_ph=61)
# PSV se pri optimalizaci vztahuje k 28 ohm - realny napajeci bod Yagi
for p in params: pass
base = build(START); base.z0 = 28.0

print("\nstart cost:", round(evaluate(base,obj).cost,3))
t=time.time()
res = optimize(base, params, obj, pop_size=30, generations=34, seed=17, polish=True,
               progress=lambda g,t_,c,txt: (print(f"  {txt}") if g%5==0 or g==t_ else None))
print(f"optimalizace {time.time()-t:.0f} s, {res.iterations} vyhodnoceni")
best = res.model
els = els_of(best)
print()
for n,(x,L) in zip(NAMES, els):
    print(f"  {n:10s} x={x:7.3f} m   délka={L:7.3f} m")
print(f"  rahno = {max(x for x,_ in els)-min(x for x,_ in els):.3f} m")
print()
print(res.evaluation.summary())
np.save("/tmp/yagi6_els.npy", np.array(els))
best.z0 = 50.0
best.save("/tmp/yagi6_free.json")
print("ULOZENO")
