"""Dokončení návrhu 6el Yagi na 10 m: napájení, kontrola NEC, chování nad zemí."""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from antopt.model import Model, Wire, Source, Ground, C0
from antopt.solver import solve, swr_from_z
from antopt.analysis import analyse, sweep, bandwidth
from antopt.farfield import azimuth_cut, elevation_cut, performance
from antopt.match import design_hairpin, tune_driven_for_hairpin, swr_with_hairpin, matched_impedance
from antopt.fileio import to_maa, to_nec

FC = 28.4
BAND = (28.0, 28.8)
NAMES = ["Reflektor", "Zářič", "D1", "D2", "D3", "D4"]
RAD = 0.008
H = 10.0

best = Model.load("/tmp/yagi6_free.json")
best.z0 = 50.0
els = [(float(w.center()[0]), w.length) for w in best.wires]

print("=" * 78)
print("GEOMETRIE PO OPTIMALIZACI (volný prostor)")
x0 = min(x for x, _ in els)
for n, (x, L) in zip(NAMES, els):
    print(f"  {n:10s} od reflektoru {x - x0:6.3f} m   délka {L:6.3f} m   "
          f"(poloviny 2× {L/2:.3f} m)")
boom = max(x for x, _ in els) - x0
print(f"  ráhno {boom:.3f} m")

# ---------------------------------------------------------------- napájení
print()
print("=" * 78)
print("NAPÁJENÍ")
z_res = analyse(best).zin
print(f"  zářič v rezonanční délce: Z = {z_res.real:.1f} {z_res.imag:+.1f} j Ω  "
      f"→ PSV(50) = {swr_from_z(z_res, 50):.2f}")

tuned, z_t = tune_driven_for_hairpin(best, 1, z0=50.0)
dl = tuned.wires[1].length - best.wires[1].length
print(f"  zkrácení zářiče o {abs(dl) * 1000:.0f} mm → Z = {z_t.real:.1f} {z_t.imag:+.1f} j Ω")
hp = design_hairpin(z_t, FC, 50.0, spacing_mm=60.0, diameter_mm=10.0)
print("  " + hp.report().replace("\n", "\n  "))

freqs = np.linspace(BAND[0], BAND[1], 17)
rows = swr_with_hairpin(tuned, hp, freqs, 50.0)
print("\n  f [MHz]   Z po vlásence        PSV")
for f, z, s in rows:
    print(f"  {f:7.3f}   {z.real:6.1f} {z.imag:+6.1f} j     {s:.2f}")
swr_max = max(s for _, _, s in rows)
print(f"  → PSV max v úseku {BAND[0]}–{BAND[1]} MHz: {swr_max:.2f}")

# ---------------------------------------------------------- výkon v pásmu
print()
print("=" * 78)
print("PARAMETRY VE VOLNÉM PROSTORU (zářič zkrácený pro vlásenku)")
perf_rows = []
for f in [28.0, 28.2, 28.4, 28.6, 28.8]:
    w = tuned.copy(); w.freq_mhz = f
    r = analyse(w)
    zm = matched_impedance(r.zin, hp.z_line, hp.length_m, f)
    perf_rows.append((f, r.gain_dbi, r.fb_db, r.fs_db, zm, swr_from_z(zm, 50)))
    print(f"  {f:.2f} MHz  G={r.gain_dbi:5.2f} dBi  F/B={r.fb_db:5.1f} dB  "
          f"F/S={r.fs_db:5.1f} dB  PSV={swr_from_z(zm, 50):.2f}")

# ---------------------------------------------------------------- vs NEC-2
print()
print("=" * 78)
print("KŘÍŽOVÁ KONTROLA PROTI NEC-2")
try:
    from PyNEC import nec_context
    tels = [(float(w.center()[0]), w.length) for w in tuned.wires]
    for f in [28.0, 28.4, 28.8]:
        w = tuned.copy(); w.freq_mhz = f
        r = analyse(w)
        c = nec_context(); g = c.get_geometry()
        for i, (x, L) in enumerate(tels):
            g.wire(i + 1, 17, x, -L / 2, 0, x, L / 2, 0, RAD, 1, 1)
        c.geometry_complete(0)
        c.ex_card(0, 2, 9, 0, 1.0, 0, 0, 0, 0, 0)
        c.fr_card(0, 1, f, 0)
        c.rp_card(0, 181, 361, 0, 5, 0, 0, 0, 0, 1.0, 1.0, 0, 0)
        zn = c.get_input_parameters(0).get_impedance()[0]
        gn = c.get_radiation_pattern(0).get_gain().max()
        print(f"  {f:.2f} MHz  AntOpt Z={r.zin.real:6.2f}{r.zin.imag:+6.2f}j G={r.gain_dbi:5.2f}"
              f"  |  NEC-2 Z={zn.real:6.2f}{zn.imag:+6.2f}j G={gn:5.2f}"
              f"  |  ΔR={abs(r.zin.real-zn.real):.2f} Ω ΔX={abs(r.zin.imag-zn.imag):.2f} Ω "
              f"ΔG={abs(r.gain_dbi-gn):.2f} dB")
except ImportError:
    print("  PyNEC není k dispozici")

# ------------------------------------------------------------- nad zemí
print()
print("=" * 78)
print(f"NAD PRŮMĚRNOU ZEMÍ VE VÝŠCE {H:.0f} m")
gnd = tuned.copy()
gnd.ground = Ground.from_name("průměrná")
for w in gnd.wires:
    w.z1 = w.z2 = H
gnd.name = f"Yagi 6 prvků 10 m, {H:.0f} m nad zemí"
gnd.freq_mhz = FC
rg = analyse(gnd)
print(f"  {FC} MHz  G={rg.gain_dbi:.2f} dBi  elevace {rg.elevation_deg:.1f}°  "
      f"F/B={rg.fb_db:.1f} dB  šířka svazku H/V {rg.beam_h_deg:.0f}°/{rg.beam_v_deg:.0f}°")
for f in [28.0, 28.4, 28.8]:
    w = gnd.copy(); w.freq_mhz = f
    r = analyse(w)
    print(f"  {f:.2f} MHz  G={r.gain_dbi:5.2f} dBi  elevace {r.elevation_deg:4.1f}°  "
          f"F/B={r.fb_db:5.1f} dB")

# ------------------------------------------------------------------ grafy
sol_free = analyse(tuned).solution
sol_gnd = rg.solution

fig = plt.figure(figsize=(13.5, 8.6), dpi=110)
fig.suptitle(f"Yagi 6 prvků, 10 m • ráhno {boom:.2f} m • AL trubka 16×1,5 • "
             f"{BAND[0]}–{BAND[1]} MHz", fontsize=13, y=0.985)

# PSV
ax = fig.add_subplot(2, 3, 1)
fs = np.linspace(27.8, 29.2, 57)
sw = swr_with_hairpin(tuned, hp, fs, 50.0)
ax.plot([r[0] for r in sw], [r[2] for r in sw], color="#1565c0", lw=1.8)
ax.axvspan(BAND[0], BAND[1], color="#4caf50", alpha=0.12)
ax.axhline(1.5, color="#bbb", ls="--", lw=0.8)
ax.axhline(2.0, color="#bbb", ls="--", lw=0.8)
ax.set_ylim(1, 3.2); ax.set_xlabel("f [MHz]"); ax.set_ylabel("PSV (50 Ω, s vlásenkou)")
ax.grid(alpha=0.3); ax.set_title("PSV", fontsize=10)

# zisk a F/B pres pasmo (volny prostor)
ax = fig.add_subplot(2, 3, 2)
fs2 = np.linspace(27.9, 29.0, 23)
gs, fbs = [], []
for f in fs2:
    w = tuned.copy(); w.freq_mhz = float(f)
    r = analyse(w, n_th=46, n_ph=91, keep_solution=False)
    gs.append(r.gain_dbi); fbs.append(r.fb_db)
ax.plot(fs2, gs, color="#6a1b9a", label="zisk [dBi]")
ax.plot(fs2, fbs, color="#ef6c00", label="F/B [dB]")
ax.axvspan(BAND[0], BAND[1], color="#4caf50", alpha=0.12)
ax.set_xlabel("f [MHz]"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
ax.set_title("Zisk a F/B (volný prostor)", fontsize=10)

# rozmerovy nakres
ax = fig.add_subplot(2, 3, 3)
for n, (x, L) in zip(NAMES, [(float(w.center()[0]), w.length) for w in tuned.wires]):
    ax.plot([x - x0, x - x0], [-L / 2, L / 2], color="#1565c0", lw=3)
    ax.text(x - x0, L / 2 + 0.18, f"{L*1000:.0f}", ha="center", fontsize=7.5)
    ax.text(x - x0, -L / 2 - 0.45, n, ha="center", fontsize=7.5, color="#555")
ax.plot([0, boom], [0, 0], color="#795548", lw=4, zorder=0)
ax.set_aspect("equal"); ax.set_xlabel("ráhno [m]"); ax.set_ylabel("[m]")
ax.set_title("Rozměry (délky v mm)", fontsize=10)
ax.set_ylim(-3.4, 3.4); ax.grid(alpha=0.25)

# azimut nad zemi
ax = fig.add_subplot(2, 3, 4, projection="polar")
ph, g = azimuth_cut(sol_gnd, rg.elevation_deg, 361)
top = math.ceil(rg.gain_dbi); rng = 40
ax.plot(np.radians(ph), np.clip(g - top + rng, 0, None), color="#1565c0", lw=1.6)
ax.fill(np.radians(ph), np.clip(g - top + rng, 0, None), color="#1565c0", alpha=0.12)
ax.set_ylim(0, rng); ax.set_yticks(np.arange(0, rng + 1, 10))
ax.set_yticklabels([f"{int(t - rng + top)}" for t in np.arange(0, rng + 1, 10)], fontsize=7)
ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
ax.set_title(f"Azimut, elevace {rg.elevation_deg:.0f}° (max {rg.gain_dbi:.2f} dBi)", fontsize=9)

# elevace nad zemi
ax = fig.add_subplot(2, 3, 5, projection="polar")
elv, g2 = elevation_cut(sol_gnd, 0.0, 361)
ax.plot(np.radians(elv), np.clip(g2 - top + rng, 0, None), color="#1565c0", lw=1.6)
ax.fill(np.radians(elv), np.clip(g2 - top + rng, 0, None), color="#1565c0", alpha=0.12)
ax.set_ylim(0, rng); ax.set_yticks(np.arange(0, rng + 1, 10))
ax.set_yticklabels([f"{int(t - rng + top)}" for t in np.arange(0, rng + 1, 10)], fontsize=7)
ax.set_theta_zero_location("E"); ax.set_theta_direction(1)
ax.set_thetamin(0); ax.set_thetamax(90)
ax.set_title(f"Elevace, {H:.0f} m nad průměrnou zemí", fontsize=9)

# azimut volny prostor
ax = fig.add_subplot(2, 3, 6, projection="polar")
rf = analyse(tuned)
ph3, g3 = azimuth_cut(rf.solution, 0.0, 361)
top3 = math.ceil(rf.gain_dbi)
ax.plot(np.radians(ph3), np.clip(g3 - top3 + rng, 0, None), color="#c62828", lw=1.6)
ax.fill(np.radians(ph3), np.clip(g3 - top3 + rng, 0, None), color="#c62828", alpha=0.10)
ax.set_ylim(0, rng); ax.set_yticks(np.arange(0, rng + 1, 10))
ax.set_yticklabels([f"{int(t - rng + top3)}" for t in np.arange(0, rng + 1, 10)], fontsize=7)
ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
ax.set_title(f"Volný prostor (max {rf.gain_dbi:.2f} dBi, F/B {rf.fb_db:.1f} dB)", fontsize=9)

fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig("/tmp/yagi6_10m.png", dpi=130, bbox_inches="tight")
print("\ngraf: /tmp/yagi6_10m.png")

# ------------------------------------------------------------------ soubory
os.makedirs("/tmp/yagi6", exist_ok=True)
tuned.name = "Yagi 6 prvku 10m - volny prostor"
tuned.save("/tmp/yagi6/yagi6_10m_volny_prostor.json")
gnd.save("/tmp/yagi6/yagi6_10m_nad_zemi.json")
open("/tmp/yagi6/yagi6_10m.maa", "w", encoding="utf-8").write(to_maa(tuned))
open("/tmp/yagi6/yagi6_10m.nec", "w", encoding="utf-8").write(to_nec(tuned))
np.save("/tmp/yagi6_final.npy", np.array([(float(w.center()[0]), w.length) for w in tuned.wires]))
print("soubory:", os.listdir("/tmp/yagi6"))
