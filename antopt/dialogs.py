"""Dialogy GUI — průvodci, úpravy geometrie, VF kalkulátory, 3D diagram."""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

from .model import Model, Wire, Source, Load, Ground, MATERIALS, GROUND_TYPES
from . import geometry_ops as go
from . import hfcalc as hf
from . import wizards as wz
from .solver import solve, swr_from_z

PAD = 6


# ==========================================================================
#  obecný formulář
# ==========================================================================
class FormDialog(tk.Toplevel):
    """Modální dialog se sadou pojmenovaných polí.

    ``fields`` = [(klíč, popisek, druh, výchozí, volby)], druh je
    'f' číslo, 'i' celé číslo, 'combo', 'check', 'text'.
    """

    def __init__(self, parent, title: str, fields: Sequence[tuple],
                 ok_text: str = "Použít", note: str = "", width: int = 22):
        super().__init__(parent)
        self.title(title)
        self.transient(parent)
        self.resizable(False, False)
        self.result: Optional[Dict[str, Any]] = None
        self._fields = fields
        self._vars: Dict[str, tk.Variable] = {}

        body = ttk.Frame(self, padding=PAD)
        body.pack(fill="both", expand=True)
        if note:
            ttk.Label(body, text=note, wraplength=430, justify="left",
                      foreground="#555").grid(row=0, column=0, columnspan=2,
                                              sticky="w", pady=(0, PAD))
        r = 1
        for spec in fields:
            key, label, kind, default = spec[0], spec[1], spec[2], spec[3]
            opts = spec[4] if len(spec) > 4 else None
            if kind == "check":
                v = tk.BooleanVar(value=bool(default))
                ttk.Checkbutton(body, text=label, variable=v).grid(
                    row=r, column=0, columnspan=2, sticky="w", pady=2)
            else:
                ttk.Label(body, text=label + ":").grid(row=r, column=0,
                                                       sticky="e", padx=(0, 6), pady=2)
                v = tk.StringVar(value=str(default))
                if kind == "combo":
                    ttk.Combobox(body, textvariable=v, values=list(opts or []),
                                 state="readonly", width=width).grid(
                        row=r, column=1, sticky="w", pady=2)
                else:
                    ttk.Entry(body, textvariable=v, width=width).grid(
                        row=r, column=1, sticky="w", pady=2)
            self._vars[key] = v
            r += 1

        btn = ttk.Frame(body)
        btn.grid(row=r, column=0, columnspan=2, pady=(PAD, 0), sticky="e")
        ttk.Button(btn, text=ok_text, command=self._ok).pack(side="left", padx=4)
        ttk.Button(btn, text="Zrušit", command=self.destroy).pack(side="left")
        self.bind("<Return>", lambda e: self._ok())
        self.bind("<Escape>", lambda e: self.destroy())
        self.update_idletasks()
        try:
            x = parent.winfo_rootx() + 120
            y = parent.winfo_rooty() + 90
            self.geometry(f"+{x}+{y}")
        except tk.TclError:
            pass
        self.grab_set()

    def _ok(self):
        out: Dict[str, Any] = {}
        for spec in self._fields:
            key, label, kind = spec[0], spec[1], spec[2]
            raw = self._vars[key].get()
            if kind == "check":
                out[key] = bool(raw)
            elif kind in ("f", "i"):
                try:
                    val = float(str(raw).replace(",", "."))
                except ValueError:
                    messagebox.showerror("Neplatná hodnota",
                                         f"„{label}“ musí být číslo.", parent=self)
                    return
                out[key] = int(round(val)) if kind == "i" else val
            else:
                out[key] = raw
        self.result = out
        self.destroy()


def ask_form(parent, title, fields, ok_text="Použít", note="", width=22):
    dlg = FormDialog(parent, title, fields, ok_text, note, width)
    parent.wait_window(dlg)
    return dlg.result


# ==========================================================================
#  průvodci tvorbou antény
# ==========================================================================
def wizard_dialog(parent) -> Optional[Model]:
    """Vybere průvodce a vytvoří novou anténu."""
    pick = ask_form(parent, "Nová anténa — průvodce",
                    [("name", "Typ antény", "combo", wz.WIZARDS[0].name,
                      [w.name for w in wz.WIZARDS])],
                    ok_text="Dál →",
                    note="Průvodce vytvoří rozumný výchozí model. Přesné "
                         "doladění pak nech na optimalizátoru.")
    if not pick:
        return None
    w = wz.WIZARD_BY_NAME[pick["name"]]
    fields = []
    for f in w.fields:
        if f.kind == "ground":
            fields.append((f.key, f.label, "combo", f.default, list(GROUND_TYPES)))
        elif f.kind == "material":
            fields.append((f.key, f.label, "combo", f.default, list(MATERIALS)))
        else:
            fields.append((f.key, f.label, f.kind, f.default))
    vals = ask_form(parent, f"{w.name} — parametry", fields, ok_text="Vytvořit")
    if not vals:
        return None
    if w.name == "Yagi" and not vals.get("boom"):
        vals["boom"] = None
    try:
        return w.fn(**vals)
    except Exception as e:
        messagebox.showerror("Průvodce selhal", str(e), parent=parent)
        return None


# ==========================================================================
#  úpravy geometrie
# ==========================================================================
def _sel_note(sel: Optional[Sequence[int]]) -> str:
    if sel is None:
        return "Použije se na všechny dráty."
    return "Použije se na drát č. " + ", ".join(str(i + 1) for i in sel) + "."


def move_dialog(parent, model: Model, sel=None) -> bool:
    v = ask_form(parent, "Posunout", [
        ("dx", "Posun X [m]", "f", 0.0),
        ("dy", "Posun Y [m]", "f", 0.0),
        ("dz", "Posun Z [m]", "f", 0.0)], note=_sel_note(sel))
    if not v:
        return False
    go.move(model, v["dx"], v["dy"], v["dz"], sel)
    return True


def rotate_dialog(parent, model: Model, sel=None) -> bool:
    v = ask_form(parent, "Otočit", [
        ("angle", "Úhel [°]", "f", 90.0),
        ("axis", "Kolem osy", "combo", "z", ["x", "y", "z"]),
        ("cx", "Střed X [m]", "f", 0.0),
        ("cy", "Střed Y [m]", "f", 0.0),
        ("cz", "Střed Z [m]", "f", 0.0)], note=_sel_note(sel))
    if not v:
        return False
    go.rotate(model, v["angle"], v["axis"], (v["cx"], v["cy"], v["cz"]), sel)
    return True


def mirror_dialog(parent, model: Model, sel=None) -> bool:
    v = ask_form(parent, "Zrcadlit", [
        ("plane", "Podle roviny kolmé na osu", "combo", "x", ["x", "y", "z"]),
        ("coord", "Souřadnice roviny [m]", "f", 0.0),
        ("copy", "Ponechat i původní (vytvořit kopii)", "check", True)],
        note=_sel_note(sel))
    if not v:
        return False
    go.mirror(model, v["plane"], v["coord"], sel, v["copy"])
    return True


def scale_dialog(parent, model: Model, sel=None) -> bool:
    v = ask_form(parent, "Změnit měřítko", [
        ("factor", "Poměr", "f", 1.0),
        ("radius", "Škálovat i poloměry vodičů", "check", True)],
        note=_sel_note(sel) + " Střed škálování je počátek soustavy.")
    if not v or v["factor"] <= 0:
        return False
    go.scale(model, v["factor"], None, v["radius"], sel)
    return True


def rescale_freq_dialog(parent, model: Model) -> bool:
    v = ask_form(parent, "Přeladit anténu na jiný kmitočet", [
        ("f", "Nový kmitočet [MHz]", "f", round(model.freq_mhz * 2, 4)),
        ("radius", "Škálovat i poloměry vodičů", "check", True),
        ("keep_h", "Nechat výšku nad zemí beze změny", "check", True)],
        note="Celá geometrie se přepočítá poměrem kmitočtů — z antény na jedno "
             "pásmo tak vznikne totéž na jiném.")
    if not v:
        return False
    try:
        f = go.rescale_to_frequency(model, v["f"], v["radius"], v["keep_h"])
    except ValueError as e:
        messagebox.showerror("Chyba", str(e), parent=parent)
        return False
    messagebox.showinfo("Přeladěno",
                        f"Rozměry přepočteny poměrem {f:.4f}.", parent=parent)
    return True


def stack_dialog(parent, model: Model) -> bool:
    lam = model.wavelength
    v = ask_form(parent, "Vytvořit stoh / řadu", [
        ("nz", "Počet nad sebou", "i", 2),
        ("dz", "Svislá rozteč [m]", "f", round(lam / 2, 3)),
        ("nx", "Počet vedle sebe", "i", 1),
        ("dx", "Vodorovná rozteč [m]", "f", round(lam / 2, 3)),
        ("feed_all", "Napájet všechny prvky", "check", True),
        ("phase", "Fázový posun mezi kopiemi [°]", "f", 0.0)],
        note="Z jedné antény udělá stoh nebo řadu kopií. Nejvýš 64 kopií.")
    if not v:
        return False
    try:
        go.make_stack(model, v["nx"], v["nz"], v["dx"], v["dz"],
                      v["feed_all"], v["phase"])
    except ValueError as e:
        messagebox.showerror("Chyba", str(e), parent=parent)
        return False
    return True


def polar_wire_dialog(parent, model: Model) -> bool:
    v = ask_form(parent, "Přidat drát polárně", [
        ("x", "Počátek X [m]", "f", 0.0),
        ("y", "Počátek Y [m]", "f", 0.0),
        ("z", "Počátek Z [m]", "f", round(model.wavelength / 2, 2)),
        ("length", "Délka [m]", "f", round(model.wavelength / 4, 3)),
        ("az", "Azimut [°]", "f", 0.0),
        ("ze", "Zenitový úhel [°]", "f", 90.0),
        ("dia", "Průměr vodiče [mm]", "f", 2.0),
        ("nseg", "Segmentů", "i", 21)],
        note="Zenitový úhel: 0° svisle vzhůru, 90° vodorovně, 180° dolů.")
    if not v:
        return False
    model.wires.append(go.polar_to_wire((v["x"], v["y"], v["z"]), v["length"],
                                        v["az"], v["ze"], v["dia"] / 2000.0,
                                        max(1, v["nseg"])))
    return True


# --------------------------------------------------------------------------
class TaperDialog(tk.Toplevel):
    """Zúžený (teleskopický) prvek — Edit → Taper Wire Set."""

    def __init__(self, parent, model: Model, wire: int):
        super().__init__(parent)
        self.transient(parent)
        self.model = model
        # pracuj vždy s CELÝM prvkem, ne s jedním drátem — jinak by se
        # u už poskládaného prvku přestavěla jen jedna trubka
        self.wire = go.element_wires(model, wire)
        self.ok = False
        _, _, total = go._group_axis(model, self.wire)
        half = total / 2.0
        self.half_mm = half * 1000.0
        n = len(self.wire)
        self.title("Zúžený prvek — dráty "
                   + ", ".join(str(i + 1) for i in self.wire[:6])
                   + ("…" if n > 6 else ""))

        body = ttk.Frame(self, padding=PAD)
        body.pack(fill="both", expand=True)
        existing = go.element_sections(model, self.wire)
        note = ("Prvek je už poskládaný — sekce níž jsou jeho současné rozměry, "
                "můžeš je rovnou přepsat.\n" if n > 1 else "")
        ttk.Label(body, wraplength=520, justify="left", foreground="#555",
                  text=f"{note}Prvek se poskládá symetricky ze sekcí zadaných od "
                       f"středu ven. Součet délek musí dát polovinu prvku "
                       f"({half * 1000:.0f} mm), jinak se délka prvku změní.\n"
                       f"Segmenty budou v celém prvku stejně dlouhé — právě "
                       f"sousedství různě dlouhých segmentů na skoku průměru "
                       f"dělá u zúžených prvků největší chybu."
                  ).pack(anchor="w", pady=(0, PAD))

        cols = ("len", "dia")
        self.tree = ttk.Treeview(body, columns=cols, show="headings", height=7)
        self.tree.heading("len", text="délka sekce [mm]")
        self.tree.heading("dia", text="průměr [mm]")
        self.tree.column("len", width=140, anchor="e")
        self.tree.column("dia", width=120, anchor="e")
        self.tree.pack(fill="both", expand=True)

        d0 = model.wires[self.wire[-1]].radius * 2000.0
        if len(existing) > 1:
            preset = [(s.length, s.radius * 2000.0) for s in existing]
        else:
            preset = [(half * 0.45, d0 * 1.6), (half * 0.30, d0 * 1.25),
                      (half * 0.25, d0)]
        for L, d in preset:
            self.tree.insert("", "end", values=(f"{L * 1000:.0f}", f"{d:.1f}"))

        ed = ttk.Frame(body)
        ed.pack(fill="x", pady=(PAD, 0))
        ttk.Label(ed, text="délka [mm]:").pack(side="left")
        self.v_len = tk.StringVar(value="500")
        ttk.Entry(ed, textvariable=self.v_len, width=9).pack(side="left", padx=3)
        ttk.Label(ed, text="průměr [mm]:").pack(side="left", padx=(8, 0))
        self.v_dia = tk.StringVar(value=f"{d0:.1f}")
        ttk.Entry(ed, textvariable=self.v_dia, width=9).pack(side="left", padx=3)
        ttk.Button(ed, text="Přidat", command=self._add).pack(side="left", padx=4)
        ttk.Button(ed, text="Změnit", command=self._edit).pack(side="left")
        ttk.Button(ed, text="Smazat", command=self._del).pack(side="left", padx=4)

        self.lbl = ttk.Label(body, text="")
        self.lbl.pack(anchor="w", pady=(PAD, 0))
        btn = ttk.Frame(body)
        btn.pack(fill="x", pady=(PAD, 0))
        ttk.Button(btn, text="Vytvořit prvek", command=self._ok).pack(side="left")
        ttk.Button(btn, text="Zrušit", command=self.destroy).pack(side="left", padx=4)
        self._refresh()
        self.grab_set()

    def _rows(self):
        out = []
        for i in self.tree.get_children():
            L, d = self.tree.set(i, "len"), self.tree.set(i, "dia")
            try:
                out.append((float(L.replace(",", ".")), float(d.replace(",", "."))))
            except ValueError:
                pass
        return out

    def _refresh(self):
        rows = self._rows()
        tot = sum(L for L, _ in rows)
        half = self.half_mm
        self.lbl.configure(
            text=f"Součet {tot:.0f} mm  →  celková délka prvku {2 * tot / 1000:.4f} m "
                 f"(původní {2 * half / 1000:.4f} m)",
            foreground="#2e7d32" if abs(tot - half) < 1 else "#a05000")

    def _vals(self):
        try:
            return (float(self.v_len.get().replace(",", ".")),
                    float(self.v_dia.get().replace(",", ".")))
        except ValueError:
            messagebox.showerror("Chyba", "Zadej čísla.", parent=self)
            return None

    def _add(self):
        v = self._vals()
        if v:
            self.tree.insert("", "end", values=(f"{v[0]:.0f}", f"{v[1]:.1f}"))
            self._refresh()

    def _edit(self):
        sel = self.tree.selection()
        v = self._vals()
        if sel and v:
            self.tree.item(sel[0], values=(f"{v[0]:.0f}", f"{v[1]:.1f}"))
            self._refresh()

    def _del(self):
        for i in self.tree.selection():
            self.tree.delete(i)
        self._refresh()

    def _ok(self):
        rows = self._rows()
        if not rows:
            messagebox.showerror("Chyba", "Zadej aspoň jednu sekci.", parent=self)
            return
        secs = [go.TaperSection(L / 1000.0, d / 2000.0) for L, d in rows]
        try:
            go.taper_element(self.model, self.wire, secs)
        except ValueError as e:
            messagebox.showerror("Chyba", str(e), parent=self)
            return
        self.ok = True
        self.destroy()


def taper_dialog(parent, model: Model, wire: int) -> bool:
    dlg = TaperDialog(parent, model, wire)
    parent.wait_window(dlg)
    return dlg.ok


# --------------------------------------------------------------------------
class ElementDialog(tk.Toplevel):
    """Editor prvků — pracuje s celými prvky včetně zúžení."""

    def __init__(self, parent, model: Model, on_change: Callable):
        super().__init__(parent)
        self.title("Editor prvků")
        self.transient(parent)
        self.model, self.on_change = model, on_change
        body = ttk.Frame(self, padding=PAD)
        body.pack(fill="both", expand=True)
        ttk.Label(body, foreground="#555", wraplength=560, justify="left",
                  text="Prvek = skupina kolineárních propojených drátů. Změna "
                       "délky zachová poměry sekcí u zúžených prvků. Hodnoty "
                       "se mění dvojklikem; sloupec „hrot“ vysune jen koncovou "
                       "trubku, jako se prvek ladí doopravdy.").pack(anchor="w",
                                                                    pady=(0, PAD))
        cols = ("n", "x", "y", "z", "len", "tip", "tap", "wires")
        self.tree = ttk.Treeview(body, columns=cols, show="headings", height=10)
        for k, t, w_ in (("n", "#", 34), ("x", "X [m]", 88), ("y", "Y [m]", 88),
                         ("z", "Z [m]", 88), ("len", "délka [m]", 96),
                         ("tip", "hrot [m]", 88),
                         ("tap", "průřez", 128), ("wires", "dráty", 110)):
            self.tree.heading(k, text=t)
            self.tree.column(k, width=w_, anchor="e" if k not in ("tap", "wires") else "w")
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<Double-1>", self._edit)
        btn = ttk.Frame(body)
        btn.pack(fill="x", pady=(PAD, 0))
        ttk.Button(btn, text="Zúžení prvku…", command=self._taper).pack(side="left")
        ttk.Button(btn, text="Zavřít", command=self.destroy).pack(side="right")
        self.refresh()
        self.grab_set()

    def refresh(self):
        for i in self.tree.get_children():
            self.tree.delete(i)
        self.els = go.find_elements(self.model)
        for k, el in enumerate(self.els, 1):
            self.tree.insert("", "end", values=(
                k, f"{el.center[0]:.4f}", f"{el.center[1]:.4f}",
                f"{el.center[2]:.4f}", f"{el.length:.4f}",
                f"{go.element_tip_length(self.model, el):.4f}",
                el.describe(self.model),
                "+".join(str(i + 1) for i in el.wires)))

    def _selected(self):
        sel = self.tree.selection()
        if not sel:
            return None
        return self.els[self.tree.index(sel[0])]

    def _taper(self):
        el = self._selected()
        if el is None:
            messagebox.showinfo("Vyber prvek",
                                "Nejdřív v tabulce vyber prvek.", parent=self)
            return
        if taper_dialog(self, self.model, el.wires[0]):
            self.refresh()
            self.on_change()

    def _edit(self, event):
        row = self.tree.identify_row(event.y)
        col = self.tree.identify_column(event.x)
        if not row or col not in ("#2", "#3", "#4", "#5", "#6"):
            return
        i = self.tree.index(row)
        el = self.els[i]
        keymap = {"#2": ("x", "Poloha X [m]", el.center[0]),
                  "#3": ("y", "Poloha Y [m]", el.center[1]),
                  "#4": ("z", "Poloha Z [m]", el.center[2]),
                  "#5": ("len", "Délka prvku [m]", el.length),
                  "#6": ("tip", "Délka koncové trubky [m]",
                         go.element_tip_length(self.model, el))}
        key, label, cur = keymap[col]
        v = ask_form(self, "Úprava prvku", [("v", label, "f", round(float(cur), 4))])
        if not v:
            return
        if key == "len":
            go.set_element_length(self.model, el, v["v"])
        elif key == "tip":
            go.set_element_tip(self.model, el, v["v"])
        else:
            go.set_element_position(self.model, el, key, v["v"])
        self.refresh()
        self.on_change()


# ==========================================================================
#  VF kalkulátory
# ==========================================================================
class CalcWindow(tk.Toplevel):
    """Okno s výstupním textem a tlačítkem Spočítat."""

    def __init__(self, parent, title: str, fields, compute: Callable[[dict], str],
                 note: str = "", height: int = 16, width: int = 76):
        super().__init__(parent)
        self.title(title)
        self.transient(parent)
        self.compute = compute
        body = ttk.Frame(self, padding=PAD)
        body.pack(fill="both", expand=True)
        if note:
            ttk.Label(body, text=note, wraplength=540, justify="left",
                      foreground="#555").pack(anchor="w", pady=(0, PAD))
        form = ttk.Frame(body)
        form.pack(fill="x")
        self._fields = fields
        self._vars = {}
        for r, spec in enumerate(fields):
            key, label, kind, default = spec[0], spec[1], spec[2], spec[3]
            opts = spec[4] if len(spec) > 4 else None
            col = (r % 2) * 2
            row = r // 2
            ttk.Label(form, text=label + ":").grid(row=row, column=col,
                                                   sticky="e", padx=(0, 4), pady=2)
            if kind == "check":
                v = tk.BooleanVar(value=bool(default))
                ttk.Checkbutton(form, variable=v).grid(row=row, column=col + 1,
                                                       sticky="w", padx=(0, 14))
            else:
                v = tk.StringVar(value=str(default))
                if kind == "combo":
                    ttk.Combobox(form, textvariable=v, values=list(opts or []),
                                 state="readonly", width=20).grid(
                        row=row, column=col + 1, sticky="w", padx=(0, 14), pady=2)
                else:
                    ttk.Entry(form, textvariable=v, width=14).grid(
                        row=row, column=col + 1, sticky="w", padx=(0, 14), pady=2)
            self._vars[key] = v
        self.txt = tk.Text(body, height=height, width=width, wrap="word",
                           font=("TkFixedFont", 10))
        self.txt.pack(fill="both", expand=True, pady=(PAD, 0))
        b = ttk.Frame(body)
        b.pack(fill="x", pady=(PAD, 0))
        ttk.Button(b, text="Spočítat", command=self.run).pack(side="left")
        ttk.Button(b, text="Zavřít", command=self.destroy).pack(side="right")
        self.run()

    def values(self):
        out = {}
        for spec in self._fields:
            key, label, kind = spec[0], spec[1], spec[2]
            raw = self._vars[key].get()
            if kind == "check":
                out[key] = bool(raw)
            elif kind in ("f", "i"):
                try:
                    val = float(str(raw).replace(",", "."))
                except ValueError:
                    return None
                out[key] = int(round(val)) if kind == "i" else val
            else:
                out[key] = raw
        return out

    def run(self):
        v = self.values()
        self.txt.delete("1.0", "end")
        if v is None:
            self.txt.insert("end", "Některá hodnota není číslo.\n")
            return
        try:
            self.txt.insert("end", self.compute(v))
        except Exception as e:
            self.txt.insert("end", f"Chyba: {e}\n")


def resonance_dialog(parent, freq_mhz: float):
    def compute(v):
        f = v["f"]
        out = [f"Kmitočet {f:g} MHz\n"]
        if v["l_uh"] > 0:
            out.append(f"  L = {v['l_uh']:g} µH  →  X = +{hf.reactance_l(v['l_uh'], f):.1f} Ω")
        if v["c_pf"] > 0:
            out.append(f"  C = {v['c_pf']:g} pF  →  X = {hf.reactance_c(v['c_pf'], f):.1f} Ω")
        if v["l_uh"] > 0 and v["c_pf"] > 0:
            out.append(f"  LC rezonuje na {hf.resonant_frequency(v['l_uh'], v['c_pf']):.4f} MHz")
        if v["x"] != 0:
            out.append("")
            out.append(f"  Pro reaktanci {v['x']:+g} Ω na {f:g} MHz potřebuješ:")
            if v["x"] > 0:
                out.append(f"    cívku {hf.l_for_reactance(v['x'], f):.4f} µH")
            else:
                out.append(f"    kondenzátor {hf.c_for_reactance(v['x'], f):.1f} pF")
        out.append("")
        out.append("Návrh jednovrstvé vzduchové cívky:")
        d, ln, t = v["coil_d"], v["coil_l"], v["turns"]
        if d > 0 and ln > 0 and t > 0:
            out.append(f"  Ø {d:g} mm, délka {ln:g} mm, {t:g} závitů "
                       f"→ L = {hf.coil_inductance(d, ln, t):.3f} µH")
        if v["target_uh"] > 0 and d > 0 and ln > 0:
            n = hf.coil_turns(v["target_uh"], d, ln)
            out.append(f"  Pro {v['target_uh']:g} µH při Ø {d:g} mm a délce {ln:g} mm: "
                       f"{n:.1f} závitů")
            if v["wire_d"] > 0:
                out.append(f"  Drát Ø {v['wire_d']:g} mm těsně vinutý zabere "
                           f"{hf.coil_length_for_turns(n, v['wire_d']):.0f} mm")
        return "\n".join(out) + "\n"

    CalcWindow(parent, "Rezonance a cívky", [
        ("f", "Kmitočet [MHz]", "f", round(freq_mhz, 4)),
        ("x", "Chci reaktanci [Ω]", "f", 0.0),
        ("l_uh", "L [µH]", "f", 0.0),
        ("c_pf", "C [pF]", "f", 0.0),
        ("coil_d", "Cívka — průměr [mm]", "f", 30.0),
        ("coil_l", "Cívka — délka [mm]", "f", 50.0),
        ("turns", "Cívka — závitů", "f", 12.0),
        ("target_uh", "Chci cívku [µH]", "f", 0.0),
        ("wire_d", "Průměr drátu [mm]", "f", 1.5),
    ], compute, note="Reaktance prvků, rezonance LC a návrh vzduchové cívky "
                     "Wheelerovým vzorcem.")


def lc_match_dialog(parent, z_load: complex, freq_mhz: float, z0: float):
    def compute(v):
        z = complex(v["r"], v["x"])
        out = [f"Zátěž {z.real:g} {z.imag:+g} j Ω  →  {v['z0']:g} Ω  "
               f"na {v['f']:g} MHz",
               f"PSV bez přizpůsobení: {swr_from_z(z, v['z0']):.2f}", ""]
        sols = hf.lc_match(z, v["z0"], v["f"])
        if not sols:
            out.append("Pro tuhle impedanci L-článek nevyjde — zkus jinou "
                       "topologii nebo transformační vedení.")
        for i, s in enumerate(sols, 1):
            out.append(f"Řešení {i}:")
            out.append(s.report(v["f"]))
            out.append("")
        out.append("Sériový prvek je blíž anténě podle popisu u řešení. "
                   "Ztráty v cívce nejsou započteny — drž Q vinutí vysoké.")
        return "\n".join(out) + "\n"

    CalcWindow(parent, "LC přizpůsobovací článek", [
        ("r", "Zátěž R [Ω]", "f", round(z_load.real, 2)),
        ("x", "Zátěž X [Ω]", "f", round(z_load.imag, 2)),
        ("f", "Kmitočet [MHz]", "f", round(freq_mhz, 4)),
        ("z0", "Cílová impedance [Ω]", "f", z0),
    ], compute, note="L-článek ze dvou reaktancí. Vypíše obě možná zapojení.")


def line_match_dialog(parent, z_load: complex, freq_mhz: float, z0: float):
    names = [l.name for l in hf.LINES]

    def compute(v):
        z = complex(v["r"], v["x"])
        line = hf.LINE_BY_NAME[v["line"]]
        stub_line = hf.LINE_BY_NAME[v["stub_line"]]
        out = [f"Zátěž {z.real:g} {z.imag:+g} j Ω na {v['f']:g} MHz, "
               f"napáječ {line.name} ({line.z0:g} Ω, vf {line.vf:g})",
               f"PSV v napáječi: {swr_from_z(z, line.z0):.2f}", ""]

        out.append("── Přizpůsobení jedním pahýlem ──")
        for shorted in (True, False):
            sols = hf.single_stub_match(z, line.z0, v["f"], line.vf,
                                        stub_line.vf, stub_line.z0, shorted)
            if not sols:
                out.append(f"  {'zkratovaný' if shorted else 'otevřený'}: řešení nenalezeno")
            for s in sols:
                out.append(s.report(stub_line.vf))
        out.append("")

        out.append("── Vložená transformační sekce ──")
        sec = hf.LINE_BY_NAME[v["sec_line"]]
        res = hf.series_section_match(z, line.z0, sec.z0, v["f"], line.vf, sec.vf)
        if not res:
            out.append("  řešení nenalezeno")
        for d, ls, z2 in res:
            out.append(f"  {sec.name} ({sec.z0:g} Ω): začátek {d * 1000:.0f} mm "
                       f"od antény, délka {ls * 1000:.0f} mm  →  "
                       f"Z = {z2.real:.1f}{z2.imag:+.1f} j Ω "
                       f"(PSV {swr_from_z(z2, v['z0']):.2f})")
        out.append("")
        out.append("Délky platí pro uvedený činitel zkrácení — u reálného kabelu "
                   "si ho ověř, liší se kus od kusu.")
        return "\n".join(out) + "\n"

    CalcWindow(parent, "Přizpůsobení vedením", [
        ("r", "Zátěž R [Ω]", "f", round(z_load.real, 2)),
        ("x", "Zátěž X [Ω]", "f", round(z_load.imag, 2)),
        ("f", "Kmitočet [MHz]", "f", round(freq_mhz, 4)),
        ("z0", "Cílová impedance [Ω]", "f", z0),
        ("line", "Napáječ", "combo", "RG-213 U", names),
        ("stub_line", "Pahýl z", "combo", "RG-213 U", names),
        ("sec_line", "Transformační sekce", "combo", "RG-11", names),
    ], compute, height=20,
        note="Pahýl (Stub match) i vložená sekce jiné impedance (Line match).")


def feedline_dialog(parent, z_load: complex, freq_mhz: float, z0: float):
    names = [l.name for l in hf.LINES]

    def compute(v):
        z = complex(v["r"], v["x"])
        line = hf.LINE_BY_NAME[v["line"]]
        zin, sa, st, loss = hf.coax_feed(z, line, v["len"], v["f"], v["z0"])
        eff = 100.0 * 10 ** (-loss / 10.0)
        return "\n".join([
            f"{line.name}: Z₀ {line.z0:g} Ω, vf {line.vf:g}, "
            f"útlum {line.loss_db_per_100m(v['f']):.2f} dB/100 m na {v['f']:g} MHz",
            "",
            f"  impedance antény      {z.real:.1f} {z.imag:+.1f} j Ω",
            f"  PSV u antény          {sa:.2f}",
            f"  délka napáječe        {v['len']:g} m",
            "",
            f"  impedance u TX        {zin.real:.1f} {zin.imag:+.1f} j Ω",
            f"  PSV u TX (Z₀={v['z0']:g} Ω)   {st:.2f}",
            f"  celková ztráta        {loss:.2f} dB   →  na anténu dojde "
            f"{eff:.0f} % výkonu",
            "",
            "PSV u TX bývá nižší než u antény — to není zlepšení, jen ztráty "
            "v kabelu maskují odraz.",
        ]) + "\n"

    CalcWindow(parent, "Napáječ", [
        ("r", "Anténa R [Ω]", "f", round(z_load.real, 2)),
        ("x", "Anténa X [Ω]", "f", round(z_load.imag, 2)),
        ("f", "Kmitočet [MHz]", "f", round(freq_mhz, 4)),
        ("len", "Délka napáječe [m]", "f", 30.0),
        ("line", "Kabel", "combo", "RG-213 U", names),
        ("z0", "Vztažná impedance [Ω]", "f", z0),
    ], compute, height=14,
        note="Co uvidí vysílač na konci kabelu, včetně ztrát zvýšených odrazem.")
