"""Desktopové GUI (Tkinter + matplotlib)."""
from __future__ import annotations

import math
import os
import queue
import threading
import traceback
from typing import Callable, List, Optional

import numpy as np

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from .model import Model, Wire, Source, Load, Ground, MATERIALS, GROUND_TYPES
from .solver import solve, swr_from_z
from .farfield import performance, azimuth_cut, elevation_cut, far_field
from .analysis import analyse, sweep, bandwidth, Result
from .fileio import to_nec, from_nec, to_maa, from_maa
from .examples import EXAMPLES
from .optimize import (Parameter, Objective, optimize, evaluate,
                       suggest_parameters, read_param, PARAM_KINDS, free_params)
from . import dialogs as dlg
from . import geometry_ops as go
from . import wizards as wz
from .analysis import find_resonance, q_factor, q_estimate
from . import engines

PAD = 6


# ==========================================================================
#  editovatelná tabulka
# ==========================================================================
class EditableTable(ttk.Frame):
    """Treeview s editací buňky po dvojkliku."""

    def __init__(self, parent, columns, on_change: Optional[Callable] = None,
                 height: int = 8):
        super().__init__(parent)
        self.columns = columns              # [(key, label, width, kind)]
        self.on_change = on_change
        keys = [c[0] for c in columns]
        self.tree = ttk.Treeview(self, columns=keys, show="headings", height=height,
                                 selectmode="browse")
        for key, label, width, kind in columns:
            self.tree.heading(key, text=label)
            self.tree.column(key, width=width, anchor="e" if kind != "s" else "w",
                             stretch=False)
        vs = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        hs = ttk.Scrollbar(self, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscroll=vs.set, xscroll=hs.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vs.grid(row=0, column=1, sticky="ns")
        hs.grid(row=1, column=0, sticky="ew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self._editor = None
        self.tree.bind("<Double-1>", self._begin_edit)
        self.tree.bind("<Return>", self._begin_edit)

    # ------------------------------------------------------------------
    def set_rows(self, rows: List[list]):
        sel = self.tree.selection()
        keep = self.tree.index(sel[0]) if sel else None
        for i in self.tree.get_children():
            self.tree.delete(i)
        for r in rows:
            self.tree.insert("", "end", values=[self._fmt(v, c[3])
                                                for v, c in zip(r, self.columns)])
        kids = self.tree.get_children()
        if keep is not None and kids:
            self.tree.selection_set(kids[min(keep, len(kids) - 1)])

    @staticmethod
    def _fmt(v, kind):
        if kind == "f":
            return f"{float(v):.4f}"
        if kind == "f2":
            return f"{float(v):.5f}"
        if kind == "i":
            return str(int(v))
        return str(v)

    def selected_index(self) -> Optional[int]:
        sel = self.tree.selection()
        if not sel:
            return None
        return self.tree.index(sel[0])

    def select(self, idx: int):
        kids = self.tree.get_children()
        if 0 <= idx < len(kids):
            self.tree.selection_set(kids[idx])
            self.tree.see(kids[idx])

    # ------------------------------------------------------------------
    def _begin_edit(self, event=None):
        self._end_edit()
        if event is not None and event.type == tk.EventType.ButtonPress:
            row = self.tree.identify_row(event.y)
            col = self.tree.identify_column(event.x)
        else:
            sel = self.tree.selection()
            row = sel[0] if sel else ""
            col = "#1"
        if not row or not col:
            return
        ci = int(col[1:]) - 1
        if ci < 0 or ci >= len(self.columns):
            return
        x, y, w, h = self.tree.bbox(row, col)
        value = self.tree.set(row, self.columns[ci][0])
        var = tk.StringVar(value=value)
        ed = ttk.Entry(self.tree, textvariable=var, justify="right")
        ed.place(x=x, y=y, width=w, height=h)
        ed.focus_set()
        ed.selection_range(0, "end")
        self._editor = (ed, row, ci, var)
        ed.bind("<Return>", lambda e: self._commit())
        ed.bind("<Escape>", lambda e: self._end_edit())
        ed.bind("<FocusOut>", lambda e: self._commit())
        ed.bind("<Tab>", lambda e: self._commit(next_col=True))

    def _commit(self, next_col=False):
        if not self._editor:
            return "break"
        ed, row, ci, var = self._editor
        key, label, width, kind = self.columns[ci]
        txt = var.get().strip().replace(",", ".")
        self._end_edit()
        try:
            val = float(txt) if kind in ("f", "f2") else (int(float(txt)) if kind == "i" else txt)
        except ValueError:
            messagebox.showerror("Neplatná hodnota", f"„{txt}“ není číslo.")
            return "break"
        idx = self.tree.index(row)
        if self.on_change:
            self.on_change(idx, key, val)
        return "break"

    def _end_edit(self):
        if self._editor:
            self._editor[0].destroy()
            self._editor = None


# ==========================================================================
#  pracovní vlákno
# ==========================================================================
class Worker:
    def __init__(self, root: tk.Misc):
        self.root = root
        self.q: "queue.Queue" = queue.Queue()
        self.cancel = threading.Event()
        self.busy = False
        self._poll()

    def run(self, fn, on_done, on_error=None, on_progress=None):
        if self.busy:
            messagebox.showinfo("Počítá se", "Počkej, až doběhne předchozí výpočet.")
            return False
        self.busy = True
        self.cancel.clear()

        def target():
            try:
                res = fn(self.cancel, lambda *a: self.q.put(("progress", on_progress, a)))
                self.q.put(("done", on_done, res))
            except Exception:
                self.q.put(("error", on_error, traceback.format_exc()))
        threading.Thread(target=target, daemon=True).start()
        return True

    def stop(self):
        self.cancel.set()

    def _poll(self):
        try:
            while True:
                kind, cb, payload = self.q.get_nowait()
                if kind == "progress":
                    if cb:
                        cb(*payload)
                elif kind == "done":
                    self.busy = False
                    if cb:
                        cb(payload)
                else:
                    self.busy = False
                    if cb:
                        cb(payload)
                    else:
                        messagebox.showerror("Chyba výpočtu", payload)
        except queue.Empty:
            pass
        self.root.after(60, self._poll)


# ==========================================================================
#  hlavní okno
# ==========================================================================
class App(ttk.Frame):
    def __init__(self, master: tk.Tk):
        super().__init__(master, padding=PAD)
        self.master = master
        self.pack(fill="both", expand=True)
        self.model = EXAMPLES["Yagi 3 prvky 20 m"]()
        self.path: Optional[str] = None
        self.result: Optional[Result] = None
        self.sweep_results: List[Result] = []
        self.opt_params: List[Parameter] = []
        self.worker = Worker(master)

        self._build_menu()
        self.status = ttk.Label(self, text="Připraveno", anchor="w",
                                relief="sunken", padding=(6, 3))
        self.status.pack(side="bottom", fill="x", pady=(PAD, 0))
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True)
        self._build_geometry_tab()
        self._build_feed_tab()
        self._build_calc_tab()
        self._build_pattern_tab()
        self._build_opt_tab()
        self.refresh_all()

    # ------------------------------------------------------------------ menu
    def _build_menu(self):
        m = tk.Menu(self.master)
        f = tk.Menu(m, tearoff=0)
        f.add_command(label="Nový", command=self.new_model, accelerator="Ctrl+N")
        f.add_command(label="Otevřít…", command=self.open_file, accelerator="Ctrl+O")
        f.add_command(label="Uložit", command=self.save_file, accelerator="Ctrl+S")
        f.add_command(label="Uložit jako…", command=lambda: self.save_file(True))
        f.add_separator()
        f.add_command(label="Importovat NEC / MMANA…", command=self.import_file)
        f.add_command(label="Exportovat NEC…", command=lambda: self.export_as("nec"))
        f.add_command(label="Exportovat MMANA (.maa)…", command=lambda: self.export_as("maa"))
        f.add_separator()
        f.add_command(label="Konec", command=self.master.destroy)
        m.add_cascade(label="Soubor", menu=f)

        nw = tk.Menu(m, tearoff=0)
        nw.add_command(label="Průvodce novou anténou…", command=self.run_wizard)
        nw.add_separator()
        ex = tk.Menu(nw, tearoff=0)
        for name in EXAMPLES:
            ex.add_command(label=name, command=lambda n=name: self.load_example(n))
        nw.add_cascade(label="Hotové příklady", menu=ex)
        m.add_cascade(label="Nová anténa", menu=nw)

        ed = tk.Menu(m, tearoff=0)
        ed.add_command(label="Posunout…", command=lambda: self._edit_op(dlg.move_dialog))
        ed.add_command(label="Otočit…", command=lambda: self._edit_op(dlg.rotate_dialog))
        ed.add_command(label="Zrcadlit…", command=lambda: self._edit_op(dlg.mirror_dialog))
        ed.add_command(label="Změnit měřítko…", command=lambda: self._edit_op(dlg.scale_dialog))
        ed.add_separator()
        ed.add_command(label="Přeladit na jiný kmitočet…", command=self.op_rescale)
        ed.add_command(label="Vytvořit stoh / řadu…", command=self.op_stack)
        ed.add_separator()
        ed.add_command(label="Zúžený (teleskopický) prvek…", command=self.op_taper)
        ed.add_command(label="Přidat drát polárně…", command=self.op_polar)
        ed.add_command(label="Editor prvků…", command=self.op_elements)
        m.add_cascade(label="Úpravy", menu=ed)

        tl = tk.Menu(m, tearoff=0)
        tl.add_command(label="Rezonance a cívky…",
                       command=lambda: dlg.resonance_dialog(self.master, self.model.freq_mhz))
        tl.add_command(label="LC přizpůsobovací článek…",
                       command=lambda: dlg.lc_match_dialog(self.master, self._zin(),
                                                           self.model.freq_mhz, self.model.z0))
        tl.add_command(label="Přizpůsobení vedením (pahýl, sekce)…",
                       command=lambda: dlg.line_match_dialog(self.master, self._zin(),
                                                             self.model.freq_mhz, self.model.z0))
        tl.add_command(label="Napáječ a jeho ztráty…",
                       command=lambda: dlg.feedline_dialog(self.master, self._zin(),
                                                           self.model.freq_mhz, self.model.z0))
        tl.add_command(label="Vlásenka (hairpin)…", command=self.match_dialog)
        m.add_cascade(label="Nástroje", menu=tl)

        h = tk.Menu(m, tearoff=0)
        h.add_command(label="O programu / omezení modelu", command=self.show_about)
        m.add_cascade(label="Nápověda", menu=h)
        self.master.config(menu=m)
        self.master.bind("<Control-n>", lambda e: self.new_model())
        self.master.bind("<Control-o>", lambda e: self.open_file())
        self.master.bind("<Control-s>", lambda e: self.save_file())
        self.master.bind("<F5>", lambda e: self.do_calc())

    # -------------------------------------------------------------- geometrie
    def _build_geometry_tab(self):
        tab = ttk.Frame(self.nb, padding=PAD)
        self.nb.add(tab, text="Geometrie")
        left = ttk.Frame(tab)
        left.pack(side="left", fill="both", expand=True)

        head = ttk.Frame(left)
        head.pack(fill="x")
        ttk.Label(head, text="Název:").pack(side="left")
        self.var_name = tk.StringVar()
        e = ttk.Entry(head, textvariable=self.var_name, width=34)
        e.pack(side="left", padx=(4, 12))
        e.bind("<FocusOut>", lambda ev: self._set_name())
        e.bind("<Return>", lambda ev: self._set_name())
        ttk.Label(head, text="Kmitočet [MHz]:").pack(side="left")
        self.var_freq = tk.StringVar()
        ef = ttk.Entry(head, textvariable=self.var_freq, width=10)
        ef.pack(side="left", padx=4)
        ef.bind("<FocusOut>", lambda ev: self._set_freq())
        ef.bind("<Return>", lambda ev: self._set_freq())
        self.lbl_lambda = ttk.Label(head, text="")
        self.lbl_lambda.pack(side="left", padx=8)

        cols = [("n", "#", 34, "s"),
                ("x1", "X1 [m]", 78, "f"), ("y1", "Y1 [m]", 78, "f"), ("z1", "Z1 [m]", 78, "f"),
                ("x2", "X2 [m]", 78, "f"), ("y2", "Y2 [m]", 78, "f"), ("z2", "Z2 [m]", 78, "f"),
                ("radius", "poloměr [m]", 100, "f2"), ("nseg", "segm.", 52, "i"),
                ("length", "délka [m]", 82, "f")]

        self.lbl_check = ttk.Label(left, text="", foreground="#a05000",
                                   wraplength=660, justify="left")
        self.lbl_check.pack(side="bottom", fill="x", pady=(PAD, 0))
        btns = ttk.Frame(left)
        btns.pack(side="bottom", fill="x", pady=(PAD, 0))
        for text, cmd in [("Přidat drát", self.add_wire),
                          ("Duplikovat", self.dup_wire),
                          ("Smazat", self.del_wire),
                          ("Auto segmentace", self.auto_seg)]:
            ttk.Button(btns, text=text, command=cmd).pack(side="left", padx=(0, 4))

        self.tbl_wires = EditableTable(left, cols, self._wire_edited, height=12)
        self.tbl_wires.pack(fill="both", expand=True, pady=(PAD, 0))
        self.tbl_wires.tree.bind("<<TreeviewSelect>>", lambda e: self.draw_geometry())

        right = ttk.Frame(tab)
        right.pack(side="left", fill="both", expand=True, padx=(PAD, 0))
        self.var_show_current = tk.BooleanVar(value=True)
        ttk.Checkbutton(right, text="Barvit podle proudu (po výpočtu)",
                        variable=self.var_show_current,
                        command=self.draw_geometry).pack(side="bottom", anchor="w")
        self.fig_geo = Figure(figsize=(4.6, 4.2), dpi=100)
        self.ax_geo = self.fig_geo.add_subplot(111, projection="3d")
        self.cv_geo = FigureCanvasTkAgg(self.fig_geo, master=right)
        self.cv_geo.get_tk_widget().pack(fill="both", expand=True)

    # ------------------------------------------------------------- napájení
    def _build_feed_tab(self):
        tab = ttk.Frame(self.nb, padding=PAD)
        self.nb.add(tab, text="Napájení a prostředí")

        box = ttk.LabelFrame(tab, text="Zdroje", padding=PAD)
        box.pack(fill="x")
        cols = [("wire", "drát", 60, "i"), ("pos", "poloha 0-1", 90, "f"),
                ("voltage", "napětí [V]", 90, "f"), ("phase", "fáze [°]", 90, "f")]
        self.tbl_src = EditableTable(box, cols, self._src_edited, height=4)
        self.tbl_src.pack(fill="x")
        b = ttk.Frame(box)
        b.pack(fill="x", pady=(4, 0))
        ttk.Button(b, text="Přidat", command=self.add_src).pack(side="left")
        ttk.Button(b, text="Smazat", command=self.del_src).pack(side="left", padx=4)

        box2 = ttk.LabelFrame(tab, text="Soustředné zátěže", padding=PAD)
        box2.pack(fill="x", pady=PAD)
        cols2 = [("wire", "drát", 60, "i"), ("pos", "poloha 0-1", 90, "f"),
                 ("kind", "typ", 60, "s"), ("r", "R [Ω]", 80, "f"),
                 ("x", "X [Ω]", 80, "f"), ("l_uh", "L [µH]", 80, "f"),
                 ("c_pf", "C [pF]", 80, "f")]
        self.tbl_load = EditableTable(box2, cols2, self._load_edited, height=4)
        self.tbl_load.pack(fill="x")
        b2 = ttk.Frame(box2)
        b2.pack(fill="x", pady=(4, 0))
        ttk.Button(b2, text="Přidat R+jX", command=lambda: self.add_load("RX")).pack(side="left")
        ttk.Button(b2, text="Přidat RLC", command=lambda: self.add_load("RLC")).pack(side="left", padx=4)
        ttk.Button(b2, text="Smazat", command=self.del_load).pack(side="left")

        box3 = ttk.LabelFrame(tab, text="Prostředí", padding=PAD)
        box3.pack(fill="x")
        g = ttk.Frame(box3)
        g.pack(fill="x")
        ttk.Label(g, text="Zem:").grid(row=0, column=0, sticky="w")
        self.var_ground = tk.StringVar()
        cb = ttk.Combobox(g, textvariable=self.var_ground, values=list(GROUND_TYPES),
                          state="readonly", width=22)
        cb.grid(row=0, column=1, padx=4, pady=2)
        cb.bind("<<ComboboxSelected>>", lambda e: self._set_ground())

        ttk.Label(g, text="ε_r:").grid(row=0, column=2, sticky="e", padx=(12, 2))
        self.var_eps = tk.StringVar()
        ttk.Entry(g, textvariable=self.var_eps, width=8).grid(row=0, column=3)
        ttk.Label(g, text="σ [S/m]:").grid(row=0, column=4, sticky="e", padx=(12, 2))
        self.var_sig = tk.StringVar()
        ttk.Entry(g, textvariable=self.var_sig, width=10).grid(row=0, column=5)
        ttk.Button(g, text="Použít", command=self._set_ground_manual).grid(row=0, column=6, padx=6)

        ttk.Label(g, text="Materiál:").grid(row=1, column=0, sticky="w")
        self.var_mat = tk.StringVar()
        cbm = ttk.Combobox(g, textvariable=self.var_mat, values=list(MATERIALS),
                           state="readonly", width=22)
        cbm.grid(row=1, column=1, padx=4, pady=2)
        cbm.bind("<<ComboboxSelected>>", lambda e: self._set_mat())

        ttk.Label(g, text="Z₀ [Ω]:").grid(row=1, column=2, sticky="e", padx=(12, 2))
        self.var_z0 = tk.StringVar()
        ez = ttk.Entry(g, textvariable=self.var_z0, width=8)
        ez.grid(row=1, column=3)
        ez.bind("<FocusOut>", lambda e: self._set_z0())
        ez.bind("<Return>", lambda e: self._set_z0())

        ttk.Label(g, text="Ztráty zemn. systému [Ω]:").grid(row=1, column=4, sticky="e", padx=(12, 2))
        self.var_gloss = tk.StringVar()
        eg = ttk.Entry(g, textvariable=self.var_gloss, width=10)
        eg.grid(row=1, column=5)
        eg.bind("<FocusOut>", lambda e: self._set_gloss())
        eg.bind("<Return>", lambda e: self._set_gloss())

        ttk.Label(box3, foreground="#606060", justify="left", wraplength=760,
                  text="Reálná zem se v impedanci počítá zjednodušeně (obraz nad dokonalou zemí), "
                       "ve vyzařování přesněji přes Fresnelovy koeficienty. U vertikálů "
                       "napájených proti zemi zadej ztráty zemního systému ručně.").pack(
            anchor="w", pady=(6, 0))

    # --------------------------------------------------------------- výpočet
    def _build_calc_tab(self):
        tab = ttk.Frame(self.nb, padding=PAD)
        self.nb.add(tab, text="Výpočet")
        top = ttk.Frame(tab)
        top.pack(fill="x")
        ttk.Button(top, text="Spočítat (F5)", command=self.do_calc).pack(side="left")
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Label(top, text="Rozmítání od").pack(side="left")
        self.var_f1 = tk.StringVar(value="13.9")
        ttk.Entry(top, textvariable=self.var_f1, width=8).pack(side="left", padx=3)
        ttk.Label(top, text="do").pack(side="left")
        self.var_f2 = tk.StringVar(value="14.4")
        ttk.Entry(top, textvariable=self.var_f2, width=8).pack(side="left", padx=3)
        ttk.Label(top, text="MHz, kroků").pack(side="left")
        self.var_fn = tk.StringVar(value="31")
        ttk.Entry(top, textvariable=self.var_fn, width=5).pack(side="left", padx=3)
        self.var_full_sweep = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="i zisk a F/B (pomalé)",
                        variable=self.var_full_sweep).pack(side="left", padx=6)
        ttk.Button(top, text="Rozmítat", command=self.do_sweep).pack(side="left", padx=4)
        ttk.Button(top, text="Kolem kmitočtu ±5 %",
                   command=self.sweep_around).pack(side="left")
        ttk.Button(top, text="Rezonance", command=self.do_resonance).pack(side="left", padx=4)
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Label(top, text="Jádro:").pack(side="left")
        self.var_engine = tk.StringVar(value=engines.default_name())
        cb_eng = ttk.Combobox(top, textvariable=self.var_engine, width=9,
                              state="readonly", values=engines.available_engines())
        cb_eng.pack(side="left", padx=3)
        cb_eng.bind("<<ComboboxSelected>>", lambda e: self._engine_changed())
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Label(top, text="F/B výsek [°]:").pack(side="left")
        self.var_fbsec = tk.StringVar(value="0")
        ttk.Combobox(top, textvariable=self.var_fbsec, width=5, state="readonly",
                     values=["0", "60", "90", "120", "180"]).pack(side="left", padx=3)
        ttk.Button(top, text="Návrh přizpůsobení…",
                   command=self.match_dialog).pack(side="left", padx=6)

        body = ttk.Frame(tab)
        body.pack(fill="both", expand=True, pady=PAD)
        left = ttk.Frame(body)
        left.pack(side="left", fill="y")
        self.txt_res = tk.Text(left, width=42, height=17, wrap="none")
        self.txt_res.pack(fill="y", expand=True)
        self.txt_res.configure(font=("TkFixedFont", 10))

        hist = ttk.LabelFrame(left, text="Historie výpočtů", padding=4)
        hist.pack(fill="both", pady=(PAD, 0))
        hcols = ("f", "z", "swr", "g", "fb", "note")
        self.tree_hist = ttk.Treeview(hist, columns=hcols, show="headings", height=6)
        for k, t, w_ in (("f", "MHz", 62), ("z", "Z [Ω]", 118), ("swr", "PSV", 46),
                         ("g", "dBi", 50), ("fb", "F/B", 46), ("note", "model", 120)):
            self.tree_hist.heading(k, text=t)
            self.tree_hist.column(k, width=w_, anchor="e" if k != "note" else "w")
        self.tree_hist.pack(fill="both", expand=True)
        hb = ttk.Frame(hist)
        hb.pack(fill="x", pady=(3, 0))
        ttk.Button(hb, text="Vyčistit", command=self.clear_history).pack(side="left")
        ttk.Button(hb, text="Uložit CSV…", command=self.save_history).pack(side="left", padx=4)
        self.history: List[dict] = []

        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)
        self.fig_sweep = Figure(figsize=(5.8, 4.0), dpi=100)
        self.cv_sweep = FigureCanvasTkAgg(self.fig_sweep, master=right)
        tb = NavigationToolbar2Tk(self.cv_sweep, right, pack_toolbar=False)
        tb.update()
        tb.pack(side="bottom", fill="x")
        self.cv_sweep.get_tk_widget().pack(fill="both", expand=True)

    # --------------------------------------------------------------- diagram
    def _build_pattern_tab(self):
        tab = ttk.Frame(self.nb, padding=PAD)
        self.nb.add(tab, text="Vyzařovací diagram")
        top = ttk.Frame(tab)
        top.pack(fill="x")
        ttk.Label(top, text="Vodorovný řez v elevaci [°]:").pack(side="left")
        self.var_el = tk.StringVar(value="auto")
        ttk.Entry(top, textvariable=self.var_el, width=7).pack(side="left", padx=3)
        ttk.Label(top, text="Svislý řez v azimutu [°]:").pack(side="left", padx=(10, 0))
        self.var_az = tk.StringVar(value="auto")
        ttk.Entry(top, textvariable=self.var_az, width=7).pack(side="left", padx=3)
        ttk.Label(top, text="Rozsah [dB]:").pack(side="left", padx=(10, 0))
        self.var_range = tk.StringVar(value="40")
        ttk.Combobox(top, textvariable=self.var_range, width=4, state="readonly",
                     values=["20", "30", "40", "50", "60"]).pack(side="left", padx=3)
        ttk.Label(top, text="Polarizace:").pack(side="left", padx=(10, 0))
        self.var_pol = tk.StringVar(value="Celkem")
        ttk.Combobox(top, textvariable=self.var_pol, width=13, state="readonly",
                     values=["Celkem", "Svislá (V)", "Vodorovná (H)",
                             "Celkem + V + H"]).pack(side="left", padx=3)
        self.var_marks = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="značky −3 dB", variable=self.var_marks).pack(side="left", padx=(10, 0))
        ttk.Button(top, text="Překreslit", command=self.draw_pattern).pack(side="left", padx=8)
        ttk.Button(top, text="3D diagram", command=self.show_3d).pack(side="left", padx=4)
        ttk.Button(top, text="Uložit obrázek…", command=self.save_pattern).pack(side="left")
        for v in (self.var_range, self.var_pol):
            v.trace_add("write", lambda *a: self.draw_pattern())
        self.var_marks.trace_add("write", lambda *a: self.draw_pattern())
        self._cursor_art = []

        self.fig_pat = Figure(figsize=(8.6, 4.2), dpi=100)
        self.cv_pat = FigureCanvasTkAgg(self.fig_pat, master=tab)
        tbp = NavigationToolbar2Tk(self.cv_pat, tab, pack_toolbar=False)
        tbp.update()
        tbp.pack(side="bottom", fill="x")
        self.cv_pat.get_tk_widget().pack(fill="both", expand=True, pady=PAD)

    # ---------------------------------------------------------- optimalizace
    def _build_opt_tab(self):
        tab = ttk.Frame(self.nb, padding=PAD)
        self.nb.add(tab, text="Optimalizace")

        left = ttk.Frame(tab)
        left.pack(side="left", fill="both", expand=True)

        pbox = ttk.LabelFrame(left, text="Co se smí měnit", padding=PAD)
        pbox.pack(fill="both", expand=True)
        cols = [("kind", "veličina", 130, "s"), ("wires", "dráty", 64, "s"),
                ("lo", "min", 84, "f"), ("hi", "max", 84, "f"),
                ("step", "krok", 60, "f"), ("link", "vazba", 78, "s"),
                ("now", "nyní", 84, "f")]
        self.tbl_par = EditableTable(pbox, cols, self._par_edited, height=9)
        self.tbl_par.pack(fill="both", expand=True)
        pb = ttk.Frame(pbox)
        pb.pack(fill="x", pady=(4, 0))
        ttk.Button(pb, text="Doplnit obvyklé", command=self.fill_params).pack(side="left")
        ttk.Button(pb, text="Přidat…", command=self.add_param).pack(side="left", padx=4)
        ttk.Button(pb, text="Smazat", command=self.del_param).pack(side="left")
        ttk.Button(pb, text="Vyprázdnit", command=self.clear_params).pack(side="left", padx=4)

        obox = ttk.LabelFrame(left, text="Cíl", padding=PAD)
        obox.pack(fill="x", pady=PAD)
        r = 0
        ttk.Label(obox, text="Kmitočty [MHz], oddělené čárkou:").grid(row=r, column=0,
                                                                     sticky="w", columnspan=2)
        self.var_ofreqs = tk.StringVar(value="")
        ttk.Entry(obox, textvariable=self.var_ofreqs, width=32).grid(row=r, column=2,
                                                                    columnspan=3, sticky="w")
        r += 1
        self.opt_vars = {}
        specs = [("w_gain", "váha zisku", "1.0"), ("w_fb", "váha F/B", "0.5"),
                 ("w_swr", "váha PSV", "2.0"), ("target_swr", "cílové PSV", "1.5"),
                 ("w_r", "váha R", "0.0"), ("target_r", "cílové R [Ω]", "50"),
                 ("w_x", "váha X", "0.0"), ("target_x", "cílové X [Ω]", "0")]
        for i, (key, label, default) in enumerate(specs):
            row, col = r + i // 2, (i % 2) * 3
            ttk.Label(obox, text=label + ":").grid(row=row, column=col, sticky="e", padx=(0, 4))
            v = tk.StringVar(value=default)
            ttk.Entry(obox, textvariable=v, width=8).grid(row=row, column=col + 1,
                                                          sticky="w", padx=(0, 12))
            self.opt_vars[key] = v

        rbox = ttk.LabelFrame(left, text="Hledání", padding=PAD)
        rbox.pack(fill="x")
        ttk.Label(rbox, text="populace:").grid(row=0, column=0, sticky="e")
        self.var_pop = tk.StringVar(value="20")
        ttk.Entry(rbox, textvariable=self.var_pop, width=6).grid(row=0, column=1, padx=(2, 10))
        ttk.Label(rbox, text="generací:").grid(row=0, column=2, sticky="e")
        self.var_gen = tk.StringVar(value="20")
        ttk.Entry(rbox, textvariable=self.var_gen, width=6).grid(row=0, column=3, padx=(2, 10))
        self.var_polish = tk.BooleanVar(value=True)
        ttk.Checkbutton(rbox, text="doladit", variable=self.var_polish).grid(row=0, column=4)
        ttk.Button(rbox, text="Spustit", command=self.do_optimize).grid(row=0, column=5, padx=8)
        ttk.Button(rbox, text="Zastavit", command=self.worker.stop).grid(row=0, column=6)
        self.pb_opt = ttk.Progressbar(rbox, length=220, mode="determinate")
        self.pb_opt.grid(row=1, column=0, columnspan=7, sticky="ew", pady=(6, 0))

        right = ttk.Frame(tab)
        right.pack(side="left", fill="both", expand=True, padx=(PAD, 0))
        self.txt_opt = tk.Text(right, height=14, width=52, wrap="word")
        self.txt_opt.pack(fill="both", expand=True)
        self.txt_opt.configure(font=("TkFixedFont", 10))
        self.fig_opt = Figure(figsize=(4.6, 2.3), dpi=100)
        self.ax_opt = self.fig_opt.add_subplot(111)
        self.cv_opt = FigureCanvasTkAgg(self.fig_opt, master=right)
        self.cv_opt.get_tk_widget().pack(fill="both", expand=True, pady=(PAD, 0))
        bb = ttk.Frame(right)
        bb.pack(fill="x")
        ttk.Button(bb, text="Přijmout výsledek do modelu",
                   command=self.accept_opt).pack(side="left")
        self._opt_candidate: Optional[Model] = None

    # ==================================================================
    #  obnova zobrazení
    # ==================================================================
    def refresh_all(self):
        m = self.model
        self.var_name.set(m.name)
        self.var_freq.set(f"{m.freq_mhz:g}")
        self.lbl_lambda.set_text = None
        self.lbl_lambda.configure(text=f"λ = {m.wavelength:.3f} m")
        self.var_mat.set(m.material)
        self.var_z0.set(f"{m.z0:g}")
        self.var_gloss.set(f"{m.ground_loss_r:g}")
        self.var_eps.set(f"{m.ground.eps_r:g}")
        self.var_sig.set(f"{m.ground.sigma:g}")
        self.var_ground.set(self._ground_name())
        self.refresh_wires()
        self.refresh_src()
        self.refresh_loads()
        self.refresh_params()
        self.draw_geometry()
        self.master.title(f"AntOpt – {m.name}" + (f"  [{os.path.basename(self.path)}]"
                                                  if self.path else ""))

    def _ground_name(self) -> str:
        g = self.model.ground
        if g.kind == "free":
            return "volný prostor"
        if g.kind == "perfect":
            return "dokonalá zem"
        for name, spec in GROUND_TYPES.items():
            if isinstance(spec, tuple) and abs(spec[0] - g.eps_r) < 1e-6 and abs(spec[1] - g.sigma) < 1e-9:
                return name
        return "průměrná"

    def refresh_wires(self):
        rows = [[i + 1, w.x1, w.y1, w.z1, w.x2, w.y2, w.z2, w.radius, w.nseg, w.length]
                for i, w in enumerate(self.model.wires)]
        self.tbl_wires.set_rows(rows)
        msgs = self.model.validate()
        self.lbl_check.configure(text="⚠ " + "  ".join(msgs) if msgs else "✓ Model vypadá v pořádku.",
                                 foreground="#a05000" if msgs else "#2e7d32")

    def refresh_src(self):
        self.tbl_src.set_rows([[s.wire + 1, s.pos, s.voltage, s.phase]
                               for s in self.model.sources])

    def refresh_loads(self):
        self.tbl_load.set_rows([[l.wire + 1, l.pos, l.kind, l.r, l.x, l.l_uh, l.c_pf]
                                for l in self.model.loads])

    def refresh_params(self):
        rows = []
        for p in self.opt_params:
            try:
                now = read_param(self.model, p)
            except (IndexError, ValueError):
                now = float("nan")
            rows.append([p.kind, "+".join(str(i + 1) for i in p.wires), p.lo, p.hi,
                         p.step, p.linked_to() or "—", now])
        self.tbl_par.set_rows(rows)

    # ==================================================================
    #  editace modelu
    # ==================================================================
    def _set_name(self):
        self.model.name = self.var_name.get()
        self.master.title(f"AntOpt – {self.model.name}")

    def _set_freq(self):
        try:
            self.model.freq_mhz = float(self.var_freq.get().replace(",", "."))
        except ValueError:
            self.var_freq.set(f"{self.model.freq_mhz:g}")
            return
        self.lbl_lambda.configure(text=f"λ = {self.model.wavelength:.3f} m")
        self.refresh_wires()

    def _set_z0(self):
        try:
            self.model.z0 = float(self.var_z0.get().replace(",", "."))
        except ValueError:
            self.var_z0.set(f"{self.model.z0:g}")

    def _set_gloss(self):
        try:
            self.model.ground_loss_r = float(self.var_gloss.get().replace(",", "."))
        except ValueError:
            self.var_gloss.set(f"{self.model.ground_loss_r:g}")

    def _set_mat(self):
        self.model.material = self.var_mat.get()

    def _set_ground(self):
        self.model.ground = Ground.from_name(self.var_ground.get())
        self.var_eps.set(f"{self.model.ground.eps_r:g}")
        self.var_sig.set(f"{self.model.ground.sigma:g}")
        self.refresh_wires()
        self.draw_geometry()

    def _set_ground_manual(self):
        try:
            eps = float(self.var_eps.get().replace(",", "."))
            sig = float(self.var_sig.get().replace(",", "."))
        except ValueError:
            messagebox.showerror("Chyba", "ε_r a σ musí být čísla.")
            return
        if self.model.ground.kind == "free":
            self.model.ground = Ground("real", eps, sig)
            self.var_ground.set("průměrná")
        else:
            self.model.ground.eps_r = eps
            self.model.ground.sigma = sig
        self.refresh_wires()

    def _wire_edited(self, idx, key, val):
        if key == "n" or key == "length":
            self.refresh_wires()
            return
        w = self.model.wires[idx]
        if key == "nseg":
            w.nseg = max(1, int(val))
        else:
            setattr(w, key, float(val))
        self.refresh_wires()
        self.tbl_wires.select(idx)
        self.draw_geometry()

    def _src_edited(self, idx, key, val):
        s = self.model.sources[idx]
        if key == "wire":
            s.wire = max(0, min(len(self.model.wires) - 1, int(val) - 1))
        elif key == "pos":
            s.pos = min(1.0, max(0.0, float(val)))
        else:
            setattr(s, key, float(val))
        self.refresh_src()
        self.draw_geometry()

    def _load_edited(self, idx, key, val):
        l = self.model.loads[idx]
        if key == "wire":
            l.wire = max(0, min(len(self.model.wires) - 1, int(val) - 1))
        elif key == "pos":
            l.pos = min(1.0, max(0.0, float(val)))
        elif key == "kind":
            l.kind = "RLC" if str(val).upper().startswith("RLC") else "RX"
        else:
            setattr(l, key, float(val))
        self.refresh_loads()

    def _par_edited(self, idx, key, val):
        p = self.opt_params[idx]
        if key in ("lo", "hi", "step"):
            setattr(p, key, float(val))
        elif key == "link":
            self._set_link(idx, str(val))
        elif key == "kind" and str(val) in PARAM_KINDS:
            p.kind = str(val)
        elif key == "wires":
            try:
                p.wires = [int(x) - 1 for x in str(val).replace(" ", "").split("+")]
            except ValueError:
                pass
        self.refresh_params()

    # ---------------------------------------------------------- tlačítka
    def add_wire(self):
        lam = self.model.wavelength
        self.model.wires.append(Wire(-lam / 4, 0, lam / 2, lam / 4, 0, lam / 2,
                                     radius=0.001, nseg=21))
        self.refresh_wires()
        self.tbl_wires.select(len(self.model.wires) - 1)
        self.draw_geometry()

    def dup_wire(self):
        i = self.tbl_wires.selected_index()
        if i is None:
            return
        w = self.model.wires[i]
        self.model.wires.insert(i + 1, Wire(w.x1, w.y1, w.z1, w.x2, w.y2, w.z2,
                                            w.radius, w.nseg, w.name))
        self.refresh_wires()
        self.draw_geometry()

    def del_wire(self):
        i = self.tbl_wires.selected_index()
        if i is None or len(self.model.wires) <= 1:
            return
        del self.model.wires[i]
        for s in self.model.sources:
            if s.wire >= len(self.model.wires):
                s.wire = len(self.model.wires) - 1
            elif s.wire > i:
                s.wire -= 1
        for l in self.model.loads:
            if l.wire >= len(self.model.wires):
                l.wire = len(self.model.wires) - 1
            elif l.wire > i:
                l.wire -= 1
        self.refresh_all()

    def auto_seg(self):
        self.model.auto_segment()
        self.refresh_wires()

    def add_src(self):
        self.model.sources.append(Source(0, 0.5, 1.0))
        self.refresh_src()

    def del_src(self):
        i = self.tbl_src.selected_index()
        if i is not None and len(self.model.sources) > 1:
            del self.model.sources[i]
            self.refresh_src()

    def add_load(self, kind):
        self.model.loads.append(Load(0, 0.5, kind))
        self.refresh_loads()

    def del_load(self):
        i = self.tbl_load.selected_index()
        if i is not None:
            del self.model.loads[i]
            self.refresh_loads()

    # ==================================================================
    #  vykreslení geometrie
    # ==================================================================
    def draw_geometry(self):
        ax = self.ax_geo
        ax.clear()
        m = self.model
        sel = self.tbl_wires.selected_index()

        geo = None
        if self.var_show_current.get() and self.result is not None:
            try:
                geo = engines.get(self.result.engine).geometry(self.result)
            except Exception:
                geo = None
            if geo is not None and geo.magnitude.size:
                mx = float(np.max(geo.magnitude))
                geo = type(geo)(geo.a, geo.b,
                                geo.magnitude / (mx if mx > 0 else 1.0))
            else:
                geo = None

        if geo is not None:
            cmap = matplotlib.colormaps["viridis"]
            for si in range(len(geo.a)):
                a, b = geo.a[si], geo.b[si]
                ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                        color=cmap(float(geo.magnitude[si])), linewidth=2.4)
        else:
            for i, w in enumerate(m.wires):
                col = "#d32f2f" if i == sel else "#1565c0"
                ax.plot([w.x1, w.x2], [w.y1, w.y2], [w.z1, w.z2],
                        color=col, linewidth=2.6 if i == sel else 1.8)

        for s in m.sources:
            if 0 <= s.wire < len(m.wires):
                w = m.wires[s.wire]
                p = w.a + s.pos * (w.b - w.a)
                ax.scatter([p[0]], [p[1]], [p[2]], color="#e53935", s=45, marker="o")
        for l in m.loads:
            if 0 <= l.wire < len(m.wires):
                w = m.wires[l.wire]
                p = w.a + l.pos * (w.b - w.a)
                ax.scatter([p[0]], [p[1]], [p[2]], color="#8e24aa", s=35, marker="s")

        lo, hi = m.bounds()
        if m.ground.kind != "free":
            lo = np.minimum(lo, [lo[0], lo[1], 0.0])
        c = 0.5 * (lo + hi)
        r = max(np.max(hi - lo), 1e-3) * 0.6
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(min(c[2] - r, 0.0 if m.ground.kind != "free" else c[2] - r), c[2] + r)
        if m.ground.kind != "free":
            gx = np.array([c[0] - r, c[0] + r])
            gy = np.array([c[1] - r, c[1] + r])
            GX, GY = np.meshgrid(gx, gy)
            ax.plot_surface(GX, GY, np.zeros_like(GX), alpha=0.15, color="#795548")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.set_title("Geometrie" + ("  (barva = proud)" if geo is not None else ""))
        self.fig_geo.tight_layout()
        self.cv_geo.draw_idle()

    # ==================================================================
    #  výpočty
    # ==================================================================
    def set_status(self, text):
        self.status.configure(text=text)

    def do_calc(self):
        msgs = self.model.validate()
        hard = [m for m in msgs if "nulovou délku" in m or "poloměr" in m]
        if hard:
            messagebox.showerror("Model není v pořádku", "\n".join(hard))
            return
        model = self.model.copy()
        self.set_status("Počítám…")

        try:
            sec = float(self.var_fbsec.get())
        except (ValueError, AttributeError):
            sec = 0.0

        eng = self.engine_name()

        def job(cancel, progress):
            r = analyse(model, fb_sector_deg=sec, engine=eng)
            try:
                r_q = q_estimate(model, engine=eng)
            except Exception:
                r_q = (float("nan"), False, float("nan"), float("nan"))
            return r, r_q

        def done(payload):
            res, qf = payload
            self.result = res
            self._last_q = qf
            self._show_result(res)
            self.add_history(res)
            self.draw_geometry()
            self.draw_pattern()
            self.set_status("Hotovo.")

        self.worker.run(job, done, on_error=self._err)

    def _err(self, tb):
        self.set_status("Chyba.")
        messagebox.showerror("Chyba výpočtu", tb[-2000:])

    def _show_result(self, r: Result):
        m = self.model
        z = r.zin
        g = abs((z - m.z0) / (z + m.z0))
        lines = [
            f"Jádro           {r.engine}",
            f"Kmitočet        {r.freq_mhz:.4f} MHz",
            f"Vlnová délka    {m.wavelength:.3f} m",
            "",
            f"Vstupní impedance  {z.real:.2f} {z.imag:+.2f} j Ω",
            f"PSV (Z₀={m.z0:g} Ω)      {r.swr:.3f}",
            f"Činitel odrazu     {g:.4f}  ({20 * math.log10(max(g, 1e-9)):.1f} dB)",
            "",
            f"Zisk               {r.gain_dbi:.2f} dBi   ({r.gain_dbi - 2.15:.2f} dBd)",
            f"Předozadní poměr   {r.fb_db:.1f} dB",
            f"Poměr před/stranou {r.fs_db:.1f} dB",
            f"Elevace maxima     {r.elevation_deg:.1f} °",
            f"Azimut maxima      {r.azimuth_deg:.1f} °",
            f"Šířka svazku H/V   {r.beam_h_deg:.1f}° / {r.beam_v_deg:.1f}°",
            f"Účinnost (integr.) {100 * r.efficiency:.1f} %",
        ]
        qi = getattr(self, "_last_q", None)
        if qi and np.isfinite(qi[0]):
            q, ok, qlo, qhi = qi
            lines.insert(13, f"Činitel jakosti Q  {q:.1f}" if ok
                         else f"Činitel jakosti Q  {qlo:.0f}–{qhi:.0f} (nestabilní)")
        if r.solution is not None:
            lines += ["", f"Segmentů           {r.solution.mesh.nseg}",
                      f"Neznámých          {r.solution.mesh.nbasis}"]
        self.txt_res.delete("1.0", "end")
        self.txt_res.insert("1.0", "\n".join(lines))

    # ------------------------------------------------------------- rozmítání
    def sweep_around(self):
        f = self.model.freq_mhz
        self.var_f1.set(f"{f * 0.95:.4f}")
        self.var_f2.set(f"{f * 1.05:.4f}")
        self.do_sweep()

    def do_sweep(self):
        try:
            f1 = float(self.var_f1.get().replace(",", "."))
            f2 = float(self.var_f2.get().replace(",", "."))
            n = int(self.var_fn.get())
        except ValueError:
            messagebox.showerror("Chyba", "Meze rozmítání musí být čísla.")
            return
        model = self.model.copy()
        full = self.var_full_sweep.get()

        eng = self.engine_name()

        def job(cancel, progress):
            def cb(i, total):
                progress(f"Rozmítání {i}/{total}")
                return not cancel.is_set()
            return sweep(model, f1, f2, n, full=full, progress=cb, engine=eng)

        def done(res):
            self.sweep_results = res
            self.draw_sweep()
            bw = bandwidth(res, 2.0)
            if bw:
                self.set_status(f"Rozmítání hotovo. Šířka pásma PSV<2: "
                                f"{bw[0]:.3f}–{bw[1]:.3f} MHz ({bw[2]:.0f} kHz)")
            else:
                self.set_status("Rozmítání hotovo. PSV nikde neklesne pod 2.")

        self.worker.run(job, done, on_error=self._err,
                        on_progress=lambda t: self.set_status(t))

    def draw_sweep(self):
        self.fig_sweep.clear()
        rs = self.sweep_results
        if not rs:
            return
        f = np.array([r.freq_mhz for r in rs])
        s = np.array([r.swr for r in rs])
        rr = np.array([r.zin.real for r in rs])
        xx = np.array([r.zin.imag for r in rs])
        has_gain = np.isfinite([r.gain_dbi for r in rs]).all()

        n = 3 if has_gain else 2
        ax1 = self.fig_sweep.add_subplot(n, 1, 1)
        ax1.plot(f, s, color="#1565c0")
        ax1.axhline(2.0, color="#bbb", ls="--", lw=0.8)
        ax1.axhline(1.5, color="#ddd", ls="--", lw=0.8)
        ax1.set_ylabel("PSV")
        ax1.set_ylim(1, min(6, max(3, np.nanmax(s) * 1.1)))
        ax1.grid(alpha=0.3)

        ax2 = self.fig_sweep.add_subplot(n, 1, 2, sharex=ax1)
        ax2.plot(f, rr, label="R", color="#2e7d32")
        ax2.plot(f, xx, label="X", color="#c62828")
        ax2.axhline(0, color="#bbb", lw=0.8)
        ax2.set_ylabel("Z [Ω]")
        ax2.legend(fontsize=8, loc="best")
        ax2.grid(alpha=0.3)

        if has_gain:
            ax3 = self.fig_sweep.add_subplot(n, 1, 3, sharex=ax1)
            ax3.plot(f, [r.gain_dbi for r in rs], label="zisk [dBi]", color="#6a1b9a")
            ax3.plot(f, [r.fb_db for r in rs], label="F/B [dB]", color="#ef6c00")
            ax3.legend(fontsize=8, loc="best")
            ax3.grid(alpha=0.3)
            ax3.set_xlabel("f [MHz]")
        else:
            ax2.set_xlabel("f [MHz]")
        self.fig_sweep.tight_layout()
        self.cv_sweep.draw_idle()

    # ---------------------------------------------------------- přizpůsobení
    def match_dialog(self):
        """Návrh vlásenky (hairpin) pro napájecí bod."""
        from .match import (design_hairpin, tune_driven_for_hairpin,
                            swr_with_hairpin, required_reactance)
        dlg = tk.Toplevel(self.master)
        dlg.title("Návrh přizpůsobení — vlásenka (hairpin)")
        dlg.transient(self.master)
        top = ttk.Frame(dlg, padding=PAD)
        top.pack(fill="x")

        ttk.Label(top, text="Zářič je drát č.:").grid(row=0, column=0, sticky="e")
        v_wire = tk.StringVar(value=str(self.model.sources[0].wire + 1
                                        if self.model.sources else 1))
        ttk.Entry(top, textvariable=v_wire, width=6).grid(row=0, column=1, padx=(2, 12))
        ttk.Label(top, text="Cílová impedance [Ω]:").grid(row=0, column=2, sticky="e")
        v_z0 = tk.StringVar(value=f"{self.model.z0:g}")
        ttk.Entry(top, textvariable=v_z0, width=6).grid(row=0, column=3, padx=(2, 12))

        ttk.Label(top, text="Vodiče vlásenky Ø [mm]:").grid(row=1, column=0, sticky="e")
        v_d = tk.StringVar(value="10")
        ttk.Entry(top, textvariable=v_d, width=6).grid(row=1, column=1, padx=(2, 12))
        ttk.Label(top, text="Rozteč [mm]:").grid(row=1, column=2, sticky="e")
        v_s = tk.StringVar(value="60")
        ttk.Entry(top, textvariable=v_s, width=6).grid(row=1, column=3, padx=(2, 12))

        v_tune = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="zkrátit zářič na přesnou podmínku vlásenky",
                        variable=v_tune).grid(row=2, column=0, columnspan=4,
                                              sticky="w", pady=(4, 0))

        txt = tk.Text(dlg, width=76, height=22, wrap="word", font=("TkFixedFont", 10))
        txt.pack(fill="both", expand=True, padx=PAD, pady=PAD)
        state = {"model": None}

        def run():
            txt.delete("1.0", "end")
            try:
                wi = int(v_wire.get()) - 1
                z0 = float(v_z0.get().replace(",", "."))
                dia = float(v_d.get().replace(",", "."))
                sp = float(v_s.get().replace(",", "."))
            except ValueError:
                txt.insert("end", "Zadané hodnoty musí být čísla.\n")
                return
            base = self.model.copy()
            base.z0 = z0
            z_now = solve(base).zin
            txt.insert("end", f"Napájecí bod nyní: {z_now.real:.1f} {z_now.imag:+.1f} j Ω  "
                              f"(PSV {swr_from_z(z_now, z0):.2f})\n")
            if z_now.real >= z0:
                txt.insert("end", f"\nOdpor je už {z_now.real:.0f} Ω ≥ {z0:g} Ω — "
                                  f"vlásenka nepomůže. Zkus přímé napájení, "
                                  f"gama článek nebo transformační vedení.\n")
                return
            need = required_reactance(z_now.real, z0)
            txt.insert("end", f"Pro vlásenku potřebuje zářič X ≈ {need:.1f} Ω "
                              f"(tj. být kapacitní).\n\n")
            model = base
            if v_tune.get():
                try:
                    model, z_t = tune_driven_for_hairpin(base, wi, z0)
                except (ValueError, IndexError) as e:
                    txt.insert("end", f"Ladění zářiče selhalo: {e}\n")
                    return
                dl = model.wires[wi].length - base.wires[wi].length
                txt.insert("end", f"Zářič {'zkrátit' if dl < 0 else 'prodloužit'} o "
                                  f"{abs(dl) * 1000:.0f} mm  →  nová délka "
                                  f"{model.wires[wi].length:.4f} m\n")
                txt.insert("end", f"Nová impedance: {z_t.real:.1f} {z_t.imag:+.1f} j Ω\n\n")
            else:
                z_t = z_now
            hp = design_hairpin(z_t, model.freq_mhz, z0, sp, dia)
            if hp is None:
                txt.insert("end", "Vlásenku pro tuto impedanci navrhnout nelze.\n")
                return
            txt.insert("end", hp.report() + "\n\n")
            f0 = model.freq_mhz
            fs = np.linspace(f0 * 0.975, f0 * 1.025, 11)
            txt.insert("end", "  f [MHz]    Z po vlásence        PSV\n")
            for f, z, s in swr_with_hairpin(model, hp, fs, z0):
                txt.insert("end", f"  {f:8.3f}   {z.real:6.1f} {z.imag:+6.1f} j     {s:.2f}\n")
            state["model"] = model
            txt.see("1.0")

        def accept():
            if state["model"] is None:
                messagebox.showinfo("Nic k převzetí", "Nejdřív spusť výpočet.", parent=dlg)
                return
            self.model = state["model"]
            self.result = None
            self.refresh_all()
            dlg.destroy()
            self.set_status("Zářič upraven pro vlásenku.")

        b = ttk.Frame(dlg, padding=(PAD, 0, PAD, PAD))
        b.pack(fill="x")
        ttk.Button(b, text="Spočítat", command=run).pack(side="left")
        ttk.Button(b, text="Převzít upravený zářič do modelu",
                   command=accept).pack(side="left", padx=6)
        ttk.Button(b, text="Zavřít", command=dlg.destroy).pack(side="right")
        run()

    # -------------------------------------------------------------- diagramy
    def draw_pattern(self):
        if self.result is None:
            return
        eng = engines.get(self.result.engine)
        try:
            rng = float(self.var_range.get().replace(",", "."))
        except ValueError:
            rng = 40.0
        el_txt = self.var_el.get().strip().lower()
        az_txt = self.var_az.get().strip().lower()
        try:
            el = (self.result.elevation_deg if el_txt in ("", "auto")
                  else float(el_txt.replace(",", ".")))
            az = (self.result.azimuth_deg if az_txt in ("", "auto")
                  else float(az_txt.replace(",", ".")))
        except ValueError:
            el, az = self.result.elevation_deg, self.result.azimuth_deg

        pol = self.var_pol.get()
        if pol.startswith("Svislá"):
            comps = [("total", "V", "#c62828")]
        elif pol.startswith("Vodorovná"):
            comps = [("total", "H", "#2e7d32")]
        elif pol.startswith("Celkem +"):
            comps = [("total", "celkem", "#1565c0"), ("v", "V", "#c62828"),
                     ("h", "H", "#2e7d32")]
        else:
            comps = [("total", "celkem", "#1565c0")]
        if pol.startswith("Svislá"):
            comps = [("v", "svislá (V)", "#c62828")]
        elif pol.startswith("Vodorovná"):
            comps = [("h", "vodorovná (H)", "#2e7d32")]

        grounded = self.model.ground.kind != "free"
        gmax = self.result.gain_dbi
        self.fig_pat.clear()
        self._cursor_art = []

        ang_h, pat_h = eng.cut_azimuth(self.result, el, 361)
        ax1 = self.fig_pat.add_subplot(121, projection="polar")
        self._polar_mmana(
            ax1, ang_h, pat_h, comps, gmax, rng, kind="azimut",
            title=f"Vodorovný diagram — elevace {el:.1f}°",
            mark_angle=self.result.azimuth_deg)

        ang_v, pat_v = eng.cut_vertical(self.result, az, 361)
        ax2 = self.fig_pat.add_subplot(122, projection="polar")
        self._polar_mmana(
            ax2, ang_v, pat_v, comps, gmax, rng,
            kind="vert_gnd" if grounded else "vert_free",
            title=f"Svislý diagram — azimut {az:.1f}°",
            mark_angle=self.result.elevation_deg if grounded else None)

        info = (f"max {gmax:.2f} dBi ({gmax - 2.15:.2f} dBd)   "
                f"F/B {self.result.fb_db:.1f} dB   "
                f"F/S {self.result.fs_db:.1f} dB   "
                f"úhel vyzařování {self.result.elevation_deg:.1f}°   "
                f"svazek {self.result.beam_h_deg:.0f}°/{self.result.beam_v_deg:.0f}°   "
                f"{self.result.freq_mhz:.4f} MHz   [{self.result.engine}]")
        self.fig_pat.text(0.5, 0.012, info, ha="center", fontsize=9, color="#333")
        bottom = 0.075
        if len(comps) > 1:
            self.fig_pat.legend(handles=[
                matplotlib.lines.Line2D([], [], color=c, lw=1.8, label=lbl)
                for _, lbl, c in comps], loc="lower center", ncol=3,
                frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.042))
            bottom = 0.115
        self.fig_pat.tight_layout(rect=(0, bottom, 1, 0.93))
        self._pat_axes = [(ax1, ang_h, pat_h, "azimut", gmax, rng),
                          (ax2, ang_v, pat_v, "vert", gmax, rng)]
        if not hasattr(self, "_pat_cid"):
            self._pat_cid = self.cv_pat.mpl_connect("button_press_event",
                                                    self._pattern_click)
        self.cv_pat.draw_idle()

    # ------------------------------------------------------------------
    def _polar_mmana(self, ax, ang_deg, pattern, comps, gmax, rng, kind,
                     title, mark_angle=None):
        """Polární diagram ve stylu MMANA: prstence po 10 dB, obzor, značky."""
        top = gmax
        ang = np.radians(ang_deg)

        if kind == "azimut":
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ticks = np.arange(0, 360, 30)
            labels = [f"{int(t)}°" for t in ticks]
            ax.set_rlabel_position(115)
        else:
            ax.set_theta_zero_location("E")
            ax.set_theta_direction(1)
            if kind == "vert_gnd":
                ax.set_thetamin(0)
                ax.set_thetamax(180)
                ticks = np.arange(0, 181, 15)
                labels = [f"{int(t if t <= 90 else 180 - t)}°" for t in ticks]
                ax.set_rlabel_position(90)
            else:
                ticks = np.arange(0, 360, 30)
                labels = [f"{int(t)}°" for t in ticks]
                ax.set_rlabel_position(115)
        ax.set_thetagrids(ticks, labels, fontsize=7.5)

        # radiální mřížka po 10 dB, popisky relativně k maximu
        step = 10.0
        rticks = np.arange(0, rng + 0.1, step)
        ax.set_ylim(0, rng)
        ax.set_yticks(rticks)
        ax.set_yticklabels([("0 dB" if abs(t - rng) < 1e-6 else f"{int(t - rng)}")
                            for t in rticks], fontsize=7, color="#777")
        ax.grid(True, color="#c8c8c8", lw=0.7, alpha=0.9)
        ax.set_axisbelow(True)

        # složky se kreslí odzadu, celkový průběh navrch
        for n_i, (key, lbl, col) in enumerate(reversed(comps)):
            g = pattern.component(key)
            r = np.clip(g - top + rng, 0, None)
            last = n_i == len(comps) - 1
            ax.plot(ang, r, color=col, lw=2.0 if last and len(comps) > 1 else 1.7,
                    zorder=4 + n_i)
            if len(comps) == 1:
                ax.fill(ang, r, color=col, alpha=0.10, zorder=3)

        # obzor jako vodorovná čára
        if kind == "vert_gnd":
            for a in (0.0, math.pi):
                ax.plot([a, a], [0, rng], color="#795548", lw=2.2,
                        solid_capstyle="butt", zorder=5)

        # směr maxima + značky −3 dB
        gt = pattern.component(comps[0][0])
        if mark_angle is not None:
            am = math.radians(mark_angle if kind != "azimut" else mark_angle)
            ax.plot([am, am], [0, rng], color="#e53935", lw=1.0, ls="--",
                    alpha=0.85, zorder=6)
        if self.var_marks.get():
            gmx = float(np.max(gt))
            closed = kind in ("azimut", "vert_free")
            for a3 in self._half_power_angles(ang_deg, gt, gmx, closed=closed):
                r3 = np.clip(gmx - 3.0 - top + rng, 0, None)
                ax.plot([math.radians(a3)], [r3], marker="o", ms=4.2,
                        color="#e53935", zorder=7)

        ax.set_title(title, fontsize=10, pad=12)

    @staticmethod
    def _half_power_angles(ang_deg, gain, gmax, closed=False):
        """Úhly, kde zisk prochází −3 dB pod maximem hlavního laloku.

        ``closed`` = řez je uzavřená smyčka (plných 360°), hledá se i přes okraj.
        """
        out = []
        thr = gmax - 3.0
        n = len(gain)
        i0 = int(np.argmax(gain))
        for direction in (1, -1):
            i = i0
            for _ in range(n - 1):
                j = i + direction
                if closed:
                    j %= n
                elif j < 0 or j >= n:
                    break
                if (gain[i] - thr) * (gain[j] - thr) <= 0:
                    denom = gain[j] - gain[i]
                    t = (thr - gain[i]) / denom if denom else 0.0
                    a_i, a_j = float(ang_deg[i]), float(ang_deg[j])
                    if closed and abs(a_j - a_i) > 180.0:
                        a_j += 360.0 if a_j < a_i else -360.0
                    out.append((a_i + t * (a_j - a_i)) % 360.0)
                    break
                i = j
        return out

    def _pattern_click(self, event):
        """Měřicí vektor jako v MMANA — klik ukáže úhel a zisk."""
        if event.inaxes is None or not hasattr(self, "_pat_axes"):
            return
        for art in self._cursor_art:
            try:
                art.remove()
            except (ValueError, NotImplementedError):
                pass
        self._cursor_art = []
        for ax, ang_deg, pat, kind, gmax, rng in self._pat_axes:
            if ax is not event.inaxes:
                continue
            if event.xdata is None:
                return
            a = math.degrees(event.xdata) % 360.0
            g = pat.component(self._active_component())
            i = int(np.argmin(np.abs(((ang_deg - a + 180) % 360) - 180)))
            gv = float(g[i])
            r = float(np.clip(gv - gmax + rng, 0, rng))
            ln, = ax.plot([math.radians(ang_deg[i])] * 2, [0, r],
                          color="#00897b", lw=1.2, ls="-", zorder=8)
            pt, = ax.plot([math.radians(ang_deg[i])], [r], marker="o", ms=5,
                          color="#00897b", zorder=9)
            lab = "elevace" if kind == "vert" else "azimut"
            shown = ang_deg[i] if kind == "azimut" else (
                ang_deg[i] if ang_deg[i] <= 90 else 180 - ang_deg[i])
            tx = ax.annotate(f"{lab} {shown:.0f}°\n{gv:.2f} dBi\n({gv - gmax:+.1f} dB)",
                             xy=(math.radians(ang_deg[i]), r),
                             xytext=(6, 6), textcoords="offset points",
                             fontsize=8, color="#00695c",
                             bbox=dict(boxstyle="round,pad=0.25", fc="#e0f2f1",
                                       ec="#00897b", lw=0.6), zorder=10)
            self._cursor_art = [ln, pt, tx]
            self.cv_pat.draw_idle()
            return

    def _active_component(self) -> str:
        p = self.var_pol.get()
        if p.startswith("Svislá"):
            return "v"
        if p.startswith("Vodorovná"):
            return "h"
        return "total"

    def save_pattern(self):
        p = filedialog.asksaveasfilename(defaultextension=".png",
                                         filetypes=[("PNG", "*.png"), ("PDF", "*.pdf")])
        if p:
            self.fig_pat.savefig(p, dpi=160, bbox_inches="tight")
            self.set_status(f"Uloženo: {p}")

    # ==================================================================
    #  optimalizace
    # ==================================================================
    def fill_params(self):
        self.opt_params = suggest_parameters(self.model)
        self.refresh_params()

    def clear_params(self):
        self.opt_params = []
        self.refresh_params()

    def del_param(self):
        i = self.tbl_par.selected_index()
        if i is not None:
            del self.opt_params[i]
            self.refresh_params()

    def add_param(self):
        dlg = tk.Toplevel(self.master)
        dlg.title("Nový parametr")
        dlg.transient(self.master)
        dlg.grab_set()
        ttk.Label(dlg, text="Veličina:").grid(row=0, column=0, sticky="e", padx=6, pady=4)
        kind = tk.StringVar(value="delka")
        ttk.Combobox(dlg, textvariable=kind, values=list(PARAM_KINDS),
                     state="readonly", width=16).grid(row=0, column=1, padx=6)
        ttk.Label(dlg, text="Dráty (např. 1 nebo 1+3):").grid(row=1, column=0, sticky="e", padx=6)
        wires = tk.StringVar(value="1")
        ttk.Entry(dlg, textvariable=wires, width=16).grid(row=1, column=1, padx=6)
        ttk.Label(dlg, text="Rozsah ± [%]:").grid(row=2, column=0, sticky="e", padx=6)
        span = tk.StringVar(value="10")
        ttk.Entry(dlg, textvariable=span, width=16).grid(row=2, column=1, padx=6, pady=4)

        def ok():
            try:
                idxs = [int(x) - 1 for x in wires.get().replace(" ", "").split("+")]
                sp = float(span.get().replace(",", ".")) / 100.0
                p = Parameter(kind.get(), idxs, 0.0, 1.0)
                cur = read_param(self.model, p)
                if cur == 0:
                    p.lo, p.hi = -0.1 * self.model.wavelength, 0.1 * self.model.wavelength
                else:
                    p.lo, p.hi = cur * (1 - sp), cur * (1 + sp)
                    if p.lo > p.hi:
                        p.lo, p.hi = p.hi, p.lo
                self.opt_params.append(p)
                self.refresh_params()
                dlg.destroy()
            except (ValueError, IndexError) as e:
                messagebox.showerror("Chyba", str(e), parent=dlg)
        ttk.Button(dlg, text="Přidat", command=ok).grid(row=3, column=0, columnspan=2, pady=8)

    def _objective(self) -> Objective:
        def num(key, default):
            try:
                return float(self.opt_vars[key].get().replace(",", "."))
            except (ValueError, KeyError):
                return default
        freqs = []
        for tok in self.var_ofreqs.get().replace(";", ",").split(","):
            tok = tok.strip().replace(",", ".")
            if tok:
                try:
                    freqs.append(float(tok))
                except ValueError:
                    pass
        try:
            sec = float(self.var_fbsec.get())
        except (ValueError, AttributeError):
            sec = 0.0
        return Objective(freqs_mhz=freqs, fb_sector_deg=sec,
                         engine=self.engine_name(),
                         w_gain=num("w_gain", 1.0),
                         w_fb=num("w_fb", 0.5), w_swr=num("w_swr", 2.0),
                         target_swr=num("target_swr", 1.5),
                         w_r=num("w_r", 0.0), target_r=num("target_r", 50.0),
                         w_x=num("w_x", 0.0), target_x=num("target_x", 0.0))

    def do_optimize(self):
        if not self.opt_params:
            messagebox.showinfo("Chybí parametry",
                                "Nejdřív vyber, co se smí měnit (tlačítko „Doplnit obvyklé“).")
            return
        obj = self._objective()
        base = self.model.copy()
        params = [Parameter(p.kind, list(p.wires), p.lo, p.hi, p.endpoint, p.axis,
                            p.label, p.step, p.link, p.link_factor, p.link_offset)
                  for p in self.opt_params]
        try:
            pop = int(self.var_pop.get())
            gen = int(self.var_gen.get())
        except ValueError:
            pop, gen = 20, 20
        polish = self.var_polish.get()

        start = evaluate(base, obj)
        self.txt_opt.delete("1.0", "end")
        self.txt_opt.insert("end", "VÝCHOZÍ STAV\n" + start.summary() + "\n\n")
        self.pb_opt.configure(maximum=gen, value=0)

        def job(cancel, progress):
            def cb(g, total, best, txt):
                progress(g, total, best, txt)
                return not cancel.is_set()
            return optimize(base, params, obj, pop_size=pop, generations=gen,
                            polish=polish, progress=cb)

        def prog(g, total, best, txt):
            self.pb_opt.configure(value=min(g, total))
            self.set_status("Optimalizace: " + txt)

        def done(res):
            self._opt_candidate = res.model
            txt = ["VÝSLEDEK"]
            for p, v in zip(params, res.values):
                txt.append(f"  {p.describe():34s} {v:9.4f} m")
            txt.append("")
            txt.append(res.evaluation.summary())
            txt.append("")
            txt.append(f"cena {start.cost:.3f} → {res.cost:.3f}   ({res.iterations} vyhodnocení)")
            self.txt_opt.insert("end", "\n".join(txt) + "\n")
            self.txt_opt.see("end")
            self.ax_opt.clear()
            self.ax_opt.plot(res.history, color="#1565c0")
            self.ax_opt.set_xlabel("generace")
            self.ax_opt.set_ylabel("cena")
            self.ax_opt.grid(alpha=0.3)
            self.fig_opt.tight_layout()
            self.cv_opt.draw_idle()
            self.pb_opt.configure(value=self.pb_opt["maximum"])
            self.set_status("Optimalizace hotova. Můžeš výsledek přijmout do modelu.")

        self.worker.run(job, done, on_error=self._err, on_progress=prog)

    def accept_opt(self):
        if self._opt_candidate is None:
            messagebox.showinfo("Nic k přijetí", "Nejdřív spusť optimalizaci.")
            return
        self.model = self._opt_candidate
        self._opt_candidate = None
        self.result = None
        self.refresh_all()
        self.nb.select(0)
        self.set_status("Optimalizovaná geometrie převzata do modelu.")


    # ==================================================================
    #  výpočetní jádro
    # ==================================================================
    def engine_name(self) -> str:
        return getattr(self, "var_engine", None).get() if hasattr(self, "var_engine") \
            else engines.default_name()

    def _engine_changed(self):
        name = self.engine_name()
        engines.set_default(name)
        self.result = None
        self.sweep_results = []
        self.draw_geometry()
        eng = engines.get(name)
        self.set_status(f"Jádro: {name} — {eng.description}  "
                        f"(přepočítej klávesou F5)")

    # ==================================================================
    #  průvodci a úpravy geometrie
    # ==================================================================
    def _zin(self) -> complex:
        """Poslední spočtená impedance, jinak spočítá rychle teď."""
        if self.result is not None:
            return self.result.zin
        try:
            return solve(self.model).zin
        except Exception:
            return complex(50.0, 0.0)

    def run_wizard(self):
        m = dlg.wizard_dialog(self.master)
        if m is None:
            return
        self.model = m
        self.path = None
        self.result = None
        self.sweep_results = []
        self.opt_params = []
        self.refresh_all()
        f = self.model.freq_mhz
        self.var_f1.set(f"{f * 0.97:.4f}")
        self.var_f2.set(f"{f * 1.03:.4f}")
        self.set_status(f"Vytvořeno průvodcem: {m.name}")

    def _selected_wires(self):
        i = self.tbl_wires.selected_index()
        if i is None:
            return None
        if messagebox.askyesno("Rozsah úpravy",
                               f"Použít jen na vybraný drát č. {i + 1}?\n"
                               f"„Ne“ použije úpravu na celou anténu.",
                               default="no"):
            return [i]
        return None

    def _edit_op(self, fn):
        sel = self._selected_wires()
        if fn(self.master, self.model, sel):
            self._after_edit()

    def _after_edit(self):
        self.result = None
        self.sweep_results = []
        self.refresh_all()
        self.set_status("Geometrie upravena — přepočítej (F5).")

    def op_rescale(self):
        if dlg.rescale_freq_dialog(self.master, self.model):
            f = self.model.freq_mhz
            self.var_f1.set(f"{f * 0.97:.4f}")
            self.var_f2.set(f"{f * 1.03:.4f}")
            self._after_edit()

    def op_stack(self):
        if dlg.stack_dialog(self.master, self.model):
            self._after_edit()

    def op_polar(self):
        if dlg.polar_wire_dialog(self.master, self.model):
            self._after_edit()

    def op_taper(self):
        i = self.tbl_wires.selected_index()
        if i is None:
            messagebox.showinfo("Vyber drát",
                                "Nejdřív v tabulce vyber drát, ze kterého má "
                                "vzniknout zúžený prvek.")
            return
        if dlg.taper_dialog(self.master, self.model, i):
            self._after_edit()

    def op_elements(self):
        d = dlg.ElementDialog(self.master, self.model, self._after_edit)
        self.master.wait_window(d)

    # ==================================================================
    #  rezonance a historie
    # ==================================================================
    def do_resonance(self):
        model = self.model.copy()
        self.set_status("Hledám rezonanci…")

        eng = self.engine_name()

        def job(cancel, progress):
            return find_resonance(model, engine=eng)

        def done(res):
            if res is None:
                self.set_status("V rozsahu ±20 % rezonance není.")
                messagebox.showinfo("Rezonance",
                                    "V rozsahu ±20 % od aktuálního kmitočtu "
                                    "anténa nerezonuje.")
                return
            f, z = res
            self.set_status(f"Rezonance na {f:.4f} MHz, Z = {z.real:.1f} Ω")
            if messagebox.askyesno("Rezonance",
                                   f"Anténa rezonuje na {f:.4f} MHz "
                                   f"(Z = {z.real:.1f} {z.imag:+.1f} j Ω).\n\n"
                                   f"Nastavit tenhle kmitočet do modelu?"):
                self.model.freq_mhz = f
                self.var_freq.set(f"{f:.4f}")
                self._set_freq()
                self.do_calc()

        self.worker.run(job, done, on_error=self._err)

    def add_history(self, r: Result):
        self.history.append({
            "f": r.freq_mhz, "r": r.zin.real, "x": r.zin.imag, "swr": r.swr,
            "g": r.gain_dbi, "fb": r.fb_db, "note": self.model.name})
        self.tree_hist.insert("", 0, values=(
            f"{r.freq_mhz:.4f}", f"{r.zin.real:.1f}{r.zin.imag:+.1f}j",
            f"{r.swr:.2f}", f"{r.gain_dbi:.2f}", f"{r.fb_db:.1f}",
            self.model.name[:24]))

    def clear_history(self):
        self.history = []
        for i in self.tree_hist.get_children():
            self.tree_hist.delete(i)

    def save_history(self):
        if not self.history:
            messagebox.showinfo("Prázdné", "Historie je prázdná.")
            return
        p = filedialog.asksaveasfilename(defaultextension=".csv",
                                         filetypes=[("CSV", "*.csv")])
        if not p:
            return
        import csv
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["f_MHz", "R_ohm", "X_ohm", "PSV", "zisk_dBi", "FB_dB", "model"])
            for h in self.history:
                w.writerow([f"{h['f']:.4f}", f"{h['r']:.2f}", f"{h['x']:.2f}",
                            f"{h['swr']:.3f}", f"{h['g']:.2f}", f"{h['fb']:.2f}",
                            h["note"]])
        self.set_status(f"Historie uložena: {p}")

    # ==================================================================
    #  3D diagram
    # ==================================================================
    def show_3d(self):
        if self.result is None:
            messagebox.showinfo("Nejdřív spočítej", "Spusť výpočet (F5).")
            return
        eng3 = engines.get(self.result.engine)
        grounded = self.model.ground.kind != "free"
        win = tk.Toplevel(self.master)
        win.title("3D vyzařovací diagram")
        top = ttk.Frame(win, padding=PAD)
        top.pack(fill="x")
        v_rng = tk.StringVar(value="30")
        ttk.Label(top, text="Rozsah [dB]:").pack(side="left")
        ttk.Combobox(top, textvariable=v_rng, width=5, state="readonly",
                     values=["20", "30", "40", "50"]).pack(side="left", padx=4)
        v_ant = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="zobrazit anténu", variable=v_ant).pack(side="left", padx=8)
        fig = Figure(figsize=(7.2, 6.0), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        cv = FigureCanvasTkAgg(fig, master=win)
        cv.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(cv, win).update()

        def draw():
            ax.clear()
            try:
                rng = float(v_rng.get())
            except ValueError:
                rng = 30.0
            n_th = 46 if grounded else 73
            th_deg = np.linspace(0.5, 89.5 if grounded else 179.5, n_th)
            g = np.empty((n_th, 97))
            for k, td in enumerate(th_deg):
                _, row = eng3.cut_azimuth(self.result, 90.0 - td, 97)
                g[k] = row.gain_dbi
            TH = np.radians(th_deg)[:, None] * np.ones((1, 97))
            PH = np.radians(np.linspace(0, 360, 97))[None, :] * np.ones((n_th, 1))
            gmax = float(np.max(g))
            r = np.clip(g - gmax + rng, 0, None) / rng
            X = r * np.sin(TH) * np.cos(PH)
            Y = r * np.sin(TH) * np.sin(PH)
            Z = r * np.cos(TH)
            norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
            ax.plot_surface(X, Y, Z, facecolors=matplotlib.colormaps["turbo"](norm(r)),
                            rstride=1, cstride=1, linewidth=0, antialiased=True,
                            shade=False, alpha=0.95)
            if v_ant.get():
                lo, hi = self.model.bounds()
                span = max(np.max(hi - lo), 1e-6)
                c = 0.5 * (lo + hi)
                for w in self.model.wires:
                    a = (w.a - c) / span * 0.9
                    b = (w.b - c) / span * 0.9
                    ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                            color="#111", lw=1.6, zorder=10)
            if grounded:
                gx = np.linspace(-1, 1, 2)
                GX, GY = np.meshgrid(gx, gx)
                ax.plot_surface(GX, GY, np.zeros_like(GX), alpha=0.10, color="#795548")
            ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
            ax.set_box_aspect((1, 1, 0.55 if grounded else 1))
            lim = 1.05
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            ax.set_zlim(0 if grounded else -lim, lim)
            ax.set_title(f"max {gmax:.2f} dBi   (povrch = {rng:.0f} dB rozsah)",
                         fontsize=10)
            fig.tight_layout()
            cv.draw_idle()

        ttk.Button(top, text="Překreslit", command=draw).pack(side="left", padx=6)
        ttk.Button(top, text="Uložit…",
                   command=lambda: self._save_fig(fig)).pack(side="left")
        draw()

    def _save_fig(self, fig):
        p = filedialog.asksaveasfilename(defaultextension=".png",
                                         filetypes=[("PNG", "*.png"), ("PDF", "*.pdf")])
        if p:
            fig.savefig(p, dpi=160, bbox_inches="tight")
            self.set_status(f"Uloženo: {p}")

    # ==================================================================
    def _set_link(self, idx: int, text: str):
        """Vazba parametru: prázdné/„—“ = volný, jinak např. #1, 0.95*#1, #1-0.2"""
        p = self.opt_params[idx]
        t = str(text).strip().replace(",", ".").replace("×", "*").replace(" ", "")
        if t in ("", "-", "—", "0"):
            p.link, p.link_factor, p.link_offset = None, 1.0, 0.0
            self.refresh_params()
            return
        import re
        m = re.match(r"^(?:([-+]?\d*\.?\d+)\*)?#(\d+)([-+]\d*\.?\d+)?$", t)
        if not m:
            messagebox.showerror(
                "Nesrozumitelná vazba",
                "Zapiš vazbu jako #2 (stejné jako parametr 2), 0.95*#2 "
                "(poměrem) nebo #2-0.15 (s posunem). Prázdné pole = volný "
                "parametr.")
            return
        fac = float(m.group(1)) if m.group(1) else 1.0
        ref = int(m.group(2)) - 1
        off = float(m.group(3)) if m.group(3) else 0.0
        if ref == idx or not (0 <= ref < len(self.opt_params)):
            messagebox.showerror("Neplatná vazba", "Parametr nemůže odkazovat sám na sebe.")
            return
        if self.opt_params[ref].link is not None:
            messagebox.showerror("Neplatná vazba",
                                 "Odkazovat lze jen na volný parametr, ne na "
                                 "další svázaný.")
            return
        p.link, p.link_factor, p.link_offset = ref, fac, off
        self.refresh_params()

    # ==================================================================
    #  soubory
    # ==================================================================
    def new_model(self):
        self.model = Model(name="Nová anténa", freq_mhz=14.1)
        lam = self.model.wavelength
        self.model.wires = [Wire(-lam / 4, 0, lam / 2, lam / 4, 0, lam / 2, 0.001, 21)]
        self.model.sources = [Source(0, 0.5, 1.0)]
        self.path = None
        self.result = None
        self.sweep_results = []
        self.opt_params = []
        self.refresh_all()

    def load_example(self, name):
        self.model = EXAMPLES[name]()
        self.path = None
        self.result = None
        self.sweep_results = []
        self.opt_params = []
        self.refresh_all()
        f = self.model.freq_mhz
        self.var_f1.set(f"{f * 0.97:.4f}")
        self.var_f2.set(f"{f * 1.03:.4f}")

    def open_file(self):
        p = filedialog.askopenfilename(
            filetypes=[("Projekt AntOpt", "*.json"), ("Vše", "*.*")])
        if not p:
            return
        try:
            self.model = Model.load(p)
        except Exception as e:
            messagebox.showerror("Nelze otevřít", str(e))
            return
        self.path = p
        self.result = None
        self.refresh_all()

    def save_file(self, as_new=False):
        p = self.path
        if as_new or not p:
            p = filedialog.asksaveasfilename(defaultextension=".json",
                                             filetypes=[("Projekt AntOpt", "*.json")])
        if not p:
            return
        self.model.save(p)
        self.path = p
        self.set_status(f"Uloženo: {p}")
        self.master.title(f"AntOpt – {self.model.name}  [{os.path.basename(p)}]")

    def import_file(self):
        p = filedialog.askopenfilename(filetypes=[
            ("Anténní modely", "*.nec *.ez *.maa *.mma *.txt"),
            ("NEC", "*.nec"), ("MMANA", "*.maa *.mma"), ("Vše", "*.*")])
        if not p:
            return
        try:
            text = open(p, "r", encoding="utf-8", errors="replace").read()
        except OSError as e:
            messagebox.showerror("Nelze číst", str(e))
            return
        ext = os.path.splitext(p)[1].lower()
        try:
            if ext in (".maa", ".mma"):
                m, warn = from_maa(text)
            elif ext in (".nec", ".ez"):
                m, warn = from_nec(text)
            else:
                try:
                    m, warn = from_nec(text)
                except Exception:
                    m, warn = from_maa(text)
        except Exception as e:
            messagebox.showerror("Import selhal", f"{e}")
            return
        self.model = m
        self.path = None
        self.result = None
        self.opt_params = []
        self.refresh_all()
        if warn:
            messagebox.showwarning("Import s výhradami", "\n".join(warn[:12]))
        self.set_status(f"Importováno: {os.path.basename(p)}")

    def export_as(self, fmt):
        ext = ".nec" if fmt == "nec" else ".maa"
        p = filedialog.asksaveasfilename(defaultextension=ext,
                                         filetypes=[(fmt.upper(), "*" + ext)])
        if not p:
            return
        text = to_nec(self.model) if fmt == "nec" else to_maa(self.model)
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)
        self.set_status(f"Exportováno: {p}")

    def show_about(self):
        messagebox.showinfo(
            "AntOpt",
            "AntOpt – modelování a optimalizace antén\n"
            "Vlastní solver metodou momentů pro tenké dráty.\n\n"
            "Ověřeno proti NEC-2: odpor do 1 %, zisk do 0,1 dB.\n\n"
            "Co model UMÍ:\n"
            "• tenké dráty libovolně v prostoru, spoje více drátů\n"
            "• volný prostor, dokonalá zem, reálná zem\n"
            "• ztráty vodiče, soustředné zátěže R+jX / RLC\n"
            "• víc zdrojů s fází (fázovaná pole)\n\n"
            "Co model NEUMÍ (a kde být opatrný):\n"
            "• reálná zem v IMPEDANCI je jen přibližná (obraz nad dokonalou "
            "zemí). U vertikálů napájených proti zemi zadej ztráty zemního "
            "systému ručně.\n"
            "• zakopané radiály, plochy a válce, dielektrika\n"
            "• velmi tlusté dráty (poloměr > λ/200) – reaktance se rozchází\n"
            "• segment musí být delší než 4× poloměr drátu\n")


def main():
    root = tk.Tk()
    root.title("AntOpt")
    try:
        root.tk.call("tk", "scaling", 1.2)
    except tk.TclError:
        pass
    style = ttk.Style()
    if "clam" in style.theme_names():
        style.theme_use("clam")
    root.geometry("1380x880")
    root.minsize(1050, 700)
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
