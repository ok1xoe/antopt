"""Rozsekání drátového modelu na segmenty a sestavení bázových funkcí.

Bázové funkce jsou po částech lineární (trojúhelníkové) proudové módy vázané
na uzly sítě.  Každá báze má nejvýš dva příspěvky, každý na jednom segmentu:

    (segment, end, coef)

kde ``end`` říká, u kterého konce segmentu má tvarová funkce hodnotu 1
(end=1 -> tvar t, end=0 -> tvar 1-t, t je parametr 0..1 od začátku k konci)
a ``coef`` je znaménko/amplituda proudu ve směru segmentu.

Uzly:
  * volný konec (stupeň 1, nad zemí)  -> žádná báze, proud = 0
  * spojení dvou segmentů            -> 1 báze (spojitý proud)
  * uzel se stupněm M >= 3           -> M-1 bází (Kirchhoff)
  * uzel na zemi (z ~ 0, zem zapnuta) -> M bází, proud smí téct do země
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .model import Model

GROUND_TOL = 1e-9


@dataclass
class Mesh:
    points: np.ndarray          # (Np, 3)
    seg_nodes: np.ndarray       # (Ns, 2) indexy uzlů
    seg_wire: np.ndarray        # (Ns,) index drátu
    radius: np.ndarray          # (Ns,)
    a: np.ndarray               # (Ns, 3) počáteční body
    b: np.ndarray               # (Ns, 3) koncové body
    u: np.ndarray               # (Ns, 3) jednotkové směry
    length: np.ndarray          # (Ns,)
    mid: np.ndarray             # (Ns, 3)
    # báze: pole s paddingem, 2 příspěvky na bázi
    bs_seg: np.ndarray          # (Nb, 2) int
    bs_end: np.ndarray          # (Nb, 2) int (0/1)
    bs_coef: np.ndarray         # (Nb, 2) float (0 = nepoužito)
    bs_node: np.ndarray         # (Nb,) uzel, ke kterému báze patří
    node_of_wirepos: dict       # (wire, node_index_on_wire) -> global node id

    @property
    def nseg(self) -> int:
        return self.seg_nodes.shape[0]

    @property
    def nbasis(self) -> int:
        return self.bs_seg.shape[0]

    def basis_at_node(self, node: int) -> int:
        idx = np.where(self.bs_node == node)[0]
        return int(idx[0]) if len(idx) else -1

    def divergence_coef(self) -> np.ndarray:
        """(Nb, 2) koeficient konstantní hustoty náboje na segmentu."""
        sign = np.where(self.bs_end == 1, 1.0, -1.0)
        L = np.where(self.bs_coef != 0, self.length[self.bs_seg], 1.0)
        return self.bs_coef * sign / L


def _key(p: np.ndarray, tol: float) -> Tuple[int, int, int]:
    return tuple(np.round(p / tol).astype(np.int64))


def build_mesh(model: Model) -> Mesh:
    lam = model.wavelength
    tol = max(1e-9, lam * 1e-7)

    points: List[np.ndarray] = []
    lookup: dict = {}

    def add_point(p: np.ndarray) -> int:
        k = _key(p, tol)
        # zkus i sousední buňky kvůli zaokrouhlení
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    kk = (k[0] + dx, k[1] + dy, k[2] + dz)
                    if kk in lookup:
                        return lookup[kk]
        idx = len(points)
        points.append(p.copy())
        lookup[k] = idx
        return idx

    seg_nodes: List[Tuple[int, int]] = []
    seg_wire: List[int] = []
    radius: List[float] = []
    node_of_wirepos: dict = {}

    for wi, w in enumerate(model.wires):
        n = max(1, int(w.nseg))
        pa, pb = w.a, w.b
        ids = []
        for i in range(n + 1):
            t = i / n
            ids.append(add_point(pa + t * (pb - pa)))
            node_of_wirepos[(wi, i)] = ids[-1]
        for i in range(n):
            seg_nodes.append((ids[i], ids[i + 1]))
            seg_wire.append(wi)
            radius.append(w.radius)

    P = np.array(points, dtype=float)
    S = np.array(seg_nodes, dtype=int)
    A = P[S[:, 0]]
    B = P[S[:, 1]]
    d = B - A
    L = np.linalg.norm(d, axis=1)
    U = d / L[:, None]

    # --- uzel -> seznam připojení (segment, end)
    attach: dict = {}
    for si, (n0, n1) in enumerate(S):
        attach.setdefault(n0, []).append((si, 0))
        attach.setdefault(n1, []).append((si, 1))

    grounded = model.ground.kind != "free"

    bs_seg, bs_end, bs_coef, bs_node = [], [], [], []

    for node, att in sorted(attach.items()):
        on_ground = grounded and abs(P[node, 2]) < 1e-6
        M = len(att)
        if on_ground:
            # proud smí odtéct do země: každý připojený segment dá jednu bázi
            for (si, e) in att:
                s = 1.0 if e == 1 else -1.0
                bs_seg.append([si, 0])
                bs_end.append([e, 0])
                bs_coef.append([s, 0.0])
                bs_node.append(node)
        elif M >= 2:
            si0, e0 = att[0]
            s0 = 1.0 if e0 == 1 else -1.0
            for k in range(1, M):
                sik, ek = att[k]
                sk = 1.0 if ek == 1 else -1.0
                bs_seg.append([si0, sik])
                bs_end.append([e0, ek])
                bs_coef.append([s0, -sk])
                bs_node.append(node)
        # M == 1 a ne na zemi -> volný konec, proud = 0

    if not bs_seg:
        raise ValueError("Model nemá žádnou bázovou funkci (zkontroluj geometrii).")

    return Mesh(
        points=P,
        seg_nodes=S,
        seg_wire=np.array(seg_wire, dtype=int),
        radius=np.array(radius, dtype=float),
        a=A, b=B, u=U, length=L, mid=0.5 * (A + B),
        bs_seg=np.array(bs_seg, dtype=int),
        bs_end=np.array(bs_end, dtype=int),
        bs_coef=np.array(bs_coef, dtype=float),
        bs_node=np.array(bs_node, dtype=int),
        node_of_wirepos=node_of_wirepos,
    )


def node_for_position(mesh: Mesh, model: Model, wire: int, pos: float) -> int:
    """Najde uzel na drátu ``wire`` nejblíž relativní poloze ``pos`` (0..1).

    Preferuje uzly, ke kterým existuje bázová funkce (tj. ne volné konce).
    """
    w = model.wires[wire]
    n = max(1, int(w.nseg))
    ideal = pos * n
    order = sorted(range(n + 1), key=lambda i: abs(i - ideal))
    for i in order:
        node = mesh.node_of_wirepos[(wire, i)]
        if mesh.basis_at_node(node) >= 0:
            return node
    raise ValueError(f"Na drátu {wire + 1} není použitelný uzel pro zdroj/zátěž.")
