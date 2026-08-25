#!/usr/bin/env python3
"""Vygeneruje ikonu AntOpt (Yagi na ráhnu) do build/icon.png a .ico/.icns.

Spouští se jen při přípravě balíčku; k běhu programu není potřeba.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def draw(size: int = 1024):
    from PIL import Image, ImageDraw
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    s = size / 1024.0
    # zaoblený podklad
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=int(220 * s),
                        fill=(21, 101, 192, 255))
    # ráhno uprostřed
    bx = size // 2
    d.rounded_rectangle([bx - int(24 * s), int(160 * s), bx + int(24 * s), int(880 * s)],
                        radius=int(24 * s), fill=(255, 255, 255, 255))
    # prvky: reflektor, zářič, direktory (zkracují se, rozteče rostou)
    els = [(250, 372), (390, 340), (540, 322), (700, 306), (850, 292)]
    for y, half in els:
        yy = int(y * s)
        hw = int(half * s)
        th = int(30 * s)
        d.rounded_rectangle([bx - hw, yy - th // 2, bx + hw, yy + th // 2],
                            radius=th // 2, fill=(255, 255, 255, 255))
    # napájecí bod na zářiči
    d.ellipse([bx - int(58 * s), int(390 * s) - int(58 * s),
               bx + int(58 * s), int(390 * s) + int(58 * s)],
              fill=(255, 193, 7, 255))
    return img


def main() -> int:
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        print("Ikona se nevygeneruje — chybí Pillow (pip install pillow).")
        print("Nevadí, aplikace se sestaví bez vlastní ikony.")
        return 0
    img = draw()
    png = os.path.join(HERE, "icon.png")
    img.save(png)
    print("uloženo:", png)

    ico = os.path.join(HERE, "icon.ico")
    img.save(ico, sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print("uloženo:", ico)

    icns = os.path.join(HERE, "icon.icns")
    if sys.platform == "darwin":
        # na Macu má přednost systémový iconutil — jeho výstup Finder
        # spolkne vždycky, u .icns z Pillow to jisté není
        iset = os.path.join(HERE, "icon.iconset")
        os.makedirs(iset, exist_ok=True)
        for n in (16, 32, 64, 128, 256, 512):
            img.resize((n, n)).save(os.path.join(iset, f"icon_{n}x{n}.png"))
            img.resize((n * 2, n * 2)).save(
                os.path.join(iset, f"icon_{n}x{n}@2x.png"))
        if os.system(f'iconutil -c icns "{iset}" -o "{icns}"') == 0:
            print("uloženo:", icns)
            return 0
        print("iconutil selhal, zkusím Pillow.")
    try:
        img.save(icns)
        print("uloženo:", icns)
    except Exception as e:
        print(f"icns se nepovedlo ({e}) — aplikace bude bez vlastní ikony.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
