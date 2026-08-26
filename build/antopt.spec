# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec pro AntOpt.

Sestavuje se na tom systému, pro který má výsledek být — PyInstaller
neumí křížový překlad. Na macOS vznikne AntOpt.app, na Windows
AntOpt.exe, na Linuxu spustitelný soubor AntOpt.

    pyinstaller build/antopt.spec --noconfirm
"""
import os
import sys

from PyInstaller.utils.hooks import collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, os.pardir))
# Verzi nastavuje sestavovací skript (u vydání se bere ze značky v*).
# Bez ní se sestaví jako 1.0 — hodí se při ladění na vlastním počítači.
VERZE = os.environ.get("ANTOPT_VERSION", "1.0")
IS_MAC = sys.platform == "darwin"
IS_WIN = os.name == "nt"

icon = None
for cand in (("icon.icns" if IS_MAC else None), ("icon.ico" if IS_WIN else None)):
    if cand and os.path.exists(os.path.join(SPECPATH, cand)):
        icon = os.path.join(SPECPATH, cand)

# scipy.optimize se importuje až za běhu (doladění Nelder-Meadem), takže
# by ho analýza sama nenašla
hidden = ["scipy.optimize", "scipy.special", "scipy._lib.messagestream"]
hidden += collect_submodules("matplotlib.backends")
# Panel nástrojů matplotlibu kreslí ikony přes PIL.ImageTk a ten si natahuje
# _tkinter_finder až za běhu. Bez tohohle řádku se sbalená aplikace složí
# hned při startu na „No module named 'PIL._tkinter_finder'“.
hidden += ["PIL.ImageTk", "PIL._tkinter_finder"]

a = Analysis(
    [os.path.join(ROOT, "run_antopt.py")],
    pathex=[ROOT],
    binaries=[],
    datas=[(os.path.join(ROOT, "README.md"), ".")],
    hiddenimports=hidden,
    hookspath=[],
    runtime_hooks=[],
    # bez těchhle je balík menší o stovky MB a nic z nich se nepoužívá
    excludes=["PyQt5", "PyQt6", "PySide2", "PySide6", "wx", "IPython",
              "jupyter", "notebook", "pytest", "pandas", "sphinx",
              "matplotlib.tests", "numpy.tests", "scipy.tests"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AntOpt",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,          # GUI aplikace, žádné okno terminálu
    icon=icon,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="AntOpt",
)

if IS_MAC:
    app = BUNDLE(
        coll,
        name="AntOpt.app",
        icon=icon,
        bundle_identifier="cz.ok1m.antopt",
        info_plist={
            "CFBundleName": "AntOpt",
            "CFBundleDisplayName": "AntOpt",
            "CFBundleShortVersionString": VERZE,
            "CFBundleVersion": VERZE,
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "11.0",
            "NSHumanReadableCopyright": "AntOpt — modelování a optimalizace antén",
            "CFBundleDocumentTypes": [
                {
                    "CFBundleTypeName": "Model MMANA",
                    "CFBundleTypeExtensions": ["maa", "mma"],
                    "CFBundleTypeRole": "Editor",
                },
                {
                    "CFBundleTypeName": "Model NEC",
                    "CFBundleTypeExtensions": ["nec", "ez"],
                    "CFBundleTypeRole": "Editor",
                },
            ],
        },
    )
