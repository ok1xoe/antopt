#!/usr/bin/env python3
"""Spouštěč AntOpt.

Před startem zkontroluje, že prostředí má všechno potřebné, a když ne,
poradí přesný příkaz místo strohého ImportError.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PY = f"{sys.version_info.major}.{sys.version_info.minor}"


def _in_venv() -> bool:
    return sys.prefix != sys.base_prefix


def _is_homebrew() -> bool:
    p = sys.base_prefix
    return "/Cellar/" in p or p.startswith("/opt/homebrew") or p.startswith("/usr/local/opt")


def _tk_help() -> str:
    """Návod, jak doplnit Tkinter k tomuhle konkrétnímu Pythonu."""
    lines = [
        "",
        "=" * 72,
        "  CHYBÍ TKINTER",
        "=" * 72,
        "",
        f"  Python {PY}: {sys.executable}",
        "",
        "  Tkinter není balíček z PyPI — `pip install tkinter` nikdy nebude",
        "  fungovat. Je to součást standardní knihovny, ale potřebuje C modul",
        "  `_tkinter` přeložený proti Tcl/Tk, který se instaluje zvlášť.",
        "",
    ]
    if sys.platform == "darwin":
        if _is_homebrew():
            lines += [
                "  Máš Python z Homebrew. Doplň k němu Tk:",
                "",
                f"      brew install python-tk@{PY}",
                "",
                "  Pak stačí program spustit znovu — virtuální prostředí bere",
                "  standardní knihovnu ze základního Pythonu, takže venv",
                "  není potřeba vytvářet znovu.",
                "",
                "  Kdyby to i tak nešlo, použij Python z python.org, který má",
                "  Tcl/Tk rovnou v sobě:  https://www.python.org/downloads/macos/",
            ]
        else:
            lines += [
                "  Nejjistější je Python z python.org — Tcl/Tk má rovnou v sobě:",
                "      https://www.python.org/downloads/macos/",
                "",
                "  Přes Homebrew:",
                f"      brew install python@{PY} python-tk@{PY}",
            ]
    elif sys.platform.startswith("linux"):
        lines += [
            "  Debian / Ubuntu:      sudo apt install python3-tk",
            "  Fedora / RHEL:        sudo dnf install python3-tkinter",
            "  Arch:                 sudo pacman -S tk",
        ]
    else:
        lines += [
            "  Na Windows Tkinter obsahuje instalátor z python.org — při",
            "  instalaci musí být zaškrtnuté „tcl/tk and IDLE“.",
        ]
    lines += [
        "",
        "  Kontrola, že už je k dispozici:",
        f"      {os.path.basename(sys.executable)} -c \"import tkinter; print(tkinter.TkVersion)\"",
        "",
        "  Solver a optimalizátor Tkinter nepotřebují — bez GUI se dají",
        "  používat přímo, viz README (sekce Struktura).",
        "=" * 72,
        "",
    ]
    return "\n".join(lines)


def _deps_help(missing) -> str:
    req = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    pip = "pip" if _in_venv() else "pip3"
    return "\n".join([
        "",
        "=" * 72,
        "  CHYBÍ KNIHOVNY: " + ", ".join(missing),
        "=" * 72,
        "",
        f"      {pip} install -r {req}",
        "",
        "  (AntOpt potřebuje numpy, scipy a matplotlib.)",
        "=" * 72,
        "",
    ])


def main() -> int:
    missing = []
    for mod in ("numpy", "scipy", "matplotlib"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(_deps_help(missing), file=sys.stderr)
        return 1

    try:
        import tkinter  # noqa: F401
    except ImportError:
        print(_tk_help(), file=sys.stderr)
        return 1

    from antopt.gui import main as gui_main
    try:
        gui_main()
    except tkinter.TclError as e:
        if "display" in str(e).lower() or "DISPLAY" in str(e):
            print("\nAntOpt potřebuje grafickou plochu — přes SSH bez X11 "
                  "se nespustí.\nSolver se dá volat i bez GUI, viz README.\n",
                  file=sys.stderr)
            return 1
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
