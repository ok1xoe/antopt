#!/bin/bash
# Sestaví spustitelný AntOpt pro Linux.
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"

python3 -c "import tkinter" 2>/dev/null || {
    echo "Chybí Tkinter:  sudo apt install python3-tk   (Fedora: python3-tkinter)"
    exit 1; }

VENV="$ROOT/.build-venv"
[ -d "$VENV" ] || python3 -m venv "$VENV"
PY="$VENV/bin/python"
"$PY" -m pip install -q --upgrade pip
"$PY" -m pip install -q -r "$ROOT/requirements.txt" pyinstaller pillow || exit 1
"$PY" "$ROOT/build/make_icon.py"
"$PY" -m PyInstaller "$ROOT/build/antopt.spec" --noconfirm \
    --distpath "$ROOT/dist" --workpath "$ROOT/build/work" || exit 1
"$ROOT/dist/AntOpt/AntOpt" --selftest
echo
echo "Hotovo: $ROOT/dist/AntOpt/AntOpt   ($(du -sh "$ROOT/dist/AntOpt" | cut -f1))"
