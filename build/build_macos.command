#!/bin/bash
# Sestaví AntOpt.app pro macOS. Dá se na něj rovnou poklepat ve Finderu.
#
# PyInstaller neumí křížový překlad — aplikace pro Mac musí vzniknout
# na Macu, a jen pro ten procesor, na kterém se sestavila (Apple Silicon
# nebo Intel).
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"

echo
echo "======================================================================"
echo "  Sestavení AntOpt.app"
echo "  $ROOT"
echo "======================================================================"
echo

# ---------------------------------------------------------------- Python
# Potřebujeme Python, který má Tkinter. Homebrew ho standardně nemá,
# proto se zkouší víc kandidátů a vybere se první funkční.
PYBIN=""
for cand in \
    /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 \
    /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
    /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 \
    python3.13 python3.12 python3.11 python3
do
    if command -v "$cand" >/dev/null 2>&1 || [ -x "$cand" ]; then
        if "$cand" -c "import tkinter" >/dev/null 2>&1; then
            PYBIN="$cand"
            break
        fi
    fi
done

if [ -z "$PYBIN" ]; then
    V="$(python3 -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo 3.12)"
    cat <<EOF

  NENAŠEL JSEM PYTHON S TKINTEREM

  Žádný Python na tomhle Macu neumí "import tkinter". Bez něj se GUI
  nesestaví. Tkinter není balíček z PyPI — "pip install tkinter" nikdy
  fungovat nebude.

  Dvě cesty, obě fungují:

    1) Python z python.org — Tcl/Tk má rovnou v sobě, nic dalšího netřeba:
         https://www.python.org/downloads/macos/

    2) Homebrew — k Pythonu se Tk doinstaluje zvlášť:
         brew install python-tk@$V

  Pak spusť tenhle skript znovu.

EOF
    read -r -p "  Zavři stiskem Enter. " _
    exit 1
fi

echo "  Python:  $("$PYBIN" -c 'import sys;print(sys.executable)')"
echo "  Verze:   $("$PYBIN" -c 'import sys;print(sys.version.split()[0])')"
echo "  Tk:      $("$PYBIN" -c 'import tkinter;print(tkinter.TkVersion)')"
echo "  Procesor: $(uname -m)"
echo

# ---------------------------------------------------------- prostředí
VENV="$ROOT/.build-venv"
if [ ! -d "$VENV" ]; then
    echo "  Vytvářím prostředí pro sestavení…"
    "$PYBIN" -m venv "$VENV" || exit 1
fi
PY="$VENV/bin/python"

echo "  Instaluji knihovny (poprvé to chvíli trvá)…"
"$PY" -m pip install --upgrade pip >/dev/null 2>&1
"$PY" -m pip install -r "$ROOT/requirements.txt" pyinstaller pillow || {
    echo "  Instalace knihoven selhala."; read -r -p "  Enter. " _; exit 1; }

echo "  Kreslím ikonu…"
"$PY" "$ROOT/build/make_icon.py"

# ------------------------------------------------------------ sestavení
echo
echo "  Sestavuji aplikaci — tohle trvá tak minutu až dvě…"
echo
rm -rf "$ROOT/dist/AntOpt.app" "$ROOT/dist/AntOpt"
"$PY" -m PyInstaller "$ROOT/build/antopt.spec" --noconfirm \
    --distpath "$ROOT/dist" --workpath "$ROOT/build/work" || {
    echo; echo "  Sestavení selhalo — výpis je výš."
    read -r -p "  Enter. " _; exit 1; }

APP="$ROOT/dist/AntOpt.app"
[ -d "$APP" ] || APP="$ROOT/dist/AntOpt"

# ------------------------------------------------------------- kontrola
echo
echo "  Kontroluji sestavenou aplikaci…"
BIN="$APP/Contents/MacOS/AntOpt"
[ -x "$BIN" ] || BIN="$ROOT/dist/AntOpt/AntOpt"
if [ -x "$BIN" ]; then
    "$BIN" --selftest || {
        echo; echo "  Aplikace se sestavila, ale kontrola našla chybu (viz výš)."
        read -r -p "  Enter. " _; exit 1; }
fi

# quarantine se nastavuje u stažených souborů; u vlastního sestavení
# obvykle chybí, ale po přenosu na jiný Mac by aplikaci blokovala
xattr -cr "$APP" 2>/dev/null

echo
echo "======================================================================"
echo "  HOTOVO"
echo "  $APP"
echo "  velikost: $(du -sh "$APP" | cut -f1)"
echo "======================================================================"
echo
echo "  Aplikaci můžeš přetáhnout do složky Aplikace."
echo
echo "  Když ji pošleš někomu dalšímu, macOS ji jako nepodepsanou napoprvé"
echo "  nepustí. Řešení: klepnout pravým tlačítkem a dát Otevřít, nebo"
echo "  v Terminálu spustit   xattr -dr com.apple.quarantine /cesta/AntOpt.app"
echo
open "$ROOT/dist" 2>/dev/null
read -r -p "  Zavři stiskem Enter. " _
