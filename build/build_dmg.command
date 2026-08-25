#!/bin/bash
# Zabalí hotovou AntOpt.app do instalačního obrazu AntOpt-1.0.dmg.
# Poklepat ve Finderu. Nejdřív musí proběhnout build_macos.command.
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"
APP="$ROOT/dist/AntOpt.app"
DMG="$ROOT/dist/AntOpt-1.0.dmg"

echo
echo "======================================================================"
echo "  Instalační obraz AntOpt.dmg"
echo "======================================================================"
echo

if [ ! -d "$APP" ]; then
    echo "  Nenašel jsem $APP"
    echo "  Nejdřív poklepej na build/build_macos.command."
    read -r -p "  Enter. " _
    exit 1
fi

STAGE="$(mktemp -d)/AntOpt"
mkdir -p "$STAGE"
echo "  Připravuji obsah…"
cp -R "$APP" "$STAGE/" || exit 1
# odkaz na Aplikace, aby stačilo aplikaci přetáhnout přes něj
ln -s /Applications "$STAGE/Aplikace"
cat > "$STAGE/Čti mě.txt" <<'TXT'
AntOpt — modelování a optimalizace antén

Instalace: přetáhni AntOpt.app na odkaz Aplikace vedle.

Aplikace není podepsaná vývojářským certifikátem, takže ji macOS
napoprvé nepustí. Klepni na ni pravým tlačítkem a dej Otevřít —
v tom dialogu už tlačítko Otevřít je. Podruhé se spustí normálně.
TXT

rm -f "$DMG"
echo "  Vytvářím obraz…"
hdiutil create -volname "AntOpt" -srcfolder "$STAGE" \
    -ov -format UDZO -quiet "$DMG" || {
    echo "  hdiutil selhal."; read -r -p "  Enter. " _; exit 1; }
rm -rf "$(dirname "$STAGE")"

echo
echo "======================================================================"
echo "  HOTOVO"
echo "  $DMG"
echo "  velikost: $(du -sh "$DMG" | cut -f1)"
echo "======================================================================"
echo
open "$ROOT/dist" 2>/dev/null
read -r -p "  Zavři stiskem Enter. " _
