@echo off
rem Sestaví AntOpt.exe pro Windows. Poklepat.
setlocal
cd /d "%~dp0\.."
set ROOT=%CD%

echo.
echo ======================================================================
echo   Sestaveni AntOpt.exe
echo   %ROOT%
echo ======================================================================
echo.

python -c "import tkinter" >nul 2>&1
if errorlevel 1 (
    echo   Python nema Tkinter. Preinstaluj Python z python.org a zaskrtni
    echo   volbu "tcl/tk and IDLE".  https://www.python.org/downloads/windows/
    pause
    exit /b 1
)

if not exist "%ROOT%\.build-venv" (
    echo   Vytvarim prostredi pro sestaveni...
    python -m venv "%ROOT%\.build-venv" || (pause & exit /b 1)
)
set PY=%ROOT%\.build-venv\Scripts\python.exe

echo   Instaluji knihovny (poprve to chvili trva)...
"%PY%" -m pip install --upgrade pip >nul 2>&1
"%PY%" -m pip install -r "%ROOT%\requirements.txt" pyinstaller pillow || (pause & exit /b 1)

echo   Kreslim ikonu...
"%PY%" "%ROOT%\build\make_icon.py"

echo.
echo   Sestavuji aplikaci - tohle trva tak minutu az dve...
echo.
"%PY%" -m PyInstaller "%ROOT%\build\antopt.spec" --noconfirm ^
    --distpath "%ROOT%\dist" --workpath "%ROOT%\build\work" || (pause & exit /b 1)

echo.
echo   Kontroluji sestavenou aplikaci...
"%ROOT%\dist\AntOpt\AntOpt.exe" --selftest

echo.
echo ======================================================================
echo   HOTOVO:  %ROOT%\dist\AntOpt\AntOpt.exe
echo ======================================================================
echo.
echo   Prenaset se musi CELA slozka dist\AntOpt, ne jen exe soubor.
echo.
explorer "%ROOT%\dist"
pause
