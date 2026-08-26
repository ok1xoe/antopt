@echo off
rem Sestavi instalator AntOpt-1.0.msi pro Windows.
rem
rem Potrebuje WiX Toolset v3 (candle.exe, light.exe, heat.exe):
rem   https://github.com/wixtoolset/wix3/releases
rem Nejdriv musi probehnout build_windows.bat, ktery vyrobi dist\AntOpt.
setlocal
cd /d "%~dp0\.."
set ROOT=%CD%
set DIST=%ROOT%\dist\AntOpt
set WORK=%ROOT%\build\windows

echo.
echo ======================================================================
echo   Instalator AntOpt.msi
echo ======================================================================
echo.

if not exist "%DIST%\AntOpt.exe" (
    echo   Nejdriv spust build_windows.bat - chybi %DIST%\AntOpt.exe
    pause & exit /b 1
)

where candle.exe >nul 2>&1
if errorlevel 1 (
    echo   Nenasel jsem WiX Toolset v3 ^(candle.exe^).
    echo.
    echo   Stahni a nainstaluj:
    echo       https://github.com/wixtoolset/wix3/releases
    echo   a pridej jeho slozku bin do PATH, napr.:
    echo       set PATH=%%PATH%%;C:\Program Files ^(x86^)\WiX Toolset v3.14\bin
    echo.
    echo   Mas-li WiX v4 nebo novejsi, prevede se zdroj prikazem:  wix convert
    pause & exit /b 1
)

if not exist "%ROOT%\build\icon.ico" (
    echo   Chybi ikona - spoustim make_icon.py
    "%ROOT%\.build-venv\Scripts\python.exe" "%ROOT%\build\make_icon.py"
)
copy /y "%ROOT%\build\icon.ico" "%WORK%\icon.ico" >nul

echo   Prochazim soubory aplikace...
heat.exe dir "%DIST%" -cg AppFiles -dr INSTALLDIR -gg -g1 -sfrag -srd -sreg ^
    -var var.SourceDir -out "%WORK%\files.wxs" || (pause & exit /b 1)

echo   Prekladam...
if not exist "%WORK%\obj" mkdir "%WORK%\obj"
candle.exe -nologo -dSourceDir="%DIST%" -out "%WORK%\obj\" ^
    "%WORK%\antopt.wxs" "%WORK%\files.wxs" || (pause & exit /b 1)

echo   Sestavuji MSI...
light.exe -nologo -b "%DIST%" -sice:ICE60 -out "%ROOT%\dist\AntOpt-1.0.msi" ^
    "%WORK%\obj\antopt.wixobj" "%WORK%\obj\files.wixobj" || (pause & exit /b 1)

echo.
echo ======================================================================
echo   HOTOVO:  %ROOT%\dist\AntOpt-1.0.msi
echo ======================================================================
echo.
echo   Instaluje do Program Files, vyrobi zastupce v nabidce Start
echo   i na plose a jde odinstalovat pres Aplikace a funkce.
echo.
explorer "%ROOT%\dist"
pause
