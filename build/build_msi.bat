@echo off
rem Sestavi instalator AntOpt-<verze>.msi pro Windows.
rem Verzi lze urcit promennou ANTOPT_VERSION, jinak je 1.0.0.
rem
rem Potrebuje WiX Toolset v3 (candle.exe, light.exe, heat.exe):
rem   https://github.com/wixtoolset/wix3/releases
rem Nejdriv musi probehnout build_windows.bat, ktery vyrobi dist\AntOpt.
setlocal
cd /d "%~dp0\.."
set ROOT=%CD%
set DIST=%ROOT%\dist\AntOpt
set WORK=%ROOT%\build\windows
if "%ANTOPT_VERSION%"=="" set ANTOPT_VERSION=1.0.0

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
echo   Prochazim soubory aplikace...
heat.exe dir "%DIST%" -cg AppFiles -dr INSTALLDIR -gg -g1 -sfrag -srd -sreg ^
    -var var.SourceDir -out "%WORK%\files.wxs" || (pause & exit /b 1)

rem Ikona se kopiruje az za heat, aby ji heat nepribalil mezi soubory
rem aplikace. light hleda relativni cesty vuci -b, tedy v %DIST%.
copy /y "%ROOT%\build\icon.ico" "%DIST%\icon.ico" >nul

echo   Prekladam...
if not exist "%WORK%\obj" mkdir "%WORK%\obj"
candle.exe -nologo -arch x64 -dSourceDir="%DIST%" -dVersion=%ANTOPT_VERSION% -out "%WORK%\obj\" ^
    "%WORK%\antopt.wxs" "%WORK%\files.wxs" || (pause & exit /b 1)

echo   Sestavuji MSI...
light.exe -nologo -b "%DIST%" -sice:ICE60 -out "%ROOT%\dist\AntOpt-%ANTOPT_VERSION%.msi" ^
    "%WORK%\obj\antopt.wixobj" "%WORK%\obj\files.wixobj" || (pause & exit /b 1)

echo.
echo ======================================================================
echo   HOTOVO:  %ROOT%\dist\AntOpt-%ANTOPT_VERSION%.msi
echo ======================================================================
echo.
echo   Instaluje do Program Files, vyrobi zastupce v nabidce Start
echo   i na plose a jde odinstalovat pres Aplikace a funkce.
echo.
explorer "%ROOT%\dist"
pause
