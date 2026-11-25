@echo off
setlocal enabledelayedexpansion

REM Install bundled offline wheels (napari-tiki + pure-Python microscopy readers)
"%PREFIX%\python.exe" -m pip install --no-deps --no-index --find-links="%PREFIX%\wheels" "%PREFIX%\wheels\*.whl"
if errorlevel 1 exit /b 1

REM Register Windows Start Menu & Desktop shortcut via menuinst
"%PREFIX%\python.exe" -c "from menuinst.api import install; install(r'%PREFIX%\Menu\menu.json', target_prefix=r'%PREFIX%')"
if errorlevel 1 exit /b 1
