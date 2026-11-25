@echo off
setlocal enabledelayedexpansion

REM Install bundled offline wheels (glob in Python since cmd.exe does not expand *.whl)
"%PREFIX%\python.exe" -c "import glob, os, subprocess, sys; whls = sorted(glob.glob(os.path.join(os.environ['PREFIX'], 'wheels', '*.whl'))); subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-index', *whls])"
if errorlevel 1 exit /b 1

REM Register Windows Start Menu & Desktop shortcut via menuinst
"%PREFIX%\python.exe" -c "import os; from menuinst.api import install; p = os.environ['PREFIX']; install(os.path.join(p, 'Menu', 'menu.json'), target_prefix=p)"
if errorlevel 1 exit /b 1
