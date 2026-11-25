@echo off
setlocal enabledelayedexpansion

REM Unpack bundled offline wheels, install them, and register Start Menu / Desktop shortcut
"%PREFIX%\python.exe" -c "import glob, os, shutil, subprocess, sys, tempfile, zipfile; p = os.environ['PREFIX']; zp = os.path.join(p, 'wheels.zip'); tmp = tempfile.mkdtemp(prefix='tiki_wheels_'); zipfile.ZipFile(zp).extractall(tmp); whls = sorted(glob.glob(os.path.join(tmp, '*.whl'))); subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-index', *whls]); shutil.rmtree(tmp, ignore_errors=True); os.remove(zp); from menuinst.api import install; install(os.path.join(p, 'Menu', 'menu.json'), target_prefix=p)"
if errorlevel 1 exit /b 1
