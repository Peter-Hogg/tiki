#!/usr/bin/env bash
set -euo pipefail

# Unpack bundled offline wheels, install them, and register the OS shortcut
"$PREFIX/bin/python" - <<'PY'
import glob, os, shutil, subprocess, sys, tempfile, zipfile

prefix = os.environ['PREFIX']
zip_path = os.path.join(prefix, 'wheels.zip')
tmp = tempfile.mkdtemp(prefix='tiki_wheels_')
try:
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp)
    whls = sorted(glob.glob(os.path.join(tmp, '*.whl')))
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-index', *whls])
finally:
    shutil.rmtree(tmp, ignore_errors=True)
    if os.path.exists(zip_path):
        os.remove(zip_path)

from menuinst.api import install
install(os.path.join(prefix, 'Menu', 'menu.json'), target_prefix=prefix)
PY
