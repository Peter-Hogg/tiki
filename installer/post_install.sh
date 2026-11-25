#!/usr/bin/env bash
set -euo pipefail

# Install bundled offline wheels (napari-tiki + pure-Python microscopy readers)
"$PREFIX/bin/python" -m pip install --no-deps --no-index --find-links="$PREFIX/wheels" "$PREFIX"/wheels/*.whl

# Register OS application shortcut (macOS .app / Linux .desktop) via menuinst
"$PREFIX/bin/python" -c "from menuinst.api import install; install('$PREFIX/Menu/menu.json', target_prefix='$PREFIX')"
