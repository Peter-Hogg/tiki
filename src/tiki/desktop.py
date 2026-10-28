"""``tiki --install-desktop``: an application-menu launcher (Linux).

Writes a freedesktop ``.desktop`` file and the icon into the user's data
folder (no admin rights), so TIKI appears in the app menu, can be pinned to
the dock, opens files dropped on its icon, and is offered under "Open With"
for TIFF files. ``tiki --uninstall-desktop`` removes both again.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

APP_ID = 'tiki'
ICON = Path(__file__).parent / 'resources' / 'tiki.png'
MIME_TYPES = ['image/tiff', 'image/png', 'image/jpeg']


def data_home() -> Path:
    return Path(os.environ.get('XDG_DATA_HOME') or Path.home() / '.local' / 'share')


def desktop_path() -> Path:
    return data_home() / 'applications' / f'{APP_ID}.desktop'


def icon_path() -> Path:
    return data_home() / 'icons' / 'hicolor' / '256x256' / 'apps' / f'{APP_ID}.png'


def desktop_entry(python: str | None = None) -> str:
    """The .desktop text; launches this Python's TIKI so it works whatever the PATH."""
    exec_cmd = f'{shlex.quote(python or sys.executable)} -m tiki %F'
    return '\n'.join([
        '[Desktop Entry]',
        'Type=Application',
        'Name=TIKI',
        'GenericName=Image analysis',
        'Comment=TIKI Is Kinda ImageJ: an ImageJ-style toolbar for napari',
        f'Exec={exec_cmd}',
        f'Icon={APP_ID}',
        'Terminal=false',
        'Categories=Science;Biology;DataVisualization;',
        'Keywords=ImageJ;Fiji;napari;microscopy;image;',
        f'MimeType={";".join(MIME_TYPES)};',
        f'StartupWMClass={APP_ID}',
        'StartupNotify=true',
        '',
    ])


def install() -> list[Path]:
    if not sys.platform.startswith('linux'):
        raise RuntimeError(
            'the desktop launcher is Linux-only for now; make a shortcut that runs:\n'
            f'  {sys.executable} -m tiki')
    desktop_path().parent.mkdir(parents=True, exist_ok=True)
    icon_path().parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ICON, icon_path())
    desktop_path().write_text(desktop_entry())
    desktop_path().chmod(0o755)
    _refresh()
    return [desktop_path(), icon_path()]


def uninstall() -> list[Path]:
    removed = []
    for p in (desktop_path(), icon_path()):
        if p.exists():
            p.unlink()
            removed.append(p)
    _refresh()
    return removed


def _refresh() -> None:
    """Let the desktop notice the change (best effort; harmless if tools are missing)."""
    for cmd in (['update-desktop-database', str(desktop_path().parent)],
                ['gtk-update-icon-cache', '-q', '-t', str(data_home() / 'icons' / 'hicolor')]):
        if shutil.which(cmd[0]):
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
