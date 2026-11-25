"""In-app update check against GitHub Releases (stdlib only, no extra deps)."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

from qtpy.QtCore import QUrl
from qtpy.QtGui import QDesktopServices
from qtpy.QtWidgets import QMessageBox

REPO = 'Peter-Hogg/tiki'
RELEASES_URL = f'https://github.com/{REPO}/releases'
API_URL = f'https://api.github.com/repos/{REPO}/releases/latest'


def current_version() -> str:
    try:
        from importlib.metadata import version

        return version('napari-tiki')
    except Exception:
        from . import __version__

        return __version__


def parse_version(ver: str) -> tuple[int, ...]:
    """Parse ``'v0.2.1'`` or ``'0.2.1'`` into a comparable integer tuple."""
    nums = [int(n) for n in re.findall(r'\d+', ver or '')]
    return tuple(nums) if nums else (0,)


def compare_versions(current: str, latest: str) -> int:
    """Return ``-1`` if ``latest`` is newer, ``0`` if equal, ``+1`` if ``current`` is newer."""
    a, b = parse_version(current), parse_version(latest)
    n = max(len(a), len(b))
    a = a + (0,) * (n - len(a))
    b = b + (0,) * (n - len(b))
    return (a > b) - (a < b)


def fetch_latest_release(url: str = API_URL, timeout: float = 5.0) -> dict:
    """Fetch latest release metadata from GitHub Releases API."""
    req = urllib.request.Request(
        url,
        headers={'Accept': 'application/vnd.github+json', 'User-Agent': 'napari-tiki'},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return {'version': '', 'url': RELEASES_URL, 'notes': '', 'none_published': True}
        raise RuntimeError(f'GitHub returned HTTP {exc.code}') from exc
    except Exception as exc:
        raise RuntimeError(f'Could not reach GitHub ({exc})') from exc

    tag = str(data.get('tag_name') or data.get('name') or '').lstrip('v')
    return {
        'version': tag,
        'url': str(data.get('html_url') or RELEASES_URL),
        'notes': str(data.get('body') or '').strip(),
        'none_published': False,
    }


def check_for_updates_dialog(tiki, fetcher=fetch_latest_release) -> None:
    """``Help ▸ Check for Updates…`` dialog."""
    cur = current_version()
    tiki.status('Checking for updates…')
    try:
        info = fetcher()
    except Exception as exc:
        box = QMessageBox(
            QMessageBox.Information,
            'Check for Updates',
            f'Could not check for updates right now.\n\n{exc}\n\nInstalled version: TIKI {cur}',
            QMessageBox.Close,
            tiki,
        )
        open_btn = box.addButton('Open Releases Page', QMessageBox.ActionRole)
        box.exec_()
        if box.clickedButton() is open_btn:
            QDesktopServices.openUrl(QUrl(RELEASES_URL))
        return

    if info.get('none_published') or not info.get('version'):
        tiki.status(f'TIKI {cur} (no GitHub releases published yet)')
        QMessageBox.information(
            tiki,
            'Check for Updates',
            f'You are running TIKI {cur}.\n\nNo packaged GitHub releases are published yet.',
        )
        return

    latest = info['version']
    cmp = compare_versions(cur, latest)
    if cmp < 0:
        tiki.status(f'Update available: TIKI {latest}')
        msg = f'A new version of TIKI is available!\n\nInstalled: {cur}\nLatest: {latest}'
        box = QMessageBox(QMessageBox.Information, 'Update Available', msg, QMessageBox.Close, tiki)
        if info.get('notes'):
            box.setDetailedText(info['notes'][:4000])
        dl_btn = box.addButton('Download Update', QMessageBox.AcceptRole)
        box.exec_()
        if box.clickedButton() is dl_btn:
            QDesktopServices.openUrl(QUrl(info['url']))
    else:
        tiki.status(f'TIKI {cur} is up to date')
        QMessageBox.information(
            tiki,
            'Check for Updates',
            f'You are up to date!\n\nInstalled version: TIKI {cur} (latest release: {latest}).',
        )
