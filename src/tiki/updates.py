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


def _version_key(ver: str) -> tuple[tuple[int, ...], int, str, int]:
    """Parse PEP 440 / semver strings so pre-releases sort before final releases."""
    s = (ver or '').strip().lstrip('vV')
    m = re.match(r'^(\d+(?:\.\d+)*)(?:[-._]?(a|alpha|b|beta|rc|c|pre|dev)(\d*))?', s, re.IGNORECASE)
    if not m:
        nums = tuple(int(n) for n in re.findall(r'\d+', s)) or (0,)
        return nums, 1, '', 0
    core = tuple(int(x) for x in m.group(1).split('.'))
    tag = (m.group(2) or '').lower()
    if not tag:
        return core, 1, '', 0
    tag_map = {'dev': '0_dev', 'a': '1_a', 'alpha': '1_a', 'b': '2_b', 'beta': '2_b', 'c': '3_rc', 'pre': '3_rc', 'rc': '3_rc'}
    pre_num = int(m.group(3)) if m.group(3) else 0
    return core, 0, tag_map.get(tag, tag), pre_num


def parse_version(ver: str) -> tuple[int, ...]:
    """Parse ``'v0.2.1'`` or ``'0.2.1'`` into a comparable integer tuple."""
    return _version_key(ver)[0]


def compare_versions(current: str, latest: str) -> int:
    """Return ``-1`` if ``latest`` is newer, ``0`` if equal, ``+1`` if ``current`` is newer."""
    ca, cf, ct, cn = _version_key(current)
    la, lf, lt, ln = _version_key(latest)
    n = max(len(ca), len(la))
    ca = ca + (0,) * (n - len(ca))
    la = la + (0,) * (n - len(la))
    ka = (ca, cf, ct, cn)
    kb = (la, lf, lt, ln)
    return (ka > kb) - (ka < kb)


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
        if hasattr(tiki, 'run_task'):
            info, cancelled = tiki.run_task('Check for Updates', 0, lambda _tick: fetcher())
            if cancelled:
                tiki.status('Update check cancelled')
                return
        else:
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
