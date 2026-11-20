"""Single-instance IPC so launching ``tiki file.tif`` opens in the running TIKI.

Desktop file managers and dock icons run a new ``tiki`` process for each
"Open With" or file drop. Forwarding to the existing TIKI toolbar keeps every
image window in one WindowManager so File > Close All, the Window menu, the
ROI Manager and Image Calculator see all open images together.
"""

from __future__ import annotations

import json
import os
import re
import socket
import tempfile
from pathlib import Path


def socket_name() -> str:
    """Per-user, per-display local socket name."""
    uid = os.getuid() if hasattr(os, 'getuid') else os.environ.get('USERNAME', 'user')
    display = os.environ.get('WAYLAND_DISPLAY') or os.environ.get('DISPLAY') or 'default'
    safe_display = re.sub(r'[^A-Za-z0-9_.-]', '_', display)
    return f'tiki-{uid}-{safe_display}'


def send_to_running(files: list[str], sample: bool = False, name: str | None = None) -> bool:
    """If another TIKI is listening on ``name``, hand it ``files`` and return True."""
    target = name or socket_name()
    resolved = [str(Path(f).expanduser().resolve()) for f in files]
    payload = json.dumps({'files': resolved, 'sample': bool(sample)}).encode('utf-8') + b'\n'

    if hasattr(socket, 'AF_UNIX'):
        path = Path(tempfile.gettempdir()) / target
        if path.exists():
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                    s.settimeout(0.5)
                    s.connect(str(path))
                    s.sendall(payload)
                return True
            except OSError:
                return False

    from qtpy.QtNetwork import QLocalSocket

    sock = QLocalSocket()
    sock.connectToServer(target)
    if not sock.waitForConnected(200):
        return False
    sock.write(payload)
    sock.flush()
    sock.waitForBytesWritten(1000)
    sock.disconnectFromServer()
    return True


def _activate_toolbar(tiki) -> None:
    if tiki.isMinimized():
        tiki.showNormal()
    else:
        tiki.show()
    tiki.raise_()
    tiki.activateWindow()


def _handle_message(tiki, raw: bytes) -> None:
    try:
        msg = json.loads(raw.decode('utf-8').strip())
    except Exception:
        return
    files = msg.get('files') or []
    sample = bool(msg.get('sample'))
    if files:
        tiki.open_dropped(files)
    if sample:
        tiki._run('Open Sample', tiki.open_sample_movie)
    if not files and not sample:
        _activate_toolbar(tiki)


def listen(tiki, name: str | None = None):
    """Listen for secondary ``tiki`` launches and open their files in ``tiki``."""
    from qtpy.QtCore import QTimer
    from qtpy.QtNetwork import QLocalServer

    target = name or socket_name()
    QLocalServer.removeServer(target)
    server = QLocalServer(tiki)
    opt = getattr(QLocalServer, 'UserAccessOption', None) or QLocalServer.SocketOption.UserAccessOption
    server.setSocketOptions(opt)
    active_conns: set = set()

    def on_connection() -> None:
        while server.hasPendingConnections():
            conn = server.nextPendingConnection()
            if conn is None:
                continue
            active_conns.add(conn)
            buf = bytearray()

            def drain(b=buf) -> None:
                while b'\n' in b:
                    line, _, rest = b.partition(b'\n')
                    b.clear()
                    b.extend(rest)
                    if line.strip():
                        QTimer.singleShot(0, lambda raw=bytes(line): _handle_message(tiki, raw))

            def read(c=conn, b=buf) -> None:
                b.extend(bytes(c.readAll()))
                drain(b)

            def cleanup(c=conn, b=buf) -> None:
                b.extend(bytes(c.readAll()))
                drain(b)
                if b.strip():
                    raw = bytes(b)
                    b.clear()
                    QTimer.singleShot(0, lambda r=raw: _handle_message(tiki, r))
                active_conns.discard(c)
                c.deleteLater()

            conn.readyRead.connect(read)
            conn.disconnected.connect(cleanup)
            if conn.bytesAvailable():
                read()

    server.newConnection.connect(on_connection)
    server.listen(target)
    return server
