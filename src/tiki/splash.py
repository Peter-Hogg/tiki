"""Lightweight startup splash screen shown before heavy imports (napari, scipy)."""

from __future__ import annotations

import time
from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QFont, QPainter, QPixmap
from qtpy.QtWidgets import QSplashScreen

ICON = Path(__file__).parent / 'resources' / 'tiki.png'


def build_pixmap(version: str = '') -> QPixmap:
    """Render the TIKI splash card as a crisp QPixmap."""
    w, h = 380, 150
    pm = QPixmap(w, h)
    pm.fill(QColor('#1e222a'))

    painter = QPainter(pm)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setRenderHint(QPainter.SmoothPixmapTransform)

    # Subtle border
    painter.setPen(QColor('#3b4252'))
    painter.drawRect(0, 0, w - 1, h - 1)

    # Icon on the left
    if ICON.exists():
        icon_pm = QPixmap(str(ICON)).scaled(88, 88, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        painter.drawPixmap(24, (h - icon_pm.height()) // 2 - 6, icon_pm)

    # Title & subtitle
    title_font = QFont()
    title_font.setPointSize(22)
    title_font.setBold(True)
    painter.setFont(title_font)
    painter.setPen(QColor('#eceff4'))
    painter.drawText(132, 60, 'TIKI')

    sub_font = QFont()
    sub_font.setPointSize(10)
    painter.setFont(sub_font)
    painter.setPen(QColor('#d8dee9'))
    painter.drawText(132, 84, 'TIKI Is Kinda ImageJ')

    if version:
        ver_font = QFont()
        ver_font.setPointSize(9)
        painter.setFont(ver_font)
        painter.setPen(QColor('#88c0d0'))
        painter.drawText(132, 106, f'Version {version}')

    painter.end()
    return pm


def _pump(app, ms: int = 80) -> None:
    """Pump Qt events briefly so X11/Wayland compositors actually map and paint the splash."""
    end = time.monotonic() + ms / 1000.0
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()


def show_splash(app, version: str = '', message: str = 'Starting TIKI…') -> QSplashScreen:
    """Create, center, and paint the splash screen before heavy imports block."""
    flags = Qt.SplashScreen | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
    splash = QSplashScreen(build_pixmap(version), flags)
    splash._shown_at = time.monotonic()
    screen = app.primaryScreen()
    if screen is not None:
        geo = screen.availableGeometry()
        splash.move(geo.center() - splash.rect().center())
    splash.showMessage(f'  {message}', Qt.AlignBottom | Qt.AlignLeft, QColor('#a3be8c'))
    splash.show()
    splash.raise_()
    splash.repaint()
    _pump(app, ms=80)
    return splash


def update_splash(splash: QSplashScreen | None, app, message: str) -> None:
    """Update the splash status text and flush a repaint."""
    if splash is None:
        return
    splash.showMessage(f'  {message}', Qt.AlignBottom | Qt.AlignLeft, QColor('#a3be8c'))
    splash.repaint()
    app.processEvents()


def finish_splash(splash: QSplashScreen | None, app, target_window, min_ms: int = 600) -> None:
    """Close the splash once ``target_window`` is shown (keeping a brief minimum display time)."""
    if splash is None:
        return
    shown_at = getattr(splash, '_shown_at', 0.0)
    remaining_ms = int(min_ms - (time.monotonic() - shown_at) * 1000)
    if remaining_ms > 0:
        _pump(app, ms=remaining_ms)
    splash.finish(target_window)
