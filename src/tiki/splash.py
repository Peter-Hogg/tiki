"""Lightweight startup splash screen shown before heavy imports (napari, scipy)."""

from __future__ import annotations

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


def show_splash(app, version: str = '', message: str = 'Starting TIKI…') -> QSplashScreen:
    """Create and display the splash screen immediately."""
    splash = QSplashScreen(build_pixmap(version), Qt.WindowStaysOnTopHint)
    splash.showMessage(message, Qt.AlignBottom | Qt.AlignLeft, QColor('#a3be8c'))
    splash.show()
    app.processEvents()
    return splash
