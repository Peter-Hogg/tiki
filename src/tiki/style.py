"""A fixed light look for TIKI's own windows, independent of the system theme.

Applied per window (not app-wide) so napari's image windows keep their theme.
Child widgets, menus and dialogs parented to a styled window inherit it.
"""

from __future__ import annotations

from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import QStyleFactory, QWidget

ACCENT = '#2f6fd0'
TEXT = '#1e1e1e'
MUTED = '#8a8a8a'
WINDOW = '#f3f3f3'
BASE = '#ffffff'
BORDER = '#b9b9b9'

STYLESHEET = f"""
QWidget {{
    background: {WINDOW};
    color: {TEXT};
}}
QMenuBar {{
    background: #ececec;
    border-bottom: 1px solid #d4d4d4;
}}
QMenuBar::item {{
    background: transparent;
    padding: 3px 8px;
}}
QMenuBar::item:selected, QMenuBar::item:pressed {{
    background: #cfe2f7;
    border-radius: 3px;
}}
QMenu {{
    background: {BASE};
    border: 1px solid {BORDER};
    padding: 4px 0;
}}
QMenu::item {{
    background: transparent;
    padding: 4px 28px 4px 22px;
}}
QMenu::item:selected {{
    background: {ACCENT};
    color: white;
}}
QMenu::item:disabled {{
    color: {MUTED};
}}
QMenu::separator {{
    height: 1px;
    background: #dcdcdc;
    margin: 4px 8px;
}}
QToolButton {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: 3px;
}}
QToolButton:hover {{
    background: #e2e8f0;
    border-color: #c3ccd6;
}}
QToolButton:checked {{
    background: #cfe2f7;
    border-color: {ACCENT};
}}
QStatusBar {{
    background: #e9e9e9;
    border-top: 1px solid #d4d4d4;
}}
QStatusBar QLabel, QCheckBox, QLabel {{
    background: transparent;
}}
QPushButton {{
    background: {BASE};
    border: 1px solid {BORDER};
    border-radius: 3px;
    padding: 3px 12px;
}}
QPushButton:hover {{
    border-color: {ACCENT};
}}
QPushButton:pressed {{
    background: #dde7f5;
}}
QLineEdit, QSpinBox, QComboBox {{
    background: {BASE};
    border: 1px solid {BORDER};
    border-radius: 3px;
    padding: 2px 4px;
    selection-background-color: {ACCENT};
    selection-color: white;
}}
QComboBox QAbstractItemView {{
    background: {BASE};
    selection-background-color: {ACCENT};
    selection-color: white;
}}
QTableWidget, QListWidget {{
    background: {BASE};
    alternate-background-color: #f6f8fa;
    gridline-color: #e3e3e3;
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT};
    selection-color: white;
}}
QHeaderView::section {{
    background: #ececec;
    border: none;
    border-right: 1px solid #d4d4d4;
    border-bottom: 1px solid #d4d4d4;
    padding: 3px 6px;
}}
QTableCornerButton::section {{
    background: #ececec;
    border: none;
}}
"""


def light_palette() -> QPalette:
    """Fusion's light palette, so checkboxes, arrows and scrollbars match."""
    pal = QStyleFactory.create('Fusion').standardPalette()
    for role, color in (
        (QPalette.Window, WINDOW),
        (QPalette.WindowText, TEXT),
        (QPalette.Base, BASE),
        (QPalette.Text, TEXT),
        (QPalette.ButtonText, TEXT),
        (QPalette.Highlight, ACCENT),
        (QPalette.HighlightedText, '#ffffff'),
        (QPalette.ToolTipBase, '#ffffe1'),
        (QPalette.ToolTipText, '#000000'),
    ):
        pal.setColor(role, QColor(color))
    return pal


def apply(widget: QWidget) -> None:
    widget.setPalette(light_palette())
    widget.setStyleSheet(STYLESHEET)
