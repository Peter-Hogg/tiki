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


def _check_mark_svg() -> str:
    """Write a white tick SVG once so checked boxes show a mark, not just colour."""
    import tempfile
    from pathlib import Path

    path = Path(tempfile.gettempdir()) / 'tiki_check.svg'
    if not path.exists():
        path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 13 13">'
            '<path d="M2.5 6.8 5.3 9.5 10.5 3.5" fill="none" stroke="white" stroke-width="2"/></svg>'
        )
    return path.as_posix()


_CHECK = _check_mark_svg()

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
QLineEdit, QComboBox {{
    background: {BASE};
    border: 1px solid {BORDER};
    border-radius: 3px;
    padding: 2px 4px;
    selection-background-color: {ACCENT};
    selection-color: white;
}}
QCheckBox::indicator {{
    width: 13px;
    height: 13px;
    border: 1px solid #8c8c8c;
    border-radius: 2px;
    background: {BASE};
}}
QCheckBox::indicator:hover {{
    border-color: {ACCENT};
}}
QCheckBox::indicator:checked {{
    background: {ACCENT};
    border-color: {ACCENT};
    image: url({_CHECK});
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
