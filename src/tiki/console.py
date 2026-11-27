"""Plugins ▸ Python Console: an interactive REPL bound to TIKI's live windows.

Uses only Python's standard library (``code.InteractiveConsole`` +
``rlcompleter``) so it works everywhere with zero extra dependencies.
``tiki`` is the same :class:`~tiki.macros.MacroAPI` used in macros, so
snippets tested here paste directly into ``Plugins ▸ Macros ▸ New Macro…``.
"""

from __future__ import annotations

import code
import io as py_io
import rlcompleter
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QFont, QTextCharFormat, QTextCursor
from qtpy.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from . import io, ops, style
from . import roi as roi_mod
from .macros import MacroAPI

BANNER = (
    'TIKI Python Console  (Up/Down = history, Tab = complete, Enter = run)\n'
    'Pre-loaded: tiki (MacroAPI), viewer, layer, data, np, ops, io, results, show(arr, name), app\n'
    'Note: viewer, layer, and data automatically track the active image window.\n'
)


class _ConsoleInput(QLineEdit):
    """Single-line REPL input with Up/Down history and Tab completion."""

    def __init__(self, on_history, on_complete, parent=None):
        super().__init__(parent)
        self._on_history = on_history
        self._on_complete = on_complete

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Up:
            self._on_history(-1)
            return
        if key == Qt.Key_Down:
            self._on_history(1)
            return
        if key == Qt.Key_Tab:
            self._on_complete()
            return
        super().keyPressEvent(event)


class _LocalConsole(code.InteractiveConsole):
    """InteractiveConsole that always writes tracebacks to stderr (never ``sys.excepthook``)."""

    def showtraceback(self) -> None:
        exc_type, exc_val, exc_tb = sys.exc_info()
        # Strip the top frame (InteractiveConsole.runcode) so tracebacks start at <console>
        tb = exc_tb.tb_next if exc_tb is not None else None
        lines = traceback.format_exception(exc_type, exc_val, tb)
        self.write(''.join(lines))

    def showsyntaxerror(self, filename=None, **kwargs) -> None:
        exc_type, exc_val, _ = sys.exc_info()
        lines = traceback.format_exception_only(exc_type, exc_val)
        self.write(''.join(lines))


class PythonConsoleWindow(QWidget):
    """Floating interactive Python console connected to the running TIKI app."""

    def __init__(self, app, parent=None):
        super().__init__(parent, Qt.Window)
        style.apply(self)
        self.app = app
        self.api = MacroAPI(app)
        self.setWindowTitle('TIKI — Python Console')
        self.resize(680, 420)

        self.history: list[str] = []
        self._hist_pos: int = 0
        self._hist_draft: str = ''
        self._more: bool = False

        mono = QFont('Monospace')
        mono.setStyleHint(QFont.Monospace)
        mono.setPointSize(10)

        # Header bar
        self.active_label = QLabel()
        clear_btn = QPushButton('Clear')
        copy_btn = QPushButton('Copy Output')
        reset_btn = QPushButton('Reset')
        clear_btn.clicked.connect(self.clear_output)
        copy_btn.clicked.connect(self.copy_output)
        reset_btn.clicked.connect(self.reset_console)

        top = QHBoxLayout()
        top.addWidget(self.active_label, 1)
        top.addWidget(clear_btn)
        top.addWidget(copy_btn)
        top.addWidget(reset_btn)

        # Output log
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setFont(mono)
        self.output.setStyleSheet(
            'QPlainTextEdit { background-color: #1e222a; color: #eceff4; border: 1px solid #3b4252; }'
        )

        # Prompt + input line
        self.prompt_label = QLabel('>>>')
        self.prompt_label.setFont(mono)
        self.input = _ConsoleInput(self._step_history, self._complete)
        self.input.setFont(mono)
        self.input.setPlaceholderText('Type Python here (e.g. data.shape, show(data * 2, "x2"))…')
        self.input.returnPressed.connect(self._on_enter)

        bottom = QHBoxLayout()
        bottom.addWidget(self.prompt_label)
        bottom.addWidget(self.input, 1)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.output, 1)
        layout.addLayout(bottom)

        self.namespace: dict = {}
        self.console = _LocalConsole(self.namespace)
        self._init_namespace()
        self._append(BANNER, '#88c0d0')
        self._connected = False
        self._connect_signals()
        self._update_active_badge()
        self.input.setFocus()

    def _connect_signals(self) -> None:
        if not self._connected:
            self.app.windows.active_changed.connect(self._update_active_badge)
            self._connected = True

    def showEvent(self, event):
        self._connect_signals()
        self._update_active_badge()
        super().showEvent(event)

    # ------------------------------------------------------------ namespace

    def _init_namespace(self) -> None:
        self.namespace.clear()
        self.namespace.update({
            'tiki': self.api,
            'app': self.app,
            'np': np,
            'ops': ops,
            'io': io,
            'results': self.app.results.rows,
            'show': self.api.show,
        })
        self._sync_live_vars()

    def _sync_live_vars(self) -> None:
        viewer = self.app.windows.active
        layer = roi_mod.target_image(viewer) if viewer is not None else None
        self.namespace['viewer'] = viewer
        self.namespace['layer'] = layer
        self.namespace['data'] = layer.data if layer is not None else None
        self.namespace['results'] = self.app.results.rows

    def _update_active_badge(self, _viewer=None) -> None:
        self._sync_live_vars()
        viewer = self.namespace.get('viewer')
        layer = self.namespace.get('layer')
        if viewer is None:
            self.active_label.setText('Active: (no image open)')
        elif layer is None:
            self.active_label.setText(f'Active: {viewer.title}')
        else:
            shape = '×'.join(str(s) for s in layer.data.shape)
            self.active_label.setText(f'Active: {layer.name}  ({shape}, {layer.data.dtype})')

    # ------------------------------------------------------------ execution

    def _append(self, text: str, color: str | None = None) -> None:
        if not text:
            return
        cursor = self.output.textCursor()
        cursor.movePosition(QTextCursor.End)
        fmt = QTextCharFormat()
        if color:
            fmt.setForeground(QColor(color))
        cursor.insertText(text, fmt)
        self.output.setTextCursor(cursor)
        self.output.ensureCursorVisible()

    def run_line(self, line: str) -> bool:
        """Push one line of Python to the console; returns True if more input is needed."""
        prompt = '... ' if self._more else '>>> '
        self._append(f'{prompt}{line}\n', '#81a1c1')
        if line.strip() and (not self.history or self.history[-1] != line):
            self.history.append(line)
        self._hist_pos = len(self.history)
        self._hist_draft = ''

        self._sync_live_vars()
        out_buf = py_io.StringIO()
        err_buf = py_io.StringIO()
        with redirect_stdout(out_buf), redirect_stderr(err_buf):
            try:
                self._more = self.console.push(line)
            except SystemExit:
                self._append('SystemExit caught (close the console window instead).\n', '#ebcb8b')
                self._more = False
            except BaseException:
                self.console.showtraceback()
                self._more = False

        stdout_text = out_buf.getvalue()
        stderr_text = err_buf.getvalue()
        if stdout_text:
            self._append(stdout_text, '#eceff4')
        if stderr_text:
            self._append(stderr_text, '#bf616a')
        self.prompt_label.setText('...' if self._more else '>>>')
        self._update_active_badge()
        return self._more

    def _on_enter(self) -> None:
        line = self.input.text()
        self.input.clear()
        self.run_line(line)

    # ------------------------------------------------- history & completion

    def _step_history(self, delta: int) -> None:
        if not self.history:
            return
        if self._hist_pos == len(self.history):
            self._hist_draft = self.input.text()
        new_pos = max(0, min(len(self.history), self._hist_pos + delta))
        if new_pos == self._hist_pos:
            return
        self._hist_pos = new_pos
        if new_pos == len(self.history):
            self.input.setText(self._hist_draft)
        else:
            self.input.setText(self.history[new_pos])

    def _complete(self) -> None:
        text = self.input.text()
        pos = self.input.cursorPosition()
        prefix = text[:pos]
        # Extract the identifier / attribute chain right before the cursor
        start = pos
        while start > 0 and (prefix[start - 1].isalnum() or prefix[start - 1] in '._'):
            start -= 1
        token = prefix[start:pos]
        if not token:
            return
        self._sync_live_vars()
        completer = rlcompleter.Completer(self.namespace)
        matches = []
        state = 0
        while True:
            cand = completer.complete(token, state)
            if cand is None:
                break
            if cand not in matches:
                matches.append(cand)
            state += 1
            if state > 200:
                break
        if not matches:
            return
        if len(matches) == 1:
            comp = matches[0]
            self.input.setText(text[:start] + comp + text[pos:])
            self.input.setCursorPosition(start + len(comp))
        else:
            # Longest common prefix
            common = matches[0]
            for m in matches[1:]:
                i = 0
                while i < len(common) and i < len(m) and common[i] == m[i]:
                    i += 1
                common = common[:i]
            if len(common) > len(token):
                self.input.setText(text[:start] + common + text[pos:])
                self.input.setCursorPosition(start + len(common))
            else:
                preview = '  '.join(matches[:20]) + ('  …' if len(matches) > 20 else '')
                self._append(f'{preview}\n', '#a3be8c')

    # -------------------------------------------------------------- actions

    def clear_output(self) -> None:
        self.output.clear()

    def copy_output(self) -> None:
        QApplication.clipboard().setText(self.output.toPlainText())

    def reset_console(self) -> None:
        self.console.resetbuffer()
        self._more = False
        self.prompt_label.setText('>>>')
        self._init_namespace()
        self._append('--- Console namespace reset ---\n', '#ebcb8b')

    def closeEvent(self, event):
        if self._connected:
            try:
                self.app.windows.active_changed.disconnect(self._update_active_badge)
            except Exception:
                pass
            self._connected = False
        for key in ('viewer', 'layer', 'data'):
            self.namespace[key] = None
        super().closeEvent(event)
