"""Floating TIKI windows: Results table and live Z-axis profile plot."""

from __future__ import annotations

import csv

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import ops
from . import roi as roi_mod


class ResultsTable(QWidget):
    """ImageJ's Results window: one row per Measure (Ctrl+M)."""

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle('Results')
        self.resize(640, 260)
        self.columns: list[str] = []
        self.rows: list[dict] = []
        self.table = QTableWidget(0, 0)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        buttons = QHBoxLayout()
        for text, slot in (('Copy', self.copy), ('Save As…', self.save_as), ('Clear', self.clear)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            buttons.addWidget(b)
        buttons.addStretch()
        layout = QVBoxLayout(self)
        layout.addWidget(self.table)
        layout.addLayout(buttons)

    def add_row(self, row: dict) -> None:
        for key in row:
            if key not in self.columns:
                self.columns.append(key)
                self.table.setColumnCount(len(self.columns))
                self.table.setHorizontalHeaderLabels(self.columns)
                for r, old in enumerate(self.rows):
                    self._set_cell(r, len(self.columns) - 1, old.get(key, ''))
        self.rows.append(row)
        r = self.table.rowCount()
        self.table.insertRow(r)
        for c, key in enumerate(self.columns):
            self._set_cell(r, c, row.get(key, ''))
        self.table.scrollToBottom()

    def _set_cell(self, r: int, c: int, value) -> None:
        if isinstance(value, float):
            text = f'{value:.4g}' if abs(value) < 1e5 else f'{value:.6g}'
        else:
            text = str(value)
        item = QTableWidgetItem(text)
        if isinstance(value, (int, float)):
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.table.setItem(r, c, item)

    def as_text(self, sep: str = '\t') -> str:
        lines = [sep.join([' '] + self.columns)]
        for i, row in enumerate(self.rows, 1):
            lines.append(sep.join([str(i)] + [str(row.get(k, '')) for k in self.columns]))
        return '\n'.join(lines)

    def copy(self) -> None:
        QApplication.clipboard().setText(self.as_text())

    def save_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, 'Save Results', 'Results.csv', 'CSV (*.csv)')
        if not path:
            return
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writeheader()
            writer.writerows(self.rows)

    def clear(self) -> None:
        self.rows.clear()
        self.columns.clear()
        self.table.clear()
        self.table.setRowCount(0)
        self.table.setColumnCount(0)


class ProfileWindow(QWidget):
    """Mean intensity in the current ROI along a stack axis, updated live.

    Redraws when the ROI is drawn, moved, resized or re-selected, and marks
    the frame currently shown in the image window.
    """

    def __init__(self, viewer, image, axis: int, parent=None):
        super().__init__(parent, Qt.Window)
        self.viewer = viewer
        self.image = image
        self.axis = axis
        self.roi = roi_mod.roi_layer(viewer, create=True)
        self._signature = None
        self.profile = np.array([])
        self.setWindowTitle(f'{image.name} — profile')
        self.resize(560, 320)

        self.figure = Figure(figsize=(5.6, 3.0), layout='constrained')
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot()
        (self.line,) = self.ax.plot([], [], lw=1, color='k')
        self.cursor_line = self.ax.axvline(0, color='tab:red', lw=0.8)
        self.ax.set_xlabel(roi_mod.axis_name(viewer, image, axis))
        self.ax.set_ylabel('mean in ROI')

        self.live = QCheckBox('Live')
        self.live.setChecked(True)
        self.live.toggled.connect(lambda on: on and self.schedule())
        copy = QPushButton('Copy')
        copy.clicked.connect(self.copy)
        save = QPushButton('Save CSV…')
        save.clicked.connect(self.save_csv)
        bar = QHBoxLayout()
        bar.addWidget(self.live)
        bar.addStretch()
        bar.addWidget(copy)
        bar.addWidget(save)
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
        layout.addLayout(bar)

        self._timer = QTimer(self, singleShot=True, interval=60)
        self._timer.timeout.connect(self.update_profile)
        self._connections = [
            (self.roi.events.data, self.schedule),
            (self.roi.events.set_data, self.schedule),
            (self.roi.events.highlight, self.schedule),
            (self.viewer.dims.events.current_step, self.on_step),
        ]
        for emitter, cb in self._connections:
            emitter.connect(cb)
        self.update_profile(force=True)

    def schedule(self, event=None) -> None:
        if self.live.isChecked():
            self._timer.start()

    def _roi_signature(self):
        shape = roi_mod.current_shape(self.roi)
        if shape is None:
            return None
        verts, kind = shape
        return kind, np.round(verts[:, -2:], 2).tobytes()

    def update_profile(self, force: bool = False) -> None:
        sig = self._roi_signature()
        if not force and sig == self._signature:
            return
        self._signature = sig
        shape2d = roi_mod.spatial_shape(self.image)
        current = roi_mod.current_shape(self.roi)
        mask = None if current is None else roi_mod.shape_mask(*current, shape2d)
        index = roi_mod.current_index(self.viewer, self.image)
        data = self.image.data
        if self.image.rgb:
            data = np.asarray(data)[..., :3].mean(axis=-1)
        self.profile = ops.axis_profile(data, self.axis, index, mask)
        x = np.arange(self.profile.size)
        self.line.set_data(x, self.profile)
        self.ax.set_title('whole frame' if mask is None else f'{current[1]} ROI, {int(mask.sum())} px', fontsize=9)
        self.ax.relim()
        self.ax.autoscale_view()
        self.on_step()

    def on_step(self, event=None) -> None:
        index = roi_mod.current_index(self.viewer, self.image)
        self.cursor_line.set_xdata([index[self.axis]] * 2)
        self.canvas.draw_idle()

    def copy(self) -> None:
        QApplication.clipboard().setText('\n'.join(f'{i}\t{v}' for i, v in enumerate(self.profile)))

    def save_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, 'Save profile', f'{self.image.name}_profile.csv', 'CSV (*.csv)')
        if path:
            np.savetxt(path, np.c_[np.arange(self.profile.size), self.profile], delimiter=',',
                       header='frame,mean', comments='', fmt=['%d', '%.6g'])

    def closeEvent(self, event):
        for emitter, cb in self._connections:
            try:
                emitter.disconnect(cb)
            except Exception:
                pass
        super().closeEvent(event)


class PlotWindow(QWidget):
    """A static x/y plot (Analyze > Plot Profile)."""

    def __init__(self, title: str, x, y, xlabel: str, ylabel: str, parent=None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle(title)
        self.resize(520, 300)
        self.x, self.y = np.asarray(x), np.asarray(y)
        figure = Figure(figsize=(5.2, 2.8), layout='constrained')
        canvas = FigureCanvasQTAgg(figure)
        ax = figure.add_subplot()
        ax.plot(self.x, self.y, lw=1, color='k')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        copy = QPushButton('Copy')
        copy.clicked.connect(
            lambda: QApplication.clipboard().setText('\n'.join(f'{a}\t{b}' for a, b in zip(self.x, self.y)))
        )
        layout = QVBoxLayout(self)
        layout.addWidget(canvas)
        layout.addWidget(copy, alignment=Qt.AlignRight)
