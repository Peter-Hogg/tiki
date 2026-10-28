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

from . import ops, style
from . import roi as roi_mod


class ResultsTable(QWidget):
    """ImageJ's Results window: one row per Measure (Ctrl+M)."""

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Window)
        style.apply(self)
        self.setWindowTitle('Results')
        self.resize(640, 260)
        self.columns: list[str] = []
        self.rows: list[dict] = []
        self.table = QTableWidget(0, 0)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
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
        style.apply(self)
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
        style.apply(self)
        self.setWindowTitle(title)
        self.resize(520, 300)
        self.x, self.y = np.asarray(x), np.asarray(y)
        figure = Figure(figsize=(5.2, 2.8), layout='constrained')
        canvas = FigureCanvasQTAgg(figure)
        ax = figure.add_subplot()
        ax.plot(self.x, self.y, lw=1, color='k', label=ylabel)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        self.ax, self.canvas = ax, canvas
        copy = QPushButton('Copy')
        copy.clicked.connect(
            lambda: QApplication.clipboard().setText('\n'.join(f'{a}\t{b}' for a, b in zip(self.x, self.y)))
        )
        layout = QVBoxLayout(self)
        layout.addWidget(canvas)
        layout.addWidget(copy, alignment=Qt.AlignRight)

    def add_line(self, x, y, label: str) -> None:
        self.ax.plot(x, y, lw=1, color='tab:red', label=label)
        self.ax.legend(fontsize=8, frameon=False)
        self.canvas.draw_idle()


class TracesWindow(QWidget):
    """One trace per ROI or label through a stack (cells x frames).

    Stacked lines for up to ``HEATMAP_ABOVE`` traces, a heatmap beyond that.
    Exports the raw (or ΔF/F) array as .npy (cells x frames) or CSV.
    """

    HEATMAP_ABOVE = 20

    def __init__(self, title, names, traces, viewer=None, image=None, axis=0, xlabel='frame', parent=None):
        from qtpy.QtWidgets import QComboBox

        super().__init__(parent, Qt.Window)
        style.apply(self)
        self.setWindowTitle(title)
        self.resize(720, 460)
        self.names = [str(n) for n in names]
        self.raw = np.asarray(traces, dtype=np.float64)
        self.viewer, self.image, self.axis, self.xlabel = viewer, image, axis, xlabel

        self.figure = Figure(figsize=(7, 4), layout='constrained')
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.dff = QCheckBox('ΔF/F (F0 = median)')
        self.dff.toggled.connect(lambda _: self.draw())
        self.mode = QComboBox()
        self.mode.addItems(['Stacked', 'Overlay', 'Heatmap'])
        self.mode.setCurrentText('Heatmap' if len(self.names) > self.HEATMAP_ABOVE else 'Stacked')
        self.mode.currentTextChanged.connect(lambda _: self.draw())
        bar = QHBoxLayout()
        bar.addWidget(self.dff)
        bar.addWidget(self.mode)
        bar.addStretch()
        for text, slot in (('Copy', self.copy), ('Save .npy…', self.save_npy), ('Save CSV…', self.save_csv)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            bar.addWidget(b)
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
        layout.addLayout(bar)

        self._cursor_artists = []
        if viewer is not None:
            viewer.dims.events.current_step.connect(self.on_step)
        self.draw()

    @property
    def traces(self) -> np.ndarray:
        return ops.dff(self.raw) if self.dff.isChecked() else self.raw

    def draw(self) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot()
        data = self.traces
        n, t = data.shape
        x = np.arange(t)
        mode = self.mode.currentText()
        ylabel = 'ΔF/F' if self.dff.isChecked() else 'mean intensity'
        if mode == 'Heatmap':
            im = ax.imshow(data, aspect='auto', interpolation='nearest', cmap='magma',
                           extent=(-0.5, t - 0.5, n - 0.5, -0.5))
            self.figure.colorbar(im, ax=ax, label=ylabel, shrink=0.8)
            ax.set_ylabel('ROI')
            if n <= 40:
                ax.set_yticks(range(n), self.names, fontsize=7)
        elif mode == 'Overlay':
            for name, tr in zip(self.names, data):
                ax.plot(x, tr, lw=0.8, label=name)
            ax.set_ylabel(ylabel)
            if n <= 12:
                ax.legend(fontsize=7, frameon=False, loc='upper right')
        else:
            span = np.nanpercentile(data, 99.5, axis=1) - np.nanpercentile(data, 0.5, axis=1)
            step = 1.1 * np.nanmax(span) if np.isfinite(span).any() and np.nanmax(span) > 0 else 1.0
            for i, (name, tr) in enumerate(zip(self.names, data)):
                offset = -i * step
                ax.plot(x, tr - np.nanmedian(tr) + offset, lw=0.8, color=f'C{i % 10}')
                ax.text(-0.01, offset, name, transform=ax.get_yaxis_transform(), ha='right', va='center',
                        fontsize=7, color=f'C{i % 10}')
            ax.set_yticks([])
            ax.spines[['left', 'top', 'right']].set_visible(False)
            ax.set_title(f'{ylabel}, one trace per ROI (offset for clarity)', fontsize=9)
        ax.set_xlabel(self.xlabel)
        ax.set_xlim(-0.5, t - 0.5)
        self.ax = ax
        self._cursor_artists = [ax.axvline(0, color='tab:red', lw=0.8)]
        self.on_step()

    def on_step(self, event=None) -> None:
        if self.viewer is None or self.image is None or not self._cursor_artists:
            self.canvas.draw_idle()
            return
        try:
            index = roi_mod.current_index(self.viewer, self.image)
        except Exception:
            return
        self._cursor_artists[0].set_xdata([index[self.axis]] * 2)
        self.canvas.draw_idle()

    def table_text(self, sep='\t') -> str:
        lines = [sep.join(['frame'] + self.names)]
        for t, row in enumerate(self.traces.T):
            lines.append(sep.join([str(t)] + [f'{v:.6g}' for v in row]))
        return '\n'.join(lines)

    def copy(self) -> None:
        QApplication.clipboard().setText(self.table_text())

    def save_npy(self, path: str | None = None) -> None:
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, 'Save traces (cells x frames)', 'traces.npy', 'NumPy (*.npy)')
            if not path:
                return
        np.save(path, self.traces)
        names = str(path)[:-4] + '_names.txt' if str(path).endswith('.npy') else str(path) + '_names.txt'
        with open(names, 'w') as f:
            f.write('\n'.join(self.names) + '\n')

    def save_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, 'Save traces', 'traces.csv', 'CSV (*.csv)')
        if path:
            with open(path, 'w') as f:
                f.write(self.table_text(','))

    def closeEvent(self, event):
        if self.viewer is not None:
            try:
                self.viewer.dims.events.current_step.disconnect(self.on_step)
            except Exception:
                pass
        super().closeEvent(event)


class HistogramWindow(QWidget):
    """Analyze > Histogram: counts plus ImageJ's summary statistics."""

    def __init__(self, title: str, stats: dict, parent=None):
        super().__init__(parent, Qt.Window)
        style.apply(self)
        self.setWindowTitle(title)
        self.resize(460, 340)
        self.stats = stats
        figure = Figure(figsize=(4.6, 2.4), layout='constrained')
        canvas = FigureCanvasQTAgg(figure)
        ax = figure.add_subplot()
        edges, counts = stats['edges'], stats['counts']
        ax.bar((edges[:-1] + edges[1:]) / 2, counts, width=edges[1] - edges[0], color='#333333', linewidth=0)
        ax.set_xlabel('value')
        ax.set_ylabel('count')
        from qtpy.QtWidgets import QLabel

        text = QLabel(
            f"N: {stats['n']}    Mean: {stats['mean']:.4g}    StdDev: {stats['std']:.4g}\n"
            f"Min: {stats['min']:.4g}    Max: {stats['max']:.4g}    Mode: {stats['mode']:.4g}"
        )
        copy = QPushButton('Copy list')
        copy.clicked.connect(self.copy)
        bar = QHBoxLayout()
        bar.addWidget(text, 1)
        bar.addWidget(copy)
        layout = QVBoxLayout(self)
        layout.addWidget(canvas)
        layout.addLayout(bar)

    def copy(self) -> None:
        edges, counts = self.stats['edges'], self.stats['counts']
        QApplication.clipboard().setText(
            '\n'.join(['bin start\tcount'] + [f'{e:.6g}\t{c}' for e, c in zip(edges[:-1], counts)]))
