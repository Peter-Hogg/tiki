"""Floating TIKI windows: Results table and live Z-axis profile plot."""

from __future__ import annotations

import csv

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import axes as axes_mod
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
        self.column_units: dict[str, str] = {}
        self._summary_win = None
        self.table = QTableWidget(0, 0)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        buttons = QHBoxLayout()
        for text, slot in (
            ('Copy', self.copy),
            ('Save As…', self.save_as),
            ('Summarize', self.summarize),
            ('Clear', self.clear),
        ):
            b = QPushButton(text)
            b.clicked.connect(lambda _checked=False, s=slot: s())
            buttons.addWidget(b)
        buttons.addStretch()
        layout = QVBoxLayout(self)
        layout.addWidget(self.table)
        layout.addLayout(buttons)

    def _header_labels(self) -> list[str]:
        return [f'{k} ({self.column_units[k]})' if self.column_units.get(k) else k for k in self.columns]

    def set_units(self, units: dict[str, str], replace: bool = False) -> None:
        clean = {k: u for k, u in units.items() if u}
        if replace:
            self.column_units = clean
        else:
            self.column_units.update(clean)
        if self.columns:
            self.table.setHorizontalHeaderLabels(self._header_labels())

    def clear_units(self) -> None:
        self.column_units.clear()
        if self.columns:
            self.table.setHorizontalHeaderLabels(self._header_labels())

    def add_row(self, row: dict) -> None:
        self.add_rows([row])

    def add_rows(self, rows: list[dict]) -> None:
        if not rows:
            return
        new_cols = False
        for row in rows:
            for key in row:
                if key not in self.columns:
                    self.columns.append(key)
                    new_cols = True
        self.table.setUpdatesEnabled(False)
        try:
            if new_cols:
                prev_col_count = self.table.columnCount()
                self.table.setColumnCount(len(self.columns))
                self.table.setHorizontalHeaderLabels(self._header_labels())
                for r, old in enumerate(self.rows):
                    for c in range(prev_col_count, len(self.columns)):
                        self._set_cell(r, c, old.get(self.columns[c], ''))
            start_r = self.table.rowCount()
            self.table.setRowCount(start_r + len(rows))
            for offset, row in enumerate(rows):
                self.rows.append(row)
                r = start_r + offset
                for c, key in enumerate(self.columns):
                    self._set_cell(r, c, row.get(key, ''))
        finally:
            self.table.setUpdatesEnabled(True)
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
        lines = [sep.join([' '] + self._header_labels())]
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
            writer = csv.writer(f)
            writer.writerow(self._header_labels())
            for row in self.rows:
                writer.writerow([row.get(k, '') for k in self.columns])

    def summary_rows(self) -> tuple[list[str], list[dict]]:
        """Compute Mean, SD, Min, Max across numeric columns of ``self.rows``."""
        num_cols: list[str] = []
        col_vals: dict[str, np.ndarray] = {}
        for k in self.columns:
            vals = []
            for row in self.rows:
                v = row.get(k)
                if isinstance(v, (int, float, np.number)) and not isinstance(v, bool) and np.isfinite(float(v)):
                    vals.append(float(v))
            if vals:
                num_cols.append(k)
                col_vals[k] = np.asarray(vals, dtype=np.float64)
        if not num_cols:
            return [], []
        specs = (
            ('Mean', lambda a: float(np.mean(a))),
            ('SD', lambda a: float(np.std(a, ddof=1)) if a.size > 1 else 0.0),
            ('Min', lambda a: float(np.min(a))),
            ('Max', lambda a: float(np.max(a))),
        )
        out = []
        for stat, fn in specs:
            r = {'Stat': stat}
            for k in num_cols:
                r[k] = fn(col_vals[k])
            out.append(r)
        return num_cols, out

    def summarize(self, show: bool = True) -> list[dict]:
        """Compute summary statistics across Results rows and optionally open Summary window."""
        num_cols, summary = self.summary_rows()
        if not summary:
            if show:
                QMessageBox.information(self, 'Summary', 'No numeric measurements in Results to summarize.')
            return []
        if show:
            if self._summary_win is not None:
                self._summary_win.close()
            headers = ['Stat'] + [
                f'{k} ({self.column_units[k]})' if self.column_units.get(k) else k for k in num_cols
            ]
            win = QWidget(self, Qt.Window)
            style.apply(win)
            win.setWindowTitle(f'Summary of Results (N = {len(self.rows)})')
            win.resize(620, 200)
            tbl = QTableWidget(len(summary), len(headers))
            tbl.setEditTriggers(QTableWidget.NoEditTriggers)
            tbl.setAlternatingRowColors(True)
            tbl.setHorizontalHeaderLabels(headers)
            for r_i, srow in enumerate(summary):
                for c_i, key in enumerate(['Stat'] + num_cols):
                    val = srow[key]
                    text = f'{val:.4g}' if isinstance(val, float) else str(val)
                    item = QTableWidgetItem(text)
                    if isinstance(val, float):
                        item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    tbl.setItem(r_i, c_i, item)
            tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            copy_btn = QPushButton('Copy')
            copy_btn.clicked.connect(
                lambda: QApplication.clipboard().setText(
                    '\n'.join(
                        [
                            '\t'.join(headers),
                            *[
                                '\t'.join(
                                    f'{s[k]:.6g}' if isinstance(s[k], float) else str(s[k])
                                    for k in ['Stat'] + num_cols
                                )
                                for s in summary
                            ],
                        ]
                    )
                )
            )
            lay = QVBoxLayout(win)
            lay.addWidget(tbl)
            lay.addWidget(copy_btn, alignment=Qt.AlignRight)
            self._summary_win = win
            win.show()
            win.raise_()
        return summary

    def clear(self) -> None:
        self.rows.clear()
        self.columns.clear()
        self.column_units.clear()
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
        self._last_other_idx = None
        self.profile = np.array([])
        self.x = np.array([])
        ax_info = axes_mod.get(image).axes[axis]
        self.step = float(ax_info.step) if (ax_info.step and ax_info.step > 0) else 1.0
        self.unit = ax_info.unit if (ax_info.step and ax_info.unit) else None
        self.axis_label = roi_mod.axis_name(viewer, image, axis)
        self.setWindowTitle(f'{image.name} — profile')
        self.resize(560, 320)

        self.figure = Figure(figsize=(5.6, 3.0), layout='constrained')
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot()
        (self.line,) = self.ax.plot([], [], lw=1, color='k')
        self.cursor_line = self.ax.axvline(0, color='tab:red', lw=0.8)
        self.ax.set_xlabel(f'{self.axis_label} ({self.unit})' if self.unit else self.axis_label)
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
        if self.image not in self.viewer.layers:
            return
        index = roi_mod.current_index(self.viewer, self.image)
        other_idx = tuple(index[d] for d in range(len(index) - 2) if d != self.axis)
        sig = (self._roi_signature(), other_idx)
        if not force and sig == self._signature:
            return
        self._signature = sig
        self._last_other_idx = other_idx
        shape2d = roi_mod.spatial_shape(self.image)
        current = roi_mod.current_shape(self.roi)
        mask = None if current is None else roi_mod.shape_mask(*current, shape2d)
        self.profile = ops.axis_profile(self.image.data, self.axis, index, mask)
        ax_info = axes_mod.get(self.image).axes[self.axis]
        self.step = float(ax_info.step) if (ax_info.step and ax_info.step > 0) else 1.0
        self.unit = ax_info.unit if (ax_info.step and ax_info.unit) else None
        self.axis_label = roi_mod.axis_name(self.viewer, self.image, self.axis)
        self.ax.set_xlabel(f'{self.axis_label} ({self.unit})' if self.unit else self.axis_label)
        self.x = np.arange(self.profile.size) * self.step
        self.line.set_data(self.x, self.profile)
        self.ax.set_title('whole frame' if mask is None else f'{current[1]} ROI, {int(mask.sum())} px', fontsize=9)
        self.ax.relim()
        self.ax.autoscale_view()
        self.on_step()

    def on_step(self, event=None) -> None:
        if self.image not in self.viewer.layers:
            return
        index = roi_mod.current_index(self.viewer, self.image)
        other_idx = tuple(index[d] for d in range(len(index) - 2) if d != self.axis)
        if self._last_other_idx is not None and other_idx != self._last_other_idx:
            self.schedule()
        self.cursor_line.set_xdata([index[self.axis] * self.step] * 2)
        self.canvas.draw_idle()

    def copy(self) -> None:
        x_vals = self.x if len(self.x) == len(self.profile) else np.arange(self.profile.size) * self.step
        lines = [f'{x:.6g}\t{v}' if (self.unit or self.step != 1.0) else f'{int(round(x))}\t{v}'
                 for x, v in zip(x_vals, self.profile)]
        QApplication.clipboard().setText('\n'.join(lines))

    def save_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, 'Save profile', f'{self.image.name}_profile.csv', 'CSV (*.csv)')
        if path:
            x_vals = self.x if len(self.x) == len(self.profile) else np.arange(self.profile.size) * self.step
            header = f'{self.axis_label} ({self.unit}),mean' if self.unit else 'frame,mean'
            fmt = ['%.6g', '%.6g'] if (self.unit or self.step != 1.0) else ['%d', '%.6g']
            np.savetxt(path, np.c_[x_vals, self.profile], delimiter=',',
                       header=header, comments='', fmt=fmt)

    def closeEvent(self, event):
        self._timer.stop()
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


class LineProfileWindow(QWidget):
    """Analyze > Plot Profile: live intensity profile along a straight line or polyline."""

    def __init__(self, viewer, image, width: int = 1, parent=None):
        from qtpy.QtWidgets import QLabel, QSpinBox

        super().__init__(parent, Qt.Window)
        style.apply(self)
        self.viewer = viewer
        self.image = image
        self.roi = roi_mod.roi_layer(viewer, create=True)
        self.dist = np.array([])
        self.values = np.array([])
        self._signature = None
        self.setWindowTitle(f'Plot of {image.name}')
        self.resize(540, 320)

        self.figure = Figure(figsize=(5.4, 2.9), layout='constrained')
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot()
        (self.line,) = self.ax.plot([], [], lw=1, color='k')
        self.ax.set_ylabel('gray value')

        self.live = QCheckBox('Live', checked=True)
        self.live.toggled.connect(lambda on: on and self.schedule())
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 200)
        self.width_spin.setValue(max(1, int(width)))
        self.width_spin.setSuffix(' px')
        self.width_spin.setToolTip('Average intensity across this many pixels perpendicular to the line')
        self.width_spin.valueChanged.connect(lambda _: self.update_profile(force=True))

        copy = QPushButton('Copy')
        copy.clicked.connect(self.copy)
        save = QPushButton('Save CSV…')
        save.clicked.connect(self.save_csv)
        bar = QHBoxLayout()
        bar.addWidget(self.live)
        bar.addWidget(QLabel('Width:'))
        bar.addWidget(self.width_spin)
        bar.addStretch()
        bar.addWidget(copy)
        bar.addWidget(save)
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
        layout.addLayout(bar)

        self._timer = QTimer(self, singleShot=True, interval=50)
        self._timer.timeout.connect(self.update_profile)
        self._connections = [
            (self.roi.events.data, self.schedule),
            (self.roi.events.set_data, self.schedule),
            (self.roi.events.highlight, self.schedule),
            (self.viewer.dims.events.current_step, lambda _=None: self.schedule(force=True)),
        ]
        for emitter, cb in self._connections:
            emitter.connect(cb)
        self.update_profile(force=True)

    def schedule(self, event=None, force: bool = False) -> None:
        if self.live.isChecked():
            if force:
                self._signature = None
            self._timer.start()

    def _signature_now(self):
        current = roi_mod.current_shape(self.roi)
        if current is None or current[1] not in ('line', 'path'):
            return None
        verts, kind = current
        idx = roi_mod.current_index(self.viewer, self.image)[:-2]
        return kind, np.round(verts[:, -2:], 2).tobytes(), idx, self.width_spin.value()

    def update_profile(self, force: bool = False) -> None:
        from . import axes as axes_mod

        if self.image not in self.viewer.layers:
            return
        sig = self._signature_now()
        if sig is None:
            return
        if not force and sig == self._signature:
            return
        self._signature = sig
        current = roi_mod.current_shape(self.roi)
        if current is None or current[1] not in ('line', 'path'):
            return
        plane = roi_mod.current_plane(self.viewer, self.image)
        scale = (float(self.image.scale[-2]), float(self.image.scale[-1]))
        self.dist, self.values = ops.line_profile(plane, current[0], width=self.width_spin.value(), scale=scale)
        self.line.set_data(self.dist, self.values)
        info = axes_mod.get(self.image)
        y_ax = info.axes[info.index('Y')] if info.index('Y') is not None else None
        unit = y_ax.unit if (y_ax is not None and y_ax.step and y_ax.unit) else 'px'
        self.ax.set_xlabel(f'distance ({unit})')
        w = self.width_spin.value()
        self.ax.set_title(f'{current[1]} ({len(self.values)} samples, width {w} px)', fontsize=9)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw_idle()

    def copy(self) -> None:
        QApplication.clipboard().setText('\n'.join(f'{d:.6g}\t{v:.6g}' for d, v in zip(self.dist, self.values)))

    def save_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, 'Save line profile', f'{self.image.name}_line.csv', 'CSV (*.csv)')
        if path:
            np.savetxt(
                path,
                np.c_[self.dist, self.values],
                delimiter=',',
                header='distance,value',
                comments='',
                fmt='%.6g',
            )

    def closeEvent(self, event):
        self._timer.stop()
        for emitter, cb in self._connections:
            try:
                emitter.disconnect(cb)
            except Exception:
                pass
        super().closeEvent(event)


class TracesWindow(QWidget):
    """One trace per ROI or label through a stack (cells x frames).

    Stacked lines for up to ``HEATMAP_ABOVE`` traces, a heatmap beyond that.
    Exports the raw (or ΔF/F) array as .npy (cells x frames) or CSV.
    """

    HEATMAP_ABOVE = 20

    def __init__(
        self,
        title,
        names,
        traces,
        viewer=None,
        image=None,
        axis=0,
        xlabel='frame',
        step: float | None = None,
        unit: str | None = None,
        parent=None,
    ):
        from qtpy.QtWidgets import QComboBox

        super().__init__(parent, Qt.Window)
        style.apply(self)
        self.setWindowTitle(title)
        self.resize(720, 460)
        self.names = [str(n) for n in names]
        self.raw = np.asarray(traces, dtype=np.float64)
        self.viewer, self.image, self.axis, self.xlabel = viewer, image, axis, xlabel
        self.step = float(step) if (step and step > 0) else 1.0
        self.unit = unit if (step and unit) else None

        self.figure = Figure(figsize=(7, 4), layout='constrained')
        self.canvas = FigureCanvasQTAgg(self.figure)
        from qtpy.QtWidgets import QDoubleSpinBox, QLabel

        self.dff = QCheckBox('ΔF/F (F0 = median)')
        self.dff.toggled.connect(lambda _: self.draw())
        self.offset = QDoubleSpinBox(decimals=2, minimum=-1e9, maximum=1e9)
        self.offset.setPrefix('offset ')
        self.offset.setToolTip('Subtracted before ΔF/F (camera dark level). Pre-set to the data minimum '
                               'when traces go below zero, since ΔF/F needs a positive F0.')
        self.offset.setValue(ops.suggested_offset(self.raw) if np.isfinite(self.raw).any() else 0.0)
        self.offset.valueChanged.connect(lambda _: self.dff.isChecked() and self.draw())
        self.offset_note = QLabel('below zero: offset set to minimum' if self.offset.value() < 0 else '')
        self.offset_note.setStyleSheet('color: #8a5a00;')
        self.mode = QComboBox()
        self.mode.addItems(['Stacked', 'Overlay', 'Heatmap'])
        self.mode.setCurrentText('Heatmap' if len(self.names) > self.HEATMAP_ABOVE else 'Stacked')
        self.mode.currentTextChanged.connect(lambda _: self.draw())
        bar = QHBoxLayout()
        bar.addWidget(self.dff)
        bar.addWidget(self.offset)
        bar.addWidget(self.offset_note)
        bar.addWidget(self.mode)
        bar.addStretch()
        for text, slot in (('Copy', self.copy), ('Save .npy…', self.save_npy), ('Save CSV…', self.save_csv)):
            b = QPushButton(text)
            b.clicked.connect(lambda _checked=False, s=slot: s())
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
        return ops.dff(self.raw, self.offset.value()) if self.dff.isChecked() else self.raw

    def draw(self) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot()
        data = self.traces
        n, t = data.shape
        x = np.arange(t) * self.step
        x_lo, x_hi = -0.5 * self.step, (t - 0.5) * self.step
        mode = self.mode.currentText()
        ylabel = 'ΔF/F' if self.dff.isChecked() else 'mean intensity'
        if mode == 'Heatmap':
            im = ax.imshow(data, aspect='auto', interpolation='nearest', cmap='magma',
                           extent=(x_lo, x_hi, n - 0.5, -0.5))
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
        ax.set_xlabel(f'{self.xlabel} ({self.unit})' if self.unit else self.xlabel)
        ax.set_xlim(x_lo, x_hi)
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
        self._cursor_artists[0].set_xdata([index[self.axis] * self.step] * 2)
        self.canvas.draw_idle()

    def table_text(self, sep='\t') -> str:
        x_head = f'{self.xlabel} ({self.unit})' if self.unit else 'frame'
        lines = [sep.join([x_head] + self.names)]
        calibrated = self.unit is not None or self.step != 1.0
        for t, row in enumerate(self.traces.T):
            x_str = f'{t * self.step:.6g}' if calibrated else str(t)
            lines.append(sep.join([x_str] + [f'{v:.6g}' for v in row]))
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


def format_image_info(viewer, image) -> tuple[str, str]:
    """Return ``(summary_text, raw_metadata_text)`` for ``Image ▸ Show Info…``."""
    data = image.data
    plane = roi_mod.current_plane(viewer, image)
    info = axes_mod.get(image)
    nbytes = int(np.prod(data.shape)) * int(np.dtype(data.dtype).itemsize)
    backing = ''
    if isinstance(data, np.memmap) or isinstance(getattr(data, 'base', None), np.memmap):
        backing = ' (memory-mapped)'
    elif hasattr(data, 'compute'):
        backing = ' (lazy Dask)'

    lines = [
        f'Title:         {image.name}',
        f'Path:          {image.metadata.get("path", "(not saved)")}',
        f'Shape:         {tuple(int(s) for s in data.shape)}',
        f'Type:          {data.dtype}',
        f'Size:          {nbytes / 2**20:.2f} MB{backing}',
        f'Current slice: {roi_mod.current_index(viewer, image)[:-2] or "(2D)"}',
        f'Slice range:   {float(np.nanmin(plane)):.4g} .. {float(np.nanmax(plane)):.4g}',
    ]
    if not image.rgb:
        lo, hi = map(float, image.contrast_limits)
        lines.append(f'Display range: {lo:.4g} .. {hi:.4g}   (LUT: {image.colormap.name})')

    lines += ['', f'Axes & Calibration ({info.source}' + ('' if info.confident else ', not confident') + ')']
    for i, ax in enumerate(info.axes):
        step_str = f'{ax.step:.6g} {ax.unit or ""}'.strip() if ax.step else 'uncalibrated'
        lines.append(f'  [{i}] {ax.role} ({info.labels[i]}): size={ax.size}, step={step_str}')
    if info.z_assumed:
        lines.append('  Note: Z step absent; 3D display assumes isotropic voxels matching XY.')
    for note in info.notes:
        lines.append(f'  Note: {note}')

    acq = image.metadata.get('acquisition') or {}
    ch_names = image.metadata.get('channel_names') or []
    ch_list = acq.get('channels') or []
    if ch_names or ch_list:
        lines += ['', 'Channels']
        n_ch = max(len(ch_names), len(ch_list))
        for k in range(n_ch):
            entry = ch_list[k] if k < len(ch_list) and isinstance(ch_list[k], dict) else {}
            cname = (ch_names[k] if k < len(ch_names) else '') or entry.get('name') or entry.get('fluor') or f'C{k + 1}'
            extras = []
            if entry.get('fluor') and entry.get('fluor') != cname:
                extras.append(f'fluor={entry["fluor"]}')
            if entry.get('excitation_nm'):
                extras.append(f'Ex={entry["excitation_nm"]:g} nm')
            if entry.get('emission_nm'):
                extras.append(f'Em={entry["emission_nm"]:g} nm')
            if entry.get('pinhole_um'):
                extras.append(f'pinhole={entry["pinhole_um"]:g} µm')
            if entry.get('exposure_ms'):
                extras.append(f'exposure={entry["exposure_ms"]:g} ms')
            if entry.get('acquisition_mode'):
                extras.append(entry['acquisition_mode'])
            lines.append(f'  C{k + 1}: {cname}' + (f'  ({", ".join(extras)})' if extras else ''))

    acq_lines = []
    for key, label in (
        ('image_name', 'Image name'),
        ('acquisition_date', 'Acquisition date'),
        ('software', 'Software'),
        ('artist', 'Artist'),
        ('description', 'Description'),
    ):
        if acq.get(key):
            acq_lines.append(f'  {label}: {acq[key]}')
    mic = acq.get('microscope')
    if isinstance(mic, dict) and mic:
        parts = [f'{k}={v}' for k, v in mic.items()]
        acq_lines.append(f'  Microscope: {", ".join(parts)}')
    obj = acq.get('objective')
    if isinstance(obj, dict) and obj:
        parts = []
        if obj.get('model') or obj.get('name'):
            parts.append(str(obj.get('model') or obj.get('name')))
        if obj.get('magnification'):
            parts.append(f'{obj["magnification"]:g}x')
        if obj.get('na'):
            parts.append(f'NA {obj["na"]:g}')
        if obj.get('immersion'):
            parts.append(str(obj['immersion']))
        if obj.get('working_distance'):
            parts.append(f'WD {obj["working_distance"]:g}')
        acq_lines.append(f'  Objective: {", ".join(parts)}')
    det = acq.get('detector')
    if isinstance(det, dict) and det:
        acq_lines.append(f'  Detector: {", ".join(f"{k}={v}" for k, v in det.items())}')
    stage = acq.get('stage_label')
    if isinstance(stage, dict) and stage:
        acq_lines.append(f'  Stage: {", ".join(f"{k}={v}" for k, v in stage.items())}')
    for extra_key in ('scanimage', 'micromanager', 'info', 'settings'):
        sub = acq.get(extra_key)
        if isinstance(sub, dict) and sub:
            acq_lines.append(f'  {extra_key}:')
            for k, v in sub.items():
                acq_lines.append(f'    {k}: {v}')
    if acq_lines:
        lines += ['', 'Acquisition Metadata'] + acq_lines

    raw = image.metadata.get('raw_metadata') or {}
    raw_sections = [f'=== {title} ===\n{str(body).strip()}' for title, body in raw.items() if body]
    return '\n'.join(lines), '\n\n'.join(raw_sections)


class InfoWindow(QWidget):
    """Scrollable Image ▸ Show Info… window with optional raw XML/header drawer."""

    def __init__(self, viewer, image, parent=None):
        super().__init__(parent, Qt.Window)
        style.apply(self)
        self.image = image
        self.setWindowTitle(f'Info — {image.name}')
        self.resize(620, 480)

        summary_text, raw_text = format_image_info(viewer, image)
        self.summary_text = summary_text
        self.raw_text = raw_text

        mono = QFont('Monospace')
        mono.setStyleHint(QFont.TypeWriter)

        self.text = QPlainTextEdit(readOnly=True)
        self.text.setFont(mono)
        self.text.setPlainText(summary_text)

        self.raw_box = QPlainTextEdit(readOnly=True)
        self.raw_box.setFont(mono)
        self.raw_box.setPlainText(raw_text)
        self.raw_box.setVisible(False)

        self.raw_btn = QPushButton('Show Raw Metadata')
        self.raw_btn.setCheckable(True)
        self.raw_btn.setVisible(bool(raw_text))
        self.raw_btn.toggled.connect(self._toggle_raw)

        copy_btn = QPushButton('Copy')
        copy_btn.clicked.connect(self.copy)
        save_btn = QPushButton('Save .txt…')
        save_btn.clicked.connect(self.save_txt)

        bar = QHBoxLayout()
        bar.addWidget(self.raw_btn)
        bar.addStretch()
        bar.addWidget(copy_btn)
        bar.addWidget(save_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(self.text, 2)
        layout.addWidget(self.raw_box, 1)
        layout.addLayout(bar)

    def _toggle_raw(self, on: bool) -> None:
        self.raw_box.setVisible(on)
        self.raw_btn.setText('Hide Raw Metadata' if on else 'Show Raw Metadata')

    def full_text(self) -> str:
        if self.raw_text:
            return self.summary_text + '\n\n' + self.raw_text
        return self.summary_text

    def copy(self) -> None:
        QApplication.clipboard().setText(self.full_text())

    def save_txt(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, 'Save Info', f'{self.image.name}_info.txt', 'Text (*.txt)')
        if path:
            with open(path, 'w') as f:
                f.write(self.full_text() + '\n')
