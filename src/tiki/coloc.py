"""Analyze > Colocalization: Pearson, Spearman, Manders, Costes thresholds & cytofluorogram."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import ops, process, style
from . import roi as roi_mod

_MAX_PLOT_POINTS = 250_000


class ColocalizationWindow(QWidget):
    """Interactive cytofluorogram scatter/density plot and colocalization statistics table."""

    def __init__(
        self,
        tiki,
        name1: str,
        name2: str,
        ch1_vals: np.ndarray,
        ch2_vals: np.ndarray,
        stats: dict,
        scope_label: str = '',
    ):
        super().__init__(None, Qt.Window)
        style.apply(self)
        self.tiki = tiki
        self.name1 = name1
        self.name2 = name2
        self.stats = dict(stats)
        self.scope_label = scope_label
        self.setWindowTitle(f'Colocalization — {name1} vs {name2}')
        self.resize(560, 680)

        self.figure = Figure(figsize=(5.0, 4.2), dpi=100, constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot()
        self._draw_cytofluorogram(ch1_vals, ch2_vals)

        rows = [
            ('Pearson r', f'{stats["Pearson_r"]:.4f}'),
            ('Spearman ρ', f'{stats["Spearman_rho"]:.4f}'),
            ('Overlap R', f'{stats["Overlap_R"]:.4f}'),
            ('Manders M1 (ch1 in ch2>0)', f'{stats["Manders_M1"]:.4f}'),
            ('Manders M2 (ch2 in ch1>0)', f'{stats["Manders_M2"]:.4f}'),
            ('Threshold T1 / T2', f'{stats["Threshold_1"]:.4g} / {stats["Threshold_2"]:.4g}'),
            ('Thresholded M1 (tM1)', f'{stats["tM1"]:.4f}'),
            ('Thresholded M2 (tM2)', f'{stats["tM2"]:.4f}'),
            ('Pearson r above thresholds', f'{stats["Pearson_above"]:.4f}'),
            ('Colocalized fraction', f'{stats["Coloc_fraction"]:.4f}'),
            ('Pairs analyzed (N)', f'{stats["N"]}'),
        ]
        self.table = QTableWidget(len(rows), 2)
        self.table.setHorizontalHeaderLabels(['Metric', 'Value'])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        for r, (k, v) in enumerate(rows):
            self.table.setItem(r, 0, QTableWidgetItem(k))
            item = QTableWidgetItem(v)
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.table.setItem(r, 1, item)
        self.table.resizeColumnToContents(0)

        bar = QHBoxLayout()
        for text, slot in (
            ('Add to Results', self.add_to_results),
            ('Copy', self.copy),
            ('Save Plot…', self.save_plot),
        ):
            b = QPushButton(text)
            b.clicked.connect(slot)
            bar.addWidget(b)
        bar.addStretch()

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas, 3)
        layout.addWidget(self.table, 2)
        layout.addLayout(bar)

    def _draw_cytofluorogram(self, x: np.ndarray, y: np.ndarray) -> None:
        ax = self.ax
        ax.clear()
        finite = np.isfinite(x) & np.isfinite(y)
        px = np.asarray(x[finite], dtype=np.float64)
        py = np.asarray(y[finite], dtype=np.float64)
        if px.size > _MAX_PLOT_POINTS:
            step = max(1, px.size // _MAX_PLOT_POINTS)
            px = px[::step]
            py = py[::step]

        if px.size >= 2 and float(px.max()) > float(px.min()) and float(py.max()) > float(py.min()):
            counts, xe, ye = np.histogram2d(px, py, bins=96)
            if (counts > 0).any():
                vmax = max(2.0, float(counts.max()))
                norm = LogNorm(vmin=1.0, vmax=vmax)
                mesh = ax.pcolormesh(xe, ye, counts.T, cmap='viridis', norm=norm, shading='auto')
                self.figure.colorbar(mesh, ax=ax, label='pixel count', shrink=0.85)
            slope = float(self.stats.get('Slope', 0.0))
            intercept = float(self.stats.get('Intercept', 0.0))
            if np.isfinite(slope) and slope > 0:
                xs = np.array([float(px.min()), float(px.max())])
                ys = slope * xs + intercept
                ax.plot(xs, ys, '--', color='#ff9800', lw=1.0, label='Costes fit')
        elif px.size:
            ax.scatter(px, py, s=8, alpha=0.6)

        t1 = float(self.stats.get('Threshold_1', 0.0))
        t2 = float(self.stats.get('Threshold_2', 0.0))
        if np.isfinite(t1) and t1 != 0.0:
            ax.axvline(t1, color='crimson', ls=':', lw=1.0)
        if np.isfinite(t2) and t2 != 0.0:
            ax.axhline(t2, color='crimson', ls=':', lw=1.0)
        ax.set_xlabel(self.name1)
        ax.set_ylabel(self.name2)
        title = f'Cytofluorogram (r = {self.stats["Pearson_r"]:.3f}'
        if self.scope_label:
            title += f', {self.scope_label}'
        title += ')'
        ax.set_title(title, fontsize=9)
        self.canvas.draw_idle()

    def add_to_results(self) -> None:
        self.tiki._sync_result_units({})
        row = {
            'Ch1': self.name1,
            'Ch2': self.name2,
            'Scope': self.scope_label or 'plane',
            'Pearson_r': self.stats['Pearson_r'],
            'Spearman_rho': self.stats['Spearman_rho'],
            'Overlap_R': self.stats['Overlap_R'],
            'Manders_M1': self.stats['Manders_M1'],
            'Manders_M2': self.stats['Manders_M2'],
            'tM1': self.stats['tM1'],
            'tM2': self.stats['tM2'],
            'T1': self.stats['Threshold_1'],
            'T2': self.stats['Threshold_2'],
            'Coloc_fraction': self.stats['Coloc_fraction'],
            'N': self.stats['N'],
        }
        self.tiki.results.add_row(row)
        self.tiki.show_results()

    def copy(self) -> None:
        lines = [f'{self.table.item(r, 0).text()}\t{self.table.item(r, 1).text()}' for r in range(self.table.rowCount())]
        QApplication.clipboard().setText('\n'.join(lines))

    def save_plot(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, 'Save Colocalization Plot', 'colocalization.png', 'PNG (*.png);;SVG (*.svg)')
        if path:
            self.figure.savefig(path, dpi=200)


def colocalization_dialog(tiki):
    """``Analyze ▸ Colocalization…``: compare any two open greyscale layers."""
    all_imgs = [
        (label, v, ly)
        for label, v, ly in process._all_images(tiki)
        if not ly.rgb
    ]
    if len(all_imgs) < 2:
        process._error(
            'Colocalization needs at least two open greyscale layers (split channels or open two images)'
        )

    dlg, form = process._form(tiki, 'Colocalization')
    ch1_box = QComboBox()
    ch2_box = QComboBox()
    for idx, (label, v, ly) in enumerate(all_imgs):
        ch1_box.addItem(label, (v, ly))
        ch2_box.addItem(label, (v, ly))

    # Prefer two distinct layers in the active window
    active_v = tiki.windows.active
    active_indices = [i for i, (_, v, _) in enumerate(all_imgs) if v is active_v]
    if len(active_indices) >= 2:
        ch1_box.setCurrentIndex(active_indices[0])
        ch2_box.setCurrentIndex(active_indices[1])
    else:
        ch1_box.setCurrentIndex(0)
        ch2_box.setCurrentIndex(1)

    scope_box = QComboBox()
    scope_box.addItem('Current 2D plane', 'plane')
    scope_box.addItem('Whole volume / stack', 'whole')
    roi_box = QCheckBox('Restrict to active ROI', checked=True)
    thresh_box = QComboBox()
    thresh_box.addItem('Costes automatic', 'costes')
    thresh_box.addItem('Manual thresholds', 'manual')
    thresh_box.addItem('None (unthresholded)', 'none')
    t1_spin = QDoubleSpinBox(decimals=3, minimum=-1e12, maximum=1e12, value=0.0)
    t2_spin = QDoubleSpinBox(decimals=3, minimum=-1e12, maximum=1e12, value=0.0)
    note = QLabel('')
    note.setStyleSheet('color: #b00020;')

    form.addRow('Channel 1:', ch1_box)
    form.addRow('Channel 2:', ch2_box)
    form.addRow('Scope:', scope_box)
    form.addRow(roi_box)
    form.addRow('Thresholding:', thresh_box)
    form.addRow('Manual T1:', t1_spin)
    form.addRow('Manual T2:', t2_spin)
    form.addRow(note)
    buttons = process._buttons(dlg, form)

    state = {'has_roi': False, 'has_stack': False}

    def _sync(_=None):
        d1 = ch1_box.currentData()
        d2 = ch2_box.currentData()
        if not d1 or not d2:
            buttons.button(buttons.Ok).setEnabled(False)
            return
        v1, l1 = d1
        v2, l2 = d2
        is_manual = thresh_box.currentData() == 'manual'
        t1_spin.setVisible(is_manual)
        t2_spin.setVisible(is_manual)
        if form.labelForField(t1_spin):
            form.labelForField(t1_spin).setVisible(is_manual)
        if form.labelForField(t2_spin):
            form.labelForField(t2_spin).setVisible(is_manual)

        cur_roi = tiki.current_roi(v1)
        state['has_roi'] = cur_roi is not None and cur_roi[1] not in ('line', 'path')
        roi_box.setVisible(state['has_roi'])
        if not state['has_roi']:
            roi_box.setChecked(False)

        same_stack = l1.data.shape == l2.data.shape and l1.data.ndim > 2
        same_plane = l1.data.shape[-2:] == l2.data.shape[-2:]
        state['has_stack'] = same_stack
        scope_box.setEnabled(same_stack)
        if not same_stack:
            scope_box.setCurrentIndex(0)

        if l1 is l2:
            note.setText('Choose two different channels')
            buttons.button(buttons.Ok).setEnabled(False)
        elif not same_plane:
            note.setText(f'Spatial shapes differ: {l1.data.shape[-2:]} vs {l2.data.shape[-2:]}')
            buttons.button(buttons.Ok).setEnabled(False)
        else:
            note.setText('')
            buttons.button(buttons.Ok).setEnabled(True)

    ch1_box.currentIndexChanged.connect(_sync)
    ch2_box.currentIndexChanged.connect(_sync)
    thresh_box.currentIndexChanged.connect(_sync)
    _sync()

    if dlg.exec_() != QDialog.Accepted:
        return None
    v1, l1 = ch1_box.currentData()
    v2, l2 = ch2_box.currentData()
    use_stack = state['has_stack'] and scope_box.currentData() == 'whole'
    if use_stack:
        arr1 = np.asarray(l1.data)
        arr2 = np.asarray(l2.data)
        scope_label = 'whole stack'
    else:
        arr1 = np.asarray(roi_mod.current_plane(v1, l1))
        arr2 = np.asarray(roi_mod.current_plane(v2, l2))
        scope_label = 'current plane'

    mask = None
    if state['has_roi'] and roi_box.isChecked():
        cur = tiki.current_roi(v1)
        if cur is not None and cur[1] not in ('line', 'path'):
            mask = roi_mod.shape_mask(*cur, arr1.shape[-2:])
            scope_label += ' + ROI'

    mode = thresh_box.currentData()
    t1 = t1_spin.value() if mode == 'manual' else None
    t2 = t2_spin.value() if mode == 'manual' else None
    costes = mode == 'costes'
    stats = ops.colocalization_stats(arr1, arr2, mask=mask, thresh1=t1, thresh2=t2, costes=costes)

    if mask is not None:
        plot_mask = np.broadcast_to(mask, arr1.shape) if mask.ndim < arr1.ndim else mask
        plot_x, plot_y = arr1[plot_mask], arr2[plot_mask]
    else:
        plot_x, plot_y = arr1.ravel(), arr2.ravel()
    win = ColocalizationWindow(tiki, l1.name, l2.name, plot_x, plot_y, stats, scope_label=scope_label)
    tiki.keep_window(win)
    win.show()
    win.raise_()
    tiki.record(
        '# Colocalization: select two layers with tiki.image(...); stats via tiki.ops.colocalization_stats(...)'
    )
    tiki.status(f'Colocalization ({l1.name} vs {l2.name}): r = {stats["Pearson_r"]:.4f}, M1 = {stats["Manders_M1"]:.4f}')
    return win
