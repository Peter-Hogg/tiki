"""Image > Adjust > Threshold: ImageJ's dialog, with napari Labels as output.

The preview is a 2D Labels layer laid over the image (it follows the slider
and the current slice). Apply adds a full-size Labels layer to the same
window, either a 0/1 mask or one id per connected object.
"""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from napari.layers import Labels
from napari.utils.colormaps import DirectLabelColormap
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from . import axes as axes_mod
from . import ops, style
from . import roi as roi_mod

PREVIEW_NAME = 'threshold preview'
COLORS = {'Red': 'red', 'Green': 'lime', 'Blue': 'dodgerblue', 'Yellow': 'yellow', 'Cyan': 'cyan', 'Magenta': 'magenta'}
STEPS = 1000  # slider resolution
STACK_SAMPLE_PLANES = 32


class ThresholdWindow(QWidget):
    def __init__(self, tiki):
        super().__init__(None, Qt.Window)
        style.apply(self)
        self.tiki = tiki
        self.viewer = None
        self.image = None
        self.preview: Labels | None = None
        self.lo_val = self.hi_val = 0.0
        self.vmin, self.vmax = 0.0, 1.0
        self._updating = False
        self.setWindowTitle('Threshold')

        self.figure = Figure(figsize=(3.4, 0.9), dpi=100)
        self.figure.subplots_adjust(0.01, 0.02, 0.99, 0.98)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setFixedHeight(90)
        self.ax = self.figure.add_subplot()
        self.ax.set_axis_off()
        self.percent = QLabel('')

        self.lo_slider, self.hi_slider = QSlider(Qt.Horizontal), QSlider(Qt.Horizontal)
        self.lo_spin, self.hi_spin = QDoubleSpinBox(), QDoubleSpinBox()
        grid = QGridLayout()
        for row, (slider, spin) in enumerate(((self.lo_slider, self.lo_spin), (self.hi_slider, self.hi_spin))):
            slider.setRange(0, STEPS)
            spin.setKeyboardTracking(False)
            spin.setMinimumWidth(90)
            grid.addWidget(slider, row, 0)
            grid.addWidget(spin, row, 1)
        self.lo_slider.valueChanged.connect(lambda v: self._from_slider('lo', v))
        self.hi_slider.valueChanged.connect(lambda v: self._from_slider('hi', v))
        self.lo_spin.valueChanged.connect(lambda v: self._set_range(v, self.hi_val))
        self.hi_spin.valueChanged.connect(lambda v: self._set_range(self.lo_val, v))

        self.method = QComboBox()
        self.method.addItems(ops.THRESHOLD_METHODS)
        self.method.currentTextChanged.connect(lambda _: self.auto())
        self.color = QComboBox()
        self.color.addItems(list(COLORS))
        self.color.currentTextChanged.connect(lambda _: self._recolor())
        combos = QHBoxLayout()
        combos.addWidget(self.method)
        combos.addWidget(self.color)

        self.dark = QCheckBox('Dark background', checked=True)
        self.dark.toggled.connect(lambda _: self.auto())
        self.stack_hist = QCheckBox('Stack histogram')
        self.stack_hist.toggled.connect(lambda _: self.refresh_histogram())
        self.split = QCheckBox('Split objects (labels)', checked=True)
        self.split.setToolTip('Apply gives each connected object its own label id, like Analyze Particles')
        self.whole_stack = QCheckBox('Apply to whole stack', checked=True)
        self.min_size = QSpinBox()
        self.min_size.setRange(0, 10**7)
        self.min_size.setSuffix(' px')
        self.min_size.setToolTip('Objects smaller than this are removed when applying')
        checks = QGridLayout()
        checks.addWidget(self.dark, 0, 0)
        checks.addWidget(self.stack_hist, 0, 1)
        checks.addWidget(self.split, 1, 0)
        checks.addWidget(self.whole_stack, 1, 1)
        min_row = QHBoxLayout()
        min_row.addWidget(QLabel('Min object size:'))
        min_row.addWidget(self.min_size)
        min_row.addStretch()

        buttons = QHBoxLayout()
        buttons.addStretch()
        for text, slot in (('Auto', self.auto), ('Apply', self.apply), ('Reset', self.reset)):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, s=slot, t=text: self.tiki._run(f'Threshold {t}', s))
            buttons.addWidget(b)
        buttons.addStretch()

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
        layout.addWidget(self.percent)
        layout.addLayout(grid)
        layout.addLayout(combos)
        layout.addLayout(checks)
        layout.addLayout(min_row)
        layout.addLayout(buttons)
        layout.setSpacing(6)
        self.setFixedWidth(380)
        self.setFixedHeight(self.sizeHint().height())

        self._redraw = QTimer(self, singleShot=True, interval=30)
        self._redraw.timeout.connect(self.update_preview)

    # ------------------------------------------------------------ target

    def set_target(self, viewer) -> None:
        """Follow the current image window, as ImageJ's dialog does."""
        image = roi_mod.target_image(viewer) if viewer is not None else None
        if viewer is self.viewer and image is self.image:
            return
        self._detach()
        self.viewer, self.image = viewer, image
        if image is None:
            self.setWindowTitle('Threshold (no image)')
            return
        self.setWindowTitle(f'Threshold — {image.name}')
        viewer.dims.events.current_step.connect(self._on_step)
        dtype = np.asarray(image.data[(0,) * (image.data.ndim - 2)][:1, :1]).dtype
        decimals = 0 if np.issubdtype(dtype, np.integer) else 4
        for spin in (self.lo_spin, self.hi_spin):
            spin.setDecimals(decimals)
        self.refresh_histogram()
        self.auto()

    def _detach(self) -> None:
        self._remove_preview()
        if self.viewer is not None:
            try:
                self.viewer.dims.events.current_step.disconnect(self._on_step)
            except Exception:
                pass
        self.viewer = self.image = None

    def _on_step(self, event=None) -> None:
        if not self.stack_hist.isChecked():
            self.refresh_histogram(keep_range=True)
        self._redraw.start()

    # --------------------------------------------------------- histogram

    def _plane(self) -> np.ndarray:
        return roi_mod.current_plane(self.viewer, self.image)

    def _hist_values(self) -> np.ndarray:
        if not self.stack_hist.isChecked() or self.image.ndim < 3:
            return self._plane()
        data = self.image.data
        lead = data.shape[: self.image.ndim - 2]
        n = int(np.prod(lead))
        picks = np.unique(np.linspace(0, n - 1, min(n, STACK_SAMPLE_PLANES)).astype(int))
        planes = [np.asarray(data[np.unravel_index(i, lead)]) for i in picks]
        if self.image.rgb:
            planes = [p[..., :3].mean(-1) for p in planes]
        return np.stack(planes)

    def refresh_histogram(self, keep_range: bool = False) -> None:
        if self.image is None:
            return
        values = np.asarray(self._hist_values(), dtype=np.float64).ravel()
        values = values[np.isfinite(values)]
        if values.size == 0:
            return
        self.vmin, self.vmax = float(values.min()), float(values.max())
        if self.vmax == self.vmin:
            self.vmax = self.vmin + 1
        self.counts, self.edges = np.histogram(values, bins=256, range=(self.vmin, self.vmax))
        self._values = values
        for spin in (self.lo_spin, self.hi_spin):
            spin.blockSignals(True)
            spin.setRange(min(self.vmin, -1e12), max(self.vmax, 1e12))
            spin.blockSignals(False)
        if keep_range:
            self._sync_widgets()
        self._draw_histogram()

    def _draw_histogram(self) -> None:
        self.ax.clear()
        self.ax.set_axis_off()
        centers = (self.edges[:-1] + self.edges[1:]) / 2
        self.ax.bar(centers, self.counts, width=self.edges[1] - self.edges[0], color='#333333', linewidth=0)
        self.ax.axvspan(self.lo_val, self.hi_val, facecolor=COLORS[self.color.currentText()], alpha=0.25,
                        edgecolor='red', linewidth=1)
        self.ax.set_xlim(self.vmin, self.vmax)
        self.ax.set_ylim(0, max(1, np.percentile(self.counts[self.counts > 0], 99) * 1.1) if self.counts.any() else 1)
        self.canvas.draw_idle()

    # ------------------------------------------------------------- range

    def _to_slider(self, value: float) -> int:
        return int(round((value - self.vmin) / (self.vmax - self.vmin) * STEPS))

    def _from_slider(self, which: str, step: int) -> None:
        if self._updating:
            return
        value = self.vmin + step / STEPS * (self.vmax - self.vmin)
        if which == 'lo':
            self._set_range(value, max(value, self.hi_val))
        else:
            self._set_range(min(value, self.lo_val), value)

    def _set_range(self, lo: float, hi: float) -> None:
        if self._updating or self.image is None:
            return
        self.lo_val, self.hi_val = float(min(lo, hi)), float(max(lo, hi))
        self._sync_widgets()
        self._draw_histogram()
        self._redraw.start()

    def _sync_widgets(self) -> None:
        self._updating = True
        try:
            self.lo_slider.setValue(self._to_slider(self.lo_val))
            self.hi_slider.setValue(self._to_slider(self.hi_val))
            self.lo_spin.setValue(self.lo_val)
            self.hi_spin.setValue(self.hi_val)
        finally:
            self._updating = False

    def auto(self) -> None:
        if self.image is None:
            return
        try:
            t = ops.auto_threshold(self._values, self.method.currentText())
        except Exception as exc:  # e.g. Minimum on a unimodal histogram
            self.tiki.status(f'Threshold {self.method.currentText()}: {exc}')
            return
        if self.dark.isChecked():
            self._set_range(t, self.vmax)
        else:
            self._set_range(self.vmin, t)

    def reset(self) -> None:
        self._remove_preview()
        if self.image is not None:
            self.lo_val, self.hi_val = self.vmin, self.vmax
            self._sync_widgets()
            self._draw_histogram()
            self.percent.setText('')

    # ----------------------------------------------------------- preview

    def _colormap(self, value: int = 1) -> DirectLabelColormap:
        return DirectLabelColormap(color_dict={None: 'transparent', 0: 'transparent', value: COLORS[self.color.currentText()]})

    def _recolor(self) -> None:
        if self.preview is not None:
            self.preview.colormap = self._colormap()
        if self.image is not None:
            self._draw_histogram()

    def update_preview(self) -> None:
        if self.image is None or self.viewer is None:
            return
        mask = ops.threshold_mask(self._plane(), self.lo_val, self.hi_val)
        self.percent.setText(f'{100 * mask.mean():.2f} %')
        data = mask.astype(np.uint8)
        if self.preview is None or self.preview not in self.viewer.layers:
            selected = self.viewer.layers.selection.active
            self.preview = Labels(
                data,
                name=PREVIEW_NAME,
                opacity=0.6,
                colormap=self._colormap(),
                **axes_mod.geometry(self.image, ndim=2),
            )
            self.viewer.add_layer(self.preview)
            # keep ROI drawing working: give selection back to what had it
            if selected is not None:
                self.viewer.layers.selection.active = selected
            roi = roi_mod.roi_layer(self.viewer, create=False)
            if roi is not None:  # ROIs stay on top of the preview
                self.viewer.layers.move(self.viewer.layers.index(roi), len(self.viewer.layers))
        else:
            self.preview.data = data

    def _remove_preview(self) -> None:
        if self.preview is not None and self.viewer is not None and self.preview in self.viewer.layers:
            self.viewer.layers.remove(self.preview)
        self.preview = None

    # -------------------------------------------------------------- apply

    def apply(self) -> Labels | None:
        from .app import TikiError

        if self.image is None:
            raise TikiError('no image open')
        image, viewer = self.image, self.viewer
        split, min_size = self.split.isChecked(), self.min_size.value()
        if self.whole_stack.isChecked() or image.ndim == 2:
            data = image.data
            if image.rgb:
                data = np.asarray(data)[..., :3].mean(-1)
            out = ops.threshold_stack(data, self.lo_val, self.hi_val, split, min_size)
            geometry = axes_mod.geometry(image)
            suffix = ''
        else:
            mask = ops.threshold_mask(self._plane(), self.lo_val, self.hi_val)
            out = ops.label_objects(mask, min_size) if split else (
                (ops.label_objects(mask, min_size) > 0) if min_size > 1 else mask).astype(np.uint8)
            geometry = axes_mod.geometry(image, ndim=2)
            idx = roi_mod.current_index(viewer, image)[:-2]
            suffix = ' slice ' + ','.join(str(i + 1) for i in idx)
        self._remove_preview()
        kind = 'labels' if split else 'mask'
        layer = Labels(out, name=f'{image.name} {kind}{suffix}', opacity=0.6, **geometry)
        if not split:
            layer.colormap = self._colormap()
        viewer.add_layer(layer)
        roi = roi_mod.roi_layer(viewer, create=False)
        if roi is not None:
            viewer.layers.move(viewer.layers.index(roi), len(viewer.layers))
        # Hand the result straight to the user for touch-up painting: select it
        # and open napari's label controls (brush, eraser, fill, brush size).
        self.tiki.set_tool('hand')
        viewer.layers.selection.active = layer
        self.tiki.windows.show_layer_controls(viewer)
        n = int(out.max()) if split else None
        self.tiki.status(
            f'Threshold {self.lo_val:.4g}–{self.hi_val:.4g}: '
            + (f'{n} objects' if split else f'{100 * (out > 0).mean():.2f} % of pixels')
            + f' → layer "{layer.name}"'
        )
        return layer

    def closeEvent(self, event):
        self._detach()
        super().closeEvent(event)
