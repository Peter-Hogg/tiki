"""Image > Adjust > Brightness/Contrast and Image > Lookup Tables."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from napari.layers import Image
from napari.utils.colormaps import Colormap
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from . import ops, process, style
from . import roi as roi_mod
from .layer_select import LayerCombo

STEPS = 1000
SATURATION_CYCLE = (0.35, 1.0, 2.5, 5.0, 10.0)


def _cyan_hot_colormap() -> Colormap:
    return Colormap(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.45, 0.75, 1.0],
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        name='cyan_hot',
    )


LUTS: dict[str, str | Colormap] = {
    'Grays': 'gray',
    'Inverted Grays': 'gray_r',
    'Fire': 'fire',
    'Cyan Hot': _cyan_hot_colormap(),
    'HiLo': 'HiLo',
    'Green': 'green',
    'Magenta': 'magenta',
    'Cyan': 'cyan',
    'Red': 'red',
    'Blue': 'blue',
    'Yellow': 'yellow',
    'Viridis': 'viridis',
    'Magma': 'magma',
    'Inferno': 'inferno',
    'Blue-White-Red': 'bwr',
}


def apply_lut(layer: Image, name: str) -> None:
    """Set ``layer.colormap`` by display name (e.g. 'Fire') or colormap name."""
    if name in LUTS:
        layer.colormap = LUTS[name]
        return
    low = name.lower()
    for label, cmap in LUTS.items():
        cmap_name = cmap if isinstance(cmap, str) else cmap.name
        if label.lower() == low or cmap_name.lower() == low:
            layer.colormap = cmap
            return
    layer.colormap = name


def lut_label(layer: Image) -> str:
    """Friendly LUT name for ``layer.colormap``."""
    cname = layer.colormap.name.lower()
    for label, cmap in LUTS.items():
        val = cmap if isinstance(cmap, str) else cmap.name
        if val.lower() == cname:
            return label
    return layer.colormap.name


def layer_lut_bytes(layer: Image) -> np.ndarray:
    """Return ``layer.colormap`` as a ``(3, 256)`` uint8 ImageJ LUT."""
    x = np.linspace(0.0, 1.0, 256)
    rgba = np.asarray(layer.colormap.map(x))
    rgb = np.clip(np.round(rgba[:, :3] * 255.0), 0, 255).astype(np.uint8)
    return rgb.T


class BrightnessContrastWindow(QWidget):
    """Floating ImageJ-style Brightness & Contrast dialog with live histogram."""

    def __init__(self, tiki):
        super().__init__(None, Qt.Window)
        style.apply(self)
        self.tiki = tiki
        self.viewer = None
        self.image: Image | None = None
        self._updating = False
        self._auto_step = 0
        self.slider_min, self.slider_max = 0.0, 1.0
        self.counts = np.zeros(1)
        self.edges = np.array([0.0, 1.0])
        self._plane_range = (0.0, 1.0)
        self.setWindowTitle('B&C')

        self.image_box = LayerCombo(None, Image)
        self.image_box.layer_changed.connect(self._set_image)
        self.lut_box = QComboBox()
        self.lut_box.addItems(list(LUTS))
        self.lut_box.currentTextChanged.connect(self._on_lut_selected)

        top = QHBoxLayout()
        top.addWidget(QLabel('Image:'))
        top.addWidget(self.image_box, 1)
        top.addWidget(QLabel('LUT:'))
        top.addWidget(self.lut_box)

        self.figure = Figure(figsize=(3.5, 0.95), dpi=100)
        self.figure.subplots_adjust(0.01, 0.04, 0.99, 0.96)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setFixedSize(370, 96)
        self.ax = self.figure.add_subplot()
        self.ax.set_axis_off()
        self.range_label = QLabel('')

        self.lo_slider, self.hi_slider = QSlider(Qt.Horizontal), QSlider(Qt.Horizontal)
        self.lo_spin, self.hi_spin = QDoubleSpinBox(), QDoubleSpinBox()
        grid = QGridLayout()
        for row, (title, slider, spin) in enumerate((
            ('Minimum', self.lo_slider, self.lo_spin),
            ('Maximum', self.hi_slider, self.hi_spin),
        )):
            slider.setRange(0, STEPS)
            spin.setKeyboardTracking(False)
            spin.setRange(-1e12, 1e12)
            spin.setMinimumWidth(95)
            grid.addWidget(QLabel(title + ':'), row, 0)
            grid.addWidget(slider, row, 1)
            grid.addWidget(spin, row, 2)
        self.lo_slider.valueChanged.connect(lambda v: self._from_slider('lo', v))
        self.hi_slider.valueChanged.connect(lambda v: self._from_slider('hi', v))
        self.lo_spin.valueChanged.connect(lambda v: self._from_spin('lo', v))
        self.hi_spin.valueChanged.connect(lambda v: self._from_spin('hi', v))

        buttons = QHBoxLayout()
        self.action_buttons: list[QPushButton] = []
        for text, slot in (
            ('Auto', self.auto),
            ('Reset', self.reset),
            ('Set…', self.set_dialog),
            ('Propagate', self.propagate),
        ):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, s=slot, t=text: self.tiki._run(f'B&C {t}', s))
            self.action_buttons.append(b)
            buttons.addWidget(b)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.canvas)
        layout.addWidget(self.range_label)
        layout.addLayout(grid)
        layout.addLayout(buttons)
        layout.setSpacing(6)
        layout.setSizeConstraint(QVBoxLayout.SetFixedSize)

        self._redraw = QTimer(self, singleShot=True, interval=40)
        self._redraw.timeout.connect(self.refresh_histogram)

    # ------------------------------------------------------------ target

    def set_target(self, viewer) -> None:
        if viewer is self.viewer:
            self.image_box.refresh(current=roi_mod.target_image(viewer) if viewer else None)
            return
        self._detach()
        self.viewer = viewer
        if viewer is not None:
            viewer.dims.events.current_step.connect(self._schedule_hist)
            viewer.layers.selection.events.active.connect(self._on_layer_selection)
        self.image_box.set_viewer(viewer, current=roi_mod.target_image(viewer) if viewer else None)
        self._set_image(self.image_box.layer())

    def _on_layer_selection(self, event=None) -> None:
        if self.viewer is None:
            return
        target = roi_mod.target_image(self.viewer)
        if target is not None and target is not self.image:
            idx = self.image_box.findData(target)
            if idx >= 0:
                self.image_box.setCurrentIndex(idx)

    def _disconnect_image(self) -> None:
        if self.image is None:
            return
        for ev, cb in (
            (self.image.events.contrast_limits, self._on_layer_contrast),
            (self.image.events.colormap, self._on_layer_colormap),
        ):
            try:
                ev.disconnect(cb)
            except Exception:
                pass

    def _detach(self) -> None:
        self._redraw.stop()
        self._disconnect_image()
        if self.viewer is not None:
            for ev, cb in (
                (self.viewer.dims.events.current_step, self._schedule_hist),
                (self.viewer.layers.selection.events.active, self._on_layer_selection),
            ):
                try:
                    ev.disconnect(cb)
                except Exception:
                    pass
        self.viewer = self.image = None
        self.image_box.blockSignals(True)
        self.image_box.set_viewer(None)
        self.image_box.blockSignals(False)

    def _set_image(self, image: Image | None) -> None:
        self._disconnect_image()
        self.image = image
        self._auto_step = 0
        enabled = image is not None and self.viewer is not None
        for w in self.action_buttons + [self.lo_slider, self.hi_slider, self.lo_spin, self.hi_spin, self.lut_box]:
            w.setEnabled(enabled)
        if not enabled:
            self.setWindowTitle('B&C (no image)')
            self.range_label.setText('')
            self.ax.clear()
            self.ax.set_axis_off()
            self.canvas.draw_idle()
            return

        self.setWindowTitle(f'B&C — {image.name}')
        image.events.contrast_limits.connect(self._on_layer_contrast)
        image.events.colormap.connect(self._on_layer_colormap)

        sample = process._sample(image.data, 8 if hasattr(image.data, 'compute') else 32)
        finite = sample[np.isfinite(sample)]
        lo, hi = map(float, image.contrast_limits)
        r0, r1 = map(float, image.contrast_limits_range)
        smin = float(finite.min()) if finite.size else lo
        smax = float(finite.max()) if finite.size else hi
        self.slider_min = min(smin, r0, lo)
        self.slider_max = max(smax, r1, hi)
        if self.slider_max <= self.slider_min:
            self.slider_max = self.slider_min + 1.0

        is_int = np.issubdtype(np.asarray(sample[:1]).dtype, np.integer)
        decimals = 0 if is_int else 4
        step = max((self.slider_max - self.slider_min) / 100.0, 1.0 if is_int else 0.01)
        for spin in (self.lo_spin, self.hi_spin):
            spin.blockSignals(True)
            spin.setDecimals(decimals)
            spin.setSingleStep(step)
            spin.blockSignals(False)

        self._sync_lut_box()
        self.refresh_histogram()
        self._sync_widgets()

    def _sync_lut_box(self) -> None:
        if self.image is None:
            return
        label = lut_label(self.image)
        self.lut_box.blockSignals(True)
        if self.lut_box.findText(label) < 0:
            self.lut_box.addItem(label)
        self.lut_box.setCurrentText(label)
        self.lut_box.blockSignals(False)

    def _on_lut_selected(self, name: str) -> None:
        if self.image is not None and name:
            apply_lut(self.image, name)

    def _on_layer_colormap(self, event=None) -> None:
        self._sync_lut_box()

    def _on_layer_contrast(self, event=None) -> None:
        if self._updating or self.image is None:
            return
        lo, hi = map(float, self.image.contrast_limits)
        self.slider_min = min(self.slider_min, lo)
        self.slider_max = max(self.slider_max, hi)
        self._sync_widgets()
        self._draw_histogram()

    # ------------------------------------------------------------ histogram

    def _schedule_hist(self, event=None) -> None:
        self._redraw.start()

    def refresh_histogram(self) -> None:
        if self.image is None or self.viewer is None or self.image not in self.viewer.layers:
            return
        plane = roi_mod.current_plane(self.viewer, self.image)
        vals = np.asarray(plane, dtype=np.float64).ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return
        pmin, pmax = float(vals.min()), float(vals.max())
        self._plane_range = (pmin, pmax)
        lo, hi = map(float, self.image.contrast_limits)
        hmin = min(pmin, lo)
        hmax = max(pmax, hi)
        if hmax <= hmin:
            hmax = hmin + 1.0
        self.counts, self.edges = np.histogram(vals, bins=256, range=(hmin, hmax))
        self._draw_histogram()

    def _draw_histogram(self) -> None:
        if self.image is None:
            return
        lo, hi = map(float, self.image.contrast_limits)
        pmin, pmax = self._plane_range
        self.ax.clear()
        self.ax.set_axis_off()
        centers = (self.edges[:-1] + self.edges[1:]) / 2.0
        width = self.edges[1] - self.edges[0]
        self.ax.bar(centers, self.counts, width=width, color='#333333', linewidth=0)
        nz = self.counts[self.counts > 0]
        ymax = max(1.0, float(np.percentile(nz, 99) * 1.15)) if nz.size else 1.0
        self.ax.plot([lo, hi], [0.0, ymax], color='#2f6fd0', lw=1.4)
        self.ax.axvline(lo, color='#2f6fd0', lw=0.8, ls='--')
        self.ax.axvline(hi, color='#b00020', lw=0.8, ls='--')
        self.ax.set_xlim(self.edges[0], self.edges[-1])
        self.ax.set_ylim(0.0, ymax)
        self.canvas.draw_idle()
        self.range_label.setText(f'Display: {lo:.4g} – {hi:.4g}   (slice: {pmin:.4g} – {pmax:.4g})')

    # ------------------------------------------------------------- controls

    def _to_slider(self, value: float) -> int:
        span = self.slider_max - self.slider_min
        if span <= 0:
            return 0
        pos = (value - self.slider_min) / span * STEPS
        return int(np.clip(round(pos), 0, STEPS))

    def _from_slider(self, which: str, step: int) -> None:
        if self._updating or self.image is None:
            return
        val = self.slider_min + step / STEPS * (self.slider_max - self.slider_min)
        lo, hi = map(float, self.image.contrast_limits)
        if which == 'lo':
            self.set_limits(val, max(val, hi))
        else:
            self.set_limits(min(val, lo), val)

    def _from_spin(self, which: str, value: float) -> None:
        if self._updating or self.image is None:
            return
        lo, hi = map(float, self.image.contrast_limits)
        if which == 'lo':
            self.set_limits(value, max(value, hi))
        else:
            self.set_limits(min(value, lo), value)

    def _sync_widgets(self) -> None:
        if self.image is None:
            return
        lo, hi = map(float, self.image.contrast_limits)
        self._updating = True
        try:
            self.lo_slider.setValue(self._to_slider(lo))
            self.hi_slider.setValue(self._to_slider(hi))
            self.lo_spin.setValue(lo)
            self.hi_spin.setValue(hi)
        finally:
            self._updating = False

    def set_limits(self, lo: float, hi: float) -> None:
        if self.image is None:
            return
        lo, hi = float(lo), float(hi)
        if hi <= lo:
            is_int = np.issubdtype(self.image.dtype, np.integer)
            eps = 1.0 if is_int else max(1e-6, abs(lo) * 1e-6, 1e-3)
            hi = lo + eps
        r0, r1 = map(float, self.image.contrast_limits_range)
        self.slider_min = min(self.slider_min, r0, lo)
        self.slider_max = max(self.slider_max, r1, hi)
        self._updating = True
        try:
            self.image.contrast_limits_range = (self.slider_min, self.slider_max)
            self.image.contrast_limits = (lo, hi)
        finally:
            self._updating = False
        self._sync_widgets()
        self._draw_histogram()

    def auto(self) -> None:
        if self.image is None or self.viewer is None:
            return
        sat = SATURATION_CYCLE[self._auto_step % len(SATURATION_CYCLE)]
        self._auto_step += 1
        plane = roi_mod.current_plane(self.viewer, self.image)
        current = self.tiki.current_roi(self.viewer)
        if current is not None and current[1] not in ('line', 'path'):
            sel = plane[roi_mod.shape_mask(*current, plane.shape)]
            if sel.size:
                plane = sel
        finite = plane[np.isfinite(plane)]
        if finite.size == 0:
            return
        lo, hi = ops.saturated_limits(finite, sat)
        self.set_limits(lo, hi)
        self.tiki.status(f'B&C Auto ({sat:g}% saturated): {lo:.4g}–{hi:.4g}')

    def reset(self) -> None:
        if self.image is None:
            return
        self._auto_step = 0
        sample = process._sample(self.image.data, 8 if hasattr(self.image.data, 'compute') else 32)
        finite = sample[np.isfinite(sample)]
        if finite.size == 0:
            return
        lo, hi = float(finite.min()), float(finite.max())
        if hi <= lo:
            hi = lo + 1.0
        self.slider_min, self.slider_max = lo, hi
        self.set_limits(lo, hi)
        self.tiki.status(f'B&C Reset: {lo:.4g}–{hi:.4g}')

    def set_dialog(self) -> None:
        if self.image is None:
            return
        lo_cur, hi_cur = map(float, self.image.contrast_limits)
        dlg = QDialog(self)
        style.apply(dlg)
        dlg.setWindowTitle('Set Display Range')
        form = QFormLayout(dlg)
        lo_box = QDoubleSpinBox(decimals=self.lo_spin.decimals(), minimum=-1e12, maximum=1e12, value=lo_cur)
        hi_box = QDoubleSpinBox(decimals=self.hi_spin.decimals(), minimum=-1e12, maximum=1e12, value=hi_cur)
        prop = QCheckBox('Propagate to all other open windows')
        form.addRow('Minimum:', lo_box)
        form.addRow('Maximum:', hi_box)
        form.addRow(prop)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)
        if dlg.exec_() != QDialog.Accepted:
            return
        self.set_limits(lo_box.value(), hi_box.value())
        if prop.isChecked():
            self.propagate()

    def propagate(self) -> int:
        if self.image is None:
            return 0
        lo, hi = map(float, self.image.contrast_limits)
        ch = self.image.metadata.get('channel')
        updated = 0
        for v in self.tiki.windows.viewers:
            if v is self.viewer:
                continue
            target = None
            if ch is not None:
                target = next(
                    (
                        ly
                        for ly in v.layers
                        if isinstance(ly, Image)
                        and ly.name not in roi_mod.PREVIEW_NAMES
                        and ly.metadata.get('channel') == ch
                    ),
                    None,
                )
            if target is None:
                target = roi_mod.target_image(v)
            if target is None:
                continue
            r0, r1 = map(float, target.contrast_limits_range)
            target.contrast_limits_range = (min(r0, lo), max(r1, hi))
            target.contrast_limits = (lo, hi)
            updated += 1
        self.tiki.status(f'Propagated display range {lo:.4g}–{hi:.4g} to {updated} window(s)')
        return updated

    def closeEvent(self, event):
        self._detach()
        super().closeEvent(event)
