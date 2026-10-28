"""Process > Filters and Subtract Background, with a live preview.

Unlike ImageJ, filtering is non-destructive: OK adds the result as a new
image layer in the same window (on top, so it becomes the current image),
and the original stays underneath.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from napari.layers import Image
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
)

from . import axes as axes_mod
from . import ops, style
from . import roi as roi_mod

PREVIEW_NAME = 'filter preview'


@dataclass
class Param:
    key: str
    label: str
    default: float
    minimum: float
    maximum: float
    step: float = 1.0
    decimals: int = 1
    suffix: str = ''


@dataclass
class FilterSpec:
    title: str
    short: str  # suffix for the result layer name
    params: list[Param]
    flags: list[tuple[str, str, bool]]  # (key, label, default)
    func: object  # func(plane, **values) -> plane

    def name_suffix(self, values: dict) -> str:
        parts = [f'{p.key}={values[p.key]:g}' for p in self.params]
        parts += [k for k, _, _ in self.flags if values.get(k)]
        return f'{self.short} ' + ' '.join(parts)


FILTERS = {
    'gaussian': FilterSpec(
        'Gaussian Blur', 'gauss',
        [Param('sigma', 'Sigma (radius):', 2.0, 0.1, 100, 0.5, 2, ' px')], [],
        lambda plane, sigma: ops.gaussian_blur(plane, sigma),
    ),
    'median': FilterSpec(
        'Median', 'median',
        [Param('radius', 'Radius:', 2.0, 0.5, 50, 0.5, 1, ' px')], [],
        lambda plane, radius: ops.median_filter(plane, radius),
    ),
    'background': FilterSpec(
        'Subtract Background', 'bgsub',
        [Param('radius', 'Rolling ball radius:', 50.0, 1, 1000, 5, 1, ' px')],
        [('light', 'Light background', False)],
        lambda plane, radius, light=False: ops.subtract_background(plane, radius, light),
    ),
}


class FilterDialog(QDialog):
    def __init__(self, tiki, key: str, viewer, image):
        super().__init__(tiki)
        style.apply(self)
        self.tiki, self.spec, self.viewer, self.image = tiki, FILTERS[key], viewer, image
        self.preview: Image | None = None
        self.setWindowTitle(f'{self.spec.title} — {image.name}')
        form = QFormLayout(self)
        self.inputs: dict[str, object] = {}
        for p in self.spec.params:
            spin = QDoubleSpinBox()
            spin.setRange(p.minimum, p.maximum)
            spin.setSingleStep(p.step)
            spin.setDecimals(p.decimals)
            spin.setValue(p.default)
            spin.setSuffix(p.suffix)
            spin.setKeyboardTracking(False)
            spin.valueChanged.connect(lambda _: self.schedule())
            form.addRow(p.label, spin)
            self.inputs[p.key] = spin
        for k, label, default in self.spec.flags:
            box = QCheckBox(label, checked=default)
            box.toggled.connect(lambda _: self.schedule())
            form.addRow(box)
            self.inputs[k] = box
        self.whole_stack = QCheckBox('Process whole stack', checked=True)
        if image.ndim > 2:
            n = int(np.prod(image.data.shape[: image.ndim - 2]))
            self.whole_stack.setText(f'Process whole stack ({n} planes)')
            form.addRow(self.whole_stack)
        self.show_preview = QCheckBox('Preview', checked=True)
        self.show_preview.toggled.connect(lambda on: self.update_preview() if on else self._remove_preview())
        form.addRow(self.show_preview)
        self.info = QLabel('')
        form.addRow(self.info)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

        self._timer = QTimer(self, singleShot=True, interval=150)
        self._timer.timeout.connect(self.update_preview)
        viewer.dims.events.current_step.connect(self.schedule)
        self.update_preview()

    def values(self) -> dict:
        out = {}
        for k, w in self.inputs.items():
            out[k] = w.isChecked() if isinstance(w, QCheckBox) else w.value()
        return out

    def schedule(self, event=None) -> None:
        if self.show_preview.isChecked():
            self._timer.start()

    def update_preview(self) -> None:
        if not self.show_preview.isChecked():
            return
        plane = roi_mod.current_plane(self.viewer, self.image)
        result = self.spec.func(plane, **self.values())
        lo, hi = ops.saturated_limits(result)
        if self.preview is None or self.preview not in self.viewer.layers:
            self.preview = Image(result, name=PREVIEW_NAME, colormap=self.image.colormap,
                                 contrast_limits=(lo, hi), **axes_mod.geometry(self.image, ndim=2))
            self.viewer.add_layer(self.preview)
            self.tiki.reapply_tool(self.viewer)
        else:
            self.preview.data = result
            self.preview.contrast_limits = (lo, hi)

    def _remove_preview(self) -> None:
        if self.preview is not None and self.preview in self.viewer.layers:
            self.viewer.layers.remove(self.preview)
        self.preview = None

    def done(self, result: int) -> None:
        self._timer.stop()
        try:
            self.viewer.dims.events.current_step.disconnect(self.schedule)
        except Exception:
            pass
        self._remove_preview()
        super().done(result)


def run_filter(tiki, key: str, viewer, image, values: dict, whole_stack: bool = True) -> Image | None:
    """Filter ``image`` and add the result as a new layer; returns it (None if cancelled)."""
    spec = FILTERS[key]
    func = lambda plane: spec.func(plane, **values)  # noqa: E731
    if image.ndim > 2 and whole_stack:
        data = image.data
        if image.rgb:
            from .app import TikiError

            raise TikiError('filters on RGB stacks are not supported yet')
        with tiki.progress(spec.title, int(np.prod(data.shape[:-2]))) as tick:
            result = ops.map_planes(func, data, progress=tick)
        if tick.cancelled:
            tiki.status(f'{spec.title} cancelled')
            return None
        geometry = axes_mod.geometry(image)
    else:
        result = func(roi_mod.current_plane(viewer, image))
        geometry = axes_mod.geometry(image, ndim=2)
    lo, hi = ops.saturated_limits(result if result.ndim == 2 else result[(0,) * (result.ndim - 2)])
    info = axes_mod.get(image) if result.ndim == image.ndim else axes_mod.guess(result.shape)
    layer = Image(result, name=f'{image.name} {spec.name_suffix(values)}', colormap=image.colormap,
                  contrast_limits=(lo, hi), **geometry,
                  metadata={'axes': axes_mod.fit(info, result.shape).to_dict()})
    viewer.add_layer(layer)
    tiki.reapply_tool(viewer)
    tiki.status(f'{spec.title}: new layer "{layer.name}" (original kept underneath)')
    return layer
