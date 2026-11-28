"""Process > Noise and Process > FFT commands with live preview and undo support."""

from __future__ import annotations

import numpy as np
from napari.layers import Image
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
)

from . import axes as axes_mod
from . import nd, ops, process, style
from . import roi as roi_mod


def _apply_plane_op(
    tiki,
    title: str,
    suffix: str,
    fn,
    whole_stack: bool = True,
    out_dtype=None,
    record_cmd: str | None = None,
    viewer=None,
    image: Image | None = None,
):
    """Apply a 2D image function to the current plane or every plane in a stack."""
    if viewer is None or image is None:
        viewer, image = tiki.require_image()
    if image.rgb:
        process._error(f'{title} works on greyscale images (split channels or convert first)')
    use_stack = image.ndim > 2 and whole_stack
    dtype_out = image.data.dtype if out_dtype is None else np.dtype(out_dtype)
    if use_stack:
        if not process._confirm_size(tiki, image.data.shape, title):
            return None
        total = int(np.prod(image.data.shape[: image.ndim - 2]))
        out, cancelled = tiki.run_task(
            title, total, lambda tick: ops.map_planes(fn, image.data, out_dtype=dtype_out, progress=tick)
        )
        if cancelled:
            return None
        lo, hi = ops.saturated_limits(out[(0,) * (out.ndim - 2)])
        layer = process._add_result(tiki, viewer, image, out, f'{image.name} {suffix}', image.colormap, (lo, hi))
    else:
        plane = roi_mod.current_plane(viewer, image)
        out = np.asarray(fn(plane), dtype=dtype_out)
        lo, hi = ops.saturated_limits(out)
        if image.ndim > 2:
            info = axes_mod.AxesInfo(axes_mod.get(image).axes[-2:], axes_mod.get(image).source, True)
            layer = Image(
                out,
                name=f'{image.name} {suffix}',
                colormap=image.colormap,
                contrast_limits=(lo, hi),
                metadata={'axes': info.to_dict()},
                **axes_mod.geometry(image, ndim=2),
            )
            viewer.add_layer(layer)
            tiki.reapply_tool(viewer)
            tiki.push_undo_layer(title, viewer, layer)
        else:
            layer = process._add_result(tiki, viewer, image, out, f'{image.name} {suffix}', image.colormap, (lo, hi))
    if record_cmd:
        tiki.record(record_cmd)
    tiki.status(f'{title} → "{layer.name}"')
    return layer


def _phys_radius_text(image: Image | None, radius_px: float, label: str = 'Radius') -> str:
    if image is None:
        return ''
    info = axes_mod.get(image)
    u = axes_mod.spatial_unit(info)
    if not u:
        return ''
    sy = float(info.axes[-2].step or 1.0)
    sx = float(info.axes[-1].step or sy)
    return f'{label} ≈ {radius_px * 0.5 * (sy + sx):.4g} {u}'


# ----------------------------------------------------------------------- Noise


def despeckle_command(tiki):
    """``Process ▸ Noise ▸ Despeckle``: 3×3 median filter across plane or stack."""
    return _apply_plane_op(
        tiki, 'Despeckle', 'despeckled', ops.despeckle, record_cmd="tiki.run('Despeckle')"
    )


def add_noise_command(tiki):
    """``Process ▸ Noise ▸ Add Noise``: zero-mean Gaussian noise (σ = 25)."""
    return _apply_plane_op(
        tiki,
        'Add Noise',
        'noise',
        lambda pl: ops.add_gaussian_noise(pl, 25.0),
        record_cmd="tiki.run('Add Noise')",
    )


def run_add_noise(tiki, viewer, image: Image, sigma: float = 25.0, whole_stack: bool = True):
    """Add zero-mean Gaussian noise with standard deviation ``sigma``."""
    sig = float(sigma)
    use_stack = image.ndim > 2 and bool(whole_stack)
    return _apply_plane_op(
        tiki,
        'Add Specified Noise',
        f'noise σ={sig:g}',
        lambda pl: ops.add_gaussian_noise(pl, sig),
        whole_stack=use_stack,
        viewer=viewer,
        image=image,
        record_cmd=f'tiki.add_noise({sig!r}, whole_stack={use_stack!r})',
    )


def add_specified_noise_dialog(tiki):
    """``Process ▸ Noise ▸ Add Specified Noise…``: Gaussian noise with chosen σ."""
    viewer, image = tiki.require_image()
    if image.rgb:
        process._error('Add Specified Noise works on greyscale images')
    dlg, form = process._form(tiki, 'Add Specified Noise')
    sigma_spin = QDoubleSpinBox(decimals=2, minimum=0.0, maximum=1e6, singleStep=5.0, value=25.0)
    whole = QCheckBox('Process whole stack', checked=True)
    has_stack = image.ndim > 2
    whole.setVisible(has_stack)
    if has_stack:
        n = int(np.prod(image.data.shape[: image.ndim - 2]))
        whole.setText(f'Process whole stack ({n} planes)')
    form.addRow('Standard deviation (σ):', sigma_spin)
    form.addRow(whole)
    process._buttons(dlg, form)
    if dlg.exec_() != QDialog.Accepted:
        return None
    return run_add_noise(
        tiki,
        viewer,
        image,
        sigma=sigma_spin.value(),
        whole_stack=has_stack and whole.isChecked(),
    )


def salt_and_pepper_command(tiki):
    """``Process ▸ Noise ▸ Salt and Pepper``: impulse noise on 5% of pixels."""
    return _apply_plane_op(
        tiki,
        'Salt and Pepper',
        'salt-pepper',
        lambda pl: ops.salt_and_pepper(pl, 0.05),
        record_cmd="tiki.run('Salt and Pepper')",
    )


class RemoveOutliersDialog(QDialog):
    """``Process ▸ Noise ▸ Remove Outliers…`` with live slice preview."""

    def __init__(self, tiki, viewer, image: Image):
        from .layer_select import LayerCombo

        super().__init__(tiki)
        style.apply(self)
        self.tiki = tiki
        self.viewer = viewer
        self.image = image
        self.preview: Image | None = None
        self.has_stack = False
        self.setWindowTitle('Remove Outliers')

        form = QFormLayout(self)
        self.image_box = LayerCombo(viewer, Image, current=image)
        self.radius_spin = QDoubleSpinBox(decimals=1, minimum=0.5, maximum=50.0, singleStep=0.5, value=2.0)
        self.radius_spin.setSuffix(' px')
        self.phys_label = QLabel('')
        self.thresh_spin = QDoubleSpinBox(decimals=2, minimum=0.0, maximum=1e9, singleStep=10.0, value=50.0)
        self.which_box = QComboBox()
        self.which_box.addItems(['Bright', 'Dark', 'Both'])
        self.whole_stack = QCheckBox('Process whole stack', checked=True)
        self.show_preview = QCheckBox('Preview', checked=True)

        form.addRow('Image:', self.image_box)
        form.addRow('Radius:', self.radius_spin)
        form.addRow('', self.phys_label)
        form.addRow('Threshold:', self.thresh_spin)
        form.addRow('Which outliers:', self.which_box)
        form.addRow(self.whole_stack)
        form.addRow(self.show_preview)
        process._buttons(self, form)

        self._timer = QTimer(self, singleShot=True, interval=150)
        self._timer.timeout.connect(self.update_preview)
        self.image_box.layer_changed.connect(self._set_image)
        self.radius_spin.valueChanged.connect(lambda _: self._sync_and_schedule())
        self.thresh_spin.valueChanged.connect(lambda _: self.schedule())
        self.which_box.currentTextChanged.connect(lambda _: self.schedule())
        self.show_preview.toggled.connect(lambda on: self.update_preview() if on else self._remove_preview())
        viewer.dims.events.current_step.connect(self.schedule)
        self._sync_options()
        self.update_preview()

    def _sync_options(self) -> None:
        n = int(np.prod(self.image.data.shape[: self.image.ndim - 2])) if self.image.ndim > 2 else 0
        self.has_stack = n > 0
        self.whole_stack.setVisible(self.has_stack)
        self.whole_stack.setText(f'Process whole stack ({n} planes)')
        self.phys_label.setText(_phys_radius_text(self.image, self.radius_spin.value()))

    def _sync_and_schedule(self) -> None:
        self.phys_label.setText(_phys_radius_text(self.image, self.radius_spin.value()))
        self.schedule()

    def _set_image(self, image) -> None:
        if image is None or image is self.image:
            return
        self._remove_preview()
        self.image = image
        self._sync_options()
        self.schedule()

    def options(self) -> dict:
        return {
            'radius': self.radius_spin.value(),
            'threshold': self.thresh_spin.value(),
            'which': self.which_box.currentText().lower(),
        }

    def schedule(self, _event=None) -> None:
        if self.show_preview.isChecked():
            self._timer.start()

    def update_preview(self) -> None:
        if not self.show_preview.isChecked():
            return
        if self.image is None or self.image not in self.viewer.layers or self.image.rgb:
            return
        plane = roi_mod.current_plane(self.viewer, self.image)
        result = ops.remove_outliers(plane, **self.options())
        lo, hi = ops.saturated_limits(result)
        if self.preview is None or self.preview not in self.viewer.layers:
            self.preview = Image(
                result,
                name='filter preview',
                colormap=self.image.colormap,
                contrast_limits=(lo, hi),
                **axes_mod.geometry(self.image, ndim=2),
            )
            self.viewer.add_layer(self.preview)
            self.tiki.reapply_tool(self.viewer)
        else:
            self.preview.data = result
            self.preview.contrast_limits = (lo, hi)

    def _remove_preview(self) -> None:
        if self.preview is not None and self.preview in self.viewer.layers:
            self.viewer.layers.remove(self.preview)
        self.preview = None

    def cleanup(self) -> None:
        self._timer.stop()
        try:
            self.viewer.dims.events.current_step.disconnect(self.schedule)
        except Exception:
            pass
        self._remove_preview()
        self.image_box.blockSignals(True)
        self.image_box.set_viewer(None)
        self.image_box.blockSignals(False)

    def done(self, result: int) -> None:
        self.cleanup()
        super().done(result)


def run_remove_outliers(
    tiki,
    viewer,
    image: Image,
    radius: float = 2.0,
    threshold: float = 50.0,
    which: str = 'bright',
    whole_stack: bool = True,
):
    """Remove bright/dark intensity outliers on ``image`` (plane or stack) and record it."""
    if image.rgb:
        process._error('Remove Outliers works on greyscale images (split channels or convert first)')
    r = float(radius)
    thr = float(threshold)
    w = str(which).lower()
    use_stack = image.ndim > 2 and bool(whole_stack)
    return _apply_plane_op(
        tiki,
        'Remove Outliers',
        f'outliers r={r:g} thr={thr:g}',
        lambda pl: ops.remove_outliers(pl, radius=r, threshold=thr, which=w),
        whole_stack=use_stack,
        out_dtype=image.data.dtype,
        viewer=viewer,
        image=image,
        record_cmd=f'tiki.remove_outliers(radius={r!r}, threshold={thr!r}, which={w!r}, whole_stack={use_stack!r})',
    )


def remove_outliers_dialog(tiki):
    """``Process ▸ Noise ▸ Remove Outliers…``."""
    viewer, image = tiki.require_image()
    if image.rgb:
        process._error('Remove Outliers works on greyscale images (split channels or convert first)')
    dlg = RemoveOutliersDialog(tiki, viewer, image)
    try:
        accepted = dlg.exec_() == QDialog.Accepted
    finally:
        dlg.cleanup()
    if not accepted or dlg.image is None:
        return None
    opts = dlg.options()
    return run_remove_outliers(
        tiki,
        viewer,
        dlg.image,
        radius=opts['radius'],
        threshold=opts['threshold'],
        which=opts['which'],
        whole_stack=dlg.has_stack and dlg.whole_stack.isChecked(),
    )


def run_remove_nans(tiki, viewer, image: Image, radius: float = 2.0, whole_stack: bool = True):
    """Replace non-finite pixels on ``image`` (plane or stack) with local median."""
    if image.rgb:
        process._error('Remove NaNs works on greyscale images')
    r = float(radius)
    use_stack = image.ndim > 2 and bool(whole_stack)
    return _apply_plane_op(
        tiki,
        'Remove NaNs',
        f'no-nan r={r:g}',
        lambda pl: ops.remove_nans(pl, r),
        whole_stack=use_stack,
        out_dtype=np.float32 if np.issubdtype(image.data.dtype, np.floating) else image.data.dtype,
        viewer=viewer,
        image=image,
        record_cmd=f'tiki.remove_nans(radius={r!r}, whole_stack={use_stack!r})',
    )


def remove_nans_dialog(tiki):
    """``Process ▸ Noise ▸ Remove NaNs…``."""
    viewer, image = tiki.require_image()
    if image.rgb:
        process._error('Remove NaNs works on greyscale images')
    dlg, form = process._form(tiki, 'Remove NaNs')
    radius_spin = QDoubleSpinBox(decimals=1, minimum=1.0, maximum=50.0, singleStep=1.0, value=2.0)
    radius_spin.setSuffix(' px')
    phys_label = QLabel(_phys_radius_text(image, radius_spin.value()))
    radius_spin.valueChanged.connect(lambda v: phys_label.setText(_phys_radius_text(image, v)))
    whole = QCheckBox('Process whole stack', checked=True)
    has_stack = image.ndim > 2
    whole.setVisible(has_stack)
    if has_stack:
        n = int(np.prod(image.data.shape[: image.ndim - 2]))
        whole.setText(f'Process whole stack ({n} planes)')
    form.addRow('Radius:', radius_spin)
    form.addRow('', phys_label)
    form.addRow(whole)
    process._buttons(dlg, form)
    if dlg.exec_() != QDialog.Accepted:
        return None
    return run_remove_nans(
        tiki,
        viewer,
        image,
        radius=radius_spin.value(),
        whole_stack=has_stack and whole.isChecked(),
    )


# ------------------------------------------------------------------------- FFT


def fft_command(tiki):
    """``Process ▸ FFT ▸ FFT``: open shifted 2D log-power spectrum + stored complex FFT."""
    viewer, image = tiki.require_image()
    if image.rgb:
        process._error('FFT works on greyscale images (split channels or convert first)')
    plane = roi_mod.current_plane(viewer, image)
    power, spec = ops.fft_forward(plane)
    src_axes = axes_mod.AxesInfo(
        axes_mod.get(image).axes[-2:], axes_mod.get(image).source, axes_mod.get(image).confident
    )
    fft_info = axes_mod.AxesInfo(
        [axes_mod.Axis('Y', power.shape[0]), axes_mod.Axis('X', power.shape[1])],
        'FFT spectrum',
        True,
    )
    meta = {
        'fft_complex': spec,
        'fft_shape': tuple(plane.shape),
        'fft_source_axes': src_axes.to_dict(),
    }
    v = tiki.open_data(power, f'FFT of {image.name}', axes=fft_info, colormap='viridis', metadata=meta)
    tiki.record("tiki.run('FFT')")
    tiki.status(f'FFT → "{v.title}" (keep open for Inverse FFT)')
    return v


def inverse_fft_command(tiki):
    """``Process ▸ FFT ▸ Inverse FFT``: reconstruct spatial image from FFT spectrum."""
    _viewer, image = tiki.require_image()
    spec = image.metadata.get('fft_complex')
    if spec is None:
        process._error('Inverse FFT needs an FFT window created by Process ▸ FFT ▸ FFT')
    spec_use = np.asarray(spec)
    cur_power = np.asarray(image.data, dtype=np.float32)
    if cur_power.shape == spec_use.shape:
        orig_power = np.log1p(np.abs(spec_use)).astype(np.float32)
        if not np.allclose(cur_power, orig_power, equal_nan=True):
            old_mag = np.abs(spec_use)
            new_mag = np.where(
                np.isfinite(cur_power) & (cur_power > 0.0),
                np.maximum(np.expm1(np.clip(cur_power, 0.0, 80.0)), 0.0),
                0.0,
            )
            with np.errstate(divide='ignore', invalid='ignore'):
                weight = np.where(old_mag > 0, new_mag / old_mag, 0.0)
            spec_use = (spec_use * weight).astype(np.complex64)
    rec = ops.fft_inverse(spec_use, shape=image.metadata.get('fft_shape'))
    src_dict = image.metadata.get('fft_source_axes')
    if isinstance(src_dict, dict):
        src_info = axes_mod.AxesInfo.from_dict(src_dict)
        info = axes_mod.fit(src_info, rec.shape)
    else:
        info = axes_mod.guess(rec.shape)
    v = tiki.open_data(rec, f'Inverse {image.name}', axes=info)
    tiki.record("tiki.run('Inverse FFT')")
    tiki.status(f'Inverse FFT → "{v.title}"')
    return v


class BandpassDialog(QDialog):
    """``Process ▸ FFT ▸ Bandpass Filter…`` with stripe suppression and live preview."""

    def __init__(self, tiki, viewer, image: Image):
        from .layer_select import LayerCombo

        super().__init__(tiki)
        style.apply(self)
        self.tiki = tiki
        self.viewer = viewer
        self.image = image
        self.preview: Image | None = None
        self.has_stack = False
        self.setWindowTitle('FFT Bandpass Filter')

        form = QFormLayout(self)
        self.image_box = LayerCombo(viewer, Image, current=image)
        self.large_spin = QDoubleSpinBox(decimals=1, minimum=0.0, maximum=10000.0, singleStep=5.0, value=40.0)
        self.large_spin.setSuffix(' px')
        self.small_spin = QDoubleSpinBox(decimals=1, minimum=0.0, maximum=1000.0, singleStep=1.0, value=3.0)
        self.small_spin.setSuffix(' px')
        self.phys_label = QLabel('')
        self.stripes_box = QComboBox()
        self.stripes_box.addItems(['None', 'Horizontal', 'Vertical'])
        self.tol_spin = QDoubleSpinBox(decimals=1, minimum=0.5, maximum=45.0, singleStep=1.0, value=5.0)
        self.tol_spin.setSuffix('°')
        self.autoscale_box = QCheckBox('Autoscale after filtering')
        self.whole_stack = QCheckBox('Process whole stack', checked=True)
        self.show_preview = QCheckBox('Preview', checked=True)

        form.addRow('Image:', self.image_box)
        form.addRow('Filter large structures down to:', self.large_spin)
        form.addRow('Filter small structures up to:', self.small_spin)
        form.addRow('', self.phys_label)
        form.addRow('Suppress stripes:', self.stripes_box)
        form.addRow('Tolerance of direction:', self.tol_spin)
        form.addRow(self.autoscale_box)
        form.addRow(self.whole_stack)
        form.addRow(self.show_preview)
        process._buttons(self, form)

        self._timer = QTimer(self, singleShot=True, interval=150)
        self._timer.timeout.connect(self.update_preview)
        self.image_box.layer_changed.connect(self._set_image)
        self.large_spin.valueChanged.connect(lambda _: self._sync_and_schedule())
        self.small_spin.valueChanged.connect(lambda _: self._sync_and_schedule())
        self.stripes_box.currentTextChanged.connect(lambda _: self._sync_and_schedule())
        self.tol_spin.valueChanged.connect(lambda _: self.schedule())
        self.autoscale_box.toggled.connect(lambda _: self.schedule())
        self.show_preview.toggled.connect(lambda on: self.update_preview() if on else self._remove_preview())
        viewer.dims.events.current_step.connect(self.schedule)
        self._sync_options()
        self.update_preview()

    def _sync_options(self) -> None:
        n = int(np.prod(self.image.data.shape[: self.image.ndim - 2])) if self.image.ndim > 2 else 0
        self.has_stack = n > 0
        self.whole_stack.setVisible(self.has_stack)
        self.whole_stack.setText(f'Process whole stack ({n} planes)')
        self.tol_spin.setEnabled(self.stripes_box.currentText() != 'None')
        parts = [
            _phys_radius_text(self.image, self.large_spin.value(), 'Large'),
            _phys_radius_text(self.image, self.small_spin.value(), 'Small'),
        ]
        self.phys_label.setText('; '.join(p for p in parts if p))

    def _sync_and_schedule(self) -> None:
        self._sync_options()
        self.schedule()

    def _set_image(self, image) -> None:
        if image is None or image is self.image:
            return
        self._remove_preview()
        self.image = image
        self._sync_options()
        self.schedule()

    def options(self) -> dict:
        return {
            'filter_large': self.large_spin.value(),
            'filter_small': self.small_spin.value(),
            'suppress_stripes': self.stripes_box.currentText().lower(),
            'tolerance': self.tol_spin.value(),
            'autoscale': self.autoscale_box.isChecked(),
        }

    def schedule(self, _event=None) -> None:
        if self.show_preview.isChecked():
            self._timer.start()

    def update_preview(self) -> None:
        if not self.show_preview.isChecked():
            return
        if self.image is None or self.image not in self.viewer.layers or self.image.rgb:
            return
        plane = roi_mod.current_plane(self.viewer, self.image)
        result = ops.bandpass_filter(plane, **self.options())
        lo, hi = ops.saturated_limits(result)
        if self.preview is None or self.preview not in self.viewer.layers:
            self.preview = Image(
                result,
                name='filter preview',
                colormap=self.image.colormap,
                contrast_limits=(lo, hi),
                **axes_mod.geometry(self.image, ndim=2),
            )
            self.viewer.add_layer(self.preview)
            self.tiki.reapply_tool(self.viewer)
        else:
            self.preview.data = result
            self.preview.contrast_limits = (lo, hi)

    def _remove_preview(self) -> None:
        if self.preview is not None and self.preview in self.viewer.layers:
            self.viewer.layers.remove(self.preview)
        self.preview = None

    def cleanup(self) -> None:
        self._timer.stop()
        try:
            self.viewer.dims.events.current_step.disconnect(self.schedule)
        except Exception:
            pass
        self._remove_preview()
        self.image_box.blockSignals(True)
        self.image_box.set_viewer(None)
        self.image_box.blockSignals(False)

    def done(self, result: int) -> None:
        self.cleanup()
        super().done(result)


def run_bandpass(
    tiki,
    viewer,
    image: Image,
    filter_large: float = 40.0,
    filter_small: float = 3.0,
    suppress_stripes: str = 'none',
    tolerance: float = 5.0,
    autoscale: bool = False,
    whole_stack: bool = True,
):
    """Run 2D FFT Bandpass Filter on ``image`` (plane or stack) and record it."""
    if image.rgb:
        process._error('Bandpass Filter works on greyscale images (split channels or convert first)')
    fl = float(filter_large)
    fs = float(filter_small)
    ss = str(suppress_stripes).lower()
    tol = float(tolerance)
    auto = bool(autoscale)
    use_stack = image.ndim > 2 and bool(whole_stack)
    return _apply_plane_op(
        tiki,
        'Bandpass Filter',
        f'bandpass {fs:g}–{fl:g}',
        lambda pl: ops.bandpass_filter(
            pl,
            filter_large=fl,
            filter_small=fs,
            suppress_stripes=ss,
            tolerance=tol,
            autoscale=auto,
        ),
        whole_stack=use_stack,
        out_dtype=np.float32,
        viewer=viewer,
        image=image,
        record_cmd=(
            f'tiki.bandpass_filter(filter_large={fl!r}, filter_small={fs!r}, '
            f'suppress_stripes={ss!r}, tolerance={tol!r}, autoscale={auto!r}, whole_stack={use_stack!r})'
        ),
    )


def bandpass_dialog(tiki):
    """``Process ▸ FFT ▸ Bandpass Filter…``."""
    viewer, image = tiki.require_image()
    if image.rgb:
        process._error('Bandpass Filter works on greyscale images (split channels or convert first)')
    dlg = BandpassDialog(tiki, viewer, image)
    try:
        accepted = dlg.exec_() == QDialog.Accepted
    finally:
        dlg.cleanup()
    if not accepted or dlg.image is None:
        return None
    opts = dlg.options()
    return run_bandpass(
        tiki,
        viewer,
        dlg.image,
        filter_large=opts['filter_large'],
        filter_small=opts['filter_small'],
        suppress_stripes=opts['suppress_stripes'],
        tolerance=opts['tolerance'],
        autoscale=opts['autoscale'],
        whole_stack=dlg.has_stack and dlg.whole_stack.isChecked(),
    )
