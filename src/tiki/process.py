"""Process > ΔF/F, Bleach Correction, Image Calculator; Stacks > Kymograph;
Analyze > Set Scale. Dialogs collect options; the maths lives in ``ops``.

Every command is non-destructive: results are new layers (same window) or
new windows, never an overwrite of the source data.
"""

from __future__ import annotations

import numpy as np
from napari.layers import Image
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QMessageBox,
    QSpinBox,
)

from . import axes as axes_mod
from . import ops, style
from . import roi as roi_mod

LARGE_RESULT_BYTES = 2 * 2**30


def _error(message, fix=None):
    from .app import TikiError

    raise TikiError(message, fix=fix)


def _fill_axes(box: QComboBox, image, prefer: str) -> None:
    """Stack axes of ``image`` (never C), defaulting to the first of ``prefer``."""
    box.clear()
    if image is None:
        return
    info = axes_mod.get(image)
    for d in range(image.ndim - 2):
        if info.axes[d].role != 'C':
            box.addItem(f'{info.labels[d]} ({image.data.shape[d]})', d)
    default = axes_mod.stack_axis(info, prefer)
    if default is not None and box.findData(default) >= 0:
        box.setCurrentIndex(box.findData(default))


def _stack_axis_box(image, prefer='TZ', image_box=None) -> QComboBox:
    """Dropdown of stack axes; follows ``image_box`` when the image changes."""
    box = QComboBox()
    _fill_axes(box, image, prefer)
    if image_box is not None:
        image_box.layer_changed.connect(lambda layer: _fill_axes(box, layer, prefer))
    return box


def _form(parent, title):
    dlg = QDialog(parent)
    style.apply(dlg)
    dlg.setWindowTitle(title)
    return dlg, QFormLayout(dlg)


def _buttons(dlg, form):
    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)
    form.addRow(buttons)
    return buttons


def _confirm_size(parent, shape, what: str) -> bool:
    nbytes = int(np.prod(shape)) * 4
    if nbytes <= LARGE_RESULT_BYTES:
        return True
    answer = QMessageBox.question(
        parent, what,
        f'The result is {nbytes / 2**30:.1f} GB of float32 in memory. Continue?\n\n'
        'Tip: Duplicate a crop or a substack first to work on less data.')
    return answer == QMessageBox.Yes


def _add_result(tiki, viewer, source, data, name, colormap=None, limits=None, axes=None, drop=None):
    """Add ``data`` as a layer aligned with ``source`` and keep the ROI tool usable.

    ``drop`` names the source axis that ``data`` lacks (e.g. T for an F0 image).
    """
    info = axes or axes_mod.fit(axes_mod.get(source), data.shape)
    kwargs = dict(name=name, metadata={'axes': info.to_dict()}, **axes_mod.geometry(source, drop=drop))
    if colormap:
        kwargs['colormap'] = colormap
    if limits:
        kwargs['contrast_limits'] = limits
    layer = Image(data, **kwargs)
    viewer.add_layer(layer)
    tiki.reapply_tool(viewer)
    return layer


# ------------------------------------------------------------------ ΔF/F

def dff_dialog(tiki) -> Image | None:
    from .layer_select import LayerCombo

    viewer, image = tiki.require_image()
    if image.ndim < 3:
        _error('ΔF/F needs a movie or stack')
    dlg, form = _form(tiki, 'ΔF/F')
    image_box = LayerCombo(viewer, Image, current=image)
    form.addRow('Image:', image_box)
    axis_box = _stack_axis_box(image, 'TZ', image_box)
    form.addRow('Along:', axis_box)
    method = QComboBox()
    method.addItems(ops.F0_METHODS)
    form.addRow('F0:', method)
    pct = QDoubleSpinBox(decimals=1, minimum=0, maximum=100, value=10)
    pct.setSuffix(' %')
    n_first = QSpinBox(minimum=1, maximum=10**6, value=10)
    window = QSpinBox(minimum=3, maximum=10**6, value=101, singleStep=2)
    window.setSuffix(' frames')
    form.addRow('Percentile:', pct)
    form.addRow('First frames:', n_first)
    form.addRow('Window:', window)
    offset = QDoubleSpinBox(decimals=2, minimum=-1e9, maximum=1e9, value=0)
    offset.setToolTip('Subtracted from F before ΔF/F, e.g. the camera dark level')
    form.addRow('Offset:', offset)
    add_f0 = QCheckBox('Also add the F0 image')
    form.addRow(add_f0)

    def show_params(_=None):
        m = method.currentText()
        for w, used in ((pct, m in ('percentile', 'sliding percentile')), (n_first, m == 'first frames'),
                        (window, m == 'sliding percentile')):
            w.setEnabled(used)

    method.currentTextChanged.connect(show_params)
    show_params()
    _buttons(dlg, form)
    if dlg.exec_() != QDialog.Accepted:
        return None
    image = image_box.layer()
    if image is None or axis_box.currentData() is None:
        _error('choose an image with a stack axis')
    if not _confirm_size(tiki, image.data.shape, 'ΔF/F'):
        return None
    m = method.currentText()
    params = {'percentile': pct.value(), 'n_first': n_first.value(), 'window': window.value()}
    with tiki.progress('ΔF/F', 1) as tick:
        result = ops.dff_movie(image.data, axis_box.currentData(), m, offset.value(), progress=tick,
                               return_f0=add_f0.isChecked(), **params)
    if tick.cancelled:
        tiki.status('ΔF/F cancelled')
        return None
    dff, f0 = result if add_f0.isChecked() else (result, None)
    finite = dff[np.isfinite(dff)]
    span = float(np.percentile(np.abs(finite), 99.5)) if finite.size else 1.0
    if f0 is not None and f0.shape == image.data.shape:
        _add_result(tiki, viewer, image, f0, f'{image.name} F0')
    elif f0 is not None:
        axis = axis_box.currentData()
        _add_result(tiki, viewer, image, np.squeeze(f0, axis), f'{image.name} F0',
                    axes=axes_mod.get(image).without(axis), drop=axis)
    layer = _add_result(tiki, viewer, image, dff, f'{image.name} ΔF/F ({m})', 'bwr', (-span, span))
    tiki.status(f'ΔF/F ({m}{", offset " + format(offset.value(), "g") if offset.value() else ""}) '
                f'→ "{layer.name}"; colour range ±{span:.3g}')
    return layer


# ------------------------------------------------------- bleach correction

def bleach_dialog(tiki) -> Image | None:
    from .layer_select import LayerCombo

    viewer, image = tiki.require_image()
    if image.ndim < 3:
        _error('bleach correction needs a movie or stack')
    dlg, form = _form(tiki, 'Bleach Correction')
    image_box = LayerCombo(viewer, Image, current=image)
    form.addRow('Image:', image_box)
    axis_box = _stack_axis_box(image, 'T', image_box)
    form.addRow('Along:', axis_box)
    method = QComboBox()
    method.addItem('Exponential fit', 'exponential')
    method.addItem('Simple ratio', 'ratio')
    method.setItemData(0, 'Divide by a fitted single-exponential decay. Refuses when there is no decay.',
                       Qt.ToolTipRole)
    method.setItemData(1, 'Scale every frame to the first frame\'s mean. Removes ALL changes in whole-frame '
                       'brightness, including real global signals; use with care.', Qt.ToolTipRole)
    form.addRow('Method:', method)
    background = QDoubleSpinBox(decimals=2, minimum=-1e9, maximum=1e9, value=0)
    background.setToolTip('Background intensity excluded from the correction (camera offset)')
    form.addRow('Background:', background)
    _buttons(dlg, form)
    if dlg.exec_() != QDialog.Accepted:
        return None
    image = image_box.layer()
    if not _confirm_size(tiki, image.data.shape, 'Bleach Correction'):
        return None
    axis = axis_box.currentData()
    with tiki.progress('Bleach Correction', image.data.shape[axis]) as tick:
        try:
            corrected, info = ops.bleach_correct(image.data, axis, method.currentData(), background.value(),
                                                 progress=tick)
        except ValueError as exc:
            means = [float(np.mean(np.take(image.data, k, axis=axis))) for k in range(image.data.shape[axis])]
            _plot_bleach(tiki, image.name, {'means': np.asarray(means)}, axes_mod.get(image).labels[axis])
            _error(str(exc))
    if tick.cancelled:
        tiki.status('Bleach correction cancelled')
        return None
    lo, hi = ops.saturated_limits(corrected[(0,) * (corrected.ndim - 2)])
    layer = _add_result(tiki, viewer, image, corrected, f'{image.name} bleach-corrected', image.colormap, (lo, hi))
    msg = f'Bleach correction ({method.currentText().lower()}) → "{layer.name}"'
    if 'tau_frames' in info:
        step = axes_mod.get(image).axes[axis]
        tau = info['tau_frames']
        msg += f'; τ = {tau:.3g} frames' + (f' = {tau * step.step:.3g} {step.unit}' if step.step else '')
        msg += f' (fit R² = {info["r2"]:.3f})'
    tiki.status(msg)
    _plot_bleach(tiki, image.name, info, axes_mod.get(image).labels[axis])
    return layer


def _plot_bleach(tiki, name, info, xlabel):
    from .widgets import PlotWindow

    win = PlotWindow(f'Bleaching — {name}', np.arange(len(info['means'])), info['means'], xlabel,
                     'mean intensity')
    if 'fit' in info:
        win.add_line(np.arange(len(info['fit'])), info['fit'], 'exponential fit')
    tiki.keep_window(win)
    win.show()


# --------------------------------------------------------------- kymograph

def kymograph_dialog(tiki):
    from .layer_select import LayerCombo

    viewer, image = tiki.require_image()
    current = tiki.current_roi(viewer)
    if current is None or current[1] not in ('line', 'path'):
        _error('draw a straight line or a path along the structure first (line tool)')
    if image.ndim < 3:
        _error('a kymograph needs a movie or stack')
    dlg, form = _form(tiki, 'Kymograph / Reslice')
    image_box = LayerCombo(viewer, Image, current=image)
    form.addRow('Image:', image_box)
    axis_box = _stack_axis_box(image, 'TZ', image_box)
    form.addRow('Along:', axis_box)
    width = QSpinBox(minimum=1, maximum=200, value=1)
    width.setSuffix(' px')
    width.setToolTip('Average this many pixels across the line')
    form.addRow('Line width:', width)
    _buttons(dlg, form)
    if dlg.exec_() != QDialog.Accepted:
        return None
    image = image_box.layer()
    axis = axis_box.currentData()
    info = axes_mod.get(image)
    index = roi_mod.current_index(viewer, image)
    data = image.data
    if image.rgb:
        _error('kymographs of RGB images are not supported; split or convert first')
    n_t = data.shape[axis]
    with tiki.progress('Kymograph', n_t) as tick:
        kymo = ops.kymograph(data, axis, index, current[0], width.value(), progress=tick)
    if tick.cancelled:
        return None
    a, y = info.axes[axis], info.axes[-1]
    rows = f'{info.labels[axis]}' + (f', 1 px = {a.step:g} {a.unit}' if a.step else '')
    cols = 'distance' + (f', 1 px = {y.step:g} {y.unit}' if y.step else '')
    k_info = axes_mod.AxesInfo([axes_mod.Axis('Y', kymo.shape[0]), axes_mod.Axis('X', kymo.shape[1])],
                               'kymograph', True, [f'rows: {rows}; columns: {cols}'])
    new = tiki.open_data(kymo, f'Kymo_{image.name}', axes=k_info)
    tiki.status(f'Kymograph: rows = {rows}; columns = {cols}')
    return new


# --------------------------------------------------------- image calculator

def _all_images(tiki):
    out = []
    for v in tiki.windows.viewers:
        for ly in v.layers:
            if isinstance(ly, Image) and ly.name not in roi_mod.PREVIEW_NAMES:
                out.append((f'{v.title} : {ly.name}' if ly.name != v.title else v.title, v, ly))
    return out


def calculator_dialog(tiki):
    images = _all_images(tiki)
    if not images:
        _error('no image open')
    dlg, form = _form(tiki, 'Image Calculator')
    box1, box2 = QComboBox(), QComboBox()
    for label, _, _ in images:
        box1.addItem(label)
        box2.addItem(label)
    _, active = tiki.windows.active_image()
    start = next((i for i, (_, _, ly) in enumerate(images) if ly is active), 0)
    box1.setCurrentIndex(start)
    box2.setCurrentIndex(min(start + 1, len(images) - 1))
    op = QComboBox()
    op.addItems(list(ops.CALC_OPS))
    note = QLabel('Result is 32-bit float in a new window. Image 2 may be a single plane '
                  '(applied to every plane of Image 1).')
    note.setWordWrap(True)
    form.addRow('Image 1:', box1)
    form.addRow('Operation:', op)
    form.addRow('Image 2:', box2)
    form.addRow(note)
    _buttons(dlg, form)
    if dlg.exec_() != QDialog.Accepted:
        return None
    (_, v1, a), (_, _, b) = images[box1.currentIndex()], images[box2.currentIndex()]
    if a.rgb or b.rgb:
        _error('Image Calculator works on greyscale images')
    sa, sb = a.data.shape, b.data.shape
    if sa[-len(sb):] != sb:
        _error(f'Image 2 {sb} must match the last axes of Image 1 {sa}\n'
               '(same size, or one plane/volume to apply to every position)')
    if not _confirm_size(tiki, sa, 'Image Calculator'):
        return None
    with tiki.progress('Image Calculator', int(np.prod(sa[:-2])) or 1) as tick:
        result = ops.image_calculator(a.data, b.data, op.currentText(), progress=tick)
    if tick.cancelled:
        return None
    return tiki.open_data(result, f'Result of {a.name}', axes=axes_mod.fit(axes_mod.get(a), result.shape))


# --------------------------------------------------------------- set scale

def set_scale_dialog(tiki):
    viewer, image = tiki.require_image()
    current = tiki.current_roi(viewer)
    pixels = 0.0
    if current is not None and current[1] in ('line', 'path'):
        verts = np.asarray(current[0])[:, -2:]
        pixels = float(np.linalg.norm(np.diff(verts, axis=0), axis=1).sum())
    info = axes_mod.get(image)
    y = info.axes[info.index('Y')]
    dlg, form = _form(tiki, 'Set Scale')
    dist_px = QDoubleSpinBox(decimals=2, minimum=0, maximum=1e9, value=pixels)
    known = QDoubleSpinBox(decimals=4, minimum=0, maximum=1e9, value=(pixels * y.step) if y.step and pixels else 0)
    unit = QComboBox()
    unit.setEditable(True)
    unit.addItems(['µm', 'nm', 'mm'])
    if y.unit:
        unit.setCurrentText(y.unit)
    form.addRow('Distance in pixels:', dist_px)
    form.addRow('Known distance:', known)
    form.addRow('Unit:', unit)
    hint = QLabel('Draw a line across a feature of known length first to fill in the pixels.'
                  if not pixels else f'From the current line: {pixels:.2f} px')
    hint.setWordWrap(True)
    form.addRow(hint)
    buttons = _buttons(dlg, form)

    def validate(_=None):
        buttons.button(QDialogButtonBox.Ok).setEnabled(dist_px.value() > 0 and known.value() > 0)

    dist_px.valueChanged.connect(validate)
    known.valueChanged.connect(validate)
    validate()
    if dlg.exec_() != QDialog.Accepted:
        return None
    step = known.value() / dist_px.value()
    for r in 'YX':
        a = info.axes[info.index(r)]
        a.step, a.unit = step, unit.currentText()
    info.source, info.confident = 'set by you', True
    axes_mod.apply(viewer, image, info)
    viewer.reset_view()
    tiki.status(f'Scale: {1 / step:.4g} pixels per {unit.currentText()} ({step:.4g} {unit.currentText()}/pixel)')
    return step
