"""Process > ΔF/F, Bleach Correction, Image Calculator; Stacks > Kymograph;
Analyze > Set Scale. Dialogs collect options; the maths lives in ``ops``.

Every command is non-destructive: results are new layers (same window) or
new windows, never an overwrite of the source data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from napari.layers import Image, Labels
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QSpinBox,
    QWidget,
)

from . import axes as axes_mod
from . import io, nd, ops, style
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
    offset_note = QLabel('')
    offset_note.setWordWrap(True)
    form.addRow(offset_note)

    def suggest_offset(layer=None):
        layer = layer or image_box.layer()
        if layer is None:
            return
        low = ops.suggested_offset(_sample(layer.data))
        offset.setValue(low)
        offset_note.setText(
            f'Values go below zero (minimum ≈ {low:g}), so the offset is set to the minimum: ΔF/F '
            'needs a positive F0. Change it if you know the true dark level.' if low < 0 else '')

    image_box.layer_changed.connect(suggest_offset)
    suggest_offset()
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


def _sample(data, n: int = 32) -> np.ndarray:
    """A few evenly spaced planes of a (possibly huge) array, for quick statistics."""
    lead = data.shape[:-2]
    total = int(np.prod(lead)) if lead else 1
    picks = np.unique(np.linspace(0, total - 1, min(n, total)).astype(int))
    return np.stack([np.asarray(data[np.unravel_index(i, lead)] if lead else data) for i in picks])


# ---------------------------------------------------------- correlation image

def correlation_image(tiki):
    viewer, image = tiki.require_image()
    info = axes_mod.get(image)
    axis = axes_mod.stack_axis(info, 'T')
    if axis is None or image.ndim < 3:
        _error('a correlation image needs a movie (a T axis)', fix='properties')
    if info.axes[axis].role != 'T':
        tiki.status(f'No T axis: correlating along {info.labels[axis]} (set T in Image ▸ Properties if wrong)')
    if image.rgb:
        _error('correlation images need a greyscale movie')
    index = roi_mod.current_index(viewer, image)
    with tiki.progress('Correlation Image', image.data.shape[-2]) as tick:
        cn = ops.correlation_image(image.data, axis, index, progress=tick)
    if tick.cancelled:
        return None
    keep = [i for i in range(len(info.axes)) if i >= image.ndim - 2]
    cn_info = axes_mod.AxesInfo([info.axes[i] for i in keep], info.source, True,
                                ['local correlation with 8 neighbours along ' + info.labels[axis]])
    new = tiki.open_data(cn, f'Cn_{image.name}', axes=cn_info)
    new.layers[0].contrast_limits = (0, max(0.1, float(np.percentile(cn, 99.9))))
    tiki.status(f'Correlation image of {image.name}: bright = pixels that co-fluctuate with their '
                'neighbours (active cells). Threshold it to find cells.')
    return new


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


# ------------------------------------------------------ duplicate & substack

def duplicate_dialog(tiki, substack_only: bool = False):
    from .layer_select import LayerCombo

    viewer, image = tiki.require_image()
    if substack_only and image.ndim < 3:
        _error('Make Substack needs a movie or stack, not a single image')

    dlg, form = _form(tiki, 'Make Substack' if substack_only else 'Duplicate')
    image_box = LayerCombo(viewer, Image, current=image)
    stem = Path(image.name).stem
    title_edit = QLineEdit(tiki.windows.unique_title(f'{stem}-1.tif'))
    box = tiki._roi_box(viewer, image)
    crop_box = QCheckBox('Crop to current selection', checked=box is not None)
    dup_stack = QCheckBox('Duplicate stack', checked=True)
    axes_container = QWidget()
    axes_form = QFormLayout(axes_container)
    axes_form.setContentsMargins(0, 0, 0, 0)
    note = QLabel('')
    note.setStyleSheet('color: #b00020;')
    note.setWordWrap(True)

    form.addRow('Image:', image_box)
    form.addRow('Title:', title_edit)
    form.addRow(crop_box)
    form.addRow(dup_stack)
    form.addRow(axes_container)
    form.addRow(note)
    buttons = _buttons(dlg, form)

    state: dict = {'has_box': box is not None, 'has_stack': image.ndim > 2, 'rows': []}

    def validate(_=None):
        img = image_box.layer()
        if img is None:
            buttons.button(QDialogButtonBox.Ok).setEnabled(False)
            return
        use_stack = substack_only or (state['has_stack'] and dup_stack.isChecked())
        for _d, edit, size in state['rows']:
            edit.setEnabled(use_stack)
        if not use_stack:
            note.setText('')
            buttons.button(QDialogButtonBox.Ok).setEnabled(True)
            return
        try:
            for _d, edit, size in state['rows']:
                ops.parse_slice_spec(edit.text(), size)
            note.setText('')
            buttons.button(QDialogButtonBox.Ok).setEnabled(True)
        except ValueError as exc:
            note.setText(str(exc))
            buttons.button(QDialogButtonBox.Ok).setEnabled(False)

    def rebuild_axes(_=None):
        while axes_form.rowCount():
            axes_form.removeRow(0)
        state['rows'] = []
        img = image_box.layer()
        if img is None:
            validate()
            return
        cur_box = tiki._roi_box(viewer, img)
        state['has_box'] = cur_box is not None
        crop_box.setVisible(cur_box is not None)
        crop_box.setChecked(cur_box is not None)
        state['has_stack'] = img.ndim > 2
        dup_stack.setVisible(state['has_stack'] and not substack_only)
        if substack_only:
            dup_stack.setChecked(True)
        info = axes_mod.get(img)
        for d in range(img.ndim - 2):
            n = img.data.shape[d]
            edit = QLineEdit(f'1-{n}')
            edit.setToolTip('1-based inclusive range (e.g. 1-50, 1-100-2, or 1, 4, 7-9)')
            edit.textChanged.connect(validate)
            axes_form.addRow(f'{info.labels[d]} range (1–{n}):', edit)
            state['rows'].append((d, edit, n))
        axes_container.setVisible(bool(state['rows']))
        validate()
        dlg.adjustSize()

    image_box.layer_changed.connect(rebuild_axes)
    dup_stack.toggled.connect(validate)
    rebuild_axes()

    if dlg.exec_() != QDialog.Accepted:
        return None
    img = image_box.layer()
    if img is None:
        return None
    use_stack = substack_only or (state['has_stack'] and dup_stack.isChecked())
    ranges = {d: edit.text() for d, edit, _ in state['rows']} if use_stack and state['rows'] else None
    return tiki.duplicate(
        image=img,
        title=title_edit.text().strip() or None,
        crop=state['has_box'] and crop_box.isChecked(),
        stack=use_stack,
        ranges=ranges,
    )


# ------------------------------------------------------------------ binning

def bin_dialog(tiki):
    from .layer_select import LayerCombo

    viewer, image = tiki.require_image()
    dlg, form = _form(tiki, 'Bin')
    image_box = LayerCombo(viewer, Image, current=image)
    xy_spin = QSpinBox(minimum=1, maximum=64, value=2)
    xy_spin.setToolTip('Binning factor applied to Y and X')
    stack_container = QWidget()
    stack_form = QFormLayout(stack_container)
    stack_form.setContentsMargins(0, 0, 0, 0)
    method_box = QComboBox()
    method_box.addItems(list(ops.BIN_METHODS))
    out_label = QLabel('')

    form.addRow('Image:', image_box)
    form.addRow('X / Y factor:', xy_spin)
    form.addRow(stack_container)
    form.addRow('Method:', method_box)
    form.addRow('Output size:', out_label)
    buttons = _buttons(dlg, form)

    state: dict = {'stack_spins': {}}

    def current_factors(img) -> tuple[int, ...]:
        factors = [1] * img.data.ndim
        spatial_end = img.data.ndim - 1 if img.rgb else img.data.ndim
        for d, spin in state['stack_spins'].items():
            factors[d] = spin.value()
        factors[spatial_end - 2] = xy_spin.value()
        factors[spatial_end - 1] = xy_spin.value()
        return tuple(factors)

    def update_summary(_=None):
        img = image_box.layer()
        if img is None:
            buttons.button(QDialogButtonBox.Ok).setEnabled(False)
            return
        factors = current_factors(img)
        out_shape = tuple(s // f for s, f in zip(img.data.shape, factors))
        ok = all(n >= 1 for n in out_shape)
        out_label.setText(f'{out_shape}' if ok else 'Factor exceeds image dimension')
        out_label.setStyleSheet('' if ok else 'color: #b00020;')
        buttons.button(QDialogButtonBox.Ok).setEnabled(ok)

    def rebuild_stack(_=None):
        while stack_form.rowCount():
            stack_form.removeRow(0)
        state['stack_spins'] = {}
        img = image_box.layer()
        if img is not None:
            info = axes_mod.get(img)
            for d in range(img.ndim - 2):
                if info.axes[d].role == 'C':
                    continue
                spin = QSpinBox(minimum=1, maximum=max(1, img.data.shape[d]), value=1)
                spin.valueChanged.connect(update_summary)
                stack_form.addRow(f'{info.labels[d]} factor:', spin)
                state['stack_spins'][d] = spin
        stack_container.setVisible(bool(state['stack_spins']))
        update_summary()
        dlg.adjustSize()

    image_box.layer_changed.connect(rebuild_stack)
    xy_spin.valueChanged.connect(update_summary)
    rebuild_stack()

    if dlg.exec_() != QDialog.Accepted:
        return None
    img = image_box.layer()
    if img is None:
        return None
    factors = current_factors(img)
    out_shape = tuple(s // f for s, f in zip(img.data.shape, factors))
    if not _confirm_size(tiki, out_shape, 'Bin'):
        return None
    method = ops.BIN_METHODS[method_box.currentText()]
    with tiki.progress('Bin', out_shape[0]) as tick:
        result = ops.bin_image(img.data, factors, method, progress=tick)
    if tick.cancelled:
        tiki.status('Bin cancelled')
        return None

    info = axes_mod.get(img)
    new_axes = [
        axes_mod.Axis(
            ax.role,
            result.shape[d],
            ax.step * factors[d] if ax.step else None,
            ax.unit if ax.step else None,
        )
        for d, ax in enumerate(info.axes)
    ]
    new_info = axes_mod.AxesInfo(new_axes, info.source, info.confident, list(info.notes))
    viewer_out = tiki.open_data(result, f'Binned_{img.name}', axes=new_info, colormap=img.colormap)
    tiki.status(f'Binned {img.name} ({method_box.currentText()}) → {result.shape}')
    return viewer_out


# ----------------------------------------------------------- image sequence

def sequence_dialog(tiki, folder: str | Path | None = None):
    if folder is None:
        folder = QFileDialog.getExistingDirectory(tiki, 'Import Image Sequence', str(Path.home()))
        if not folder:
            return None
    files = io.find_sequence_files(folder)
    if not files:
        _error(f'No supported image files found in {Path(folder).name}')

    dlg, form = _form(tiki, f'Import Image Sequence — {Path(folder).name}')
    pattern = QLineEdit()
    pattern.setPlaceholderText('optional substring or *.tif')
    start = QSpinBox(minimum=1, maximum=len(files), value=1)
    count = QSpinBox(minimum=1, maximum=len(files), value=len(files))
    step = QSpinBox(minimum=1, maximum=len(files), value=1)
    axis_box = QComboBox()
    axis_box.addItem('T (time)', 'T')
    axis_box.addItem('Z (depth)', 'Z')
    axis_box.addItem('? (unknown)', '?')
    step_val = QDoubleSpinBox(decimals=6, minimum=0, maximum=1e9, value=0)
    step_val.setSpecialValueText('uncalibrated')
    unit_box = QComboBox()
    unit_box.setEditable(True)
    unit_box.addItems(['s', 'ms', 'min', 'µm', 'nm', 'mm'])
    summary = QLabel('')

    form.addRow('Folder:', QLabel(f'{Path(folder).name} ({len(files)} files)'))
    form.addRow('File name contains:', pattern)
    form.addRow('First image:', start)
    form.addRow('Number of images:', count)
    form.addRow('Increment:', step)
    form.addRow('Stack axis:', axis_box)
    form.addRow('Step size:', step_val)
    form.addRow('Unit:', unit_box)
    form.addRow(summary)
    buttons = _buttons(dlg, form)

    def sync_unit(_=None):
        role = axis_box.currentData()
        if role == 'T':
            unit_box.setCurrentText('s')
        elif role == 'Z':
            unit_box.setCurrentText('µm')

    def update_count(_=None):
        matched = io.find_sequence_files(folder, pattern.text())
        sliced = matched[start.value() - 1 :: step.value()][: count.value()]
        summary.setText(f'Will open {len(sliced)} of {len(matched)} matching file(s) lazily'
                        + (f' (starting with {sliced[0].name})' if sliced else ''))
        buttons.button(QDialogButtonBox.Ok).setEnabled(len(sliced) > 0)

    axis_box.currentIndexChanged.connect(sync_unit)
    pattern.textChanged.connect(update_count)
    start.valueChanged.connect(update_count)
    count.valueChanged.connect(update_count)
    step.valueChanged.connect(update_count)
    sync_unit()
    update_count()

    if dlg.exec_() != QDialog.Accepted:
        return None
    return tiki.open_sequence(
        folder,
        pattern=pattern.text(),
        start=start.value(),
        count=count.value(),
        step=step.value(),
        axis_role=axis_box.currentData(),
        axis_step=step_val.value() or None,
        axis_unit=unit_box.currentText() or None,
    )


# --------------------------------------------------------- motion correction

def motion_correct_dialog(tiki):
    from .layer_select import LayerCombo
    from .widgets import PlotWindow

    viewer, image = tiki.require_image()
    if image.ndim < 3:
        _error('Motion Correction needs a movie or stack')
    if axes_mod.get(image).index('C') is not None:
        _error(
            f'{image.name} has a channel axis: split channels first, then register one channel '
            'and apply its shifts to the others',
            fix='split',
        )

    dlg, form = _form(tiki, 'Motion Correction (Rigid)')
    image_box = LayerCombo(viewer, Image, current=image)
    axis_box = _stack_axis_box(image, 'TZ', image_box)
    ref_box = QComboBox()
    ref_box.addItem('Median of sampled frames', 'median')
    ref_box.addItem('Mean of first N frames', 'first_mean')
    ref_box.addItem('Current frame', 'current')
    n_tmpl = QSpinBox(minimum=1, maximum=10000, value=200)
    up_box = QComboBox()
    up_box.addItem('1/10 pixel', 10)
    up_box.addItem('1/4 pixel', 4)
    up_box.addItem('Whole pixel (fast)', 1)
    max_shift = QDoubleSpinBox(decimals=1, minimum=1.0, maximum=1000.0, value=32.0)
    max_shift.setSuffix(' px')
    other_box = QComboBox()
    add_tmpl = QCheckBox('Also add template image', checked=False)

    form.addRow('Image:', image_box)
    form.addRow('Along:', axis_box)
    form.addRow('Reference:', ref_box)
    form.addRow('Template frames:', n_tmpl)
    form.addRow('Subpixel precision:', up_box)
    form.addRow('Max shift:', max_shift)
    form.addRow('Also apply shifts to:', other_box)
    form.addRow(add_tmpl)
    buttons = _buttons(dlg, form)

    def refresh_others(_=None):
        src = image_box.layer()
        ax = axis_box.currentData()
        other_box.clear()
        other_box.addItem('None', None)
        ok = src is not None and ax is not None and axes_mod.get(src).index('C') is None and not src.rgb
        buttons.button(QDialogButtonBox.Ok).setEnabled(ok)
        if not ok:
            return
        n_t = src.data.shape[ax]
        spatial = src.data.shape[-2:]
        for v, ly in ((vv, ll) for _, vv, ll in _all_images(tiki)):
            if ly is src or ly.rgb or ly.ndim < 3:
                continue
            if ly.data.shape[-2:] != spatial:
                continue
            o_ax = ax if ax < ly.ndim - 2 and ly.data.shape[ax] == n_t else None
            if o_ax is None:
                o_ax = next((d for d in range(ly.ndim - 2) if ly.data.shape[d] == n_t), None)
            if o_ax is not None:
                label = f'{v.title} : {ly.name}' if ly.name != v.title else ly.name
                other_box.addItem(label, (v, ly, o_ax))

    def toggle_tmpl(_=None):
        n_tmpl.setEnabled(ref_box.currentData() != 'current')

    image_box.layer_changed.connect(refresh_others)
    axis_box.currentIndexChanged.connect(refresh_others)
    ref_box.currentIndexChanged.connect(toggle_tmpl)
    refresh_others()
    toggle_tmpl()

    if dlg.exec_() != QDialog.Accepted:
        return None
    src = image_box.layer()
    ax = axis_box.currentData()
    if src is None or ax is None:
        _error('choose an image with a stack axis')
    if axes_mod.get(src).index('C') is not None:
        _error('split channels before running Motion Correction', fix='split')
    if src.rgb:
        _error('Motion Correction requires a greyscale movie')
    if not _confirm_size(tiki, src.data.shape, 'Motion Correction'):
        return None

    index = roi_mod.current_index(viewer, src)
    n_t = src.data.shape[ax]
    with tiki.progress('Motion Correction', 2 * n_t) as tick:
        corrected, mc_info = ops.motion_correct(
            src.data,
            axis=ax,
            template=ref_box.currentData(),
            n_template=n_tmpl.value(),
            ref_index=index[ax],
            upsample=up_box.currentData(),
            max_shift=max_shift.value(),
            index=index,
            progress=tick,
        )
    if tick.cancelled or mc_info.get('cancelled'):
        tiki.status('Motion correction cancelled')
        return None

    lo, hi = ops.saturated_limits(corrected[(0,) * (corrected.ndim - 2)])
    layer = _add_result(tiki, viewer, src, corrected, f'{src.name} registered', src.colormap, (lo, hi))
    if add_tmpl.isChecked():
        tmpl = mc_info['template']
        t_lo, t_hi = ops.saturated_limits(tmpl)
        src_info = axes_mod.get(src)
        t_info = axes_mod.AxesInfo([axes_mod.Axis(a.role, a.size, a.step, a.unit) for a in src_info.axes[-2:]],
                                   'motion template', True)
        t_layer = Image(tmpl, name=f'{src.name} template', colormap=src.colormap, contrast_limits=(t_lo, t_hi),
                        metadata={'axes': t_info.to_dict()}, **axes_mod.geometry(src, ndim=2))
        viewer.add_layer(t_layer)
        tiki.reapply_tool(viewer)

    shifts = mc_info['shifts']
    other = other_box.currentData()
    if other is not None:
        other_v, other_ly, o_ax = other
        with tiki.progress(f'Applying shifts to {other_ly.name}', n_t) as tick2:
            other_reg = ops.apply_shifts(other_ly.data, shifts, axis=o_ax, progress=tick2)
        if not tick2.cancelled:
            o_lo, o_hi = ops.saturated_limits(other_reg[(0,) * (other_reg.ndim - 2)])
            _add_result(tiki, other_v, other_ly, other_reg, f'{other_ly.name} registered', other_ly.colormap, (o_lo, o_hi))

    src_info = axes_mod.get(src)
    y_ax = src_info.axes[src_info.index('Y')]
    unit = y_ax.unit if y_ax.step and y_ax.unit else 'px'
    scale_y = float(src.scale[-2]) if y_ax.step else 1.0
    scale_x = float(src.scale[-1]) if src_info.axes[src_info.index('X')].step else 1.0
    x_vals = np.arange(n_t)
    dy_vals = shifts[:, 0] * scale_y
    dx_vals = shifts[:, 1] * scale_x
    plot = PlotWindow(f'Motion shifts — {src.name}', x_vals, dy_vals, src_info.labels[ax], f'shift ({unit})')
    plot.ax.lines[0].set_label('dy')
    plot.add_line(x_vals, dx_vals, 'dx')
    tiki.keep_window(plot)
    plot.show()

    max_y, max_x = np.abs(shifts).max(axis=0)
    tiki.status(f'Motion Correction → "{layer.name}"; max |shift| dy={max_y:.2f} px, dx={max_x:.2f} px')
    return layer


# ------------------------------------------------------- labels & watershed

def _labels_info(viewer, layer: Labels) -> axes_mod.AxesInfo:
    if layer.metadata.get('axes'):
        return axes_mod.get(layer)
    shape = tuple(layer.data.shape[: layer.ndim])
    ref = roi_mod.target_image(viewer)
    if ref is not None:
        ref_info = axes_mod.get(ref)
        if len(ref_info.axes) >= layer.ndim:
            tail = axes_mod.AxesInfo(
                [axes_mod.Axis(a.role, n, a.step, a.unit) for a, n in zip(ref_info.axes[-layer.ndim :], shape)],
                ref_info.source,
                ref_info.confident,
            )
            if not tail.problems(shape):
                return tail
    return axes_mod.guess(shape)


def _labels_z_ratio(viewer, layer: Labels) -> float:
    info = _labels_info(viewer, layer)
    z, y = info.index('Z'), info.index('Y')
    if z is not None and y is not None:
        sz, sy = info.axes[z].step, info.axes[y].step
        if sz and sy:
            return float(sy / sz)
    return 1.0


class WatershedDialog(QDialog):
    PREVIEW_NAME = 'watershed preview'

    def __init__(self, tiki, viewer, layer: Labels):
        from .layer_select import LayerCombo

        super().__init__(tiki)
        style.apply(self)
        self.tiki, self.viewer, self.layer = tiki, viewer, layer
        self.preview: Labels | None = None
        self.has_z = False
        self.has_stack = False
        self.setWindowTitle('Watershed (Labels)')

        form = QFormLayout(self)
        self.labels_box = LayerCombo(viewer, Labels, current=layer)
        self.labels_box.layer_changed.connect(self._set_layer)
        self.scope = QComboBox()
        self.three_d = QCheckBox('3D (split through Z, scaled by voxel size)', checked=True)
        self.min_dist = QDoubleSpinBox(decimals=1, minimum=1.0, maximum=500.0, singleStep=1.0, value=5.0)
        self.min_dist.setSuffix(' px')
        self.min_dist.setKeyboardTracking(False)
        self.fill_holes = QCheckBox('Fill holes before splitting', checked=True)
        self.show_preview = QCheckBox('Preview (current slice)', checked=True)

        form.addRow('Labels:', self.labels_box)
        form.addRow('Apply to:', self.scope)
        form.addRow(self.three_d)
        form.addRow('Min peak distance:', self.min_dist)
        form.addRow(self.fill_holes)
        form.addRow(self.show_preview)
        buttons = _buttons(self, form)
        self._ok = buttons.button(QDialogButtonBox.Ok)

        self._timer = QTimer(self, singleShot=True, interval=120)
        self._timer.timeout.connect(self.update_preview)

        self.scope.currentIndexChanged.connect(lambda _: self._update_3d_enabled())
        self.min_dist.valueChanged.connect(lambda _: self.schedule())
        self.fill_holes.toggled.connect(lambda _: self.schedule())
        self.show_preview.toggled.connect(lambda on: self.update_preview() if on else self._remove_preview())
        viewer.dims.events.current_step.connect(self.schedule)

        self._update_options()
        self.update_preview()

    def _update_options(self) -> None:
        self._ok.setEnabled(self.layer is not None)
        if self.layer is None:
            return
        info = _labels_info(self.viewer, self.layer)
        self.has_z = info.index('Z') is not None
        self.has_stack = self.layer.ndim > 2
        self.scope.blockSignals(True)
        self.scope.clear()
        for key, label in nd.scopes(info):
            self.scope.addItem(label, key)
        keys = [self.scope.itemData(i) for i in range(self.scope.count())]
        default = nd.WHOLE if nd.WHOLE in keys else nd.PLANE
        self.scope.setCurrentIndex(keys.index(default))
        self.scope.blockSignals(False)
        self.scope.setVisible(self.has_stack)
        self.three_d.setVisible(self.has_z)
        self._update_3d_enabled()

    def _update_3d_enabled(self) -> None:
        self.three_d.setEnabled(self.has_z and self.scope_key() != nd.PLANE)

    def scope_key(self) -> str:
        return self.scope.currentData() if self.has_stack else nd.PLANE

    def use_3d(self) -> bool:
        return self.has_z and self.three_d.isChecked() and self.scope_key() != nd.PLANE

    def _set_layer(self, layer) -> None:
        if layer is None or layer is self.layer:
            return
        self._remove_preview()
        self.layer = layer
        self._update_options()
        self.schedule()

    def schedule(self, event=None) -> None:
        if self.show_preview.isChecked():
            self._timer.start()

    def update_preview(self) -> None:
        if not self.show_preview.isChecked() or self.layer is None:
            return
        plane = self.tiki.labels_plane(self.viewer, self.layer)
        result = ops.watershed_labels(plane, min_distance=self.min_dist.value(), fill_holes=self.fill_holes.isChecked())
        if self.preview is None or self.preview not in self.viewer.layers:
            selected = self.viewer.layers.selection.active
            self.preview = Labels(
                result,
                name=self.PREVIEW_NAME,
                opacity=0.6,
                **axes_mod.geometry(self.layer, ndim=2),
            )
            self.viewer.add_layer(self.preview)
            if selected is not None and selected in self.viewer.layers:
                self.viewer.layers.selection.active = selected
            roi = roi_mod.roi_layer(self.viewer, create=False)
            if roi is not None:
                self.viewer.layers.move(self.viewer.layers.index(roi), len(self.viewer.layers))
        else:
            self.preview.data = result

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
        self.labels_box.blockSignals(True)
        self.labels_box.set_viewer(None)
        self.labels_box.blockSignals(False)

    def done(self, result: int) -> None:
        self.cleanup()
        super().done(result)


def watershed_dialog(tiki):
    viewer, _ = tiki.require_image()
    layer = tiki.labels_layer(viewer)
    dlg = WatershedDialog(tiki, viewer, layer)
    try:
        accepted = dlg.exec_() == QDialog.Accepted
    finally:
        dlg.cleanup()
    if not accepted or dlg.layer is None:
        return None

    layer = dlg.layer
    info = _labels_info(viewer, layer)
    scope = dlg.scope_key()
    by_volume = dlg.use_3d()
    min_dist = dlg.min_dist.value()
    fill = dlg.fill_holes.isChecked()
    zr = _labels_z_ratio(viewer, layer)
    shape = tuple(layer.data.shape[: layer.ndim])

    ref = roi_mod.target_image(viewer)
    if ref is not None and len(axes_mod.get(ref).axes) >= layer.ndim:
        full_idx = roi_mod.current_index(viewer, ref)
        index = full_idx[-layer.ndim :]
    else:
        point = layer.world_to_data(viewer.dims.point[-layer.ndim :])
        index = tuple(int(np.clip(np.round(point[d]), 0, shape[d] - 1)) for d in range(layer.ndim))

    if scope == nd.PLANE or layer.ndim == 2:
        out = ops.watershed_labels(tiki.labels_plane(viewer, layer), min_distance=min_dist, fill_holes=fill)
        geometry = axes_mod.geometry(layer, ndim=2)
    elif scope == nd.VOLUME:
        (block,) = nd.blocks(info, shape, index, nd.VOLUME, False)
        vol = np.asarray(layer.data[block])
        if by_volume:
            out = ops.watershed_labels(vol, min_distance=min_dist, z_ratio=zr, fill_holes=fill)
        else:
            planes = [(k, slice(None), slice(None)) for k in range(vol.shape[0])]
            out = ops.watershed_blocks(vol, planes, min_distance=min_dist, fill_holes=fill)
        geometry = axes_mod.geometry(layer, ndim=3)
    else:
        blocks = nd.blocks(info, shape, index, nd.WHOLE, by_volume)
        with tiki.progress('Watershed', len(blocks)) as tick:
            out = ops.watershed_blocks(layer.data, blocks, min_distance=min_dist, z_ratio=zr,
                                       fill_holes=fill, progress=tick)
        if tick.cancelled:
            tiki.status('Watershed cancelled')
            return None
        geometry = axes_mod.geometry(layer)

    new_layer = Labels(out, name=f'{layer.name} watershed', opacity=0.6,
                       metadata={'axes': info.to_dict()} if layer.ndim > 2 else {}, **geometry)
    viewer.add_layer(new_layer)
    viewer.layers.selection.active = new_layer
    roi = roi_mod.roi_layer(viewer, create=False)
    if roi is not None:
        viewer.layers.move(viewer.layers.index(roi), len(viewer.layers))
    n_after = len(np.unique(out)) - (1 if (out == 0).any() else 0)
    tiki.status(f'Watershed: {n_after} objects → "{new_layer.name}"')
    return new_layer


def fill_holes_command(tiki):
    viewer, _ = tiki.require_image()
    layer = tiki.labels_layer(viewer)
    info = _labels_info(viewer, layer)
    if layer.ndim == 2:
        out = ops.fill_holes_labels(np.asarray(layer.data))
    else:
        total = int(np.prod(layer.data.shape[:-2]))
        with tiki.progress('Fill Holes', total) as tick:
            out = ops.map_planes(ops.fill_holes_labels, layer.data, progress=tick)
        if tick.cancelled:
            tiki.status('Fill Holes cancelled')
            return None
    new_layer = Labels(out, name=f'{layer.name} filled', opacity=0.6,
                       metadata={'axes': info.to_dict()} if layer.ndim > 2 else {}, **axes_mod.geometry(layer))
    viewer.add_layer(new_layer)
    viewer.layers.selection.active = new_layer
    roi = roi_mod.roi_layer(viewer, create=False)
    if roi is not None:
        viewer.layers.move(viewer.layers.index(roi), len(viewer.layers))
    tiki.status(f'Filled holes → "{new_layer.name}"')
    return new_layer


# ----------------------------------------- selections, transforms, formats

def _require_area_roi(tiki):
    viewer, image = tiki.require_image()
    current = tiki.current_roi(viewer)
    if current is None or current[1] in ('line', 'path'):
        _error('draw an area selection (rectangle, oval, polygon or freehand) first')
    return viewer, image, current


def enlarge_dialog(tiki):
    viewer, image, current = _require_area_roi(tiki)
    dlg, form = _form(tiki, 'Enlarge / Shrink Selection')
    dist = QDoubleSpinBox(decimals=1, minimum=-500.0, maximum=500.0, singleStep=1.0, value=5.0)
    dist.setSuffix(' px')
    dist.setToolTip('Positive values enlarge the selection; negative values shrink it')
    form.addRow('Enlarge by:', dist)
    _buttons(dlg, form)
    if dlg.exec_() != QDialog.Accepted:
        return None
    shape2d = roi_mod.spatial_shape(image)
    mask = roi_mod.shape_mask(*current, shape2d)
    new_mask = ops.enlarge_mask(mask, dist.value())
    poly = ops.mask_to_polygon(new_mask)
    if poly is None:
        _error('selection vanished after shrinking')
    tiki._last_roi = (np.asarray(current[0]).copy(), current[1])
    layer = tiki.roi_layer(viewer)
    layer.data = []
    layer.add_polygons([poly])
    tiki.status(f'Selection enlarged by {dist.value():+g} px ({int(new_mask.sum())} px)')
    return poly


def make_band_dialog(tiki):
    viewer, image, current = _require_area_roi(tiki)
    dlg, form = _form(tiki, 'Make Band (Annulus)')
    width = QDoubleSpinBox(decimals=1, minimum=0.5, maximum=500.0, singleStep=1.0, value=5.0)
    width.setSuffix(' px')
    gap = QDoubleSpinBox(decimals=1, minimum=0.0, maximum=200.0, singleStep=1.0, value=0.0)
    gap.setSuffix(' px')
    gap.setToolTip('Optional gap between the ROI boundary and the inner edge of the band')
    form.addRow('Band width:', width)
    form.addRow('Gap from ROI:', gap)
    _buttons(dlg, form)
    if dlg.exec_() != QDialog.Accepted:
        return None
    shape2d = roi_mod.spatial_shape(image)
    mask = roi_mod.shape_mask(*current, shape2d)
    band = ops.make_band_mask(mask, width.value(), gap.value())
    if not band.any():
        _error('band is empty')
    name = f'band ({width.value():g} px)' if gap.value() == 0 else f'band ({width.value():g} px, gap {gap.value():g})'
    layer = Labels(band.astype(np.uint8), name=name, opacity=0.6, **axes_mod.geometry(image, ndim=2))
    viewer.add_layer(layer)
    tiki.reapply_tool(viewer)
    tiki.status(f'Created "{name}" Labels layer ({int(band.sum())} px)')
    return layer


def clear_selection(tiki, outside: bool = False):
    viewer, image, current = _require_area_roi(tiki)
    shape2d = roi_mod.spatial_shape(image)
    mask = roi_mod.shape_mask(*current, shape2d)
    index = roi_mod.current_index(viewer, image)
    whole = image.ndim > 2
    if whole and not _confirm_size(tiki, image.data.shape, 'Clear Outside' if outside else 'Clear'):
        return None
    total = int(np.prod(image.data.shape[: image.ndim - 2])) if whole else 1
    with tiki.progress('Clear Outside' if outside else 'Clear', total) as tick:
        out = ops.clear_mask(
            image.data,
            mask,
            outside=outside,
            fill_value=0.0,
            index=index,
            whole_stack=whole,
            rgb=bool(image.rgb),
            progress=tick,
        )
    if tick.cancelled:
        tiki.status('Clear cancelled')
        return None
    suffix = 'cleared outside' if outside else 'cleared'
    layer = _add_result(tiki, viewer, image, out, f'{image.name} {suffix}', image.colormap,
                        tuple(map(float, image.contrast_limits)) if not image.rgb else None)
    tiki.status(f'{"Clear Outside" if outside else "Clear"} → "{layer.name}"')
    return layer


def rotate_dialog(tiki):
    viewer, image = tiki.require_image()
    dlg, form = _form(tiki, 'Rotate')
    angle = QDoubleSpinBox(decimals=2, minimum=-360.0, maximum=360.0, singleStep=5.0, value=15.0)
    angle.setSuffix('°')
    angle.setToolTip('Clockwise rotation angle in degrees')
    enlarge = QCheckBox('Enlarge image to fit result', checked=True)
    form.addRow('Angle:', angle)
    form.addRow(enlarge)
    _buttons(dlg, form)
    if dlg.exec_() != QDialog.Accepted:
        return None
    if not _confirm_size(tiki, image.data.shape, 'Rotate'):
        return None
    total = int(np.prod(image.data.shape[: image.ndim - 2])) if image.ndim > 2 else 1
    with tiki.progress('Rotate', total) as tick:
        out = ops.rotate_angle(image.data, angle.value(), resize=enlarge.isChecked(),
                               rgb=bool(image.rgb), progress=tick)
    if tick.cancelled:
        tiki.status('Rotate cancelled')
        return None
    info = axes_mod.fit(axes_mod.get(image), out.shape[:-1] if image.rgb else out.shape)
    kw: dict = {'axes': info}
    if not image.rgb:
        kw['colormap'] = image.colormap
    v = tiki.open_data(out, f'Rot{angle.value():g}_{image.name}', **kw)
    tiki.status(f'Rotated {image.name} by {angle.value():g}°')
    return v


def set_measurements_dialog(tiki):
    from qtpy.QtWidgets import QGridLayout

    dlg = QDialog(tiki)
    style.apply(dlg)
    dlg.setWindowTitle('Set Measurements')
    grid = QGridLayout(dlg)
    boxes: dict[str, QCheckBox] = {}
    cols = 2
    for i, key in enumerate(ops.MEASUREMENT_KEYS):
        cb = QCheckBox(key, checked=key in tiki.measurements)
        boxes[key] = cb
        grid.addWidget(cb, i // cols, i % cols)
    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)
    grid.addWidget(buttons, (len(ops.MEASUREMENT_KEYS) + cols - 1) // cols, 0, 1, cols)
    if dlg.exec_() != QDialog.Accepted:
        return None
    chosen = {k for k, cb in boxes.items() if cb.isChecked()}
    if not chosen:
        chosen = {'Area', 'Mean'}
    tiki.measurements = chosen
    tiki.status(f'Set Measurements: {", ".join(k for k in ops.MEASUREMENT_KEYS if k in chosen)}')
    return chosen


def open_zarr_dialog(tiki):
    folder = QFileDialog.getExistingDirectory(tiki, 'Open OME-Zarr (.zarr)', str(Path.home()))
    if not folder:
        return None
    tiki.open_paths([folder])
    return tiki.windows.active
