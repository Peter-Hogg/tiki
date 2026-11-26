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
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
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
    ax_info = axes_mod.get(image)
    ax_meta = ax_info.axes[axis]
    with tiki.progress('Bleach Correction', image.data.shape[axis]) as tick:
        try:
            corrected, info = ops.bleach_correct(image.data, axis, method.currentData(), background.value(),
                                                 progress=tick)
        except ValueError as exc:
            means = [float(np.mean(np.take(image.data, k, axis=axis))) for k in range(image.data.shape[axis])]
            _plot_bleach(
                tiki,
                image.name,
                {'means': np.asarray(means)},
                ax_info.labels[axis],
                step=ax_meta.step,
                unit=ax_meta.unit,
            )
            _error(str(exc))
    if tick.cancelled:
        tiki.status('Bleach correction cancelled')
        return None
    lo, hi = ops.saturated_limits(corrected[(0,) * (corrected.ndim - 2)])
    layer = _add_result(tiki, viewer, image, corrected, f'{image.name} bleach-corrected', image.colormap, (lo, hi))
    msg = f'Bleach correction ({method.currentText().lower()}) → "{layer.name}"'
    if 'tau_frames' in info:
        tau = info['tau_frames']
        msg += f'; τ = {tau:.3g} frames' + (f' = {tau * ax_meta.step:.3g} {ax_meta.unit}' if ax_meta.step else '')
        msg += f' (fit R² = {info["r2"]:.3f})'
    tiki.status(msg)
    _plot_bleach(tiki, image.name, info, ax_info.labels[axis], step=ax_meta.step, unit=ax_meta.unit)
    return layer


def _plot_bleach(tiki, name, info, xlabel, step: float | None = None, unit: str | None = None):
    from .widgets import PlotWindow

    scale_x = float(step) if (step and step > 0) else 1.0
    x = np.arange(len(info['means'])) * scale_x
    label_x = f'{xlabel} ({unit})' if (step and unit) else xlabel
    win = PlotWindow(f'Bleaching — {name}', x, info['means'], label_x, 'mean intensity')
    if 'fit' in info:
        win.add_line(x, info['fit'], 'exponential fit')
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
    a, y, x_ax = info.axes[axis], info.axes[-2], info.axes[-1]
    rows = f'{info.labels[axis]}' + (f', 1 px = {a.step:g} {a.unit}' if a.step else '')
    cols = 'distance' + (f', 1 px = {x_ax.step:g} {x_ax.unit}' if x_ax.step else '')
    k_y_step = None
    k_y_unit = None
    if a.role == 'Z' and a.step:
        k_y_unit = x_ax.unit or y.unit or a.unit
        k_y_step = axes_mod.convert_unit(a.step, a.unit, k_y_unit) if (a.unit and k_y_unit) else float(a.step)
    k_x_step = x_ax.step if x_ax.step else None
    k_x_unit = x_ax.unit if x_ax.step else None
    k_info = axes_mod.AxesInfo(
        [
            axes_mod.Axis('Y', kymo.shape[0], k_y_step, k_y_unit if k_y_step else None),
            axes_mod.Axis('X', kymo.shape[1], k_x_step, k_x_unit),
        ],
        'kymograph',
        True,
        [f'rows: {rows}; columns: {cols}'],
    )
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

def _xy_step_info(image) -> tuple[float, str | None]:
    info = axes_mod.get(image)
    u = axes_mod.spatial_unit(info)
    if not u:
        return 1.0, None
    sy = float(info.axes[-2].step or 1.0)
    sx = float(info.axes[-1].step or sy)
    return 0.5 * (sy + sx), u


def set_scale_dialog(tiki):
    viewer, image = tiki.require_image()
    current = tiki.current_roi(viewer)
    pixels = 0.0
    if current is not None and current[1] in ('line', 'path'):
        verts = np.asarray(current[0])[:, -2:]
        pixels = float(np.linalg.norm(np.diff(verts, axis=0), axis=1).sum())
    info = axes_mod.get(image)
    y = info.axes[info.index('Y')]
    x_ax = info.axes[info.index('X')]
    cur_aspect = (float(y.step) / float(x_ax.step)) if (y.step and x_ax.step and x_ax.step > 0) else 1.0

    dlg, form = _form(tiki, 'Set Scale')
    dist_px = QDoubleSpinBox(decimals=2, minimum=0, maximum=1e9, value=pixels)
    known = QDoubleSpinBox(decimals=4, minimum=0, maximum=1e9, value=(pixels * y.step) if y.step and pixels else 0)
    aspect = QDoubleSpinBox(decimals=4, minimum=0.001, maximum=1000.0, value=cur_aspect)
    aspect.setToolTip('Ratio of Y pixel size to X pixel size (1.0 for square pixels)')
    unit = QComboBox()
    unit.setEditable(True)
    unit.addItems(['µm', 'nm', 'mm'])
    if y.unit:
        unit.setCurrentText(y.unit)
    global_box = QCheckBox('Global (apply to all open windows)')
    summary = QLabel('')
    hint = QLabel('Draw a line across a feature of known length first to fill in the pixels.'
                  if not pixels else f'From the current line: {pixels:.2f} px')
    hint.setWordWrap(True)

    form.addRow('Distance in pixels:', dist_px)
    form.addRow('Known distance:', known)
    form.addRow('Pixel aspect ratio (Y/X):', aspect)
    form.addRow('Unit:', unit)
    form.addRow(global_box)
    form.addRow(summary)
    form.addRow(hint)

    btn_row = QHBoxLayout()
    remove_btn = QPushButton('Remove Scale')
    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)
    btn_row.addWidget(remove_btn)
    btn_row.addWidget(buttons)
    form.addRow(btn_row)

    removed = {'flag': False}

    def _on_remove():
        removed['flag'] = True
        dlg.accept()

    remove_btn.clicked.connect(_on_remove)

    def validate(_=None):
        ok = dist_px.value() > 0 and known.value() > 0
        buttons.button(QDialogButtonBox.Ok).setEnabled(ok)
        if ok:
            sx_val = known.value() / dist_px.value()
            u_val = unit.currentText() or 'µm'
            summary.setText(f'Scale: {1 / sx_val:.4g} px/{u_val} ({sx_val:.4g} {u_val}/px)')
        else:
            summary.setText('')

    dist_px.valueChanged.connect(validate)
    known.valueChanged.connect(validate)
    aspect.valueChanged.connect(validate)
    unit.currentTextChanged.connect(validate)
    validate()
    if dlg.exec_() != QDialog.Accepted:
        return None

    targets = (
        [
            (v, ly)
            for v in tiki.windows.viewers
            for ly in v.layers
            if isinstance(ly, Image) and ly.name not in roi_mod.PREVIEW_NAMES
        ]
        if global_box.isChecked()
        else [(viewer, image)]
    )

    if removed['flag']:
        for v, ly in targets:
            ly_info = axes_mod.get(ly)
            for r in 'YX':
                idx = ly_info.index(r)
                if idx is not None:
                    ly_info.axes[idx].step = None
                    ly_info.axes[idx].unit = None
            ly_info.source, ly_info.confident = 'uncalibrated', True
            axes_mod.apply(v, ly, ly_info)
            v.reset_view()
        tiki.status('Scale removed')
        return 0.0

    sx_step = known.value() / dist_px.value()
    sy_step = sx_step * aspect.value()
    u_txt = unit.currentText() or 'µm'
    for v, ly in targets:
        ly_info = axes_mod.get(ly)
        y_i, x_i = ly_info.index('Y'), ly_info.index('X')
        if y_i is not None:
            ly_info.axes[y_i].step, ly_info.axes[y_i].unit = sy_step, u_txt
        if x_i is not None:
            ly_info.axes[x_i].step, ly_info.axes[x_i].unit = sx_step, u_txt
        ly_info.source, ly_info.confident = 'set by you', True
        axes_mod.apply(v, ly, ly_info)
        v.reset_view()
    tiki.status(f'Scale: {1 / sx_step:.4g} pixels per {u_txt} ({sx_step:.4g} {u_txt}/pixel)')
    return sx_step


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
    ax_obj = src_info.axes[ax]
    x_step = float(ax_obj.step) if (ax_obj.step and ax_obj.step > 0) else 1.0
    x_unit = ax_obj.unit if (ax_obj.step and ax_obj.unit) else None
    x_vals = np.arange(n_t) * x_step
    x_label = f'{src_info.labels[ax]} ({x_unit})' if x_unit else src_info.labels[ax]
    dy_vals = shifts[:, 0] * scale_y
    dx_vals = shifts[:, 1] * scale_x
    plot = PlotWindow(f'Motion shifts — {src.name}', x_vals, dy_vals, x_label, f'shift ({unit})')
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
        self.phys_label = QLabel('')
        self.fill_holes = QCheckBox('Fill holes before splitting', checked=True)
        self.show_preview = QCheckBox('Preview (current slice)', checked=True)

        form.addRow('Labels:', self.labels_box)
        form.addRow('Apply to:', self.scope)
        form.addRow(self.three_d)
        form.addRow('Min peak distance:', self.min_dist)
        form.addRow('', self.phys_label)
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

    def _update_phys_label(self) -> None:
        if self.layer is None:
            self.phys_label.setText('')
            return
        info = _labels_info(self.viewer, self.layer)
        u = axes_mod.spatial_unit(info)
        if not u:
            self.phys_label.setText('')
            return
        sy = float(info.axes[-2].step or 1.0)
        sx = float(info.axes[-1].step or sy)
        self.phys_label.setText(f'≈ {self.min_dist.value() * 0.5 * (sy + sx):.4g} {u}')

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
        self._update_phys_label()

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
        self._update_phys_label()
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
    phys = QLabel('')
    step_xy, u = _xy_step_info(image)

    def _sync_phys(_=None):
        phys.setText(f'≈ {dist.value() * step_xy:+g} {u}' if u else '')

    dist.valueChanged.connect(_sync_phys)
    _sync_phys()
    form.addRow('Enlarge by:', dist)
    if u:
        form.addRow('', phys)
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
    phys = QLabel('')
    step_xy, u = _xy_step_info(image)

    def _sync_phys(_=None):
        if u:
            phys.setText(f'width ≈ {width.value() * step_xy:.4g} {u}, gap ≈ {gap.value() * step_xy:.4g} {u}')
        else:
            phys.setText('')

    width.valueChanged.connect(_sync_phys)
    gap.valueChanged.connect(_sync_phys)
    _sync_phys()
    form.addRow('Band width:', width)
    form.addRow('Gap from ROI:', gap)
    if u:
        form.addRow('', phys)
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


def stack_to_hyperstack_dialog(tiki):
    """``Image ▸ Stacks ▸ Stack to Hyperstack…``: reshape a flat stack into C / Z / T axes."""
    _viewer, image = tiki.require_image()
    spatial_ndim = 3 if image.rgb else 2
    if image.data.ndim < spatial_ndim + 1:
        _error('Need a stack (3D or higher) to convert to a hyperstack.')
    n_total = int(np.prod(image.data.shape[:-spatial_ndim]))

    dlg, form = _form(tiki, 'Stack to Hyperstack')
    order_box = QComboBox()
    for label, code in (
        ('xytzc — channel blocks [1..N, 1..N]', 'xytzc'),
        ('xyczt — interleaved channels [c1, c2, c1, c2, …]', 'xyczt'),
        ('xyztc — Z blocks, then T, then C', 'xyztc'),
        ('xyzct — Z blocks per channel, then T', 'xyzct'),
        ('xytcz — T blocks per channel, then Z', 'xytcz'),
        ('xyctz — interleaved channels, Z slowest', 'xyctz'),
    ):
        order_box.addItem(label, code)

    default_c = 2 if n_total % 2 == 0 and n_total >= 2 else 1
    c_spin = QSpinBox(minimum=1, maximum=n_total, value=default_c)
    z_spin = QSpinBox(minimum=1, maximum=n_total, value=1)
    t_spin = QSpinBox(minimum=1, maximum=n_total, value=n_total // default_c)
    check_label = QLabel()
    split_box = QCheckBox('Split channels immediately')
    split_box.setChecked(default_c > 1)

    form.addRow('Order:', order_box)
    form.addRow('Channels (c):', c_spin)
    form.addRow('Slices (z):', z_spin)
    form.addRow('Frames (t):', t_spin)
    form.addRow('', check_label)
    form.addRow(split_box)
    buttons = _buttons(dlg, form)
    ok_btn = buttons.button(QDialogButtonBox.Ok)

    updating = {'busy': False}

    def _refresh_status():
        c, z, t = c_spin.value(), z_spin.value(), t_spin.value()
        prod = c * z * t
        ok = prod == n_total
        if ok:
            check_label.setText(f'{c} × {z} × {t} = {prod} planes  ✓')
            check_label.setStyleSheet('color: #2e7d32;')
        else:
            check_label.setText(f'{c} × {z} × {t} = {prod}  (must equal {n_total})')
            check_label.setStyleSheet('color: #c62828;')
        if ok_btn is not None:
            ok_btn.setEnabled(ok)
        split_box.setEnabled(c > 1)

    def _auto_frames():
        if updating['busy']:
            return
        c, z = c_spin.value(), z_spin.value()
        if c * z > 0 and n_total % (c * z) == 0:
            updating['busy'] = True
            t_spin.setValue(n_total // (c * z))
            updating['busy'] = False
        split_box.setChecked(c > 1)
        _refresh_status()

    c_spin.valueChanged.connect(lambda _: _auto_frames())
    z_spin.valueChanged.connect(lambda _: _auto_frames())
    t_spin.valueChanged.connect(lambda _: _refresh_status())
    _refresh_status()

    if dlg.exec_() != QDialog.Accepted:
        return None

    c, z, t = c_spin.value(), z_spin.value(), t_spin.value()
    order = order_box.currentData()
    try:
        reshaped, roles = ops.stack_to_hyperstack(image.data, c, z, t, order=order, rgb=bool(image.rgb))
    except ValueError as exc:
        _error(str(exc))

    src_info = axes_mod.get(image)
    ay, ax_x = src_info.axes[-2], src_info.axes[-1]
    new_axes = []
    for dim, role in enumerate(roles):
        size = int(reshaped.shape[dim])
        if role == 'Y':
            new_axes.append(axes_mod.Axis('Y', size, ay.step, ay.unit))
        elif role == 'X':
            new_axes.append(axes_mod.Axis('X', size, ax_x.step, ax_x.unit))
        else:
            new_axes.append(axes_mod.Axis(role, size))
    info = axes_mod.AxesInfo(axes=new_axes, source='hyperstack', confident=True)

    kw: dict = {'axes': info, 'rgb': bool(image.rgb)}
    if not image.rgb:
        kw['colormap'] = image.colormap
        kw['contrast_limits'] = tuple(image.contrast_limits)
    v = tiki.open_data(reshaped, f'{image.name} ({roles})', **kw)
    if split_box.isChecked() and c > 1 and not image.rgb:
        tiki.split_channels()
    tiki.status(f'Hyperstack {roles}: {c} ch × {z} z × {t} t ({order})')
    return v


# ------------------------------------ everyday image & stack commands

def math_dialog(tiki):
    """``Process ▸ Math…``: scalar arithmetic and unary transforms (32-bit float result)."""
    from .layer_select import LayerCombo

    viewer, image = tiki.require_image()
    if image.rgb:
        _error('Process ▸ Math works on greyscale images (split channels or convert first)')
    dlg, form = _form(tiki, 'Math')
    image_box = LayerCombo(viewer, Image, current=image)
    op_box = QComboBox()
    op_box.addItems(list(ops.MATH_OPS))
    val_spin = QDoubleSpinBox(decimals=4, minimum=-1e9, maximum=1e9, value=1.0)
    whole = QCheckBox('Process whole stack', checked=True)

    form.addRow('Image:', image_box)
    form.addRow('Operation:', op_box)
    form.addRow('Value:', val_spin)
    form.addRow(whole)
    _buttons(dlg, form)

    def _sync(_=None):
        op_name = op_box.currentText()
        val_spin.setEnabled(op_name not in ops.UNARY_MATH_OPS)
        img = image_box.layer()
        has_stack = img is not None and img.ndim > 2
        whole.setVisible(has_stack)
        if has_stack:
            n = int(np.prod(img.data.shape[: img.ndim - 2]))
            whole.setText(f'Process whole stack ({n} planes)')

    op_box.currentTextChanged.connect(_sync)
    image_box.layer_changed.connect(_sync)
    _sync()

    if dlg.exec_() != QDialog.Accepted:
        return None
    img = image_box.layer()
    if img is None:
        return None
    if img.rgb:
        _error('Process ▸ Math works on greyscale images')
    op_name = op_box.currentText()
    val = val_spin.value()
    use_stack = img.ndim > 2 and whole.isChecked()
    if use_stack:
        if not _confirm_size(tiki, img.data.shape, f'Math ({op_name})'):
            return None
        total = int(np.prod(img.data.shape[: img.ndim - 2]))
        with tiki.progress(f'Math ({op_name})', total) as tick:
            out = ops.math_op(img.data, op_name, val, progress=tick)
        if tick.cancelled:
            return None
        axes_out = axes_mod.fit(axes_mod.get(img), out.shape)
        drop = None
    else:
        plane = roi_mod.current_plane(viewer, img)
        out = ops.math_op(plane, op_name, val)
        axes_out = axes_mod.fit(
            axes_mod.AxesInfo(axes_mod.get(img).axes[-2:], axes_mod.get(img).source, True),
            out.shape,
        )
        drop = None

    finite = out[np.isfinite(out)]
    lo, hi = ops.saturated_limits(finite if out.ndim == 2 else out[(0,) * (out.ndim - 2)])
    suffix = op_name if op_name in ops.UNARY_MATH_OPS else f'{op_name} {val:g}'
    if out.ndim == img.ndim:
        layer = _add_result(tiki, viewer, img, out, f'{img.name} ({suffix})', img.colormap, (lo, hi), axes=axes_out, drop=drop)
    else:
        layer = Image(
            out,
            name=f'{img.name} ({suffix})',
            colormap=img.colormap,
            contrast_limits=(lo, hi),
            metadata={'axes': axes_out.to_dict()},
            **axes_mod.geometry(img, ndim=2),
        )
        viewer.add_layer(layer)
        tiki.reapply_tool(viewer)
    tiki.status(f'Math ({suffix}) → "{layer.name}"')
    return layer


def resize_dialog(tiki):
    """``Image ▸ Adjust ▸ Size…``: resize spatial X/Y and update physical pixel size."""
    from .layer_select import LayerCombo

    viewer, image = tiki.require_image()
    dlg, form = _form(tiki, 'Resize (Size)')
    image_box = LayerCombo(viewer, Image, current=image)
    x_scale = QDoubleSpinBox(decimals=4, minimum=0.01, maximum=100.0, singleStep=0.1, value=1.0)
    y_scale = QDoubleSpinBox(decimals=4, minimum=0.01, maximum=100.0, singleStep=0.1, value=1.0)
    w_spin = QSpinBox(minimum=1, maximum=32768, suffix=' px')
    h_spin = QSpinBox(minimum=1, maximum=32768, suffix=' px')
    constrain = QCheckBox('Constrain aspect ratio', checked=True)
    interp = QComboBox()
    interp.addItem('Bilinear', 1)
    interp.addItem('Bicubic', 3)
    interp.addItem('Nearest neighbor', 0)

    form.addRow('Image:', image_box)
    form.addRow('X scale:', x_scale)
    form.addRow('Y scale:', y_scale)
    form.addRow('Width:', w_spin)
    form.addRow('Height:', h_spin)
    form.addRow(constrain)
    form.addRow('Interpolation:', interp)
    _buttons(dlg, form)

    state = {'busy': False, 'h': 1, 'w': 1}

    def _init_dims(_=None):
        img = image_box.layer()
        if img is None:
            return
        h, w = roi_mod.spatial_shape(img)
        state['h'], state['w'] = h, w
        state['busy'] = True
        try:
            x_scale.setValue(1.0)
            y_scale.setValue(1.0)
            w_spin.setValue(w)
            h_spin.setValue(h)
        finally:
            state['busy'] = False

    def _from_scale(which: str):
        if state['busy']:
            return
        state['busy'] = True
        try:
            if constrain.isChecked():
                val = x_scale.value() if which == 'x' else y_scale.value()
                x_scale.setValue(val)
                y_scale.setValue(val)
            w_spin.setValue(max(1, int(round(state['w'] * x_scale.value()))))
            h_spin.setValue(max(1, int(round(state['h'] * y_scale.value()))))
        finally:
            state['busy'] = False

    def _from_pixels(which: str):
        if state['busy']:
            return
        state['busy'] = True
        try:
            if which == 'w':
                sx = w_spin.value() / max(state['w'], 1)
                x_scale.setValue(sx)
                if constrain.isChecked():
                    y_scale.setValue(sx)
                    h_spin.setValue(max(1, int(round(state['h'] * sx))))
            else:
                sy = h_spin.value() / max(state['h'], 1)
                y_scale.setValue(sy)
                if constrain.isChecked():
                    x_scale.setValue(sy)
                    w_spin.setValue(max(1, int(round(state['w'] * sy))))
        finally:
            state['busy'] = False

    image_box.layer_changed.connect(_init_dims)
    x_scale.valueChanged.connect(lambda _: _from_scale('x'))
    y_scale.valueChanged.connect(lambda _: _from_scale('y'))
    w_spin.valueChanged.connect(lambda _: _from_pixels('w'))
    h_spin.valueChanged.connect(lambda _: _from_pixels('h'))
    _init_dims()

    if dlg.exec_() != QDialog.Accepted:
        return None
    img = image_box.layer()
    if img is None:
        return None
    old_h, old_w = roi_mod.spatial_shape(img)
    new_h, new_w = h_spin.value(), w_spin.value()
    spatial_end = img.data.ndim - 1 if img.rgb else img.data.ndim
    lead = img.data.shape[: max(0, spatial_end - 2)]
    out_shape = lead + (new_h, new_w) + ((img.data.shape[-1],) if img.rgb else ())
    if not _confirm_size(tiki, out_shape, 'Resize'):
        return None

    total = int(np.prod(lead)) if lead else 1
    with tiki.progress('Resize', total) as tick:
        out = ops.resize_image(img.data, (new_h, new_w), order=interp.currentData(), rgb=bool(img.rgb), progress=tick)
    if tick.cancelled:
        return None

    src_info = axes_mod.get(img)
    new_axes = [axes_mod.Axis(a.role, a.size, a.step, a.unit) for a in src_info.axes]
    new_axes[-2] = axes_mod.Axis('Y', new_h, (src_info.axes[-2].step * old_h / new_h) if src_info.axes[-2].step else None, src_info.axes[-2].unit)
    new_axes[-1] = axes_mod.Axis('X', new_w, (src_info.axes[-1].step * old_w / new_w) if src_info.axes[-1].step else None, src_info.axes[-1].unit)
    new_info = axes_mod.AxesInfo(new_axes, src_info.source, src_info.confident, list(src_info.notes))
    kw: dict = {'axes': new_info, 'rgb': bool(img.rgb)}
    if not img.rgb:
        kw['colormap'] = img.colormap
    v = tiki.open_data(out, f'Resize_{img.name}', **kw)
    tiki.status(f'Resized {img.name} to {new_w}×{new_h} px')
    return v


def morphology_dialog(tiki):
    """``Process ▸ Labels ▸ Morphology…``: erode, dilate, open, or close Labels."""
    from .layer_select import LayerCombo

    viewer, _ = tiki.require_image()
    layer = tiki.labels_layer(viewer)
    dlg, form = _form(tiki, 'Labels Morphology')
    labels_box = LayerCombo(viewer, Labels, current=layer)
    op_box = QComboBox()
    for label, key in (('Open', 'open'), ('Close', 'close'), ('Erode', 'erode'), ('Dilate', 'dilate')):
        op_box.addItem(label, key)
    radius_spin = QDoubleSpinBox(decimals=1, minimum=1.0, maximum=200.0, singleStep=1.0, value=1.0)
    radius_spin.setSuffix(' px')
    phys_label = QLabel('')
    scope_box = QComboBox()
    three_d = QCheckBox('3D (through Z, scaled by voxel size)', checked=True)

    form.addRow('Labels:', labels_box)
    form.addRow('Operation:', op_box)
    form.addRow('Radius:', radius_spin)
    form.addRow('', phys_label)
    form.addRow('Apply to:', scope_box)
    form.addRow(three_d)
    _buttons(dlg, form)

    state = {'has_z': False, 'has_stack': False}

    def _update_opts(_=None):
        ly = labels_box.layer()
        if ly is None:
            return
        info = _labels_info(viewer, ly)
        state['has_z'] = info.index('Z') is not None
        state['has_stack'] = ly.ndim > 2
        scope_box.blockSignals(True)
        scope_box.clear()
        for key, lbl in nd.scopes(info):
            scope_box.addItem(lbl, key)
        keys = [scope_box.itemData(i) for i in range(scope_box.count())]
        default = nd.WHOLE if nd.WHOLE in keys else nd.PLANE
        scope_box.setCurrentIndex(keys.index(default))
        scope_box.blockSignals(False)
        scope_box.setVisible(state['has_stack'])
        three_d.setVisible(state['has_z'])
        three_d.setEnabled(state['has_z'] and scope_box.currentData() != nd.PLANE)
        u = axes_mod.spatial_unit(info)
        if u:
            sy = float(info.axes[-2].step or 1.0)
            sx = float(info.axes[-1].step or sy)
            phys_label.setText(f'≈ {radius_spin.value() * 0.5 * (sy + sx):.4g} {u}')
        else:
            phys_label.setText('')

    labels_box.layer_changed.connect(_update_opts)
    scope_box.currentIndexChanged.connect(lambda _: three_d.setEnabled(state['has_z'] and scope_box.currentData() != nd.PLANE))
    radius_spin.valueChanged.connect(lambda _: _update_opts())
    _update_opts()

    if dlg.exec_() != QDialog.Accepted:
        return None
    ly = labels_box.layer()
    if ly is None:
        return None
    op_key = op_box.currentData()
    rad = radius_spin.value()
    info = _labels_info(viewer, ly)
    scope = scope_box.currentData() if state['has_stack'] else nd.PLANE
    by_volume = state['has_z'] and three_d.isChecked() and scope != nd.PLANE
    zr = _labels_z_ratio(viewer, ly)
    shape = tuple(ly.data.shape[: ly.ndim])
    ref = roi_mod.target_image(viewer)
    if ref is not None and len(axes_mod.get(ref).axes) >= ly.ndim:
        index = roi_mod.current_index(viewer, ref)[-ly.ndim :]
    else:
        point = ly.world_to_data(viewer.dims.point[-ly.ndim :])
        index = tuple(int(np.clip(np.round(point[d]), 0, shape[d] - 1)) for d in range(ly.ndim))

    if scope == nd.PLANE or ly.ndim == 2:
        out = ops.morphology_labels(tiki.labels_plane(viewer, ly), op=op_key, radius=rad)
        geom = axes_mod.geometry(ly, ndim=2)
    else:
        blocks = nd.blocks(info, shape, index, scope, by_volume)
        out_arr = np.zeros(shape, dtype=np.int32)
        with tiki.progress(f'Morphology ({op_key})', len(blocks)) as tick:
            for n, b in enumerate(blocks, 1):
                blk = np.asarray(ly.data[b])
                out_arr[b] = ops.morphology_labels(blk, op=op_key, radius=rad, z_ratio=zr if blk.ndim == 3 else 1.0)
                if tick(n, len(blocks)) is False:
                    return None
        if scope == nd.VOLUME:
            out = out_arr[blocks[0]]
            geom = axes_mod.geometry(ly, ndim=3)
        else:
            out = out_arr
            geom = axes_mod.geometry(ly)

    new_ly = Labels(
        out,
        name=f'{ly.name} {op_key}',
        opacity=0.6,
        metadata={'axes': info.to_dict()} if out.ndim > 2 else {},
        **geom,
    )
    viewer.add_layer(new_ly)
    viewer.layers.selection.active = new_ly
    roi = roi_mod.roi_layer(viewer, create=False)
    if roi is not None:
        viewer.layers.move(viewer.layers.index(roi), len(viewer.layers))
    tiki.status(f'Labels {op_key} (r={rad:g} px) → "{new_ly.name}"')
    return new_ly


def find_edges_command(tiki):
    """``Process ▸ Filters ▸ Find Edges``: Sobel edge magnitude."""
    viewer, image = tiki.require_image()
    if image.rgb:
        _error('Find Edges works on greyscale images (split channels or convert first)')
    if image.ndim > 2:
        if not _confirm_size(tiki, image.data.shape, 'Find Edges'):
            return None
        total = int(np.prod(image.data.shape[: image.ndim - 2]))
        with tiki.progress('Find Edges', total) as tick:
            out = ops.map_planes(ops.find_edges, image.data, out_dtype=np.float32, progress=tick)
        if tick.cancelled:
            return None
    else:
        out = ops.find_edges(image.data)
    lo, hi = ops.saturated_limits(out if out.ndim == 2 else out[(0,) * (out.ndim - 2)])
    layer = _add_result(tiki, viewer, image, out, f'{image.name} edges', image.colormap, (lo, hi))
    tiki.status(f'Find Edges → "{layer.name}"')
    return layer


def concatenate_dialog(tiki):
    """``Image ▸ Stacks ▸ Concatenate…``: join 2 or more open images along an axis."""
    images = _all_images(tiki)
    if len(images) < 2:
        _error('Concatenate needs at least two open images')

    dlg, form = _form(tiki, 'Concatenate')
    list_w = QListWidget()
    _, active = tiki.windows.active_image()
    ref_shape = active.data.shape[-2:] if active is not None else images[0][2].data.shape[-2:]
    ref_ndim = active.data.ndim if active is not None else images[0][2].data.ndim
    checked_count = 0
    for idx, (label, v, ly) in enumerate(images):
        item = QListWidgetItem(f'{label}  {ly.data.shape}')
        item.setData(Qt.UserRole, (v, ly))
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        match = ly.data.ndim == ref_ndim and ly.data.shape[-2:] == ref_shape
        should_check = (ly is active) or (match and checked_count < 2)
        item.setCheckState(Qt.Checked if should_check else Qt.Unchecked)
        if should_check:
            checked_count += 1
        list_w.addItem(item)

    up_btn = QPushButton('Move Up')
    down_btn = QPushButton('Move Down')
    move_bar = QHBoxLayout()
    move_bar.addWidget(up_btn)
    move_bar.addWidget(down_btn)
    move_bar.addStretch()

    def _move(delta: int):
        row = list_w.currentRow()
        if row < 0:
            return
        new_row = row + delta
        if not (0 <= new_row < list_w.count()):
            return
        it = list_w.takeItem(row)
        list_w.insertItem(new_row, it)
        list_w.setCurrentRow(new_row)

    up_btn.clicked.connect(lambda: _move(-1))
    down_btn.clicked.connect(lambda: _move(1))

    axis_box = QComboBox()
    note = QLabel('')
    note.setStyleSheet('color: #b00020;')
    form.addRow('Images (checked, in order):', list_w)
    form.addRow('', move_bar)
    form.addRow('Concatenate along:', axis_box)
    form.addRow(note)
    buttons = _buttons(dlg, form)

    def _selected():
        out = []
        for i in range(list_w.count()):
            it = list_w.item(i)
            if it.checkState() == Qt.Checked:
                out.append(it.data(Qt.UserRole))
        return out

    def _refresh_axes(_=None):
        sel = _selected()
        cur = axis_box.currentData()
        axis_box.blockSignals(True)
        axis_box.clear()
        ok = len(sel) >= 2
        if sel:
            first_ly = sel[0][1]
            if first_ly.ndim == 2:
                for role, lbl in (('T', 'New T (time) axis'), ('Z', 'New Z (depth) axis'), ('?', 'New stack axis (?)')):
                    axis_box.addItem(lbl, role)
            else:
                info0 = axes_mod.get(first_ly)
                for d in range(first_ly.ndim - 2):
                    axis_box.addItem(f'{info0.labels[d]} (axis {d})', d)
                if cur is not None and axis_box.findData(cur) >= 0:
                    axis_box.setCurrentIndex(axis_box.findData(cur))
        axis_box.blockSignals(False)
        msg = '' if ok else 'Select at least 2 images'
        if ok:
            ndims = {ly.data.ndim for _, ly in sel}
            if len(ndims) != 1:
                ok = False
                msg = 'Selected images must have the same number of dimensions'
        note.setText(msg)
        buttons.button(QDialogButtonBox.Ok).setEnabled(ok)

    list_w.itemChanged.connect(_refresh_axes)
    _refresh_axes()

    if dlg.exec_() != QDialog.Accepted:
        return None
    sel = _selected()
    if len(sel) < 2:
        _error('select at least two images to concatenate')

    first_ly = sel[0][1]
    arrays = [ly.data for _, ly in sel]
    ax_choice = axis_box.currentData()
    try:
        if first_ly.ndim == 2:
            out = ops.concatenate_stacks(arrays, axis=0)
            src_info = axes_mod.get(first_ly)
            new_axes = [axes_mod.Axis(ax_choice if isinstance(ax_choice, str) else 'T', out.shape[0])] + list(src_info.axes)
            new_info = axes_mod.AxesInfo(new_axes, 'concatenated', True)
        else:
            ax = int(ax_choice)
            out = ops.concatenate_stacks(arrays, axis=ax)
            new_info = axes_mod.fit(axes_mod.get(first_ly), out.shape)
    except ValueError as exc:
        _error(str(exc))

    kw: dict = {'axes': new_info, 'rgb': bool(first_ly.rgb)}
    if not first_ly.rgb:
        kw['colormap'] = first_ly.colormap
    v = tiki.open_data(out, f'Concat_{first_ly.name}', **kw)
    tiki.status(f'Concatenated {len(sel)} images → {out.shape}')
    return v


def images_to_stack_dialog(tiki):
    """``Image ▸ Stacks ▸ Images to Stack…``: combine open matching images into one stack."""
    images = _all_images(tiki)
    if len(images) < 2:
        _error('Images to Stack needs at least two open images')

    _, active = tiki.windows.active_image()
    ref_ly = active if active is not None else images[0][2]
    ref_shape = ref_ly.data.shape

    dlg, form = _form(tiki, 'Images to Stack')
    filt_edit = QLineEdit()
    filt_edit.setPlaceholderText('optional title substring')
    axis_box = QComboBox()
    axis_box.addItem('T (time)', 'T')
    axis_box.addItem('Z (depth)', 'Z')
    axis_box.addItem('C (channel)', 'C')
    axis_box.addItem('? (unknown)', '?')
    summary = QLabel('')

    form.addRow('Title contains:', filt_edit)
    form.addRow('Stack axis role:', axis_box)
    form.addRow(summary)
    buttons = _buttons(dlg, form)

    def _matching():
        q = filt_edit.text().strip().lower()
        out = []
        for label, _v, ly in images:
            if ly.data.shape == ref_shape and (not q or q in ly.name.lower() or q in label.lower()):
                out.append(ly)
        return out

    def _refresh(_=None):
        matched = _matching()
        ok = len(matched) >= 2
        summary.setText(
            f'Will stack {len(matched)} image(s) of shape {ref_shape}'
            if ok
            else f'Need at least 2 matching images of shape {ref_shape} (found {len(matched)})'
        )
        summary.setStyleSheet('' if ok else 'color: #b00020;')
        buttons.button(QDialogButtonBox.Ok).setEnabled(ok)

    filt_edit.textChanged.connect(_refresh)
    _refresh()

    if dlg.exec_() != QDialog.Accepted:
        return None
    matched = _matching()
    if len(matched) < 2:
        _error('need at least two matching open images')

    out = ops.images_to_stack([ly.data for ly in matched])
    role = axis_box.currentData()
    ref_info = axes_mod.get(matched[0])
    new_axes = [axes_mod.Axis(role, len(matched))] + [
        axes_mod.Axis('?' if a.role == role else a.role, a.size, a.step, a.unit) for a in ref_info.axes
    ]
    new_info = axes_mod.AxesInfo(new_axes, ref_info.source, '?' not in ''.join(a.role for a in new_axes[:-2]))
    kw: dict = {'axes': new_info, 'rgb': bool(matched[0].rgb)}
    if not matched[0].rgb:
        kw['colormap'] = matched[0].colormap
    v = tiki.open_data(out, 'Stack', **kw)
    tiki.status(f'Stacked {len(matched)} images → {out.shape}')
    return v


def reverse_stack_command(tiki):
    """``Image ▸ Stacks ▸ Reverse``: reverse slice order along a stack axis."""
    _viewer, image = tiki.require_image()
    spatial_end = image.data.ndim - 1 if image.rgb else image.data.ndim
    if spatial_end < 3:
        _error('Reverse needs a stack or movie, not a single 2D image')
    info = axes_mod.get(image)
    candidates = [d for d in range(spatial_end - 2) if info.axes[d].role != 'C']
    if not candidates:
        candidates = list(range(spatial_end - 2))

    if len(candidates) == 1:
        ax = candidates[0]
    else:
        dlg, form = _form(tiki, 'Reverse Stack')
        axis_box = QComboBox()
        for d in candidates:
            axis_box.addItem(f'{info.labels[d]} ({image.data.shape[d]})', d)
        form.addRow('Reverse along:', axis_box)
        _buttons(dlg, form)
        if dlg.exec_() != QDialog.Accepted:
            return None
        ax = axis_box.currentData()

    out = ops.reverse_stack(image.data, axis=ax)
    new_info = axes_mod.fit(info, out.shape[:-1] if image.rgb else out.shape)
    kw: dict = {'axes': new_info, 'rgb': bool(image.rgb)}
    if not image.rgb:
        kw['colormap'] = image.colormap
    v = tiki.open_data(out, f'Rev_{image.name}', **kw)
    tiki.status(f'Reversed {image.name} along {info.labels[ax]}')
    return v


def remove_slices_dialog(tiki):
    """``Image ▸ Stacks ▸ Remove Slices…``: drop a 1-based slice range from a stack."""
    from .layer_select import LayerCombo

    viewer, image = tiki.require_image()
    if image.ndim < 3:
        _error('Remove Slices needs a stack or movie')
    dlg, form = _form(tiki, 'Remove Slices')
    image_box = LayerCombo(viewer, Image, current=image)
    axis_box = QComboBox()
    spec_edit = QLineEdit('1')
    spec_edit.setToolTip('1-based inclusive indices to remove (e.g. 1, 1-5, 1-20-2)')
    note = QLabel('')

    form.addRow('Image:', image_box)
    form.addRow('Axis:', axis_box)
    form.addRow('Slices to remove:', spec_edit)
    form.addRow(note)
    buttons = _buttons(dlg, form)

    def _fill_all_axes(_=None):
        img = image_box.layer()
        axis_box.blockSignals(True)
        axis_box.clear()
        if img is not None:
            info = axes_mod.get(img)
            for d in range(img.ndim - 2):
                axis_box.addItem(f'{info.labels[d]} ({img.data.shape[d]})', d)
            default = axes_mod.stack_axis(info, 'TZ')
            if default is not None and axis_box.findData(default) >= 0:
                axis_box.setCurrentIndex(axis_box.findData(default))
        axis_box.blockSignals(False)
        _validate()

    def _validate(_=None):
        img = image_box.layer()
        ax = axis_box.currentData()
        if img is None or ax is None:
            buttons.button(QDialogButtonBox.Ok).setEnabled(False)
            note.setText('Need a stack axis')
            return
        size = img.data.shape[ax]
        try:
            parsed = ops.parse_slice_spec(spec_edit.text(), size)
            n_rem = len(range(*parsed.indices(size))) if isinstance(parsed, slice) else len(set(parsed.tolist()))
            n_keep = size - n_rem
            if n_keep < 1:
                raise ValueError('cannot remove all slices')
            note.setText(f'Will remove {n_rem} slice(s); {n_keep} remaining')
            note.setStyleSheet('')
            buttons.button(QDialogButtonBox.Ok).setEnabled(True)
        except ValueError as exc:
            note.setText(str(exc))
            note.setStyleSheet('color: #b00020;')
            buttons.button(QDialogButtonBox.Ok).setEnabled(False)

    image_box.layer_changed.connect(_fill_all_axes)
    axis_box.currentIndexChanged.connect(_validate)
    spec_edit.textChanged.connect(_validate)
    _fill_all_axes()

    if dlg.exec_() != QDialog.Accepted:
        return None
    img = image_box.layer()
    ax = axis_box.currentData()
    out = ops.remove_slices(img.data, ax, spec_edit.text(), rgb=bool(img.rgb))
    new_info = axes_mod.fit(axes_mod.get(img), out.shape[:-1] if img.rgb else out.shape)
    kw: dict = {'axes': new_info, 'rgb': bool(img.rgb)}
    if not img.rgb:
        kw['colormap'] = img.colormap
    v = tiki.open_data(out, f'Trim_{img.name}', **kw)
    tiki.status(f'Removed slices ({spec_edit.text()}) → {out.shape}')
    return v


def hyperstack_to_stack_command(tiki):
    """``Image ▸ Stacks ▸ Hyperstack to Stack``: flatten hyperstack axes into 3D stack."""
    _viewer, image = tiki.require_image()
    spatial_ndim = 3 if image.rgb else 2
    if image.data.ndim <= spatial_ndim + 1:
        _error('Hyperstack to Stack needs a 4D+ hyperstack (e.g. TZCYX or ZCYX)')
    src_info = axes_mod.get(image)
    out = ops.hyperstack_to_stack(image.data, rgb=bool(image.rgb))
    ay, ax_x = src_info.axes[-2], src_info.axes[-1]
    new_info = axes_mod.AxesInfo(
        [
            axes_mod.Axis('?', out.shape[0]),
            axes_mod.Axis('Y', ay.size, ay.step, ay.unit),
            axes_mod.Axis('X', ax_x.size, ax_x.step, ax_x.unit),
        ],
        'flattened hyperstack',
        False,
        [f'flattened from {src_info.roles}'],
    )
    kw: dict = {'axes': new_info, 'rgb': bool(image.rgb)}
    if not image.rgb:
        kw['colormap'] = image.colormap
    v = tiki.open_data(out, f'Stack_{image.name}', **kw)
    tiki.status(f'Flattened {src_info.roles} → stack {out.shape}')
    return v


def merge_channels_dialog(tiki):
    """``Image ▸ Color ▸ Merge Channels…``: combine separate images into a composite stack."""
    from . import contrast

    slots = [
        ('C1 (Red):', 'Red'),
        ('C2 (Green):', 'Green'),
        ('C3 (Blue):', 'Blue'),
        ('C4 (Gray):', 'Grays'),
        ('C5 (Cyan):', 'Cyan'),
        ('C6 (Magenta):', 'Magenta'),
        ('C7 (Yellow):', 'Yellow'),
    ]
    all_imgs = [
        (label, ly)
        for label, _v, ly in _all_images(tiki)
        if not ly.rgb and axes_mod.get(ly).index('C') is None
    ]
    if not all_imgs:
        _error('Merge Channels needs at least one open greyscale image without a C axis')

    _, active = tiki.windows.active_image()
    ref_shape = active.data.shape if (active is not None and not active.rgb) else all_imgs[0][1].data.shape
    matching = [ly for _, ly in all_imgs if ly.data.shape == ref_shape]

    dlg, form = _form(tiki, 'Merge Channels')
    combos: list[tuple[QComboBox, str]] = []
    for idx, (label, lut_name) in enumerate(slots):
        cb = QComboBox()
        cb.addItem('<None>', None)
        for img_label, ly in all_imgs:
            cb.addItem(img_label, ly)
        if idx < len(matching) and idx < 3:
            cb.setCurrentIndex(cb.findData(matching[idx]))
        form.addRow(label, cb)
        combos.append((cb, lut_name))

    split_box = QCheckBox('Create composite (split channels in new window)', checked=True)
    note = QLabel('')
    note.setStyleSheet('color: #b00020;')
    form.addRow(split_box)
    form.addRow(note)
    buttons = _buttons(dlg, form)

    def _validate(_=None):
        chosen = [cb.currentData() for cb, _ in combos if cb.currentData() is not None]
        if not chosen:
            note.setText('Choose at least one image')
            buttons.button(QDialogButtonBox.Ok).setEnabled(False)
            return
        shapes = {ly.data.shape for ly in chosen}
        if len(shapes) != 1:
            note.setText('Selected channels must have matching shapes')
            buttons.button(QDialogButtonBox.Ok).setEnabled(False)
            return
        note.setText('')
        buttons.button(QDialogButtonBox.Ok).setEnabled(True)

    for cb, _ in combos:
        cb.currentIndexChanged.connect(_validate)
    _validate()

    if dlg.exec_() != QDialog.Accepted:
        return None

    chosen = [(cb.currentData(), lut_name) for cb, lut_name in combos if cb.currentData() is not None]
    if not chosen:
        return None
    layers_sel = [ly for ly, _ in chosen]
    merged = ops.images_to_stack([ly.data for ly in layers_sel])
    first = layers_sel[0]
    first_info = axes_mod.get(first)
    c_axes = [axes_mod.Axis('C', len(layers_sel))] + [
        axes_mod.Axis(a.role, a.size, a.step, a.unit) for a in first_info.axes
    ]
    new_info = axes_mod.AxesInfo(c_axes, first_info.source, first_info.confident)

    luts = []
    for _ly, lut_name in chosen:
        cmap = contrast.LUTS[lut_name]
        from napari.layers import Image as _Img

        tmp_ly = _Img(np.zeros((2, 2), dtype=np.uint8), colormap=cmap)
        luts.append(contrast.layer_lut_bytes(tmp_ly))

    ch_names = [ly.name for ly in layers_sel]
    v = tiki.open_data(
        merged,
        'Composite',
        axes=new_info,
        metadata={'imagej_luts': luts, 'channel_names': ch_names},
    )
    if split_box.isChecked() and len(layers_sel) > 1:
        tiki.split_channels()
    tiki.status(f'Merged {len(layers_sel)} channel(s) → "{v.title}"')
    return v
