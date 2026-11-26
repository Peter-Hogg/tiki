"""Image > Color: Split Channels, Merge Channels and the Channels tool.

Splitting keeps the channels in the *same* window as separate, additively
blended layers (napari's strength), each a view into the original data, so
no memory is copied. Merge puts the original layer back.
"""

from __future__ import annotations

import numpy as np
from napari.layers import Image
from napari.utils.colormaps import Colormap
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QCheckBox, QComboBox, QGridLayout, QLabel, QPushButton, QSlider, QWidget

from . import axes as axes_mod
from . import contrast, ops, style
from . import roi as roi_mod

DEFAULT_COLORS = {
    1: ['gray'],
    2: ['cyan', 'magenta'],
    3: ['red', 'green', 'blue'],
    4: ['red', 'green', 'blue', 'gray'],
}
COLOR_CHOICES = list(contrast.LUTS)
_SLIDER_STEPS = 1000


def _imagej_colormaps(luts, n: int) -> list | None:
    """Colormaps from ImageJ LUTs stored in the file (one per channel)."""
    if luts is None or len(luts) != n:
        return None
    out = []
    for k, lut in enumerate(luts):
        lut = np.asarray(lut, dtype=float)
        if lut.shape != (3, 256):
            return None
        rgba = np.column_stack([np.clip(lut.T / 255.0, 0.0, 1.0), np.ones(256)])
        out.append(Colormap(rgba, name=f'imagej_c{k + 1}'))
    return out


def _keep_position(viewer):
    """Remember slider positions by axis name; returns a function restoring them."""
    saved = dict(zip(viewer.dims.axis_labels, viewer.dims.point))

    def restore():
        for i, label in enumerate(viewer.dims.axis_labels):
            if label in saved and label not in ('y', 'x'):
                viewer.dims.set_point(i, saved[label])

    return restore


def split(viewer, image: Image) -> list[Image]:
    """Replace ``image`` by one layer per channel (C axis) in the same window."""
    from .app import TikiError

    info = axes_mod.get(image)
    c = info.index('C')
    if c is None:
        raise TikiError(
            f'{image.name} has no channel (C) axis.\n\nIf one of its axes is really channels, '
            'mark it as C in Image ▸ Properties.',
            fix='properties',
        )
    n = image.data.shape[c]
    if n > 8:
        raise TikiError(f'the C axis has {n} entries; is it really channels? Check Image ▸ Properties.',
                        fix='properties')
    colormaps = _imagej_colormaps(image.metadata.get('imagej_luts'), n) or DEFAULT_COLORS.get(n, ['gray'] * n)
    ch_names = image.metadata.get('channel_names')
    rest = info.without(c)
    stem = image.name
    position = viewer.layers.index(image)
    layers = []
    for k in range(n):
        index = (slice(None),) * c + (k,)
        data = image.data[index]  # a view: memory-mapped files stay on disk
        cname = str(ch_names[k]).strip() if (isinstance(ch_names, (list, tuple)) and k < len(ch_names) and ch_names[k]) else ''
        layer_name = f'C{k + 1} ({cname})-{stem}' if cname else f'C{k + 1}-{stem}'
        meta = {
            'axes': rest.to_dict(),
            'path': image.metadata.get('path'),
            'split_from': image,
            'channel': k,
        }
        if cname:
            meta['channel_name'] = cname
        if 'acquisition' in image.metadata:
            meta['acquisition'] = image.metadata['acquisition']
        if 'raw_metadata' in image.metadata:
            meta['raw_metadata'] = image.metadata['raw_metadata']
        layer = Image(data, name=layer_name, colormap=colormaps[k], blending='additive',
                      **axes_mod.geometry(image, drop=c), metadata=meta)
        plane = np.asarray(data[(0,) * (data.ndim - 2)])
        layer.contrast_limits = ops.saturated_limits(plane)
        layers.append(layer)
    restore = _keep_position(viewer)
    viewer.layers.remove(image)
    for k, layer in enumerate(layers):
        viewer.layers.insert(position + k, layer)
        axes_mod.apply(viewer, layer, rest)
    restore()
    return layers


def merge(viewer) -> Image:
    """Undo Split Channels: put the original multichannel layer back."""
    from .app import TikiError

    parts = [ly for ly in viewer.layers if isinstance(ly, Image) and 'split_from' in ly.metadata]
    if not parts:
        raise TikiError('nothing to merge: this window has no split channels')
    original = parts[0].metadata['split_from']
    group = [ly for ly in parts if ly.metadata['split_from'] is original]
    position = min(viewer.layers.index(ly) for ly in group)
    restore = _keep_position(viewer)
    for ly in group:
        viewer.layers.remove(ly)
    viewer.layers.insert(position, original)
    axes_mod.apply(viewer, original, axes_mod.get(original))
    restore()
    viewer.layers.selection.active = original
    return original


class ChannelsTool(QWidget):
    """Image > Color > Channels Tool: show/hide, LUT and per-channel contrast."""

    def __init__(self, tiki):
        super().__init__(None, Qt.Window)
        style.apply(self)
        self.tiki = tiki
        self._layer_connections: list[tuple] = []
        self.setWindowTitle('Channels')
        self.grid = QGridLayout(self)
        self.refresh()

    def _disconnect_layers(self) -> None:
        for ev, cb in self._layer_connections:
            try:
                ev.disconnect(cb)
            except Exception:
                pass
        self._layer_connections.clear()

    def refresh(self, viewer=None) -> None:
        self._disconnect_layers()
        viewer = viewer or self.tiki.windows.active
        while self.grid.count():
            w = self.grid.takeAt(0).widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        images = [] if viewer is None else [
            ly for ly in viewer.layers if isinstance(ly, Image) and ly.name not in roi_mod.PREVIEW_NAMES
        ]
        if not images:
            self.grid.addWidget(QLabel('No image layers in the current window'), 0, 0)
            self.adjustSize()
            return

        for col, head in enumerate(('Channel', 'LUT', 'Min', 'Max', 'Range', '')):
            self.grid.addWidget(QLabel(f'<b>{head}</b>') if head else QLabel(''), 0, col)

        for row, layer in enumerate(images, start=1):
            show = QCheckBox(layer.name, checked=layer.visible)
            show.toggled.connect(lambda on, ly=layer: setattr(ly, 'visible', on))

            color = QComboBox()
            for label, cmap in contrast.LUTS.items():
                color.addItem(label, cmap)
            current_label = contrast.lut_label(layer)
            if color.findText(current_label) < 0:
                color.insertItem(0, current_label, layer.colormap)
            color.setCurrentText(current_label)
            color.currentIndexChanged.connect(
                lambda idx, cb=color, ly=layer: setattr(ly, 'colormap', cb.itemData(idx))
                if cb.itemData(idx) is not None
                else None
            )

            lo_slider = QSlider(Qt.Horizontal)
            hi_slider = QSlider(Qt.Horizontal)
            for sl in (lo_slider, hi_slider):
                sl.setRange(0, _SLIDER_STEPS)
                sl.setMinimumWidth(75)
            range_label = QLabel()
            range_label.setMinimumWidth(82)
            auto_btn = QPushButton('Auto')

            state = {'updating': False}

            def sync_row(_event=None, ly=layer, lo_s=lo_slider, hi_s=hi_slider, lbl=range_label, st=state):
                if st['updating']:
                    return
                st['updating'] = True
                try:
                    lo, hi = map(float, ly.contrast_limits)
                    r0, r1 = map(float, ly.contrast_limits_range)
                    r0, r1 = min(r0, lo), max(r1, hi)
                    span = r1 - r0 if r1 > r0 else 1.0
                    lo_s.setValue(int(np.clip(round((lo - r0) / span * _SLIDER_STEPS), 0, _SLIDER_STEPS)))
                    hi_s.setValue(int(np.clip(round((hi - r0) / span * _SLIDER_STEPS), 0, _SLIDER_STEPS)))
                    lbl.setText(f'{lo:.3g}–{hi:.3g}')
                finally:
                    st['updating'] = False

            def apply_slider(which, step, ly=layer, lo_s=lo_slider, hi_s=hi_slider, lbl=range_label, st=state):
                if st['updating']:
                    return
                lo, hi = map(float, ly.contrast_limits)
                r0, r1 = map(float, ly.contrast_limits_range)
                r0, r1 = min(r0, lo), max(r1, hi)
                span = r1 - r0 if r1 > r0 else 1.0
                val = r0 + step / _SLIDER_STEPS * span
                if which == 'lo':
                    new_lo, new_hi = val, max(val, hi)
                else:
                    new_lo, new_hi = min(val, lo), val
                if new_hi <= new_lo:
                    eps = 1.0 if np.issubdtype(ly.dtype, np.integer) else max(1e-6, abs(new_lo) * 1e-6, 1e-3)
                    new_hi = new_lo + eps
                st['updating'] = True
                try:
                    ly.contrast_limits_range = (min(r0, new_lo), max(r1, new_hi))
                    ly.contrast_limits = (float(new_lo), float(new_hi))
                    if which == 'lo' and new_hi > hi:
                        hi_s.setValue(int(np.clip(round((new_hi - r0) / span * _SLIDER_STEPS), 0, _SLIDER_STEPS)))
                    elif which == 'hi' and new_lo < lo:
                        lo_s.setValue(int(np.clip(round((new_lo - r0) / span * _SLIDER_STEPS), 0, _SLIDER_STEPS)))
                    lbl.setText(f'{new_lo:.3g}–{new_hi:.3g}')
                finally:
                    st['updating'] = False

            def sync_cmap(_event=None, ly=layer, cb=color):
                lbl = contrast.lut_label(ly)
                cb.blockSignals(True)
                if cb.findText(lbl) < 0:
                    cb.insertItem(0, lbl, ly.colormap)
                cb.setCurrentText(lbl)
                cb.blockSignals(False)

            def auto_row(_=False, v=viewer, ly=layer):
                plane = roi_mod.current_plane(v, ly)
                lo, hi = ops.saturated_limits(plane)
                r0, r1 = map(float, ly.contrast_limits_range)
                ly.contrast_limits_range = (min(r0, lo), max(r1, hi))
                ly.contrast_limits = (lo, hi)

            lo_slider.valueChanged.connect(lambda val, fn=apply_slider: fn('lo', val))
            hi_slider.valueChanged.connect(lambda val, fn=apply_slider: fn('hi', val))
            auto_btn.clicked.connect(auto_row)
            layer.events.contrast_limits.connect(sync_row)
            layer.events.colormap.connect(sync_cmap)
            self._layer_connections.append((layer.events.contrast_limits, sync_row))
            self._layer_connections.append((layer.events.colormap, sync_cmap))
            sync_row()

            self.grid.addWidget(show, row, 0)
            self.grid.addWidget(color, row, 1)
            self.grid.addWidget(lo_slider, row, 2)
            self.grid.addWidget(hi_slider, row, 3)
            self.grid.addWidget(range_label, row, 4)
            self.grid.addWidget(auto_btn, row, 5)
        sync_btn = QPushButton('Sync to All Windows')
        sync_btn.clicked.connect(lambda _=False, v=viewer: self.tiki.sync_display(v))
        self.grid.addWidget(sync_btn, len(images) + 1, 0, 1, 6)
        self.adjustSize()

    def closeEvent(self, event):
        self._disconnect_layers()
        super().closeEvent(event)
