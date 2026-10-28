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
from qtpy.QtWidgets import QCheckBox, QComboBox, QGridLayout, QLabel, QWidget

from . import axes as axes_mod
from . import ops, style
from . import roi as roi_mod

DEFAULT_COLORS = {
    1: ['gray'],
    2: ['cyan', 'magenta'],
    3: ['red', 'green', 'blue'],
    4: ['red', 'green', 'blue', 'gray'],
}
COLOR_CHOICES = ['gray', 'green', 'magenta', 'red', 'blue', 'cyan', 'yellow', 'gray_r', 'viridis', 'magma']


def _imagej_colormaps(luts, n: int) -> list | None:
    """Colormaps from ImageJ LUTs stored in the file (one per channel)."""
    if luts is None or len(luts) != n:
        return None
    out = []
    for k, lut in enumerate(luts):
        lut = np.asarray(lut, dtype=float)
        if lut.shape != (3, 256):
            return None
        top = lut[:, -1] / 255.0
        out.append(Colormap([[0, 0, 0, 1], [*top, 1]], name=f'imagej_c{k + 1}'))
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
    rest = info.without(c)
    stem = image.name
    position = viewer.layers.index(image)
    layers = []
    for k in range(n):
        index = (slice(None),) * c + (k,)
        data = image.data[index]  # a view: memory-mapped files stay on disk
        layer = Image(data, name=f'C{k + 1}-{stem}', colormap=colormaps[k], blending='additive',
                      **axes_mod.geometry(image, drop=c),
                      metadata={'axes': rest.to_dict(), 'path': image.metadata.get('path'),
                                'split_from': image, 'channel': k})
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
    """Image > Color > Channels Tool: show/hide and recolour channel layers."""

    def __init__(self, tiki):
        super().__init__(None, Qt.Window)
        style.apply(self)
        self.tiki = tiki
        self.setWindowTitle('Channels')
        self.grid = QGridLayout(self)
        self.refresh()

    def refresh(self, viewer=None) -> None:
        viewer = viewer or self.tiki.windows.active
        while self.grid.count():
            w = self.grid.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        images = [] if viewer is None else [
            ly for ly in viewer.layers if isinstance(ly, Image) and ly.name not in roi_mod.PREVIEW_NAMES
        ]
        if not images:
            self.grid.addWidget(QLabel('No image layers in the current window'), 0, 0)
            return
        for row, layer in enumerate(images):
            show = QCheckBox(layer.name, checked=layer.visible)
            show.toggled.connect(lambda on, ly=layer: setattr(ly, 'visible', on))
            color = QComboBox()
            names = list(COLOR_CHOICES)
            current = layer.colormap.name
            if current not in names:
                names.insert(0, current)
            color.addItems(names)
            color.setCurrentText(current)
            color.currentTextChanged.connect(
                lambda name, ly=layer: setattr(ly, 'colormap', name) if name in COLOR_CHOICES else None)
            self.grid.addWidget(show, row, 0)
            self.grid.addWidget(color, row, 1)
        self.adjustSize()
