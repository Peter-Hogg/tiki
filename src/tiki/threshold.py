"""Image > Adjust > Threshold: ImageJ's dialog, with napari Labels as output.

Works on any layer of the current window (dropdown) and on the part of it
you choose: this plane, this Z-volume, or the whole stack. With a Z axis,
objects can be connected through Z (true 3D objects). The histogram and Auto
use the same scope, and the preview shows a volume in 3D view. Apply adds a
Labels layer to the window: a 0/1 mask or one id per connected object.
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
from . import nd, ops, style
from . import roi as roi_mod

PREVIEW_NAME = 'threshold preview'
COLORS = {'Red': 'red', 'Green': 'lime', 'Blue': 'dodgerblue', 'Yellow': 'yellow', 'Cyan': 'cyan', 'Magenta': 'magenta'}
STEPS = 1000  # slider resolution
STACK_SAMPLE_PLANES = 32
VOLUME_SAMPLE_PLANES = 64


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
        self.info = None
        self.setWindowTitle('Threshold')

        from .layer_select import LayerCombo

        self.image_box = LayerCombo(None)
        self.image_box.layer_changed.connect(self._set_image)
        image_row = QHBoxLayout()
        image_row.addWidget(QLabel('Image:'))
        image_row.addWidget(self.image_box, 1)
        self.channel_note = QLabel('')
        self.channel_note.setWordWrap(True)
        self.split_button = QPushButton('Split Channels')
        self.split_button.clicked.connect(lambda: self.tiki._run('Split Channels', self._split_channels))
        note_row = QHBoxLayout()
        note_row.addWidget(self.channel_note, 1)
        note_row.addWidget(self.split_button)
        self.note_widget = QWidget()
        self.note_widget.setLayout(note_row)
        self.note_widget.setVisible(False)

        self.figure = Figure(figsize=(3.4, 0.9), dpi=100)
        self.figure.subplots_adjust(0.01, 0.02, 0.99, 0.98)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setFixedSize(370, 90)
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

        self.scope = QComboBox()
        self.scope.setToolTip('Which part of the image the histogram, Auto, preview and Apply use')
        self.scope.currentIndexChanged.connect(lambda _: self._scope_changed())
        scope_row = QHBoxLayout()
        scope_row.addWidget(QLabel('Apply to:'))
        scope_row.addWidget(self.scope, 1)
        self.connect_z = QCheckBox('Connect objects through Z (3D)', checked=True)
        self.connect_z.setToolTip('Objects touching in neighbouring slices are one 3D object (26-connected)')
        self.connect_z.toggled.connect(lambda _: self._scope_changed())

        self.dark = QCheckBox('Dark background', checked=True)
        self.dark.toggled.connect(lambda _: self.auto())
        self.split = QCheckBox('Split objects (labels)', checked=True)
        self.split.setToolTip('Apply gives each connected object its own label id, like Analyze Particles')
        self.min_size = QSpinBox()
        self.min_size.setRange(0, 10**7)
        self.min_size.setSuffix(' px')
        self.min_size.setToolTip('Objects smaller than this are removed when applying')
        checks = QGridLayout()
        checks.addWidget(self.dark, 0, 0)
        checks.addWidget(self.split, 0, 1)
        min_row = QHBoxLayout()
        min_row.addWidget(QLabel('Min object size:'))
        min_row.addWidget(self.min_size)
        min_row.addStretch()

        buttons = QHBoxLayout()
        buttons.addStretch()
        self.action_buttons = []
        for text, slot in (('Auto', self.auto), ('Apply', self.apply), ('Reset', self.reset)):
            b = QPushButton(text)
            self.action_buttons.append(b)
            b.clicked.connect(lambda _=False, s=slot, t=text: self.tiki._run(f'Threshold {t}', s))
            buttons.addWidget(b)
        buttons.addStretch()

        layout = QVBoxLayout(self)
        layout.addLayout(image_row)
        layout.addWidget(self.note_widget)
        layout.addWidget(self.canvas)
        layout.addWidget(self.percent)
        layout.addLayout(grid)
        layout.addLayout(combos)
        layout.addLayout(scope_row)
        layout.addWidget(self.connect_z)
        layout.addLayout(checks)
        layout.addLayout(min_row)
        layout.addLayout(buttons)
        layout.setSpacing(6)
        layout.setSizeConstraint(QVBoxLayout.SetFixedSize)  # shrink-wrap: no gaps when rows hide

        self._redraw = QTimer(self, singleShot=True, interval=30)
        self._redraw.timeout.connect(self.update_preview)

    # ------------------------------------------------------------ target

    def set_target(self, viewer) -> None:
        """Follow the current image window, as ImageJ's dialog does."""
        if viewer is self.viewer:
            self.image_box.refresh()
            return
        self._detach()
        self.viewer = viewer
        if viewer is not None:
            viewer.dims.events.current_step.connect(self._on_step)
            viewer.dims.events.ndisplay.connect(self._on_step)
        self.image_box.set_viewer(viewer, current=roi_mod.target_image(viewer) if viewer else None)
        self._set_image(self.image_box.layer())

    def _set_image(self, image) -> None:
        self._remove_preview()
        self.image = image
        if image is None:
            self.setWindowTitle('Threshold (no image)')
            self.info = None
            return
        self.setWindowTitle(f'Threshold — {image.name}')
        self.info = axes_mod.get(image)
        c = self.info.index('C')
        blocked = c is not None
        self.note_widget.setVisible(blocked)
        if blocked:
            n = image.data.shape[c]
            self.channel_note.setText(f'<b>{image.name}</b> has {n} channels. Threshold works on one '
                                      'channel at a time: split them, then pick a channel above.')
        for w in self.action_buttons + [self.lo_slider, self.hi_slider, self.lo_spin, self.hi_spin, self.scope]:
            w.setEnabled(not blocked)
        if blocked:
            self.percent.setText('')
            return
        z = self.info.index('Z')
        self.scope.blockSignals(True)
        self.scope.clear()
        for key, label in nd.scopes(self.info):
            self.scope.addItem(label, key)
        keys = [self.scope.itemData(i) for i in range(self.scope.count())]
        default = nd.VOLUME if nd.VOLUME in keys else nd.WHOLE if nd.WHOLE in keys else nd.PLANE
        self.scope.setCurrentIndex(keys.index(default))
        self.scope.blockSignals(False)
        self.connect_z.setVisible(z is not None)
        dtype = np.asarray(image.data[(0,) * (image.data.ndim - 2)][:1, :1]).dtype
        decimals = 0 if np.issubdtype(dtype, np.integer) else 4
        for spin in (self.lo_spin, self.hi_spin):
            spin.setDecimals(decimals)
        self._scope_changed(auto=True)

    def _split_channels(self) -> None:
        self.tiki.windows.set_active(self.viewer)
        self.tiki.windows.active.layers.selection.active = self.image
        self.tiki.split_channels()
        self.image_box.refresh()

    def _detach(self) -> None:
        self._remove_preview()
        if self.viewer is not None:
            for ev in (self.viewer.dims.events.current_step, self.viewer.dims.events.ndisplay):
                try:
                    ev.disconnect(self._on_step)
                except Exception:
                    pass
        self.viewer = self.image = None

    def _scope_key(self) -> str:
        return self.scope.currentData() or nd.PLANE

    def _three_d(self) -> bool:
        """Objects are connected through Z for this scope."""
        return (self.info is not None and self.info.index('Z') is not None
                and self.connect_z.isChecked() and self._scope_key() != nd.PLANE)

    def _scope_changed(self, auto: bool = False) -> None:
        if self.image is None or self.info is None or self.info.index('C') is not None:
            return
        self.connect_z.setEnabled(self._scope_key() != nd.PLANE)
        self.min_size.setSuffix(' voxels' if self._three_d() else ' px')
        self.refresh_histogram(keep_range=not auto)
        if auto:
            self.auto()
        self._redraw.start()

    def _on_step(self, event=None) -> None:
        if self.image is None or self.info is None or self.info.index('C') is not None:
            return
        if self._scope_key() != nd.WHOLE:
            self.refresh_histogram(keep_range=True)
        self._redraw.start()

    def _read(self, block) -> np.ndarray:
        data = np.asarray(self.image.data[block])
        return data[..., :3].mean(-1) if self.image.rgb else data

    def _index(self) -> tuple:
        return roi_mod.current_index(self.viewer, self.image)

    def _blocks(self, scope: str | None = None) -> list[tuple]:
        scope = scope or self._scope_key()
        shape = tuple(self.image.data.shape[: self.image.ndim])
        return nd.blocks(self.info, shape, self._index(), scope, by_volume=self._three_d())

    # --------------------------------------------------------- histogram

    def _plane(self) -> np.ndarray:
        return self._read(nd.blocks(self.info, self.image.data.shape[: self.image.ndim], self._index(),
                                    nd.PLANE, False)[0])

    def _hist_values(self) -> np.ndarray:
        """Pixels the histogram and Auto use: the plane, the volume or a stack sample."""
        scope = self._scope_key()
        if scope == nd.PLANE or self.image.ndim < 3:
            return self._plane()
        shape = tuple(self.image.data.shape[: self.image.ndim])
        if scope == nd.VOLUME:
            (block,) = nd.blocks(self.info, shape, self._index(), nd.VOLUME, False)
            z = self.info.index('Z')
            nz = shape[z]
            if nz > VOLUME_SAMPLE_PLANES:  # read an even sample of slices from huge volumes
                block = list(block)
                block[z] = np.unique(np.linspace(0, nz - 1, VOLUME_SAMPLE_PLANES).astype(int))
                block = tuple(block)
            return self._read(block)
        planes = nd.blocks(self.info, shape, self._index(), nd.WHOLE, False)
        picks = np.unique(np.linspace(0, len(planes) - 1, min(len(planes), STACK_SAMPLE_PLANES)).astype(int))
        return np.stack([self._read(planes[i]) for i in picks])

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

    def _volume_preview_ok(self) -> bool:
        """Show the preview as a volume: 3D view, a Z axis right before Y, X."""
        z = self.info.index('Z') if self.info else None
        return (self.viewer.dims.ndisplay == 3 and z is not None and z == self.image.ndim - 3
                and self._scope_key() != nd.PLANE)

    def update_preview(self) -> None:
        if self.image is None or self.viewer is None or self.info is None or self.info.index('C') is not None:
            return
        if self._volume_preview_ok():
            (block,) = nd.blocks(self.info, self.image.data.shape[: self.image.ndim], self._index(), nd.VOLUME, False)
            mask = ops.threshold_mask(self._read(block), self.lo_val, self.hi_val)
        else:
            mask = ops.threshold_mask(self._plane(), self.lo_val, self.hi_val)
        self.percent.setText(f'{100 * mask.mean():.2f} % of this ' + ('volume' if mask.ndim == 3 else 'plane'))
        data = mask.astype(np.uint8)
        if self.preview is not None and (self.preview not in self.viewer.layers or self.preview.ndim != data.ndim):
            self._remove_preview()
        if self.preview is None:
            selected = self.viewer.layers.selection.active
            self.preview = Labels(
                data,
                name=PREVIEW_NAME,
                opacity=0.6,
                colormap=self._colormap(),
                **axes_mod.geometry(self.image, ndim=data.ndim),
            )
            self.viewer.add_layer(self.preview)
            # keep ROI drawing working: give selection back to what had it
            if selected is not None and selected in self.viewer.layers:
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
        if self.info.index('C') is not None:
            raise TikiError('split the channels first and threshold one channel (button at the top)')
        image, viewer, info = self.image, self.viewer, self.info
        split, min_size = self.split.isChecked(), self.min_size.value()
        scope, three_d = self._scope_key(), self._three_d()
        shape = tuple(image.data.shape[: image.ndim])
        z = info.index('Z')
        if scope == nd.PLANE:
            (block,) = self._blocks()
            out = ops.segment(self._read(block), self.lo_val, self.hi_val, split, min_size)
            geometry = axes_mod.geometry(image, ndim=2)
            suffix = f' ({nd.block_label(info, block)})' if image.ndim > 2 else ''
        elif scope == nd.VOLUME and z == image.ndim - 3:
            # one Z-volume, shown at every other position (like a 2D ROI on a movie)
            (block,) = nd.blocks(info, shape, self._index(), nd.VOLUME, False)
            vol = self._read(block)
            if three_d:
                out = ops.segment(vol, self.lo_val, self.hi_val, split, min_size)
            else:
                planes = [(k, slice(None), slice(None)) for k in range(vol.shape[0])]
                out = ops.threshold_blocks(vol, self.lo_val, self.hi_val, planes, split, min_size)
            geometry = axes_mod.geometry(image, ndim=3)
            suffix = f' ({nd.block_label(info, block)})'
        else:
            blocks = self._blocks()
            data = image.data if not image.rgb else _RgbAsGray(image.data)
            with self.tiki.progress('Threshold', len(blocks)) as tick:
                out = ops.threshold_blocks(data, self.lo_val, self.hi_val, blocks, split, min_size, progress=tick)
            if tick.cancelled:
                self.tiki.status('Threshold cancelled')
                return None
            geometry = axes_mod.geometry(image)
            suffix = ''
        self._remove_preview()
        kind = ('3D labels' if three_d else 'labels') if split else 'mask'
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
        n = len(np.unique(out)) - (1 if (out == 0).any() else 0) if split else None
        self.tiki.status(
            f'Threshold {self.lo_val:.4g}–{self.hi_val:.4g}: '
            + (f'{n} {"3D " if three_d else ""}objects' if split else f'{100 * (out > 0).mean():.2f} % of pixels')
            + f' → layer "{layer.name}"'
        )
        return layer

    def closeEvent(self, event):
        self._detach()
        super().closeEvent(event)


class _RgbAsGray:
    """Index an RGB(A) array as its grey-level mean, block by block."""

    def __init__(self, data):
        self.data = data
        self.shape = data.shape[:-1]
        self.ndim = len(self.shape)

    def __getitem__(self, key):
        return np.asarray(self.data[key])[..., :3].mean(-1)
