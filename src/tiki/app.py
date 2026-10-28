"""The TIKI toolbar: ImageJ's main window, driving napari image windows."""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import numpy as np
from napari.layers import Image, Labels, Points, Shapes
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import (
    QAction,
    QActionGroup,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from . import axes as axes_mod
from . import io, ops, style
from . import roi as roi_mod
from .icons import tool_icon
from .manager import WindowManager
from .widgets import ProfileWindow, ResultsTable

# name, tooltip, napari mode, layer kind
TOOLS = [
    ('rectangle', 'Rectangle', 'add_rectangle', 'shapes'),
    ('oval', 'Oval', 'add_ellipse', 'shapes'),
    ('polygon', 'Polygon (double-click to close)', 'add_polygon', 'shapes'),
    ('freehand', 'Freehand', 'add_polygon_lasso', 'shapes'),
    ('line', 'Straight line', 'add_line', 'shapes'),
    ('point', 'Point', 'add', 'points'),
    ('select', 'Move / edit ROI', 'select', 'shapes'),
    ('hand', 'Scrolling tool (pan & zoom)', 'pan_zoom', None),
    ('brush', 'Paint labels (brush size in the layer controls)', 'paint', 'labels'),
    ('eraser', 'Erase labels', 'erase', 'labels'),
]
DRAW_MODES = {'add_rectangle', 'add_ellipse', 'add_polygon', 'add_polygon_lasso', 'add_line'}

STATUS_HOLD_MS = 4000


class TikiError(Exception):
    """An expected problem the user can fix (no image open, wrong ROI type...).

    ``fix`` names a command that resolves it (e.g. ``'properties'``); the
    error dialog then offers a button for it.
    """

    def __init__(self, message: str, fix: str | None = None):
        super().__init__(message)
        self.fix = fix


FIXES = {'properties': ('Open Properties…', 'properties')}


class Tiki(QMainWindow):
    def __init__(self):
        super().__init__()
        style.apply(self)
        self.setWindowTitle('TIKI')
        self.setAcceptDrops(True)
        self.tool = 'hand'  # nothing gets drawn until a drawing tool is picked
        self.results = ResultsTable()
        self.profiles: list[ProfileWindow] = []
        self._message_until = 0.0
        self._actions: dict[str, QAction] = {}
        self.threshold_window = None
        self.roi_manager = None
        self.traces: list = []  # open TracesWindows (exported by Send to Notebook)

        self._build_menus()
        self.windows = WindowManager(
            reserved_shortcuts=[a.shortcut().toString() for a in self._actions.values() if not a.shortcut().isEmpty()]
        )
        self.windows.active_changed.connect(self._on_active_changed)
        self.windows.cursor_moved.connect(self._on_cursor)
        self.windows.ndisplay_changed.connect(self._on_ndisplay)
        self._build_toolbar()
        self.status('TIKI Is Kinda ImageJ — File ▸ Open or drop an image here')

    # ------------------------------------------------------------------ UI

    def _build_toolbar(self) -> None:
        central = QWidget()
        row = QHBoxLayout(central)
        row.setContentsMargins(4, 2, 4, 2)
        row.setSpacing(1)
        self.tool_group = QActionGroup(self)
        self.tool_group.setExclusive(True)
        for name, tip, _mode, _kind in TOOLS:
            action = QAction(tool_icon(name), tip, self, checkable=True)
            action.setData(name)
            action.triggered.connect(lambda _=False, n=name: self.set_tool(n))
            self.tool_group.addAction(action)
            button = QToolButton()
            button.setDefaultAction(action)
            button.setAutoRaise(True)
            button.setFixedSize(30, 30)
            row.addWidget(button)
            if name == self.tool:
                action.setChecked(True)
        row.addStretch()
        self.setCentralWidget(central)
        self.status_label = QLabel()
        self.status_label.setMinimumWidth(420)
        self.statusBar().addWidget(self.status_label, 1)
        self.statusBar().setSizeGripEnabled(False)
        self.resize(560, self.sizeHint().height())

    def _add(self, menu: QMenu, text: str, slot, shortcut: str | None = None) -> QAction:
        action = menu.addAction(text)
        action.triggered.connect(lambda _=False: self._run(text, slot))
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
            action.setShortcutContext(Qt.ApplicationShortcut)
        self._actions[f'{menu.title()}>{text}'.replace('&', '')] = action
        return action

    def _build_menus(self) -> None:
        mb = self.menuBar()
        m = mb.addMenu('&File')
        self._add(m, 'Open…', self.open_dialog, 'Ctrl+O')
        samples = m.addMenu('Open Samples')
        self._add(samples, 'Synthetic movie', self.open_sample_movie)
        self._add(samples, 'Blobs', lambda: self._open_skimage('blobs'))
        self._add(samples, 'Camera', lambda: self._open_skimage('camera'))
        m.addSeparator()
        self._add(m, 'Close', self.close_active, 'Ctrl+W')
        self._add(m, 'Close All', self.windows_close_all, 'Ctrl+Shift+W')
        self._add(m, 'Save As TIFF…', self.save_as, 'Ctrl+S')
        export = m.addMenu('Export')
        self._add(export, 'Send to Notebook…', self.send_to_notebook, 'Ctrl+Shift+N')
        m.addSeparator()
        self._add(m, 'Quit', self.close, 'Ctrl+Q')

        m = mb.addMenu('&Edit')
        sel = m.addMenu('Selection')
        self._add(sel, 'Select All', self.select_all, 'Ctrl+A')
        self._add(sel, 'Select None', self.select_none, 'Ctrl+Shift+A')

        m = mb.addMenu('&Image')
        adjust = m.addMenu('Adjust')
        self._add(adjust, 'Brightness/Contrast…', self.toggle_layer_controls, 'Ctrl+Shift+C')
        self._add(adjust, 'Threshold…', self.threshold, 'Ctrl+Shift+T')
        self._add(adjust, 'Enhance Contrast', self.enhance_contrast, 'Ctrl+Shift+E')
        self._add(adjust, 'Reset Display Range', self.reset_contrast)
        m.addSeparator()
        self._add(m, 'Show Info…', self.show_info, 'Ctrl+I')
        self._add(m, 'Properties…', self.properties, 'Ctrl+Shift+P')
        color = m.addMenu('Color')
        self._add(color, 'Split Channels', self.split_channels)
        self._add(color, 'Merge Channels', self.merge_channels)
        self._add(color, 'Channels Tool…', self.channels_tool, 'Ctrl+Shift+Z')
        self._add(m, 'Crop', self.crop, 'Ctrl+Shift+X')
        self._add(m, 'Duplicate…', self.duplicate, 'Ctrl+Shift+D')
        stacks = m.addMenu('Stacks')
        self._add(stacks, 'Z Project…', self.z_project_dialog)
        self._add(stacks, 'Plot Z-axis Profile', self.plot_z_profile, 'Ctrl+Shift+Y')
        stacks.addSeparator()
        self._add(stacks, 'Split Channels ', self.split_channels)  # same command as Image ▸ Color

        m = mb.addMenu('&Process')
        filt = m.addMenu('Filters')
        self._add(filt, 'Gaussian Blur…', lambda: self.filter_dialog('gaussian'))
        self._add(filt, 'Median…', lambda: self.filter_dialog('median'))
        self._add(m, 'Subtract Background…', lambda: self.filter_dialog('background'))

        m = mb.addMenu('&Analyze')
        self._add(m, 'Measure', self.measure, 'Ctrl+M')
        self._add(m, 'Analyze Particles…', self.analyze_particles)
        self._add(m, 'Clear Results', self.results.clear)
        self._add(m, 'Show Results', self.show_results)
        m.addSeparator()
        self._add(m, 'Plot Profile', self.plot_line_profile, 'Ctrl+K')
        traces = m.addMenu('Traces')
        self._add(traces, 'From ROI Manager', self.traces_from_manager)
        self._add(traces, 'From Labels', self.traces_from_labels)
        tools = m.addMenu('Tools')
        self._add(tools, 'ROI Manager…', self.show_roi_manager)
        self._add(tools, 'Add to Manager', self.add_to_manager, 'Ctrl+T')

        self.window_menu = mb.addMenu('&Window')
        self.window_menu.aboutToShow.connect(self._fill_window_menu)

        m = mb.addMenu('&Help')
        self._add(m, 'Find Commands…', self.find_commands, 'Ctrl+L')
        self._add(m, 'About TIKI', self.about)

    def _fill_window_menu(self) -> None:
        menu = self.window_menu
        menu.clear()
        toggle = menu.addAction('Show Layer List')
        toggle.triggered.connect(self.toggle_layer_list)
        menu.addAction('Show Results').triggered.connect(self.show_results)
        menu.addSeparator()
        if not self.windows.viewers:
            menu.addAction('(no images)').setEnabled(False)
        for viewer in self.windows.viewers:
            action = menu.addAction(viewer.title)
            action.setCheckable(True)
            action.setChecked(viewer is self.windows.active)
            action.triggered.connect(lambda _=False, v=viewer: self.windows.activate(v))

    # ------------------------------------------------------------- status

    def status(self, text: str, hold: bool = True) -> None:
        self.status_label.setText(text)
        if hold:
            self._message_until = time.monotonic() + STATUS_HOLD_MS / 1000

    def _run(self, name: str, slot) -> None:
        """Run a menu command; problems become a message box, never a crash.

        PyQt6 aborts the whole process on an exception escaping a slot, so
        nothing is re-raised here.
        """
        name = name.rstrip('…')
        try:
            slot()
        except TikiError as exc:
            self.report(name, str(exc), fix=exc.fix)
        except Exception as exc:
            traceback.print_exc()
            self.report(name, f'{type(exc).__name__}: {exc}\n\n(full traceback in the terminal)', error=True)

    def report(self, title: str, message: str, error: bool = False, fix: str | None = None) -> None:
        """Show a problem the way ImageJ does: a small dialog plus the status bar.

        With ``fix`` the dialog gets a button that runs the fixing command.
        """
        self.status(f'{title}: {message.splitlines()[0]}')
        if fix in FIXES:
            label, method = FIXES[fix]
            box = QMessageBox(QMessageBox.Warning, title, message, QMessageBox.Close, self)
            button = box.addButton(label, QMessageBox.ActionRole)
            box.exec_()
            if box.clickedButton() is button:
                self._run(label, getattr(self, method))
            return
        box = QMessageBox.critical if error else QMessageBox.warning
        box(self, title, message)

    def _on_cursor(self, viewer, position) -> None:
        if time.monotonic() < self._message_until:
            return
        image = roi_mod.target_image(viewer)
        if image is None:
            return
        ndim = image.ndim
        coords = image.world_to_data(position[-ndim:])
        shape = image.data.shape[:ndim]
        idx = tuple(int(np.floor(c + 0.5)) for c in coords)
        parts = [f'x={idx[-1]}', f'y={idx[-2]}']
        for d in range(ndim - 2):
            parts.append(f'{roi_mod.axis_name(viewer, image, d)}={idx[d]}')
        if all(0 <= i < s for i, s in zip(idx, shape)):
            value = image.data[idx]
            if np.ndim(value) == 0:
                v = value.item()
                parts.append(f'value={v:.4g}' if isinstance(v, float) else f'value={v}')
            else:
                parts.append('value=' + ','.join(str(x) for x in np.asarray(value).tolist()))
        self.status_label.setText(f'{viewer.title}   ' + '  '.join(parts))

    # ----------------------------------------------------------------- tools

    def set_tool(self, name: str) -> None:
        self.tool = name
        for action in self.tool_group.actions():
            action.setChecked(action.data() == name)
        if self.windows.active is not None:
            self.apply_tool(self.windows.active)
        tip = next(t for n, t, _, _ in TOOLS if n == name)
        self.status(tip)

    def apply_tool(self, viewer) -> None:
        """Put ``viewer`` in the mode for the current tool (ImageJ tools are global).

        In 3D view the mouse rotates the volume, so drawing tools stand down
        (otherwise a drag meant to rotate draws shapes in 3D space).
        """
        if roi_mod.target_image(viewer) is None:
            return
        _, _, mode, kind = next(t for t in TOOLS if t[0] == self.tool)
        if viewer.dims.ndisplay == 3:
            for layer in viewer.layers:
                if isinstance(layer, (Shapes, Points, Labels)) and layer.mode != 'pan_zoom':
                    layer.mode = 'pan_zoom'
            viewer.layers.selection.active = roi_mod.target_image(viewer)
            return
        if kind == 'shapes':
            layer = self._roi(viewer)
        elif kind == 'points':
            layer = roi_mod.points_layer(viewer)
        elif kind == 'labels':
            layer = self._labels_for_painting(viewer)
            self.windows.show_layer_controls(viewer)
        else:
            for name in (roi_mod.ROI_NAME, roi_mod.POINTS_NAME):
                if name in viewer.layers:
                    viewer.layers[name].mode = 'pan_zoom'
            return
        viewer.layers.selection.active = layer
        layer.mode = mode

    def _labels_for_painting(self, viewer) -> Labels:
        """The selected Labels layer, else the topmost one, else a new empty one."""
        from .threshold import PREVIEW_NAME

        active = viewer.layers.selection.active
        if isinstance(active, Labels) and active.name != PREVIEW_NAME:
            return active
        labels = [ly for ly in viewer.layers if isinstance(ly, Labels) and ly.name != PREVIEW_NAME]
        if labels:
            return labels[-1]
        image = roi_mod.target_image(viewer)
        shape = image.data.shape[: image.ndim]
        geometry = axes_mod.geometry(image)
        if np.prod(shape) * 2 > 1 * 2**30:  # >1 GB for a whole movie: paint one plane
            shape, geometry = shape[-2:], axes_mod.geometry(image, ndim=2)
            self.status('Movie too large for full-size labels; painting a single 2D plane')
        layer = Labels(np.zeros(shape, np.uint16), name='labels', opacity=0.6, **geometry)
        viewer.add_layer(layer)
        roi = roi_mod.roi_layer(viewer, create=False)
        if roi is not None:
            viewer.layers.move(viewer.layers.index(roi), len(viewer.layers))
        return layer

    def _roi(self, viewer) -> Shapes:
        existed = roi_mod.roi_layer(viewer, create=False) is not None
        layer = roi_mod.roi_layer(viewer)
        if not existed:
            layer.mouse_drag_callbacks.append(self._replace_previous_roi)
        return layer

    def _replace_previous_roi(self, layer, event):
        """ImageJ keeps one selection: a new ROI replaces the old (Shift adds)."""
        if layer.mode not in DRAW_MODES:
            return
        keep_all = 'Shift' in event.modifiers
        yield
        while event.type == 'mouse_move':
            yield
        if not keep_all:
            QTimer.singleShot(0, lambda: self._prune(layer))

    @staticmethod
    def _prune(layer: Shapes) -> None:
        if layer._is_creating or layer.nshapes < 2:
            return
        layer.selected_data = set(range(layer.nshapes - 1))
        layer.remove_selected()

    def _on_ndisplay(self, viewer) -> None:
        self.apply_tool(viewer)
        if viewer.dims.ndisplay == 3:
            image = roi_mod.target_image(viewer)
            note = ''
            if image is not None and axes_mod.get(image).z_assumed:
                note = ' z-step unknown, so voxels are drawn as cubes: set it in Image ▸ Properties.'
            self.status('3D view: drag to rotate, shift-drag to pan; drawing tools work in 2D.' + note)
            self._message_until = time.monotonic() + 8

    def _on_active_changed(self, viewer) -> None:
        if viewer is not None:
            self.apply_tool(viewer)
        if self.threshold_window is not None and self.threshold_window.isVisible():
            self.threshold_window.set_target(viewer)
        if self.roi_manager is not None and self.roi_manager.isVisible():
            self.roi_manager.refresh_overlay(viewer)
        self._refresh_channels_tool(viewer)

    # ------------------------------------------------------------ helpers

    def require_image(self):
        viewer, image = self.windows.active_image()
        if image is None:
            raise TikiError('no image open')
        return viewer, image

    def open_data(self, data, name, **kwargs):
        viewer = self.windows.new_window(data, name, **kwargs)
        self.apply_tool(viewer)
        return viewer

    def current_roi(self, viewer):
        return roi_mod.current_shape(roi_mod.roi_layer(viewer, create=False))

    # ------------------------------------------------------------- File

    def open_paths(self, paths) -> None:
        for path in paths:
            t0 = time.perf_counter()
            try:
                data, meta = io.read_image(path)
            except Exception as exc:
                traceback.print_exc()
                self.report('Open', f'Could not open {Path(path).name}:\n{exc}')
                continue
            metadata = {'path': meta['path']}
            if meta.get('imagej_luts') is not None:
                metadata['imagej_luts'] = meta['imagej_luts']
            info = meta['axes']
            self.open_data(data, meta['name'], axes=info, metadata=metadata)
            msg = f'Opened {meta["name"]}: {info.describe()} {data.dtype} [{info.source}]'
            if info.notes:
                msg += ' — ' + '; '.join(info.notes)
            if not info.confident:
                msg += ' — check Image ▸ Properties (Ctrl+Shift+P)'
            self.status(msg)
            if not info.confident or info.notes:
                self._message_until = time.monotonic() + 10
            if 'warning' in meta:
                self.report('Open', f'{meta["name"]}: {meta["warning"]}')

    def open_dialog(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, 'Open', str(Path.home()), io.OPEN_FILTER)
        if paths:
            self.open_paths(paths)

    def open_sample_movie(self) -> None:
        self.open_data(io.synthetic_movie(), 'synthetic_movie.tif')

    def _open_skimage(self, name: str) -> None:
        from skimage import data

        self.open_data(getattr(data, name)(), f'{name}.tif')

    def close_active(self) -> None:
        viewer = self.windows.active
        if viewer is not None:
            self.windows.close(viewer)

    def windows_close_all(self) -> None:
        self.windows.close_all()

    def save_as(self) -> None:
        viewer, image = self.require_image()
        default = Path(image.metadata.get('path', image.name)).with_suffix('.tif').name
        path, _ = QFileDialog.getSaveFileName(self, 'Save As TIFF', default, 'TIFF (*.tif)')
        if path:
            scale = tuple(image.scale) if np.any(np.asarray(image.scale) != 1) else None
            io.save_tiff(path, image.data, scale)
            self.status(f'Saved {path}')

    # ------------------------------------------------------------- Edit

    def select_all(self) -> None:
        viewer, image = self.require_image()
        h, w = roi_mod.spatial_shape(image)
        layer = self._roi(viewer)
        corners = [[-0.5, -0.5], [-0.5, w - 0.5], [h - 0.5, w - 0.5], [h - 0.5, -0.5]]
        layer.data = []
        layer.add_rectangles(np.array([corners], dtype=float))

    def select_none(self) -> None:
        viewer = self.windows.active
        layer = roi_mod.roi_layer(viewer, create=False) if viewer else None
        if layer is not None and layer.nshapes:
            layer.data = []

    # ------------------------------------------------------------- Image

    def _set_dock(self, dock_name: str) -> None:
        viewer = self.windows.active
        if viewer is None:
            raise TikiError('no image open')
        dock = getattr(viewer.window._qt_viewer, dock_name)
        dock.setVisible(not dock.isVisible())

    def toggle_layer_controls(self) -> None:
        viewer, image = self.require_image()
        viewer.layers.selection.active = image
        self._set_dock('dockLayerControls')

    def toggle_layer_list(self) -> None:
        self._set_dock('dockLayerList')

    def properties(self) -> None:
        from .properties import PropertiesDialog

        viewer, image = self.require_image()
        dlg = PropertiesDialog(self, viewer, image)
        if dlg.exec_() == QDialog.Accepted:
            info = dlg.apply()
            self.status(f'{image.name}: {info.describe()}')

    def split_channels(self) -> None:
        from . import channels

        viewer, image = self.require_image()
        layers = channels.split(viewer, image)
        self.reapply_tool(viewer)
        self._refresh_channels_tool(viewer)
        self.status(f'Split {image.name} into {len(layers)} channel layers (Image ▸ Color ▸ Merge Channels undoes)')

    def merge_channels(self) -> None:
        from . import channels

        viewer, _ = self.require_image()
        layer = channels.merge(viewer)
        self.reapply_tool(viewer)
        self._refresh_channels_tool(viewer)
        self.status(f'Merged channels back into {layer.name}')

    def channels_tool(self) -> None:
        from .channels import ChannelsTool

        if getattr(self, 'channels_window', None) is None:
            self.channels_window = ChannelsTool(self)
        self.channels_window.refresh()
        self.channels_window.show()
        self.channels_window.raise_()

    def _refresh_channels_tool(self, viewer=None) -> None:
        win = getattr(self, 'channels_window', None)
        if win is not None and win.isVisible():
            win.refresh(viewer)

    def threshold(self) -> None:
        from .threshold import ThresholdWindow

        viewer, image = self.require_image()
        if self.threshold_window is None:
            self.threshold_window = ThresholdWindow(self)
        win = self.threshold_window
        win.show()
        win.raise_()
        win.set_target(viewer)
        win.update_preview()

    def enhance_contrast(self) -> None:
        viewer, image = self.require_image()
        current = self.current_roi(viewer)
        plane = roi_mod.current_plane(viewer, image)
        if current is not None and current[1] not in ('line', 'path'):
            plane = plane[roi_mod.shape_mask(*current, plane.shape)]
        lo, hi = ops.saturated_limits(plane)
        image.contrast_limits_range = (min(lo, image.contrast_limits_range[0]), max(hi, image.contrast_limits_range[1]))
        image.contrast_limits = (lo, hi)
        self.status(f'Display range {lo:.4g}–{hi:.4g} (0.35% saturated)')

    def reset_contrast(self) -> None:
        viewer, image = self.require_image()
        image.reset_contrast_limits()

    def show_info(self) -> None:
        viewer, image = self.require_image()
        data = image.data
        plane = roi_mod.current_plane(viewer, image)
        lines = [
            f'Title: {image.name}',
            f'Path: {image.metadata.get("path", "(not saved)")}',
            f'Shape: {data.shape}',
            f'Type: {data.dtype}',
            f'Size: {data.nbytes / 2**20:.1f} MB' + (' (memory-mapped)' if isinstance(data, np.memmap) else ''),
            f'Scale: {tuple(float(s) for s in image.scale)}',
            f'Current slice: {roi_mod.current_index(viewer, image)[:-2]}',
            f'Slice min/max: {plane.min():.4g} / {plane.max():.4g}',
        ]
        QMessageBox.information(self, f'Info for {image.name}', '\n'.join(lines))

    def _roi_box(self, viewer, image):
        current = self.current_roi(viewer)
        if current is None or current[1] in ('line', 'path'):
            return None
        return roi_mod.shape_box(current[0], roi_mod.spatial_shape(image))

    def duplicate(self) -> None:
        viewer, image = self.require_image()
        box = self._roi_box(viewer, image)
        data = image.data if box is None else ops.crop_to_box(image.data, box)
        stem = Path(image.name).stem
        self.open_data(np.array(data), f'{stem}-1.tif', axes=axes_mod.fit(axes_mod.get(image), data.shape))

    def crop(self) -> None:
        viewer, image = self.require_image()
        box = self._roi_box(viewer, image)
        if box is None:
            raise TikiError('Crop needs an area selection')
        image.data = np.array(ops.crop_to_box(image.data, box))
        self.select_none()
        viewer.reset_view()

    def z_project_dialog(self) -> None:
        viewer, image = self.require_image()
        if image.ndim < 3:
            raise TikiError('Z Project needs a stack')
        dlg = QDialog(self)
        dlg.setWindowTitle('ZProjection')
        form = QFormLayout(dlg)
        info = axes_mod.get(image)
        axis_box = QComboBox()
        for d in range(image.ndim - 2):
            if info.axes[d].role == 'C':
                continue  # never average channels together
            axis_box.addItem(f'{roi_mod.axis_name(viewer, image, d)} ({image.data.shape[d]})', d)
        if axis_box.count() == 0:
            raise TikiError('no axis to project along (only channels besides Y and X)', fix='properties')
        default = axes_mod.stack_axis(info, prefer='ZT')
        if default is not None and axis_box.findData(default) >= 0:
            axis_box.setCurrentIndex(axis_box.findData(default))
        start, stop = QSpinBox(), QSpinBox()
        method = QComboBox()
        method.addItems(list(ops.PROJECTIONS))

        def update_range():
            n = image.data.shape[axis_box.currentData()]
            for spin in (start, stop):
                spin.setRange(1, n)
            start.setValue(1)
            stop.setValue(n)

        axis_box.currentIndexChanged.connect(update_range)
        update_range()
        if axis_box.count() > 1:
            form.addRow('Project along:', axis_box)
        form.addRow('Start slice:', start)
        form.addRow('Stop slice:', stop)
        form.addRow('Projection type:', method)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)
        if dlg.exec_() != QDialog.Accepted:
            return
        self.z_project(image, ops.PROJECTIONS[method.currentText()], axis_box.currentData(),
                       start.value() - 1, stop.value())

    def z_project(self, image: Image, method: str, axis: int = 0, start: int = 0, stop: int | None = None):
        t0 = time.perf_counter()
        sl = [slice(None)] * image.data.ndim
        sl[axis] = slice(start, stop)
        data = image.data[tuple(sl)]
        result = ops.z_project(data, method, axis)
        prefix = {'max': 'MAX', 'mean': 'AVG', 'sum': 'SUM', 'std': 'STD', 'median': 'MED', 'min': 'MIN'}[method]
        info = axes_mod.fit(axes_mod.get(image), data.shape).without(axis)
        viewer = self.open_data(result, f'{prefix}_{image.name}', axes=info)
        self.status(f'Z Project ({method}) in {time.perf_counter() - t0:.2f} s')
        return viewer

    def plot_z_profile(self) -> None:
        viewer, image = self.require_image()
        axis = self._stack_axis(viewer, image)
        window = ProfileWindow(viewer, image, axis)
        window.destroyed.connect(lambda _=None, w=window: self.profiles.remove(w) if w in self.profiles else None)
        window.setAttribute(Qt.WA_DeleteOnClose)
        self.profiles.append(window)
        window.show()
        # drawing the ROI needs the image window, not the plot, to be on top
        self.windows.activate(viewer)

    # ----------------------------------------------------------- Analyze

    def measure(self) -> None:
        viewer, image = self.require_image()
        plane = roi_mod.current_plane(viewer, image)
        index = roi_mod.current_index(viewer, image)
        base = {'Label': image.name}
        if image.ndim > 2:
            base['Slice'] = ','.join(str(i + 1) for i in index[:-2])

        points = roi_mod.points_layer(viewer, create=False)
        if self.tool == 'point' and points is not None and len(points.data):
            for p in points.data:
                y, x = (int(np.floor(c + 0.5)) for c in p[-2:])
                if 0 <= y < plane.shape[0] and 0 <= x < plane.shape[1]:
                    self.results.add_row(base | {'X': x, 'Y': y, 'Value': float(plane[y, x])})
            self.show_results()
            return

        current = self.current_roi(viewer)
        mask = None if current is None else roi_mod.shape_mask(*current, plane.shape)
        row = base | ops.mask_stats(plane, mask, roi_mod.pixel_area(image))
        if current is not None and current[1] in ('line', 'path'):
            verts = np.asarray(current[0])[:, -2:] * np.asarray(image.scale[-2:])
            row['Length'] = float(np.linalg.norm(np.diff(verts, axis=0), axis=1).sum())
        self.results.add_row(row)
        self.show_results()

    def show_results(self) -> None:
        self.results.show()
        self.results.raise_()

    def plot_line_profile(self) -> None:
        from skimage.measure import profile_line

        from .widgets import PlotWindow

        viewer, image = self.require_image()
        current = self.current_roi(viewer)
        if current is None or current[1] != 'line':
            raise TikiError('Plot Profile needs a straight line selection')
        src, dst = np.asarray(current[0])[:, -2:]
        plane = roi_mod.current_plane(viewer, image)
        values = profile_line(plane, src, dst, order=1, mode='constant', reduce_func=None)
        dist = np.linspace(0, np.linalg.norm((dst - src) * image.scale[-2:]), len(values))
        win = PlotWindow(f'Plot of {image.name}', dist, values, 'distance', 'gray value')
        win.setAttribute(Qt.WA_DeleteOnClose)
        self.profiles.append(win)
        win.destroyed.connect(lambda _=None, w=win: self.profiles.remove(w) if w in self.profiles else None)
        win.show()

    # ----------------------------------------------- ROI Manager & traces

    def roi_layer(self, viewer) -> Shapes:
        return self._roi(viewer)

    def reapply_tool(self, viewer) -> None:
        """After adding a layer: keep the live ROI on top and the current tool armed."""
        roi = roi_mod.roi_layer(viewer, create=False)
        if roi is not None:
            viewer.layers.move(viewer.layers.index(roi), len(viewer.layers))
        self.apply_tool(viewer)

    def labels_layer(self, viewer) -> Labels:
        """The selected Labels layer, else the topmost one (previews excluded)."""
        active = viewer.layers.selection.active
        if isinstance(active, Labels) and active.name not in roi_mod.PREVIEW_NAMES:
            return active
        labels = [ly for ly in viewer.layers if isinstance(ly, Labels) and ly.name not in roi_mod.PREVIEW_NAMES]
        if not labels:
            raise TikiError('no Labels layer in this window (use Image ▸ Adjust ▸ Threshold… and Apply)')
        return labels[-1]

    def labels_plane(self, viewer, layer: Labels) -> np.ndarray:
        """The 2D labels on the current slice (a 2D layer is used as is)."""
        data = layer.data
        if data.ndim == 2:
            return np.asarray(data)
        point = layer.world_to_data(viewer.dims.point[-layer.ndim:])
        idx = tuple(int(np.clip(np.round(point[d]), 0, data.shape[d] - 1)) for d in range(data.ndim - 2))
        return np.asarray(data[idx])

    def show_roi_manager(self):
        from .roi_manager import RoiManager

        if self.roi_manager is None:
            self.roi_manager = RoiManager(self)
        self.roi_manager.show()
        self.roi_manager.raise_()
        return self.roi_manager

    def add_to_manager(self) -> None:
        self.show_roi_manager().add_current()

    def _stack_axis(self, viewer, image) -> int:
        """Axis for profiles/traces: T if there is one, else Z, else an unknown stack axis."""
        if image.ndim < 3:
            raise TikiError('this needs a stack or movie, not a single image')
        axis = axes_mod.stack_axis(axes_mod.get(image))
        if axis is None:
            raise TikiError('this image has no T or Z axis to run along (only channels?)', fix='properties')
        return axis

    def show_traces(self, viewer, image, names, masks=None, labels=None):
        """Compute traces (masks list or a 2D labels image) and open a TracesWindow."""
        from .widgets import TracesWindow

        axis = self._stack_axis(viewer, image)
        index = roi_mod.current_index(viewer, image)
        data = image.data
        n_t = data.shape[axis]
        t0 = time.perf_counter()
        with self.progress('Traces', n_t) as tick:
            if labels is not None:
                ids, traces = ops.label_traces(data, axis, index, labels, progress=tick)
                names = [f'label {int(i)}' for i in ids]
            else:
                traces = ops.roi_traces(data, axis, index, masks, progress=tick)
        if tick.cancelled:
            self.status('Traces cancelled')
            return None
        if len(names) == 0:
            raise TikiError('nothing to plot')
        win = TracesWindow(f'Traces — {image.name}', names, traces, viewer, image, axis,
                           xlabel=roi_mod.axis_name(viewer, image, axis))
        win.setAttribute(Qt.WA_DeleteOnClose)
        self.traces.append(win)
        self.profiles.append(win)
        win.destroyed.connect(lambda _=None, w=win: [lst.remove(w) for lst in (self.traces, self.profiles) if w in lst])
        win.show()
        self.status(f'{len(names)} traces x {n_t} frames in {time.perf_counter() - t0:.1f} s')
        return win

    def traces_from_manager(self) -> None:
        self.show_roi_manager().multi_plot()

    def traces_from_labels(self) -> None:
        viewer, image = self.require_image()
        layer = self.labels_layer(viewer)
        self.show_traces(viewer, image, [], labels=self.labels_plane(viewer, layer))

    def progress(self, title: str, total: int):
        return _Progress(self, title, total)

    # ------------------------------------------------- Analyze Particles

    def analyze_particles(self) -> None:
        viewer, image = self.require_image()
        layer = self.labels_layer(viewer)
        dlg = QDialog(self)
        dlg.setWindowTitle('Analyze Particles')
        form = QFormLayout(dlg)
        form.addRow(QLabel(f'Labels: <b>{layer.name}</b><br>Intensity: <b>{image.name}</b>'))
        min_size = QSpinBox()
        min_size.setRange(0, 10**7)
        min_size.setSuffix(' px')
        form.addRow('Min size:', min_size)
        exclude = QCheckBox('Exclude on edges')
        clear = QCheckBox('Clear results', checked=True)
        stack = QCheckBox('Whole stack')
        form.addRow(exclude)
        form.addRow(clear)
        if layer.data.ndim > 2:
            form.addRow(stack)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)
        if dlg.exec_() != QDialog.Accepted:
            return
        self.particles(viewer, image, layer, min_size.value(), exclude.isChecked(), clear.isChecked(),
                       stack.isChecked() and layer.data.ndim > 2)

    def particles(self, viewer, image, layer, min_size=0, exclude_edges=False, clear=True, whole_stack=False):
        from skimage.segmentation import clear_border

        if clear:
            self.results.clear()
        sy, sx = image.scale[-2:]
        if whole_stack:
            lead = layer.data.shape[:-2]
            planes = [(idx, np.asarray(layer.data[idx]), np.asarray(image.data[idx])) for idx in np.ndindex(*lead)]
        else:
            idx = roi_mod.current_index(viewer, image)[:-2]
            planes = [(idx, self.labels_plane(viewer, layer), roi_mod.current_plane(viewer, image))]
        total = 0
        for idx, lab, img in planes:
            if lab.max() == 1:  # a 0/1 mask: split into objects first
                lab = ops.label_objects(lab > 0)
            if exclude_edges:
                lab = clear_border(lab)
            if min_size > 1:
                sizes = np.bincount(lab.ravel())
                lab = np.where(sizes[lab] >= min_size, lab, 0)
            for row in ops.particle_table(lab, img, (sy, sx)):
                if idx:
                    row = {'Slice': ','.join(str(i + 1) for i in idx)} | row
                self.results.add_row(row)
                total += 1
        self.show_results()
        self.status(f'Analyze Particles: {total} objects from "{layer.name}"')
        return total

    # --------------------------------------------------- Process filters

    def filter_dialog(self, key: str) -> None:
        from .filters import FilterDialog, run_filter

        viewer, image = self.require_image()
        dlg = FilterDialog(self, key, viewer, image)
        if dlg.exec_() != QDialog.Accepted:
            return
        run_filter(self, key, viewer, image, dlg.values(), dlg.whole_stack.isChecked())

    # ---------------------------------------------------- Send to notebook

    def send_to_notebook(self, folder: str | None = None) -> str:
        from .export import export_session

        viewer, image = self.require_image()
        if folder is None:
            src = image.metadata.get('path')
            start = Path(src).parent if src else Path.home()
            base = QFileDialog.getExistingDirectory(self, 'Export session into folder', str(start))
            if not base:
                return ''
            folder = str(Path(base) / f'{Path(image.name).stem}_tiki')
        manager = self.roi_manager.rois if self.roi_manager is not None else []
        traces = [(w.windowTitle(), w.names, w.traces) for w in self.traces if w.viewer is viewer]
        out, snippet = export_session(folder, viewer, image, manager, traces,
                                      self.results.rows, self.results.columns)
        QApplication.clipboard().setText(snippet)
        self.status(f'Exported to {out}; loader code copied to clipboard')
        if not getattr(self, '_quiet_export', False):
            QMessageBox.information(
                self, 'Send to Notebook',
                f'Wrote {out}\n\nPaste into a notebook (already on your clipboard):\n\n'
                f"exec(open(r'{out / 'load_session.py'}').read())\n\nor open load_session.py.",
            )
        return snippet

    # -------------------------------------------------------------- Help

    def find_commands(self) -> None:
        dlg = CommandFinder(self._actions, self)
        dlg.exec_()

    def about(self) -> None:
        QMessageBox.about(
            self,
            'About TIKI',
            'TIKI Is Kinda ImageJ\n\nAn ImageJ-style toolbar for napari image windows.',
        )

    # ------------------------------------------------------------ Qt events

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        self._run('Open', lambda: self.open_paths(paths))

    def closeEvent(self, event):
        self.windows.close_all()
        self.results.close()
        for w in (self.threshold_window, self.roi_manager, getattr(self, 'channels_window', None)):
            if w is not None:
                w.close()
        for w in list(self.profiles):
            w.close()
        event.accept()
        QApplication.instance().quit()


def install_excepthook(tiki: Tiki) -> None:
    """Catch errors outside menu commands (mouse callbacks, timers, live plots).

    Without this PyQt6 calls abort() on any exception raised inside Qt.
    """

    def hook(exc_type, exc, tb):
        traceback.print_exception(exc_type, exc, tb)
        try:
            tiki.status(f'Error: {exc_type.__name__}: {exc} (see terminal)')
        except Exception:
            pass

    sys.excepthook = hook


class _Progress:
    """``with tiki.progress(title, total) as tick:`` - tick(done, total) -> keep going?"""

    def __init__(self, parent, title: str, total: int):
        from qtpy.QtWidgets import QProgressDialog

        self.cancelled = False
        self.dialog = QProgressDialog(f'{title}…', 'Cancel', 0, max(total, 1), parent)
        self.dialog.setWindowTitle(title)
        self.dialog.setWindowModality(Qt.WindowModal)
        self.dialog.setMinimumDuration(400)

    def __enter__(self):
        return self

    def __call__(self, done: int, total: int) -> bool:
        self.dialog.setMaximum(max(total, 1))
        self.dialog.setValue(done)
        QApplication.processEvents()
        self.cancelled = self.dialog.wasCanceled()
        return not self.cancelled

    def __exit__(self, *exc):
        self.dialog.reset()
        self.dialog.close()
        return False


class CommandFinder(QDialog):
    """ImageJ's Find Commands (Ctrl+L): type to filter, Enter to run."""

    def __init__(self, actions: dict[str, QAction], parent=None):
        super().__init__(parent)
        self.setWindowTitle('Find Commands')
        self.resize(380, 360)
        self.actions = actions
        self.query = QLineEdit(placeholderText='Search commands…')
        self.list = QListWidget()
        layout = QVBoxLayout(self)
        layout.addWidget(self.query)
        layout.addWidget(self.list)
        self.query.textChanged.connect(self.refresh)
        self.query.returnPressed.connect(self.run_current)
        self.list.itemActivated.connect(lambda _item: self.run_current())
        self.refresh('')

    def refresh(self, text: str) -> None:
        self.list.clear()
        words = text.lower().split()
        for path, action in sorted(self.actions.items(), key=lambda kv: kv[0].split('>')[-1]):
            if all(w in path.lower() for w in words):
                label = path.split('>')[-1] + f'    ({path.split(">")[0]})'
                if not action.shortcut().isEmpty():
                    label += f'   {action.shortcut().toString()}'
                item = QListWidgetItem(label)
                item.setData(Qt.UserRole, path)
                self.list.addItem(item)
        if self.list.count():
            self.list.setCurrentRow(0)

    def run_current(self) -> None:
        item = self.list.currentItem()
        if item is None:
            return
        action = self.actions[item.data(Qt.UserRole)]
        self.accept()
        action.trigger()
