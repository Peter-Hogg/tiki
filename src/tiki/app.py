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
    """An expected problem the user can fix (no image open, wrong ROI type...)."""


class Tiki(QMainWindow):
    def __init__(self):
        super().__init__()
        style.apply(self)
        self.setWindowTitle('TIKI')
        self.setAcceptDrops(True)
        self.tool = 'rectangle'
        self.results = ResultsTable()
        self.profiles: list[ProfileWindow] = []
        self._message_until = 0.0
        self._actions: dict[str, QAction] = {}
        self.threshold_window = None

        self._build_menus()
        self.windows = WindowManager(
            reserved_shortcuts=[a.shortcut().toString() for a in self._actions.values() if not a.shortcut().isEmpty()]
        )
        self.windows.active_changed.connect(self._on_active_changed)
        self.windows.cursor_moved.connect(self._on_cursor)
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
        self._add(m, 'Crop', self.crop, 'Ctrl+Shift+X')
        self._add(m, 'Duplicate…', self.duplicate, 'Ctrl+Shift+D')
        stacks = m.addMenu('Stacks')
        self._add(stacks, 'Z Project…', self.z_project_dialog)
        self._add(stacks, 'Plot Z-axis Profile', self.plot_z_profile, 'Ctrl+Shift+Y')

        m = mb.addMenu('&Analyze')
        self._add(m, 'Measure', self.measure, 'Ctrl+M')
        self._add(m, 'Clear Results', self.results.clear)
        self._add(m, 'Show Results', self.show_results)
        m.addSeparator()
        self._add(m, 'Plot Profile', self.plot_line_profile, 'Ctrl+K')

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
            self.report(name, str(exc))
        except Exception as exc:
            traceback.print_exc()
            self.report(name, f'{type(exc).__name__}: {exc}\n\n(full traceback in the terminal)', error=True)

    def report(self, title: str, message: str, error: bool = False) -> None:
        """Show a problem the way ImageJ does: a small dialog plus the status bar."""
        self.status(f'{title}: {message.splitlines()[0]}')
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
        """Put ``viewer`` in the mode for the current tool (ImageJ tools are global)."""
        if roi_mod.target_image(viewer) is None:
            return
        _, _, mode, kind = next(t for t in TOOLS if t[0] == self.tool)
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
        scale, translate = image.scale, image.translate
        if np.prod(shape) * 2 > 1 * 2**30:  # >1 GB for a whole movie: paint one plane
            shape, scale, translate = shape[-2:], scale[-2:], translate[-2:]
            self.status('Movie too large for full-size labels; painting a single 2D plane')
        layer = Labels(np.zeros(shape, np.uint16), name='labels', scale=scale, translate=translate, opacity=0.6)
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

    def _on_active_changed(self, viewer) -> None:
        if viewer is not None:
            self.apply_tool(viewer)
        if self.threshold_window is not None and self.threshold_window.isVisible():
            self.threshold_window.set_target(viewer)

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
            kwargs = {'metadata': {'path': meta['path']}}
            if 'scale' in meta:
                kwargs['scale'] = meta['scale']
            self.open_data(data, meta['name'], **kwargs)
            self.status(f'Opened {meta["name"]} {data.shape} {data.dtype} in {time.perf_counter() - t0:.2f} s')
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
        index = roi_mod.current_index(viewer, image)
        lead = list(index[:-2])
        corners = [lead + [y, x] for y, x in ((-0.5, -0.5), (-0.5, w - 0.5), (h - 0.5, w - 0.5), (h - 0.5, -0.5))]
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
        self.open_data(np.array(data), f'{stem}-1.tif', scale=image.scale)

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
        axis_box = QComboBox()
        for d in range(image.ndim - 2):
            axis_box.addItem(f'{roi_mod.axis_name(viewer, image, d)} ({image.data.shape[d]})', d)
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
        if image.ndim > 3:
            form.addRow('Axis:', axis_box)
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
        scale = [s for d, s in enumerate(image.scale) if d != axis]
        viewer = self.open_data(result, f'{prefix}_{image.name}', scale=scale)
        self.status(f'Z Project ({method}) in {time.perf_counter() - t0:.2f} s')
        return viewer

    def plot_z_profile(self) -> None:
        viewer, image = self.require_image()
        if image.ndim < 3:
            raise TikiError('Plot Z-axis Profile needs a stack')
        axis = 0
        last = viewer.dims.last_used - (viewer.dims.ndim - image.ndim)
        if 0 <= last < image.ndim - 2:
            axis = last
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
        if self.threshold_window is not None:
            self.threshold_window.close()
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
