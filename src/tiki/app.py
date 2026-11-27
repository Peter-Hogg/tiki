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
    QDoubleSpinBox,
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
from . import io, nd, ops, recent, style
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
    ('polyline', 'Segmented line (double-click to finish)', 'add_path', 'shapes'),
    ('wand', 'Magic Wand (click to flood-select; Edit ▸ Selection ▸ Wand Tolerance…)', 'pan_zoom', 'wand'),
    ('point', 'Point', 'add', 'points'),
    ('select', 'Move / edit ROI', 'select', 'shapes'),
    ('hand', 'Scrolling tool (pan & zoom)', 'pan_zoom', None),
    ('brush', 'Paint labels (brush size in the layer controls)', 'paint', 'labels'),
    ('eraser', 'Erase labels', 'erase', 'labels'),
]
DRAW_MODES = {'add_rectangle', 'add_ellipse', 'add_polygon', 'add_polygon_lasso', 'add_line', 'add_path'}

STATUS_HOLD_MS = 4000


class TikiError(Exception):
    """An expected problem the user can fix (no image open, wrong ROI type...).

    ``fix`` names a command that resolves it (e.g. ``'properties'``); the
    error dialog then offers a button for it.
    """

    def __init__(self, message: str, fix: str | None = None):
        super().__init__(message)
        self.fix = fix


FIXES = {'properties': ('Open Properties…', 'properties'), 'split': ('Split Channels', 'split_channels')}


class Tiki(QMainWindow):
    FOLDER_SEQUENCE_MIN = 4

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
        self._action_slots: dict[str, tuple[str, object]] = {}
        self.contrast_window = None
        self.threshold_window = None
        self.roi_manager = None
        self.console_window = None
        self.recorder_window = None
        self.traces: list = []  # open TracesWindows (exported by Send to Notebook)
        self._last_roi: tuple[np.ndarray, str] | None = None
        self._undo_stack: list[dict] = []
        self.wand_tolerance: float = 10.0
        self.measurements: set[str] = set(ops.DEFAULT_MEASUREMENTS)

        self._build_menus()
        self.windows = WindowManager(
            reserved_shortcuts=[a.shortcut().toString() for a in self._actions.values() if not a.shortcut().isEmpty()]
        )
        self.windows.active_changed.connect(self._on_active_changed)
        self.windows.windows_changed.connect(self._prune_undo_stack)
        self.windows.cursor_moved.connect(self._on_cursor)
        self.windows.ndisplay_changed.connect(self._on_ndisplay)
        self.windows.files_dropped.connect(self.open_dropped)
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
        key = f'{menu.title()}>{text}'.replace('&', '')
        self._actions[key] = action
        self._action_slots[key] = (text, slot)
        return action

    def _build_menus(self) -> None:
        from .contrast import LUTS

        mb = self.menuBar()
        m = mb.addMenu('&File')
        self._add(m, 'Open…', self.open_dialog, 'Ctrl+O')
        self.recent_menu = m.addMenu('Open Recent')
        self.recent_menu.aboutToShow.connect(self._fill_recent_menu)
        imp = m.addMenu('Import')
        self._add(imp, 'Image Sequence…', lambda: self._process('sequence_dialog'), 'Ctrl+Shift+I')
        self._add(imp, 'OME-Zarr (.zarr)…', lambda: self._process('open_zarr_dialog'))
        samples = m.addMenu('Open Samples')
        self._add(samples, 'Synthetic movie', self.open_sample_movie)
        samples.addSeparator()
        self._add(samples, 'Cells 3D (2-ch Z-stack)', lambda: self._open_skimage('cells3d'))
        self._add(samples, 'Human Mitosis (2D nuclei)', lambda: self._open_skimage('human_mitosis'))
        self._add(samples, 'Mouse Kidney (3-ch Z-stack)', lambda: self._open_skimage('kidney'))
        self._add(samples, 'Lily Confocal (4-ch)', lambda: self._open_skimage('lily'))
        self._add(samples, 'Brain MRI (Z-stack)', lambda: self._open_skimage('brain'))
        self._add(samples, 'Cell (brightfield 2D)', lambda: self._open_skimage('cell'))
        self._add(samples, 'Immunohistochemistry (RGB)', lambda: self._open_skimage('immunohistochemistry'))
        self._add(samples, 'Retina (RGB)', lambda: self._open_skimage('retina'))
        self._add(samples, 'Microaneurysms (2D)', lambda: self._open_skimage('microaneurysms'))
        samples.addSeparator()
        self._add(samples, 'Blobs', lambda: self._open_skimage('binary_blobs'))
        self._add(samples, 'Camera', lambda: self._open_skimage('camera'))
        m.addSeparator()
        self._add(m, 'Close', self.close_active, 'Ctrl+W')
        self._add(m, 'Close All', self.windows_close_all, 'Ctrl+Shift+W')
        self._add(m, 'Save As TIFF…', self.save_as, 'Ctrl+S')
        export = m.addMenu('Export')
        self._add(export, 'Snapshot (PNG / SVG)…', lambda: self._publish('snapshot_dialog'))
        self._add(export, 'Export Movie (MP4 / GIF / AVI)…', lambda: self._publish('export_movie_dialog'))
        export.addSeparator()
        self._add(export, 'Send to Notebook…', self.send_to_notebook, 'Ctrl+Shift+N')
        m.addSeparator()
        self._add(m, 'Quit', self.close, 'Ctrl+Q')

        m = mb.addMenu('&Edit')
        self._add(m, 'Undo', self.undo, 'Ctrl+Z')
        m.addSeparator()
        sel = m.addMenu('Selection')
        self._add(sel, 'Select All', self.select_all, 'Ctrl+A')
        self._add(sel, 'Select None', self.select_none, 'Ctrl+Shift+A')
        self._add(sel, 'Restore Selection', self.restore_selection, 'Ctrl+Shift+R')
        sel.addSeparator()
        self._add(sel, 'Enlarge / Shrink…', lambda: self._process('enlarge_dialog'))
        self._add(sel, 'Make Band…', lambda: self._process('make_band_dialog'))
        self._add(sel, 'Wand Tolerance…', self.set_wand_tolerance)
        m.addSeparator()
        self._add(m, 'Clear', lambda: self._process('clear_selection', outside=False))
        self._add(m, 'Clear Outside', lambda: self._process('clear_selection', outside=True))
        m.addSeparator()
        self._add(m, 'Invert', self.invert)

        m = mb.addMenu('&Image')
        img_type = m.addMenu('Type')
        self._add(img_type, '8-bit', lambda: self.convert_type('uint8'))
        self._add(img_type, '16-bit', lambda: self.convert_type('uint16'))
        self._add(img_type, '32-bit', lambda: self.convert_type('float32'))
        self._add(img_type, 'RGB Color', self.flatten_rgb)
        adjust = m.addMenu('Adjust')
        self._add(adjust, 'Brightness/Contrast…', self.brightness_contrast, 'Ctrl+Shift+C')
        self._add(adjust, 'Threshold…', self.threshold, 'Ctrl+Shift+T')
        self._add(adjust, 'Enhance Contrast', self.enhance_contrast, 'Ctrl+Shift+E')
        self._add(adjust, 'Reset Display Range', self.reset_contrast)
        self._add(adjust, 'Sync Display Across Windows', self.sync_display)
        adjust.addSeparator()
        self._add(adjust, 'Size…', lambda: self._process('resize_dialog'))
        luts = m.addMenu('Lookup Tables')
        for lut_name in LUTS:
            self._add(luts, lut_name, lambda n=lut_name: self.set_lut(n))
        m.addSeparator()
        self._add(m, 'Show Info…', self.show_info, 'Ctrl+I')
        self._add(m, 'Properties…', self.properties, 'Ctrl+Shift+P')
        color = m.addMenu('Color')
        self._add(color, 'Split Channels', self.split_channels)
        self._add(color, 'Merge Channels', self.merge_channels)
        self._add(color, 'Channels Tool…', self.channels_tool, 'Ctrl+Shift+Z')
        transform = m.addMenu('Transform')
        self._add(transform, 'Flip Horizontally', lambda: self.flip('horizontal'))
        self._add(transform, 'Flip Vertically', lambda: self.flip('vertical'))
        self._add(transform, 'Rotate 90° Right', lambda: self.rotate_90(-1))
        self._add(transform, 'Rotate 90° Left', lambda: self.rotate_90(1))
        self._add(transform, 'Rotate…', lambda: self._process('rotate_dialog'))
        self._add(m, 'Crop', self.crop, 'Ctrl+Shift+X')
        self._add(m, 'Duplicate…', lambda: self._process('duplicate_dialog'), 'Ctrl+Shift+D')
        stacks = m.addMenu('Stacks')
        self._add(stacks, 'Z Project…', self.z_project_dialog)
        self._add(stacks, 'Make Substack…', lambda: self._process('duplicate_dialog', substack_only=True))
        self._add(stacks, 'Stack to Hyperstack…', lambda: self._process('stack_to_hyperstack_dialog'))
        self._add(stacks, 'Hyperstack to Stack', lambda: self._process('hyperstack_to_stack_command'))
        self._add(stacks, 'Make Montage…', lambda: self._publish('make_montage_dialog'))
        self._add(stacks, 'Orthogonal Views', lambda: self._publish('orthogonal_views'), 'Ctrl+Shift+H')
        self._add(stacks, 'Bin…', lambda: self._process('bin_dialog'))
        self._add(stacks, 'Plot Z-axis Profile', self.plot_z_profile, 'Ctrl+Shift+Y')
        self._add(stacks, 'Kymograph / Reslice…', lambda: self._process('kymograph_dialog'))
        self._add(stacks, 'Correlation Image', lambda: self._process('correlation_image'))
        self._add(stacks, 'Concatenate…', lambda: self._process('concatenate_dialog'))
        self._add(stacks, 'Images to Stack…', lambda: self._process('images_to_stack_dialog'))
        self._add(stacks, 'Reverse', lambda: self._process('reverse_stack_command'))
        self._add(stacks, 'Remove Slices…', lambda: self._process('remove_slices_dialog'))
        stacks.addSeparator()
        self._add(stacks, 'Split Channels ', self.split_channels)  # same command as Image ▸ Color

        m = mb.addMenu('&Process')
        self._add(m, 'Smooth', lambda: self._process('smooth_command'), 'Ctrl+Shift+S')
        self._add(m, 'Sharpen', lambda: self._process('sharpen_command'))
        self._add(m, 'Find Edges', lambda: self._process('find_edges_command'))
        self._add(m, 'Edge Detection…', lambda: self._process('edge_detection_dialog'))
        self._add(m, 'Find Maxima…', lambda: self._process('find_maxima_dialog'))
        m.addSeparator()
        filt = m.addMenu('Filters')
        self._add(filt, 'Convolve…', lambda: self._process('convolve_dialog'))
        filt.addSeparator()
        self._add(filt, 'Gaussian Blur…', lambda: self.filter_dialog('gaussian'))
        self._add(filt, 'Mean…', lambda: self.filter_dialog('mean'))
        self._add(filt, 'Median…', lambda: self.filter_dialog('median'))
        self._add(filt, 'Minimum…', lambda: self.filter_dialog('minimum'))
        self._add(filt, 'Maximum…', lambda: self.filter_dialog('maximum'))
        self._add(filt, 'Variance…', lambda: self.filter_dialog('variance'))
        self._add(filt, 'Top Hat…', lambda: self.filter_dialog('tophat'))
        self._add(filt, 'Unsharp Mask…', lambda: self.filter_dialog('unsharp'))
        noise_menu = m.addMenu('Noise')
        self._add(noise_menu, 'Despeckle', lambda: self._fft_noise('despeckle_command'))
        self._add(noise_menu, 'Remove Outliers…', lambda: self._fft_noise('remove_outliers_dialog'))
        self._add(noise_menu, 'Remove NaNs…', lambda: self._fft_noise('remove_nans_dialog'))
        noise_menu.addSeparator()
        self._add(noise_menu, 'Add Noise', lambda: self._fft_noise('add_noise_command'))
        self._add(noise_menu, 'Add Specified Noise…', lambda: self._fft_noise('add_specified_noise_dialog'))
        self._add(noise_menu, 'Salt and Pepper', lambda: self._fft_noise('salt_and_pepper_command'))
        fft_menu = m.addMenu('FFT')
        self._add(fft_menu, 'FFT', lambda: self._fft_noise('fft_command'))
        self._add(fft_menu, 'Inverse FFT', lambda: self._fft_noise('inverse_fft_command'))
        self._add(fft_menu, 'Bandpass Filter…', lambda: self._fft_noise('bandpass_dialog'))
        self._add(m, 'Subtract Background…', lambda: self.filter_dialog('background'))
        self._add(m, 'Math…', lambda: self._process('math_dialog'))
        m.addSeparator()
        self._add(m, 'ΔF/F…', lambda: self._process('dff_dialog'))
        self._add(m, 'Bleach Correction…', lambda: self._process('bleach_dialog'))
        self._add(m, 'Motion Correction…', lambda: self._process('motion_correct_dialog'))
        self._add(m, 'Image Calculator…', lambda: self._process('calculator_dialog'))
        labels_menu = m.addMenu('Labels')
        self._add(labels_menu, 'Watershed…', lambda: self._process('watershed_dialog'))
        self._add(labels_menu, 'Fill Holes', lambda: self._process('fill_holes_command'))
        self._add(labels_menu, 'Morphology…', lambda: self._process('morphology_dialog'))
        labels_menu.addSeparator()
        self._add(labels_menu, 'Distance Map', lambda: self._process('distance_map_command'))
        self._add(labels_menu, 'Skeletonize', lambda: self._process('skeletonize_command'))
        self._add(labels_menu, 'Outline', lambda: self._process('outline_command'))
        self._add(labels_menu, 'Voronoi…', lambda: self._process('voronoi_dialog'))
        batch_menu = m.addMenu('Batch')
        self._add(batch_menu, 'Convert…', lambda: self._batch('batch_convert_dialog'))
        self._add(batch_menu, 'Run Macro…', lambda: self._batch('batch_macro_dialog'))

        m = mb.addMenu('&Analyze')
        self._add(m, 'Measure', self.measure, 'Ctrl+M')
        self._add(m, 'Analyze Particles…', self.analyze_particles)
        self._add(m, 'Histogram', self.histogram, 'Ctrl+H')
        self._add(m, 'Colocalization…', lambda: self._coloc('colocalization_dialog'))
        self._add(m, 'Set Scale…', lambda: self._process('set_scale_dialog'))
        self._add(m, 'Set Measurements…', lambda: self._process('set_measurements_dialog'))
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
        tools.addSeparator()
        self._add(tools, 'Scale Bar…', lambda: self._publish('snapshot_dialog', focus_scale_bar=True))

        m = mb.addMenu('&Plugins')
        self._add(m, 'Python Console…', self.show_console, 'Ctrl+Shift+J')
        m.addSeparator()
        macros = m.addMenu('Macros')
        self.macros_menu = macros
        self._add(macros, 'Record…', self.show_macro_recorder)
        macros.addSeparator()
        self._add(macros, 'New Macro…', self.new_macro, 'Ctrl+Shift+M')
        self._add(macros, 'Edit Macro…', self.edit_macro)
        self._add(macros, 'Run Macro…', self.run_macro_file)
        self._add(macros, 'Show Macros Folder', self.show_macros_folder)
        self._macro_fixed = len(macros.actions())
        macros.aboutToShow.connect(self._fill_macros_menu)

        self.window_menu = mb.addMenu('&Window')
        self.window_menu.aboutToShow.connect(self._fill_window_menu)

        m = mb.addMenu('&Help')
        self._add(m, 'Find Commands…', self.find_commands, 'Ctrl+L')
        self._add(m, 'Check for Updates…', self.check_for_updates)
        self._add(m, 'About TIKI', self.about)

    def _fill_window_menu(self) -> None:
        menu = self.window_menu
        menu.clear()
        toggle = menu.addAction('Show Layer List')
        toggle.triggered.connect(self.toggle_layer_list)
        controls = menu.addAction('Show Layer Controls')
        controls.triggered.connect(self.toggle_layer_controls)
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
            tb = traceback.format_exc()
            traceback.print_exc()
            details = format_error_report(name, exc, tb)
            self.report(
                name,
                f'{type(exc).__name__}: {exc}\n\nClick "Show Details" or "Copy Error Details" to share this error.',
                error=True,
                details=details,
            )

    def run_command(self, command: str, raise_errors: bool = False):
        """Run a registered menu command by menu path or leaf label."""
        raw = command.strip().lower()
        wanted = command.rstrip('.…').strip().lower()
        matched = None
        for path, (text, slot) in self._action_slots.items():
            if path.lower() == raw:
                matched = (text, slot)
                break
        if matched is None:
            for path, (text, slot) in self._action_slots.items():
                if path.split('>')[-1].rstrip('.…').strip().lower() == wanted:
                    matched = (text, slot)
                    break
        if matched is None:
            raise ValueError(f'no TIKI command called {command!r}')
        text, slot = matched
        if raise_errors:
            return slot()
        return self._run(text, slot)

    def report(
        self,
        title: str,
        message: str,
        error: bool = False,
        fix: str | None = None,
        details: str | None = None,
    ) -> None:
        """Show a problem the way ImageJ does: a small dialog plus the status bar.

        With ``fix`` the dialog gets a button that runs the fixing command.
        With ``details`` the dialog shows a collapsible traceback and a
        "Copy Error Details" button (useful when launched without a terminal).
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
        if details:
            box = QMessageBox(QMessageBox.Critical if error else QMessageBox.Warning, title, message, QMessageBox.Close, self)
            box.setDetailedText(details)
            copy_btn = box.addButton('Copy Error Details', QMessageBox.ActionRole)
            box.exec_()
            if box.clickedButton() is copy_btn:
                QApplication.clipboard().setText(details)
                self.status('Copied error details to clipboard')
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
        if ndim < 2:
            return
        coords = image.world_to_data(position[-ndim:])
        if not np.all(np.isfinite(coords)):
            return
        shape = image.data.shape[:ndim]
        idx = tuple(int(np.floor(c + 0.5)) for c in coords)
        info = axes_mod.get(image)
        x_ax = info.axes[-1]
        y_ax = info.axes[-2]
        x_str = f'x={idx[-1]} ({idx[-1] * x_ax.step:.4g} {x_ax.unit})' if (x_ax.step and x_ax.unit) else f'x={idx[-1]}'
        y_str = f'y={idx[-2]} ({idx[-2] * y_ax.step:.4g} {y_ax.unit})' if (y_ax.step and y_ax.unit) else f'y={idx[-2]}'
        parts = [x_str, y_str]
        for d in range(ndim - 2):
            ax = info.axes[d]
            name = roi_mod.axis_name(viewer, image, d)
            if ax.role in ('T', 'Z') and ax.step and ax.unit:
                parts.append(f'{name}={idx[d]} ({idx[d] * ax.step:.4g} {ax.unit})')
            else:
                parts.append(f'{name}={idx[d]}')
        if all(0 <= i < s for i, s in zip(idx, shape)):
            value = np.asarray(image.data[idx])
            if value.ndim == 0:
                v = value.item()
                parts.append(f'value={v:.4g}' if isinstance(v, float) else f'value={v}')
            else:
                parts.append('value=' + ','.join(str(x) for x in value.tolist()))
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
        elif kind == 'wand':
            for name in (roi_mod.ROI_NAME, roi_mod.POINTS_NAME):
                if name in viewer.layers:
                    viewer.layers[name].mode = 'pan_zoom'
            if self._on_wand_click not in viewer.mouse_drag_callbacks:
                viewer.mouse_drag_callbacks.append(self._on_wand_click)
            viewer.layers.selection.active = roi_mod.target_image(viewer)
            return
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

    def _on_wand_click(self, viewer, event):
        if self.tool != 'wand' or viewer.dims.ndisplay == 3:
            return
        image = roi_mod.target_image(viewer)
        if image is None:
            return
        coords = image.world_to_data(event.position[-image.ndim :])
        y, x = int(np.floor(coords[-2] + 0.5)), int(np.floor(coords[-1] + 0.5))
        self._run('Magic Wand', lambda: self.wand_select(viewer, y, x, add='Shift' in event.modifiers))

    def wand_select(self, viewer, y: int, x: int, add: bool = False):
        image = roi_mod.target_image(viewer)
        if image is None:
            raise TikiError('no image open')
        plane = roi_mod.current_plane(viewer, image)
        if not (0 <= y < plane.shape[0] and 0 <= x < plane.shape[1]):
            return None
        mask = ops.wand_mask(plane, y, x, tolerance=self.wand_tolerance)
        poly = ops.mask_to_polygon(mask)
        if poly is None:
            self.status(f'Magic Wand: region at ({x}, {y}) is too small to outline')
            return None
        layer = self._roi(viewer)
        if not add:
            prev = roi_mod.current_shape(layer)
            if prev is not None:
                self._last_roi = (prev[0].copy(), prev[1])
            layer.data = []
        layer.add_polygons([poly])
        self.status(f'Magic Wand (tol={self.wand_tolerance:g}): {int(mask.sum())} px at ({x}, {y})')
        return poly

    def set_wand_tolerance(self) -> None:
        from qtpy.QtWidgets import QInputDialog

        val, ok = QInputDialog.getDouble(
            self,
            'Wand Tolerance',
            'Tolerance (intensity difference from clicked pixel):',
            self.wand_tolerance,
            0.0,
            1e9,
            2,
        )
        if ok:
            self.wand_tolerance = float(val)
            self.status(f'Magic Wand tolerance = {self.wand_tolerance:g}')

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

    def _prune(self, layer: Shapes) -> None:
        if layer._is_creating or layer.nshapes < 2:
            return
        if layer.nshapes >= 2:
            self._last_roi = (np.asarray(layer.data[-2]).copy(), layer.shape_type[-2])
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
        if self.contrast_window is not None and self.contrast_window.isVisible():
            self.contrast_window.set_target(viewer)
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

    def open_scene_entry(self, data, meta: dict, scene_idx: int | None = None, raise_errors: bool = False):
        """Open one loaded ``(data, meta)`` entry in a new window and return the viewer."""
        metadata = {'path': meta['path']}
        if scene_idx is not None:
            metadata['scene'] = scene_idx
        for key in ('imagej_luts', 'channel_names', 'acquisition', 'raw_metadata', 'container_key'):
            if meta.get(key) is not None:
                metadata[key] = meta[key]
        info = meta['axes']
        open_kw: dict = {'axes': info, 'metadata': metadata}
        rgb = data.ndim == len(info.axes) + 1 and data.shape[-1] in (3, 4)
        if meta.get('first_plane') is not None and not rgb:
            open_kw['contrast_limits'] = ops.saturated_limits(meta['first_plane'])
        luts = meta.get('imagej_luts')
        if info.index('C') is None and not rgb and luts and len(luts) == 1:
            lut = np.asarray(luts[0], dtype=float)
            if lut.shape == (3, 256):
                ramp = np.arange(256, dtype=float)
                if not all(np.allclose(lut[ch], ramp, atol=1.5) for ch in range(3)):
                    from napari.utils.colormaps import Colormap

                    rgba = np.column_stack([np.clip(lut.T / 255.0, 0.0, 1.0), np.ones(256)])
                    open_kw['colormap'] = Colormap(rgba, name='imagej_lut')
        viewer = self.open_data(data, meta['name'], **open_kw)
        msg = f'Opened {meta["name"]}: {info.describe()} {data.dtype} [{info.source}]'
        if info.notes:
            msg += ' — ' + '; '.join(info.notes)
        if not info.confident:
            msg += ' — check Image ▸ Properties (Ctrl+Shift+P)'
        self.status(msg)
        if not info.confident or info.notes:
            self._message_until = time.monotonic() + 10
        if 'warning' in meta:
            if raise_errors:
                self.status(f'{meta["name"]}: {meta["warning"]}')
            else:
                self.report('Open', f'{meta["name"]}: {meta["warning"]}')
        return viewer

    def open_paths(self, paths, raise_errors: bool = False) -> list:
        opened: list = []
        for path in paths:
            try:
                loaded = io.read_multi(path)
            except Exception as exc:
                if raise_errors:
                    raise
                traceback.print_exc()
                self.report('Open', f'Could not open {Path(path).name}:\n{exc}')
                continue
            for scene_idx, (data, meta) in enumerate(loaded):
                v = self.open_scene_entry(
                    data,
                    meta,
                    scene_idx=scene_idx if len(loaded) > 1 else None,
                    raise_errors=raise_errors,
                )
                opened.append(v)
            self.record(f'tiki.open({str(path)!r})')
            recent.add(path)
        return opened

    def open_sequence(self, folder_or_paths, **kwargs):
        t0 = time.perf_counter()
        data, meta = io.read_sequence(folder_or_paths, **kwargs)
        info = meta['axes']
        open_kw: dict = {'axes': info, 'metadata': {'path': meta['path']}}
        if not meta.get('rgb'):
            open_kw['contrast_limits'] = ops.saturated_limits(meta['first_plane'])
        viewer = self.open_data(data, meta['name'], **open_kw)
        msg = (
            f'Opened sequence {meta["name"]}: {info.describe()} {data.dtype} '
            f'({meta["n_files"]} files, lazy) in {time.perf_counter() - t0:.2f} s'
        )
        if not info.confident:
            msg += ' — check Image ▸ Properties (Ctrl+Shift+P)'
        self.status(msg)
        if not info.confident:
            self._message_until = time.monotonic() + 10
        return viewer

    def open_dialog(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, 'Open', str(Path.home()), io.OPEN_FILTER)
        if paths:
            self.open_paths(paths)

    def open_sample_movie(self) -> None:
        self.open_data(io.synthetic_movie(), 'synthetic_movie.tif')

    def _open_skimage(self, name: str) -> None:
        from skimage import data

        loader_name = 'binary_blobs' if name == 'blobs' else name
        loader = getattr(data, loader_name, None)
        if loader is None:
            raise TikiError(f'Sample dataset {name!r} is not available in this scikit-image version.')
        self.status(f'Loading sample: {name}…')
        QApplication.processEvents()
        try:
            arr = loader()
        except Exception as exc:
            raise TikiError(
                f'Could not load sample {name!r}.\n\n'
                f'Larger scikit-image biology datasets download once from the internet (and are cached locally).\n\n'
                f'Details: {exc}'
            ) from exc

        kw: dict = {}
        # Provide physical axes / channel order for 3D & multichannel biology samples
        if loader_name == 'cells3d' and arr.ndim == 4:
            # (60, 2, 256, 256) ZCYX — 0.29 µm Z, 0.26 µm XY (Allen Institute)
            kw['axes'] = axes_mod.AxesInfo(
                [
                    axes_mod.Axis('Z', arr.shape[0], 0.29, 'µm'),
                    axes_mod.Axis('C', arr.shape[1]),
                    axes_mod.Axis('Y', arr.shape[2], 0.26, 'µm'),
                    axes_mod.Axis('X', arr.shape[3], 0.26, 'µm'),
                ],
                source='skimage',
                confident=True,
            )
        elif loader_name == 'kidney' and arr.ndim == 4:
            # (16, 512, 512, 3) ZYXC -> (16, 3, 512, 512) ZCYX so Split Channels works
            arr = np.moveaxis(arr, -1, 1)
            kw['axes'] = axes_mod.AxesInfo(
                [
                    axes_mod.Axis('Z', arr.shape[0]),
                    axes_mod.Axis('C', arr.shape[1]),
                    axes_mod.Axis('Y', arr.shape[2]),
                    axes_mod.Axis('X', arr.shape[3]),
                ],
                source='skimage',
                confident=True,
            )
        elif loader_name == 'lily' and arr.ndim == 3 and arr.shape[-1] == 4:
            # (922, 922, 4) YXC -> (4, 922, 922) CYX (4-channel confocal)
            arr = np.moveaxis(arr, -1, 0)
            kw['axes'] = axes_mod.AxesInfo(
                [
                    axes_mod.Axis('C', arr.shape[0]),
                    axes_mod.Axis('Y', arr.shape[1]),
                    axes_mod.Axis('X', arr.shape[2]),
                ],
                source='skimage',
                confident=True,
            )
        elif loader_name == 'brain' and arr.ndim == 3:
            kw['axes'] = axes_mod.AxesInfo(
                [
                    axes_mod.Axis('Z', arr.shape[0]),
                    axes_mod.Axis('Y', arr.shape[1]),
                    axes_mod.Axis('X', arr.shape[2]),
                ],
                source='skimage',
                confident=True,
            )
        elif loader_name == 'binary_blobs':
            arr = (arr.astype(np.uint8)) * 255

        self.open_data(arr, f'{loader_name}.tif', **kw)
        if loader_name in ('cells3d', 'kidney', 'lily'):
            self.split_channels()
        self.status(f'Opened sample {loader_name} ({arr.shape}, {arr.dtype})')

    def close_active(self) -> None:
        viewer = self.windows.active
        if viewer is not None:
            self.windows.close(viewer)

    def windows_close_all(self) -> None:
        self.windows.close_all()

    def save_layer(self, path: str | Path, layer=None, viewer=None, merge_split: bool = False) -> Path:
        from . import contrast

        viewer = viewer or self.windows.active
        if viewer is None:
            raise TikiError('no image open')
        if merge_split:
            parts = sorted(
                [ly for ly in viewer.layers if isinstance(ly, Image) and 'split_from' in ly.metadata],
                key=lambda ly: ly.metadata.get('channel', 0),
            )
            if not parts:
                raise TikiError('this window has no split channels to merge')
            original = parts[0].metadata['split_from']
            info = axes_mod.get(original)
            c = info.index('C')
            if c is None or len(parts) != original.data.shape[c]:
                data = np.stack([np.asarray(ly.data) for ly in parts], axis=0)
                p_info = axes_mod.get(parts[0])
                info = axes_mod.AxesInfo(
                    [axes_mod.Axis('C', len(parts))] + list(p_info.axes),
                    p_info.source,
                    p_info.confident,
                )
            else:
                data = np.stack([np.asarray(ly.data) for ly in parts], axis=c)
            luts = [contrast.layer_lut_bytes(ly) for ly in parts]
            ch_names = [ly.metadata.get('channel_name') or '' for ly in parts]
            io.save_tiff(
                path,
                data,
                axes=info,
                luts=luts,
                channel_names=ch_names if any(ch_names) else None,
            )
            self.record(f'tiki.save_tiff({str(path)!r}, merge_split=True)')
            self.status(f'Saved hyperstack {path}')
            return Path(path)

        if layer is None:
            active = viewer.layers.selection.active
            if isinstance(active, (Image, Labels)) and active.name not in roi_mod.PREVIEW_NAMES:
                layer = active
            else:
                layer = roi_mod.target_image(viewer)
        if layer is None:
            raise TikiError('no image or labels layer to save')

        if isinstance(layer, Labels):
            info = None
            if layer.metadata.get('axes'):
                info = axes_mod.get(layer)
            else:
                ref = roi_mod.target_image(viewer)
                shape = tuple(layer.data.shape[: layer.ndim])
                if ref is not None:
                    ref_info = axes_mod.get(ref)
                    if len(ref_info.axes) >= layer.ndim:
                        cand = axes_mod.AxesInfo(
                            [axes_mod.Axis(a.role, n, a.step, a.unit)
                             for a, n in zip(ref_info.axes[-layer.ndim :], shape)],
                            ref_info.source,
                            ref_info.confident,
                        )
                        if not cand.problems(shape):
                            info = cand
                if info is None:
                    info = axes_mod.guess(shape)
            io.save_tiff(path, layer.data, axes=info)
            self.status(f'Saved labels "{layer.name}" to {path}')
            return Path(path)

        info = axes_mod.get(layer)
        luts = None
        if info.index('C') is None and not layer.rgb and layer.colormap.name != 'gray':
            luts = [contrast.layer_lut_bytes(layer)]
        ch_names = layer.metadata.get('channel_names')
        if not ch_names and layer.metadata.get('channel_name'):
            ch_names = [layer.metadata['channel_name']]
        io.save_tiff(path, layer.data, axes=info, luts=luts, channel_names=ch_names)
        self.record(f'tiki.save_tiff({str(path)!r})')
        self.status(f'Saved {path}')
        return Path(path)

    def save_as(self) -> None:
        viewer = self.windows.active
        if viewer is None:
            raise TikiError('no image open')
        savable = [
            ly for ly in viewer.layers
            if isinstance(ly, (Image, Labels)) and ly.name not in roi_mod.PREVIEW_NAMES
        ]
        if not savable:
            raise TikiError('no image or labels layer to save')
        parts = [ly for ly in savable if isinstance(ly, Image) and 'split_from' in ly.metadata]
        merge_split = False
        chosen = savable[-1]

        if parts or len(savable) > 1:
            dlg = QDialog(self)
            style.apply(dlg)
            dlg.setWindowTitle('Save As TIFF')
            form = QFormLayout(dlg)
            what = QComboBox()
            if parts:
                orig = parts[0].metadata['split_from']
                what.addItem(f'Merged hyperstack ({orig.name}, {len(parts)} channels)', ('merge', orig))
            active = viewer.layers.selection.active
            for ly in savable:
                kind = 'Labels' if isinstance(ly, Labels) else 'Image'
                what.addItem(f'{ly.name} ({kind})', ('layer', ly))
            if not parts and active in savable:
                idx = next(i for i in range(what.count()) if what.itemData(i) == ('layer', active))
                what.setCurrentIndex(idx)
            form.addRow('Save:', what)
            buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            buttons.accepted.connect(dlg.accept)
            buttons.rejected.connect(dlg.reject)
            form.addRow(buttons)
            if dlg.exec_() != QDialog.Accepted:
                return
            mode, target = what.currentData()
            if mode == 'merge':
                merge_split = True
                default_name = Path(target.metadata.get('path', target.name)).with_suffix('.tif').name
            else:
                chosen = target
                default_name = Path(chosen.metadata.get('path', chosen.name)).with_suffix('.tif').name
        else:
            chosen = savable[0]
            default_name = Path(chosen.metadata.get('path', chosen.name)).with_suffix('.tif').name

        path, _ = QFileDialog.getSaveFileName(
            self, 'Save As TIFF', default_name, 'TIFF (*.tif *.tiff);;OME-TIFF (*.ome.tif)'
        )
        if path:
            self.save_layer(path, layer=chosen, viewer=viewer, merge_split=merge_split)

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
            cur = roi_mod.current_shape(layer)
            if cur is not None:
                self._last_roi = (cur[0].copy(), cur[1])
            layer.data = []

    UNDO_MAX = 10
    UNDO_MAX_BYTES = 256 * 2**20

    @staticmethod
    def _estimate_nbytes(data) -> int | None:
        nb = getattr(data, 'nbytes', None)
        if isinstance(nb, (int, np.integer)):
            return int(nb)
        shape = getattr(data, 'shape', None)
        dtype = getattr(data, 'dtype', None)
        if shape is not None and dtype is not None:
            try:
                return int(np.prod(shape)) * int(np.dtype(dtype).itemsize)
            except Exception:
                return None
        return None

    def push_undo_data(self, label: str, viewer, layer, in_place: bool = False) -> bool:
        """Snapshot ``layer.data`` and axes before an edit without materializing oversized lazy arrays."""
        raw = layer.data
        est = self._estimate_nbytes(raw)
        if est is not None and est > self.UNDO_MAX_BYTES:
            return False
        if hasattr(raw, 'compute') and not in_place:
            snapshot = raw
            snap_bytes = 0
        else:
            arr = np.asarray(raw)
            if arr.nbytes > self.UNDO_MAX_BYTES:
                return False
            snapshot = arr.copy()
            snap_bytes = int(arr.nbytes)
        entry = {
            'kind': 'data',
            'label': label,
            'viewer': viewer,
            'layer': layer,
            'data': snapshot,
            'bytes': snap_bytes,
            'axes': axes_mod.get(layer).to_dict(),
            'had_modified': 'modified' in layer.metadata,
            'modified': layer.metadata.get('modified'),
        }
        self._undo_stack.append(entry)
        self._trim_undo_stack()
        return True

    def push_undo_layer(self, label: str, viewer, layer) -> None:
        """Register a newly created layer so Edit ▸ Undo removes it."""
        self._undo_stack.append(
            {'kind': 'layer', 'label': label, 'viewer': viewer, 'layer': layer, 'bytes': 0}
        )
        self._trim_undo_stack()

    def _trim_undo_stack(self) -> None:
        while len(self._undo_stack) > self.UNDO_MAX:
            self._undo_stack.pop(0)
        total_bytes = sum(e.get('bytes', 0) for e in self._undo_stack)
        while total_bytes > self.UNDO_MAX_BYTES and self._undo_stack:
            total_bytes -= self._undo_stack.pop(0).get('bytes', 0)

    def _prune_undo_stack(self) -> None:
        open_viewers = set(self.windows.viewers)
        if not self._undo_stack:
            return
        self._undo_stack = [e for e in self._undo_stack if e['viewer'] in open_viewers]

    def undo(self) -> None:
        """Undo the last Labels paint stroke, destructive crop, or added layer."""
        viewer = self.windows.active
        if viewer is not None:
            active = viewer.layers.selection.active
            if isinstance(active, Labels) and len(getattr(active, '_undo_history', ())):
                active.undo()
                self.status(f'Undo paint on "{active.name}"')
                return

        while self._undo_stack:
            entry = self._undo_stack.pop()
            v, ly = entry['viewer'], entry['layer']
            if v not in self.windows.viewers or ly not in v.layers:
                continue
            if entry['kind'] == 'data':
                ly.data = entry['data']
                if entry.get('axes'):
                    axes_mod.apply(v, ly, axes_mod.AxesInfo.from_dict(entry['axes']))
                if entry.get('had_modified'):
                    ly.metadata['modified'] = entry.get('modified')
                else:
                    ly.metadata.pop('modified', None)
                v.reset_view()
            else:
                v.layers.remove(ly)
                self.reapply_tool(v)
            self.status(f'Undo {entry["label"]}')
            return

        raise TikiError('nothing to undo')

    def restore_selection(self) -> None:
        viewer, _ = self.require_image()
        if self._last_roi is None:
            raise TikiError('no previous selection to restore')
        verts, kind = self._last_roi
        layer = self._roi(viewer)
        cur = roi_mod.current_shape(layer)
        if cur is not None:
            self._last_roi = (cur[0].copy(), cur[1])
        layer.data = []
        layer.add(verts, shape_type=kind)
        self.status(f'Restored {kind} selection')

    def invert(self):
        viewer, image = self.require_image()
        total = int(np.prod(image.data.shape[: image.ndim - 2])) if image.ndim > 2 else 1
        with self.progress('Invert', total) as tick:
            out = ops.invert_image(image.data, rgb=bool(image.rgb), progress=tick)
        if tick.cancelled:
            self.status('Invert cancelled')
            return None
        info = axes_mod.fit(axes_mod.get(image), out.shape[:-1] if image.rgb else out.shape)
        kw: dict = {'axes': info, 'rgb': bool(image.rgb)}
        if not image.rgb:
            kw['colormap'] = image.colormap
        v = self.open_data(out, f'Inv_{image.name}', **kw)
        self.record("tiki.run('Invert')")
        self.status(f'Inverted {image.name}')
        return v

    # ------------------------------------------------------------- Image

    def flip(self, axis: str = 'horizontal'):
        viewer, image = self.require_image()
        out = ops.flip_image(image.data, axis=axis, rgb=bool(image.rgb))
        info = axes_mod.fit(axes_mod.get(image), out.shape[:-1] if image.rgb else out.shape)
        kw: dict = {'axes': info}
        if not image.rgb:
            kw['colormap'] = image.colormap
        tag = 'FlipH' if axis == 'horizontal' else 'FlipV'
        v = self.open_data(out, f'{tag}_{image.name}', **kw)
        self.record(f"tiki.run({'Flip Horizontally' if axis == 'horizontal' else 'Flip Vertically'!r})")
        self.status(f'Flipped {image.name} ({axis})')
        return v

    def rotate_90(self, k: int = -1):
        viewer, image = self.require_image()
        out = ops.rotate_90(image.data, k=k, rgb=bool(image.rgb))
        info = axes_mod.get(image)
        new_axes = [axes_mod.Axis(a.role, a.size, a.step, a.unit) for a in info.axes]
        # Swapping spatial height/width swaps Y and X sizes and steps
        y_ax, x_ax = new_axes[-2], new_axes[-1]
        new_axes[-2] = axes_mod.Axis('Y', out.shape[-3 if image.rgb else -2], x_ax.step, x_ax.unit)
        new_axes[-1] = axes_mod.Axis('X', out.shape[-2 if image.rgb else -1], y_ax.step, y_ax.unit)
        new_info = axes_mod.AxesInfo(new_axes, info.source, info.confident, list(info.notes))
        kw: dict = {'axes': new_info}
        if not image.rgb:
            kw['colormap'] = image.colormap
        tag = 'Rot90R' if k < 0 else 'Rot90L'
        v = self.open_data(out, f'{tag}_{image.name}', **kw)
        self.record(f"tiki.run({'Rotate 90° Right' if k < 0 else 'Rotate 90° Left'!r})")
        self.status(f'Rotated {image.name} 90° {"right" if k < 0 else "left"}')
        return v

    def convert_type(self, target_dtype: str):
        viewer, image = self.require_image()
        if image.rgb and target_dtype != 'uint8':
            raise TikiError('RGB images can only be converted via Image ▸ Type ▸ 8-bit or split first')
        scale_range = None
        if not image.rgb and target_dtype in ('uint8', 'uint16'):
            scale_range = tuple(map(float, image.contrast_limits))
        total = int(np.prod(image.data.shape[: image.ndim - 2])) if image.ndim > 2 else 1
        with self.progress(f'Convert to {target_dtype}', total) as tick:
            out = ops.convert_type(image.data, target_dtype, scale_range=scale_range, progress=tick)
        if tick.cancelled:
            self.status('Type conversion cancelled')
            return None
        info = axes_mod.fit(axes_mod.get(image), out.shape[:-1] if image.rgb else out.shape)
        kw: dict = {'axes': info}
        if not image.rgb:
            kw['colormap'] = image.colormap
        v = self.open_data(out, f'{target_dtype}_{image.name}', **kw)
        label = {'uint8': '8-bit', 'uint16': '16-bit', 'float32': '32-bit'}.get(target_dtype)
        if label:
            self.record(f'tiki.run({label!r})')
        self.status(f'Converted {image.name} to {target_dtype}')
        return v

    def flatten_rgb(self):
        viewer, image = self.require_image()
        visible = [
            ly for ly in viewer.layers
            if isinstance(ly, Image) and ly.visible and ly.name not in roi_mod.PREVIEW_NAMES and not ly.rgb
        ]
        if not visible:
            raise TikiError('no visible greyscale image layers to flatten into RGB')
        ref_shape = visible[0].data.shape
        same = [ly for ly in visible if ly.data.shape == ref_shape]
        if any(axes_mod.get(ly).index('C') is not None for ly in same):
            raise TikiError('split channels first before flattening to RGB', fix='split')
        total = int(np.prod(ref_shape[:-2])) if len(ref_shape) > 2 else 1
        specs = [(ly.data, ly.colormap, tuple(map(float, ly.contrast_limits))) for ly in same]
        with self.progress('RGB Color', total) as tick:
            rgb = ops.composite_rgb(specs, progress=tick)
        if tick.cancelled:
            self.status('RGB conversion cancelled')
            return None
        info = axes_mod.fit(axes_mod.get(same[0]), rgb.shape[:-1])
        v = self.open_data(rgb, f'RGB_{viewer.title}', axes=info, rgb=True)
        self.status(f'Flattened {len(same)} layer(s) to RGB')
        return v

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
        if axes_mod.get(image).index('C') is None:
            imgs = [ly for ly in viewer.layers if isinstance(ly, Image) and ly.name not in roi_mod.PREVIEW_NAMES]
            if len(imgs) > 1:
                for ly in imgs:
                    kw: dict = {'axes': axes_mod.get(ly), 'rgb': bool(ly.rgb)}
                    if not ly.rgb:
                        kw['colormap'] = ly.colormap
                        kw['contrast_limits'] = tuple(map(float, ly.contrast_limits))
                    self.open_data(ly.data, ly.name, **kw)
                self.status(f'Opened {len(imgs)} channels in separate windows')
                return
        layers = channels.split(viewer, image)
        self.reapply_tool(viewer)
        self._refresh_channels_tool(viewer)
        self.record("tiki.run('Split Channels')")
        self.status(f'Split {image.name} into {len(layers)} channel layers (run Split Channels again for separate windows)')

    def merge_channels(self):
        from . import channels

        viewer = self.windows.active
        if viewer is not None and any(
            isinstance(ly, Image) and 'split_from' in ly.metadata for ly in viewer.layers
        ):
            layer = channels.merge(viewer)
            self.reapply_tool(viewer)
            self._refresh_channels_tool(viewer)
            self.record("tiki.run('Merge Channels')")
            self.status(f'Merged channels back into {layer.name}')
            return layer
        return self._process('merge_channels_dialog')

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

    def brightness_contrast(self) -> None:
        from .contrast import BrightnessContrastWindow

        viewer, _ = self.require_image()
        if self.contrast_window is None:
            self.contrast_window = BrightnessContrastWindow(self)
        win = self.contrast_window
        win.show()
        win.raise_()
        win.set_target(viewer)

    def set_lut(self, name: str) -> None:
        from . import contrast

        viewer, image = self.require_image()
        contrast.apply_lut(image, name)
        self._refresh_channels_tool(viewer)
        self.record(f'tiki.set_lut({name!r})')
        self.status(f'{image.name}: LUT → {name}')

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
        self.record("tiki.run('Enhance Contrast')")
        self.status(f'Display range {lo:.4g}–{hi:.4g} (0.35% saturated)')

    def reset_contrast(self) -> None:
        viewer, image = self.require_image()
        image.reset_contrast_limits()
        self.record("tiki.run('Reset Display Range')")

    def sync_display(self, viewer=None) -> int:
        from . import contrast

        if viewer is None:
            viewer, _ = self.require_image()
        return contrast.sync_display_across_windows(self, viewer)

    def show_info(self):
        from .widgets import InfoWindow

        viewer, image = self.require_image()
        win = InfoWindow(viewer, image)
        self.keep_window(win)
        win.show()
        win.raise_()
        return win

    def _roi_box(self, viewer, image):
        current = self.current_roi(viewer)
        if current is None or current[1] in ('line', 'path'):
            return None
        return roi_mod.shape_box(current[0], roi_mod.spatial_shape(image))

    def duplicate(self, image=None, title: str | None = None, crop: bool = True,
                  stack: bool = True, ranges: dict | None = None):
        viewer, active = self.require_image()
        image = image or active
        box = self._roi_box(viewer, image) if crop else None
        info = axes_mod.get(image)
        specs: dict = {}
        if not stack and image.ndim > 2:
            idx = roi_mod.current_index(viewer, image)
            for d in range(image.ndim - 2):
                specs[d] = idx[d]
        elif ranges:
            specs = dict(ranges)
        result, dropped, mults = ops.substack(
            image.data, specs=specs, box=box, squeeze_single=True, rgb=bool(image.rgb)
        )
        new_axes = []
        for d, ax in enumerate(info.axes):
            if d in dropped:
                continue
            out_d = len(new_axes)
            step = ax.step * mults[d] if (ax.step and d in mults) else ax.step
            new_axes.append(axes_mod.Axis(ax.role, result.shape[out_d], step, ax.unit))
        new_info = axes_mod.AxesInfo(new_axes, info.source, info.confident, list(info.notes))
        expected_shape = result.shape[:-1] if image.rgb and result.ndim == len(new_axes) + 1 else result.shape
        if new_info.problems(expected_shape):
            new_info = axes_mod.fit(info, expected_shape)
        stem = Path(image.name).stem
        open_kw: dict = {'axes': new_info}
        if not image.rgb:
            open_kw['colormap'] = image.colormap
        if hasattr(result, 'compute') and not image.rgb:
            plane = np.asarray(result[(0,) * max(0, result.ndim - 2)])
            open_kw['contrast_limits'] = ops.saturated_limits(plane)
        return self.open_data(result, title or f'{stem}-1.tif', **open_kw)

    def crop(self) -> None:
        viewer, image = self.require_image()
        box = self._roi_box(viewer, image)
        if box is None:
            raise TikiError('Crop needs an area selection')
        y0, y1, x0, x1 = box
        if y1 <= y0 or x1 <= x0:
            raise TikiError('Selection is outside the image')
        self.push_undo_data('Crop', viewer, image)
        image.data = np.array(ops.crop_to_box(image.data, box, rgb=bool(image.rgb)))
        image.metadata['modified'] = True
        expected_shape = image.data.shape[:-1] if image.rgb else image.data.shape
        image.metadata['axes'] = axes_mod.fit(axes_mod.get(image), expected_shape).to_dict()
        self.select_none()
        viewer.reset_view()
        self.record("tiki.run('Crop')")
        self.status(f'Cropped {image.name} → {image.data.shape}')

    def z_project_dialog(self) -> None:
        from .layer_select import LayerCombo

        viewer, image = self.require_image()
        dlg = QDialog(self)
        dlg.setWindowTitle('ZProjection')
        form = QFormLayout(dlg)
        image_box = LayerCombo(viewer, Image, current=image)
        form.addRow('Image:', image_box)
        axis_box = QComboBox()
        start, stop = QSpinBox(), QSpinBox()
        method = QComboBox()
        method.addItems(list(ops.PROJECTIONS))
        note = QLabel('')
        note.setStyleSheet('color: #b00020;')

        def update_range(_=None):
            img = image_box.layer()
            if img is None or axis_box.currentData() is None:
                return
            n = img.data.shape[axis_box.currentData()]
            for spin in (start, stop):
                spin.setRange(1, n)
            start.setValue(1)
            stop.setValue(n)

        def update_axes(_=None):
            img = image_box.layer()
            axis_box.blockSignals(True)
            axis_box.clear()
            if img is not None:
                info = axes_mod.get(img)
                for d in range(img.ndim - 2):
                    if info.axes[d].role == 'C':
                        continue  # never average channels together
                    axis_box.addItem(f'{roi_mod.axis_name(viewer, img, d)} ({img.data.shape[d]})', d)
                default = axes_mod.stack_axis(info, prefer='ZT')
                if default is not None and axis_box.findData(default) >= 0:
                    axis_box.setCurrentIndex(axis_box.findData(default))
            axis_box.blockSignals(False)
            ok = axis_box.count() > 0
            note.setText('' if ok else 'This image has no stack axis to project along.')
            buttons.button(QDialogButtonBox.Ok).setEnabled(ok)
            update_range()

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        image_box.layer_changed.connect(update_axes)
        axis_box.currentIndexChanged.connect(update_range)
        update_axes()
        form.addRow('Project along:', axis_box)
        form.addRow('Start slice:', start)
        form.addRow('Stop slice:', stop)
        form.addRow('Projection type:', method)
        form.addRow(note)
        form.addRow(buttons)
        if dlg.exec_() != QDialog.Accepted:
            return
        self.z_project(image_box.layer(), ops.PROJECTIONS[method.currentText()], axis_box.currentData(),
                       start.value() - 1, stop.value())

    def z_project(self, image: Image, method: str, axis: int = 0, start: int = 0, stop: int | None = None):
        t0 = time.perf_counter()
        n = image.data.shape[axis]
        s0 = max(0, int(start))
        s1 = n if stop is None else min(n, int(stop))
        if s1 <= s0:
            raise TikiError('Z Project start slice must be less than or equal to stop slice')
        sl = [slice(None)] * image.data.ndim
        sl[axis] = slice(s0, s1)
        data = image.data[tuple(sl)]
        result = ops.z_project(data, method, axis)
        prefix = {'max': 'MAX', 'mean': 'AVG', 'sum': 'SUM', 'std': 'STD', 'median': 'MED', 'min': 'MIN'}[method]
        fit_shape = data.shape[:-1] if image.rgb else data.shape
        info = axes_mod.fit(axes_mod.get(image), fit_shape).without(axis)
        viewer = self.open_data(result, f'{prefix}_{image.name}', axes=info, rgb=bool(image.rgb))
        self.record(f'tiki.z_project({method!r}, axis={axis!r}, start={s0!r}, stop={s1!r})')
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

    def _sync_result_units(self, units: dict[str, str], reset: bool = False) -> None:
        if reset or not self.results.rows:
            self.results.set_units(units, replace=True)
            return
        if not units:
            if self.results.column_units:
                self.results.clear_units()
            return
        prev_x = self.results.column_units.get('X')
        new_x = units.get('X')
        if prev_x is None:
            self.results.clear_units()
        elif prev_x != new_x:
            self.results.clear_units()
            self.status(f'Note: Results mix {prev_x} and {new_x}; column units cleared')
        else:
            self.results.set_units(units, replace=False)

    def measure(self) -> None:
        viewer, image = self.require_image()
        plane = roi_mod.current_plane(viewer, image)
        index = roi_mod.current_index(viewer, image)
        base = {'Label': image.name}
        if image.ndim > 2:
            base['Slice'] = ','.join(str(i + 1) for i in index[:-2])

        info = axes_mod.get(image)
        scale_yx = (float(image.scale[-2]), float(image.scale[-1]))
        unit = axes_mod.spatial_unit(info)
        new_units = {'Area': f'{unit}²', 'Length': unit, 'X': unit, 'Y': unit, 'Perim.': unit, 'Feret': unit} if unit else {}
        self._sync_result_units(new_units)

        points = roi_mod.points_layer(viewer, create=False)
        if self.tool == 'point' and points is not None and len(points.data):
            sy, sx = scale_yx
            for p in points.data:
                y, x = (int(np.floor(c + 0.5)) for c in p[-2:])
                if 0 <= y < plane.shape[0] and 0 <= x < plane.shape[1]:
                    x_val = x if (unit is None and sx == 1.0) else float(x * sx)
                    y_val = y if (unit is None and sy == 1.0) else float(y * sy)
                    self.results.add_row(base | {'X': x_val, 'Y': y_val, 'Value': float(plane[y, x])})
            self.show_results()
            return

        current = self.current_roi(viewer)
        mask = None if current is None else roi_mod.shape_mask(*current, plane.shape)
        row = base | ops.mask_stats(plane, mask, roi_mod.pixel_area(image), scale=scale_yx, include=self.measurements)
        if current is not None and current[1] in ('line', 'path'):
            verts = np.asarray(current[0])[:, -2:] * np.asarray(scale_yx)
            row['Length'] = float(np.linalg.norm(np.diff(verts, axis=0), axis=1).sum())
        self.results.add_row(row)
        self.record("tiki.run('Measure')")
        self.show_results()

    def show_results(self) -> None:
        self.results.show()
        self.results.raise_()

    def plot_line_profile(self):
        from .widgets import LineProfileWindow

        viewer, image = self.require_image()
        current = self.current_roi(viewer)
        if current is None or current[1] not in ('line', 'path'):
            raise TikiError('Plot Profile needs a straight line or segmented line selection')
        win = LineProfileWindow(viewer, image)
        self.keep_window(win)
        win.show()
        self.windows.activate(viewer)
        return win

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
        info = axes_mod.get(image)
        win = TracesWindow(
            f'Traces — {image.name}', names, traces, viewer, image, axis,
            xlabel=roi_mod.axis_name(viewer, image, axis),
            step=info.axes[axis].step, unit=info.display_unit(axis),
        )
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
        from .layer_select import LayerCombo

        viewer, image = self.require_image()
        self.labels_layer(viewer)  # fails gently if there are no labels at all
        dlg = QDialog(self)
        dlg.setWindowTitle('Analyze Particles')
        form = QFormLayout(dlg)
        labels_box = LayerCombo(viewer, Labels)
        image_box = LayerCombo(viewer, Image, current=image)
        form.addRow('Labels:', labels_box)
        form.addRow('Measure intensity in:', image_box)
        scope = QComboBox()
        three_d = QCheckBox('Measure objects in 3D (volume, z centroid)', checked=True)
        min_size = QDoubleSpinBox()
        min_size.setRange(0.0, 1e9)
        min_size.setDecimals(0)
        max_size = QDoubleSpinBox()
        max_size.setRange(0.0, 1e9)
        max_size.setDecimals(0)
        max_size.setSpecialValueText('∞')
        phys_box = QCheckBox('Use physical units for size filter', checked=True)
        state = {'has_z': False, 'unit': None, 'unit_size': 1.0}

        def refresh_scopes(_=None):
            img = image_box.layer()
            info = axes_mod.get(img) if img is not None else None
            current = scope.currentData()
            scope.clear()
            for key, label in nd.scopes(info) if info else []:
                scope.addItem(label, key)
            if current and scope.findData(current) >= 0:
                scope.setCurrentIndex(scope.findData(current))
            state['has_z'] = info is not None and info.index('Z') is not None
            three_d.setVisible(state['has_z'])
            state['unit'] = axes_mod.spatial_unit(info) if info is not None else None
            phys_box.setVisible(bool(state['unit']))
            update_units()

        def update_units(_=None):
            img = image_box.layer()
            is_3d = state['has_z'] and three_d.isChecked() and scope.currentData() != nd.PLANE
            u = state['unit']
            if img is not None and u:
                (sz, sy, sx), _, _ = axes_mod.physical_scale_3d(axes_mod.get(img))
                state['unit_size'] = (sz * sy * sx) if is_3d else (sy * sx)
            else:
                state['unit_size'] = 1.0
            use_phys = bool(u and phys_box.isChecked())
            suffix = (f' {u}³' if is_3d else f' {u}²') if use_phys else (' voxels' if is_3d else ' px')
            decimals = 4 if use_phys else 0
            for spin in (min_size, max_size):
                spin.setSuffix(suffix)
                spin.setDecimals(decimals)

        image_box.layer_changed.connect(refresh_scopes)
        scope.currentIndexChanged.connect(update_units)
        three_d.toggled.connect(update_units)
        phys_box.toggled.connect(update_units)
        refresh_scopes()
        form.addRow('Measure:', scope)
        form.addRow(three_d)
        form.addRow(phys_box)
        form.addRow('Min size:', min_size)
        form.addRow('Max size:', max_size)
        exclude = QCheckBox('Exclude on edges')
        clear = QCheckBox('Clear results', checked=True)
        form.addRow(exclude)
        form.addRow(clear)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)
        if dlg.exec_() != QDialog.Accepted:
            return
        if labels_box.layer() is None or image_box.layer() is None:
            raise TikiError('choose a Labels layer and an image')
        unit_size = state['unit_size'] if (state['unit'] and phys_box.isChecked()) else 1.0
        min_px = (min_size.value() / unit_size) if min_size.value() > 0 else 0
        max_px = (max_size.value() / unit_size) if max_size.value() > 0 else None
        self.particles(viewer, image_box.layer(), labels_box.layer(), min_px, exclude.isChecked(),
                       clear.isChecked(), scope.currentData() or nd.PLANE,
                       state['has_z'] and three_d.isChecked(), max_size=max_px)

    def particles(self, viewer, image, layer, min_size=0, exclude_edges=False, clear=True,
                  scope: str = 'plane', three_d: bool = True, whole_stack: bool | None = None,
                  max_size: float | None = None):
        """Measure every object of Labels ``layer`` in ``image`` (plane, volume or whole stack)."""
        from skimage.segmentation import clear_border

        if whole_stack is not None:  # older keyword
            scope = nd.WHOLE if whole_stack else nd.PLANE
        info = axes_mod.get(image)
        if info.index('C') is not None:
            raise TikiError(f'{image.name} has a channel axis: split channels and measure one channel',
                            fix='split')
        labels = layer.data
        ln, ndim = labels.ndim, image.ndim
        if ln > ndim:
            raise TikiError(f'"{layer.name}" has more axes than "{image.name}"')
        z = info.index('Z')
        labels_have_z = z is not None and ln >= ndim - z
        by_volume = three_d and labels_have_z and scope != nd.PLANE
        if scope == nd.VOLUME and not labels_have_z:
            raise TikiError(f'"{layer.name}" is a single plane; measure "This plane" or threshold the volume')
        shape = tuple(image.data.shape[:ndim])
        blocks = nd.blocks(info, shape, roi_mod.current_index(viewer, image), scope, by_volume)
        phys_3d, phys_unit, _ = axes_mod.physical_scale_3d(info)
        if clear:
            self.results.clear()
        p_units: dict[str, str] = {}
        if phys_unit:
            if by_volume:
                p_units = {'Volume': f'{phys_unit}³', 'X': phys_unit, 'Y': phys_unit, 'Z': phys_unit}
            else:
                p_units = {
                    'Area': f'{phys_unit}²', 'X': phys_unit, 'Y': phys_unit,
                    'Perim.': phys_unit, 'Feret': phys_unit,
                }
        self._sync_result_units(p_units, reset=clear)
        total = 0
        with self.progress('Analyze Particles', len(blocks)) as tick:
            for n, block in enumerate(blocks):
                lab = np.asarray(labels[nd.for_layer(block, ln)])
                img = np.asarray(image.data[block])
                if image.rgb:
                    img = img[..., :3].mean(-1)
                if lab.shape != img.shape:  # e.g. 2D labels against a volume: measure plane by plane
                    lab = np.broadcast_to(lab, img.shape)
                if lab.max() == 1:  # a 0/1 mask: split into objects first
                    lab = ops.label_objects(lab > 0)
                if exclude_edges:
                    lab = clear_border(lab)
                if lab.min() < 0:
                    lab = np.where(lab > 0, lab, 0)
                want_min = min_size > 1 or (0 < min_size <= 1 and phys_unit)
                want_max = max_size is not None and max_size > 0
                if want_min or want_max:
                    sizes = np.bincount(lab.ravel())
                    keep = np.ones_like(sizes, dtype=bool)
                    keep[0] = False
                    if want_min:
                        keep &= sizes >= min_size
                    if want_max:
                        keep &= sizes <= max_size
                    lab = np.where(keep[lab], lab, 0)
                block_scale = [image.scale[d] for d, b in enumerate(block) if not isinstance(b, int)]
                if phys_unit:
                    scale_arg = phys_3d if (lab.ndim == 3 and by_volume) else phys_3d[1:]
                else:
                    scale_arg = block_scale[-2:] if (lab.ndim == 3 and not by_volume) else block_scale
                if lab.ndim == 3 and not by_volume:  # labels with Z, measured slice by slice
                    rows = []
                    for k in range(lab.shape[0]):
                        for row in ops.particle_table(lab[k], img[k], scale_arg[-2:]):
                            rows.append({'z': k + 1} | row)
                else:
                    rows = ops.particle_table(lab, img, scale_arg)
                where = nd.block_label(info, block)
                for row in rows:
                    if where:
                        row = {'Slice': where} | row
                    self.results.add_row(row)
                    total += 1
                if tick(n + 1, len(blocks)) is False:
                    break
        self.show_results()
        self.record(
            f'tiki.analyze_particles(min_size={min_size!r}, max_size={max_size!r}, '
            f'exclude_edges={exclude_edges!r}, clear={clear!r}, scope={scope!r}, three_d={three_d!r})'
        )
        self.status(f'Analyze Particles: {total} {"3D " if by_volume else ""}objects from "{layer.name}"')
        return total

    # --------------------------------------------------- Process filters

    def filter_dialog(self, key: str) -> None:
        from .filters import FilterDialog, run_filter

        viewer, image = self.require_image()
        dlg = FilterDialog(self, key, viewer, image)
        if dlg.exec_() != QDialog.Accepted:
            return
        run_filter(self, key, viewer, dlg.image, dlg.values(),
                   dlg.use_whole_stack(), dlg.use_3d())

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
                                      self.results.rows, self.results.columns,
                                      results_units=self.results.column_units)
        quiet = getattr(self, '_quiet_export', False)
        # Clipboard access can block indefinitely on headless Linux. Quiet
        # mode is used by tests and programmatic callers, which already get
        # the snippet as the return value.
        if not quiet:
            QApplication.clipboard().setText(snippet)
        self.status(f'Exported to {out}' + ('' if quiet else '; loader code copied to clipboard'))
        if not quiet:
            QMessageBox.information(
                self, 'Send to Notebook',
                f'Wrote {out}\n\nPaste into a notebook (already on your clipboard):\n\n'
                f"exec(open(r'{out / 'load_session.py'}').read())\n\nor open load_session.py.",
            )
        return snippet

    # ---------------------------------------------------------------- Macros

    def new_macro(self, path=None):
        from .macros import MacroEditor

        editor = MacroEditor(self, path)
        self.keep_window(editor)
        editor.show()
        return editor

    def edit_macro(self):
        from . import macros

        macros.MACROS_DIR.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getOpenFileName(self, 'Edit macro', str(macros.MACROS_DIR), 'Python (*.py)')
        if path:
            self.new_macro(path)

    def run_macro_file(self, path: str | None = None) -> bool:
        from . import macros

        if path is None:
            macros.MACROS_DIR.mkdir(parents=True, exist_ok=True)
            path, _ = QFileDialog.getOpenFileName(self, 'Run macro', str(macros.MACROS_DIR), 'Python (*.py)')
            if not path:
                return False
        code = Path(path).read_text()
        ok, out, _ = macros.run_code(self, code, str(path))
        if out:
            self.log(f'── {Path(path).name}\n{out}')
        if not ok:
            raise TikiError(f'macro {Path(path).name} stopped:\n\n{out.strip()[-1500:]}')
        self.status(f'Macro {Path(path).name} finished')
        return ok

    def show_log(self):
        from .macros import LogWindow

        if getattr(self, 'log_window', None) is None:
            self.log_window = LogWindow()
        self.log_window.show()
        self.log_window.raise_()
        return self.log_window

    def log(self, text: str) -> None:
        self.show_log().append(text)

    def show_macro_recorder(self):
        from .macros import MacroRecorderWindow

        if self.recorder_window is None:
            self.recorder_window = MacroRecorderWindow(self)
        self.recorder_window.show()
        self.recorder_window.raise_()
        return self.recorder_window

    def record(self, code: str) -> None:
        if self.recorder_window is not None and not self.recorder_window.paused:
            self.recorder_window.record(code)

    def show_console(self):
        from .console import PythonConsoleWindow

        if self.console_window is None:
            self.console_window = PythonConsoleWindow(self)
        self.console_window.show()
        self.console_window.raise_()
        self.console_window.activateWindow()
        self.console_window.input.setFocus()
        return self.console_window

    def show_macros_folder(self) -> None:
        from qtpy.QtCore import QUrl
        from qtpy.QtGui import QDesktopServices

        from . import macros

        macros.MACROS_DIR.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(macros.MACROS_DIR)))
        self.status(f'Macros folder: {macros.MACROS_DIR} (every .py here appears in Plugins ▸ Macros)')

    def _fill_macros_menu(self) -> None:
        from . import macros

        menu = self.macros_menu
        for action in menu.actions()[self._macro_fixed:]:
            menu.removeAction(action)
        found = macros.list_macros()
        menu.addSeparator()
        if not found:
            menu.addAction('(save macros to the macros folder to list them here)').setEnabled(False)
        for path in found:
            name = path.stem.replace('_', ' ')
            action = menu.addAction(name)
            action.setToolTip(str(path))
            action.triggered.connect(lambda _=False, p=str(path): self._run(p, lambda: self.run_macro_file(p)))

    # ------------------------------------------- Process extras & helpers

    def _process(self, name: str, **kwargs):
        from . import process

        return getattr(process, name)(self, **kwargs)

    def _fft_noise(self, name: str, **kwargs):
        from . import fft_noise

        return getattr(fft_noise, name)(self, **kwargs)

    def _coloc(self, name: str, **kwargs):
        from . import coloc

        return getattr(coloc, name)(self, **kwargs)

    def _batch(self, name: str, **kwargs):
        from . import batch

        return getattr(batch, name)(self, **kwargs)

    def _publish(self, name: str, **kwargs):
        from . import publish

        return getattr(publish, name)(self, **kwargs)

    def keep_window(self, win) -> None:
        """Keep a floating plot alive until it is closed (and close it with TIKI)."""
        win.setAttribute(Qt.WA_DeleteOnClose)
        self.profiles.append(win)
        win.destroyed.connect(lambda _=None, w=win: self.profiles.remove(w) if w in self.profiles else None)

    def histogram(self) -> None:
        from .widgets import HistogramWindow

        viewer, image = self.require_image()
        plane = roi_mod.current_plane(viewer, image)
        current = self.current_roi(viewer)
        where = 'whole plane'
        if current is not None and current[1] not in ('line', 'path'):
            plane = plane[roi_mod.shape_mask(*current, plane.shape)]
            where = 'selection'
        stats = ops.histogram_stats(plane)
        if stats['n'] == 0:
            raise TikiError('no finite pixel values to histogram')
        idx = roi_mod.current_index(viewer, image)[:-2]
        pos = f' ({nd.block_label(axes_mod.get(image), idx + (slice(None), slice(None)))})' if idx else ''
        win = HistogramWindow(f'Histogram of {image.name}{pos}, {where}', stats)
        self.keep_window(win)
        win.show()

    def _fill_recent_menu(self) -> None:
        menu = self.recent_menu
        menu.clear()
        items = recent.load()
        if not items:
            menu.addAction('(none yet)').setEnabled(False)
            return
        for path in items:
            action = menu.addAction(Path(path).name)
            action.setToolTip(path)
            action.setStatusTip(path)
            exists = Path(path).exists()
            action.setEnabled(exists)
            if not exists:
                action.setText(f'{Path(path).name} (missing)')
            action.triggered.connect(lambda _=False, p=path: self._run('Open', lambda: self.open_paths([p])))
        menu.addSeparator()
        menu.addAction('Clear List').triggered.connect(recent.clear)

    # -------------------------------------------------------------- Help

    def find_commands(self) -> None:
        dlg = CommandFinder(self._actions, self)
        dlg.exec_()

    def check_for_updates(self) -> None:
        from . import updates

        updates.check_for_updates_dialog(self)

    def about(self) -> None:
        from . import updates

        QMessageBox.about(
            self,
            'About TIKI',
            f'TIKI Is Kinda ImageJ (v{updates.current_version()})\n\n'
            'An ImageJ-style toolbar for napari image windows.',
        )

    # ------------------------------------------------------------ Qt events

    def dragEnterEvent(self, event):
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        if not paths:
            return
        event.acceptProposedAction()
        kinds = io.sort_dropped(paths)
        parts = [f'{len(v)} {k[:-1] if len(v) == 1 else k}' for k, v in kinds.items() if v and k != 'skipped']
        self.status('Drop to open ' + (', '.join(parts) if parts else 'nothing TIKI can read'), hold=False)

    def dragLeaveEvent(self, event):
        self.status('', hold=False)

    def dropEvent(self, event):
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        event.acceptProposedAction()
        self.open_dropped(paths)

    def _ask_folder_drop(self, folder: Path, count: int) -> str:
        box = QMessageBox(
            QMessageBox.Question,
            'Open Folder',
            f'"{folder.name}" contains {count} image files.\n\nOpen them as one lazy stack or as separate windows?',
            QMessageBox.Cancel,
            self,
        )
        as_stack = box.addButton('Open as Stack', QMessageBox.AcceptRole)
        options = box.addButton('Sequence Options…', QMessageBox.ActionRole)
        separate = box.addButton('Separate Windows', QMessageBox.ActionRole)
        box.setDefaultButton(as_stack)
        box.exec_()
        clicked = box.clickedButton()
        if clicked is as_stack:
            return 'stack'
        if clicked is options:
            return 'options'
        if clicked is separate:
            return 'separate'
        return 'cancel'

    def open_dropped(self, paths) -> None:
        """Open whatever was dropped, ImageJ-style: images (and folders of images)
        in new windows, .roi/.zip into the ROI Manager, .py macros in the editor."""
        self.raise_()
        remaining = []
        opened_sequence = False
        for raw in paths:
            p = Path(raw)
            if p.is_dir() and p.suffix.lower() != '.zarr':
                seq_files = io.find_sequence_files(p)
                if len(seq_files) >= self.FOLDER_SEQUENCE_MIN:
                    mode = self._ask_folder_drop(p, len(seq_files))
                    if mode == 'stack':
                        self._run('Open Sequence', lambda p=p: self.open_sequence(p))
                        opened_sequence = True
                        continue
                    if mode == 'options':
                        self._run('Import Image Sequence', lambda p=p: self._process('sequence_dialog', folder=p))
                        opened_sequence = True
                        continue
                    if mode == 'cancel':
                        continue
            remaining.append(raw)

        kinds = io.sort_dropped(remaining)
        if kinds['images']:
            self._run('Open', lambda: self.open_paths([str(p) for p in kinds['images']]))
        for p in kinds['rois']:
            self._run('Open ROIs', lambda p=p: self.show_roi_manager().open(str(p)))
        for p in kinds['macros']:
            self._run('Open Macro', lambda p=p: self.new_macro(str(p)))
        if kinds['skipped']:
            names = ', '.join(p.name for p in kinds['skipped'][:5])
            self.report('Open', f'TIKI cannot open: {names}\n\nSupported: images (.tif, .tiff, .czi, .nd2, .lif, '
                        '.zarr, .h5, .mat, .npy, .png, .jpg), folders of images, ImageJ ROIs (.roi, .zip) and macros (.py).')
        if not opened_sequence and not any(kinds.values()):
            self.status('Nothing to open')

    def closeEvent(self, event):
        for w in list(self.profiles):
            if not w.close():
                event.ignore()
                return
        srv = getattr(self, '_instance_server', None)
        if srv is not None:
            srv.close()
        self.windows.close_all()
        self.results.close()
        for w in (self.contrast_window, self.threshold_window, self.roi_manager, self.console_window,
                  self.recorder_window, getattr(self, 'channels_window', None),
                  getattr(self, 'log_window', None), getattr(self.results, '_summary_win', None)):
            if w is not None:
                w.close()
        event.accept()
        QApplication.instance().quit()


def format_error_report(context: str, exc: BaseException, tb_text: str) -> str:
    """Format an unexpected error with environment info for easy copy/paste."""
    import platform

    from . import updates

    try:
        import napari

        napari_ver = getattr(napari, '__version__', 'unknown')
    except Exception:
        napari_ver = 'unknown'
    header = '\n'.join([
        f'Command: {context}',
        f'TIKI: {updates.current_version()}',
        f'napari: {napari_ver}',
        f'Python: {platform.python_version()} ({platform.platform()})',
        f'Error: {type(exc).__name__}: {exc}',
        '',
    ])
    return header + tb_text.strip() + '\n'


def install_excepthook(tiki: Tiki) -> None:
    """Catch errors outside menu commands (mouse callbacks, timers, live plots).

    Without this PyQt6 calls abort() on any exception raised inside Qt.
    """
    in_hook = {'active': False}

    def hook(exc_type, exc, tb):
        tb_text = ''.join(traceback.format_exception(exc_type, exc, tb))
        traceback.print_exception(exc_type, exc, tb)
        if in_hook['active']:
            return
        in_hook['active'] = True
        try:
            details = format_error_report('Unexpected error', exc, tb_text)
            tiki.report(
                'Unexpected Error',
                f'{exc_type.__name__}: {exc}\n\nClick "Show Details" or "Copy Error Details" to share this error.',
                error=True,
                details=details,
            )
        except Exception:
            try:
                tiki.status(f'Error: {exc_type.__name__}: {exc}')
            except Exception:
                pass
        finally:
            in_hook['active'] = False

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
