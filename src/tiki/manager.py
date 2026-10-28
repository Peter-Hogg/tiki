"""Tracks the napari image windows and which one is "current", ImageJ-style.

Commands act on the most recently activated image window. Activating the
TIKI toolbar itself (or a dialog) does not change the current image.
"""

from __future__ import annotations

import napari
import numpy as np
from qtpy.QtCore import QEvent, QObject, QTimer, Signal
from qtpy.QtGui import QKeySequence, QShortcut
from qtpy.QtWidgets import QAction, QMenu

from . import roi as roi_mod


class WindowManager(QObject):
    active_changed = Signal(object)  # napari.Viewer | None
    windows_changed = Signal()
    cursor_moved = Signal(object, object)  # viewer, world position

    def __init__(self, reserved_shortcuts: list[str] | None = None):
        super().__init__()
        self.viewers: list[napari.Viewer] = []
        self._active: napari.Viewer | None = None
        self.reserved_shortcuts = [QKeySequence(s) for s in (reserved_shortcuts or [])]

    # -- creating windows -------------------------------------------------

    def unique_title(self, name: str) -> str:
        titles = {v.title for v in self.viewers}
        if name not in titles:
            return name
        stem, dot, ext = name.rpartition('.')
        if not dot:
            stem, ext = name, ''
        i = 1
        while True:
            cand = f'{stem}-{i}' + (f'.{ext}' if ext else '')
            if cand not in titles:
                return cand
            i += 1

    def new_window(self, data: np.ndarray | None = None, name: str = 'Untitled', **image_kwargs) -> napari.Viewer:
        """Open ``data`` in a new, slimmed-down napari window and make it current."""
        title = self.unique_title(name)
        viewer = napari.Viewer(title=title, show=True)
        self._slim(viewer)
        if data is not None:
            image_kwargs.setdefault('name', title)
            layer = viewer.add_image(data, **image_kwargs)
            n_stack = layer.ndim - 2
            stack_names = {1: ['z'], 2: ['t', 'z'], 3: ['t', 'z', 'c']}.get(
                n_stack, [f'dim{d}' for d in range(n_stack)]
            )
            viewer.dims.axis_labels = stack_names + ['y', 'x']
            # ImageJ opens stacks at the first slice; napari at the middle.
            viewer.dims.current_step = (0,) * viewer.dims.ndim
        self.register(viewer)
        return viewer

    def _slim(self, viewer: napari.Viewer) -> None:
        """Hide the layer docks so a window is mostly just the image."""
        qt_viewer = viewer.window._qt_viewer
        qt_viewer.dockLayerList.setVisible(False)
        qt_viewer.dockLayerControls.setVisible(False)

    def show_layer_controls(self, viewer: napari.Viewer, visible: bool = True) -> None:
        """Show napari's controls for the selected layer (brush size, contrast...)."""
        viewer.window._qt_viewer.dockLayerControls.setVisible(visible)

    def _strip_shortcuts(self, viewer: napari.Viewer) -> None:
        """Remove napari menu shortcuts that TIKI uses app-wide (e.g. Ctrl+M)."""
        if not self.reserved_shortcuts or viewer not in self.viewers:
            return
        reserved = {s.toString() for s in self.reserved_shortcuts}
        qwin = viewer.window._qt_window
        # napari's app-model actions are shared and parented to the QApplication,
        # so reach them through the window's menus as well as its children.
        actions = set(qwin.findChildren(QAction))
        for menu in qwin.findChildren(QMenu):
            actions.update(menu.actions())
        for action in actions:
            keep = [s for s in action.shortcuts() if s.toString() not in reserved]
            if len(keep) != len(action.shortcuts()):
                action.setShortcuts(keep)
        for shortcut in qwin.findChildren(QShortcut):
            if shortcut.key().toString() in reserved:
                shortcut.setEnabled(False)

    def register(self, viewer: napari.Viewer) -> None:
        if viewer in self.viewers:
            return
        self.viewers.append(viewer)
        qwin = viewer.window._qt_window
        qwin.installEventFilter(self)
        self._strip_shortcuts(viewer)
        # napari rebuilds menus lazily; strip again once the window settles.
        QTimer.singleShot(0, lambda: self._strip_shortcuts(viewer))
        viewer.cursor.events.position.connect(
            lambda event, v=viewer: self.cursor_moved.emit(v, v.cursor.position)
        )
        self.set_active(viewer)
        self.windows_changed.emit()

    # -- active window ----------------------------------------------------

    @property
    def active(self) -> napari.Viewer | None:
        if self._active is not None and self._active not in self.viewers:
            self._active = None
        return self._active

    def set_active(self, viewer: napari.Viewer | None) -> None:
        if viewer is self._active:
            return
        self._active = viewer
        self.active_changed.emit(viewer)

    def activate(self, viewer: napari.Viewer) -> None:
        """Bring ``viewer`` to the front and make it current."""
        qwin = viewer.window._qt_window
        qwin.showNormal() if qwin.isMinimized() else qwin.show()
        qwin.raise_()
        qwin.activateWindow()
        self.set_active(viewer)

    def viewer_for_window(self, qwin) -> napari.Viewer | None:
        for v in self.viewers:
            if v.window._qt_window is qwin:
                return v
        return None

    def eventFilter(self, obj, event):
        etype = event.type()
        if etype == QEvent.WindowActivate:
            viewer = self.viewer_for_window(obj)
            if viewer is not None:
                self.set_active(viewer)
        elif etype == QEvent.Close:
            viewer = self.viewer_for_window(obj)
            if viewer is not None:
                # closeEvent may still be refused (confirm dialog); check after.
                QTimer.singleShot(0, lambda v=viewer: self._check_closed(v))
        return False

    def _check_closed(self, viewer: napari.Viewer) -> None:
        try:
            visible = viewer.window._qt_window.isVisible()
        except (RuntimeError, AttributeError):  # window already torn down
            visible = False
        if not visible:
            self.unregister(viewer)

    def unregister(self, viewer: napari.Viewer) -> None:
        if viewer not in self.viewers:
            return
        self.viewers.remove(viewer)
        if self._active is viewer:
            self.set_active(self.viewers[-1] if self.viewers else None)
        self.windows_changed.emit()

    def close(self, viewer: napari.Viewer) -> None:
        self.unregister(viewer)
        viewer.close()

    def close_all(self) -> None:
        for v in list(self.viewers):
            self.close(v)

    # -- convenience -------------------------------------------------------

    def active_image(self):
        viewer = self.active
        if viewer is None:
            return None, None
        return viewer, roi_mod.target_image(viewer)
