"""Analyze > Tools > ROI Manager, with Labels conversion and multi-ROI traces.

Like ImageJ's, the manager is global: its ROIs can be measured or plotted
on whichever image window is current. Selecting an entry makes it the
current selection in that window.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from napari.layers import Labels, Shapes
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from . import ops, style
from . import roi as roi_mod
from . import rois as R

OVERLAY_NAME = 'ROI Manager'
OVERLAY_COLOR = 'cyan'


class RoiManager(QWidget):
    def __init__(self, tiki):
        super().__init__(None, Qt.Window)
        style.apply(self)
        self.tiki = tiki
        self.rois: list[R.Roi] = []
        self._syncing = False
        self.setWindowTitle('ROI Manager')
        self.resize(330, 420)

        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list.itemSelectionChanged.connect(self._on_selection)
        self.list.itemDoubleClicked.connect(lambda _item: self._run('Rename', self.rename))

        buttons = QVBoxLayout()
        for text, slot, tip in (
            ('Add [Ctrl+T]', self.add_current, 'Add the current selection'),
            ('Update', self.update_selected, 'Replace the selected entry with the current selection'),
            ('Delete', self.delete, 'Delete selected entries (all if none selected)'),
            ('Rename…', self.rename, ''),
            ('Measure', self.measure, 'Measure selected (or all) ROIs on the current slice'),
            ('Multi Plot', self.multi_plot, 'Mean trace of each ROI through the stack'),
            ('From Labels', self.from_labels, 'One ROI per object in the Labels layer (current slice)'),
            ('To Labels', self.to_labels, 'Rasterise ROIs into a new Labels layer'),
            ('Open…', self.open, 'Read ImageJ .roi / .zip'),
            ('Save…', self.save, 'Write ImageJ .roi / .zip'),
            ('Deselect', self.list.clearSelection, ''),
        ):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, s=slot, t=text: self._run(t.split(' [')[0], s))
            buttons.addWidget(b)
        self.show_all = QCheckBox('Show All', checked=True)
        self.show_all.toggled.connect(lambda _: self.refresh_overlay())
        self.show_labels = QCheckBox('Labels', checked=True)
        self.show_labels.toggled.connect(lambda _: self.refresh_overlay())
        buttons.addWidget(self.show_all)
        buttons.addWidget(self.show_labels)
        buttons.addStretch()

        row = QHBoxLayout(self)
        row.addWidget(self.list, 1)
        row.addLayout(buttons)

    def _run(self, name, slot):
        self.tiki._run(name, slot)

    # ----------------------------------------------------------- helpers

    def _error(self, message):
        from .app import TikiError

        raise TikiError(message)

    def _targets(self) -> list[int]:
        """Selected rows, or every row if none are selected (ImageJ's rule)."""
        rows = sorted(i.row() for i in self.list.selectedIndexes())
        return rows or list(range(len(self.rois)))

    def _refresh_list(self, select: list[int] | None = None) -> None:
        self._syncing = True
        try:
            self.list.clear()
            for r in self.rois:
                self.list.addItem(r.name)
            for i in select or []:
                if 0 <= i < self.list.count():
                    self.list.item(i).setSelected(True)
        finally:
            self._syncing = False
        self.refresh_overlay()

    def add(self, rois: list[R.Roi]) -> None:
        start = len(self.rois)
        for r in rois:
            if not r.name:
                r.name = r.default_name(len(self.rois) + 1)
            self.rois.append(r)
        self._refresh_list()
        self.list.scrollToBottom()
        self.tiki.status(f'ROI Manager: {len(self.rois)} ROIs (+{len(self.rois) - start})')

    def _current_roi(self) -> R.Roi:
        viewer = self.tiki.windows.active
        if viewer is None:
            self._error('no image open')
        current = roi_mod.current_shape(roi_mod.roi_layer(viewer, create=False))
        if current is None:
            self._error('draw a selection first')
        verts, kind = current
        return R.Roi('', kind, np.asarray(verts)[:, -2:].copy())

    # ----------------------------------------------------------- actions

    def add_current(self) -> None:
        roi = self._current_roi()
        roi.name = roi.default_name(len(self.rois) + 1)
        self.add([roi])

    def update_selected(self) -> None:
        rows = sorted(i.row() for i in self.list.selectedIndexes())
        if len(rows) != 1:
            self._error('select exactly one entry to update')
        new = self._current_roi()
        new.name = self.rois[rows[0]].name
        self.rois[rows[0]] = new
        self._refresh_list(select=rows)

    def delete(self) -> None:
        rows = sorted(i.row() for i in self.list.selectedIndexes())
        if not rows:
            if not self.rois:
                return
            if QMessageBox.question(self, 'ROI Manager', f'Delete all {len(self.rois)} ROIs?') != QMessageBox.Yes:
                return
            rows = list(range(len(self.rois)))
        for i in reversed(rows):
            del self.rois[i]
        self._refresh_list()

    def rename(self) -> None:
        rows = sorted(i.row() for i in self.list.selectedIndexes())
        if len(rows) != 1:
            self._error('select one entry to rename')
        name, ok = QInputDialog.getText(self, 'Rename', 'Name:', text=self.rois[rows[0]].name)
        if ok and name:
            self.rois[rows[0]].name = name
            self._refresh_list(select=rows)

    def measure(self) -> None:
        viewer, image = self.tiki.require_image()
        if not self.rois:
            self._error('the ROI Manager is empty')
        plane = roi_mod.current_plane(viewer, image)
        index = roi_mod.current_index(viewer, image)
        for i in self._targets():
            roi = self.rois[i]
            row = {'Label': image.name, 'ROI': roi.name}
            if image.ndim > 2:
                row['Slice'] = ','.join(str(j + 1) for j in index[:-2])
            row |= ops.mask_stats(plane, roi.mask(plane.shape), roi_mod.pixel_area(image))
            self.tiki.results.add_row(row)
        self.tiki.show_results()

    def multi_plot(self) -> None:
        viewer, image = self.tiki.require_image()
        rows = self._targets()
        if not rows:
            self._error('the ROI Manager is empty')
        shape2d = roi_mod.spatial_shape(image)
        masks = [self.rois[i].mask(shape2d) for i in rows]
        self.tiki.show_traces(viewer, image, [self.rois[i].name for i in rows], masks=masks)

    def from_labels(self) -> None:
        viewer, image = self.tiki.require_image()
        layer = self.tiki.labels_layer(viewer)
        plane = self.tiki.labels_plane(viewer, layer)
        rois = R.rois_from_labels(plane)
        if not rois:
            self._error(f'"{layer.name}" has no objects on this slice')
        self.add(rois)

    def to_labels(self) -> None:
        viewer, image = self.tiki.require_image()
        rows = self._targets()
        if not rows:
            self._error('the ROI Manager is empty')
        data = R.labels_from_rois([self.rois[i] for i in rows], roi_mod.spatial_shape(image))
        layer = Labels(data, name='ROI Manager labels', opacity=0.6, **_plane_kw(image))
        viewer.add_layer(layer)
        self.tiki.reapply_tool(viewer)
        self.tiki.status(f'{len(rows)} ROIs → Labels layer "{layer.name}" (ids in list order)')

    def open(self, path: str | None = None) -> None:
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, 'Open ROIs', str(Path.home()), 'ImageJ ROIs (*.roi *.zip)')
            if not path:
                return
        rois, skipped = R.load_imagej(path)
        self.add(rois)
        if skipped:
            self.tiki.status(f'Opened {len(rois)} ROIs; skipped {skipped} unsupported (composite/text) ROIs')

    def save(self, path: str | None = None) -> None:
        if not self.rois:
            self._error('the ROI Manager is empty')
        rows = self._targets()
        if path is None:
            default = 'RoiSet.zip' if len(rows) > 1 else f'{self.rois[rows[0]].name}.roi'
            path, _ = QFileDialog.getSaveFileName(self, 'Save ROIs', default, 'ImageJ ROIs (*.zip *.roi)')
            if not path:
                return
        R.save_imagej(path, [self.rois[i] for i in rows])
        self.tiki.status(f'Saved {len(rows)} ROIs to {path}')

    # ------------------------------------------------------- selection

    def _on_selection(self) -> None:
        if self._syncing:
            return
        rows = sorted(i.row() for i in self.list.selectedIndexes())
        viewer = self.tiki.windows.active
        if len(rows) == 1 and viewer is not None and roi_mod.target_image(viewer) is not None:
            roi = self.rois[rows[0]]
            if roi.kind != 'point':
                layer = self.tiki.roi_layer(viewer)
                layer.data = []
                layer.add(roi.yx, shape_type=roi.kind)
            if roi.position is not None:
                lead = viewer.dims.ndim - 2
                if len(roi.position) <= lead:
                    step = list(viewer.dims.current_step)
                    step[lead - len(roi.position) : lead] = roi.position
                    viewer.dims.current_step = tuple(step)
        self.refresh_overlay()

    # ---------------------------------------------------------- overlay

    def refresh_overlay(self, viewer=None) -> None:
        """Draw all manager ROIs (cyan, named) on the current image window."""
        viewer = viewer or self.tiki.windows.active
        for v in self.tiki.windows.viewers:
            if v is not viewer and OVERLAY_NAME in v.layers:
                v.layers.remove(OVERLAY_NAME)
        if viewer is None:
            return
        image = roi_mod.target_image(viewer)
        shapes = [r for r in self.rois if r.kind != 'point']
        if not self.isVisible() or not self.show_all.isChecked() or image is None or not shapes:
            if OVERLAY_NAME in viewer.layers:
                viewer.layers.remove(OVERLAY_NAME)
            return
        data = [r.yx for r in shapes]
        kinds = [r.kind for r in shapes]
        features = {'name': [r.name for r in shapes]}
        text = {'string': '{name}', 'color': OVERLAY_COLOR, 'size': 8, 'anchor': 'upper_left'} \
            if self.show_labels.isChecked() else None
        selected = viewer.layers.selection.active
        if OVERLAY_NAME in viewer.layers:
            viewer.layers.remove(OVERLAY_NAME)
        layer = Shapes(data, shape_type=kinds, name=OVERLAY_NAME, features=features, text=text,
                       edge_color=OVERLAY_COLOR, face_color='transparent', edge_width=1,
                       **_plane_kw(image))
        viewer.add_layer(layer)
        roi = roi_mod.roi_layer(viewer, create=False)
        if roi is not None:  # the live selection stays on top
            viewer.layers.move(viewer.layers.index(roi), len(viewer.layers))
        if selected is not None and selected in viewer.layers:
            viewer.layers.selection.active = selected

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh_overlay()

    def hideEvent(self, event):
        super().hideEvent(event)
        for v in self.tiki.windows.viewers:
            if OVERLAY_NAME in v.layers:
                v.layers.remove(OVERLAY_NAME)


def _plane_kw(image) -> dict:
    from . import axes as axes_mod

    return axes_mod.geometry(image, ndim=2)
