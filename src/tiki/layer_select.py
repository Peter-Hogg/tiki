"""A dropdown of one window's layers, so dialogs can say which layer to use."""

from __future__ import annotations

from napari.layers import Image, Labels
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QComboBox

from . import roi as roi_mod

PREVIEWS = roi_mod.PREVIEW_NAMES | {'threshold preview'}


class LayerCombo(QComboBox):
    """Lists the window's layers of ``kind`` (previews and TIKI overlays hidden).

    Stays in sync as layers are added, removed or renamed, and keeps the
    current choice when it still exists.
    """

    layer_changed = Signal(object)  # the selected layer or None

    def __init__(self, viewer, kind=Image, current=None, parent=None):
        super().__init__(parent)
        self.kind = kind
        self.viewer = None
        self.currentIndexChanged.connect(lambda _: self.layer_changed.emit(self.layer()))
        self.set_viewer(viewer, current)

    def set_viewer(self, viewer, current=None) -> None:
        if self.viewer is not None:
            for ev in (self.viewer.layers.events.inserted, self.viewer.layers.events.removed):
                try:
                    ev.disconnect(self.refresh)
                except Exception:
                    pass
        self.viewer = viewer
        if viewer is not None:
            viewer.layers.events.inserted.connect(self.refresh)
            viewer.layers.events.removed.connect(self.refresh)
        self.refresh(current=current)

    def layers(self) -> list:
        if self.viewer is None:
            return []
        return [ly for ly in self.viewer.layers if isinstance(ly, self.kind) and ly.name not in PREVIEWS]

    def refresh(self, event=None, current=None) -> None:
        keep = current if current is not None else self.layer()
        layers = self.layers()
        self.blockSignals(True)
        self.clear()
        for ly in layers:
            self.addItem(ly.name, ly)
        if keep in layers:
            self.setCurrentIndex(layers.index(keep))
        elif layers:
            default = roi_mod.target_image(self.viewer) if self.kind is Image else None
            if self.kind is Labels and isinstance(self.viewer.layers.selection.active, Labels):
                default = self.viewer.layers.selection.active
            self.setCurrentIndex(layers.index(default) if default in layers else len(layers) - 1)
        self.blockSignals(False)
        if self.layer() is not keep:
            self.layer_changed.emit(self.layer())

    def layer(self):
        return self.currentData() if self.count() else None
