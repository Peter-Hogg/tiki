import numpy as np
import pytest

from tiki import roi
from tiki.app import Tiki
from tiki.io import synthetic_movie


@pytest.fixture
def tiki(qtbot):
    t = Tiki()
    qtbot.addWidget(t)
    t.show()
    yield t
    t.windows.close_all()


def _rect(layer, y0, y1, x0, x1, z=0):
    layer.add_rectangles(np.array([[[z, y0, x0], [z, y0, x1], [z, y1, x1], [z, y1, x0]]], float))


def test_open_measure_project_profile(tiki, qtbot):
    movie = synthetic_movie(n_frames=50, size=64)
    v1 = tiki.open_data(movie, 'movie.tif')
    assert tiki.windows.active is v1
    layer = v1.layers[roi.ROI_NAME]  # rectangle tool made the ROI layer
    assert v1.layers.selection.active is layer and layer.mode == 'add_rectangle'

    _rect(layer, 10, 20, 10, 30)
    tiki.measure()
    row = tiki.results.rows[-1]
    assert row['Area'] == 10 * 20
    np.testing.assert_allclose(row['Mean'], movie[0, 10:20, 10:30].mean(), rtol=1e-6)

    tiki.z_project(v1.layers['movie.tif'], 'max')
    v2 = tiki.windows.active
    assert v2 is not v1 and v2.title == 'MAX_movie.tif'
    np.testing.assert_array_equal(v2.layers[0].data, movie.max(0))

    # commands follow the active window
    tiki.windows.activate(v1)
    tiki.plot_z_profile()
    prof = tiki.profiles[-1]
    np.testing.assert_allclose(prof.profile, movie[:, 10:20, 10:30].mean(axis=(1, 2)), rtol=1e-6)

    # live update when the ROI changes
    layer.data = []
    _rect(layer, 0, 5, 0, 5)
    qtbot.waitUntil(lambda: np.allclose(prof.profile, movie[:, 0:5, 0:5].mean(axis=(1, 2)), rtol=1e-6), timeout=2000)


def test_duplicate_crops_to_roi_and_titles_unique(tiki):
    movie = synthetic_movie(n_frames=5, size=32)
    v = tiki.open_data(movie, 'a.tif')
    _rect(v.layers[roi.ROI_NAME], 4, 12, 8, 24)
    tiki.duplicate()
    assert tiki.windows.active.layers[0].data.shape == (5, 8, 16)
    tiki.windows.activate(v)
    tiki.duplicate()
    titles = [w.title for w in tiki.windows.viewers]
    assert len(set(titles)) == len(titles) == 3


def test_new_roi_replaces_old(tiki):
    v = tiki.open_data(np.zeros((32, 32)), 'img.tif')
    layer = v.layers[roi.ROI_NAME]
    layer.add_rectangles(np.array([[[1, 1], [1, 5], [5, 5], [5, 1]]], float))
    layer.add_rectangles(np.array([[[8, 8], [8, 12], [12, 12], [12, 8]]], float))
    tiki._prune(layer)
    assert layer.nshapes == 1
    np.testing.assert_allclose(layer.data[0][0], [8, 8])


def test_close_unregisters(tiki, qtbot):
    v1 = tiki.open_data(np.zeros((8, 8)), 'one')
    v2 = tiki.open_data(np.zeros((8, 8)), 'two')
    tiki.close_active()
    assert tiki.windows.viewers == [v1]
    assert tiki.windows.active is v1


def test_napari_ctrl_m_removed(tiki, qtbot):
    from qtpy.QtWidgets import QAction

    v = tiki.open_data(np.zeros((8, 8)), 'img')
    qtbot.wait(50)
    keys = {s.toString() for a in v.window._qt_window.findChildren(QAction) for s in a.shortcuts()}
    assert 'Ctrl+M' not in keys and 'Ctrl+O' not in keys


def test_app_shortcut_from_image_window(tiki, qtbot):
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QApplication

    v = tiki.open_data(np.ones((16, 16)), 'img')
    qwin = v.window._qt_window
    qwin.activateWindow()
    canvas = v.window._qt_viewer.canvas.native
    canvas.setFocus()
    qtbot.wait(50)
    qtbot.keyClick(canvas, Qt.Key_M, Qt.ControlModifier)
    qtbot.waitUntil(lambda: len(tiki.results.rows) == 1, timeout=2000)


def _drag(layer, start, end, modifiers=()):
    from napari.utils._test_utils import read_only_mouse_event
    from napari.utils.interactions import mouse_move_callbacks, mouse_press_callbacks, mouse_release_callbacks

    mouse_press_callbacks(layer, read_only_mouse_event(type='mouse_press', position=start, modifiers=list(modifiers)))
    mouse_move_callbacks(layer, read_only_mouse_event(type='mouse_move', is_dragging=True, position=end,
                                                      modifiers=list(modifiers)))
    mouse_release_callbacks(layer, read_only_mouse_event(type='mouse_release', position=end, modifiers=list(modifiers)))


def test_drawing_replaces_previous_roi_unless_shift(tiki, qtbot):
    v = tiki.open_data(np.zeros((64, 64)), 'img')
    layer = v.layers[roi.ROI_NAME]
    _drag(layer, (5, 5), (15, 20))
    qtbot.wait(20)
    assert layer.nshapes == 1
    _drag(layer, (30, 30), (40, 50))
    qtbot.wait(20)
    assert layer.nshapes == 1
    np.testing.assert_allclose(layer.data[0].min(axis=0), [30, 30])
    _drag(layer, (50, 5), (60, 10), modifiers=['Shift'])
    qtbot.wait(20)
    assert layer.nshapes == 2
