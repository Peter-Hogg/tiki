import numpy as np
import pytest

from tiki import roi
from tiki.app import Tiki
from tiki.io import synthetic_movie


@pytest.fixture
def tiki(qtbot):
    t = Tiki()
    # napari's layer-control panels use OpenGL, which the offscreen test
    # platform lacks; record the request instead of showing them.
    t.windows.shown_controls = []
    t.windows.show_layer_controls = lambda v, visible=True: t.windows.shown_controls.append(v)
    t.set_tool('rectangle')  # most tests draw ROIs; the app default is the pan tool
    qtbot.addWidget(t)
    t.show()
    yield t
    t.windows.close_all()


def _rect(layer, y0, y1, x0, x1):
    layer.add_rectangles(np.array([[[y0, x0], [y0, x1], [y1, x1], [y1, x0]]], float))


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


def test_command_errors_are_reported_not_raised(tiki, monkeypatch):
    from qtpy.QtWidgets import QMessageBox

    shown = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda *a: shown.append(('warning', a[1], a[2])))
    monkeypatch.setattr(QMessageBox, 'critical', lambda *a: shown.append(('critical', a[1], a[2])))
    tiki.open_data(np.zeros((16, 16)), 'img')
    tiki._actions['Image>Crop'].trigger()  # no ROI drawn
    tiki._actions['Analyze>Plot Profile'].trigger()
    tiki._run('Boom', lambda: 1 / 0)
    assert [s[0] for s in shown] == ['warning', 'warning', 'critical']
    assert 'area selection' in shown[0][2]
    assert 'ZeroDivisionError' in shown[2][2]


def test_truncated_tiff_opens_with_warning(tmp_path):
    import tifffile

    from tiki import io

    path = tmp_path / 'short.tif'
    with tifffile.TiffWriter(path) as tw:
        tw.write(np.zeros((5, 8, 8), np.uint16), metadata={'shape': [9, 8, 8]})
    data, meta = io.read_image(path)
    assert data.shape[-2:] == (8, 8)


def test_threshold_preview_follows_window_and_apply_makes_labels(tiki, qtbot):
    movie = np.zeros((4, 32, 32), np.uint16)
    movie[:, 4:12, 4:12] = 1000
    movie[:, 20:30, 20:30] = 1000
    v1 = tiki.open_data(movie, 'movie.tif')
    tiki.threshold()
    win = tiki.threshold_window
    assert win.image is v1.layers['movie.tif']
    assert win.lo_val > 0 and win.hi_val == 1000  # dark background: objects above
    qtbot.waitUntil(lambda: win.preview is not None, timeout=1000)
    assert win.preview.data.sum() == 64 + 100
    assert v1.layers.selection.active.name == roi.ROI_NAME  # ROI tool still armed

    layer = win.apply()
    assert layer.data.shape == movie.shape and layer.data.max() == 8  # 2 objects x 4 slices
    assert 'threshold preview' not in [ly.name for ly in v1.layers]
    # result is handed over for painting
    assert v1.layers.selection.active is layer
    assert v1 in tiki.windows.shown_controls
    tiki.set_tool('brush')
    assert v1.layers.selection.active is layer and layer.mode == 'paint'

    v2 = tiki.open_data(np.ones((16, 16)), 'flat.tif')
    assert win.viewer is v2
    win.update_preview()
    assert 'threshold preview' not in [ly.name for ly in v1.layers]


def test_brush_creates_labels_layer_when_none(tiki):
    from napari.layers import Labels

    v = tiki.open_data(np.zeros((3, 16, 16), np.uint16), 'img')
    tiki.set_tool('eraser')
    layer = v.layers.selection.active
    assert isinstance(layer, Labels) and layer.data.shape == (3, 16, 16) and layer.mode == 'erase'
    tiki.set_tool('rectangle')
    assert v.layers.selection.active.name == roi.ROI_NAME


def test_roi_layer_is_2d_and_persists_across_slices(tiki):
    movie = synthetic_movie(n_frames=10, size=32)
    v = tiki.open_data(movie, 'm.tif')
    layer = v.layers[roi.ROI_NAME]
    assert layer.ndim == 2
    _rect(layer, 2, 10, 2, 10)
    v.dims.set_current_step(0, 7)
    tiki.measure()
    np.testing.assert_allclose(tiki.results.rows[-1]['Mean'], movie[7, 2:10, 2:10].mean(), rtol=1e-6)
    assert tiki.results.rows[-1]['Slice'] == '8'


# ------------------------------------------------------- new features

def _two_cell_movie(n=40):
    rng = np.random.default_rng(5)
    movie = rng.normal(100, 2, (n, 48, 48)).astype(np.float32)
    movie[:, 5:15, 5:15] += 50 + 30 * np.sin(np.arange(n) / 3)[:, None, None]
    movie[:, 30:42, 28:40] += 80
    return movie


def test_analyze_particles_from_threshold(tiki):
    movie = _two_cell_movie()
    v = tiki.open_data(movie, 'cells.tif')
    tiki.threshold()
    tiki.threshold_window._set_range(140, 1000)
    tiki.threshold_window.apply()
    n = tiki.particles(v, v.layers['cells.tif'], tiki.labels_layer(v), min_size=5)
    assert n == 2
    areas = sorted(r['Area'] for r in tiki.results.rows)
    assert areas == [100, 144]
    assert 'Circ.' in tiki.results.columns and 'Solidity' in tiki.results.columns


def test_roi_manager_measure_traces_save_and_labels(tiki, tmp_path):
    movie = _two_cell_movie()
    v = tiki.open_data(movie, 'cells.tif')
    mgr = tiki.show_roi_manager()
    layer = v.layers[roi.ROI_NAME]
    _rect(layer, 4.5, 14.5, 4.5, 14.5)
    tiki.add_to_manager()
    _rect(layer, 29.5, 41.5, 27.5, 39.5)
    tiki.add_to_manager()
    assert [r.name[:4] for r in mgr.rois] == ['0001', '0002']
    assert 'ROI Manager' in v.layers  # Show All overlay

    mgr.measure()
    assert [r['Area'] for r in tiki.results.rows[-2:]] == [100, 144]

    win = tiki.show_traces(v, v.layers['cells.tif'], [r.name for r in mgr.rois],
                           masks=[r.mask((48, 48)) for r in mgr.rois])
    assert win.raw.shape == (2, 40)
    np.testing.assert_allclose(win.raw[0], movie[:, 5:15, 5:15].mean(axis=(1, 2)), rtol=1e-6)

    path = tmp_path / 'RoiSet.zip'
    mgr.save(str(path))
    mgr.rois.clear()
    mgr.open(str(path))
    assert len(mgr.rois) == 2 and mgr.rois[1].mask((48, 48)).sum() == 144

    mgr.to_labels()
    labels = v.layers['ROI Manager labels']
    assert set(np.unique(labels.data)) == {0, 1, 2}
    mgr.rois.clear()
    v.layers.selection.active = labels
    mgr.from_labels()
    assert len(mgr.rois) == 2

    # selecting an entry makes it the current selection
    mgr.list.item(1).setSelected(True)
    assert roi.current_shape(layer)[0].mean(axis=0) == pytest.approx(mgr.rois[1].yx.mean(axis=0))


def test_traces_from_labels_and_dff(tiki):
    movie = _two_cell_movie()
    v = tiki.open_data(movie, 'cells.tif')
    tiki.threshold()
    tiki.threshold_window._set_range(140, 1000)
    w = tiki.threshold_window
    w.scope.setCurrentIndex(w.scope.findData('plane'))
    tiki.threshold_window.apply()
    tiki.traces_from_labels()
    win = tiki.traces[-1]
    assert win.raw.shape[0] == 2 and win.raw.shape[1] == 40
    win.dff.setChecked(True)
    assert np.allclose(np.nanmedian(win.traces, axis=1), 0)


def test_filter_preview_and_result_layer(tiki, qtbot):
    from tiki.filters import FilterDialog, run_filter

    movie = _two_cell_movie(5)
    v = tiki.open_data(movie, 'cells.tif')
    dlg = FilterDialog(tiki, 'gaussian', v, v.layers['cells.tif'])
    assert 'filter preview' in v.layers
    assert roi.target_image(v).name == 'cells.tif'  # preview never becomes "the image"
    assert v.layers[-1].name == roi.ROI_NAME
    dlg.reject()
    assert 'filter preview' not in v.layers
    layer = run_filter(tiki, 'gaussian', v, v.layers['cells.tif'], {'sigma': 1.5})
    assert layer.data.shape == movie.shape and layer.data.dtype == np.float32
    assert roi.target_image(v) is layer
    assert v.layers[-1].name == roi.ROI_NAME
    bg = run_filter(tiki, 'background', v, v.layers['cells.tif'], {'radius': 20, 'light': False},
                    whole_stack=False)
    assert bg.data.ndim == 2


def test_send_to_notebook_roundtrip(tiki, tmp_path):
    movie = _two_cell_movie(10)
    v = tiki.open_data(movie, 'cells.tif')
    _rect(v.layers[roi.ROI_NAME], 4.5, 14.5, 4.5, 14.5)
    tiki.add_to_manager()
    tiki.measure()
    tiki.traces_from_manager()
    tiki.threshold()
    tiki.threshold_window.apply()
    tiki._quiet_export = True
    snippet = tiki.send_to_notebook(str(tmp_path / 'sess'))
    ns = {}
    exec(snippet, ns)
    np.testing.assert_array_equal(ns['image'], movie)
    assert ns['traces'].shape == (1, 10)          # one manager ROI x 10 frames
    assert ns['traces_names'] == [tiki.roi_manager.rois[0].name]
    assert len(ns['rois']) == 2                   # manager ROI + live selection
    assert ns['roi_masks'].shape == (2, 48, 48) and ns['roi_masks'][0].sum() == 100
    (lab,) = ns['labels'].values()
    assert lab.shape == movie.shape
    assert (tmp_path / 'sess' / 'RoiSet.zip').exists() and (tmp_path / 'sess' / 'results.csv').exists()


# ------------------------------------------------------------- nD axes

@pytest.fixture
def hyperstack(tmp_path):
    import tifffile

    rng = np.random.default_rng(0)
    data = rng.integers(100, 200, (6, 4, 2, 24, 20)).astype(np.uint16)
    data[:, :, 1, 5:10, 5:10] += 1000  # bright object in channel 2 only
    path = tmp_path / 'hyper.tif'
    tifffile.imwrite(path, data, imagej=True, resolution=(4, 4),
                     metadata={'axes': 'TZCYX', 'spacing': 2.0, 'finterval': 0.5, 'unit': 'um'})
    return path, data


def test_open_hyperstack_labels_scale(tiki, hyperstack):
    path, data = hyperstack
    tiki.open_paths([str(path)])
    v = tiki.windows.active
    assert tuple(v.dims.axis_labels) == ('t', 'z', 'c', 'y', 'x')
    assert tuple(v.layers[path.name].scale) == pytest.approx((0.5, 2.0, 1.0, 0.25, 0.25))
    assert 'TZCYX' not in tiki.status_label.text() and 'T6' in tiki.status_label.text()


def test_split_and_merge_channels_are_views(tiki, hyperstack):
    path, data = hyperstack
    tiki.open_paths([str(path)])
    v = tiki.windows.active
    original = v.layers[path.name]
    v.dims.set_current_step(0, 3)  # t = 3
    v.dims.set_current_step(1, 2)  # z = 2
    tiki.split_channels()
    assert v.dims.current_step[:2] == (3, 2)  # still on the same frame and slice
    chans = [ly for ly in v.layers if ly.name.startswith('C')]
    assert [ly.name for ly in chans] == [f'C1-{path.name}', f'C2-{path.name}']
    assert all(ly.blending == 'additive' for ly in chans)
    assert chans[1].data.shape == (6, 4, 24, 20)
    assert np.shares_memory(chans[1].data, original.data)
    assert tuple(v.dims.axis_labels) == ('t', 'z', 'y', 'x')
    tiki.merge_channels()
    assert v.layers[path.name] is original and tuple(v.dims.axis_labels) == ('t', 'z', 'c', 'y', 'x')


def test_z_project_keeps_t_and_c_and_traces_run_along_t(tiki, hyperstack):
    path, data = hyperstack
    tiki.open_paths([str(path)])
    v = tiki.windows.active
    image = v.layers[path.name]
    proj = tiki.z_project(image, 'max', axis=1)
    assert proj.layers[0].data.shape == (6, 2, 24, 20)
    assert tuple(proj.dims.axis_labels) == ('t', 'c', 'y', 'x')
    np.testing.assert_array_equal(proj.layers[0].data, data.max(axis=1))
    tiki.windows.activate(v)
    assert tiki._stack_axis(v, image) == 0  # T, not Z or C


def test_split_without_channels_offers_properties(tiki, monkeypatch):
    from qtpy.QtWidgets import QMessageBox

    seen = {}

    def fake_exec(box):
        seen['text'] = box.text()
        seen['buttons'] = [b.text() for b in box.buttons()]
        return 0

    monkeypatch.setattr(QMessageBox, 'exec_', fake_exec)
    monkeypatch.setattr(QMessageBox, 'exec', fake_exec, raising=False)
    tiki.open_data(np.zeros((5, 8, 8)), 'movie.npy')
    tiki._actions['Color>Split Channels'].trigger()
    assert 'no channel' in seen['text']
    assert 'Open Properties…' in seen['buttons']


def test_properties_relabel_and_remember(tiki, tmp_path, monkeypatch):
    from tiki import axes
    from tiki.properties import PropertiesDialog

    monkeypatch.setattr(axes, 'MEMORY_FILE', tmp_path / 'mem.json')
    path = tmp_path / 'cache.npy'
    np.save(path, np.zeros((30, 8, 8), np.float32))
    tiki.open_paths([str(path)])
    v = tiki.windows.active
    image = v.layers['cache.npy']
    dlg = PropertiesDialog(tiki, v, image)
    role, step, unit = dlg.rows[0]
    role.setCurrentIndex(axes.ROLES.index('X'))  # invalid: two X axes
    assert not dlg.buttons.button(dlg.buttons.Ok).isEnabled()
    role.setCurrentIndex(axes.ROLES.index('T'))
    step.setValue(0.001)
    assert dlg.buttons.button(dlg.buttons.Ok).isEnabled()
    dlg.remember.setChecked(True)
    dlg.apply()
    assert v.dims.axis_labels[0] == 't' and image.scale[0] == pytest.approx(0.001)
    np.save(tmp_path / 'other.npy', np.zeros((99, 8, 8), np.float32))
    tiki.open_paths([str(tmp_path / 'other.npy')])
    assert tiki.windows.active.dims.axis_labels[0] == 't'


def test_3d_view_disarms_drawing_and_assumes_cubic_voxels(tiki, hyperstack):
    import tifffile

    path, data = hyperstack
    nozstep = path.with_name('noz.tif')
    tifffile.imwrite(nozstep, data, imagej=True, resolution=(4, 4), metadata={'axes': 'TZCYX', 'unit': 'um'})
    tiki.open_paths([str(nozstep)])
    v = tiki.windows.active
    image = v.layers['noz.tif']
    assert tuple(image.scale)[1] == pytest.approx(0.25)  # z drawn like x/y
    assert str(image.units[1]) == 'micrometer'
    roi_layer = v.layers[roi.ROI_NAME]
    assert roi_layer.mode == 'add_rectangle'
    v.dims.ndisplay = 3
    assert roi_layer.mode == 'pan_zoom' and v.layers.selection.active is image
    assert 'drawing tools work in 2D' in tiki.status_label.text()
    v.dims.ndisplay = 2
    assert roi_layer.mode == 'add_rectangle'


def test_view_toolbar_present(tiki):
    from qtpy.QtWidgets import QToolBar

    v = tiki.open_data(np.zeros((4, 8, 8)), 'img')
    bars = [b for b in v.window._qt_window.findChildren(QToolBar) if b.objectName() == 'tiki_view_toolbar']
    assert len(bars) == 1


def test_default_tool_is_pan_and_draws_nothing(qtbot):
    t = Tiki()
    qtbot.addWidget(t)
    assert t.tool == 'hand'
    v = t.open_data(np.zeros((4, 8, 8)), 'img')
    assert roi.ROI_NAME not in v.layers  # no drawing layer until a drawing tool is chosen
    t.windows.close_all()


def test_calibrated_image_is_framed_by_camera(tiki, hyperstack):
    path, data = hyperstack
    tiki.open_paths([str(path)])
    v = tiki.windows.active
    image = v.layers[path.name]
    (lo, hi) = image.extent.world
    centre = (np.asarray(lo[-2:]) + np.asarray(hi[-2:])) / 2
    np.testing.assert_allclose(v.camera.center[-2:], centre, atol=0.5)  # in µm, not pixels


# ------------------------------------------------------- 3D threshold etc.

def _sphere_movie(nt=3):
    from test_ops import _spheres

    return np.stack([_spheres() for _ in range(nt)])  # T, Z, Y, X


def _open_with_axes(tiki, data, roles, name, steps=None):
    from tiki import axes

    info = axes.AxesInfo([axes.Axis(r, n, (steps or {}).get(r), 'µm' if (steps or {}).get(r) else None)
                          for r, n in zip(roles, data.shape)], 'test', True)
    return tiki.open_data(data, name, axes=info)


def test_threshold_zstack_3d_objects(tiki):
    from test_ops import _spheres

    v = _open_with_axes(tiki, _spheres(), 'ZYX', 'zstack.tif', {'Z': 2.0, 'Y': 0.5, 'X': 0.5})
    tiki.threshold()
    w = tiki.threshold_window
    assert [w.scope.itemData(i) for i in range(w.scope.count())] == ['plane', 'whole']
    assert w.scope.currentData() == 'whole' and w.connect_z.isVisible()
    w._set_range(500, 1000)
    layer = w.apply()
    assert layer.data.shape == (20, 40, 40) and len(np.unique(layer.data)) == 3  # bg + 2 spheres
    assert '3D labels' in layer.name
    n = tiki.particles(v, v.layers['zstack.tif'], layer, scope='whole', three_d=True)
    assert n == 2 and 'Volume' in tiki.results.columns and 'Z' in tiki.results.columns


def test_threshold_movie_volume_and_whole_unique_ids(tiki):
    v = _open_with_axes(tiki, _sphere_movie(), 'TZYX', 'tzyx.tif')
    tiki.threshold()
    w = tiki.threshold_window
    assert w.scope.currentData() == 'volume'
    w._set_range(500, 1000)
    vol = w.apply()
    assert vol.data.shape == (20, 40, 40) and vol.data.max() == 2
    tiki.threshold()
    w.scope.setCurrentIndex(w.scope.findData('whole'))
    w._set_range(500, 1000)
    whole = w.apply()
    assert whole.data.shape == (3, 20, 40, 40)
    assert len(np.unique(whole.data)) == 1 + 3 * 2  # 2 objects per t, unique ids
    w.connect_z.setChecked(False)
    w._set_range(500, 1000)
    flat = w.apply()
    assert flat.data.max() > 6  # per-plane objects


def test_threshold_image_dropdown_and_channel_gate(tiki, hyperstack):
    path, data = hyperstack
    tiki.open_paths([str(path)])
    v = tiki.windows.active
    tiki.threshold()
    w = tiki.threshold_window
    assert w.note_widget.isVisibleTo(w) and not w.action_buttons[1].isEnabled()
    w._split_channels()
    names = [w.image_box.itemText(i) for i in range(w.image_box.count())]
    assert names == [f'C1-{path.name}', f'C2-{path.name}']
    w.image_box.setCurrentIndex(1)
    assert w.image.name == f'C2-{path.name}' and w.action_buttons[1].isEnabled()
    w._set_range(900, 2000)
    layer = w.apply()
    assert layer.data.ndim in (3, 4)


def test_filter_3d_option(tiki):
    from test_ops import _spheres
    from tiki.filters import FilterDialog, run_filter

    v = _open_with_axes(tiki, _spheres().astype(np.float32), 'ZYX', 'z.tif', {'Z': 2.0, 'Y': 0.5, 'X': 0.5})
    image = v.layers['z.tif']
    dlg = FilterDialog(tiki, 'gaussian', v, image)
    assert dlg.has_z and dlg.use_3d()
    dlg.reject()
    flat = run_filter(tiki, 'gaussian', v, image, {'sigma': 2}, whole_stack=True, three_d=False)
    deep = run_filter(tiki, 'gaussian', v, image, {'sigma': 2}, whole_stack=True, three_d=True)
    assert flat.data.shape == deep.data.shape == image.data.shape
    assert not np.allclose(flat.data, deep.data) and '3D' in deep.name
