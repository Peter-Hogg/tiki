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
    monkeypatch.setattr(
        QMessageBox,
        'exec_',
        lambda box: shown.append(('critical' if box.icon() == QMessageBox.Critical else 'warning', box.windowTitle(), box.text())) or QMessageBox.Close,
    )
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

    units = {'T': 's', 'Z': 'µm', 'Y': 'µm', 'X': 'µm'}
    info = axes.AxesInfo([axes.Axis(r, n, (steps or {}).get(r), units.get(r) if (steps or {}).get(r) else None)
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


# ------------------------------------------------------ process extras

@pytest.fixture
def accept_dialogs(monkeypatch):
    """Make modal dialogs return OK immediately, after an optional configure(dlg)."""
    from qtpy.QtWidgets import QDialog

    state = {'configure': None}

    def fake_exec(dlg):
        if state['configure']:
            state['configure'](dlg)
        return QDialog.Accepted

    monkeypatch.setattr(QDialog, 'exec_', fake_exec)
    monkeypatch.setattr(QDialog, 'exec', fake_exec, raising=False)
    return state


def _combo(dlg, text):
    """The QComboBox in ``dlg`` that offers ``text``."""
    from qtpy.QtWidgets import QComboBox

    for box in dlg.findChildren(QComboBox):
        if box.findText(text) >= 0:
            return box
    raise AssertionError(f'no combo offers {text!r}')


def _tyx_movie(tiki, n=50):
    rng = np.random.default_rng(1)
    movie = (200 + rng.normal(0, 1, (n, 24, 30))).astype(np.float32)
    movie[20:25, 10:14, 10:14] += 100
    return _open_with_axes(tiki, movie, 'TYX', 'movie.tif', {'T': 0.01, 'Y': 0.5, 'X': 0.5}), movie


def test_dff_dialog(tiki, accept_dialogs):
    from tiki import process

    v, movie = _tyx_movie(tiki)
    layer = process.dff_dialog(tiki)
    assert layer.name == 'movie.tif ΔF/F (median)' and layer.colormap.name == 'bwr'
    assert layer.data[22, 12, 12] == pytest.approx(0.5, abs=0.03)

    def first_frames_with_f0(dlg):
        _combo(dlg, 'movie.tif').setCurrentText('movie.tif')  # not the ΔF/F layer now on top
        _combo(dlg, 'first frames').setCurrentText('first frames')
        from qtpy.QtWidgets import QCheckBox

        next(b for b in dlg.findChildren(QCheckBox) if 'F0' in b.text()).setChecked(True)

    accept_dialogs['configure'] = first_frames_with_f0
    process.dff_dialog(tiki)
    assert 'movie.tif F0' in v.layers and v.layers['movie.tif F0'].data.shape == (24, 30)
    assert 'movie.tif ΔF/F (first frames)' in v.layers


def test_bleach_dialog_exponential(tiki, accept_dialogs):
    from tiki import process

    n = 80
    decay = (100 + 400 * np.exp(-np.arange(n) / 20)).astype(np.float32)
    movie = np.ones((n, 16, 16), np.float32) * decay[:, None, None]
    v = _open_with_axes(tiki, movie, 'TYX', 'bleach.tif', {'T': 0.5})
    layer = process.bleach_dialog(tiki)  # exponential fit is the default
    means = layer.data.mean(axis=(1, 2))
    assert np.allclose(means, means[0], rtol=1e-3)
    assert 'τ = 20' in tiki.status_label.text() and '= 10 s' in tiki.status_label.text()


def test_kymograph_dialog(tiki, accept_dialogs):
    from tiki import process

    n = 15
    movie = np.zeros((n, 20, 40), np.float32)
    for k in range(n):
        movie[k, 9:12, 5 + 2 * k] = 50
    v = _open_with_axes(tiki, movie, 'TYX', 'move.tif', {'T': 0.1, 'Y': 0.2, 'X': 0.2})
    v.layers[roi.ROI_NAME].add_lines(np.array([[10.0, 1.0], [10.0, 38.0]]))
    kv = process.kymograph_dialog(tiki)
    kymo = kv.layers[0].data
    assert kymo.shape == (15, 38)
    assert np.all(np.diff(kymo.argmax(axis=1)) == 2)  # 2 px per frame
    assert 'rows = t, 1 px = 0.1 s' in tiki.status_label.text()


def test_kymograph_needs_a_line(tiki, accept_dialogs, monkeypatch):
    from tiki import process
    from tiki.app import TikiError

    _tyx_movie(tiki)
    with pytest.raises(TikiError, match='line'):
        process.kymograph_dialog(tiki)


def test_image_calculator_dialog(tiki, accept_dialogs):
    from tiki import process

    a = np.full((4, 8, 8), 10, np.uint16)
    b = np.full((8, 8), 3, np.uint16)
    tiki.open_data(a, 'a.tif')
    tiki.open_data(b, 'flat.tif')
    tiki.windows.activate(tiki.windows.viewers[0])

    def configure(dlg):
        _combo(dlg, 'Subtract').setCurrentText('Subtract')
        from qtpy.QtWidgets import QComboBox

        boxes = [bx for bx in dlg.findChildren(QComboBox) if bx.findText('flat.tif') >= 0]
        boxes[0].setCurrentText('a.tif')
        boxes[1].setCurrentText('flat.tif')

    accept_dialogs['configure'] = configure
    result = process.calculator_dialog(tiki)
    assert result.title == 'Result of a.tif'
    np.testing.assert_array_equal(result.layers[0].data, np.full((4, 8, 8), 7, np.float32))


def test_set_scale_from_line(tiki, accept_dialogs):
    from qtpy.QtWidgets import QDoubleSpinBox

    from tiki import process

    v = tiki.open_data(np.zeros((64, 64), np.uint8), 'img.tif')
    layer = v.layers[roi.ROI_NAME]
    layer.add_lines(np.array([[10.0, 5.0], [10.0, 55.0]]))  # 50 px

    def configure(dlg):
        spins = dlg.findChildren(QDoubleSpinBox)
        assert spins[0].value() == pytest.approx(50)
        spins[1].setValue(10.0)  # 10 µm

    accept_dialogs['configure'] = configure
    step = process.set_scale_dialog(tiki)
    assert step == pytest.approx(0.2)
    assert tuple(v.layers['img.tif'].scale) == pytest.approx((0.2, 0.2))
    assert tuple(layer.scale) == pytest.approx((0.2, 0.2))  # the ROI moved with the image


def test_histogram_window(tiki):
    v = tiki.open_data(np.arange(100, dtype=np.uint16).reshape(10, 10), 'ramp.tif')
    tiki.histogram()
    win = tiki.profiles[-1]
    assert win.stats['n'] == 100 and win.stats['max'] == 99
    _rect(v.layers[roi.ROI_NAME], -0.5, 1.5, -0.5, 9.5)  # first two rows
    tiki.histogram()
    assert tiki.profiles[-1].stats['n'] == 20 and 'selection' in tiki.profiles[-1].windowTitle()


def test_open_recent(tiki, tmp_path, monkeypatch):
    from tiki import recent

    monkeypatch.setattr(recent, 'RECENT_FILE', tmp_path / 'recent.json')
    paths = []
    for name in ('one.npy', 'two.npy'):
        np.save(tmp_path / name, np.zeros((4, 4)))
        paths.append(tmp_path / name)
        tiki.open_paths([str(tmp_path / name)])
    assert recent.load() == [str(paths[1].resolve()), str(paths[0].resolve())]
    paths[0].unlink()
    tiki._fill_recent_menu()
    texts = [a.text() for a in tiki.recent_menu.actions()]
    assert texts[:2] == ['two.npy', 'one.npy (missing)']
    assert not tiki.recent_menu.actions()[1].isEnabled()


def test_correlation_image_and_negative_offset_traces(tiki, accept_dialogs):
    from tiki import process

    rng = np.random.default_rng(3)
    movie = (rng.normal(-250, 2, (200, 20, 24))).astype(np.float32)  # negative baseline
    sig = np.abs(rng.normal(0, 1, 200)).astype(np.float32)
    movie[:, 5:10, 5:10] += 20 * sig[:, None, None]
    v = _open_with_axes(tiki, movie, 'TYX', 'neg.tif', {'T': 0.033})
    cn_win = process.correlation_image(tiki)
    cn = cn_win.layers[0].data
    assert cn.shape == (20, 24) and cn[6:9, 6:9].mean() > 0.5 > cn[15:, 15:].mean()
    tiki.windows.activate(v)
    win = tiki.show_traces(v, v.layers['neg.tif'], ['cell'], masks=[np.pad(np.ones((5, 5), bool), ((5, 10), (5, 14)))])
    assert win.offset.value() < -250  # pre-set to the data minimum
    win.dff.setChecked(True)
    assert np.nanmax(win.traces) > 0.5  # transients point up, not down


# ------------------------------------------------------------ drag & drop

def _drop(widget, paths):
    from qtpy.QtCore import QMimeData, QPointF, Qt, QUrl
    from qtpy.QtGui import QDragEnterEvent, QDropEvent
    from qtpy.QtWidgets import QApplication

    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    pos = QPointF(10, 10)
    enter = QDragEnterEvent(pos.toPoint(), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(widget, enter)
    drop = QDropEvent(pos, Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(widget, drop)
    return enter, drop


def test_drop_files_folder_rois_macro_on_toolbar(tiki, tmp_path, monkeypatch):
    from tiki import macros, recent
    from tiki import rois as R

    monkeypatch.setattr(recent, 'RECENT_FILE', tmp_path / 'recent.json')
    folder = tmp_path / 'stack'
    folder.mkdir()
    for k in range(2):
        np.save(folder / f'img{k}.npy', np.zeros((4, 4)))
    (folder / 'notes.txt').write_text('ignored in folders')
    single = tmp_path / 'single.npy'
    np.save(single, np.ones((5, 5)))
    roi_zip = tmp_path / 'RoiSet.zip'
    R.save_imagej(roi_zip, [R.Roi('a', 'rectangle', np.array([[0.5, 0.5], [0.5, 2.5], [2.5, 2.5], [2.5, 0.5]]))])
    macro = tmp_path / 'hello.py'
    macro.write_text("print('hi')")
    reported = []
    monkeypatch.setattr(tiki, 'report', lambda *a, **k: reported.append(a))
    enter, drop = _drop(tiki, [single, folder, roi_zip, macro, tmp_path / 'data.csv'])
    assert enter.isAccepted()
    titles = [v.title for v in tiki.windows.viewers]
    assert titles == ['single.npy', 'img0.npy', 'img1.npy']
    assert [r.name for r in tiki.roi_manager.rois] == ['a']
    assert any(isinstance(w, macros.MacroEditor) and w.path == macro for w in tiki.profiles)
    assert reported and 'data.csv' in reported[0][1]


def test_drop_on_image_window_opens_new_window_not_layer(tiki, tmp_path, monkeypatch):
    import tifffile

    from tiki import recent

    monkeypatch.setattr(recent, 'RECENT_FILE', tmp_path / 'recent.json')
    v = tiki.open_data(np.zeros((8, 8)), 'first')
    movie = tmp_path / 'movie.tif'
    tifffile.imwrite(movie, np.zeros((10, 8, 8), np.uint16), imagej=True, metadata={'axes': 'TYX', 'finterval': 0.1})
    n_layers = len(v.layers)
    _drop(v.window._qt_viewer, [movie])
    assert len(v.layers) == n_layers  # napari did not add it as a layer
    new = tiki.windows.viewers[-1]
    assert new.title == 'movie.tif' and new.dims.axis_labels[0] == 't'  # TIKI read its axes


def test_close_all_after_dialogs_and_multiple_layers(tiki, qtbot):
    from tiki.filters import FilterDialog, run_filter

    v1 = tiki.open_data(np.ones((5, 16, 16), np.float32), 'm1.tif')
    v2 = tiki.open_data(np.ones((5, 16, 16), np.float32), 'm2.tif')
    filt = run_filter(tiki, 'gaussian', v2, v2.layers['m2.tif'], {'sigma': 1.0})
    tiki.plot_z_profile()
    prof = tiki.profiles[-1]
    tiki.threshold()
    tiki.threshold_window.close()
    dlg = FilterDialog(tiki, 'gaussian', v2, filt)
    dlg.reject()
    tiki.windows.activate(v1)
    activations = []
    tiki.windows.active_changed.connect(activations.append)

    tiki._actions['File>Close All'].trigger()
    qtbot.wait(220)

    assert tiki.windows.viewers == []
    assert tiki.windows.active is None
    assert activations == [None]  # no cascading activation through closing windows
    assert list(v2.layers) == []
    prof.update_profile(force=True)  # safe after its viewer closed


def test_single_instance_forwards_to_running_toolbar(tiki, qtbot, tmp_path, monkeypatch):
    from tiki import __main__, instance, recent

    monkeypatch.setattr(recent, 'RECENT_FILE', tmp_path / 'recent.json')
    sock = f'tiki-test-{tmp_path.name}'
    monkeypatch.setattr(instance, 'socket_name', lambda: sock)
    server = instance.listen(tiki, sock)
    try:
        a, b = tmp_path / 'a.npy', tmp_path / 'b.npy'
        np.save(a, np.zeros((6, 6)))
        np.save(b, np.ones((6, 6)))
        assert __main__.main([str(a)]) == 0
        assert __main__.main([str(b)]) == 0
        qtbot.waitUntil(lambda: [v.title for v in tiki.windows.viewers] == ['a.npy', 'b.npy'], timeout=2000)

        tiki._actions['File>Close All'].trigger()
        assert tiki.windows.viewers == []
    finally:
        server.close()


def test_brightness_contrast_window_luts_and_channels_tool(tiki, hyperstack):
    img1 = np.arange(100, dtype=np.uint16).reshape(10, 10)
    img2 = np.arange(200, 300, dtype=np.uint16).reshape(10, 10)
    v1 = tiki.open_data(img1, 'one.tif')
    v2 = tiki.open_data(img2, 'two.tif')

    tiki.brightness_contrast()
    bc = tiki.contrast_window
    assert bc is not None and bc.isVisible() and bc.viewer is v2

    bc.set_limits(210, 280)
    assert tuple(v2.layers['two.tif'].contrast_limits) == pytest.approx((210.0, 280.0))
    assert bc.lo_spin.value() == pytest.approx(210.0) and bc.hi_spin.value() == pytest.approx(280.0)

    bc.propagate()
    assert tuple(v1.layers['one.tif'].contrast_limits) == pytest.approx((210.0, 280.0))

    bc.auto()
    lo_a, hi_a = v2.layers['two.tif'].contrast_limits
    assert 200.0 <= lo_a < hi_a <= 299.0

    bc.reset()
    assert tuple(v2.layers['two.tif'].contrast_limits) == pytest.approx((200.0, 299.0))

    tiki.set_lut('Fire')
    assert v2.layers['two.tif'].colormap.name == 'fire'
    tiki.set_lut('HiLo')
    assert v2.layers['two.tif'].colormap.name.lower() == 'hilo'

    tiki.windows.activate(v1)
    assert bc.viewer is v1

    path, _ = hyperstack
    tiki.open_paths([str(path)])
    hv = tiki.windows.active
    tiki.split_channels()
    tiki.channels_tool()
    ct = tiki.channels_window
    assert ct.isVisible()
    from qtpy.QtWidgets import QPushButton

    auto_buttons = [b for b in ct.findChildren(QPushButton) if b.text() == 'Auto']
    assert len(auto_buttons) == 2
    auto_buttons[0].click()
    ch1 = hv.layers[f'C1-{path.name}']
    assert ch1.contrast_limits[0] < ch1.contrast_limits[1]


def test_duplicate_substack_and_bin_dialogs(tiki, accept_dialogs):
    from qtpy.QtWidgets import QCheckBox, QLineEdit, QSpinBox

    from tiki import process

    v, movie = _tyx_movie(tiki, n=12)
    _rect(v.layers[roi.ROI_NAME], 2, 12, 4, 20)

    def configure_stepped(dlg):
        edits = dlg.findChildren(QLineEdit)
        edits[-1].setText('1-10-2')

    accept_dialogs['configure'] = configure_stepped
    sub_v = process.duplicate_dialog(tiki)
    assert sub_v.layers[0].data.shape == (5, 10, 16)
    assert sub_v.layers[0].scale[0] == pytest.approx(0.02)

    tiki.windows.activate(v)
    accept_dialogs['configure'] = None
    sub2 = process.duplicate_dialog(tiki, substack_only=True)
    assert sub2.layers[0].data.shape == (12, 10, 16)

    tiki.windows.activate(v)

    def configure_plane_only(dlg):
        boxes = dlg.findChildren(QCheckBox)
        next(b for b in boxes if 'stack' in b.text().lower()).setChecked(False)

    accept_dialogs['configure'] = configure_plane_only
    plane_v = process.duplicate_dialog(tiki)
    assert plane_v.layers[0].data.ndim == 2

    tiki.windows.activate(v)

    def configure_bin(dlg):
        spins = dlg.findChildren(QSpinBox)
        for sp in spins:
            sp.setValue(2)

    accept_dialogs['configure'] = configure_bin
    bin_v = process.bin_dialog(tiki)
    assert bin_v.layers[0].data.shape == (6, 12, 15)
    assert tuple(bin_v.layers[0].scale) == pytest.approx((0.02, 1.0, 1.0))


def test_sequence_dialog_and_smart_folder_drop(tiki, tmp_path, accept_dialogs, monkeypatch):
    import tifffile

    from tiki import process

    seq = tmp_path / 'frames'
    seq.mkdir()
    for i in range(1, 6):
        tifffile.imwrite(seq / f'frame{i}.tif', np.full((12, 14), i * 10, dtype=np.uint16))

    accept_dialogs['configure'] = None
    v = process.sequence_dialog(tiki, folder=seq)
    assert v.layers[0].data.shape == (5, 12, 14)
    assert hasattr(v.layers[0].data, 'compute')

    monkeypatch.setattr(tiki, '_ask_folder_drop', lambda folder, n: 'stack')
    tiki.open_dropped([str(seq)])
    assert len(tiki.windows.viewers) == 2
    assert tiki.windows.active.layers[0].data.shape == (5, 12, 14)


def test_save_hyperstack_split_channels_and_labels(tiki, hyperstack, tmp_path):
    from napari.layers import Labels

    from tiki import io

    path, _ = hyperstack
    tiki.open_paths([str(path)])
    v = tiki.windows.active
    tiki.split_channels()
    c1 = v.layers[f'C1-{path.name}']
    c2 = v.layers[f'C2-{path.name}']
    v.layers.selection.active = c1
    tiki.set_lut('Green')
    v.layers.selection.active = c2
    tiki.set_lut('Fire')

    out = tmp_path / 'merged_out.tif'
    tiki.save_layer(out, viewer=v, merge_split=True)
    reopened, meta = io.read_image(out)
    assert reopened.shape == (6, 4, 2, 24, 20)
    assert meta['axes'].roles == 'TZCYX'
    assert meta['axes'].scale == pytest.approx([0.5, 2.0, 1.0, 0.25, 0.25])
    assert len(meta['imagej_luts']) == 2

    lab_data = np.zeros((24, 20), dtype=np.int32)
    lab_data[4:10, 4:10] = 7
    lab_layer = Labels(lab_data, name='cells')
    v.add_layer(lab_layer)
    lab_out = tmp_path / 'cells_labels.tif'
    tiki.save_layer(lab_out, layer=lab_layer, viewer=v)
    lab_back, lab_meta = io.read_image(lab_out)
    assert lab_back[6, 6] == 7 and lab_meta['axes'].scale == pytest.approx([0.25, 0.25])


def test_motion_correction_dialog_dual_channel(tiki, accept_dialogs):
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QComboBox
    from scipy.ndimage import gaussian_filter
    from scipy.ndimage import shift as nd_shift

    from tiki import process

    rng = np.random.default_rng(11)
    field = gaussian_filter(rng.normal(0, 1, (100, 100)).astype(np.float32), 2) * 50 + 100
    shifts = rng.uniform(-2.5, 2.5, (15, 2))
    c1 = np.stack([nd_shift(field, s, order=3)[25:75, 25:75] for s in shifts]).astype(np.float32)
    c2 = np.stack([nd_shift(field * 2, s, order=3)[25:75, 25:75] for s in shifts]).astype(np.float32)
    hyper = np.stack([c1, c2], axis=1)
    v = _open_with_axes(tiki, hyper, 'TCYX', 'dual.tif', {'T': 0.1, 'Y': 0.5, 'X': 0.5})
    tiki.split_channels()
    v.layers.selection.active = v.layers['C1-dual.tif']

    def configure(dlg):
        for cb in dlg.findChildren(QComboBox):
            idx = cb.findText('C2-dual.tif', Qt.MatchContains)
            if idx >= 0 and cb.findText('None') >= 0:
                cb.setCurrentIndex(idx)

    accept_dialogs['configure'] = configure
    reg1 = process.motion_correct_dialog(tiki)
    assert reg1 is not None and 'C1-dual.tif registered' in v.layers
    assert 'C2-dual.tif registered' in v.layers
    inner = (slice(None), slice(5, -5), slice(5, -5))
    assert v.layers['C2-dual.tif registered'].data[inner].std(0).mean() < 0.4 * c2[inner].std(0).mean()
    assert any('Motion shifts' in w.windowTitle() for w in tiki.profiles)


def test_watershed_dialog_preview_cleanup_and_fill_holes(tiki, accept_dialogs):
    from napari.layers import Labels

    from tiki import process

    yy, xx = np.mgrid[:64, :64]
    img = np.zeros((64, 64), dtype=np.float32)
    img[(yy - 20) ** 2 + (xx - 16) ** 2 <= 8**2] = 500
    img[(yy - 20) ** 2 + (xx - 27) ** 2 <= 8**2] = 500
    donut = ((yy - 46) ** 2 + (xx - 45) ** 2 <= 8**2) & ((yy - 46) ** 2 + (xx - 45) ** 2 >= 3**2)
    img[donut] = 500
    v = tiki.open_data(img, 'cells.tif')
    labels = Labels((img > 100).astype(np.int32), name='mask')
    v.add_layer(labels)
    v.layers.selection.active = labels

    accept_dialogs['configure'] = None
    ws = process.watershed_dialog(tiki)
    assert 'watershed preview' not in [ly.name for ly in v.layers]
    assert ws is not None and ws.data.max() == 3
    assert ws.data[46, 45] > 0

    v.layers.selection.active = labels
    filled = process.fill_holes_command(tiki)
    assert filled is not None and filled.data[46, 45] > 0


def test_open_czi_multi_scene_and_split_channels(tiki, tmp_path, monkeypatch):
    from tiki import axes, io

    info = axes.AxesInfo(
        [
            axes.Axis('C', 2),
            axes.Axis('Z', 3, 1.0, 'µm'),
            axes.Axis('Y', 10, 0.2, 'µm'),
            axes.Axis('X', 10, 0.2, 'µm'),
        ],
        'CZI metadata',
        True,
    )
    lut_g = io._czi_color_to_lut('#FF00FF00')
    lut_m = io._czi_color_to_lut('#FFFF00FF')
    p = tmp_path / 'zeiss.czi'
    p.write_bytes(b'dummy')
    monkeypatch.setattr(
        io,
        'read_czi_scenes',
        lambda _path: [
            (np.zeros((2, 3, 10, 10), np.uint16), {'name': 'zeiss.czi [S0]', 'path': str(p), 'axes': info, 'imagej_luts': [lut_g, lut_m]}),
            (np.ones((2, 3, 10, 10), np.uint16), {'name': 'zeiss.czi [S1]', 'path': str(p), 'axes': info, 'imagej_luts': [lut_g, lut_m]}),
        ],
    )
    tiki.open_paths([str(p)])
    assert [v.title for v in tiki.windows.viewers] == ['zeiss.czi [S0]', 'zeiss.czi [S1]']
    v = tiki.windows.active
    assert tuple(v.dims.axis_labels) == ('c', 'z', 'y', 'x')
    tiki.split_channels()
    assert [ly.name for ly in v.layers if ly.name.startswith('C')] == ['C1-zeiss.czi [S1]', 'C2-zeiss.czi [S1]']


def test_live_polyline_plot_profile(tiki, qtbot):
    img = np.zeros((30, 30), dtype=np.float32)
    img[9:12, :] = 100.0
    img[:, 19:22] = 200.0
    v = _open_with_axes(tiki, img, 'YX', 'lines.tif', {'Y': 0.5, 'X': 0.5})
    layer = v.layers[roi.ROI_NAME]
    layer.add_paths(np.array([[[10.0, 2.0], [10.0, 20.0], [25.0, 20.0]]]))
    win = tiki.plot_line_profile()
    assert len(win.values) > 20 and win.values[5] == pytest.approx(100.0, abs=1.0)
    assert 'µm' in win.ax.get_xlabel()
    win.width_spin.setValue(3)
    assert win.values[-3] == pytest.approx(200.0, abs=1.0)
    layer.data = []
    layer.add_lines(np.array([[20.0, 1.0], [20.0, 28.0]]))
    qtbot.waitUntil(lambda: win.values[5] < 10.0, timeout=2000)


def test_wand_restore_enlarge_band_and_clear(tiki, accept_dialogs):
    from tiki import process

    img = np.full((48, 48), 10.0, dtype=np.float32)
    img[16:28, 18:30] = 150.0
    v = tiki.open_data(img, 'cell.tif')
    tiki.wand_tolerance = 20.0
    poly = tiki.wand_select(v, 20, 22)
    assert poly is not None and v.layers[roi.ROI_NAME].nshapes == 1

    tiki.select_none()
    assert v.layers[roi.ROI_NAME].nshapes == 0
    tiki.restore_selection()
    assert v.layers[roi.ROI_NAME].nshapes == 1

    accept_dialogs['configure'] = None
    enlarged = process.enlarge_dialog(tiki)
    assert enlarged is not None
    band_layer = process.make_band_dialog(tiki)
    assert band_layer is not None and band_layer.data.sum() > 0

    cleared = process.clear_selection(tiki, outside=True)
    assert cleared is not None and cleared.data[0, 0] == 0.0 and cleared.data[20, 22] == 150.0


def test_transforms_type_conversion_and_rgb_flatten(tiki, hyperstack, accept_dialogs):
    from tiki import process

    arr = np.arange(20, dtype=np.uint16).reshape(4, 5)
    v = _open_with_axes(tiki, arr, 'YX', 'rect.tif', {'Y': 0.5, 'X': 1.0})
    fh = tiki.flip('horizontal')
    np.testing.assert_array_equal(fh.layers[0].data, arr[:, ::-1])

    tiki.windows.activate(v)
    r90 = tiki.rotate_90(-1)
    assert r90.layers[0].data.shape == (5, 4)
    assert tuple(r90.layers[0].scale) == pytest.approx((1.0, 0.5))

    tiki.windows.activate(v)
    accept_dialogs['configure'] = None
    rot_v = process.rotate_dialog(tiki)
    assert rot_v is not None and rot_v.layers[0].data.dtype == np.float32

    tiki.windows.activate(v)
    u8_v = tiki.convert_type('uint8')
    assert u8_v.layers[0].data.dtype == np.uint8

    path, _ = hyperstack
    tiki.open_paths([str(path)])
    tiki.split_channels()
    rgb_v = tiki.flatten_rgb()
    assert rgb_v.layers[0].rgb and rgb_v.layers[0].data.shape == (6, 4, 24, 20, 3)


def test_set_measurements_and_unit_headers(tiki, accept_dialogs):
    from qtpy.QtWidgets import QCheckBox

    from tiki import process

    img = np.arange(100, dtype=np.float32).reshape(10, 10)
    v = _open_with_axes(tiki, img, 'YX', 'cal.tif', {'Y': 0.5, 'X': 0.5})
    _rect(v.layers[roi.ROI_NAME], 1.5, 5.5, 2.5, 7.5)

    def configure(dlg):
        for cb in dlg.findChildren(QCheckBox):
            if cb.text() in ('Median', 'Centroid', 'Feret'):
                cb.setChecked(True)

    accept_dialogs['configure'] = configure
    process.set_measurements_dialog(tiki)
    tiki.measure()
    row = tiki.results.rows[-1]
    assert row['Area'] == pytest.approx(20 * 0.25)
    assert 'Median' in row and 'X' in row and 'Feret' in row
    headers = [tiki.results.table.horizontalHeaderItem(c).text() for c in range(tiki.results.table.columnCount())]
    assert 'Area (µm²)' in headers and 'X (µm)' in headers


def test_snapshot_png_svg_and_scale_bar(tiki, tmp_path):
    from PIL import Image as PILImage

    from tiki import publish

    arr = np.linspace(0, 255, 40 * 50, dtype=np.uint8).reshape(40, 50)
    v = _open_with_axes(tiki, arr, 'YX', 'snap.tif', {'Y': 0.5, 'X': 0.5})
    _rect(v.layers[roi.ROI_NAME], 5, 20, 10, 30)

    png = tmp_path / 'out.png'
    svg = tmp_path / 'out.svg'
    publish.save_snapshot(v, png, dpi=100, scale_bar=True, show_rois=True)
    publish.save_snapshot(v, svg, scale_bar=True, show_rois=True, timestamp='t = 1.0 s')
    assert png.exists() and png.stat().st_size > 100
    assert svg.exists() and '<svg' in svg.read_text() and 'µm' in svg.read_text()
    with PILImage.open(png) as im:
        assert im.size[0] > 0 and im.size[1] > 0


def test_export_movie_gif_and_mp4(tiki, tmp_path):
    from PIL import Image as PILImage

    from tiki import publish

    stack = np.zeros((6, 24, 30), dtype=np.uint8)
    for k in range(6):
        stack[k, 8:16, 4 + k * 3 : 10 + k * 3] = 220
    v = _open_with_axes(tiki, stack, 'TYX', 'mov.tif', {'T': 0.1, 'Y': 0.5, 'X': 0.5})

    gif = tmp_path / 'movie.gif'
    publish.export_movie(v, gif, axis=0, indices=[0, 2, 4], fps=10.0, scale_bar=True, show_timestamp=True)
    assert gif.exists() and gif.stat().st_size > 100
    with PILImage.open(gif) as im:
        assert getattr(im, 'n_frames', 1) == 3

    if publish.find_ffmpeg() is not None:
        mp4 = tmp_path / 'movie.mp4'
        publish.export_movie(v, mp4, axis=0, indices=[0, 1, 2, 3], fps=10.0, scale_bar=False, show_timestamp=False)
        assert mp4.exists() and mp4.stat().st_size > 100


def test_make_montage_dialog_and_orthogonal_views(tiki, accept_dialogs):
    from tiki import publish

    vol = np.arange(8 * 20 * 24, dtype=np.uint16).reshape(8, 20, 24)
    v = _open_with_axes(tiki, vol, 'ZYX', 'zstack.tif', {'Z': 2.0, 'Y': 0.5, 'X': 0.5})

    accept_dialogs['configure'] = None
    publish.make_montage_dialog(tiki)
    montage_v = tiki.windows.active
    assert montage_v is not v and 'Montage' in montage_v.title
    assert montage_v.layers[0].data.ndim == 2

    tiki.windows.activate(v)
    ortho = publish.orthogonal_views(tiki)
    assert ortho in tiki.profiles
    ortho.set_coords(z=5, y=10, x=12)
    assert v.dims.current_step[0] == 5
    assert 'Z=5' in ortho.coord_label.text() and 'µm' in ortho.coord_label.text()
    ortho.close()


def test_splash_updates_and_error_report(tiki, monkeypatch):
    from qtpy.QtWidgets import QApplication, QMessageBox

    from tiki import app as app_mod
    from tiki import splash, updates

    # 1. Splash pixmap builds cleanly with bundled icon
    pm = splash.build_pixmap('0.1.0')
    assert not pm.isNull() and pm.width() == 380 and pm.height() == 150

    # 2. Version comparison & update dialog (mocked network)
    assert updates.compare_versions('0.0.1', '0.1.0') == -1
    assert updates.compare_versions('0.1.0', 'v0.1.0') == 0
    assert updates.compare_versions('0.2.0', '0.1.9') == 1

    seen = {}

    def fake_exec(box):
        seen['title'] = box.windowTitle()
        seen['text'] = box.text()
        seen['details'] = box.detailedText()
        seen['clicked'] = None
        for btn in box.buttons():
            if btn.text() == 'Copy Error Details':
                seen['clicked'] = btn
                break
        return QMessageBox.Close

    monkeypatch.setattr(QMessageBox, 'exec_', fake_exec)
    monkeypatch.setattr(QMessageBox, 'clickedButton', lambda _self: seen.get('clicked'))
    updates.check_for_updates_dialog(
        tiki,
        fetcher=lambda: {'version': '9.9.9', 'url': 'https://example.com', 'notes': 'New features', 'none_published': False},
    )
    assert seen['title'] == 'Update Available' and '9.9.9' in seen['text']

    # 3. Unexpected error dialog includes formatted details and copies to clipboard
    def boom():
        raise RuntimeError('kaboom')

    tiki._run('Explode', boom)
    assert 'kaboom' in seen['text']
    assert 'TIKI:' in seen['details'] and 'RuntimeError: kaboom' in seen['details']
    assert 'RuntimeError: kaboom' in QApplication.clipboard().text()
    assert app_mod.format_error_report('Test', ValueError('x'), 'tb').startswith('Command: Test')


def test_python_console(tiki):
    arr = np.arange(20, dtype=np.uint16).reshape(4, 5)
    tiki.open_data(arr, 'cells.tif')
    con = tiki.show_console()
    assert 'cells.tif' in con.active_label.text()

    # Single-line eval against live `data`
    assert con.run_line('data.shape') is False
    assert '(4, 5)' in con.output.toPlainText()

    # Multi-line block + show() helper
    assert con.run_line('def scale2(x):') is True
    assert con.run_line('    return x * 2') is True
    assert con.run_line('') is False
    assert con.run_line('show(scale2(data), "doubled")') is False
    assert 'doubled' in tiki.windows.active.title

    # History stepping + error capture without crash
    con._step_history(-1)
    assert 'scale2(data)' in con.input.text()
    assert con.run_line('1 / 0') is False
    assert 'ZeroDivisionError' in con.output.toPlainText()

    # Tab completion
    con.input.setText('ops.make_mon')
    con.input.setCursorPosition(len('ops.make_mon'))
    con._complete()
    assert con.input.text().startswith('ops.make_montage')
    con.close()


def test_cursor_readout_on_dask_array(tiki):
    import dask.array as da

    arr = da.from_array(np.arange(16, dtype=np.uint16).reshape(4, 4), chunks=(2, 2))
    v = tiki.open_data(arr, 'lazy.tif')
    tiki._message_until = 0.0
    tiki._on_cursor(v, (2.0, 3.0))
    assert 'value=11' in tiki.status_label.text()


def test_stack_to_hyperstack_dialog_on_dask_sequence(tiki, accept_dialogs):
    import dask.array as da

    from tiki import process

    # Simulate an image sequence [ch1: 0..4, ch2: 100..104] as a lazy Dask array
    ch1 = np.arange(5, dtype=np.uint16)[:, None, None] * np.ones((1, 8, 8), dtype=np.uint16)
    ch2 = (100 + np.arange(5, dtype=np.uint16))[:, None, None] * np.ones((1, 8, 8), dtype=np.uint16)
    seq = da.from_array(np.concatenate([ch1, ch2], axis=0), chunks=(1, 8, 8))
    tiki.open_data(seq, 'sequence')

    # Default dialog settings for an even stack: order=xytzc (blocks), C=2, Z=1, T=5, Split channels=True
    accept_dialogs['configure'] = None
    v = process.stack_to_hyperstack_dialog(tiki)
    assert v is not None
    img_layers = [ly for ly in v.layers if ly.__class__.__name__ == 'Image']
    assert len(img_layers) == 2
    assert img_layers[0].data.shape == (5, 8, 8) and img_layers[1].data.shape == (5, 8, 8)
    assert int(np.asarray(img_layers[0].data[2, 0, 0])) == 2
    assert int(np.asarray(img_layers[1].data[2, 0, 0])) == 102

    # Calling Split Channels again on already-split channel layers opens each in its own window
    before = len(tiki.windows.viewers)
    tiki.split_channels()
    assert len(tiki.windows.viewers) == before + 2


def test_open_skimage_biology_samples(tiki):
    # Test bundled offline biology samples + Blobs alias
    tiki._open_skimage('cell')
    assert 'cell.tif' in tiki.windows.active.title
    assert tiki.windows.active.layers[0].data.ndim == 2

    tiki._open_skimage('immunohistochemistry')
    assert tiki.windows.active.layers[0].rgb

    tiki._open_skimage('blobs')
    assert tiki.windows.active.layers[0].data.dtype == np.uint8


def test_show_info_window_and_raw_metadata(tiki, tmp_path):
    import tifffile

    ome_xml = '''<?xml version="1.0" encoding="UTF-8"?>
    <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
      <Instrument ID="Instrument:0">
        <Objective ID="Objective:0" Manufacturer="Nikon" Model="Plan Apo 60x" LensNA="1.4"/>
      </Instrument>
      <Image ID="Image:0" Name="CalibStack">
        <Pixels DimensionOrder="XYCZT" Type="uint16" SizeX="16" SizeY="12" SizeZ="2" SizeC="2" SizeT="1"
                PhysicalSizeX="0.2" PhysicalSizeXUnit="&#181;m"
                PhysicalSizeY="0.2" PhysicalSizeYUnit="&#181;m"
                PhysicalSizeZ="1.0" PhysicalSizeZUnit="&#181;m">
          <Channel ID="Channel:0:0" Name="DAPI" Color="-16711681"/>
          <Channel ID="Channel:0:1" Name="GFP" Color="16711935"/>
        </Pixels>
      </Image>
    </OME>'''
    path = tmp_path / 'meta.ome.tif'
    arr = np.zeros((2, 2, 12, 16), dtype=np.uint16)
    with tifffile.TiffWriter(path) as tif:
        tif.write(arr, photometric='minisblack', description=ome_xml)

    tiki.open_paths([str(path)])
    win = tiki.show_info()
    summary_txt = win.text.toPlainText()
    assert 'CalibStack' in summary_txt
    assert 'Plan Apo 60x' in summary_txt
    assert 'DAPI' in summary_txt and 'GFP' in summary_txt
    assert win.raw_btn.isVisibleTo(win)
    assert not win.raw_box.isVisibleTo(win)
    win.raw_btn.setChecked(True)
    assert win.raw_box.isVisibleTo(win)
    assert 'OME-XML' in win.raw_box.toPlainText()
    win.close()


def test_multi_series_ome_and_channel_named_split(tiki, tmp_path):
    import tifffile

    ome_path = tmp_path / 'two_series.ome.tif'
    s0 = np.zeros((2, 8, 8), dtype=np.uint16)
    s1 = np.ones((2, 8, 8), dtype=np.uint16) * 5
    with tifffile.TiffWriter(ome_path, ome=True) as tif:
        tif.write(s0, photometric='minisblack', metadata={'axes': 'CYX', 'PhysicalSizeX': 0.25, 'PhysicalSizeY': 0.25})
        tif.write(s1, photometric='minisblack', metadata={'axes': 'CYX', 'PhysicalSizeX': 0.5, 'PhysicalSizeY': 0.5})

    before = len(tiki.windows.viewers)
    tiki.open_paths([str(ome_path)])
    assert len(tiki.windows.viewers) == before + 2

    # Split Channels uses channel_names from metadata and Merge restores
    from tiki import axes

    arr = np.zeros((2, 10, 10), dtype=np.uint16)
    arr[0] = 10
    arr[1] = 200
    info = axes.AxesInfo(
        [axes.Axis('C', 2), axes.Axis('Y', 10, 0.5, 'µm'), axes.Axis('X', 10, 0.5, 'µm')],
        'test', True,
    )
    v = tiki.open_data(arr, 'ch_named.tif', axes=info, metadata={'channel_names': ['DAPI', 'RFP']})
    tiki.split_channels()
    names = [ly.name for ly in v.layers if ly.__class__.__name__ == 'Image']
    assert names == ['C1 (DAPI)-ch_named.tif', 'C2 (RFP)-ch_named.tif']
    tiki.merge_channels()
    assert [ly.name for ly in v.layers if ly.__class__.__name__ == 'Image'] == ['ch_named.tif']


def test_calibrated_point_roi_manager_particles_summarize_and_unit_mismatch(tiki, tmp_path, monkeypatch):
    from napari.layers import Labels
    from qtpy.QtWidgets import QFileDialog
    from tiki import axes

    img = np.arange(20 * 20, dtype=np.float32).reshape(20, 20)
    info = axes.AxesInfo(
        [axes.Axis('Y', 20, 0.5, 'µm'), axes.Axis('X', 20, 0.5, 'µm')],
        'test', True,
    )
    v = tiki.open_data(img, 'calib.tif', axes=info)

    # Point tool records calibrated X/Y coordinates
    tiki.set_tool('point')
    pts = roi.points_layer(v)
    pts.add(np.array([[4.0, 6.0]]))
    tiki.measure()
    assert tiki.results.rows[-1]['X'] == pytest.approx(3.0)
    assert tiki.results.rows[-1]['Y'] == pytest.approx(2.0)
    assert tiki.results.column_units.get('X') == 'µm'

    # Line ROI in ROI Manager records calibrated Length
    tiki.results.clear()
    tiki.set_tool('rectangle')
    tiki.roi_layer(v).add(np.array([[0.0, 0.0], [0.0, 10.0]]), shape_type='line')
    mgr = tiki.show_roi_manager()
    mgr.add_current()
    mgr.measure()
    assert tiki.results.rows[-1]['Length'] == pytest.approx(5.0)
    assert tiki.results.column_units.get('Length') == 'µm'
    assert 'Length (µm)' in tiki.results.as_text()

    # ResultsTable.summarize produces Mean / SD / Min / Max
    summary = tiki.results.summarize(show=False)
    assert [s['Stat'] for s in summary] == ['Mean', 'SD', 'Min', 'Max']

    # CSV save includes unit headers
    csv_out = tmp_path / 'res.csv'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *a, **kw: (str(csv_out), 'CSV (*.csv)'))
    tiki.results.save_as()
    assert 'Length (µm)' in csv_out.read_text().splitlines()[0]

    # Unit mismatch (measuring an uncalibrated image) clears shared unit headers
    tiki.open_data(np.ones((10, 10), np.float32), 'uncalib.tif')
    tiki.measure()
    assert tiki.results.column_units == {}

    # 3D calibrated Analyze Particles (mixed nm in Z and µm in XY -> Volume in µm³ + max_size filter)
    vol = np.zeros((6, 16, 16), dtype=np.float32)
    lab = np.zeros((6, 16, 16), dtype=np.int32)
    lab[1:3, 2:4, 2:4] = 1  # 8 voxels
    lab[1:4, 8:12, 8:12] = 2  # 48 voxels
    vol[lab > 0] = 50.0
    info3 = axes.AxesInfo(
        [axes.Axis('Z', 6, 500.0, 'nm'), axes.Axis('Y', 16, 0.5, 'µm'), axes.Axis('X', 16, 0.5, 'µm')],
        'test', True,
    )
    v3 = tiki.open_data(vol, 'vol3d.tif', axes=info3)
    ly_lab = v3.add_labels(lab, name='objs', scale=info3.scale)
    # 500 nm = 0.5 µm -> voxel volume = 0.5 * 0.5 * 0.5 = 0.125 µm³; max_size=20 voxels keeps only label 1
    n_kept = tiki.particles(v3, v3.layers['vol3d.tif'], ly_lab, max_size=20, scope='whole', three_d=True)
    assert n_kept == 1
    assert tiki.results.rows[0]['Volume'] == pytest.approx(8 * 0.125)
    assert tiki.results.column_units['Volume'] == 'µm³'

    # Send to Notebook saves axes + results_units into session.json
    import json

    tiki._quiet_export = True
    tiki.send_to_notebook(str(tmp_path / 'nb'))
    manifest = json.loads((tmp_path / 'nb' / 'session.json').read_text())
    assert manifest['results_units']['Volume'] == 'µm³'
    assert manifest['axes']['axes'][0]['unit'] == 'nm'
    if isinstance(mgr, object):
        mgr.close()


def test_calibrated_profiles_traces_and_set_scale_options(tiki, qtbot, accept_dialogs):
    from qtpy.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QPushButton
    from tiki import axes, process

    mov = np.arange(5 * 12 * 12, dtype=np.float32).reshape(5, 12, 12)
    info = axes.AxesInfo(
        [axes.Axis('T', 5, 0.5, 's'), axes.Axis('Y', 12, 0.25, 'µm'), axes.Axis('X', 12, 0.25, 'µm')],
        'test', True,
    )
    v1 = tiki.open_data(mov, 't1.tif', axes=info)
    v2 = tiki.open_data(np.ones((8, 8), np.float32), 't2.tif')

    # Calibrated Z/T profile
    tiki.windows.activate(v1)
    _rect(tiki.roi_layer(v1), 2, 6, 2, 6)
    tiki.plot_z_profile()
    prof = tiki.profiles[-1]
    assert prof.unit == 's'
    np.testing.assert_allclose(prof.x, np.arange(5) * 0.5)
    assert '(s)' in prof.ax.get_xlabel()

    # Calibrated TracesWindow
    mask = np.zeros((12, 12), bool)
    mask[2:6, 2:6] = True
    tw = tiki.show_traces(v1, v1.layers['t1.tif'], ['ROI 1'], masks=[mask])
    assert tw.unit == 's' and tw.step == pytest.approx(0.5)
    assert '(s)' in tw.ax.get_xlabel()
    assert tw.table_text().splitlines()[0].startswith('t (s)')

    # Calibrated cursor readout includes t step in seconds
    tiki._message_until = 0.0
    tiki._on_cursor(v1, (2.0 * 0.5, 4.0 * 0.25, 4.0 * 0.25))
    assert '1 s' in tiki.status_label.text()

    # Set Scale: aspect ratio + Global across v1 and v2, then Remove Scale
    def cfg_global(dlg):
        spins = dlg.findChildren(QDoubleSpinBox)
        spins[0].setValue(10.0)   # dist px
        spins[1].setValue(5.0)    # known -> sx = 0.5
        spins[2].setValue(2.0)    # aspect -> sy = 1.0
        dlg.findChild(QComboBox).setCurrentText('nm')
        dlg.findChild(QCheckBox).setChecked(True)  # Global

    accept_dialogs['configure'] = cfg_global
    tiki.windows.activate(v2)
    process.set_scale_dialog(tiki)
    i1 = axes.get(v1.layers['t1.tif'])
    i2 = axes.get(v2.layers['t2.tif'])
    assert i1.axes[-1].step == pytest.approx(0.5) and i1.axes[-2].step == pytest.approx(1.0) and i1.axes[-1].unit == 'nm'
    assert i2.axes[-1].step == pytest.approx(0.5) and i2.axes[-2].step == pytest.approx(1.0)

    def cfg_remove(dlg):
        spins = dlg.findChildren(QDoubleSpinBox)
        spins[0].setValue(10.0)
        spins[1].setValue(5.0)
        for btn in dlg.findChildren(QPushButton):
            if 'Remove' in btn.text():
                btn.click()
                break

    accept_dialogs['configure'] = cfg_remove
    process.set_scale_dialog(tiki)
    i2_after = axes.get(v2.layers['t2.tif'])
    assert i2_after.axes[-1].step is None and i2_after.axes[-1].unit is None


def test_new_image_stack_and_process_commands(tiki, accept_dialogs):
    from qtpy.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit, QListWidget, QSpinBox
    from tiki import axes, filters, process

    # 1. Edit ▸ Invert
    arr = np.array([[0, 50, 255]], dtype=np.uint8)
    v = tiki.open_data(arr, 'base.tif')
    v_inv = tiki.invert()
    np.testing.assert_array_equal(v_inv.layers[0].data, np.array([[255, 205, 0]], dtype=np.uint8))

    # 2. Image ▸ Adjust ▸ Size… updates pixels and physical pixel step
    info_cal = axes.AxesInfo([axes.Axis('Y', 8, 1.0, 'µm'), axes.Axis('X', 10, 1.0, 'µm')], 'test', True)
    v_sz = tiki.open_data(np.arange(80, dtype=np.uint16).reshape(8, 10), 'res.tif', axes=info_cal)

    def cfg_resize(dlg):
        spins = dlg.findChildren(QDoubleSpinBox)
        spins[0].setValue(2.0)  # X scale -> 20x16, step 0.5 µm

    accept_dialogs['configure'] = cfg_resize
    v_resized = process.resize_dialog(tiki)
    assert v_resized.layers[0].data.shape == (16, 20)
    assert axes.get(v_resized.layers[0]).axes[-1].step == pytest.approx(0.5)

    # 3. Process ▸ Math… & Process ▸ Filters ▸ Find Edges + mean/unsharp
    tiki.windows.activate(v_sz)

    def cfg_math(dlg):
        _combo(dlg, 'Add').setCurrentText('Add')
        dlg.findChild(QDoubleSpinBox).setValue(5.0)

    accept_dialogs['configure'] = cfg_math
    math_ly = process.math_dialog(tiki)
    np.testing.assert_allclose(math_ly.data, v_sz.layers['res.tif'].data + 5.0)

    tiki.windows.activate(v_sz)
    edge_ly = process.find_edges_command(tiki)
    assert 'edges' in edge_ly.name and edge_ly.data.dtype == np.float32

    mean_ly = filters.run_filter(tiki, 'mean', v_sz, v_sz.layers['res.tif'], {'radius': 1.0}, whole_stack=False)
    sharp_ly = filters.run_filter(tiki, 'unsharp', v_sz, v_sz.layers['res.tif'], {'sigma': 1.0, 'weight': 0.5}, whole_stack=False)
    assert 'mean' in mean_ly.name and 'unsharp' in sharp_ly.name

    # Physical readout check in FilterDialog
    fdlg = filters.FilterDialog(tiki, 'mean', v_sz, v_sz.layers['res.tif'])
    assert 'µm' in fdlg.info.text()
    fdlg.reject()

    # 4. Process ▸ Labels ▸ Morphology…
    lab = np.zeros((10, 10), dtype=np.int32)
    lab[3:7, 3:7] = 1
    lab[0, 0] = 2
    v_sz.add_labels(lab, name='seg')

    def cfg_morph(dlg):
        _combo(dlg, 'Open').setCurrentText('Open')

    accept_dialogs['configure'] = cfg_morph
    morph_ly = process.morphology_dialog(tiki)
    assert morph_ly.data[0, 0] == 0 and morph_ly.data[5, 5] == 1

    # 5. Images to Stack, Concatenate, Reverse, Remove Slices, Hyperstack to Stack, Merge Channels
    p1 = np.zeros((8, 8), dtype=np.uint16)
    p2 = np.ones((8, 8), dtype=np.uint16) * 7
    vp1 = tiki.open_data(p1, 'plane1.tif')
    vp2 = tiki.open_data(p2, 'plane2.tif')

    accept_dialogs['configure'] = lambda dlg: None
    v_stk = process.images_to_stack_dialog(tiki)
    assert v_stk.layers[0].data.shape[0] >= 2

    # Concatenate explicitly checked stacks vp1 & vp2 along new T axis
    def cfg_concat(dlg):
        lw = dlg.findChild(QListWidget)
        for i in range(lw.count()):
            it = lw.item(i)
            ly = it.data(0x0100)[1] if hasattr(it.data(0x0100), '__getitem__') else None
            from qtpy.QtCore import Qt

            it.setCheckState(Qt.Checked if ly in (vp1.layers[0], vp2.layers[0]) else Qt.Unchecked)

    accept_dialogs['configure'] = cfg_concat
    v_cat = process.concatenate_dialog(tiki)
    assert v_cat.layers[0].data.shape == (2, 8, 8)
    assert v_cat.layers[0].data[1].min() == 7

    # Reverse + Remove Slices on a 4-frame stack
    stk4 = np.arange(4 * 6 * 6, dtype=np.uint16).reshape(4, 6, 6)
    tiki.open_data(stk4, 'stk4.tif', axes=axes.AxesInfo([axes.Axis('T', 4, 1.0, 's'), axes.Axis('Y', 6), axes.Axis('X', 6)], 'test', True))
    v_rev = process.reverse_stack_command(tiki)
    np.testing.assert_array_equal(v_rev.layers[0].data[0], stk4[-1])

    accept_dialogs['configure'] = lambda dlg: dlg.findChild(QLineEdit).setText('1, 4')
    tiki.windows.activate(v_rev)
    v_trim = process.remove_slices_dialog(tiki)
    assert v_trim.layers[0].data.shape == (2, 6, 6)

    # Hyperstack to Stack
    hs = np.zeros((2, 3, 6, 6), dtype=np.uint16)
    tiki.open_data(hs, 'hs.tif', axes=axes.AxesInfo([axes.Axis('C', 2), axes.Axis('Z', 3, 1.0, 'µm'), axes.Axis('Y', 6), axes.Axis('X', 6)], 'test', True))
    v_flat = process.hyperstack_to_stack_command(tiki)
    assert v_flat.layers[0].data.shape == (6, 6, 6)

    # Multi-window Merge Channels when no split layers exist in active viewer
    def cfg_merge(dlg):
        combos = dlg.findChildren(QComboBox)
        combos[0].setCurrentIndex(combos[0].findData(vp1.layers[0]))
        combos[1].setCurrentIndex(combos[1].findData(vp2.layers[0]))
        for cb in combos[2:]:
            cb.setCurrentIndex(0)
        dlg.findChild(QCheckBox).setChecked(False)

    accept_dialogs['configure'] = cfg_merge
    tiki.windows.activate(vp1)
    v_comp = tiki.merge_channels()
    assert v_comp.layers[0].data.shape == (2, 8, 8)


def test_fiji_process_commands_smooth_sharpen_variance_tophat_convolve_maxima(tiki, accept_dialogs):
    from qtpy.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QPlainTextEdit
    from tiki import axes, filters, process

    # Menu actions match Fiji Process layout
    for key in (
        'Process>Smooth',
        'Process>Sharpen',
        'Process>Find Edges',
        'Process>Find Maxima…',
        'Filters>Convolve…',
        'Filters>Variance…',
        'Filters>Top Hat…',
    ):
        assert key in tiki._actions

    # 1. Smooth & Sharpen on a stack
    mov = np.zeros((3, 12, 12), dtype=np.float32)
    mov[:, 6, 6] = 90.0
    info = axes.AxesInfo(
        [axes.Axis('Z', 3, 1.0, 'µm'), axes.Axis('Y', 12, 0.5, 'µm'), axes.Axis('X', 12, 0.5, 'µm')],
        'test', True,
    )
    v = tiki.open_data(mov, 'spots.tif', axes=info)
    sm_ly = process.smooth_command(tiki)
    assert 'smooth' in sm_ly.name and sm_ly.data[0, 6, 6] == pytest.approx(10.0)
    sh_ly = process.sharpen_command(tiki)
    assert 'sharpen' in sh_ly.name and sh_ly.data.shape == mov.shape

    # 2. Variance & Top Hat live filters (2D & 3D)
    v.layers.selection.active = v.layers['spots.tif']
    var_ly = filters.run_filter(tiki, 'variance', v, v.layers['spots.tif'], {'radius': 1.0}, whole_stack=True, three_d=True)
    assert 'var' in var_ly.name and '3D' in var_ly.name and float(var_ly.data[0, 6, 6]) > 0.0
    th_ly = filters.run_filter(tiki, 'tophat', v, v.layers['spots.tif'], {'radius': 2.0, 'light': False}, whole_stack=True)
    assert 'tophat' in th_ly.name and float(th_ly.data[1, 6, 6]) == pytest.approx(90.0)

    # 3. Convolve… dialog validation & preview cleanup
    cdlg = process.ConvolveDialog(tiki, v, v.layers['spots.tif'])
    cdlg.kernel_edit.setPlainText('1 2\n3')
    assert not cdlg.buttons.button(cdlg.buttons.Ok).isEnabled()
    cdlg.reject()
    assert 'filter preview' not in v.layers

    def cfg_conv(dlg):
        dlg.findChild(QPlainTextEdit).setPlainText('1 1 1\n1 1 1\n1 1 1')

    accept_dialogs['configure'] = cfg_conv
    conv_ly = process.convolve_dialog(tiki)
    assert 'filter preview' not in v.layers
    assert 'convolve 3×3' in conv_ly.name and conv_ly.data[0, 6, 6] == pytest.approx(10.0)

    # 4. Find Maxima…: preview cleanup + all 5 output modes
    peaks = np.zeros((2, 16, 16), dtype=np.float32)
    peaks[0, 4, 5] = 100.0
    peaks[0, 11, 12] = 60.0
    peaks[1, 8, 8] = 80.0
    vp = tiki.open_data(peaks, 'peaks.tif', axes=axes.AxesInfo(
        [axes.Axis('T', 2, 0.5, 's'), axes.Axis('Y', 16, 0.25, 'µm'), axes.Axis('X', 16, 0.25, 'µm')],
        'test', True,
    ))

    # Live preview appears while open and removes cleanly on cancel
    fdlg = process.FindMaximaDialog(tiki, vp, vp.layers['peaks.tif'])
    assert 'maxima preview' in vp.layers
    assert '2 maxima' in fdlg.count_label.text()
    fdlg.reject()
    assert 'maxima preview' not in vp.layers

    # 4a. Point Selection
    def cfg_pts(dlg):
        dlg.findChild(QDoubleSpinBox).setValue(10.0)
        dlg.output_box.setCurrentText('Point Selection')

    accept_dialogs['configure'] = cfg_pts
    pts_ly = process.find_maxima_dialog(tiki)
    assert 'maxima preview' not in vp.layers
    assert len(pts_ly.data) == 2

    # 4b. List (calibrated µm coordinates in Results)
    tiki.results.clear()

    def cfg_list(dlg):
        dlg.findChild(QDoubleSpinBox).setValue(10.0)
        dlg.output_box.setCurrentText('List')
        dlg.whole_stack.setChecked(False)

    accept_dialogs['configure'] = cfg_list
    n_rows = process.find_maxima_dialog(tiki)
    assert n_rows == 2 and tiki.results.column_units.get('X') == 'µm'
    assert tiki.results.rows[0]['X'] == pytest.approx(5 * 0.25)
    assert tiki.results.rows[0]['Y'] == pytest.approx(4 * 0.25)

    # 4c. Count over whole stack
    tiki.results.clear()

    def cfg_count(dlg):
        dlg.findChild(QDoubleSpinBox).setValue(10.0)
        dlg.output_box.setCurrentText('Count')
        dlg.whole_stack.setChecked(True)

    accept_dialogs['configure'] = cfg_count
    n_count_rows = process.find_maxima_dialog(tiki)
    assert n_count_rows == 2
    assert [r['Count'] for r in tiki.results.rows] == [2, 1]

    # 4d. Single Points + Segmented Particles
    def cfg_single(dlg):
        dlg.findChild(QDoubleSpinBox).setValue(10.0)
        dlg.output_box.setCurrentText('Single Points')
        dlg.whole_stack.setChecked(False)

    accept_dialogs['configure'] = cfg_single
    sp_ly = process.find_maxima_dialog(tiki)
    assert int(sp_ly.data[4, 5]) >= 1 and int((sp_ly.data > 0).sum()) == 2

    def cfg_seg(dlg):
        dlg.findChild(QDoubleSpinBox).setValue(10.0)
        dlg.output_box.setCurrentText('Segmented Particles')
        dlg.whole_stack.setChecked(False)

    accept_dialogs['configure'] = cfg_seg
    seg_ly = process.find_maxima_dialog(tiki)
    assert seg_ly.data.dtype == np.int32 and int(seg_ly.data.max()) == 2


def test_edge_detection_dialog_methods_and_3d(tiki, accept_dialogs):
    from napari.layers import Image, Labels
    from tiki import axes, process

    assert 'Process>Edge Detection…' in tiki._actions

    vol = np.zeros((6, 20, 20), dtype=np.float32)
    vol[1:5, 5:15, 5:15] = 100.0
    info = axes.AxesInfo(
        [axes.Axis('Z', 6, 1.0, 'µm'), axes.Axis('Y', 20, 0.5, 'µm'), axes.Axis('X', 20, 0.5, 'µm')],
        'test', True,
    )
    v = tiki.open_data(vol, 'edge3d.tif', axes=info)
    v.dims.set_current_step(0, 2)  # slice with square object

    # Preview appears while open, physical sigma readout updates, and preview cleans up on cancel
    edlg = process.EdgeDetectionDialog(tiki, v, v.layers['edge3d.tif'])
    assert 'filter preview' in v.layers
    assert 'µm' in edlg.phys_label.text()
    edlg.reject()
    assert 'filter preview' not in v.layers

    # Canny on current plane -> uint8 Image
    def cfg_canny_img(dlg):
        dlg.method_box.setCurrentIndex(dlg.method_box.findData('canny'))
        dlg.whole_stack.setChecked(False)
        dlg.as_labels.setChecked(False)

    accept_dialogs['configure'] = cfg_canny_img
    canny_ly = process.edge_detection_dialog(tiki)
    assert isinstance(canny_ly, Image) and canny_ly.data.dtype == np.uint8
    assert 'edges (canny)' in canny_ly.name and canny_ly.data.max() == 255
    assert 'filter preview' not in v.layers

    # Canny Output as Labels layer
    v.layers.selection.active = v.layers['edge3d.tif']

    def cfg_canny_lab(dlg):
        dlg.method_box.setCurrentIndex(dlg.method_box.findData('canny'))
        dlg.whole_stack.setChecked(False)
        dlg.as_labels.setChecked(True)

    accept_dialogs['configure'] = cfg_canny_lab
    canny_lab = process.edge_detection_dialog(tiki)
    assert isinstance(canny_lab, Labels) and canny_lab.data.max() == 1

    # Anisotropic 3D Scharr across Z-stack
    v.layers.selection.active = v.layers['edge3d.tif']

    def cfg_3d_scharr(dlg):
        dlg.method_box.setCurrentIndex(dlg.method_box.findData('scharr'))
        assert dlg.has_z
        dlg.three_d.setChecked(True)
        dlg.whole_stack.setChecked(True)

    accept_dialogs['configure'] = cfg_3d_scharr
    scharr_ly = process.edge_detection_dialog(tiki)
    assert isinstance(scharr_ly, Image) and scharr_ly.data.shape == vol.shape
    assert 'scharr 3D' in scharr_ly.name and float(scharr_ly.data.max()) > 0.0


def test_noise_fft_coloc_labels_geometry_undo_sync_and_batch_ui(tiki, tmp_path, accept_dialogs):
    from napari.layers import Image, Labels
    from qtpy.QtWidgets import QDoubleSpinBox, QLineEdit, QPlainTextEdit
    from tiki import axes, batch, channels, coloc, fft_noise, process

    # Verify new menu actions are registered cleanly
    for key in (
        'Edit>Undo',
        'Adjust>Sync Display Across Windows',
        'Noise>Despeckle',
        'Noise>Remove Outliers…',
        'Noise>Remove NaNs…',
        'Noise>Add Noise',
        'Noise>Add Specified Noise…',
        'Noise>Salt and Pepper',
        'FFT>FFT',
        'FFT>Inverse FFT',
        'FFT>Bandpass Filter…',
        'Labels>Distance Map',
        'Labels>Skeletonize',
        'Labels>Outline',
        'Labels>Voronoi…',
        'Batch>Convert…',
        'Batch>Run Macro…',
        'Analyze>Colocalization…',
        'Macros>Record…',
    ):
        assert key in tiki._actions

    # 1. Noise commands + Remove Outliers preview cleanup + FFT + Inverse FFT + Bandpass
    # Keep pixels finite in the Qt VisPy viewer (test_ops verifies NaN replacement directly)
    plane = np.full((20, 20), 100.0, dtype=np.float32)
    plane[10, 10] = 250.0
    info = axes.AxesInfo([axes.Axis('Y', 20, 0.5, 'µm'), axes.Axis('X', 20, 0.5, 'µm')], 'test', True)
    v = tiki.open_data(plane, 'noisy.tif', axes=info)

    rodlg = fft_noise.RemoveOutliersDialog(tiki, v, v.layers['noisy.tif'])
    assert 'filter preview' in v.layers and 'µm' in rodlg.phys_label.text()
    rodlg.reject()
    assert 'filter preview' not in v.layers

    accept_dialogs['configure'] = lambda d: None
    ro_ly = fft_noise.remove_outliers_dialog(tiki)
    assert float(ro_ly.data[10, 10]) == pytest.approx(100.0)
    nan_ly = fft_noise.remove_nans_dialog(tiki)
    assert np.all(np.isfinite(nan_ly.data))

    fft_noise.despeckle_command(tiki)
    fft_noise.add_noise_command(tiki)
    fft_noise.salt_and_pepper_command(tiki)

    v.layers.selection.active = nan_ly
    v_fft = fft_noise.fft_command(tiki)
    assert 'FFT of' in v_fft.title and 'fft_complex' in v_fft.layers[0].metadata
    v_inv = fft_noise.inverse_fft_command(tiki)
    assert v_inv.layers[0].data.shape == (20, 20)

    tiki.windows.activate(v)
    v.layers.selection.active = nan_ly
    bp_ly = fft_noise.bandpass_dialog(tiki)
    assert 'bandpass' in bp_ly.name and 'filter preview' not in v.layers

    # 2. Labels geometry: Distance Map, Skeletonize, Outline, Voronoi
    lab = np.zeros((16, 20), dtype=np.int32)
    lab[3:12, 2:8] = 1
    lab[3:12, 12:18] = 2
    v_lab = tiki.open_data(np.zeros((16, 20), np.float32), 'geom.tif', axes=info)
    v_lab.add_labels(lab, name='seg', scale=info.scale)

    edm_ly = process.distance_map_command(tiki)
    assert 'distance map' in edm_ly.name and float(edm_ly.data.max()) > 0.0
    v_lab.layers.selection.active = v_lab.layers['seg']
    sk_ly = process.skeletonize_command(tiki)
    assert set(np.unique(sk_ly.data)) == {0, 1, 2}
    v_lab.layers.selection.active = v_lab.layers['seg']
    ol_ly = process.outline_command(tiki)
    assert ol_ly.data[7, 5] == 0 and ol_ly.data[3, 5] == 1
    v_lab.layers.selection.active = v_lab.layers['seg']
    vor_ly = process.voronoi_dialog(tiki)
    assert set(np.unique(vor_ly.data)) == {1, 2}

    # 3. Edit ▸ Undo (Crop restores original array & axes; Labels painting undo; layer undo removes added layer)
    tiki.windows.activate(v_lab)
    assert sk_ly in v_lab.layers and edm_ly in v_lab.layers and vor_ly in v_lab.layers
    tiki.undo()  # undoes Voronoi added layer
    assert vor_ly not in v_lab.layers

    v_lab.layers.selection.active = v_lab.layers['geom.tif']
    _rect(tiki.roi_layer(v_lab), 2, 10, 2, 12)
    tiki.crop()
    assert v_lab.layers['geom.tif'].data.shape == (8, 10)
    tiki.undo()
    assert v_lab.layers['geom.tif'].data.shape == (16, 20)

    paint_ly = v_lab.layers['seg']
    paint_ly._save_history(((np.array([0]), np.array([0])), np.array([0]), np.array([9])))
    paint_ly.data[0, 0] = 9
    v_lab.layers.selection.active = paint_ly
    tiki.undo()
    assert int(paint_ly.data[0, 0]) == 0

    # 4. Sync Display Across Windows (via Tiki & ChannelsTool)
    hs = np.zeros((2, 10, 10), dtype=np.float32)
    hs[0] = np.arange(100, dtype=np.float32).reshape(10, 10)
    hs[1] = hs[0] * 2.0
    info_c = axes.AxesInfo([axes.Axis('C', 2), axes.Axis('Y', 10), axes.Axis('X', 10)], 'test', True)
    va = tiki.open_data(hs, 'syncA.tif', axes=info_c)
    tiki.split_channels()
    vb = tiki.open_data(hs.copy(), 'syncB.tif', axes=info_c)
    tiki.split_channels()

    va.layers[0].colormap = 'fire'
    va.layers[0].contrast_limits = (10.0, 80.0)
    va.layers[1].visible = False
    tiki.sync_display(va)
    assert vb.layers[0].colormap.name == 'fire'
    assert tuple(map(float, vb.layers[0].contrast_limits)) == pytest.approx((10.0, 80.0))
    assert not vb.layers[1].visible

    tiki.channels_tool()
    assert 'Sync to All Windows' in [
        w.text() for w in tiki.channels_window.findChildren(type(tiki.channels_window.grid.itemAtPosition(3, 0).widget()))
    ]

    # 5. Colocalization dialog + ColocalizationWindow + Add to Results
    tiki.windows.activate(va)
    tiki.results.clear()
    accept_dialogs['configure'] = lambda dlg: None
    cwin = coloc.colocalization_dialog(tiki)
    assert cwin.stats['Pearson_r'] > 0.99
    cwin.add_to_results()
    assert len(tiki.results.rows) == 1 and tiki.results.rows[0]['Pearson_r'] > 0.99
    cwin.close()

    # 6. Batch dialogs (Convert… & Run Macro…)
    in_b = tmp_path / 'ui_batch_in'
    out_b = tmp_path / 'ui_batch_out'
    in_b.mkdir()
    import tifffile

    tifffile.imwrite(in_b / 'sample.tif', np.full((8, 8), 12, dtype=np.uint8))

    def cfg_bconv(dlg):
        edits = dlg.findChildren(QLineEdit)
        edits[0].setText(str(in_b))
        edits[1].setText(str(out_b))

    accept_dialogs['configure'] = cfg_bconv
    n_conv, errs_b = batch.batch_convert_dialog(tiki)
    assert n_conv == 1 and errs_b == [] and (out_b / 'sample.tif').is_file()

    def cfg_bmac(dlg):
        edits = dlg.findChildren(QLineEdit)
        edits[0].setText(str(in_b))
        dlg.findChild(QPlainTextEdit).setPlainText("tiki.run('Measure')")

    tiki.results.clear()
    accept_dialogs['configure'] = cfg_bmac
    n_bm, errs_bm = batch.batch_macro_dialog(tiki)
    assert n_bm == 1 and errs_bm == [] and len(tiki.results.rows) == 1


def test_review_bugfixes_export_integrity_collisions_rgb_crop_and_units(tiki, tmp_path):
    import tifffile

    from tiki import axes as axes_mod
    from tiki import export, filters

    # 1. Mixed-unit 3D z_ratio (500 nm Z vs 0.5 µm XY -> 1.0, not 0.001)
    vol = np.zeros((6, 12, 12), dtype=np.float32)
    info = axes_mod.AxesInfo(
        [axes_mod.Axis('Z', 6, 500.0, 'nm'), axes_mod.Axis('Y', 12, 0.5, 'µm'), axes_mod.Axis('X', 12, 0.5, 'µm')],
        'test',
        True,
    )
    v_unit = tiki.open_data(vol, 'units.tif', axes=info)
    assert filters.z_ratio(v_unit.layers[0]) == pytest.approx(1.0)

    # 2. RGB crop slices spatial Y/X, not the 3-channel color axis
    rgb = np.arange(20 * 30 * 3, dtype=np.uint8).reshape(20, 30, 3)
    v_rgb = tiki.open_data(rgb, 'rgb.png', rgb=True)
    _rect(v_rgb.layers[roi.ROI_NAME], 1.5, 7.5, 4.5, 11.5)
    tiki.crop()
    assert v_rgb.layers[0].data.shape == (6, 7, 3)
    np.testing.assert_array_equal(v_rgb.layers[0].data, rgb[2:8, 5:12, :])

    # 3. Round-trip notebook export: split channel + crop + colliding label names + apostrophe path
    apostrophe_dir = tmp_path / "o'brien's lab"
    apostrophe_dir.mkdir()
    src_tif = apostrophe_dir / "two'ch.tif"
    raw = np.arange(2 * 12 * 16, dtype=np.uint16).reshape(2, 12, 16)
    tifffile.imwrite(src_tif, raw, imagej=True, metadata={'axes': 'CYX'})

    tiki.open_paths([str(src_tif)])
    v_src = tiki.windows.active
    tiki.split_channels()
    ch1 = v_src.layers[0]
    v_src.layers.selection.active = ch1
    _rect(v_src.layers[roi.ROI_NAME], 1.5, 7.5, 2.5, 10.5)
    tiki.crop()
    expected = np.asarray(ch1.data).copy()
    assert expected.shape == (6, 8)

    # Colliding label names: 'a b' and 'a_b'
    lbl1 = np.full((6, 8), 1, dtype=np.int32)
    lbl2 = np.full((6, 8), 2, dtype=np.int32)
    v_src.add_labels(lbl1, name='a b')
    v_src.add_labels(lbl2, name='a_b')

    out_dir = apostrophe_dir / "session'1"
    folder, snippet = export.export_session(out_dir, v_src, ch1)
    ns: dict = {}
    exec(compile(snippet, str(folder / 'load_session.py'), 'exec'), ns)
    assert ns['image'].shape == (6, 8)
    np.testing.assert_array_equal(ns['image'], expected)
    assert int(ns['labels']['a b'][0, 0]) == 1
    assert int(ns['labels']['a_b'][0, 0]) == 2


def test_lazy_dask_undo_skips_oversized_and_preserves_laziness(tiki):
    import dask
    import dask.array as da

    from tiki import axes as axes_mod

    # 1. > 256 MB Dask array: chunks outside ROI raise if materialized during push_undo_data or crop()
    guard = {'strict': False}

    def _safe_half_chunk(z_idx, x_half):
        if guard['strict'] and x_half > 0:
            raise AssertionError('oversized Dask array materialized outside ROI during push_undo_data/crop')
        return np.zeros((1, 512, 256), dtype=np.float32)

    huge = da.concatenate(
        [
            da.concatenate(
                [
                    da.from_delayed(dask.delayed(_safe_half_chunk)(k, 0), shape=(1, 512, 256), dtype=np.float32),
                    da.from_delayed(dask.delayed(_safe_half_chunk)(k, 1), shape=(1, 512, 256), dtype=np.float32),
                ],
                axis=2,
            )
            for k in range(300)
        ],
        axis=0,
    )  # 300 * 512 * 512 * 4 ~= 300 MB > 256 MB
    info_huge = axes_mod.AxesInfo(
        [axes_mod.Axis('Z', 300), axes_mod.Axis('Y', 512), axes_mod.Axis('X', 512)],
        'test',
        True,
    )
    v_huge = tiki.open_data(huge, 'huge.tif', axes=info_huge, contrast_limits=(0.0, 1.0))
    guard['strict'] = True
    assert tiki.push_undo_data('Crop', v_huge, v_huge.layers[0]) is False
    _rect(v_huge.layers[roi.ROI_NAME], 0, 16, 0, 16)
    tiki.crop()
    assert v_huge.layers['huge.tif'].data.shape == (300, 16, 16)
    tiki.windows.close(v_huge)

    # 2. <= 256 MB Dask array: Crop snapshots lazily without computing full array, and Undo restores lazy reference
    calls = {'count': 0}

    def _chunk(z_val):
        calls['count'] += 1
        return np.full((1, 16, 16), float(z_val + 1), dtype=np.float32)

    small_dask = da.concatenate(
        [da.from_delayed(dask.delayed(_chunk)(k), shape=(1, 16, 16), dtype=np.float32) for k in range(4)],
        axis=0,
    )
    info_small = axes_mod.AxesInfo(
        [axes_mod.Axis('Z', 4), axes_mod.Axis('Y', 16), axes_mod.Axis('X', 16)],
        'test',
        True,
    )
    v_small = tiki.open_data(small_dask, 'lazy.tif', axes=info_small, contrast_limits=(0.0, 5.0))
    calls_after_open = calls['count']
    _rect(v_small.layers[roi.ROI_NAME], 2, 10, 2, 10)
    tiki.crop()
    assert v_small.layers['lazy.tif'].data.shape == (4, 8, 8)
    assert v_small.layers['lazy.tif'].metadata.get('modified') is True
    # Only the crop itself materialized chunks (1 pass), not an extra full copy in push_undo_data
    assert calls['count'] - calls_after_open <= 4

    tiki.undo()
    ly = v_small.layers['lazy.tif']
    assert ly.data is small_dask and hasattr(ly.data, 'compute')
    assert 'modified' not in ly.metadata


def test_macro_recorder_replay_reproduces_pixels_and_metadata(tiki, accept_dialogs):
    from napari.layers import Image, Labels
    from qtpy.QtWidgets import QDoubleSpinBox, QPlainTextEdit
    from tiki import axes as axes_mod
    from tiki import fft_noise, macros, process

    rng = np.random.default_rng(3)
    vol = rng.uniform(5.0, 15.0, (4, 18, 18)).astype(np.float32)
    vol[:, 5:13, 5:13] = 120.0
    vol[1, 9, 9] = 900.0
    info = axes_mod.AxesInfo(
        [axes_mod.Axis('Z', 4, 1.0, 'µm'), axes_mod.Axis('Y', 18, 0.5, 'µm'), axes_mod.Axis('X', 18, 0.5, 'µm')],
        'test',
        True,
    )

    rec = tiki.show_macro_recorder()
    v1 = tiki.open_data(vol.copy(), 'sample.tif', axes=info)
    v1.dims.set_current_step(0, 1)

    # 1. Convolve (3D stack plane-by-plane)
    accept_dialogs['configure'] = lambda d: d.findChild(QPlainTextEdit).setPlainText('1 1 1\n1 1 1\n1 1 1')
    conv_1 = process.convolve_dialog(tiki)

    # 2. Edge Detection: Canny as Labels across stack with explicit thresholds
    v1.layers.selection.active = v1.layers['sample.tif']

    def cfg_canny(dlg):
        dlg.method_box.setCurrentIndex(dlg.method_box.findData('canny'))
        dlg.sigma_spin.setValue(1.2)
        dlg.low_spin.setValue(0.05)
        dlg.high_spin.setValue(0.30)
        dlg.whole_stack.setChecked(True)
        dlg.as_labels.setChecked(True)

    accept_dialogs['configure'] = cfg_canny
    canny_1 = process.edge_detection_dialog(tiki)

    # 3. Edge Detection: 3D Scharr across Z-stack
    v1.layers.selection.active = v1.layers['sample.tif']

    def cfg_scharr3d(dlg):
        dlg.method_box.setCurrentIndex(dlg.method_box.findData('scharr'))
        dlg.three_d.setChecked(True)
        dlg.whole_stack.setChecked(True)

    accept_dialogs['configure'] = cfg_scharr3d
    scharr_1 = process.edge_detection_dialog(tiki)

    # 4. Math (single current plane only -> whole_stack=False)
    v1.layers.selection.active = v1.layers['sample.tif']

    def cfg_math(dlg):
        _combo(dlg, 'Add').setCurrentText('Multiply')
        dlg.findChild(QDoubleSpinBox).setValue(2.5)
        from qtpy.QtWidgets import QCheckBox
        dlg.findChild(QCheckBox).setChecked(False)

    accept_dialogs['configure'] = cfg_math
    math_1 = process.math_dialog(tiki)

    # 5. Find Maxima (Segmented Particles, plane)
    v1.layers.selection.active = v1.layers['sample.tif']

    def cfg_fm(dlg):
        dlg.prom_spin.setValue(15.0)
        dlg.output_box.setCurrentText('Segmented Particles')
        dlg.whole_stack.setChecked(False)

    accept_dialogs['configure'] = cfg_fm
    fm_1 = process.find_maxima_dialog(tiki)

    # 6. Remove Outliers + Bandpass Filter (stripe suppression)
    v1.layers.selection.active = v1.layers['sample.tif']
    accept_dialogs['configure'] = lambda d: None
    ro_1 = fft_noise.remove_outliers_dialog(tiki)

    v1.layers.selection.active = v1.layers['sample.tif']

    def cfg_bp(dlg):
        dlg.stripes_box.setCurrentText('Horizontal')
        dlg.whole_stack.setChecked(True)

    accept_dialogs['configure'] = cfg_bp
    bp_1 = fft_noise.bandpass_dialog(tiki)

    # 7. Resize (updates physical Y/X calibration)
    v1.layers.selection.active = v1.layers['sample.tif']

    def cfg_resize(dlg):
        spins = dlg.findChildren(QDoubleSpinBox)
        spins[0].setValue(2.0)

    accept_dialogs['configure'] = cfg_resize
    v_res_1 = process.resize_dialog(tiki)
    res_1_data = np.asarray(v_res_1.layers[0].data).copy()
    res_1_axes = axes_mod.get(v_res_1.layers[0]).to_dict()

    recorded = rec.code()
    exp_conv = (np.asarray(conv_1.data).copy(), axes_mod.get(conv_1).to_dict())
    exp_canny = (np.asarray(canny_1.data).copy(), isinstance(canny_1, Labels), axes_mod.get(canny_1).to_dict())
    exp_scharr = (np.asarray(scharr_1.data).copy(), axes_mod.get(scharr_1).to_dict())
    exp_math = (np.asarray(math_1.data).copy(), axes_mod.get(math_1).to_dict())
    exp_fm = (np.asarray(fm_1.data).copy(), axes_mod.get(fm_1).to_dict())
    exp_ro = (np.asarray(ro_1.data).copy(), axes_mod.get(ro_1).to_dict())
    exp_bp = (np.asarray(bp_1.data).copy(), axes_mod.get(bp_1).to_dict())

    # Replay recorded code on a fresh identical stack (keeping active source layer selected cleanly)
    tiki.windows.close_all()
    v2 = tiki.open_data(vol.copy(), 'sample.tif', axes=info)
    v2.dims.set_current_step(0, 1)

    lines_rebuilt = []
    for ln in recorded.splitlines():
        s = ln.strip()
        if not s or s.startswith('#'):
            continue
        lines_rebuilt.append("tiki.viewer().layers.selection.active = tiki.layer('sample.tif')")
        lines_rebuilt.append(ln)
    ok, log_txt, _ = macros.run_code(tiki, '\n'.join(lines_rebuilt))
    assert ok, log_txt
    # Running a macro must NOT re-record lines into the open Recorder window
    assert rec.code() == recorded
    rec.close()

    # Compare reproduced pixels, layer types, and calibrated axes metadata
    by_name = {ly.name: ly for ly in v2.layers}
    np.testing.assert_allclose(by_name[conv_1.name].data, exp_conv[0])
    assert axes_mod.get(by_name[conv_1.name]).to_dict()['axes'] == exp_conv[1]['axes']

    assert isinstance(by_name[canny_1.name], Labels) is exp_canny[1]
    np.testing.assert_array_equal(by_name[canny_1.name].data, exp_canny[0])

    np.testing.assert_allclose(by_name[scharr_1.name].data, exp_scharr[0])
    assert axes_mod.get(by_name[scharr_1.name]).to_dict()['axes'] == exp_scharr[1]['axes']

    np.testing.assert_allclose(by_name[math_1.name].data, exp_math[0])
    assert by_name[math_1.name].data.ndim == 2

    np.testing.assert_array_equal(by_name[fm_1.name].data, exp_fm[0])
    np.testing.assert_allclose(by_name[ro_1.name].data, exp_ro[0])
    np.testing.assert_allclose(by_name[bp_1.name].data, exp_bp[0])

    v_res_2 = tiki.windows.viewers[-1]
    assert v_res_2.title.startswith('Resize_')
    np.testing.assert_allclose(v_res_2.layers[0].data, res_1_data)
    assert axes_mod.get(v_res_2.layers[0]).to_dict()['axes'] == res_1_axes['axes']


def test_notebook_export_container_h5_and_zarr(tiki, tmp_path, monkeypatch):
    import sys
    import types

    import dask.array as da

    from tiki import export, io

    # 1. Multi-dataset HDF5 container exports selected dataset key and closes file handle
    closed_handles = []

    class FakeDataset:
        def __init__(self, arr, attrs=None):
            self._arr = np.asarray(arr)
            self.shape = self._arr.shape
            self.ndim = self._arr.ndim
            self.dtype = self._arr.dtype
            self.attrs = attrs or {}

        def __getitem__(self, item):
            return self._arr[item]

        def __array__(self, dtype=None, copy=None):
            return self._arr if dtype is None else self._arr.astype(dtype)

    h5_arr = np.arange(3 * 6 * 8, dtype=np.uint16).reshape(3, 6, 8) + 5

    class FakeH5File:
        def __init__(self, _p, _m='r'):
            self.closed = False
            self._items = {
                'small': FakeDataset(np.zeros((2, 2), np.float32)),
                'sub/stack': FakeDataset(h5_arr, {'axes': 'ZYX'}),
            }

        def visititems(self, fn):
            for k, v in self._items.items():
                fn(k, v)

        def __getitem__(self, k):
            return self._items[k]

        def close(self):
            self.closed = True

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()
            closed_handles.append(self)
            return False

    monkeypatch.setitem(sys.modules, 'h5py', types.SimpleNamespace(File=FakeH5File, Dataset=FakeDataset))
    h5_path = tmp_path / 'sample.h5'
    h5_path.write_bytes(b'dummy')
    data_h5, meta_h5 = io.read_hdf5(h5_path)
    assert meta_h5['container_key'] == 'sub/stack'
    v_h5 = tiki.open_scene_entry(data_h5, meta_h5)

    out_h5 = tmp_path / 'nb_h5'
    folder_h5, snippet_h5 = export.export_session(out_h5, v_h5, v_h5.layers[0])
    assert '_h5[\'sub/stack\']' in snippet_h5
    ns_h5: dict = {}
    exec(compile(snippet_h5, str(folder_h5 / 'load_session.py'), 'exec'), ns_h5)
    assert isinstance(ns_h5['image'], np.ndarray)
    np.testing.assert_array_equal(ns_h5['image'], h5_arr)
    assert closed_handles and closed_handles[-1].closed

    # Missing container_key on HDF5 falls back cleanly to image.npy instead of broken h5py.File
    v_h5_nokey = tiki.open_data(h5_arr, 'nokey.h5', metadata={'path': str(h5_path)})
    folder_fb, snippet_fb = export.export_session(tmp_path / 'nb_fb', v_h5_nokey, v_h5_nokey.layers[0])
    assert (folder_fb / 'image.npy').is_file() and 'image.npy' in snippet_fb

    # 2. Nested plain Zarr & OME-Zarr export selected dataset key, not root group
    zarr_arr = np.arange(2 * 6 * 6, dtype=np.float32).reshape(2, 6, 6) + 10.0

    class FakeZarrGroup:
        def __init__(self, children, attrs=None):
            self._children = children
            self.attrs = attrs or {}

        def keys(self):
            return list(self._children.keys())

        def __getitem__(self, k):
            cur = self
            for part in str(k).split('/'):
                cur = cur._children[part]
            return cur

    fake_zarr = FakeZarrGroup({'sub': FakeZarrGroup({'stack': zarr_arr})})
    monkeypatch.setitem(sys.modules, 'zarr', types.SimpleNamespace(open=lambda *_a, **_kw: fake_zarr))
    monkeypatch.setattr(da, 'from_zarr', lambda arr: da.from_array(arr, chunks=(1, 6, 6)))
    z_path = tmp_path / 'nested.zarr'
    z_path.mkdir()
    data_z, meta_z = io.read_zarr(z_path)
    assert meta_z['container_key'] == 'sub/stack'
    v_z = tiki.open_scene_entry(data_z, meta_z)

    folder_z, snippet_z = export.export_session(tmp_path / 'nb_zarr', v_z, v_z.layers[0])
    assert "['sub/stack']" in snippet_z
    ns_z: dict = {}
    exec(compile(snippet_z, str(folder_z / 'load_session.py'), 'exec'), ns_z)
    np.testing.assert_array_equal(np.asarray(ns_z['image']), zarr_arr)




