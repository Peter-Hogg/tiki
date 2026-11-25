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

    auto_buttons = ct.findChildren(QPushButton)
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
