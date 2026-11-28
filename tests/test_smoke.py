"""Smoke tests and regression tests for stability across TIKI modules."""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

import numpy as np
import pytest
import tifffile
from napari.utils.colormaps import Colormap
from qtpy.QtWidgets import QDialog, QFileDialog, QMessageBox, QPushButton

import tiki as tiki_pkg
from tiki import axes as axes_mod
from tiki import batch, instance, io, macros, nd, ops, process, publish, recent, roi
from tiki import rois as rois_mod
from tiki.app import Tiki, TikiError
from tiki.filters import FILTERS, FilterDialog


@pytest.fixture
def tiki(qtbot):
    t = Tiki()
    t.windows.shown_controls = []
    t.windows.show_layer_controls = lambda v, visible=True: t.windows.shown_controls.append(v)
    t.set_tool('rectangle')
    qtbot.addWidget(t)
    t.show()
    yield t
    t.windows.close_all()


# ----------------------------------------------------------------- imports / CLI


def test_smoke_import_all_modules():
    """Every module in the tiki package imports without side-effect crashes."""
    names = [m.name for m in pkgutil.iter_modules(tiki_pkg.__path__)]
    assert len(names) >= 20
    for name in names:
        mod = importlib.import_module(f'tiki.{name}')
        assert mod is not None


def test_smoke_cli_version_and_help():
    from tiki.__main__ import main

    with pytest.raises(SystemExit) as exc:
        main(['--version'])
    assert exc.value.code == 0

    with pytest.raises(SystemExit) as exc:
        main(['--help'])
    assert exc.value.code == 0


# ----------------------------------------------------------------- resilience


def test_recent_handles_corrupt_json(tmp_path, monkeypatch):
    f = tmp_path / 'recent.json'
    monkeypatch.setattr(recent, 'RECENT_FILE', f)
    for bad in ('', 'null', '123', '{"a": 1}', '{bad'):
        f.write_text(bad)
        assert recent.load() == []


def test_axes_memory_handles_corrupt_json(tmp_path, monkeypatch):
    mem = tmp_path / 'axes_memory.json'
    monkeypatch.setattr(axes_mod, 'MEMORY_FILE', mem)
    info = axes_mod.AxesInfo([axes_mod.Axis('T', 4), axes_mod.Axis('Y', 8), axes_mod.Axis('X', 8)], 'test', True)
    for bad in ('', 'null', '[]', '{not json'):
        mem.write_text(bad)
        assert axes_mod.recall(tmp_path / 'a.tif', (4, 8, 8)) is None
        axes_mod.remember(tmp_path / 'a.tif', (4, 8, 8), info)
        got = axes_mod.recall(tmp_path / 'a.tif', (4, 8, 8))
        assert got is not None and got.roles == 'TYX'


def test_from_tiff_zero_fps_and_negative_spacing(tmp_path):
    p = tmp_path / 'neg_z.tif'
    arr = np.zeros((3, 8, 8), dtype=np.uint8)
    tifffile.imwrite(p, arr, imagej=True, metadata={'axes': 'ZYX', 'spacing': -2.5, 'fps': 0, 'finterval': 0})
    with tifffile.TiffFile(p) as tf:
        info = axes_mod.from_tiff(tf, arr.shape)
    z_ax = info.axes[info.index('Z')]
    assert z_ax.step == pytest.approx(2.5)


def test_instance_handle_message_malformed():
    class DummyTiki:
        opened = []

        def open_dropped(self, files):
            self.opened.extend(files)

        def isMinimized(self):
            return False

        def show(self):
            pass

        def raise_(self):
            pass

        def activateWindow(self):
            pass

    d = DummyTiki()
    for raw in (b'', b'not json', b'[]', b'123', b'"hi"', b'{"files": 42}', b'\xff\xfe'):
        instance._handle_message(d, raw)
    assert d.opened == []


def test_shape_box_off_image_and_empty():
    # Entirely above/left of image: must not return negative y1/x1
    y0, y1, x0, x1 = roi.shape_box(np.array([[-20, -15], [-5, -2]], float), (32, 32))
    assert (y0, y1, x0, x1) == (0, 0, 0, 0)
    # Entirely below/right of image
    y0, y1, x0, x1 = roi.shape_box(np.array([[40, 40], [50, 50]], float), (32, 32))
    assert y0 == y1 == 32 and x0 == x1 == 32
    # Empty vertices
    assert roi.shape_box(np.empty((0, 2)), (32, 32)) == (0, 0, 0, 0)


def test_rois_empty_default_name_and_safe_save(tmp_path):
    empty = rois_mod.Roi('empty', 'polygon', np.empty((0, 2)))
    assert empty.default_name(1) == '0001-0000-0000'

    existing = tmp_path / 'keep.zip'
    existing.write_bytes(b'original-data')
    bad_roi = rois_mod.Roi('bad', 'unknown_kind', np.array([[1.0, 2.0]]))
    with pytest.raises(KeyError):
        rois_mod.save_imagej(existing, [bad_roi])
    assert existing.read_bytes() == b'original-data'


def test_nd_for_layer_higher_ndim():
    block = (slice(None), slice(None))
    got = nd.for_layer(block, 3)
    assert got == (slice(None), slice(None), slice(None))


# ----------------------------------------------------------------- ops edge cases


def test_ops_edge_cases():
    # 3D particle_table with default 2-tuple scale
    lab3 = np.zeros((4, 16, 16), dtype=np.int32)
    lab3[1:3, 4:8, 4:8] = 1
    img3 = np.ones((4, 16, 16), dtype=np.float32) * 10.0
    rows = ops.particle_table(lab3, img3)
    assert len(rows) == 1 and rows[0]['Voxels'] == 32

    # auto_threshold with NaN and empty
    arr_nan = np.array([0.0, 0.0, 10.0, 10.0, np.nan, np.inf])
    t = ops.auto_threshold(arr_nan, 'Otsu')
    assert np.isfinite(t) and 0.0 <= t <= 10.0
    with pytest.raises(ValueError):
        ops.auto_threshold(np.array([np.nan, np.inf]), 'Otsu')

    # label_traces: negative labels ignored, sparse huge IDs don't OOM, negative axis works
    movie = np.arange(5 * 6 * 6, dtype=np.float32).reshape(5, 6, 6)
    lab = np.zeros((6, 6), dtype=np.int32)
    lab[0, 0] = -1
    lab[1, 1] = 2
    lab[2, 2] = 2_000_005
    ids, tr = ops.label_traces(movie, axis=-3, index=(0, 0, 0), labels=lab)
    assert list(ids) == [2, 2_000_005]
    np.testing.assert_allclose(tr[0], movie[:, 1, 1])
    np.testing.assert_allclose(tr[1], movie[:, 2, 2])

    # roi_traces with negative axis
    mask = np.zeros((6, 6), dtype=bool)
    mask[1, 1] = True
    rtr = ops.roi_traces(movie, axis=-3, index=(0, 0, 0), masks=[mask])
    np.testing.assert_allclose(rtr[0], movie[:, 1, 1])

    # kymograph with negative axis and off-image line
    k = ops.kymograph(movie, axis=-3, index=(0, 0, 0), verts=np.array([[-10.0, -10.0], [-5.0, -5.0]]))
    assert k.shape[0] == 5 and np.all(np.isfinite(k))

    # bleach_correct rejects non-positive means[0] in ratio mode
    with pytest.raises(ValueError):
        ops.bleach_correct(np.zeros((5, 8, 8), dtype=np.float32), axis=0, method='ratio', background=0.0)

    # subtract_background validates radius > 0 and tolerates NaN with light_background
    with pytest.raises(ValueError):
        ops.subtract_background(np.ones((16, 16), dtype=np.float32), radius=0)
    nan_plane = np.full((16, 16), 50.0, dtype=np.float32)
    nan_plane[0, 0] = np.nan
    sub = ops.subtract_background(nan_plane, radius=3, light_background=True)
    assert np.isfinite(sub[8, 8])

    # invert_image on uint64 high values
    u64 = np.array([[0, 2**63 + 5, 2**64 - 1]], dtype=np.uint64)
    inv = ops.invert_image(u64)
    assert inv.dtype == np.uint64
    assert int(inv[0, 0]) == 2**64 - 1
    assert int(inv[0, 2]) == 0

    # mask_stats on RGB plane reports pixel area, not 3x
    rgb = np.full((10, 10, 3), 100, dtype=np.uint8)
    m = np.zeros((10, 10), dtype=bool)
    m[2:4, 2:5] = True  # 6 pixels
    st = ops.mask_stats(rgb, m)
    assert st['Area'] == pytest.approx(6.0)

    # colocalization_stats with costes=False and manual thresholds
    a = np.linspace(1, 100, 64).reshape(8, 8)
    b = a * 2.0 + 3.0
    col = ops.colocalization_stats(a, b, costes=False)
    assert col['Slope'] == pytest.approx(2.0, rel=1e-5)
    assert col['Intercept'] == pytest.approx(3.0, rel=1e-4)


# ----------------------------------------------------------------- io / publish


def test_save_tiff_edge_cases(tmp_path):
    # 2D RGB with 2-tuple scale and axes=None
    rgb2d = np.zeros((12, 16, 3), dtype=np.uint8)
    p1 = tmp_path / 'rgb2d.tif'
    io.save_tiff(p1, rgb2d, scale=(0.5, 0.5))
    assert p1.exists()

    # Zero scale does not raise ZeroDivisionError
    p2 = tmp_path / 'zero_scale.tif'
    io.save_tiff(p2, np.zeros((10, 10), dtype=np.uint8), scale=(0.0, 0.0))
    assert p2.exists()

    # Multiple unknown axes in OME-TIFF (no duplicate 'Z')
    arr4 = np.zeros((2, 3, 8, 8), dtype=np.int16)
    info4 = axes_mod.AxesInfo(
        [axes_mod.Axis('?', 2), axes_mod.Axis('?', 3), axes_mod.Axis('Y', 8), axes_mod.Axis('X', 8)],
        'test',
        False,
    )
    p3 = tmp_path / 'multi_unknown.ome.tif'
    io.save_tiff(p3, arr4, axes=info4)
    assert p3.exists()

    # uint16 RGB and uint8 RGBA fall back to OME-TIFF without crashing ImageJ mode
    info2 = axes_mod.AxesInfo([axes_mod.Axis('Y', 8), axes_mod.Axis('X', 8)], 'test', True)
    p4 = tmp_path / 'rgb16.tif'
    io.save_tiff(p4, np.zeros((8, 8, 3), dtype=np.uint16), axes=info2)
    assert p4.exists()
    p5 = tmp_path / 'rgba8.tif'
    io.save_tiff(p5, np.zeros((8, 8, 4), dtype=np.uint8), axes=info2)
    assert p5.exists()

    # Non-RGB stack with unknown axis and size 3 stays non-RGB (minisblack)
    arr3 = np.zeros((3, 8, 8), dtype=np.uint8)
    info3 = axes_mod.AxesInfo([axes_mod.Axis('?', 3), axes_mod.Axis('Y', 8), axes_mod.Axis('X', 8)], 'test', False)
    p6 = tmp_path / 'plain3.tif'
    io.save_tiff(p6, arr3, axes=info3)
    (got, _), = io.read_multi(p6)
    assert got.shape == (3, 8, 8)


def test_publish_roi_manager_overlay_and_nan_rgb(tiki, tmp_path):
    # NaN float RGB compositing
    rgb_nan = np.full((16, 16, 3), 0.5, dtype=np.float32)
    rgb_nan[0, 0, 0] = np.nan
    v = tiki.open_data(rgb_nan, 'nan_rgb.tif', rgb=True)
    rendered = publish.render_plane_rgb(v)
    assert rendered.dtype == np.uint8 and rendered.shape == (16, 16, 3)

    # ROI Manager overlay in save_snapshot (previously AttributeError: r.vertices)
    mgr = tiki.show_roi_manager()
    mgr.rois.append(rois_mod.Roi('cell1', 'rectangle', np.array([[2, 2], [2, 8], [8, 8], [8, 2]], float)))
    out_png = tmp_path / 'snap.png'
    publish.save_snapshot(v, out_png, show_rois=True, roi_manager=mgr, show_roi_labels=True)
    assert out_png.exists() and out_png.stat().st_size > 0

    # OrthogonalViewsWindow._to_display normalizes uint16 RGB
    rgb16_stack = np.full((4, 16, 16, 3), 32768, dtype=np.uint16)
    info = axes_mod.AxesInfo([axes_mod.Axis('Z', 4), axes_mod.Axis('Y', 16), axes_mod.Axis('X', 16)], 'test', True)
    v2 = tiki.open_data(rgb16_stack, 'rgb16z.tif', axes=info, rgb=True)
    ortho = publish.OrthogonalViewsWindow(v2, v2.layers[0], z_axis=0)
    disp = ortho._to_display(rgb16_stack[0])
    assert disp.dtype == np.uint8 and int(disp.max()) <= 130 and int(disp.min()) >= 125
    ortho.close()


# ----------------------------------------------------------------- UI / signals / lifecycle


def test_ui_button_and_menu_signal_fixes(tiki, monkeypatch):
    # 1. ResultsTable Summarize button (Qt clicked(bool=False) must not suppress show=True)
    tiki.results.add_row({'Label': 'a', 'Mean': 10.0})
    tiki.results.add_row({'Label': 'b', 'Mean': 20.0})
    sum_btn = [b for b in tiki.results.findChildren(QPushButton) if b.text() == 'Summarize'][0]
    sum_btn.click()
    assert tiki.results._summary_win is not None
    assert tiki.results._summary_win.isVisible()

    # 2. TracesWindow Save .npy button (Qt clicked(bool=False) must not pass False as path)
    from tiki.widgets import TracesWindow

    tw = TracesWindow('Traces', ['r1', 'r2'], np.ones((2, 10), dtype=np.float32))
    called = []
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *a, **k: (called.append(True) or '', ''))
    npy_btn = [b for b in tw.findChildren(QPushButton) if 'npy' in b.text()][0]
    npy_btn.click()
    assert called == [True]
    tw.close()

    # 3. MacroEditor Templates menu (Qt triggered(bool=False) must not overwrite code)
    ed = tiki.new_macro()
    templates_menu = [a.menu() for a in ed.menuBar().actions() if 'Templates' in a.text()][0]
    first_action = templates_menu.actions()[0]
    first_action.trigger()
    assert len(ed.editor.toPlainText()) > 10
    ed.close()


def test_macro_system_exit_and_editor_save_cancel_and_recorder_close(tiki, monkeypatch):
    # SystemExit in macro does not terminate process
    ok, out, _ns = macros.run_code(tiki, 'import sys\nsys.exit(2)')
    assert not ok and 'SystemExit' in out
    ok0, _out0, _ns0 = macros.run_code(tiki, 'import sys\nsys.exit(0)')
    assert ok0

    # MacroEditor closeEvent keeps editor open if user clicks Save then cancels Save As
    ed = tiki.new_macro()
    ed.editor.setPlainText('x = 1\n')
    ed.editor.document().setModified(True)
    monkeypatch.setattr(QMessageBox, 'question', lambda *a, **k: QMessageBox.Save)
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *a, **k: ('', ''))
    assert ed.close() is False
    ed.editor.document().setModified(False)
    assert ed.close() is True

    # Closing MacroRecorderWindow pauses recording
    rec = tiki.show_macro_recorder()
    assert not rec.paused
    rec.close()
    assert rec.paused


def test_contrast_custom_lut_and_threshold_edge_cases(tiki):
    # Custom ImageJ LUT switching away and back does not raise KeyError
    custom = Colormap([[0, 0, 0, 1], [0.2, 0.8, 0.4, 1]], name='imagej_custom_smoke')
    v = tiki.open_data(np.arange(64, dtype=np.uint8).reshape(8, 8), 'lut.tif', colormap=custom)
    tiki.brightness_contrast()
    bc = tiki.contrast_window
    bc.lut_box.setCurrentText('Grays')
    bc.lut_box.setCurrentText('imagej_custom_smoke')
    assert v.layers[0].colormap.name == 'imagej_custom_smoke'
    bc.close()

    # ThresholdWindow: large spinbox value doesn't OverflowError, all-NaN image doesn't AttributeError
    v_f = tiki.open_data(np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8), 'f.tif')
    tiki.threshold()
    tw = tiki.threshold_window
    assert tw._to_slider(1e10) == 1000
    assert tw._to_slider(-1e10) == 0

    v_nan = tiki.open_data(np.full((8, 8), np.nan, dtype=np.float32), 'all_nan.tif')
    tw.set_target(v_nan)
    tw.reset()
    tw.close()


def test_filter_rgb_zstack_crop_zproject_cursor_undo_channels_roimanager(tiki):
    # FilterDialog on RGB Z-stack disables 3D and previews cleanly
    rgb_z = np.zeros((4, 16, 16, 3), dtype=np.uint8)
    info = axes_mod.AxesInfo([axes_mod.Axis('Z', 4), axes_mod.Axis('Y', 16), axes_mod.Axis('X', 16)], 'test', True)
    v = tiki.open_data(rgb_z, 'rgbz.tif', axes=info, rgb=True)
    dlg = FilterDialog(tiki, 'gaussian', v, v.layers[0])
    assert not dlg.has_z
    dlg.reject()

    # z_project rejects start >= stop and preserves rgb=True
    with pytest.raises(TikiError):
        tiki.z_project(v.layers[0], 'max', axis=0, start=3, stop=2)
    v_proj = tiki.z_project(v.layers[0], 'max', axis=0, start=0, stop=2)
    assert v_proj.layers[0].rgb is True
    tiki.windows.close(v_proj)

    # crop rejects off-image ROI
    rlayer = v.layers[roi.ROI_NAME]
    rlayer.add_rectangles(np.array([[[-20, -20], [-20, -5], [-5, -5], [-5, -20]]], float))
    with pytest.raises(TikiError):
        tiki.crop()

    # _on_cursor ignores NaN coordinates
    tiki._message_until = 0.0
    tiki._on_cursor(v, (np.nan, np.nan, np.nan))

    # Undo stack prunes entries when viewer closes
    v_u = tiki.open_data(np.zeros((12, 12), dtype=np.uint8), 'undo_prune.tif')
    assert tiki.push_undo_data('test', v_u, v_u.layers[0])
    assert any(e['viewer'] is v_u for e in tiki._undo_stack)
    tiki.windows.close(v_u)
    assert not any(e['viewer'] is v_u for e in tiki._undo_stack)

    # ChannelsTool refreshes when a layer is removed
    tiki.channels_tool()
    ct = tiki.channels_window
    v.add_image(np.zeros((4, 16, 16), dtype=np.uint8), name='extra')
    v.layers.remove('extra')
    ct.close()

    # RoiManager clamps out-of-range roi.position
    mgr = tiki.show_roi_manager()
    mgr.rois.clear()
    mgr.list.clear()
    mgr.rois.append(rois_mod.Roi('deep', 'rectangle', np.array([[1, 1], [1, 5], [5, 5], [5, 1]], float), position=(999,)))
    mgr.list.addItem('deep')
    mgr.list.setCurrentRow(0)
    assert v.dims.current_step[0] == 3

    # RoiManager navigates hyperstack_pos by axis role name (T=7 on TZCYX -> T=6, Z=0, C=0)
    from tiki import axes as axes_mod

    tzc = np.zeros((10, 5, 3, 16, 16), dtype=np.uint8)
    tzc_axes = axes_mod.AxesInfo(
        [axes_mod.Axis('T', 10), axes_mod.Axis('Z', 5), axes_mod.Axis('C', 3), axes_mod.Axis('Y', 16), axes_mod.Axis('X', 16)],
        'test',
        True,
    )
    v_tzc = tiki.open_data(tzc, 'tzc.tif', axes=tzc_axes)
    v_tzc.dims.current_step = (0, 0, 0, 0, 0)
    mgr.rois.clear()
    mgr.list.clear()
    t_roi = rois_mod.Roi(
        't7',
        'rectangle',
        np.array([[1, 1], [1, 5], [5, 5], [5, 1]], float),
        position=(6,),
        meta={'hyperstack_pos': {'T': 6}},
    )
    mgr.rois.append(t_roi)
    mgr.list.addItem('t7')
    mgr.list.setCurrentRow(0)
    assert v_tzc.dims.current_step[:3] == (6, 0, 0)


def test_batch_convert_skip_multiscene(tmp_path):
    in_dir = tmp_path / 'in'
    out_dir = tmp_path / 'out'
    in_dir.mkdir()
    out_dir.mkdir()
    # Create a 2-series OME-TIFF
    src = in_dir / 'sample.ome.tif'
    with tifffile.TiffWriter(src, ome=True) as tw:
        tw.write(np.ones((8, 8), dtype=np.uint8), metadata={'axes': 'YX'})
        tw.write(np.full((8, 8), 2, dtype=np.uint8), metadata={'axes': 'YX'})
    # Pre-create sample.ome.tif in out_dir so planned_dst is None under skip
    (out_dir / 'sample.ome.tif').write_bytes(b'existing')
    ok_n, errs = batch.batch_convert(in_dir, out_dir, fmt='tiff', on_conflict='skip')
    assert ok_n == 1 and not errs
    assert (out_dir / 'sample.ome_s1.tif').exists()
    assert (out_dir / 'sample.ome_s2.tif').exists()


def test_run_task_background_thread_and_errors(tiki):
    import threading
    from qtpy.QtCore import QThread
    from qtpy.QtWidgets import QApplication

    main_tid = threading.get_ident()
    seen_threads = []

    def work(tick):
        seen_threads.append(threading.get_ident())
        assert QThread.currentThread() is not QApplication.instance().thread()
        for i in range(1, 6):
            assert tick(i, 5) is True
        return 42

    res, cancelled = tiki.run_task('Test Background', 5, work)
    assert res == 42
    assert not cancelled
    assert seen_threads and seen_threads[0] != main_tid

    # Exceptions raised on worker thread propagate cleanly to main thread
    def fail(_tick):
        raise ValueError('worker boom')

    with pytest.raises(ValueError, match='worker boom'):
        tiki.run_task('Test Error', 1, fail)


def test_napari_plugins_menu_and_ai_seg_bridge(tiki, monkeypatch):
    from tiki import ai_seg

    # Napari Plugins menu populates cleanly and includes Install / Manage Plugins…
    tiki._fill_napari_plugins_menu()
    labels = [a.text() for a in tiki.napari_plugins_menu.actions() if not a.isSeparator()]
    assert any('Manage Plugins' in t for t in labels)

    # AI segmentation backend detection reports valid mode and handles missing gracefully
    mode, detail = ai_seg.detect_backend('Cellpose')
    assert mode in ('local', 'uv', 'missing')
    assert detail

    # Block-wise run_segmentation preserves globally unique label IDs across slices
    monkeypatch.setattr(ai_seg, 'detect_backend', lambda _name: ('local', 'mocked'))
    monkeypatch.setattr(
        ai_seg,
        '_run_cellpose_local',
        lambda arr, _params: np.where(arr > 0, 1, 0).astype(np.int32),
    )
    stack = np.zeros((3, 8, 8), dtype=np.uint16)
    stack[0, 1:3, 1:3] = 10
    stack[1, 2:4, 2:4] = 10
    stack[2, 4:6, 4:6] = 10
    blocks = [(i, slice(None), slice(None)) for i in range(3)]
    out = ai_seg.run_segmentation('Cellpose', stack, {'model': 'cyto3'}, blocks=blocks)
    assert out.shape == stack.shape
    assert set(np.unique(out)) == {0, 1, 2, 3}


def test_sam_window_ui_and_commit(tiki):
    from tiki import sam as sam_mod

    img = np.zeros((32, 32), dtype=np.uint16)
    img[8:20, 8:20] = 500
    rgb = sam_mod._to_rgb_uint8(img)
    assert rgb.shape == (32, 32, 3) and rgb.dtype == np.uint8

    v = tiki.open_data(img, 'sam_test.tif')
    win = tiki.show_sam_window()
    assert win.image is not None
    # Simulate a preview mask returned by the sidecar and commit it as object #1
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:20, 8:20] = 1
    win._show_preview_mask(mask)
    win._accept_object()
    assert 'sam_test.tif sam' in v.layers
    assert int(np.asarray(v.layers['sam_test.tif sam'].data).max()) == 1

    # Scrolling to a different slice clears preview and never commits into the wrong frame
    stack = np.zeros((4, 32, 32), dtype=np.uint16)
    v3 = tiki.open_data(stack, 'sam_stack.tif')
    win.attach(v3)
    v3.dims.current_step = (0, 0, 0)
    win._show_preview_mask(mask)
    v3.dims.current_step = (2, 0, 0)
    assert win._last_mask is None
    win._accept_object()
    assert 'sam_stack.tif sam' not in v3.layers
    win.close()



