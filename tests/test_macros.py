import numpy as np
import pytest
from qtpy.QtCore import Qt

from tiki import macros, roi
from tiki.app import Tiki


@pytest.fixture
def tiki(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr(macros, 'MACROS_DIR', tmp_path / 'macros')
    t = Tiki()
    qtbot.addWidget(t)
    t.windows.show_layer_controls = lambda v, visible=True: None
    t.show()
    yield t
    t.windows.close_all()


def _movie(tiki, data, roles='TYX'):
    from tiki import axes

    info = axes.AxesInfo([axes.Axis(r, n) for r, n in zip(roles, data.shape)], 'test', True)
    return tiki.open_data(data, 'movie.tif', axes=info)


def test_run_code_output_and_errors(tiki):
    _movie(tiki, np.ones((5, 8, 8), np.float32))
    ok, out, ns = macros.run_code(tiki, "print(tiki.image().shape, tiki.axes())\nx = 3")
    assert ok and '(5, 8, 8) TYX' in out and ns['x'] == 3
    ok, out, _ = macros.run_code(tiki, "a = 1\nb = undefined_name + 1", '<macro>')
    assert not ok and 'NameError' in out and 'line 2' in out and 'site-packages' not in out
    ok, out, _ = macros.run_code(tiki, "def broken(:\n  pass")
    assert not ok and 'SyntaxError' in out


def test_api_show_labels_rois_traces_measure(tiki):
    movie = np.zeros((10, 16, 16), np.float32)
    movie[:, 4:8, 4:8] = np.arange(10)[:, None, None]
    v = _movie(tiki, movie)
    code = '''
labels = np.zeros((16, 16), int); labels[4:8, 4:8] = 1
tiki.add_labels(labels, 'my labels')
tiki.rois_from_labels(labels)
tr = tiki.traces()
tiki.measure(ROI=tiki.rois()[0].name, Max=float(tr.max()))
w = tiki.show(tiki.image()[:, ::2, ::2], 'binned', axes='TYX', scale=[0.1, 2, 2])
'''
    ok, out, ns = macros.run_code(tiki, code)
    assert ok, out
    assert 'my labels' in v.layers
    np.testing.assert_allclose(ns['tr'][0], np.arange(10))
    assert tiki.results.rows[-1]['Max'] == 9
    assert ns['w'].layers[0].data.shape == (10, 8, 8)
    assert tuple(ns['w'].layers[0].scale) == pytest.approx((0.1, 2, 2))


def test_api_run_menu_command(tiki):
    v = _movie(tiki, np.full((3, 8, 8), 5, np.float32))
    tiki.set_tool('rectangle')
    v.layers[roi.ROI_NAME].add_rectangles(np.array([[[1.5, 1.5], [1.5, 5.5], [5.5, 5.5], [5.5, 1.5]]]))
    ok, out, _ = macros.run_code(tiki, "tiki.run('Measure')")
    assert ok, out
    assert tiki.results.rows[-1]['Area'] == 16
    ok, out, _ = macros.run_code(tiki, "tiki.run('No Such Command')")
    assert not ok and 'no TIKI command' in out


def test_motion_correction_template_recovers_shifts(tiki):
    from scipy.ndimage import gaussian_filter
    from scipy.ndimage import shift as nd_shift

    rng = np.random.default_rng(0)
    field = gaussian_filter(rng.normal(0, 1, (120, 120)).astype(np.float32), 2) * 50 + 100
    true = rng.uniform(-3, 3, (30, 2))
    # frames are windows onto a moving field (no artificial borders) plus noise
    movie = np.stack([nd_shift(field, s, order=3)[30:90, 30:90] for s in true]).astype(np.float32)
    movie += rng.normal(0, 2, movie.shape).astype(np.float32)
    _movie(tiki, movie)
    ok, out, ns = macros.run_code(tiki, macros.TEMPLATES['Motion correction (rigid)'])
    assert ok, out
    est = ns['shifts']
    error = np.abs((est - est.mean(0)) + (true - true.mean(0))).mean()
    assert error < 0.2  # sub-pixel recovery of the applied motion
    inner = (slice(None), slice(6, -6), slice(6, -6))
    assert ns['out'][inner].std(axis=0).mean() < 0.4 * movie[inner].std(axis=0).mean()
    assert 'MC_movie.tif' in [w.title for w in tiki.windows.viewers]


def test_caiman_template_explains_missing_package(tiki):
    _movie(tiki, np.zeros((5, 8, 8), np.float32))
    try:
        import caiman  # noqa: F401
        pytest.skip('CaImAn installed; template would actually run')
    except ImportError:
        pass
    ok, out, _ = macros.run_code(tiki, macros.TEMPLATES['CaImAn CNMF → ROIs'])
    assert not ok and 'CaImAn is not installed' in out


def test_macros_menu_lists_and_runs_files(tiki):
    macros.MACROS_DIR.mkdir(parents=True)
    (macros.MACROS_DIR / 'say_hello.py').write_text("print('hello from a macro')\ntiki.status('hi')")
    (macros.MACROS_DIR / '_private.py').write_text("raise SystemExit")
    tiki._fill_macros_menu()
    names = [a.text() for a in tiki.macros_menu.actions()]
    assert 'say hello' in names and 'private' not in ' '.join(names)
    next(a for a in tiki.macros_menu.actions() if a.text() == 'say hello').trigger()
    assert 'hello from a macro' in tiki.log_window.text.toPlainText()


def test_editor_keys_run_and_save_not_tiki(tiki, qtbot, tmp_path, monkeypatch):
    _movie(tiki, np.zeros((2, 4, 4), np.float32))
    saved_tiff = []
    monkeypatch.setattr(tiki, 'save_as', lambda: saved_tiff.append(True))
    ed = tiki.new_macro()
    ed.editor.setPlainText("result = tiki.image().sum() + 7\nprint('ran', result)")
    ed.path = tmp_path / 'm.py'
    ed.activateWindow()
    ed.editor.setFocus()
    qtbot.keyClick(ed.editor, Qt.Key_R, Qt.ControlModifier)
    assert 'ran 7.0' in ed.output.toPlainText()
    qtbot.keyClick(ed.editor, Qt.Key_S, Qt.ControlModifier)
    assert (tmp_path / 'm.py').read_text().startswith('result =') and not saved_tiff


def test_api_select_layers_and_windows_by_name_or_index(tiki):
    from tiki import axes

    data = np.zeros((2, 3, 8, 8), np.float32)
    data[:, 1] = 7  # channel 2
    info = axes.AxesInfo([axes.Axis('T', 2), axes.Axis('C', 3), axes.Axis('Y', 8), axes.Axis('X', 8)], 't', True)
    v = tiki.open_data(data, 'two.tif', axes=info)
    tiki.split_channels()
    other = tiki.open_data(np.ones((4, 4), np.float32), 'other.tif')
    tiki.windows.activate(v)
    code = '''
names = tiki.layers()
wins = tiki.windows()
c2 = tiki.image('C2-two.tif')
top = tiki.layer(-1).name
bottom = tiki.image(0)
far = tiki.image(window='other.tif')
far_by_index = tiki.image(0, window=1)
tiki.add_labels((c2[0] > 1).astype(int), 'c2 mask', like='C2-two.tif')
lab = tiki.labels('c2 mask')
'''
    ok, out, ns = macros.run_code(tiki, code)
    assert ok, out
    assert ns['names'] == ['C1-two.tif', 'C2-two.tif', 'C3-two.tif']
    assert ns['wins'] == ['two.tif', 'other.tif']
    assert ns['c2'].max() == 7 and ns['bottom'].max() == 0
    assert ns['top'] == 'C3-two.tif'
    assert ns['far'].shape == (4, 4) and ns['far_by_index'].shape == (4, 4)
    assert ns['lab'].sum() == 64
    ok, out, _ = macros.run_code(tiki, "tiki.image('nope')")
    assert not ok and "no image layer named 'nope'" in out and 'C1-two.tif' in out
    ok, out, _ = macros.run_code(tiki, "tiki.image(window='missing')")
    assert not ok and 'open windows' in out


def test_macro_api_helpers_recorder_and_batch(tiki, tmp_path):
    import tifffile
    from tiki import batch

    # 1. MacroAPI programmatic helpers: set_scale, set_lut, filter, z_project, threshold, analyze_particles
    mov = np.zeros((4, 16, 16), dtype=np.float32)
    mov[:, 4:12, 4:12] = 100.0
    _movie(tiki, mov, roles='ZYX')
    code = """
tiki.set_scale(0.5, 0.5, unit='µm')
tiki.set_lut('Fire')
tiki.filter('gaussian', whole_stack=True, three_d=False, sigma=1.0)
tiki.threshold(40.0, 1000.0, scope='whole', split=True, connect_z=True)
n = tiki.analyze_particles(min_size=5, scope='whole', three_d=True)
p_view = tiki.z_project('max')
"""
    ok, out, ns = macros.run_code(tiki, code)
    assert ok, out
    assert ns['n'] == 1
    assert tiki.results.column_units.get('Volume') == 'µm³'
    assert ns['p_view'].title.startswith('MAX_')

    # 2. MacroRecorderWindow records actions, supports Pause/Resume and Create Macro -> Editor
    rec = tiki.show_macro_recorder()
    tiki.windows.activate(tiki.windows.viewers[0])
    tiki.invert()
    assert "tiki.run('Invert')" in rec.code()

    rec.pause_btn.setChecked(True)
    tiki.flip('horizontal')
    assert "Flip Horizontally" not in rec.code()
    rec.pause_btn.setChecked(False)

    ed = rec.create_macro()
    assert "tiki.run('Invert')" in ed.editor.toPlainText()
    ed.close()
    rec.close()

    # 3. Batch Convert (TIFF -> PNG & NPY) and Batch Macro across folder
    in_dir = tmp_path / 'batch_in'
    in_dir.mkdir()
    for idx in range(2):
        arr = np.full((3, 12, 12), 50 + idx * 40, dtype=np.uint16)
        arr[:, 3:9, 3:9] += 100
        tifffile.imwrite(in_dir / f'img_{idx + 1}.tif', arr, imagej=True, metadata={'axes': 'ZYX'})

    out_png = tmp_path / 'out_png'
    n_conv, errs = batch.batch_convert(in_dir, out_png, fmt='png')
    assert n_conv == 2 and errs == []
    assert (out_png / 'img_1.png').is_file() and (out_png / 'img_2.png').is_file()

    out_npy = tmp_path / 'out_npy'
    n_npy, errs2 = batch.batch_convert(in_dir, out_npy, fmt='npy')
    assert n_npy == 2 and errs2 == []
    assert np.load(out_npy / 'img_1.npy').shape == (3, 12, 12)

    tiki.results.clear()
    out_batch = tmp_path / 'out_batch'
    before_wins = len(tiki.windows.viewers)
    n_ok, b_errs = batch.batch_macro(
        tiki,
        in_dir,
        "tiki.z_project('max')\ntiki.run('Measure')",
        output_dir=out_batch,
        auto_save=True,
        save_results_csv=True,
        close_after=True,
    )
    assert n_ok == 2 and b_errs == []
    assert len(tiki.results.rows) == 2
    assert (out_batch / 'Results.csv').is_file()
    assert (out_batch / 'img_1.tif').is_file()
    assert len(tiki.windows.viewers) == before_wins

