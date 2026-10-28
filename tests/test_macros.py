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
