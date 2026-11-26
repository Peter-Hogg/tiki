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


def test_batch_convert_collisions_and_source_protection(tmp_path):
    from PIL import Image as PilImage
    import tifffile

    from tiki import batch

    # 1. Intra-batch stem collision: sample.tif + sample.png -> TIFF
    col_dir = tmp_path / 'col_in'
    col_dir.mkdir()
    tifffile.imwrite(col_dir / 'sample.tif', np.full((8, 8), 11, dtype=np.uint8))
    PilImage.fromarray(np.full((8, 8), 22, dtype=np.uint8)).save(col_dir / 'sample.png')

    out_err = tmp_path / 'out_err'
    with pytest.raises(ValueError, match='collision'):
        batch.batch_convert(col_dir, out_err, fmt='tiff', on_conflict='error')
    assert not out_err.exists()

    out_ren = tmp_path / 'out_ren'
    n_ok, errs = batch.batch_convert(col_dir, out_ren, fmt='tiff', on_conflict='rename')
    assert n_ok == 2 and errs == []
    assert float(tifffile.imread(out_ren / 'sample_tif.tif').mean()) == pytest.approx(11.0)
    assert float(tifffile.imread(out_ren / 'sample_png.tif').mean()) == pytest.approx(22.0)

    # 2. Same input & output directory never overwrites source files
    same_dir = tmp_path / 'same'
    same_dir.mkdir()
    src_tif = same_dir / 'orig.tif'
    tifffile.imwrite(src_tif, np.full((8, 8), 15, dtype=np.uint8))

    with pytest.raises(ValueError, match='source file'):
        batch.batch_convert(same_dir, same_dir, fmt='tiff', on_conflict='overwrite')
    assert float(tifffile.imread(src_tif).mean()) == pytest.approx(15.0)

    n_skip, e_skip = batch.batch_convert(same_dir, same_dir, fmt='tiff', on_conflict='skip')
    assert n_skip == 0 and e_skip == [] and len(list(same_dir.glob('*.tif'))) == 1

    n_in_ren, e_in_ren = batch.batch_convert(same_dir, same_dir, fmt='tiff', on_conflict='rename')
    assert n_in_ren == 1 and e_in_ren == []
    assert (same_dir / 'orig_out.tif').is_file()
    assert float(tifffile.imread(src_tif).mean()) == pytest.approx(15.0)

    # 3. Existing non-source output conflict policies
    png_only = tmp_path / 'png_only'
    png_only.mkdir()
    PilImage.fromarray(np.full((8, 8), 33, dtype=np.uint8)).save(png_only / 'item.png')

    out_ex = tmp_path / 'out_ex'
    out_ex.mkdir()
    sentinel = out_ex / 'item.tif'
    tifffile.imwrite(sentinel, np.full((8, 8), 99, dtype=np.uint8))

    with pytest.raises(ValueError, match='already exists'):
        batch.batch_convert(png_only, out_ex, fmt='tiff', on_conflict='error')
    assert float(tifffile.imread(sentinel).mean()) == pytest.approx(99.0)

    n_s, _ = batch.batch_convert(png_only, out_ex, fmt='tiff', on_conflict='skip')
    assert n_s == 0 and float(tifffile.imread(sentinel).mean()) == pytest.approx(99.0)

    n_r, _ = batch.batch_convert(png_only, out_ex, fmt='tiff', on_conflict='rename')
    assert n_r == 1 and float(tifffile.imread(out_ex / 'item_2.tif').mean()) == pytest.approx(33.0)
    assert float(tifffile.imread(sentinel).mean()) == pytest.approx(99.0)

    n_o, _ = batch.batch_convert(png_only, out_ex, fmt='tiff', on_conflict='overwrite')
    assert n_o == 1 and float(tifffile.imread(sentinel).mean()) == pytest.approx(33.0)


def test_macro_failures_and_batch_error_propagation(tiki, tmp_path):
    import tifffile

    from tiki import batch

    sentinel_win = tiki.open_data(np.full((6, 6), 42, dtype=np.uint8), 'existing.tif')
    ok, out, _ = macros.run_code(tiki, "tiki.run('Crop')")
    assert not ok and 'area selection' in out

    ok_open, out_open, _ = macros.run_code(tiki, f"tiki.open({str(tmp_path / 'missing.tif')!r})")
    assert not ok_open and out_open.strip()

    in_dir = tmp_path / 'mixed_in'
    in_dir.mkdir()
    tifffile.imwrite(in_dir / 'good.tif', np.full((6, 6), 10, dtype=np.uint8))
    (in_dir / 'bad.tif').write_bytes(b'not a valid tiff file')

    out_dir = tmp_path / 'mixed_out'
    n_ok, errs = batch.batch_macro(
        tiki,
        in_dir,
        "tiki.run('Invert')",
        output_dir=out_dir,
        auto_save=True,
        close_after=True,
    )
    assert n_ok == 1 and len(errs) == 1 and errs[0][0].name == 'bad.tif'
    assert (out_dir / 'good.tif').is_file()
    assert not (out_dir / 'bad.tif').exists()
    assert int(sentinel_win.layers[0].data[0, 0]) == 42

    # Failing command inside macro fails batch item and does NOT auto-save unchanged image
    only_good = tmp_path / 'only_good'
    only_good.mkdir()
    tifffile.imwrite(only_good / 'sample.tif', np.full((6, 6), 10, dtype=np.uint8))
    out_fail = tmp_path / 'out_fail'
    n_ok2, errs2 = batch.batch_macro(
        tiki,
        only_good,
        "tiki.run('Crop')",
        output_dir=out_fail,
        auto_save=True,
        close_after=True,
    )
    assert n_ok2 == 0 and len(errs2) == 1 and 'area selection' in errs2[0][1]
    assert not (out_fail / 'sample.tif').exists()


def test_batch_macro_multiscene_policies(tiki, tmp_path):
    import tifffile

    from tiki import batch

    in_dir = tmp_path / 'ome_in'
    in_dir.mkdir()
    ome_path = in_dir / 'three_scenes.ome.tif'
    with tifffile.TiffWriter(ome_path, ome=True) as tif:
        for val in (10, 20, 30):
            tif.write(np.full((6, 6), val, dtype=np.uint16), photometric='minisblack', metadata={'axes': 'YX'})

    # 1. scenes='all' processes every scene cleanly and writes 3 outputs
    tiki.results.clear()
    out_all = tmp_path / 'out_all'
    before = len(tiki.windows.viewers)
    n_all, e_all = batch.batch_macro(
        tiki,
        in_dir,
        "tiki.measure(Scene=scene_index, Mean=float(tiki.image().mean()))",
        output_dir=out_all,
        auto_save=True,
        scenes='all',
        close_after=True,
    )
    assert n_all == 1 and e_all == []
    assert [r['Mean'] for r in tiki.results.rows] == pytest.approx([10.0, 20.0, 30.0])
    for s in (1, 2, 3):
        assert (out_all / f'three_scenes.ome_s{s}.tif').is_file()
    assert len(tiki.windows.viewers) == before

    # 2. scenes='first' processes only scene 0
    tiki.results.clear()
    out_first = tmp_path / 'out_first'
    n_first, e_first = batch.batch_macro(
        tiki,
        in_dir,
        "tiki.measure(Scene=scene_index, Mean=float(tiki.image().mean()))",
        output_dir=out_first,
        auto_save=True,
        scenes='first',
        close_after=True,
    )
    assert n_first == 1 and e_first == []
    assert [r['Mean'] for r in tiki.results.rows] == pytest.approx([10.0])
    assert (out_first / 'three_scenes.ome.tif').is_file()
    assert len(tiki.windows.viewers) == before

    # 3. scenes='active' opens all scenes and runs once on active window, then closes all
    tiki.results.clear()
    n_act, e_act = batch.batch_macro(
        tiki,
        in_dir,
        "tiki.measure(Windows=len(tiki.windows()), Mean=float(tiki.image().mean()))",
        scenes='active',
        close_after=True,
    )
    assert n_act == 1 and e_act == []
    assert len(tiki.results.rows) == 1
    assert tiki.results.rows[0]['Windows'] == 3
    assert tiki.results.rows[0]['Mean'] == pytest.approx(30.0)
    assert len(tiki.windows.viewers) == before


