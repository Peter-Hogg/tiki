import numpy as np
import pytest

from tiki import ops, roi


def test_z_project_matches_numpy_and_chunks():
    rng = np.random.default_rng(0)
    data = rng.integers(0, 1000, (20, 37, 41)).astype(np.uint16)
    old = ops.CHUNK_BYTES
    ops.CHUNK_BYTES = 20 * 41 * 2 * 3  # force several chunks
    try:
        for method, ref in [('max', data.max(0)), ('min', data.min(0)), ('mean', data.mean(0)),
                            ('sum', data.sum(0)), ('std', data.std(0)), ('median', np.median(data, 0))]:
            out = ops.z_project(data, method)
            np.testing.assert_allclose(out, ref, rtol=1e-5)
    finally:
        ops.CHUNK_BYTES = old
    assert ops.z_project(data, 'max').dtype == np.uint16
    assert ops.z_project(data, 'mean').dtype == np.float32


def test_z_project_rejects_spatial_axis():
    with pytest.raises(ValueError):
        ops.z_project(np.zeros((3, 4, 5)), 'max', axis=1)


def test_axis_profile_uses_mask_and_fixed_index():
    data = np.zeros((2, 5, 6, 6))
    data[1, :, 2:4, 2:4] = np.arange(5)[:, None, None]
    mask = np.zeros((6, 6), bool)
    mask[2:4, 2:4] = True
    prof = ops.axis_profile(data, axis=1, index=(1, 0, 0, 0), mask=mask, chunk=2)
    np.testing.assert_allclose(prof, np.arange(5))


def test_mask_stats():
    plane = np.arange(16, dtype=float).reshape(4, 4)
    mask = np.zeros((4, 4), bool)
    mask[1:3, 1:3] = True
    s = ops.mask_stats(plane, mask, pixel_area=0.25)
    assert s['Area'] == 1.0
    assert s['Mean'] == pytest.approx(7.5)
    assert s['RawIntDen'] == pytest.approx(30)


def test_rectangle_and_ellipse_masks():
    rect = np.array([[2, 2], [2, 6], [6, 6], [6, 2]], float)
    m = roi.shape_mask(rect, 'rectangle', (10, 10))
    assert m[3:6, 3:6].all() and not m[0].any()
    e = roi.shape_mask(np.array([[0, 0], [0, 20], [20, 20], [20, 0]], float), 'ellipse', (21, 21))
    assert e[10, 10] and not e[1, 1]
    assert abs(e.sum() - np.pi * 100) / (np.pi * 100) < 0.05
    line = roi.shape_mask(np.array([[0, 0], [5, 5]], float), 'line', (8, 8))
    assert line.sum() == 6


def test_auto_threshold_methods_separate_two_populations():
    rng = np.random.default_rng(1)
    values = np.concatenate([rng.normal(100, 5, 5000), rng.normal(300, 5, 1000)])
    for method in ops.THRESHOLD_METHODS:
        t = ops.auto_threshold(values, method)
        assert values.min() <= t <= values.max(), (method, t)
        if method in ('Default', 'IsoData', 'Otsu', 'Li', 'Minimum'):
            assert 112 < t < 290, (method, t)


def test_threshold_stack_labels_unique_and_min_size():
    data = np.zeros((2, 10, 10), np.uint16)
    data[0, 1:4, 1:4] = 500   # 9 px object
    data[0, 7, 7] = 500       # 1 px speck
    data[1, 5:9, 5:9] = 500   # 16 px object
    labels = ops.threshold_stack(data, 200, 1000, split=True, min_size=2)
    assert labels.dtype == np.int32
    assert set(np.unique(labels[0])) == {0, 1} and set(np.unique(labels[1])) == {0, 2}
    mask = ops.threshold_stack(data, 200, 1000, split=False)
    assert mask.dtype == np.uint8 and mask.sum() == 9 + 1 + 16


def test_particle_table_matches_simple_shapes():
    labels = np.zeros((20, 20), np.int32)
    labels[2:6, 2:8] = 1      # 4x6 rectangle
    labels[10:15, 10:15] = 2  # 5x5 square
    img = np.full((20, 20), 10.0)
    img[labels == 2] = 50
    rows = ops.particle_table(labels, img, scale=(0.5, 0.5))
    assert [r['Label'] for r in rows] == [1, 2]
    assert rows[0]['Area'] == 24 * 0.25 and rows[1]['Mean'] == 50
    assert rows[1]['X'] == pytest.approx(12 * 0.5) and rows[1]['Y'] == pytest.approx(12 * 0.5)
    assert rows[1]['AR'] == pytest.approx(1.0)


def test_roi_and_label_traces_agree():
    rng = np.random.default_rng(3)
    data = rng.normal(100, 5, (2, 30, 16, 16))
    labels = np.zeros((16, 16), np.int32)
    labels[1:5, 1:5] = 3
    labels[8:14, 9:15] = 7
    ids, lt = ops.label_traces(data, axis=1, index=(1, 0, 0, 0), labels=labels, chunk=7)
    rt = ops.roi_traces(data, axis=1, index=(1, 0, 0, 0), masks=[labels == 3, labels == 7], chunk=7)
    assert list(ids) == [3, 7]
    np.testing.assert_allclose(lt, rt)
    np.testing.assert_allclose(rt[1], data[1, :, 8:14, 9:15].mean(axis=(1, 2)))


def test_filters_shapes_and_background():
    yy, xx = np.mgrid[:64, :64]
    ramp = (yy * 2).astype(np.uint16) + 100
    spot = ramp.copy()
    spot[30:34, 30:34] += 500
    flat = ops.subtract_background(spot, radius=10)
    assert flat.dtype == np.uint16
    assert flat[5:20, 5:20].max() < 20          # ramp removed
    assert flat[31, 31] > 400                    # small object kept
    gentle = (yy * 0.2).astype(np.uint16) + 100
    gentle[30:34, 30:34] += 500
    big = ops.subtract_background(gentle, radius=40)  # shrunken (ImageJ-style) path
    assert big[31, 31] > 400 and np.median(big) < 10
    light = ops.subtract_background(65000 - spot, radius=10, light_background=True)
    assert light[31, 31] < light[5, 5]
    assert ops.gaussian_blur(spot, 2).dtype == np.float32
    assert ops.median_filter(spot, 2).shape == spot.shape
    stack = ops.map_planes(lambda p: ops.median_filter(p, 1), np.stack([spot, spot]))
    assert stack.shape == (2, 64, 64)


def _spheres(shape=(20, 40, 40), centres=((6, 10, 10), (13, 28, 28)), r=4):
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    vol = np.full(shape, 100, np.uint16)
    for cz, cy, cx in centres:
        vol[(zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = 1000
    return vol


def test_segment_connects_through_z_only_in_3d():
    vol = _spheres()
    labels3d = ops.segment(vol, 500, 2000, split=True)
    assert labels3d.max() == 2  # two spheres = two 3D objects
    planes = [(k, slice(None), slice(None)) for k in range(vol.shape[0])]
    per_plane = ops.threshold_blocks(vol, 500, 2000, planes, split=True)
    assert per_plane.max() > 2  # every slice of every sphere is its own object


def test_particle_table_3d_volume_uses_voxel_size():
    vol = np.zeros((10, 20, 20), np.int32)
    vol[2:6, 5:10, 5:15] = 1  # 4 x 5 x 10 voxels
    rows = ops.particle_table(vol, np.ones_like(vol, float), scale=(2.0, 0.5, 0.5))
    assert len(rows) == 1
    assert rows[0]['Voxels'] == 200
    assert rows[0]['Volume'] == pytest.approx(200 * 2.0 * 0.25)
    assert rows[0]['Z'] == pytest.approx(3.5 * 2.0)


def test_nd_blocks_scopes():
    from tiki import axes, nd

    info = axes.AxesInfo([axes.Axis('T', 3), axes.Axis('Z', 4), axes.Axis('Y', 8), axes.Axis('X', 8)])
    shape, idx = (3, 4, 8, 8), (1, 2, 0, 0)
    assert [k for k, _ in nd.scopes(info)] == ['plane', 'volume', 'whole']
    assert nd.blocks(info, shape, idx, 'plane', True) == [(1, 2, slice(None), slice(None))]
    assert nd.blocks(info, shape, idx, 'volume', True) == [(1, slice(None), slice(None), slice(None))]
    assert len(nd.blocks(info, shape, idx, 'whole', True)) == 3    # one per t
    assert len(nd.blocks(info, shape, idx, 'whole', False)) == 12  # one per plane
    zyx = axes.AxesInfo([axes.Axis('Z', 4), axes.Axis('Y', 8), axes.Axis('X', 8)])
    assert [k for k, _ in nd.scopes(zyx)] == ['plane', 'whole']
    assert nd.for_layer((1, slice(None), slice(None), slice(None)), 3) == (slice(None),) * 3


def test_gaussian_3d_respects_anisotropy():
    vol = np.zeros((21, 21, 21), np.float32)
    vol[10, 10, 10] = 1
    iso = ops.gaussian_3d(vol, 2.0, 1.0)
    aniso = ops.gaussian_3d(vol, 2.0, 0.25)  # slices 4x further apart than pixels
    assert iso[12, 10, 10] == pytest.approx(iso[10, 12, 10], rel=1e-4)
    assert aniso[11, 10, 10] < aniso[10, 11, 10]  # blurred less across slices


def test_dff_movie_methods_and_chunking():
    rng = np.random.default_rng(0)
    t = np.arange(100)
    base = 200.0
    movie = np.full((100, 6, 7), base, np.float32) + rng.normal(0, 1, (100, 6, 7)).astype(np.float32)
    movie[40:45, 2, 3] += 100  # a transient
    old = ops.CHUNK_BYTES
    ops.CHUNK_BYTES = 100 * 7 * 4 * 2  # 2 rows per chunk
    try:
        d = ops.dff_movie(movie, axis=0, method='median')
    finally:
        ops.CHUNK_BYTES = old
    assert d.dtype == np.float32 and d.shape == movie.shape
    assert d[42, 2, 3] == pytest.approx(0.5, abs=0.03)
    assert abs(np.median(d[:, 0, 0])) < 1e-6
    d_off, f0 = ops.dff_movie(movie, axis=0, method='first frames', n_first=10, offset=100, return_f0=True)
    assert f0.shape == (1, 6, 7) and f0[0, 0, 0] == pytest.approx(100, abs=1)
    assert d_off[42, 2, 3] == pytest.approx(1.0, abs=0.05)  # offset doubles ΔF/F
    slide = ops.dff_movie(movie + (t * 0.5)[:, None, None].astype(np.float32), axis=0,
                          method='sliding percentile', window=21, percentile=50)
    assert abs(np.nanmedian(slide[:, 0, 0])) < 0.01  # drift removed by the sliding baseline
    zero = np.zeros((5, 4, 4), np.float32)
    assert np.isnan(ops.dff_movie(zero, axis=0)).all()


def test_bleach_correct_ratio_and_exponential():
    n = 60
    t = np.arange(n)
    decay = 100 + 300 * np.exp(-t / 15)
    movie = np.ones((n, 8, 8), np.float32) * decay[:, None, None].astype(np.float32)
    corr, info = ops.bleach_correct(movie, axis=0, method='ratio')
    assert np.allclose(corr.mean(axis=(1, 2)), corr[0].mean(), rtol=1e-4)
    corr2, info2 = ops.bleach_correct(movie, axis=0, method='exponential')
    assert info2['tau_frames'] == pytest.approx(15, rel=0.02)
    assert np.allclose(corr2.mean(axis=(1, 2)), corr2[0].mean(), rtol=1e-3)


def test_kymograph_tracks_moving_spot():
    n, h, w = 20, 16, 40
    movie = np.zeros((n, h, w), np.float32)
    for k in range(n):
        movie[k, 7:10, 5 + k] = 100  # spot moves 1 px per frame along x
    kymo = ops.kymograph(movie, axis=0, index=(0, 0, 0), verts=np.array([[8, 2], [8, 35]]), width=3, chunk=6)
    assert kymo.shape == (20, 34)
    peaks = kymo.argmax(axis=1)
    assert np.all(np.diff(peaks) == 1)  # diagonal line = constant velocity
    assert ops.line_samples(np.array([[0, 0], [0, 10], [10, 10]])).shape == (1, 21, 2)


def test_image_calculator_broadcast_and_divide():
    a = np.full((3, 4, 5), 10, np.uint16)
    b = np.full((4, 5), 4, np.uint16)
    b[0, 0] = 0
    sub = ops.image_calculator(a, b, 'Subtract')
    assert sub.dtype == np.float32 and sub[2, 1, 1] == 6  # no uint wrap-around
    div = ops.image_calculator(a, b, 'Divide')
    assert div[1, 1, 1] == 2.5 and np.isnan(div[0, 0, 0])
    with pytest.raises(ValueError):
        ops.image_calculator(a, np.zeros((3, 3)), 'Add')


def test_histogram_stats():
    s = ops.histogram_stats(np.array([1, 2, 2, 3, np.nan]), bins=3)
    assert s['n'] == 4 and s['mean'] == 2 and s['min'] == 1 and s['max'] == 3
    assert s['mode'] == pytest.approx(2.0)


def test_bleach_exponential_refuses_without_decay():
    rng = np.random.default_rng(2)
    means = 190 + rng.normal(0, 0.2, 60)
    means[18:35] -= 2  # a dip, not bleaching
    movie = np.ones((60, 4, 4), np.float32) * means[:, None, None].astype(np.float32)
    with pytest.raises(ValueError, match='no bleaching decay'):
        ops.bleach_correct(movie, axis=0, method='exponential')


def test_dff_never_flips_sign_on_negative_data():
    trace = np.full((1, 100), -250.0)
    trace[0, 40:45] += 40  # a positive transient on a negative baseline
    raw = ops.dff(trace)
    assert np.isnan(raw).all()  # F0 <= 0: refuse rather than flip
    fixed = ops.dff(trace, offset=ops.suggested_offset(trace) - 50)  # dark level 50 below the baseline
    assert fixed[0, 42] == pytest.approx(40 / 50)
    assert ops.suggested_offset(np.array([1.0, 5.0])) == 0.0


def test_correlation_image_finds_cell_not_brightness():
    rng = np.random.default_rng(0)
    movie = rng.normal(0, 1, (300, 24, 30)).astype(np.float32)
    sig = rng.normal(0, 1, 300).astype(np.float32)
    movie[:, 8:14, 8:14] += 3 * sig[:, None, None]
    movie[:, :, 20:] += 100  # bright but uncorrelated
    old = ops.CHUNK_BYTES
    ops.CHUNK_BYTES = 300 * 30 * 4 * 5  # 5-row chunks with halos
    try:
        cn = ops.correlation_image(movie, 0, (0, 0, 0))
    finally:
        ops.CHUNK_BYTES = old
    assert cn[9:13, 9:13].mean() > 0.8
    assert abs(cn[:, 22:].mean()) < 0.05
    z = (movie - movie.mean(0)) / movie.std(0)
    ref = np.mean([np.mean(z[:, 10, 10] * z[:, 10 + dy, 10 + dx])
                   for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)])
    assert cn[10, 10] == pytest.approx(ref, abs=1e-4)
