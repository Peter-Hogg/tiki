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


def test_parse_slice_spec_and_substack(tmp_path):
    assert ops.parse_slice_spec('', 10) == slice(None)
    assert ops.parse_slice_spec('all', 10) == slice(None)
    assert ops.parse_slice_spec('1-10', 10) == slice(None)
    assert ops.parse_slice_spec('2-8-2', 10) == slice(1, 8, 2)
    np.testing.assert_array_equal(ops.parse_slice_spec('1, 4, 7-9', 10), [0, 3, 6, 7, 8])
    np.testing.assert_array_equal(ops.parse_slice_spec('5', 10), [4])
    for bad in ('0-5', '5-2', '1-12', 'abc', '1,,2'):
        with pytest.raises(ValueError):
            ops.parse_slice_spec(bad, 10)

    data = np.arange(10 * 2 * 16 * 16, dtype=np.float32).reshape(10, 2, 16, 16)
    out, dropped, mults = ops.substack(data, specs={0: '1-10-2', 1: '2'}, box=(2, 10, 4, 12))
    assert out.shape == (5, 8, 8)
    assert dropped == [1]
    assert mults == {0: 2}
    np.testing.assert_array_equal(out, data[::2, 1, 2:10, 4:12])

    mm_path = tmp_path / 'mm.dat'
    mm = np.memmap(mm_path, dtype=np.float32, mode='w+', shape=(8, 12, 12))
    mm[:] = np.arange(8 * 12 * 12, dtype=np.float32).reshape(8, 12, 12)
    mm.flush()
    mm_ro = np.memmap(mm_path, dtype=np.float32, mode='r', shape=(8, 12, 12))
    view, dropped2, _ = ops.substack(mm_ro, specs={0: '2-6'})
    assert view.shape == (5, 12, 12) and dropped2 == []
    assert np.shares_memory(view, mm_ro)

    fancy, _, _ = ops.substack(mm_ro, specs={0: '1, 4, 7-8'})
    assert fancy.shape == (4, 12, 12)
    np.testing.assert_array_equal(fancy, mm_ro[[0, 3, 6, 7]])


def test_bin_image_2d_nd_and_chunks():
    arr2d = np.arange(9 * 7, dtype=np.uint16).reshape(9, 7)
    binned_mean = ops.bin_image(arr2d, (2, 3), 'mean')
    assert binned_mean.shape == (4, 2) and binned_mean.dtype == np.float32
    np.testing.assert_allclose(binned_mean[0, 0], arr2d[:2, :3].mean())
    binned_max = ops.bin_image(arr2d, (2, 3), 'max')
    assert binned_max.dtype == np.uint16 and binned_max[1, 1] == arr2d[2:4, 3:6].max()

    movie = np.arange(10 * 12 * 14, dtype=np.uint16).reshape(10, 12, 14)
    old = ops.CHUNK_BYTES
    ops.CHUNK_BYTES = 1000
    try:
        out_sum = ops.bin_image(movie, (2, 2, 2), 'sum')
        out_med = ops.bin_image(movie, (2, 2, 2), 'median')
    finally:
        ops.CHUNK_BYTES = old
    assert out_sum.shape == (5, 6, 7) and out_sum.dtype == np.float32
    ref = movie.reshape(5, 2, 6, 2, 7, 2)
    np.testing.assert_allclose(out_sum, ref.sum(axis=(1, 3, 5)))
    np.testing.assert_allclose(out_med, np.median(ref, axis=(1, 3, 5)))

    vol4d = np.ones((4, 4, 8, 8), dtype=np.float32)
    assert ops.bin_image(vol4d, (2, 2, 2, 2), 'mean').shape == (2, 2, 4, 4)


def test_motion_correct_recovers_shifts_and_clamps():
    from scipy.ndimage import gaussian_filter
    from scipy.ndimage import shift as nd_shift

    rng = np.random.default_rng(7)
    field = gaussian_filter(rng.normal(0, 1, (120, 120)).astype(np.float32), 2) * 50 + 100
    true = rng.uniform(-3, 3, (25, 2))
    c1 = np.stack([nd_shift(field, s, order=3)[30:90, 30:90] for s in true]).astype(np.float32)
    c1 += rng.normal(0, 2, c1.shape).astype(np.float32)
    c2 = np.stack([nd_shift(field * 0.5, s, order=3)[30:90, 30:90] for s in true]).astype(np.float32)

    reg1, info = ops.motion_correct(c1, axis=0, template='median', upsample=10)
    est = info['shifts']
    err = np.abs((est - est.mean(0)) + (true - true.mean(0))).mean()
    assert err < 0.2
    inner = (slice(None), slice(6, -6), slice(6, -6))
    assert reg1[inner].std(axis=0).mean() < 0.4 * c1[inner].std(axis=0).mean()

    reg2 = ops.apply_shifts(c2, est, axis=0)
    assert reg2[inner].std(axis=0).mean() < 0.35 * c2[inner].std(axis=0).mean()

    clamped, _ = ops.estimate_shifts(c1, axis=0, max_shift=1.2)
    assert np.abs(clamped).max() <= 1.2 + 1e-6


def test_fill_holes_and_watershed_2d_and_3d():
    yy, xx = np.mgrid[:64, :64]
    mask = np.zeros((64, 64), dtype=np.int32)
    # two touching disks
    mask[(yy - 20) ** 2 + (xx - 15) ** 2 <= 9**2] = 1
    mask[(yy - 20) ** 2 + (xx - 27) ** 2 <= 9**2] = 1
    # donut cell with an interior hole
    donut = ((yy - 48) ** 2 + (xx - 45) ** 2 <= 8**2) & ((yy - 48) ** 2 + (xx - 45) ** 2 >= 3**2)
    mask[donut] = 2
    # tiny isolated object
    mask[4:7, 50:53] = 3

    filled = ops.fill_holes_labels(mask)
    assert filled[48, 45] == 2  # hole filled with its own label ID

    split = ops.watershed_labels(mask, min_distance=5.0, fill_holes=True)
    assert split[48, 45] > 0  # donut filled and kept whole
    assert split[5, 51] > 0   # tiny object preserved
    assert split[20, 15] != split[20, 27] and split[20, 15] > 0 and split[20, 27] > 0
    assert len(np.unique(split)) == 5  # bg + 4 objects

    # 3D anisotropic touching spheres (z_ratio = 0.25 -> z step 4x XY pixel)
    zz, yy3, xx3 = np.ogrid[:14, :40, :40]
    vol = ((zz - 4) ** 2 * 16 + (yy3 - 15) ** 2 + (xx3 - 20) ** 2 <= 10**2) | (
        (zz - 9) ** 2 * 16 + (yy3 - 15) ** 2 + (xx3 - 20) ** 2 <= 10**2
    )
    split3d = ops.watershed_labels(vol, min_distance=4.0, z_ratio=0.25, fill_holes=True)
    assert split3d[4, 15, 20] != split3d[9, 15, 20] and split3d.max() == 2

    stack = np.stack([mask, mask])
    blk = ops.watershed_blocks(stack, [(0, slice(None), slice(None)), (1, slice(None), slice(None))], min_distance=5.0)
    assert blk[0].max() == 4 and blk[1].min() == 0 and set(np.unique(blk[1])) - {0} == {5, 6, 7, 8}


def test_line_profile_polyline_and_width():
    plane = np.zeros((30, 30), dtype=np.float32)
    plane[9:12, :] = 100.0
    plane[:, 19:22] = 200.0
    verts = np.array([[10.0, 2.0], [10.0, 20.0], [25.0, 20.0]])
    dist, vals = ops.line_profile(plane, verts, width=3, scale=(0.5, 0.5))
    assert len(dist) == len(vals) and dist[0] == 0.0
    assert dist[-1] == pytest.approx((18.0 + 15.0) * 0.5, rel=0.05)
    assert vals[5] == pytest.approx(100.0, abs=1.0)
    assert vals[-3] == pytest.approx(200.0, abs=1.0)


def test_wand_enlarge_band_and_clear():
    plane = np.full((40, 40), 10.0, dtype=np.float32)
    plane[12:22, 14:24] = 100.0
    plane[15, 16] = 105.0
    m = ops.wand_mask(plane, 15, 16, tolerance=10.0)
    assert m.sum() == 100
    poly = ops.mask_to_polygon(m)
    assert poly is not None and len(poly) >= 4

    big = ops.enlarge_mask(m, 3.0)
    small = ops.enlarge_mask(m, -2.0)
    assert big.sum() > m.sum() > small.sum() > 0

    band = ops.make_band_mask(m, width=3.0, gap=1.0)
    assert band.any() and not (band & m).any()

    stack = np.ones((3, 40, 40), dtype=np.uint16) * 50
    cleared_in = ops.clear_mask(stack, m, outside=False, whole_stack=True)
    cleared_out = ops.clear_mask(stack, m, outside=True, whole_stack=True)
    assert cleared_in[:, m].max() == 0 and cleared_in[:, ~m].min() == 50
    assert cleared_out[:, m].min() == 50 and cleared_out[:, ~m].max() == 0


def test_transforms_type_conversion_and_composite_rgb():
    from napari.utils.colormaps import Colormap

    arr = np.arange(12, dtype=np.uint16).reshape(3, 4)
    np.testing.assert_array_equal(ops.flip_image(arr, 'horizontal'), arr[:, ::-1])
    np.testing.assert_array_equal(ops.flip_image(arr, 'vertical'), arr[::-1, :])
    rot_r = ops.rotate_90(arr, -1)
    assert rot_r.shape == (4, 3)
    np.testing.assert_array_equal(ops.rotate_90(rot_r, 1), arr)

    rot_a = ops.rotate_angle(arr.astype(np.float32), 90.0, resize=True)
    assert rot_a.shape == (4, 3) and rot_a.dtype == np.float32

    u8 = ops.convert_type(np.array([[0, 500], [1000, 2000]], dtype=np.uint16), 'uint8', scale_range=(0, 1000))
    assert u8.dtype == np.uint8 and u8[0, 1] == 128 and u8[1, 0] == 255 and u8[1, 1] == 255

    g_map = Colormap([[0, 0, 0, 1], [0, 1, 0, 1]], name='g')
    r_map = Colormap([[0, 0, 0, 1], [1, 0, 0, 1]], name='r')
    c1 = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    c2 = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    rgb = ops.composite_rgb([(c1, g_map, (0.0, 1.0)), (c2, r_map, (0.0, 1.0))])
    assert rgb.shape == (2, 2, 3) and rgb.dtype == np.uint8
    assert tuple(rgb[1, 0]) == (255, 255, 0)  # green + red = yellow


def test_mask_stats_extended_measurements():
    plane = np.arange(100, dtype=np.float32).reshape(10, 10)
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:6, 3:8] = True
    row = ops.mask_stats(plane, mask, pixel_area=0.25, scale=(0.5, 0.5), include=ops.MEASUREMENT_KEYS)
    assert row['Area'] == pytest.approx(20 * 0.25)
    assert row['Median'] == pytest.approx(np.median(plane[mask]))
    assert 'X' in row and 'Y' in row and 'Perim.' in row and 'Circ.' in row and 'Feret' in row and 'Solidity' in row


def test_make_montage_and_orthogonal_slices():
    stack = np.arange(6 * 4 * 5, dtype=np.uint16).reshape(6, 4, 5)
    m = ops.make_montage(stack, cols=3, rows=2, border=1, border_value=999)
    assert m.shape == (2 * 4 + 1, 3 * 5 + 2) and m.dtype == np.uint16
    np.testing.assert_array_equal(m[:4, :5], stack[0])
    np.testing.assert_array_equal(m[5:9, 12:17], stack[5])
    assert m[4, 0] == 999  # horizontal border

    m_sub = ops.make_montage(stack, indices=[0, 2, 4], cols=2, border=0)
    assert m_sub.shape == (8, 10)
    np.testing.assert_array_equal(m_sub[4:8, :5], stack[4])
    assert m_sub[4:8, 5:10].max() == 0  # trailing empty cell

    rgb_stack = np.zeros((4, 3, 4, 3), dtype=np.uint8)
    rgb_stack[3, :, :, 0] = 200
    m_rgb = ops.make_montage(rgb_stack, cols=2, rows=2, border=1, border_value=255, rgb=True)
    assert m_rgb.shape == (7, 9, 3) and m_rgb[4:7, 5:9, 0].min() == 200 and m_rgb[3, 0, 0] == 255

    xy, xz, yz = ops.orthogonal_slices(stack, z=2, y=1, x=3)
    assert xy.shape == (4, 5) and xz.shape == (6, 5) and yz.shape == (4, 6)
    np.testing.assert_array_equal(xy, stack[2])
    np.testing.assert_array_equal(xz, stack[:, 1, :])
    np.testing.assert_array_equal(yz, stack[:, :, 3].T)
    # Out-of-bounds coordinates clamp cleanly
    xy2, _, _ = ops.orthogonal_slices(stack, z=99, y=-5, x=99)
    np.testing.assert_array_equal(xy2, stack[-1])


def test_stack_to_hyperstack_block_and_interleaved():
    # 2 channels of 5 frames laid out as [ch1: 0..4, ch2: 10..14]
    ch1 = np.arange(5, dtype=np.uint16)[:, None, None] * np.ones((1, 4, 4), dtype=np.uint16)
    ch2 = (10 + np.arange(5, dtype=np.uint16))[:, None, None] * np.ones((1, 4, 4), dtype=np.uint16)
    flat_blocks = np.concatenate([ch1, ch2], axis=0)

    hs, roles = ops.stack_to_hyperstack(flat_blocks, channels=2, slices=1, frames=5, order='xytzc')
    assert roles == 'CTYX' and hs.shape == (2, 5, 4, 4)
    np.testing.assert_array_equal(hs[0], ch1)
    np.testing.assert_array_equal(hs[1], ch2)

    # Interleaved [ch1_0, ch2_0, ch1_1, ch2_1, ...]
    interleaved = np.empty((10, 4, 4), dtype=np.uint16)
    interleaved[0::2] = ch1
    interleaved[1::2] = ch2
    hs2, roles2 = ops.stack_to_hyperstack(interleaved, channels=2, slices=1, frames=5, order='xyczt')
    assert roles2 == 'TCYX' and hs2.shape == (5, 2, 4, 4)
    np.testing.assert_array_equal(hs2[:, 0], ch1)
    np.testing.assert_array_equal(hs2[:, 1], ch2)

    with pytest.raises(ValueError):
        ops.stack_to_hyperstack(flat_blocks, channels=3, slices=1, frames=3, order='xytzc')


def test_anisotropic_mask_stats_and_particle_table():
    plane = np.ones((20, 20), dtype=np.float32) * 10.0
    mask = np.zeros((20, 20), dtype=bool)
    mask[4:10, 4:14] = True  # 6 px tall x 10 px wide
    # Anisotropic scaling: sy=2.0 µm/px, sx=0.5 µm/px => physical 12 µm tall x 5 µm wide
    stats = ops.mask_stats(plane, mask, pixel_area=1.0, scale=(2.0, 0.5), include=ops.MEASUREMENT_KEYS)
    assert stats['Area'] == pytest.approx(60.0)
    assert stats['X'] == pytest.approx(8.5 * 0.5)
    assert stats['Y'] == pytest.approx(6.5 * 2.0)
    assert stats['Perim.'] > 20.0
    assert stats['Feret'] > 10.0

    labels = np.zeros((20, 20), dtype=np.int32)
    labels[4:10, 4:14] = 1
    rows = ops.particle_table(labels, plane, scale=(2.0, 0.5))
    assert len(rows) == 1
    r = rows[0]
    assert r['Area'] == pytest.approx(60.0)
    assert r['X'] == pytest.approx(8.5 * 0.5)
    assert r['Y'] == pytest.approx(6.5 * 2.0)
    assert 'Feret' in r and r['Feret'] == pytest.approx(stats['Feret'])


def test_invert_math_resize_and_edge_filters():
    # Invert uint8, float32, RGBA (alpha preserved)
    u8 = np.array([[0, 100, 255]], dtype=np.uint8)
    np.testing.assert_array_equal(ops.invert_image(u8), np.array([[255, 155, 0]], dtype=np.uint8))

    f32 = np.array([[1.0, 3.0, 5.0]], dtype=np.float32)
    np.testing.assert_allclose(ops.invert_image(f32), np.array([[5.0, 3.0, 1.0]], dtype=np.float32))

    rgba = np.array([[[10, 20, 30, 128]]], dtype=np.uint8)
    inv_rgba = ops.invert_image(rgba, rgb=True)
    np.testing.assert_array_equal(inv_rgba, np.array([[[245, 235, 225, 128]]], dtype=np.uint8))

    # Math ops
    arr = np.array([[1.0, 4.0], [-9.0, 16.0]], dtype=np.float32)
    np.testing.assert_allclose(ops.math_op(arr, 'Add', 2.0), arr + 2.0)
    np.testing.assert_allclose(ops.math_op(arr, 'Multiply', 3.0), arr * 3.0)
    np.testing.assert_allclose(ops.math_op(arr, 'Abs'), np.abs(arr))
    np.testing.assert_allclose(ops.math_op(np.array([[4.0, 9.0]], np.float32), 'Square Root'), np.array([[2.0, 3.0]], np.float32))
    # NaN on <= 0 in Log, non-finite ignored in Divide by 0 -> nan
    div0 = ops.math_op(arr, 'Divide', 0.0)
    assert np.all(np.isnan(div0))

    # Resize stack & RGB
    stack = np.arange(2 * 8 * 10, dtype=np.uint16).reshape(2, 8, 10)
    resized = ops.resize_image(stack, (16, 20), order=1)
    assert resized.shape == (2, 16, 20) and resized.dtype == np.uint16
    rgb_img = np.zeros((3, 8, 10, 3), dtype=np.uint8)
    assert ops.resize_image(rgb_img, (4, 5), order=0, rgb=True).shape == (3, 4, 5, 3)

    # Mean, Minimum, Maximum, Unsharp Mask (2D & 3D), Find Edges
    plane = np.zeros((11, 11), dtype=np.float32)
    plane[5, 5] = 100.0
    assert ops.maximum_filter(plane, 1.0)[5, 6] == pytest.approx(100.0)
    assert ops.minimum_filter(np.pad(np.ones((5, 5)), 2), 1.0)[2, 2] == pytest.approx(0.0)
    assert 0.0 < ops.mean_filter(plane, 1.0)[5, 5] < 100.0
    sharp = ops.unsharp_mask(plane, sigma=1.0, weight=0.6)
    assert sharp[5, 5] > plane[5, 5]
    edges = ops.find_edges(np.pad(np.ones((5, 5), dtype=np.float32), 3))
    assert edges.shape == (11, 11) and edges[3, 3] > 0.0 and edges[0, 0] == 0.0

    vol = np.zeros((5, 9, 9), dtype=np.float32)
    vol[2, 4, 4] = 50.0
    assert ops.mean_3d(vol, 1.0, z_ratio=2.0).shape == vol.shape
    assert ops.maximum_3d(vol, 1.0, z_ratio=2.0)[2, 4, 5] == pytest.approx(50.0)
    assert ops.minimum_3d(vol + 10.0, 1.0, z_ratio=2.0).min() == pytest.approx(10.0)
    assert ops.unsharp_mask_3d(vol, sigma=1.0, weight=0.5, z_ratio=2.0)[2, 4, 4] > vol[2, 4, 4]


def test_morphology_and_stack_manipulation_ops():
    # Binary morphology
    bin_mask = np.zeros((15, 15), dtype=np.uint8)
    bin_mask[4:11, 4:11] = 1
    bin_mask[4, 4] = 0
    bin_mask[0, 0] = 1  # single-pixel noise
    opened = ops.morphology_labels(bin_mask, op='open', radius=1.0)
    assert opened.dtype == np.uint8 and opened[0, 0] == 0 and opened[7, 7] == 1

    # Multi-label 3D morphology keeps distinct label IDs
    lab3d = np.zeros((7, 15, 15), dtype=np.int32)
    lab3d[1:6, 2:7, 2:7] = 1
    lab3d[1:6, 8:13, 8:13] = 2
    dil = ops.morphology_labels(lab3d, op='dilate', radius=1.0, z_ratio=1.0)
    assert set(np.unique(dil)) == {0, 1, 2}
    assert dil[3, 1, 4] == 1 and dil[3, 13, 10] == 2

    # Concatenate, Images to Stack, Reverse, Remove Slices, Hyperstack to Stack
    a = np.ones((2, 4, 5), dtype=np.uint8)
    b = np.full((3, 4, 5), 2, dtype=np.float32)
    cat = ops.concatenate_stacks([a, b], axis=0)
    assert cat.shape == (5, 4, 5) and cat.dtype == np.float32
    assert np.all(cat[:2] == 1) and np.all(cat[2:] == 2)

    p1 = np.zeros((4, 5), dtype=np.uint16)
    p2 = np.ones((4, 5), dtype=np.uint16)
    stk = ops.images_to_stack([p1, p2])
    assert stk.shape == (2, 4, 5) and stk[1].min() == 1

    rev = ops.reverse_stack(np.arange(4 * 3 * 3).reshape(4, 3, 3), axis=0)
    assert rev[0, 0, 0] == 3 * 9

    trimmed = ops.remove_slices(np.arange(6 * 2 * 2).reshape(6, 2, 2), axis=0, spec='1, 3-5')
    assert trimmed.shape == (2, 2, 2)
    assert trimmed[0, 0, 0] == 4 and trimmed[1, 0, 0] == 20

    hs = np.arange(2 * 3 * 4 * 4).reshape(2, 3, 4, 4)
    flat = ops.hyperstack_to_stack(hs)
    assert flat.shape == (6, 4, 4)
    np.testing.assert_array_equal(flat[0], hs[0, 0])
    np.testing.assert_array_equal(flat[5], hs[1, 2])

