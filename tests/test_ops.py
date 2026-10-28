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
