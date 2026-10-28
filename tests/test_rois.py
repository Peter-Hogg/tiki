import numpy as np

from tiki import rois as R


def test_imagej_roundtrip_preserves_pixels(tmp_path):
    shape = (64, 64)
    items = [
        R.Roi('rect', 'rectangle', np.array([[9.5, 4.5], [9.5, 24.5], [19.5, 24.5], [19.5, 4.5]])),
        R.Roi('oval', 'ellipse', np.array([[29.5, 29.5], [29.5, 49.5], [49.5, 49.5], [49.5, 29.5]])),
        R.Roi('poly', 'polygon', np.array([[5.2, 40.1], [20.7, 60.3], [3.3, 58.9]]), position=(4,)),
        R.Roi('line', 'line', np.array([[50.0, 3.0], [60.0, 20.0]])),
    ]
    path = tmp_path / 'set.zip'
    R.save_imagej(path, items)
    back, skipped = R.load_imagej(path)
    assert skipped == 0
    assert [r.name for r in back] == ['rect', 'oval', 'poly', 'line']
    assert [r.kind for r in back] == ['rectangle', 'ellipse', 'polygon', 'line']
    assert back[2].position == (4,)
    for a, b in zip(items[:3], back[:3]):
        assert (a.mask(shape) ^ b.mask(shape)).sum() <= 0.02 * a.mask(shape).sum() + 2, a.name
    assert back[0].mask(shape).sum() == 10 * 20  # ImageJ rect 5..25 x 10..20
    np.testing.assert_allclose(back[3].yx, items[3].yx, atol=1e-3)


def test_labels_roundtrip():
    labels = np.zeros((40, 40), np.int32)
    labels[3:12, 4:15] = 5
    yy, xx = np.mgrid[:40, :40]
    labels[(yy - 28) ** 2 + (xx - 25) ** 2 < 49] = 9
    rois = R.rois_from_labels(labels)
    assert [r.meta['label'] for r in rois] == [5, 9]
    back = R.labels_from_rois(rois, labels.shape)
    for i, lid in enumerate([5, 9], start=1):
        orig, new = labels == lid, back == i
        assert (orig ^ new).sum() <= 0.1 * orig.sum(), lid
