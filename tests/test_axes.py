import numpy as np
import pytest
import tifffile

from tiki import axes, io


@pytest.fixture(autouse=True)
def tmp_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(axes, 'MEMORY_FILE', tmp_path / 'axes_memory.json')


def test_imagej_hyperstack_axes_and_calibration(tmp_path):
    data = np.zeros((4, 3, 2, 16, 12), np.uint16)
    path = tmp_path / 'hyper.tif'
    tifffile.imwrite(path, data, imagej=True, resolution=(1 / 0.25, 1 / 0.25),
                     metadata={'axes': 'TZCYX', 'spacing': 1.5, 'finterval': 0.5, 'unit': 'um'})
    out, meta = io.read_image(path)
    info = meta['axes']
    assert info.roles == 'TZCYX' and info.confident and info.source == 'ImageJ metadata'
    assert info.scale == pytest.approx([0.5, 1.5, 1.0, 0.25, 0.25])
    assert info.axes[0].unit == 's' and info.axes[3].unit == 'µm'
    assert info.labels == ['t', 'z', 'c', 'y', 'x']


def test_ome_axes(tmp_path):
    data = np.zeros((2, 5, 8, 8), np.uint8)
    path = tmp_path / 'img.ome.tif'
    tifffile.imwrite(path, data, ome=True, metadata={'axes': 'CZYX', 'PhysicalSizeX': 0.1,
                                                      'PhysicalSizeY': 0.1, 'PhysicalSizeZ': 2.0})
    info = io.read_image(path)[1]['axes']
    assert info.roles == 'CZYX' and info.source == 'OME metadata'
    assert info.scale == pytest.approx([1.0, 2.0, 0.1, 0.1])


def test_no_metadata_is_unknown_not_guessed_as_z(tmp_path):
    path = tmp_path / 'movie.npy'
    np.save(path, np.zeros((50, 8, 8), np.float32))
    info = io.read_image(path)[1]['axes']
    assert info.roles == '?YX' and not info.confident
    assert info.labels[0] == 'dim0'


def test_remembered_axes_apply_to_folder(tmp_path):
    a, b = tmp_path / 'a.npy', tmp_path / 'b.npy'
    np.save(a, np.zeros((50, 8, 8)))
    np.save(b, np.zeros((70, 8, 8)))
    info = axes.AxesInfo([axes.Axis('T', 50, 0.002, 's'), axes.Axis('Y', 8), axes.Axis('X', 8)], 'set by you', True)
    axes.remember(a, (50, 8, 8), info)
    got = io.read_image(b)[1]['axes']
    assert got.roles == 'TYX' and got.axes[0].step == 0.002 and got.axes[0].size == 70
    assert got.source == 'remembered for this folder'


def test_bad_metadata_falls_back_gently(tmp_path, monkeypatch):
    path = tmp_path / 'x.tif'
    tifffile.imwrite(path, np.zeros((3, 8, 8), np.uint8))

    def boom(*a, **k):
        raise ValueError('corrupt tag')

    monkeypatch.setattr(axes, 'from_tiff', boom)
    data, meta = io.read_image(path)
    assert data.shape == (3, 8, 8)
    assert meta['axes'].roles == '?YX'
    assert any('corrupt tag' in n for n in meta['axes'].notes)


def test_problems_and_derivation():
    info = axes.AxesInfo([axes.Axis('T', 5), axes.Axis('C', 2), axes.Axis('Y', 4), axes.Axis('X', 4)])
    assert info.problems((5, 2, 4, 4)) == []
    assert info.without(1).roles == 'TYX'
    bad = axes.AxesInfo([axes.Axis('T', 5), axes.Axis('T', 2), axes.Axis('X', 4), axes.Axis('Y', 4)])
    probs = bad.problems((5, 2, 4, 4))
    assert any('more than one T' in p for p in probs) and any('Y and X' in p for p in probs)
    assert axes.stack_axis(info) == 0
    assert axes.stack_axis(axes.AxesInfo([axes.Axis('C', 2), axes.Axis('Y', 4), axes.Axis('X', 4)])) is None
