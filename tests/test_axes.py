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


def test_imagej_slices_with_fps_are_time(tmp_path):
    path = tmp_path / 'movie_as_slices.tif'
    tifffile.imwrite(path, np.zeros((30, 8, 8), np.uint16), imagej=True, metadata={'axes': 'ZYX', 'fps': 60.0})
    info = io.read_image(path)[1]['axes']
    assert info.roles == 'TYX'
    assert info.axes[0].step == pytest.approx(1 / 60) and info.axes[0].unit == 's'
    assert any('fps' in n for n in info.notes)


def test_imagej_zstack_with_spacing_stays_z(tmp_path):
    path = tmp_path / 'zstack.tif'
    tifffile.imwrite(path, np.zeros((30, 8, 8), np.uint16), imagej=True, metadata={'axes': 'ZYX', 'spacing': 0.5, 'unit': 'um'})
    assert io.read_image(path)[1]['axes'].roles == 'ZYX'


def test_desktop_launcher_install_and_uninstall(tmp_path, monkeypatch):
    import sys

    from tiki import desktop

    monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path))
    monkeypatch.setattr(sys, 'platform', 'linux')
    monkeypatch.setattr(desktop, '_refresh', lambda: None)
    paths = desktop.install()
    entry = desktop.desktop_path().read_text()
    assert desktop.desktop_path() in paths and desktop.icon_path().exists()
    assert f'Exec={sys.executable} -m tiki %F' in entry and 'MimeType=image/tiff;' in entry
    assert desktop.ICON.exists()
    assert set(desktop.uninstall()) == set(paths) and not desktop.desktop_path().exists()


def test_save_tiff_roundtrip_hyperstack_reorders_and_preserves_luts(tmp_path):
    # Non-canonical CZYX order should transpose to ZCYX on write and preserve steps & LUTs.
    data = np.arange(2 * 3 * 12 * 10, dtype=np.uint16).reshape(2, 3, 12, 10)
    info = axes.AxesInfo(
        [
            axes.Axis('C', 2),
            axes.Axis('Z', 3, 1.5, 'µm'),
            axes.Axis('Y', 12, 0.25, 'µm'),
            axes.Axis('X', 10, 0.25, 'µm'),
        ],
        'test',
        True,
    )
    lut0 = np.zeros((3, 256), dtype=np.uint8)
    lut0[1] = np.arange(256, dtype=np.uint8)
    lut1 = np.zeros((3, 256), dtype=np.uint8)
    lut1[0] = np.arange(256, dtype=np.uint8)

    path = tmp_path / 'czyx.tif'
    io.save_tiff(path, data, axes=info, luts=[lut0, lut1])
    out, meta = io.read_image(path)
    assert meta['axes'].roles == 'ZCYX'
    assert meta['axes'].scale == pytest.approx([1.5, 1.0, 0.25, 0.25])
    np.testing.assert_array_equal(out, np.transpose(data, (1, 0, 2, 3)))
    assert len(meta['imagej_luts']) == 2
    np.testing.assert_array_equal(meta['imagej_luts'][0], lut0)

    # Single-channel stack with one custom LUT
    single = np.arange(3 * 8 * 8, dtype=np.uint16).reshape(3, 8, 8)
    s_info = axes.AxesInfo(
        [axes.Axis('T', 3, 0.2, 's'), axes.Axis('Y', 8, 0.5, 'µm'), axes.Axis('X', 8, 0.5, 'µm')],
        'test',
        True,
    )
    p_single = tmp_path / 'single_lut.tif'
    io.save_tiff(p_single, single, axes=s_info, luts=[lut1])
    _, s_meta = io.read_image(p_single)
    assert s_meta['axes'].roles == 'TYX'
    assert len(s_meta['imagej_luts']) == 1 and s_meta['imagej_luts'][0].shape == (3, 256)


def test_save_tiff_labels_and_unknown_axes_and_ome(tmp_path):
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[2:6, 2:6] = 42
    info = axes.AxesInfo([axes.Axis('Y', 10, 0.5, 'µm'), axes.Axis('X', 10, 0.5, 'µm')], 'test', True)
    p_lab = tmp_path / 'labels.tif'
    io.save_tiff(p_lab, labels, axes=info)
    out_lab, meta_lab = io.read_image(p_lab)
    assert out_lab.dtype == np.uint16
    assert out_lab[3, 3] == 42 and meta_lab['axes'].scale == pytest.approx([0.5, 0.5])

    big_labels = np.array([[0, 100_000], [1, 2]], dtype=np.int32)
    big_info = axes.AxesInfo([axes.Axis('Y', 2), axes.Axis('X', 2)], 'test', True)
    p_ome = tmp_path / 'big_labels.tif'
    io.save_tiff(p_ome, big_labels, axes=big_info)
    out_ome, _ = io.read_image(p_ome)
    assert int(out_ome[0, 1]) == 100_000

    unk = np.zeros((4, 8, 8), dtype=np.uint16)
    unk_info = axes.AxesInfo(
        [axes.Axis('?', 4), axes.Axis('Y', 8, 0.5, 'µm'), axes.Axis('X', 8, 0.5, 'µm')],
        'guess',
        False,
    )
    p_unk = tmp_path / 'unknown.tif'
    io.save_tiff(p_unk, unk, axes=unk_info)
    _, meta_unk = io.read_image(p_unk)
    assert meta_unk['axes'].roles == '?YX' and not meta_unk['axes'].confident


def test_read_sequence_natural_sort_and_lazy(tmp_path):
    seq = tmp_path / 'seq'
    seq.mkdir()
    for n in (1, 2, 10, 20):
        tifffile.imwrite(seq / f'img{n}.tif', np.full((8, 8), n, dtype=np.uint16))
    (seq / 'readme.txt').write_text('ignore')

    found = io.find_sequence_files(seq)
    assert [p.name for p in found] == ['img1.tif', 'img2.tif', 'img10.tif', 'img20.tif']

    stack, meta = io.read_sequence(seq, start=2, count=2, step=1, axis_role='T', axis_step=0.25, axis_unit='s')
    assert hasattr(stack, 'compute')
    assert stack.shape == (2, 8, 8) and stack.dtype == np.uint16
    assert meta['axes'].roles == 'TYX' and meta['axes'].scale[0] == pytest.approx(0.25)
    assert int(np.asarray(stack[0])[0, 0]) == 2 and int(np.asarray(stack[1])[0, 0]) == 10


def test_czi_axes_metadata_and_read_scenes(tmp_path, monkeypatch):
    import sys
    import types

    xml = """<ImageDocument><Metadata>
      <Scaling><Items>
        <Distance Id="X"><Value>2.5e-07</Value></Distance>
        <Distance Id="Y"><Value>2.5e-07</Value></Distance>
        <Distance Id="Z"><Value>1.5e-06</Value></Distance>
      </Items></Scaling>
      <Information><Image><Dimensions>
        <T><Positions><Interval><Increment>0.5</Increment></Interval></Positions></T>
        <Channels>
          <Channel Name="GFP"><Color>#FF00FF00</Color></Channel>
          <Channel Name="RFP"><Color>#FFFF0000</Color></Channel>
        </Channels>
      </Dimensions></Image></Information>
    </Metadata></ImageDocument>"""

    info = axes.from_czi('TCZYX', (3, 2, 4, 12, 10), xml=xml)
    assert info.roles == 'TCZYX' and info.source == 'CZI metadata' and info.confident
    assert info.scale == pytest.approx([0.5, 1.0, 1.5, 0.25, 0.25])
    assert info.axes[3].unit == 'µm' and info.axes[0].unit == 's'

    rgb_info = axes.from_czi('YXS', (8, 8, 3), coord_scales={'X': 5e-9, 'Y': 5e-9},
                             coord_units={'X': 'meter', 'Y': 'meter'})
    assert rgb_info.roles == 'YX' and rgb_info.axes[0].unit == 'nm' and rgb_info.scale == pytest.approx([5.0, 5.0])

    class FakeScene:
        def __init__(self, arr, ax_codes):
            self._arr = arr
            self.axes = ax_codes
            self.coord_scales = {'X': 2.5e-7, 'Y': 2.5e-7, 'Z': 1.5e-6}
            self.coord_units = {'X': 'meter', 'Y': 'meter', 'Z': 'meter'}
            self.channels = {'GFP': {'Color': '#FF00FF00'}, 'RFP': {'Color': '#FFFF0000'}}

        def asarray(self):
            return self._arr

    class FakeCziFile:
        def __init__(self, _path):
            self.scenes = {
                0: FakeScene(np.zeros((1, 2, 4, 12, 10, 1), dtype=np.uint16), 'BCZYX0'),
                1: FakeScene(np.ones((1, 2, 4, 12, 10, 1), dtype=np.uint16), 'BCZYX0'),
            }

        def metadata(self, **_kw):
            return xml

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    fake_mod = types.SimpleNamespace(CziFile=FakeCziFile)
    monkeypatch.setitem(sys.modules, 'czifile', fake_mod)

    czi_path = tmp_path / 'sample.czi'
    czi_path.write_bytes(b'dummy')
    assert io.sort_dropped([czi_path])['images'] == [czi_path]

    scenes = io.read_czi_scenes(czi_path)
    assert len(scenes) == 2
    data0, meta0 = scenes[0]
    assert data0.shape == (2, 4, 12, 10)  # singleton B=1 and 0=1 squeezed
    assert meta0['name'] == 'sample.czi [S0]' and meta0['axes'].roles == 'CZYX'
    assert meta0['axes'].scale == pytest.approx([1.0, 1.5, 0.25, 0.25])
    assert len(meta0['imagej_luts']) == 2
    assert meta0['imagej_luts'][0][1, -1] == 255 and meta0['imagej_luts'][0][0, -1] == 0
    assert meta0['imagej_luts'][1][0, -1] == 255 and meta0['imagej_luts'][1][1, -1] == 0


def test_ome_zarr_nd2_lif_and_hdf5_readers(tmp_path, monkeypatch):
    import sys
    import types

    # 1. OME-Zarr metadata + folder drop recognition
    zdir = tmp_path / 'sample.zarr'
    zdir.mkdir()
    assert io.sort_dropped([zdir])['images'] == [zdir]

    z_attrs = {
        'multiscales': [{
            'axes': [
                {'name': 't', 'unit': 'second'},
                {'name': 'c'},
                {'name': 'z', 'unit': 'micrometer'},
                {'name': 'y', 'unit': 'micrometer'},
                {'name': 'x', 'unit': 'micrometer'},
            ],
            'datasets': [{
                'path': '0',
                'coordinateTransformations': [{'type': 'scale', 'scale': [0.5, 1.0, 2.0, 0.25, 0.25]}],
            }],
        }],
        'omero': {'channels': [{'color': '00FF00'}, {'color': 'FF00FF'}]},
    }
    z_info, z_colors = axes.from_ome_zarr(z_attrs, (3, 2, 4, 16, 16))
    assert z_info.roles == 'TCZYX' and z_info.scale == pytest.approx([0.5, 1.0, 2.0, 0.25, 0.25])
    assert z_colors == ['00FF00', 'FF00FF']

    # 2. Nikon ND2 (multi-position)
    class FakeND2:
        def __init__(self, _p):
            self.sizes = {'P': 2, 'C': 2, 'Y': 8, 'X': 8}
            self.metadata = types.SimpleNamespace(
                channels=[
                    types.SimpleNamespace(channel=types.SimpleNamespace(colorRGB=0x00FF00)),
                    types.SimpleNamespace(channel=types.SimpleNamespace(colorRGB=0x0000FF)),
                ]
            )

        def voxel_size(self):
            return types.SimpleNamespace(x=0.3, y=0.3, z=1.0)

        def to_dask(self):
            return np.zeros((2, 2, 8, 8), dtype=np.uint16)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setitem(sys.modules, 'nd2', types.SimpleNamespace(ND2File=FakeND2))
    nd2_path = tmp_path / 'test.nd2'
    nd2_path.write_bytes(b'dummy')
    nd2_scenes = io.read_nd2_scenes(nd2_path)
    assert len(nd2_scenes) == 2
    assert nd2_scenes[0][0].shape == (2, 8, 8) and nd2_scenes[0][1]['axes'].roles == 'CYX'
    assert nd2_scenes[0][1]['axes'].scale == pytest.approx([1.0, 0.3, 0.3])
    assert len(nd2_scenes[0][1]['imagej_luts']) == 2

    # 3. Leica LIF (lazy dask)
    class FakeLifImg:
        def __init__(self, name, val):
            self.name = name
            self.dims = types.SimpleNamespace(t=2, z=1, y=6, x=6)
            self.channels = 1
            self.scale = (4.0, 4.0, None, 2.0)  # 4 px/um -> 0.25 um, 2 fps -> 0.5 s
            self._val = val

        def get_frame(self, z=0, t=0, c=0):
            return np.full((6, 6), self._val + t, dtype=np.uint16)

    class FakeLifFile:
        def __init__(self, _p):
            pass

        def get_iter_image(self):
            return iter([FakeLifImg('SeriesA', 10), FakeLifImg('SeriesB', 20)])

    monkeypatch.setitem(sys.modules, 'readlif', types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'readlif.reader', types.SimpleNamespace(LifFile=FakeLifFile))
    lif_path = tmp_path / 'exp.lif'
    lif_path.write_bytes(b'dummy')
    lif_scenes = io.read_lif_scenes(lif_path)
    assert len(lif_scenes) == 2
    arr_a, meta_a = lif_scenes[0]
    assert hasattr(arr_a, 'compute') and arr_a.shape == (2, 6, 6)
    assert meta_a['axes'].roles == 'TYX' and meta_a['axes'].scale == pytest.approx([0.5, 0.25, 0.25])
    assert int(np.asarray(arr_a[1])[0, 0]) == 11

    # 4. HDF5 (picks largest dataset and reads axes attr)
    class FakeDataset:
        def __init__(self, arr, attrs=None):
            self._arr = arr
            self.shape = arr.shape
            self.ndim = arr.ndim
            self.dtype = arr.dtype
            self.attrs = attrs or {}

        def __array__(self, dtype=None):
            return np.asarray(self._arr, dtype=dtype)

        def __getitem__(self, item):
            return self._arr[item]

    class FakeH5File:
        def __init__(self, _p, _m='r'):
            self._items = {
                'meta/small': FakeDataset(np.zeros((2, 2), np.float32)),
                'mov': FakeDataset(np.ones((5, 8, 8), np.float32), {'axes': 'TYX'}),
            }

        def visititems(self, fn):
            for k, v in self._items.items():
                fn(k, v)

        def __getitem__(self, k):
            return self._items[k]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setitem(sys.modules, 'h5py', types.SimpleNamespace(File=FakeH5File, Dataset=FakeDataset))
    h5_path = tmp_path / 'movie.h5'
    h5_path.write_bytes(b'dummy')
    h5_data, h5_meta = io.read_hdf5(h5_path)
    assert h5_data.shape == (5, 8, 8) and h5_meta['axes'].roles == 'TYX'
