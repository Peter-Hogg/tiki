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
    assert h5_data.shape == (5, 8, 8) and h5_meta['axes'].roles == 'TYX' and h5_meta['container_key'] == 'mov'


def test_parse_ome_xml_greek_units_deltat_channels_and_instrument():
    xml = '''<?xml version="1.0" encoding="UTF-8"?>
    <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
      <Instrument ID="Instrument:0">
        <Microscope Manufacturer="Zeiss" Model="LSM 980"/>
        <Objective ID="Objective:0" Manufacturer="Zeiss" Model="Plan-Apochromat 40x"
                   NominalMagnification="40" LensNA="1.3" Immersion="Oil"/>
        <Detector ID="Detector:0" Manufacturer="Hamamatsu" Model="ORCA-Fusion" Type="CMOS"/>
      </Instrument>
      <Image ID="Image:0" Name="FieldA">
        <AcquisitionDate>2026-05-01T12:00:00</AcquisitionDate>
        <StageLabel Name="Region1" X="12.5" Y="-4.0" Z="1.0"/>
        <Pixels DimensionOrder="XYCZT" Type="uint16" SizeX="64" SizeY="48" SizeZ="4" SizeC="2" SizeT="3"
                PhysicalSizeX="0.25" PhysicalSizeXUnit="μm"
                PhysicalSizeY="0.25" PhysicalSizeYUnit="μm"
                PhysicalSizeZ="1.2" PhysicalSizeZUnit="μm">
          <Channel ID="Channel:0:0" Name="DAPI" Color="-16711681"
                   ExcitationWavelength="358" ExcitationWavelengthUnit="nm"
                   EmissionWavelength="461" EmissionWavelengthUnit="nm" PinholeSize="1.0"/>
          <Channel ID="Channel:0:1" Name="GCaMP" Color="16711935"
                   ExcitationWavelength="488" EmissionWavelength="510"/>
          <Plane TheZ="0" TheC="0" TheT="0" DeltaT="0.0" DeltaTUnit="s"/>
          <Plane TheZ="0" TheC="0" TheT="1" DeltaT="250.0" DeltaTUnit="ms"/>
          <Plane TheZ="0" TheC="0" TheT="2" DeltaT="500.0" DeltaTUnit="ms"/>
        </Pixels>
      </Image>
    </OME>'''
    parsed = axes.parse_ome_xml(xml, series_index=0)
    steps = parsed['steps']
    assert steps['X'] == pytest.approx((0.25, 'µm'))
    assert steps['Y'] == pytest.approx((0.25, 'µm'))
    assert steps['Z'] == pytest.approx((1.2, 'µm'))
    assert steps['T'] == pytest.approx((0.25, 's'))
    assert any('DeltaT' in n for n in parsed['notes'])
    assert parsed['channel_names'] == ['DAPI', 'GCaMP']
    acq = parsed['acquisition']
    assert acq['image_name'] == 'FieldA'
    assert acq['objective']['model'] == 'Plan-Apochromat 40x'
    assert acq['detector']['model'] == 'ORCA-Fusion'
    assert acq['stage_label']['name'] == 'Region1'
    assert len(acq['channels']) == 2 and acq['channels'][0]['excitation_nm'] == pytest.approx(358.0)


def test_multi_series_ome_tiff_and_channel_roundtrip(tmp_path):
    ome_path = tmp_path / 'multi.ome.tif'
    s0 = np.arange(2 * 6 * 8, dtype=np.uint16).reshape(2, 6, 8)
    s1 = (np.arange(3 * 6 * 8, dtype=np.uint16) + 100).reshape(3, 6, 8)
    with tifffile.TiffWriter(ome_path, ome=True) as tif:
        tif.write(s0, photometric='minisblack', metadata={'axes': 'CYX', 'PhysicalSizeX': 0.2, 'PhysicalSizeY': 0.2})
        tif.write(s1, photometric='minisblack', metadata={'axes': 'CYX', 'PhysicalSizeX': 0.4, 'PhysicalSizeY': 0.4})
    scenes = io.read_multi(ome_path)
    assert len(scenes) == 2
    assert scenes[0][0].shape == (2, 6, 8) and scenes[1][0].shape == (3, 6, 8)
    assert scenes[0][1]['axes'].scale[-2:] == pytest.approx([0.2, 0.2])
    assert scenes[1][1]['axes'].scale[-2:] == pytest.approx([0.4, 0.4])

    # Single hyperstack roundtrip with OME Channel Name and Color via save_tiff
    out_path = tmp_path / 'roundtrip.ome.tif'
    arr = np.zeros((2, 3, 8, 8), dtype=np.uint16)
    arr[0] = 50
    arr[1] = 200
    info = axes.AxesInfo(
        [axes.Axis('C', 2), axes.Axis('Z', 3, 1.5, 'µm'), axes.Axis('Y', 8, 0.25, 'µm'), axes.Axis('X', 8, 0.25, 'µm')],
        'test', True,
    )
    red = np.zeros((3, 256), dtype=np.uint8)
    red[0] = np.arange(256)
    green = np.zeros((3, 256), dtype=np.uint8)
    green[1] = np.arange(256)
    io.save_tiff(out_path, arr, axes=info, luts=[red, green], channel_names=['mCherry', 'EGFP'])
    back, meta = io.read_image(out_path)
    assert back.shape == (2, 3, 8, 8)
    assert meta.get('channel_names') == ['mCherry', 'EGFP']
    assert meta.get('imagej_luts') is not None and len(meta['imagej_luts']) == 2
    assert meta['imagej_luts'][0][0, -1] == 255 and meta['imagej_luts'][1][1, -1] == 255


def test_scanimage_micromanager_czi_nd2_zarr_metadata(tmp_path, monkeypatch):
    import sys
    import types

    # 1. ScanImage fake TiffFile
    class FakePage:
        tags = {}

    class FakeScanImageTf:
        is_ome = False
        is_imagej = False
        is_scanimage = True
        is_micromanager = False
        ome_metadata = None
        imagej_metadata = None
        micromanager_metadata = None
        pages = [FakePage()]
        series = [types.SimpleNamespace(axes='TZYX', shape=(2, 3, 16, 16))]
        scanimage_metadata = {
            'FrameData': {
                'SI.hRoiManager.imagingFovUm': [[-5.0, -5.0], [5.0, -5.0], [5.0, 5.0], [-5.0, 5.0]],
                'SI.hStackManager.actualStackZStepSize': 2.5,
                'SI.hRoiManager.scanFrameRate': 10.0,
            }
        }

    si_info = axes.from_tiff(FakeScanImageTf(), (2, 3, 16, 16))
    assert si_info.roles == 'TZYX' and si_info.source == 'ScanImage metadata'
    assert si_info.axes[0].step == pytest.approx(0.1) and si_info.axes[1].step == pytest.approx(2.5)
    assert si_info.axes[2].step == pytest.approx(10.0 / 16)

    # 2. Micro-Manager fake TiffFile
    class FakeMMTf:
        is_ome = False
        is_imagej = False
        is_scanimage = False
        is_micromanager = True
        ome_metadata = None
        imagej_metadata = None
        scanimage_metadata = None
        pages = [FakePage()]
        series = [types.SimpleNamespace(axes='TYX', shape=(4, 16, 16))]
        micromanager_metadata = {
            'Summary': {
                'PixelSize_um': 0.325,
                'Interval_ms': 200.0,
                'ChNames': ['Brightfield'],
            }
        }

    mm_info = axes.from_tiff(FakeMMTf(), (4, 16, 16))
    assert mm_info.roles == 'TYX'
    assert mm_info.scale == pytest.approx([200.0, 0.325, 0.325])
    assert mm_info.axes[0].unit == 'ms'
    mm_meta = io.extract_tiff_metadata(FakeMMTf())
    assert mm_meta['channel_names'] == ['Brightfield']
    assert 'micromanager' in mm_meta['acquisition']

    # 3. parse_czi_metadata extended fields
    czi_xml = '''<ImageDocument><Metadata><Information>
      <Image><Dimensions>
        <T><Positions><Interval><Increment>0.5</Increment></Interval></Positions></T>
        <Channels>
          <Channel Name="CFP"><Color>#FF00FFFF</Color><ExcitationWavelength>436</ExcitationWavelength><EmissionWavelength>476</EmissionWavelength></Channel>
        </Channels>
      </Dimensions></Image>
      <Instrument>
        <Microscopes><Microscope Name="Axio Observer"><System>LSM 980</System></Microscope></Microscopes>
        <Objectives><Objective Name="Plan-Neofluar 20x"><NominalMagnification>20</NominalMagnification><LensNA>0.5</LensNA><Immersion>Air</Immersion></Objective></Objectives>
      </Instrument>
    </Information>
    <Scaling><Items>
      <Distance Id="X"><Value>3.25e-7</Value></Distance>
      <Distance Id="Y"><Value>3.25e-7</Value></Distance>
    </Items></Scaling>
    </Metadata></ImageDocument>'''
    parsed_czi = axes.parse_czi_metadata(czi_xml)
    assert parsed_czi['channel_names'] == ['CFP']
    assert parsed_czi['steps']['T'] == pytest.approx((0.5, 's'))
    assert parsed_czi['steps']['X'] == pytest.approx((0.325, 'µm'))
    assert parsed_czi['acquisition']['objective']['name'] == 'Plan-Neofluar 20x'
    assert parsed_czi['acquisition']['objective']['magnification'] == pytest.approx(20.0)
    assert parsed_czi['acquisition']['microscope']['system'] == 'LSM 980'

    # 4. ND2 experiment TimeLoop T step + channel names
    class FakeND2:
        def __init__(self, _p):
            self.sizes = {'T': 3, 'C': 2, 'Y': 8, 'X': 8}
            self.metadata = types.SimpleNamespace(
                channels=[
                    types.SimpleNamespace(
                        channel=types.SimpleNamespace(name='DAPI', colorRGB=0x00FF0000, emissionLambdaNm=460, excitationLambdaNm=358),
                        microscope=types.SimpleNamespace(objectiveName='Plan Fluor 20x', objectiveMagnification=20.0, objectiveNumericalAperture=0.75),
                    ),
                    types.SimpleNamespace(
                        channel=types.SimpleNamespace(name='FITC', colorRGB=0x0000FF00),
                        microscope=None,
                    ),
                ]
            )
            self.experiment = [types.SimpleNamespace(parameters=types.SimpleNamespace(periodMs=500.0))]

        def voxel_size(self):
            return types.SimpleNamespace(x=0.3, y=0.3, z=1.0)

        def to_dask(self):
            return np.zeros((3, 2, 8, 8), np.uint16)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setitem(sys.modules, 'nd2', types.SimpleNamespace(ND2File=FakeND2))
    nd2_path = tmp_path / 'sample.nd2'
    nd2_path.write_bytes(b'dummy')
    nd2_scenes = io.read_nd2_scenes(nd2_path)
    assert len(nd2_scenes) == 1
    nd2_meta = nd2_scenes[0][1]
    assert nd2_meta['axes'].roles == 'TCYX'
    assert nd2_meta['axes'].scale == pytest.approx([0.5, 1.0, 0.3, 0.3])
    assert nd2_meta['channel_names'] == ['DAPI', 'FITC']
    assert nd2_meta['acquisition']['objective']['model'] == 'Plan Fluor 20x'

    # 5. OME-Zarr channel labels
    zarr_attrs = {
        'multiscales': [{
            'axes': [{'name': 'c', 'type': 'channel'}, {'name': 'y', 'type': 'space', 'unit': 'micrometer'}, {'name': 'x', 'type': 'space', 'unit': 'micrometer'}],
            'datasets': [{'path': '0', 'coordinateTransformations': [{'type': 'scale', 'scale': [1, 0.5, 0.5]}]}],
        }],
        'omero': {'channels': [{'label': 'Hoechst', 'color': '0000FF'}, {'label': 'AF488', 'color': '00FF00'}]},
    }
    import dask.array as da

    class FakeZarrGroup:
        def __init__(self):
            self.attrs = zarr_attrs
            self._arr = np.zeros((2, 8, 8), np.uint16)

        def keys(self):
            return ['0']

        def __getitem__(self, k):
            return self._arr

    monkeypatch.setitem(sys.modules, 'zarr', types.SimpleNamespace(open=lambda *_a, **_kw: FakeZarrGroup()))
    monkeypatch.setattr(da, 'from_zarr', lambda arr: da.from_array(arr, chunks=(1, 8, 8)))
    z_path = tmp_path / 'test.zarr'
    z_path.mkdir()
    _, z_meta = io.read_zarr(z_path)
    assert z_meta['channel_names'] == ['Hoechst', 'AF488']
    assert z_meta['container_key'] == '0'
    assert z_meta['axes'].scale == pytest.approx([1.0, 0.5, 0.5])


def test_unit_conversion_and_physical_scale_3d():
    assert axes.convert_unit(250, 'nm', 'µm') == pytest.approx(0.25)
    assert axes.convert_unit(500, 'ms', 's') == pytest.approx(0.5)
    assert axes.convert_unit(1.5, 'μm', 'mm') == pytest.approx(0.0015)

    mixed = axes.AxesInfo(
        [axes.Axis('Z', 10, 200.0, 'nm'), axes.Axis('Y', 16, 0.5, 'µm'), axes.Axis('X', 16, 0.5, 'µm')],
        'test', True,
    )
    assert axes.spatial_unit(mixed) == 'µm'
    scale_3d, unit, z_assumed = axes.physical_scale_3d(mixed)
    assert scale_3d == pytest.approx((0.2, 0.5, 0.5))
    assert unit == 'µm' and not z_assumed

    no_z = axes.AxesInfo(
        [axes.Axis('Z', 10, None, None), axes.Axis('Y', 16, 0.4, 'µm'), axes.Axis('X', 16, 0.4, 'µm')],
        'test', True,
    )
    scale_3d_nz, unit_nz, z_assumed_nz = axes.physical_scale_3d(no_z)
    assert scale_3d_nz == pytest.approx((0.4, 0.4, 0.4))
    assert unit_nz == 'µm' and z_assumed_nz

