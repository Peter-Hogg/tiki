"""Reading and writing images. Large files are memory-mapped, never copied."""

from __future__ import annotations

import fnmatch
import logging
import re
from pathlib import Path

import numpy as np
import tifffile

from . import axes as axes_mod

OPEN_FILTER = 'Images (*.tif *.tiff *.czi *.npy *.png *.jpg);;All files (*)'
IMAGE_SUFFIXES = {'.tif', '.tiff', '.czi', '.npy', '.png', '.jpg', '.jpeg'}
ROI_SUFFIXES = {'.roi', '.zip'}
MACRO_SUFFIXES = {'.py'}


def _natural_key(path: Path) -> list[tuple[int, int | str]]:
    """Sort key that orders ``img2.tif`` before ``img10.tif``."""
    return [(0, int(s)) if s.isdigit() else (1, s.lower()) for s in re.split(r'(\d+)', path.name)]


def sort_dropped(paths) -> dict[str, list[Path]]:
    """Sort dropped paths into images, ROI files, macros and anything unsupported.

    A folder contributes its image files (not recursive), naturally sorted by
    name, as ImageJ does when a folder is dropped on its toolbar.
    """
    out: dict[str, list[Path]] = {'images': [], 'rois': [], 'macros': [], 'skipped': []}
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            out['images'] += sorted(
                (f for f in p.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES),
                key=_natural_key,
            )
            continue
        suffix = p.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            out['images'].append(p)
        elif suffix in ROI_SUFFIXES:
            out['rois'].append(p)
        elif suffix in MACRO_SUFFIXES:
            out['macros'].append(p)
        else:
            out['skipped'].append(p)
    return out


def find_sequence_files(folder: str | Path, pattern: str = '') -> list[Path]:
    """Image files in ``folder`` (non-recursive), naturally sorted and optionally filtered."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f'Not a folder: {folder}')
    files = sorted(
        (f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES),
        key=_natural_key,
    )
    pat = (pattern or '').strip()
    if pat:
        low = pat.lower()
        if '*' in pat or '?' in pat:
            files = [f for f in files if fnmatch.fnmatch(f.name.lower(), low)]
        else:
            files = [f for f in files if low in f.name.lower()]
    return files


def read_sequence(
    folder_or_paths,
    pattern: str = '',
    start: int = 1,
    count: int | None = None,
    step: int = 1,
    axis_role: str = 'T',
    axis_step: float | None = None,
    axis_unit: str | None = None,
) -> tuple[object, dict]:
    """Open a sequence of image files lazily as one dask-backed nD stack."""
    import dask
    import dask.array as da

    if isinstance(folder_or_paths, (str, Path)):
        folder = Path(folder_or_paths)
        files = find_sequence_files(folder, pattern)
        name = folder.name or 'sequence'
        path_str = str(folder)
    else:
        files = [Path(p) for p in folder_or_paths]
        pat = (pattern or '').strip()
        if pat:
            low = pat.lower()
            if '*' in pat or '?' in pat:
                files = [f for f in files if fnmatch.fnmatch(f.name.lower(), low)]
            else:
                files = [f for f in files if low in f.name.lower()]
        name = files[0].parent.name if files else 'sequence'
        path_str = str(files[0].parent) if files else ''

    start_idx = max(0, int(start) - 1)
    step_n = max(1, int(step))
    files = files[start_idx::step_n]
    if count is not None and count > 0:
        files = files[: int(count)]
    if not files:
        raise ValueError('no matching image files found')

    first_data, first_meta = read_image(files[0])
    first_arr = np.asarray(first_data)
    inner_shape = first_arr.shape
    dtype = first_arr.dtype
    inner_info: axes_mod.AxesInfo = first_meta['axes']
    rgb = len(inner_shape) == len(inner_info.axes) + 1 and inner_shape[-1] in (3, 4)

    if len(files) > 1:
        last_data, _ = read_image(files[-1])
        if last_data.shape != inner_shape:
            raise ValueError(
                f'{files[-1].name} shape {last_data.shape} does not match {files[0].name} shape {inner_shape}'
            )

    def _load_one(p: Path) -> np.ndarray:
        arr = np.asarray(read_image(p)[0])
        if arr.shape != inner_shape:
            raise ValueError(f'{p.name} shape {arr.shape} does not match {inner_shape}')
        return arr.astype(dtype, copy=False)

    delayed_frames = [
        da.from_delayed(dask.delayed(_load_one)(p), shape=inner_shape, dtype=dtype)
        for p in files
    ]
    stack = da.stack(delayed_frames, axis=0)

    role = axis_role if axis_role in ('T', 'Z', 'C', '?') else '?'
    inner_axes = [
        axes_mod.Axis('?' if (role != '?' and a.role == role) else a.role, a.size, a.step, a.unit)
        for a in inner_info.axes
    ]
    step_val = float(axis_step) if axis_step and axis_step > 0 else None
    unit_val = (axis_unit or axes_mod.DEFAULT_UNITS.get(role)) if step_val else None
    all_axes = [axes_mod.Axis(role, len(files), step_val, unit_val)] + inner_axes
    confident = '?' not in ''.join(a.role for a in all_axes[:-2])
    info = axes_mod.AxesInfo(all_axes, 'image sequence', confident, [f'{len(files)} files from {files[0].name}'])

    first_plane = first_arr[(0,) * max(0, first_arr.ndim - (3 if rgb else 2))]
    return stack, {
        'name': name,
        'path': path_str,
        'axes': info,
        'n_files': len(files),
        'first_plane': first_plane,
        'rgb': rgb,
    }


def _axes_without_metadata(path: Path, shape: tuple, rgb: bool = False) -> axes_mod.AxesInfo:
    """Folder memory first, then a cautious guess."""
    return axes_mod.recall(path, shape) or axes_mod.guess(shape, rgb=rgb)


def _czi_color_to_lut(color: str) -> np.ndarray | None:
    """Convert Zeiss ``#AARRGGBB`` or ``#RRGGBB`` hex color to a ``(3, 256)`` uint8 LUT."""
    text = (color or '').strip().lstrip('#')
    if len(text) == 8:
        text = text[2:]  # drop alpha
    if len(text) != 6:
        return None
    try:
        r, g, b = int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)
    except ValueError:
        return None
    if r == g == b == 0:
        return None
    ramp = np.linspace(0.0, 1.0, 256)
    lut = np.vstack([
        np.round(ramp * r),
        np.round(ramp * g),
        np.round(ramp * b),
    ]).astype(np.uint8)
    return lut


def _squeeze_czi(data: np.ndarray, codes: str) -> tuple[np.ndarray, str]:
    """Drop singleton non-spatial axes (e.g. B=1, H=1, 0=1) while keeping Y, X and RGB."""
    if len(codes) != data.ndim:
        return data, codes
    keep = []
    for i, (code, size) in enumerate(zip(codes, data.shape)):
        if code in ('Y', 'X'):
            keep.append(i)
        elif i == data.ndim - 1 and code in ('S', '0') and size in (3, 4):
            keep.append(i)
        elif size > 1:
            keep.append(i)
    if len(keep) == data.ndim:
        return data, codes
    drop = tuple(i for i in range(data.ndim) if i not in keep)
    return np.squeeze(data, axis=drop), ''.join(codes[i] for i in keep)


def _czi_luts(channels_meta: dict | None, xml: str, n_channels: int) -> list[np.ndarray] | None:
    colors: list[str] = []
    if channels_meta:
        for ch in channels_meta.values():
            if isinstance(ch, dict) and ch.get('Color'):
                colors.append(str(ch['Color']))
    if len(colors) != n_channels:
        _, xml_colors = axes_mod.from_czi_xml(xml)
        if xml_colors:
            colors = xml_colors
    if len(colors) != n_channels:
        return None
    luts = [_czi_color_to_lut(c) for c in colors]
    return luts if all(lut is not None for lut in luts) else None


def read_czi_scenes(path: str | Path) -> list[tuple[np.ndarray, dict]]:
    """Read all scenes of a Zeiss ``.czi`` file as ``[(data, meta), ...]``."""
    try:
        import czifile
    except ImportError as exc:
        raise RuntimeError('Opening .czi files requires czifile: pip install czifile') from exc

    path = Path(path)
    results: list[tuple[np.ndarray, dict]] = []
    with czifile.CziFile(path) as czi:
        try:
            xml = czi.metadata() if hasattr(czi, 'scenes') else czi.metadata(raw=True)
            if not isinstance(xml, str):
                xml = ''
        except Exception:
            xml = ''

        if hasattr(czi, 'scenes') and czi.scenes:
            items = list(czi.scenes.items())
            multi = len(items) > 1
            for scene_idx, czi_img in items:
                data = czi_img.asarray()
                data, codes = _squeeze_czi(data, czi_img.axes)
                try:
                    info = axes_mod.from_czi(
                        codes,
                        data.shape,
                        coord_scales=getattr(czi_img, 'coord_scales', None),
                        coord_units=getattr(czi_img, 'coord_units', None),
                        xml=xml,
                    )
                except Exception as exc:
                    info = axes_mod.guess(data.shape)
                    info.notes.append(f'could not read CZI axis metadata ({type(exc).__name__}: {exc})')
                if not info.confident:
                    info = axes_mod.recall(path, data.shape) or info
                name = f'{path.name} [S{scene_idx}]' if multi else path.name
                meta: dict = {'name': name, 'path': str(path), 'axes': info}
                c_axis = info.index('C')
                n_ch = data.shape[c_axis] if c_axis is not None else 1
                luts = _czi_luts(getattr(czi_img, 'channels', None), xml, n_ch)
                if luts is not None:
                    meta['imagej_luts'] = luts
                results.append((data, meta))
        else:
            # Compatibility with older czifile (<2026) without .scenes
            data = czi.asarray()
            data, codes = _squeeze_czi(data, getattr(czi, 'axes', ''))
            info = axes_mod.from_czi(codes, data.shape, xml=xml)
            if not info.confident:
                info = axes_mod.recall(path, data.shape) or info
            meta = {'name': path.name, 'path': str(path), 'axes': info}
            c_axis = info.index('C')
            n_ch = data.shape[c_axis] if c_axis is not None else 1
            luts = _czi_luts(None, xml, n_ch)
            if luts is not None:
                meta['imagej_luts'] = luts
            results.append((data, meta))
    return results


def read_czi(path: str | Path, scene: int | None = None) -> tuple[np.ndarray, dict]:
    """Read one scene (default: first) of a Zeiss ``.czi`` file."""
    scenes = read_czi_scenes(path)
    if not scenes:
        raise ValueError(f'{Path(path).name} contains no image scenes')
    if scene is not None:
        return scenes[scene]
    return scenes[0]


def read_image(path: str | Path) -> tuple[np.ndarray, dict]:
    """Return ``(data, meta)``; meta has ``name``, ``path`` and ``axes`` (AxesInfo).

    Never fails because of metadata: unreadable or inconsistent axis
    information falls back to a guess with a note explaining why.
    """
    path = Path(path)
    meta: dict = {'name': path.name, 'path': str(path)}
    suffix = path.suffix.lower()
    if suffix == '.npy':
        data = np.load(path, mmap_mode='r')
        meta['axes'] = _axes_without_metadata(path, data.shape)
        return data, meta
    if suffix == '.czi':
        return read_czi(path)
    if suffix in ('.tif', '.tiff'):
        # tifffile logs (rather than raises) when the header promises more
        # frames than the file holds; report that ourselves instead.
        tif_log = logging.getLogger('tifffile')
        level = tif_log.level
        tif_log.setLevel(logging.ERROR + 10)
        try:
            with tifffile.TiffFile(path) as tf:
                expected = tf.series[0].shape
                try:
                    data = tf.asarray(out='memmap')
                except Exception:
                    data = tf.asarray()
                try:
                    info = axes_mod.from_tiff(tf, data.shape)
                except Exception as exc:  # odd metadata must never block opening
                    info = axes_mod.guess(data.shape)
                    info.notes.append(f'could not read axis metadata ({type(exc).__name__}: {exc})')
                if not info.confident:
                    info = axes_mod.recall(path, data.shape) or info
                luts = (tf.imagej_metadata or {}).get('LUTs')
                if luts is not None:
                    luts_arr = np.asarray(luts)
                    if luts_arr.ndim == 2 and luts_arr.shape == (3, 256):
                        meta['imagej_luts'] = [luts_arr]
                    else:
                        meta['imagej_luts'] = [np.asarray(lut) for lut in luts]
        finally:
            tif_log.setLevel(level)
        if data.shape != tuple(expected):
            meta['warning'] = (
                f'header says {expected} but the file holds {data.shape}; '
                'opened what is there (acquisition stopped early, or split across files?)'
            )
        meta['axes'] = info
        return data, meta
    from skimage.io import imread

    data = imread(path)
    rgb = data.ndim == 3 and data.shape[-1] in (3, 4)
    meta['axes'] = _axes_without_metadata(path, data.shape, rgb=rgb)
    return data, meta


_METRIC_UNITS = {'nm': 1e-9, 'µm': 1e-6, 'um': 1e-6, 'mm': 1e-3}


def _convert_unit(value: float, src: str | None, dst: str | None) -> float:
    if src in _METRIC_UNITS and dst in _METRIC_UNITS:
        return value * (_METRIC_UNITS[src] / _METRIC_UNITS[dst])
    return value


def _ij_unit(unit: str | None) -> str:
    return 'um' if unit == 'µm' else (unit or 'um')


def save_tiff(
    path: str | Path,
    data: np.ndarray,
    scale: tuple | None = None,
    axes: axes_mod.AxesInfo | None = None,
    luts: list[np.ndarray] | None = None,
) -> None:
    """Write a calibrated ImageJ hyperstack or OME-TIFF."""
    arr = np.asarray(data)
    if arr.dtype == np.float64:
        arr = arr.astype(np.float32)
    elif arr.dtype == np.bool_:
        arr = arr.astype(np.uint8) * 255
    elif np.issubdtype(arr.dtype, np.integer) and arr.dtype not in (np.uint8, np.uint16):
        if arr.size == 0 or (int(arr.min()) >= 0 and int(arr.max()) <= 65535):
            arr = arr.astype(np.uint16)
        elif arr.dtype == np.int64 and int(arr.min()) >= -(2**31) and int(arr.max()) < 2**31:
            arr = arr.astype(np.int32)

    want_ome = str(path).lower().endswith(('.ome.tif', '.ome.tiff')) or arr.dtype not in (
        np.uint8,
        np.uint16,
        np.float32,
    )

    if axes is None:
        kwargs: dict = {}
        if scale is not None:
            kwargs['resolution'] = (1.0 / scale[-1], 1.0 / scale[-2])
            if arr.ndim >= 3 and not want_ome:
                kwargs['metadata'] = {'spacing': scale[-3]}
        if want_ome:
            tifffile.imwrite(path, arr, ome=True, **kwargs)
        else:
            tifffile.imwrite(path, arr, imagej=arr.dtype != np.int64, **kwargs)
        return

    rgb = arr.ndim == len(axes.axes) + 1 and arr.shape[-1] in (3, 4)
    expected_shape = arr.shape[:-1] if rgb else arr.shape
    if axes.problems(expected_shape):
        save_tiff(path, arr, scale=scale)
        return

    def _axis(role: str):
        idx = axes.index(role)
        return axes.axes[idx] if idx is not None else None

    y_ax, x_ax, z_ax, t_ax = _axis('Y'), _axis('X'), _axis('Z'), _axis('T')
    sy = float(y_ax.step) if y_ax and y_ax.step else None
    sx = float(x_ax.step) if x_ax and x_ax.step else sy
    xy_unit = (y_ax.unit if y_ax and y_ax.unit else None) or (x_ax.unit if x_ax else None)

    if want_ome:
        ome_meta: dict = {'axes': axes.roles.replace('?', 'Z') + ('S' if rgb else '')}
        if sx:
            ome_meta['PhysicalSizeX'] = sx
            if xy_unit:
                ome_meta['PhysicalSizeXUnit'] = xy_unit
        if sy:
            ome_meta['PhysicalSizeY'] = sy
            if xy_unit:
                ome_meta['PhysicalSizeYUnit'] = xy_unit
        if z_ax and z_ax.step:
            ome_meta['PhysicalSizeZ'] = float(z_ax.step)
            if z_ax.unit:
                ome_meta['PhysicalSizeZUnit'] = z_ax.unit
        if t_ax and t_ax.step:
            ome_meta['TimeIncrement'] = float(t_ax.step)
            if t_ax.unit:
                ome_meta['TimeIncrementUnit'] = t_ax.unit
        tifffile.imwrite(path, arr, ome=True, metadata=ome_meta)
        return

    if '?' in axes.roles[:-2]:
        res_kw: dict = {}
        if sx and sy:
            res_kw['resolution'] = (1.0 / sx, 1.0 / sy)
            res_unit = {
                'µm': tifffile.RESUNIT.MICROMETER,
                'mm': tifffile.RESUNIT.MILLIMETER,
                'cm': tifffile.RESUNIT.CENTIMETER,
            }.get(xy_unit or '')
            if res_unit is not None:
                res_kw['resolutionunit'] = res_unit
        tifffile.imwrite(path, arr, imagej=False, **res_kw)
        return

    rank = {'T': 0, 'Z': 1, 'C': 2, 'Y': 3, 'X': 4}
    perm = sorted(range(len(axes.axes)), key=lambda i: rank[axes.axes[i].role])
    if perm != list(range(len(axes.axes))):
        arr = np.transpose(arr, perm + ([arr.ndim - 1] if rgb else []))
    canon_roles = ''.join(axes.axes[i].role for i in perm)
    ij_meta: dict = {'axes': canon_roles + ('S' if rgb else '')}
    ij_kwargs: dict = {}

    if sx and sy:
        ij_kwargs['resolution'] = (1.0 / sx, 1.0 / sy)
        ij_meta['unit'] = _ij_unit(xy_unit)
    if z_ax and z_ax.step:
        z_dst_unit = xy_unit or z_ax.unit
        ij_meta['spacing'] = _convert_unit(float(z_ax.step), z_ax.unit, z_dst_unit)
        ij_meta.setdefault('unit', _ij_unit(z_dst_unit))
    if t_ax and t_ax.step:
        ij_meta['finterval'] = float(t_ax.step)
        if t_ax.unit and t_ax.unit != 's':
            ij_meta['tunit'] = t_ax.unit

    if luts is not None and not rgb:
        valid_luts = [np.asarray(lut, dtype=np.uint8) for lut in luts if np.asarray(lut).shape == (3, 256)]
        if len(valid_luts) == len(luts):
            if 'C' in canon_roles and len(valid_luts) == arr.shape[canon_roles.index('C')]:
                ij_meta['LUTs'] = valid_luts
                ij_meta['mode'] = 'composite'
            elif 'C' not in canon_roles and len(valid_luts) == 1:
                ij_meta['LUTs'] = valid_luts

    tifffile.imwrite(path, arr, imagej=True, metadata=ij_meta, **ij_kwargs)


def synthetic_movie(n_frames: int = 300, size: int = 256, seed: int = 0) -> np.ndarray:
    """A small noisy movie with a few blinking blobs, for trying out tools."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    movie = rng.normal(100, 8, (n_frames, size, size)).astype(np.float32)
    t = np.arange(n_frames)
    for _ in range(6):
        margin = min(30, size // 4)
        cy, cx = rng.uniform(margin, size - margin, 2)
        blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 6.0**2))
        trace = 60 + 40 * (rng.random(n_frames) < 0.02).astype(float)
        trace = np.convolve(trace, np.exp(-np.arange(30) / 8), mode='full')[:n_frames] / 3
        trace += 20 * np.sin(2 * np.pi * t / rng.uniform(60, 200))
        movie += (blob[None] * trace[:, None, None]).astype(np.float32)
    return movie
