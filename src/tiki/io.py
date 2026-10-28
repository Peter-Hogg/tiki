"""Reading and writing images. Large files are memory-mapped, never copied."""

from __future__ import annotations

from pathlib import Path

import logging

import numpy as np
import tifffile

OPEN_FILTER = 'Images (*.tif *.tiff *.npy *.png *.jpg);;All files (*)'


def _tiff_scale(tf: tifffile.TiffFile, ndim: int) -> tuple | None:
    """Pixel size from ImageJ/TIFF resolution tags, or None if uncalibrated."""
    try:
        page = tf.pages[0]
        xres = page.tags.get('XResolution')
        yres = page.tags.get('YResolution')
        if xres is None or yres is None:
            return None
        sx = xres.value[1] / xres.value[0]
        sy = yres.value[1] / yres.value[0]
        if sx == 1 and sy == 1:
            return None
        scale = [1.0] * ndim
        scale[-1], scale[-2] = sx, sy
        spacing = (tf.imagej_metadata or {}).get('spacing')
        if spacing and ndim >= 3:
            scale[-3] = float(spacing)
        return tuple(scale)
    except Exception:
        return None


def read_image(path: str | Path) -> tuple[np.ndarray, dict]:
    """Return ``(data, meta)``; meta has ``name`` and optionally ``scale``."""
    path = Path(path)
    meta: dict = {'name': path.name, 'path': str(path)}
    suffix = path.suffix.lower()
    if suffix == '.npy':
        return np.load(path, mmap_mode='r'), meta
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
                scale = _tiff_scale(tf, data.ndim)
        finally:
            tif_log.setLevel(level)
        if data.shape != tuple(expected):
            meta['warning'] = (
                f'header says {expected} but the file holds {data.shape}; '
                'opened what is there (acquisition stopped early, or split across files?)'
            )
        if scale is not None:
            meta['scale'] = scale
        return data, meta
    from skimage.io import imread

    return imread(path), meta


def save_tiff(path: str | Path, data: np.ndarray, scale: tuple | None = None) -> None:
    """Write an ImageJ-compatible TIFF (with calibration when known)."""
    data = np.asarray(data)
    if data.dtype == np.float64:
        data = data.astype(np.float32)
    if data.dtype == np.bool_:
        data = data.astype(np.uint8) * 255
    kwargs: dict = {}
    if scale is not None:
        kwargs['resolution'] = (1 / scale[-1], 1 / scale[-2])
        if data.ndim >= 3:
            kwargs['metadata'] = {'spacing': scale[-3]}
    tifffile.imwrite(path, data, imagej=data.dtype != np.int64, **kwargs)


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
