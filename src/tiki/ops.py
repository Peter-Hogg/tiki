"""Pure image operations (numpy in, numpy out).

Nothing in here knows about Qt or napari, so every command can be tested
and scripted without a GUI. Functions stream over memory-mapped data in
chunks so they work on movies larger than RAM.
"""

from __future__ import annotations

import numpy as np

PROJECTIONS = {
    'Max Intensity': 'max',
    'Average Intensity': 'mean',
    'Sum Slices': 'sum',
    'Standard Deviation': 'std',
    'Median': 'median',
    'Min Intensity': 'min',
}

CHUNK_BYTES = 256 * 2**20


def _rows_per_chunk(data: np.ndarray, axis: int) -> int:
    """Rows (second-last axis) per chunk so one chunk stays under CHUNK_BYTES."""
    row_bytes = data.dtype.itemsize * data.shape[-1] * data.shape[axis]
    return max(1, CHUNK_BYTES // max(row_bytes, 1))


def z_project(data: np.ndarray, method: str = 'max', axis: int = 0) -> np.ndarray:
    """Project ``data`` along ``axis`` (ImageJ Image > Stacks > Z Project).

    ``method`` is one of max, mean, sum, std, median, min. Max/min keep the
    input dtype, sum/mean/std/median return float32 like ImageJ's 32-bit output.
    """
    funcs = {
        'max': np.max,
        'min': np.min,
        'mean': np.mean,
        'sum': np.sum,
        'std': np.std,
        'median': np.median,
    }
    if method not in funcs:
        raise ValueError(f'unknown projection {method!r}')
    axis = axis % data.ndim
    if axis >= data.ndim - 2:
        raise ValueError('can only project along a non-spatial axis')
    func = funcs[method]
    out_dtype = data.dtype if method in ('max', 'min') else np.float32
    out_shape = data.shape[:axis] + data.shape[axis + 1 :]
    out = np.empty(out_shape, dtype=out_dtype)
    step = _rows_per_chunk(data, axis)
    for r0 in range(0, data.shape[-2], step):
        r1 = min(r0 + step, data.shape[-2])
        chunk = np.asarray(data[..., r0:r1, :])
        if method in ('sum', 'mean', 'std'):
            chunk = chunk.astype(np.float64, copy=False)
        out[..., r0:r1, :] = func(chunk, axis=axis)
    return out


def mask_stats(plane: np.ndarray, mask: np.ndarray | None, pixel_area: float = 1.0) -> dict:
    """ImageJ-style measurements of ``plane`` inside ``mask`` (whole plane if None)."""
    values = plane[mask] if mask is not None else plane.ravel()
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        nan = float('nan')
        return dict(Area=0.0, Mean=nan, StdDev=nan, Min=nan, Max=nan, IntDen=nan, RawIntDen=nan)
    area = values.size * pixel_area
    mean = float(values.mean())
    return dict(
        Area=area,
        Mean=mean,
        StdDev=float(values.std(ddof=1)) if values.size > 1 else 0.0,
        Min=float(values.min()),
        Max=float(values.max()),
        IntDen=area * mean,
        RawIntDen=float(values.sum()),
    )


def axis_profile(
    data: np.ndarray,
    axis: int,
    index: tuple,
    mask: np.ndarray | None,
    chunk: int = 512,
) -> np.ndarray:
    """Mean inside ``mask`` for every position along ``axis`` (Plot Z-axis Profile).

    ``index`` is the full nD position; entries for ``axis`` and the last two
    (spatial) axes are ignored. Only the mask's bounding box is read from disk.
    """
    ndim = data.ndim
    axis = axis % ndim
    if mask is None:
        mask = np.ones(data.shape[-2:], dtype=bool)
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0:
        return np.full(data.shape[axis], np.nan)
    ys = slice(rows[0], rows[-1] + 1)
    xs = slice(cols[0], cols[-1] + 1)
    sub_mask = mask[ys, xs]
    out = np.empty(data.shape[axis], dtype=np.float64)
    for t0 in range(0, data.shape[axis], chunk):
        t1 = min(t0 + chunk, data.shape[axis])
        sl = []
        for d in range(ndim - 2):
            sl.append(slice(t0, t1) if d == axis else int(index[d]))
        block = np.asarray(data[tuple(sl) + (ys, xs)], dtype=np.float64)
        out[t0:t1] = block[:, sub_mask].mean(axis=1)
    return out


def saturated_limits(plane: np.ndarray, saturated: float = 0.35) -> tuple[float, float]:
    """Display range that saturates ``saturated`` percent of pixels (Enhance Contrast)."""
    values = np.asarray(plane, dtype=np.float64).ravel()
    if values.size > 1_000_000:
        values = values[:: values.size // 1_000_000]
    lo, hi = np.percentile(values, [saturated / 2, 100 - saturated / 2])
    if hi <= lo:
        hi = lo + 1
    return float(lo), float(hi)


def crop_to_box(data: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Crop the last two axes to ``(y0, y1, x0, x1)``, clipped to the image."""
    y0, y1, x0, x1 = box
    h, w = data.shape[-2:]
    y0, y1 = max(0, y0), min(h, y1)
    x0, x1 = max(0, x0), min(w, x1)
    return data[..., y0:y1, x0:x1]
