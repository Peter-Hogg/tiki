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


# ImageJ's Auto Threshold method names -> scikit-image implementations.
# "Default" in ImageJ is its IsoData variant, which skimage's isodata matches closely.
THRESHOLD_METHODS = ['Default', 'IsoData', 'Li', 'Mean', 'Minimum', 'Otsu', 'Percentile', 'Triangle', 'Yen']


def auto_threshold(values: np.ndarray, method: str = 'Default') -> float:
    """Threshold value for ``values`` by an ImageJ-named method."""
    from skimage import filters

    values = np.asarray(values).ravel()
    if values.size > 2_000_000:
        values = values[:: values.size // 2_000_000]
    if values.min() == values.max():
        return float(values.min())
    funcs = {
        'Default': filters.threshold_isodata,
        'IsoData': filters.threshold_isodata,
        'Li': filters.threshold_li,
        'Mean': filters.threshold_mean,
        'Minimum': filters.threshold_minimum,
        'Otsu': filters.threshold_otsu,
        'Percentile': lambda v: np.percentile(v, 50),
        'Triangle': filters.threshold_triangle,
        'Yen': filters.threshold_yen,
    }
    if method not in funcs:
        raise ValueError(f'unknown threshold method {method!r}')
    return float(funcs[method](values))


def threshold_mask(plane: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Pixels with ``lower <= value <= upper``."""
    plane = np.asarray(plane)
    return (plane >= lower) & (plane <= upper)


def label_objects(mask: np.ndarray, min_size: int = 0, start: int = 0) -> np.ndarray:
    """Connected components of a 2D or 3D mask, fully connected (8 neighbours
    in 2D like Analyze Particles, 26 in 3D like the 3D Objects Counter).

    Objects smaller than ``min_size`` pixels/voxels are dropped; ids start at ``start + 1``.
    """
    from skimage.measure import label

    labels = label(mask, connectivity=mask.ndim).astype(np.int32)
    if min_size > 1 and labels.max():
        sizes = np.bincount(labels.ravel())
        small = sizes < min_size
        small[0] = False
        labels[small[labels]] = 0
        labels = label(labels > 0, connectivity=mask.ndim).astype(np.int32)
    if start:
        labels[labels > 0] += start
    return labels


def segment(block: np.ndarray, lower: float, upper: float, split: bool = True, min_size: int = 0) -> np.ndarray:
    """Threshold one 2D plane or 3D volume: int32 object labels, or a uint8 0/1 mask."""
    mask = threshold_mask(block, lower, upper)
    if split:
        return label_objects(mask, min_size)
    if min_size > 1:
        mask = label_objects(mask, min_size) > 0
    return mask.astype(np.uint8)


def threshold_blocks(data, lower: float, upper: float, blocks: list[tuple], split: bool = True,
                     min_size: int = 0, progress=None) -> np.ndarray:
    """Threshold ``data`` block by block (planes or Z-volumes, see ``tiki.nd``).

    Returns an array the shape of ``data``; with ``split`` every object id is
    unique across all blocks. Blocks are read one at a time (memmap friendly).
    """
    out = np.zeros(data.shape, dtype=np.int32 if split else np.uint8)
    next_id = 0
    for n, b in enumerate(blocks):
        result = segment(np.asarray(data[b]), lower, upper, split, min_size)
        if split and result.max():
            result[result > 0] += next_id
            next_id = int(result.max())
        out[b] = result
        if progress is not None and progress(n + 1, len(blocks)) is False:
            break
    return out


def threshold_stack(
    data: np.ndarray, lower: float, upper: float, split: bool = False, min_size: int = 0
) -> np.ndarray:
    """Threshold every 2D plane of ``data`` into a uint8 mask or int32 labels."""
    lead = data.shape[:-2]
    planes = [idx + (slice(None), slice(None)) for idx in (np.ndindex(*lead) if lead else [()])]
    return threshold_blocks(data, lower, upper, planes, split, min_size)


# ---------------------------------------------------------------- particles

def particle_table(labels: np.ndarray, intensity: np.ndarray, scale=(1.0, 1.0)) -> list[dict]:
    """Analyze Particles: one row per labelled object in a 2D or 3D label image.

    2D columns follow ImageJ (Area, Mean, StdDev, Min, Max, X, Y, Perim.,
    Circ., AR, Solidity, IntDen). 3D gives Volume, X, Y, Z and AR instead of
    the outline measures, using the voxel size so anisotropic z is handled.
    ``scale`` is the physical step of each axis of ``labels``.
    """
    from skimage.measure import regionprops

    labels = np.asarray(labels).astype(np.int32, copy=False)
    intensity = np.asarray(intensity)
    scale = tuple(float(s) for s in scale)[-labels.ndim:]
    rows = []
    if labels.ndim == 3:
        for region in regionprops(labels, intensity, spacing=scale):
            vals = region.image_intensity[region.image].astype(np.float64)
            cz, cy, cx = region.centroid  # physical units (spacing applied)
            major, minor = region.axis_major_length, region.axis_minor_length
            volume = float(region.area)
            rows.append({
                'Label': int(region.label),
                'Volume': volume,
                'Voxels': int(region.image.sum()),
                'Mean': float(vals.mean()),
                'StdDev': float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                'Min': float(vals.min()),
                'Max': float(vals.max()),
                'X': cx, 'Y': cy, 'Z': cz,
                'AR': major / minor if minor > 0 else float('nan'),
                'IntDen': volume * float(vals.mean()),
            })
        return rows
    sy, sx = scale
    for region in regionprops(labels, intensity):
        vals = region.image_intensity[region.image].astype(np.float64)
        area_px = float(region.area)
        perim_px = float(region.perimeter)
        cy, cx = region.centroid
        major, minor = region.axis_major_length, region.axis_minor_length
        rows.append({
            'Label': int(region.label),
            'Area': area_px * sy * sx,
            'Mean': float(vals.mean()),
            'StdDev': float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
            'Min': float(vals.min()),
            'Max': float(vals.max()),
            'X': cx * sx,
            'Y': cy * sy,
            'Perim.': perim_px * sx,
            'Circ.': min(1.0, 4 * np.pi * area_px / perim_px**2) if perim_px > 0 else 1.0,
            'AR': major / minor if minor > 0 else float('nan'),
            'Solidity': float(region.solidity),
            'IntDen': area_px * sy * sx * float(vals.mean()),
        })
    return rows


# ------------------------------------------------------------------- traces

def _lead_slices(ndim: int, axis: int, index: tuple, t0: int, t1: int) -> tuple:
    return tuple(slice(t0, t1) if d == axis else int(index[d]) for d in range(ndim - 2))


def roi_traces(data, axis: int, index: tuple, masks: list[np.ndarray], chunk: int = 256, progress=None) -> np.ndarray:
    """Mean inside each mask at every position along ``axis``: (n_rois, n_frames).

    Reads each chunk of frames once (only the union bounding box of all masks)
    however many ROIs there are. ``progress(done, total)`` may return False to cancel.
    """
    n_t = data.shape[axis]
    out = np.full((len(masks), n_t), np.nan)
    if not masks:
        return out
    union = np.any(masks, axis=0)
    rows, cols = np.flatnonzero(union.any(1)), np.flatnonzero(union.any(0))
    if rows.size == 0:
        return out
    ys, xs = slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)
    flat_idx = [np.flatnonzero(m[ys, xs].ravel()) for m in masks]
    for t0 in range(0, n_t, chunk):
        t1 = min(t0 + chunk, n_t)
        block = np.asarray(data[_lead_slices(data.ndim, axis, index, t0, t1) + (ys, xs)], dtype=np.float64)
        block = block.reshape(t1 - t0, -1)
        for i, idx in enumerate(flat_idx):
            if idx.size:
                out[i, t0:t1] = block[:, idx].mean(axis=1)
        if progress is not None and progress(t1, n_t) is False:
            break
    return out


def label_traces(data, axis: int, index: tuple, labels: np.ndarray, chunk: int = 256, progress=None):
    """Mean of every label in a 2D label image at each frame: ``(ids, (n_ids, n_frames))``."""
    labels = np.asarray(labels)
    ids = np.unique(labels)
    ids = ids[ids > 0]
    n_t = data.shape[axis]
    out = np.full((ids.size, n_t), np.nan)
    if ids.size == 0:
        return ids, out
    # compact ids so bincount stays small even with sparse label numbers
    compact = np.zeros(int(labels.max()) + 1, dtype=np.int64)
    compact[ids] = np.arange(1, ids.size + 1)
    lab = compact[labels].ravel()
    counts = np.bincount(lab, minlength=ids.size + 1)[1:].astype(np.float64)
    for t0 in range(0, n_t, chunk):
        t1 = min(t0 + chunk, n_t)
        block = np.asarray(data[_lead_slices(data.ndim, axis, index, t0, t1)], dtype=np.float64)
        block = block.reshape(t1 - t0, -1)
        for j in range(t1 - t0):
            sums = np.bincount(lab, weights=block[j], minlength=ids.size + 1)[1:]
            out[:, t0 + j] = sums / counts
        if progress is not None and progress(t1, n_t) is False:
            break
    return ids, out


def dff(traces: np.ndarray) -> np.ndarray:
    """ΔF/F with F0 = each trace's median."""
    f0 = np.nanmedian(traces, axis=-1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        return (traces - f0) / f0


# ------------------------------------------------------------------ filters

def gaussian_blur(plane: np.ndarray, sigma: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(np.asarray(plane, dtype=np.float32), sigma)


def median_filter(plane: np.ndarray, radius: float) -> np.ndarray:
    """Median over a disk of ``radius`` pixels (ImageJ's Median filter)."""
    from scipy.ndimage import median_filter as _median
    from skimage.morphology import disk

    r = max(int(round(radius)), 0)
    if r == 0:
        return np.asarray(plane).copy()
    return _median(np.asarray(plane), footprint=disk(r))


def subtract_background(plane: np.ndarray, radius: float, light_background: bool = False) -> np.ndarray:
    """Rolling-ball background subtraction (ImageJ's Subtract Background).

    Like ImageJ, large radii work on a shrunken image (min-pooled) and the
    background is interpolated back up, which keeps big balls fast.
    Integer images are clipped at zero.
    """
    from skimage.restoration import rolling_ball
    from skimage.transform import resize

    plane = np.asarray(plane)
    img = plane.astype(np.float32)
    # Light background: invert, subtract, invert back, so the background
    # ends up at the image maximum (white) as in ImageJ.
    top = float(img.max())
    work = top - img if light_background else img
    shrink = 8 if radius > 100 else 4 if radius > 30 else 2 if radius > 10 else 1
    if shrink > 1:
        h, w = work.shape
        hp, wp = -(-h // shrink) * shrink, -(-w // shrink) * shrink
        padded = np.pad(work, ((0, hp - h), (0, wp - w)), mode='edge')
        small = padded.reshape(hp // shrink, shrink, wp // shrink, shrink).min(axis=(1, 3))
        bg_small = rolling_ball(small, radius=radius / shrink)
        background = resize(bg_small, (hp, wp), order=1, preserve_range=True)[:h, :w]
        background = np.minimum(background, work)  # never subtract more than the pixel
    else:
        background = rolling_ball(work, radius=radius)
    result = work - background
    if light_background:
        result = top - result
    if np.issubdtype(plane.dtype, np.integer):
        info = np.iinfo(plane.dtype)
        return np.clip(np.round(result), max(info.min, 0), info.max).astype(plane.dtype)
    return result.astype(np.float32)


def gaussian_3d(volume: np.ndarray, sigma: float, z_ratio: float) -> np.ndarray:
    """Gaussian through Z: ``sigma`` in x/y pixels, z sigma scaled by voxel shape.

    ``z_ratio`` = x/y pixel size / z step, so a blur of 1 µm is 1 µm in every
    direction even when slices are further apart than pixels.
    """
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(np.asarray(volume, dtype=np.float32), (sigma * z_ratio, sigma, sigma))


def median_3d(volume: np.ndarray, radius: float, z_ratio: float) -> np.ndarray:
    """Median over an ellipsoid of ``radius`` x/y pixels (z radius scaled by voxel shape)."""
    from scipy.ndimage import median_filter as _median

    r = max(int(round(radius)), 0)
    rz = int(round(radius * z_ratio))
    if r == 0:
        return np.asarray(volume).copy()
    zz, yy, xx = np.ogrid[-rz:rz + 1, -r:r + 1, -r:r + 1]
    footprint = (xx / r) ** 2 + (yy / r) ** 2 + ((zz / rz) ** 2 if rz else zz.astype(float) * 0) <= 1
    return _median(np.asarray(volume), footprint=footprint)


def map_blocks(func, data, blocks: list[tuple], out_dtype=None, progress=None) -> np.ndarray:
    """Apply ``func`` to each block (2D plane or 3D volume) of ``data``."""
    first = func(np.asarray(data[blocks[0]]))
    out = np.empty(data.shape, dtype=out_dtype or first.dtype)
    out[blocks[0]] = first
    for n, b in enumerate(blocks[1:], start=2):
        out[b] = func(np.asarray(data[b]))
        if progress is not None and progress(n, len(blocks)) is False:
            break
    return out


def map_planes(func, data, out_dtype=None, progress=None) -> np.ndarray:
    """Apply a 2D ``func`` to every plane of ``data`` (chunk-safe for memmaps)."""
    lead = data.shape[:-2]
    first = func(np.asarray(data[(0,) * len(lead)]))
    out = np.empty(lead + first.shape, dtype=out_dtype or first.dtype)
    total = int(np.prod(lead)) if lead else 1
    for n, idx in enumerate(np.ndindex(*lead) if lead else [()]):
        out[idx] = first if n == 0 else func(np.asarray(data[idx]))
        if progress is not None and progress(n + 1, total) is False:
            break
    return out


# ---------------------------------------------------------------- ΔF/F etc.

F0_METHODS = ['median', 'mean', 'percentile', 'first frames', 'sliding percentile']


def _rows_over_axis(data, axis: int):
    """Row chunks (second-last axis) sized so one chunk of the full axis fits CHUNK_BYTES."""
    step = _rows_per_chunk(data, axis)
    for r0 in range(0, data.shape[-2], step):
        yield slice(r0, min(r0 + step, data.shape[-2]))


def baseline_f0(chunk: np.ndarray, axis: int, method: str = 'median', percentile: float = 10.0,
                n_first: int = 10, window: int = 101) -> np.ndarray:
    """F0 for ΔF/F along ``axis`` of an in-memory chunk (same shape as chunk for
    the sliding method, else with ``axis`` reduced to length 1)."""
    if method == 'median':
        return np.median(chunk, axis=axis, keepdims=True)
    if method == 'mean':
        return np.mean(chunk, axis=axis, keepdims=True)
    if method == 'percentile':
        return np.percentile(chunk, percentile, axis=axis, keepdims=True)
    if method == 'first frames':
        n = max(1, min(int(n_first), chunk.shape[axis]))
        return np.mean(np.take(chunk, range(n), axis=axis), axis=axis, keepdims=True)
    if method == 'sliding percentile':
        from scipy.ndimage import percentile_filter

        size = [1] * chunk.ndim
        size[axis] = max(1, min(int(window), chunk.shape[axis]))
        return percentile_filter(chunk, percentile, size=size, mode='nearest')
    raise ValueError(f'unknown F0 method {method!r}')


def dff_movie(data, axis: int, method: str = 'median', offset: float = 0.0, progress=None,
              return_f0: bool = False, **params):
    """(F - F0) / F0 along ``axis`` (usually T), float32, chunked over rows.

    ``offset`` (e.g. the camera dark level) is subtracted from F first. Pixels
    whose F0 is not positive become NaN rather than exploding.
    """
    axis = axis % data.ndim
    if axis >= data.ndim - 2:
        raise ValueError('ΔF/F needs a stack axis, not Y or X')
    out = np.empty(data.shape, dtype=np.float32)
    f0_shape = list(data.shape)
    if method != 'sliding percentile':
        f0_shape[axis] = 1
    f0_all = np.empty(f0_shape, dtype=np.float32) if return_f0 else None
    rows = list(_rows_over_axis(data, axis))
    for n, rs in enumerate(rows):
        chunk = np.asarray(data[..., rs, :], dtype=np.float32) - np.float32(offset)
        f0 = baseline_f0(chunk, axis, method, **params).astype(np.float32)
        with np.errstate(divide='ignore', invalid='ignore'):
            d = np.where(f0 > 0, (chunk - f0) / f0, np.nan)
        out[..., rs, :] = d
        if return_f0:
            f0_all[..., rs, :] = f0
        if progress is not None and progress(n + 1, len(rows)) is False:
            break
    return (out, f0_all) if return_f0 else out


def bleach_correct(data, axis: int, method: str = 'ratio', background: float = 0.0, progress=None):
    """Correct photobleaching along ``axis`` using each frame's mean intensity.

    ``ratio``: scale every frame to the first frame's mean (ImageJ "Simple
    ratio"). ``exponential``: fit a + b·exp(-t/τ) to the frame means and divide
    by the fit (normalised to t=0). Returns ``(corrected float32, info dict)``.
    """
    axis = axis % data.ndim
    n = data.shape[axis]
    means = np.empty(n)
    for t in range(n):
        means[t] = float(np.mean(np.take(data, t, axis=axis), dtype=np.float64)) - background
    info: dict = {'means': means}
    if method == 'ratio':
        factors = means[0] / np.where(means > 0, means, np.nan)
    elif method == 'exponential':
        from scipy.optimize import curve_fit

        t = np.arange(n, dtype=float)

        def model(t, a, b, tau):
            return a + b * np.exp(-t / tau)

        p0 = (means[-1], means[0] - means[-1], max(n / 3, 1.0))
        try:
            (a, b, tau), _ = curve_fit(model, t, means, p0=p0, maxfev=20000)
        except Exception as exc:
            raise ValueError(f'exponential fit failed ({exc}); try the ratio method') from exc
        fit = model(t, a, b, tau)
        drop = (fit[0] - fit[-1]) / fit[0] if fit[0] else 0.0
        ss_tot = float(np.sum((means - means.mean()) ** 2))
        r2 = 1 - float(np.sum((means - fit) ** 2)) / ss_tot if ss_tot > 0 else 0.0
        if not (b > 0 and 0 < tau < 20 * n and drop > 0.005 and r2 >= 0.5):
            raise ValueError(
                'no bleaching decay found: an exponential decay does not describe the frame means '
                f'(fit explains {100 * max(r2, 0):.0f}% of their variance, τ = {tau:.3g} frames, '
                f'change {100 * -drop:+.2g}%). Nothing was corrected. Check the mean-intensity plot; '
                'dips or rises may be real signal or motion.')
        info['r2'] = r2
        factors = fit[0] / fit
        info |= {'a': a, 'b': b, 'tau_frames': tau, 'fit': fit}
    else:
        raise ValueError(f'unknown bleach method {method!r}')
    factors = np.nan_to_num(factors, nan=1.0)
    out = np.empty(data.shape, dtype=np.float32)
    for t in range(n):
        idx = [slice(None)] * data.ndim
        idx[axis] = t
        frame = np.asarray(data[tuple(idx)], dtype=np.float32) - np.float32(background)
        out[tuple(idx)] = frame * np.float32(factors[t]) + np.float32(background)
        if progress is not None and progress(t + 1, n) is False:
            break
    info['factors'] = factors
    return out, info


# --------------------------------------------------------------- kymograph

def line_samples(verts: np.ndarray, width: int = 1) -> np.ndarray:
    """Sample points (width, n, 2) every pixel along a polyline, with ``width``
    parallel offsets one pixel apart (for averaging across the line)."""
    verts = np.asarray(verts, dtype=float)[:, -2:]
    pts, normals = [], []
    for p, q in zip(verts[:-1], verts[1:]):
        seg = q - p
        length = float(np.hypot(*seg))
        if length == 0:
            continue
        n = max(int(np.ceil(length)), 1)
        t = np.linspace(0, 1, n, endpoint=False)[:, None]
        pts.append(p + t * seg)
        normal = np.array([-seg[1], seg[0]]) / length
        normals.append(np.repeat(normal[None], n, axis=0))
    pts.append(verts[-1][None])
    normals.append(normals[-1][-1:] if normals else np.array([[0.0, 1.0]]))
    pts, normals = np.concatenate(pts), np.concatenate(normals)
    offsets = np.arange(width) - (width - 1) / 2
    return pts[None] + offsets[:, None, None] * normals[None]


def kymograph(data, axis: int, index: tuple, verts: np.ndarray, width: int = 1,
              chunk: int = 256, progress=None) -> np.ndarray:
    """Intensity along a line for every position of ``axis``: (n_axis, n_points).

    Bilinear sampling, averaged across ``width`` pixels. Only the line's
    bounding box is read from disk.
    """
    from scipy.ndimage import map_coordinates

    samples = line_samples(verts, width)  # (w, n, 2) in y, x
    h, w_img = data.shape[-2:]
    lo = np.clip(np.floor(samples.reshape(-1, 2).min(0)).astype(int) - 1, 0, None)
    hi = np.minimum(np.ceil(samples.reshape(-1, 2).max(0)).astype(int) + 2, (h, w_img))
    ys, xs = slice(lo[0], hi[0]), slice(lo[1], hi[1])
    local = samples - lo
    coords = local.reshape(-1, 2).T  # (2, w*n)
    n_t = data.shape[axis]
    out = np.empty((n_t, samples.shape[1]), dtype=np.float32)
    for t0 in range(0, n_t, chunk):
        t1 = min(t0 + chunk, n_t)
        sl = tuple(slice(t0, t1) if d == axis else int(index[d]) for d in range(data.ndim - 2))
        block = np.asarray(data[sl + (ys, xs)], dtype=np.float32)
        for k in range(t1 - t0):
            vals = map_coordinates(block[k], coords, order=1, mode='nearest')
            out[t0 + k] = vals.reshape(samples.shape[:2]).mean(0)
        if progress is not None and progress(t1, n_t) is False:
            break
    return out


# -------------------------------------------------------- image calculator

CALC_OPS = {
    'Add': np.add,
    'Subtract': np.subtract,
    'Multiply': np.multiply,
    'Divide': np.divide,
    'Min': np.minimum,
    'Max': np.maximum,
    'Average': lambda a, b: (a + b) / 2,
    'Difference': lambda a, b: np.abs(a - b),
}


def image_calculator(a, b, op: str, progress=None) -> np.ndarray:
    """``a <op> b`` as float32, plane by plane. ``b`` may have fewer leading axes
    than ``a`` (e.g. one flat-field frame applied to a whole movie)."""
    if op not in CALC_OPS:
        raise ValueError(f'unknown operation {op!r}')
    if a.shape[-b.ndim:] != b.shape:
        raise ValueError(f'shapes {a.shape} and {b.shape} do not match')
    func = CALC_OPS[op]
    out = np.empty(a.shape, dtype=np.float32)
    lead = a.shape[:-2]
    planes = list(np.ndindex(*lead)) if lead else [()]
    nb = b.ndim - 2
    for n, idx in enumerate(planes):
        pa = np.asarray(a[idx], dtype=np.float32)
        pb = np.asarray(b[idx[len(idx) - nb:] if nb else ()], dtype=np.float32)
        with np.errstate(divide='ignore', invalid='ignore'):
            r = func(pa, pb)
        if op == 'Divide':
            r[~np.isfinite(r)] = np.nan
        out[idx] = r
        if progress is not None and progress(n + 1, len(planes)) is False:
            break
    return out


def histogram_stats(values: np.ndarray, bins: int = 256) -> dict:
    """Histogram plus ImageJ's summary statistics (n, mean, std, min, max, mode)."""
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {'n': 0}
    lo, hi = float(v.min()), float(v.max())
    counts, edges = np.histogram(v, bins=bins, range=(lo, hi if hi > lo else lo + 1))
    mode = (edges[np.argmax(counts)] + edges[np.argmax(counts) + 1]) / 2
    return {'n': int(v.size), 'mean': float(v.mean()), 'std': float(v.std(ddof=1)) if v.size > 1 else 0.0,
            'min': lo, 'max': hi, 'mode': float(mode), 'counts': counts, 'edges': edges}
