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


MEASUREMENT_KEYS = [
    'Area',
    'Mean',
    'StdDev',
    'Min',
    'Max',
    'IntDen',
    'RawIntDen',
    'Median',
    'Centroid',
    'Perimeter',
    'Circularity',
    'Feret',
    'AR',
    'Solidity',
]
DEFAULT_MEASUREMENTS = {'Area', 'Mean', 'StdDev', 'Min', 'Max', 'IntDen', 'RawIntDen'}


def mask_stats(
    plane: np.ndarray,
    mask: np.ndarray | None,
    pixel_area: float = 1.0,
    scale: tuple[float, float] | None = None,
    include: set[str] | list[str] | None = None,
) -> dict:
    """ImageJ-style measurements of ``plane`` inside ``mask`` (whole plane if None)."""
    wanted = DEFAULT_MEASUREMENTS if include is None else set(include)
    values = plane[mask] if mask is not None else plane.ravel()
    values = np.asarray(values, dtype=np.float64)
    sy, sx = (float(scale[0]), float(scale[1])) if scale is not None else (float(np.sqrt(pixel_area)), float(np.sqrt(pixel_area)))

    if values.size == 0:
        nan = float('nan')
        row: dict = {}
        if 'Area' in wanted:
            row['Area'] = 0.0
        for key in ('Mean', 'StdDev', 'Min', 'Max', 'Median', 'IntDen', 'RawIntDen'):
            if key in wanted:
                row[key] = nan
        if 'Centroid' in wanted:
            row['X'] = nan
            row['Y'] = nan
        if 'Perimeter' in wanted:
            row['Perim.'] = nan
        if 'Circularity' in wanted:
            row['Circ.'] = nan
        if 'Feret' in wanted:
            row['Feret'] = nan
        if 'AR' in wanted:
            row['AR'] = nan
        if 'Solidity' in wanted:
            row['Solidity'] = nan
        return row

    area = values.size * pixel_area
    mean = float(values.mean())
    row = {}
    if 'Area' in wanted:
        row['Area'] = area
    if 'Mean' in wanted:
        row['Mean'] = mean
    if 'StdDev' in wanted:
        row['StdDev'] = float(values.std(ddof=1)) if values.size > 1 else 0.0
    if 'Min' in wanted:
        row['Min'] = float(values.min())
    if 'Max' in wanted:
        row['Max'] = float(values.max())
    if 'Median' in wanted:
        row['Median'] = float(np.median(values))
    if 'IntDen' in wanted:
        row['IntDen'] = area * mean
    if 'RawIntDen' in wanted:
        row['RawIntDen'] = float(values.sum())

    shape_keys = {'Centroid', 'Perimeter', 'Circularity', 'Feret', 'AR', 'Solidity'}
    if wanted & shape_keys:
        from skimage.measure import regionprops

        bin_mask = np.ones(plane.shape[:2], dtype=np.uint8) if mask is None else mask.astype(np.uint8)
        props = regionprops(bin_mask)
        if props:
            reg = props[0]
            area_px = float(reg.area)
            perim_px = float(reg.perimeter)
            cy, cx = reg.centroid
            if 'Centroid' in wanted:
                row['X'] = float(cx * sx)
                row['Y'] = float(cy * sy)
            if 'Perimeter' in wanted:
                row['Perim.'] = float(perim_px * sx)
            if 'Circularity' in wanted:
                row['Circ.'] = float(min(1.0, 4 * np.pi * area_px / perim_px**2)) if perim_px > 0 else 1.0
            if 'Feret' in wanted:
                row['Feret'] = float(reg.feret_diameter_max * sx)
            if 'AR' in wanted:
                major, minor = reg.axis_major_length, reg.axis_minor_length
                row['AR'] = float(major / minor) if minor > 0 else float('nan')
            if 'Solidity' in wanted:
                row['Solidity'] = float(reg.solidity)
    return row


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


def dff(traces: np.ndarray, offset: float = 0.0) -> np.ndarray:
    """ΔF/F with F0 = each trace's median, after subtracting ``offset``.

    Traces whose F0 is not positive become NaN: dividing by a negative F0
    would silently flip transients upside down.
    """
    f = np.asarray(traces, dtype=np.float64) - offset
    f0 = np.nanmedian(f, axis=-1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(f0 > 0, (f - f0) / f0, np.nan)


def suggested_offset(values) -> float:
    """0 for positive data; the minimum when data dips below zero (a dark offset)."""
    low = float(np.nanmin(values))
    return low if low < 0 else 0.0


def correlation_image(data, axis: int, index: tuple, progress=None) -> np.ndarray:
    """Local correlation image (as in CaImAn): each pixel's mean temporal
    correlation with its 8 neighbours along ``axis``.

    Active cells stand out regardless of brightness, so uneven illumination
    and neuropil haze matter much less than in a mean or SD projection.
    Computed in row chunks with a one-row halo, so movies need not fit in RAM.
    """
    lead = data.ndim - 2
    axis = axis % data.ndim
    h, w = data.shape[-2:]
    n_t = data.shape[axis]
    out = np.zeros((h, w), np.float64)
    rows = max(1, min(h, CHUNK_BYTES // max(1, n_t * w * 4)))
    offsets = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
    for r0 in range(0, h, rows):
        r1 = min(h, r0 + rows)
        a0, a1 = max(0, r0 - 1), min(h, r1 + 1)
        sl = tuple(slice(None) if d == axis else int(index[d]) for d in range(lead))
        # only ``axis`` is a slice among the leading axes, so time comes first
        block = np.asarray(data[sl + (slice(a0, a1), slice(None))], dtype=np.float32)
        block = block - block.mean(axis=0, dtype=np.float64).astype(np.float32)
        sd = block.std(axis=0)
        block /= np.where(sd > 0, sd, np.inf)
        core = slice(r0 - a0, r0 - a0 + (r1 - r0))
        acc = np.zeros((r1 - r0, w))
        count = np.zeros((r1 - r0, w))
        ys = np.arange(r0, r1)
        for dy, dx in offsets:
            yy = ys + dy
            valid_y = (yy >= 0) & (yy < h)
            if not valid_y.any():
                continue
            src_rows = np.arange(core.start, core.stop)[valid_y]
            nb_rows = src_rows + dy
            x0, x1 = max(0, -dx), min(w, w - dx)
            a = block[:, src_rows, x0:x1]
            b = block[:, nb_rows, x0 + dx:x1 + dx]
            acc[valid_y, x0:x1] += np.einsum('tij,tij->ij', a, b) / n_t
            count[valid_y, x0:x1] += 1
        out[r0:r1] = acc / np.maximum(count, 1)
        if progress is not None and progress(r1, h) is False:
            break
    return out.astype(np.float32)


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


# -------------------------------------------------------- substacks & binning

def parse_slice_spec(spec: str, size: int) -> slice | np.ndarray:
    """Parse a 1-based inclusive ImageJ slice specification for an axis of length ``size``.

    Supports empty / ``'all'`` (full range), single index (``'5'``), range
    (``'2-10'``), stepped range (``'1-100-2'``), and comma-separated lists
    (``'1, 4, 7-9'``). Contiguous/stepped ranges return a ``slice`` so
    memory-mapped files stay zero-copy; single values and lists return a 0-based
    integer ``np.ndarray``.
    """
    if size < 1:
        raise ValueError('axis has length 0')
    text = (spec or '').strip()
    if not text or text.lower() == 'all':
        return slice(None)

    import re

    parts = [p.strip() for p in text.split(',')]
    if any(not p for p in parts):
        raise ValueError(f'invalid slice range {spec!r}')

    if len(parts) == 1 and '-' in parts[0]:
        part = parts[0]
        if not re.fullmatch(r'\d+(?:-\d+(?:-\d+)?)?', part):
            raise ValueError(f'invalid slice range {spec!r}')
        nums = [int(x) for x in part.split('-')]
        a, b = nums[0], nums[1]
        step = nums[2] if len(nums) == 3 else 1
        if step < 1 or a < 1 or b < a or b > size:
            raise ValueError(f'slice range {part!r} out of bounds for length {size} (1–{size})')
        if a == 1 and b == size and step == 1:
            return slice(None)
        return slice(a - 1, b, None if step == 1 else step)

    indices: list[int] = []
    for part in parts:
        if not re.fullmatch(r'\d+(?:-\d+(?:-\d+)?)?', part):
            raise ValueError(f'invalid slice range {part!r}')
        nums = [int(x) for x in part.split('-')]
        if len(nums) == 1:
            k = nums[0]
            if k < 1 or k > size:
                raise ValueError(f'slice {k} out of bounds for length {size} (1–{size})')
            indices.append(k - 1)
        else:
            a, b = nums[0], nums[1]
            step = nums[2] if len(nums) == 3 else 1
            if step < 1 or a < 1 or b < a or b > size:
                raise ValueError(f'slice range {part!r} out of bounds for length {size} (1–{size})')
            indices.extend(range(a - 1, b, step))
    return np.asarray(indices, dtype=int)


def _is_lazy_backing(data) -> bool:
    if isinstance(data, np.memmap) or hasattr(data, 'compute'):
        return True
    return isinstance(getattr(data, 'base', None), np.memmap)


def substack(
    data,
    specs: dict[int, str | int | slice | np.ndarray] | None = None,
    box: tuple[int, int, int, int] | None = None,
    squeeze_single: bool = True,
    rgb: bool = False,
) -> tuple[np.ndarray, list[int], dict[int, int]]:
    """Extract a cropped and/or substack view or array from ``data``.

    Returns ``(result, dropped_axes, step_multipliers)`` where ``dropped_axes``
    are original leading axes reduced to a single index and
    ``step_multipliers`` maps original axes with ``step > 1`` to their stride.
    """
    import itertools

    spatial_end = data.ndim - 1 if rgb and data.ndim >= 3 else data.ndim
    n_lead = max(0, spatial_end - 2)
    indexer: list = [slice(None)] * data.ndim
    dropped_axes: list[int] = []
    step_multipliers: dict[int, int] = {}

    for d, raw in (specs or {}).items():
        if not (0 <= d < n_lead):
            raise ValueError(f'axis {d} is not a stack axis')
        size = data.shape[d]
        if isinstance(raw, str):
            idx = parse_slice_spec(raw, size)
        elif isinstance(raw, (int, np.integer)):
            k = int(raw)
            if k < 0 or k >= size:
                raise ValueError(f'index {k} out of bounds for axis {d} of length {size}')
            idx = np.array([k], dtype=int)
        elif isinstance(raw, slice):
            idx = raw
        else:
            idx = np.asarray(raw, dtype=int)
            if idx.size == 0 or (idx < 0).any() or (idx >= size).any():
                raise ValueError(f'indices out of bounds for axis {d} of length {size}')

        if isinstance(idx, np.ndarray) and idx.size == 1 and squeeze_single:
            indexer[d] = int(idx.item())
            dropped_axes.append(d)
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(size)
            count = len(range(start, stop, step))
            if count == 0:
                raise ValueError(f'empty selection on axis {d}')
            if count == 1 and squeeze_single:
                indexer[d] = start
                dropped_axes.append(d)
            else:
                indexer[d] = idx
                if step > 1:
                    step_multipliers[d] = step
        else:
            indexer[d] = idx

    if box is not None:
        h, w = data.shape[spatial_end - 2], data.shape[spatial_end - 1]
        y0, y1, x0, x1 = box
        y0, y1 = max(0, int(y0)), min(h, int(y1))
        x0, x1 = max(0, int(x0)), min(w, int(x1))
        if y1 <= y0 or x1 <= x0:
            raise ValueError('crop selection is empty')
        indexer[spatial_end - 2] = slice(y0, y1)
        indexer[spatial_end - 1] = slice(x0, x1)

    has_fancy = any(isinstance(indexer[d], np.ndarray) for d in range(n_lead))
    if not has_fancy:
        view = data[tuple(indexer)]
        result = view if _is_lazy_backing(data) else np.array(view, copy=True)
        return result, sorted(dropped_axes), step_multipliers

    lead_src: list[list[int]] = []
    out_lead_shape: list[int] = []
    for d in range(n_lead):
        ix = indexer[d]
        if isinstance(ix, int):
            lead_src.append([ix])
        elif isinstance(ix, slice):
            vals = list(range(*ix.indices(data.shape[d])))
            lead_src.append(vals)
            out_lead_shape.append(len(vals))
        else:
            vals = [int(v) for v in ix.tolist()]
            lead_src.append(vals)
            out_lead_shape.append(len(vals))

    sy = indexer[spatial_end - 2]
    sx = indexer[spatial_end - 1]
    ry = len(range(*sy.indices(data.shape[spatial_end - 2])))
    rx = len(range(*sx.indices(data.shape[spatial_end - 1])))
    tail_shape = (ry, rx) + ((data.shape[-1],) if rgb and data.ndim >= 3 else ())
    out = np.empty(tuple(out_lead_shape) + tail_shape, dtype=data.dtype)
    tail_slices = (sy, sx) + ((slice(None),) if rgb and data.ndim >= 3 else ())

    out_ranges = [range(n) for n in out_lead_shape]
    for src_idx, dst_idx in zip(itertools.product(*lead_src), itertools.product(*out_ranges)):
        out[dst_idx] = np.asarray(data[src_idx + tail_slices])
    return out, sorted(dropped_axes), step_multipliers


BIN_METHODS = {
    'Average': 'mean',
    'Max': 'max',
    'Min': 'min',
    'Median': 'median',
    'Sum': 'sum',
}


def _bin_block(block: np.ndarray, factors: tuple[int, ...], method: str, out_dtype) -> np.ndarray:
    interleaved: list[int] = []
    reduce_axes: list[int] = []
    for s, f in zip(block.shape, factors):
        if f > 1:
            interleaved.extend([s // f, f])
            reduce_axes.append(len(interleaved) - 1)
        else:
            interleaved.append(s)
    if not reduce_axes:
        return np.asarray(block, dtype=out_dtype).copy()
    work = block if method in ('max', 'min') else block.astype(
        np.float64 if method in ('mean', 'sum') else np.float32, copy=False
    )
    fn = {'mean': np.mean, 'max': np.max, 'min': np.min, 'median': np.median, 'sum': np.sum}[method]
    reduced = fn(work.reshape(interleaved), axis=tuple(reduce_axes))
    return np.asarray(reduced, dtype=out_dtype)


def bin_image(data, factors: tuple[int, ...], method: str = 'mean', progress=None) -> np.ndarray:
    """Downsample ``data`` by integer ``factors`` along each axis.

    Remainder pixels/frames that do not fill a full bin are trimmed at the end.
    Works chunk-wise on memory-mapped or dask arrays.
    """
    import itertools

    method = BIN_METHODS.get(method, method)
    if method not in ('mean', 'max', 'min', 'median', 'sum'):
        raise ValueError(f'unknown bin method {method!r}')
    factors = tuple(int(f) for f in factors)
    if len(factors) != data.ndim:
        raise ValueError(f'expected {data.ndim} bin factors, got {len(factors)}')
    if any(f < 1 for f in factors):
        raise ValueError('bin factors must be >= 1')

    trim = tuple((s // f) * f for s, f in zip(data.shape, factors))
    out_shape = tuple(s // f for s, f in zip(data.shape, factors))
    if any(n < 1 for n in out_shape):
        raise ValueError('bin factor exceeds image dimension')

    out_dtype = data.dtype if method in ('max', 'min') else np.float32
    if all(f == 1 for f in factors):
        return np.asarray(data, dtype=out_dtype).copy()

    out = np.empty(out_shape, dtype=out_dtype)
    work_bytes = max(int(np.dtype(data.dtype).itemsize), 8 if method in ('mean', 'sum') else 4)

    if data.ndim == 2:
        fy, fx = factors
        row_bins = max(1, CHUNK_BYTES // max(1, fy * trim[1] * work_bytes))
        total = out_shape[0]
        for b0 in range(0, out_shape[0], row_bins):
            b1 = min(out_shape[0], b0 + row_bins)
            chunk = np.asarray(data[b0 * fy : b1 * fy, : trim[1]])
            out[b0:b1] = _bin_block(chunk, factors, method, out_dtype)
            if progress is not None and progress(b1, total) is False:
                break
        return out

    lead = data.ndim - 2
    tail_trim = tuple(slice(0, t) for t in trim[lead:])
    slab_bytes = int(np.prod([factors[d] for d in range(lead)] + list(trim[lead:]))) * work_bytes

    if lead == 1 or (slab_bytes <= CHUNK_BYTES and int(np.prod(out_shape[1:lead])) == 1):
        # Single non-trivial leading axis: batch consecutive bins under CHUNK_BYTES,
        # or fall back to row chunks if even one bin exceeds CHUNK_BYTES.
        f0 = factors[0]
        per_bin = int(np.prod(trim[1:])) * f0 * work_bytes
        if per_bin <= CHUNK_BYTES:
            batch = max(1, CHUNK_BYTES // max(1, per_bin))
            fixed = (slice(0, trim[d]) for d in range(1, lead))
            fixed_t = tuple(fixed)
            for b0 in range(0, out_shape[0], batch):
                b1 = min(out_shape[0], b0 + batch)
                block = np.asarray(data[(slice(b0 * f0, b1 * f0),) + fixed_t + tail_trim])
                out[b0:b1] = _bin_block(block, factors, method, out_dtype)
                if progress is not None and progress(b1, out_shape[0]) is False:
                    break
            return out

    # General nD streaming: one output-bin tuple along leading axes at a time,
    # chunked along the first spatial axis when needed.
    fy = factors[lead]
    combos = list(itertools.product(*(range(out_shape[d]) for d in range(lead))))
    bin_lead_bytes = int(np.prod([factors[d] for d in range(lead)])) * int(np.prod(trim[lead + 1 :])) * fy * work_bytes
    row_bins = max(1, CHUNK_BYTES // max(1, bin_lead_bytes))
    y_chunks = list(range(0, out_shape[lead], row_bins))
    total = len(combos) * len(y_chunks)
    done = 0
    for idx in combos:
        lead_sl = tuple(slice(k * factors[d], (k + 1) * factors[d]) for d, k in enumerate(idx))
        for yb0 in y_chunks:
            yb1 = min(out_shape[lead], yb0 + row_bins)
            y_sl = slice(yb0 * fy, yb1 * fy)
            rest_sl = tuple(slice(0, trim[d]) for d in range(lead + 1, data.ndim))
            block = np.asarray(data[lead_sl + (y_sl,) + rest_sl])
            binned = _bin_block(block, factors, method, out_dtype)
            out[idx + (slice(yb0, yb1),)] = binned[(0,) * len(idx) + (slice(None),)]
            done += 1
            if progress is not None and progress(done, total) is False:
                return out
    return out


# ---------------------------------------------------------- motion correction

def _prep_for_correlation(img: np.ndarray, hann: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    if np.isnan(arr).any():
        fill = float(np.nanmean(arr))
        arr = np.where(np.isfinite(arr), arr, 0.0 if np.isnan(fill) else fill)
    return (arr - float(arr.mean())) * hann


def estimate_shifts(
    data,
    axis: int = 0,
    template: str | np.ndarray = 'median',
    n_template: int = 200,
    ref_index: int = 0,
    upsample: int = 10,
    max_shift: float | None = 32.0,
    index: tuple | None = None,
    progress=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate subpixel rigid ``(dy, dx)`` shifts along ``axis`` via phase cross-correlation."""
    from skimage.filters import window
    from skimage.registration import phase_cross_correlation

    axis = axis % data.ndim
    if axis >= data.ndim - 2:
        raise ValueError('motion correction needs a stack axis, not Y or X')
    n_t = data.shape[axis]
    h, w = data.shape[-2:]
    lead = data.ndim - 2
    idx = (0,) * data.ndim if index is None else index

    def frame2d(t: int) -> np.ndarray:
        sl = tuple(t if d == axis else int(idx[d]) for d in range(lead))
        return np.asarray(data[sl], dtype=np.float32)

    if isinstance(template, np.ndarray):
        ref_img = np.asarray(template, dtype=np.float32)
        if ref_img.shape != (h, w):
            raise ValueError(f'template shape {ref_img.shape} does not match frame shape {(h, w)}')
    elif template == 'median':
        picks = np.unique(np.linspace(0, n_t - 1, min(max(1, int(n_template)), n_t)).astype(int))
        ref_img = np.median(np.stack([frame2d(int(t)) for t in picks]), axis=0).astype(np.float32)
    elif template == 'first_mean':
        k = min(max(1, int(n_template)), n_t)
        acc = np.zeros((h, w), dtype=np.float64)
        for t in range(k):
            acc += frame2d(t)
        ref_img = (acc / k).astype(np.float32)
    elif template == 'current':
        ref_img = frame2d(int(np.clip(ref_index, 0, n_t - 1)))
    else:
        raise ValueError(f'unknown template mode {template!r}')

    hann = window('hann', (h, w)).astype(np.float32)
    ref_prep = _prep_for_correlation(ref_img, hann)
    shifts = np.zeros((n_t, 2), dtype=np.float64)
    up = max(1, int(upsample))
    for t in range(n_t):
        s, _, _ = phase_cross_correlation(
            ref_prep,
            _prep_for_correlation(frame2d(t), hann),
            upsample_factor=up,
            normalization=None,
        )
        if max_shift is not None and max_shift > 0:
            s = np.clip(s, -float(max_shift), float(max_shift))
        shifts[t] = s
        if progress is not None and progress(t + 1, n_t) is False:
            break
    return shifts, ref_img


def apply_shifts(
    data,
    shifts: np.ndarray,
    axis: int = 0,
    order: int = 1,
    mode: str = 'nearest',
    progress=None,
) -> np.ndarray:
    """Apply ``(n_t, 2)`` ``(dy, dx)`` shifts to every 2D plane along ``axis``."""
    import itertools

    from scipy.ndimage import shift as nd_shift

    axis = axis % data.ndim
    if axis >= data.ndim - 2:
        raise ValueError('shifts must be applied along a stack axis')
    shifts = np.asarray(shifts, dtype=np.float64)
    n_t = data.shape[axis]
    if shifts.shape != (n_t, 2):
        raise ValueError(f'shifts shape {shifts.shape} must be ({n_t}, 2)')

    out_dtype = np.float32 if order > 0 else data.dtype
    out = np.empty(data.shape, dtype=out_dtype)
    lead = data.ndim - 2
    other_axes = [d for d in range(lead) if d != axis]
    other_ranges = [range(data.shape[d]) for d in other_axes]
    other_combos = list(itertools.product(*other_ranges)) if other_axes else [()]

    for t in range(n_t):
        dy, dx = shifts[t]
        for combo in other_combos:
            pos = dict(zip(other_axes, combo))
            sl = tuple(t if d == axis else pos[d] for d in range(lead))
            plane = np.asarray(data[sl], dtype=np.float32 if order > 0 else data.dtype)
            if abs(dy) < 1e-6 and abs(dx) < 1e-6:
                out[sl] = plane
            else:
                out[sl] = nd_shift(plane, (dy, dx), order=order, mode=mode)
        if progress is not None and progress(t + 1, n_t) is False:
            break
    return out


def motion_correct(
    data,
    axis: int = 0,
    template: str | np.ndarray = 'median',
    n_template: int = 200,
    ref_index: int = 0,
    upsample: int = 10,
    max_shift: float | None = 32.0,
    order: int = 1,
    index: tuple | None = None,
    progress=None,
) -> tuple[np.ndarray, dict]:
    """Rigid motion correction along ``axis``: returns ``(corrected, info)``."""
    axis = axis % data.ndim
    n_t = data.shape[axis]
    total = 2 * n_t
    cancelled = {'flag': False}

    def p1(done, _tot):
        if progress is not None and progress(done, total) is False:
            cancelled['flag'] = True
            return False
        return True

    def p2(done, _tot):
        if progress is not None and progress(n_t + done, total) is False:
            cancelled['flag'] = True
            return False
        return True

    shifts, ref_img = estimate_shifts(
        data,
        axis=axis,
        template=template,
        n_template=n_template,
        ref_index=ref_index,
        upsample=upsample,
        max_shift=max_shift,
        index=index,
        progress=p1,
    )
    if cancelled['flag']:
        return np.empty(0, dtype=np.float32), {'shifts': shifts, 'template': ref_img, 'cancelled': True}
    corrected = apply_shifts(data, shifts, axis=axis, order=order, progress=p2)
    return corrected, {'shifts': shifts, 'template': ref_img, 'cancelled': cancelled['flag']}


# ---------------------------------------------------- labels & watershed

def fill_holes_labels(labels: np.ndarray) -> np.ndarray:
    """Fill interior background holes in a 2D or 3D mask or label image."""
    from scipy.ndimage import binary_fill_holes
    from skimage.measure import regionprops

    arr = np.asarray(labels)
    if arr.size == 0 or arr.max() == 0:
        return arr.copy()
    if arr.max() <= 1 and arr.min() >= 0:
        filled = binary_fill_holes(arr > 0)
        return filled.astype(arr.dtype if np.issubdtype(arr.dtype, np.integer) else np.uint8)
    out = arr.astype(np.int32, copy=True)
    for region in regionprops(out):
        filled = binary_fill_holes(region.image)
        view = out[region.slice]
        view[filled & (view == 0)] = int(region.label)
    return out


def watershed_labels(
    labels: np.ndarray,
    min_distance: float = 5.0,
    z_ratio: float = 1.0,
    fill_holes: bool = True,
) -> np.ndarray:
    """Split touching 2D or 3D objects using distance-transform watershed."""
    from scipy.ndimage import binary_fill_holes, distance_transform_edt, gaussian_filter
    from skimage.feature import peak_local_max
    from skimage.measure import label
    from skimage.segmentation import watershed

    arr = np.asarray(labels)
    fg = arr > 0
    if not fg.any():
        return np.zeros(arr.shape, dtype=np.int32)
    if fill_holes:
        fg = binary_fill_holes(fg)

    z_ratio = max(float(z_ratio), 1e-6)
    sampling = (1.0 / z_ratio, 1.0, 1.0) if fg.ndim == 3 else (1.0, 1.0)
    dist = distance_transform_edt(fg, sampling=sampling)
    sigma = (0.5 * z_ratio, 0.5, 0.5) if fg.ndim == 3 else 0.5
    dist_smooth = np.where(fg, gaussian_filter(dist, sigma=sigma), 0.0)

    r = max(1, int(round(min_distance)))
    if fg.ndim == 3:
        rz = max(0, int(round(min_distance * z_ratio)))
        zz, yy, xx = np.ogrid[-rz : rz + 1, -r : r + 1, -r : r + 1]
        footprint = (xx / r) ** 2 + (yy / r) ** 2 + ((zz / rz) ** 2 if rz else zz.astype(float) * 0) <= 1
    else:
        yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
        footprint = (xx / r) ** 2 + (yy / r) ** 2 <= 1

    conn = label(fg, connectivity=fg.ndim).astype(np.int32)
    coords = peak_local_max(dist_smooth, footprint=footprint, labels=conn, exclude_border=False)
    markers = np.zeros(fg.shape, dtype=np.int32)
    seen_components: set[int] = set()
    next_id = 1
    for pt in coords:
        idx = tuple(int(c) for c in pt)
        markers[idx] = next_id
        seen_components.add(int(conn[idx]))
        next_id += 1

    # Guarantee every connected component gets at least one marker so small objects are never dropped.
    n_comp = int(conn.max())
    if len(seen_components) < n_comp:
        from skimage.measure import regionprops

        for region in regionprops(conn):
            cid = int(region.label)
            if cid in seen_components:
                continue
            sub_mask = region.image
            sub_dist = np.where(sub_mask, dist_smooth[region.slice], -1.0)
            local = np.unravel_index(int(np.argmax(sub_dist)), sub_dist.shape)
            global_idx = tuple(sl.start + offset for sl, offset in zip(region.slice, local))
            markers[global_idx] = next_id
            next_id += 1

    return watershed(-dist, markers, mask=fg).astype(np.int32)


def watershed_blocks(
    data,
    blocks: list[tuple],
    min_distance: float = 5.0,
    z_ratio: float = 1.0,
    fill_holes: bool = True,
    progress=None,
) -> np.ndarray:
    """Apply distance-transform watershed block by block with globally unique label IDs."""
    out = np.zeros(data.shape, dtype=np.int32)
    next_id = 0
    for n, b in enumerate(blocks):
        result = watershed_labels(np.asarray(data[b]), min_distance=min_distance, z_ratio=z_ratio, fill_holes=fill_holes)
        if result.max():
            result[result > 0] += next_id
            next_id = int(result.max())
        out[b] = result
        if progress is not None and progress(n + 1, len(blocks)) is False:
            break
    return out


# -------------------------------------------- line profiles, selections, wand

def line_profile(
    plane: np.ndarray,
    verts: np.ndarray,
    width: int = 1,
    scale: tuple[float, float] = (1.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Intensity profile along a straight line or polyline, averaged across ``width`` px.

    Returns ``(distances, values)`` where ``distances`` uses ``scale = (sy, sx)``.
    """
    from scipy.ndimage import map_coordinates

    width = max(1, int(width))
    samples = line_samples(verts, width)  # (width, n, 2)
    coords = samples.reshape(-1, 2).T
    vals = map_coordinates(np.asarray(plane, dtype=np.float32), coords, order=1, mode='nearest')
    profile = vals.reshape(samples.shape[:2]).mean(axis=0)
    center = samples[width // 2] if width % 2 == 1 else samples.mean(axis=0)
    diffs = np.diff(center, axis=0) * np.asarray(scale, dtype=float)
    dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(diffs, axis=1))])
    return dist, profile


def wand_mask(plane: np.ndarray, y: int, x: int, tolerance: float = 0.0) -> np.ndarray:
    """8-connected flood selection starting at ``(y, x)`` within ``tolerance``."""
    from skimage.segmentation import flood

    arr = np.asarray(plane, dtype=np.float64)
    h, w = arr.shape[:2]
    if not (0 <= y < h and 0 <= x < w):
        return np.zeros((h, w), dtype=bool)
    return flood(arr, (int(y), int(x)), tolerance=max(0.0, float(tolerance)), connectivity=2)


def mask_to_polygon(mask: np.ndarray, tolerance: float = 0.5) -> np.ndarray | None:
    """Outer boundary polygon ``(N, 2)`` in ``(y, x)`` coordinates of the largest component."""
    from skimage.measure import approximate_polygon, find_contours

    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return None
    contours = find_contours(np.pad(m, 1).astype(np.uint8), 0.5)
    if not contours:
        return None
    contour = max(contours, key=len) - 1.0
    if tolerance > 0 and len(contour) > 8:
        contour = approximate_polygon(contour, tolerance)
    if contour.shape[0] > 1 and np.allclose(contour[0], contour[-1]):
        contour = contour[:-1]
    return contour if len(contour) >= 3 else None


def enlarge_mask(mask: np.ndarray, pixels: float) -> np.ndarray:
    """Dilate (``pixels > 0``) or erode (``pixels < 0``) a 2D boolean mask by Euclidean distance."""
    from scipy.ndimage import distance_transform_edt

    m = np.asarray(mask, dtype=bool)
    pixels = float(pixels)
    if pixels == 0 or not m.any():
        return m.copy()
    if pixels > 0:
        return distance_transform_edt(~m) <= pixels
    # Pad with False so objects touching the image border also erode inward
    dist = distance_transform_edt(np.pad(m, 1, constant_values=False))[1:-1, 1:-1]
    return dist > abs(pixels)


def make_band_mask(mask: np.ndarray, width: float, gap: float = 0.0) -> np.ndarray:
    """Annulus ring around ``mask`` of ``width`` px, separated by ``gap`` px."""
    m = np.asarray(mask, dtype=bool)
    width = max(0.0, float(width))
    gap = max(0.0, float(gap))
    if width == 0 or not m.any():
        return np.zeros_like(m, dtype=bool)
    inner = enlarge_mask(m, gap) if gap > 0 else m
    outer = enlarge_mask(m, gap + width)
    return outer & ~inner


def clear_mask(
    data,
    mask: np.ndarray,
    outside: bool = False,
    fill_value: float = 0.0,
    index: tuple | None = None,
    whole_stack: bool = False,
    rgb: bool = False,
    progress=None,
) -> np.ndarray:
    """Return a copy of ``data`` (or one plane) with pixels inside/outside ``mask`` set to ``fill_value``."""
    target = ~np.asarray(mask, dtype=bool) if outside else np.asarray(mask, dtype=bool)
    fill = np.asarray(fill_value, dtype=data.dtype)
    spatial_end = data.ndim - 1 if rgb and data.ndim >= 3 else data.ndim
    lead_ndim = spatial_end - 2

    if lead_ndim == 0 or not whole_stack:
        sl = tuple(int(index[d]) for d in range(lead_ndim)) if (index is not None and lead_ndim > 0) else ()
        plane = np.asarray(data[sl]).copy()
        plane[target] = fill
        return plane

    out = np.empty(data.shape, dtype=data.dtype)
    lead_shape = data.shape[:lead_ndim]
    combos = list(np.ndindex(*lead_shape))
    for n, idx in enumerate(combos):
        plane = np.asarray(data[idx]).copy()
        plane[target] = fill
        out[idx] = plane
        if progress is not None and progress(n + 1, len(combos)) is False:
            break
    return out


# -------------------------------------------- transforms & type conversion

def flip_image(data, axis: str = 'horizontal', rgb: bool = False) -> np.ndarray:
    """Flip spatial X (``'horizontal'``) or Y (``'vertical'``), returning an array."""
    spatial_end = data.ndim - 1 if rgb and data.ndim >= 3 else data.ndim
    ax = spatial_end - 1 if axis == 'horizontal' else spatial_end - 2
    if axis not in ('horizontal', 'vertical'):
        raise ValueError(f'unknown flip axis {axis!r}')
    return np.ascontiguousarray(np.flip(np.asarray(data), axis=ax))


def rotate_90(data, k: int = 1, rgb: bool = False) -> np.ndarray:
    """Rotate 90° counterclockwise ``k`` times (``k=-1`` = 90° right / clockwise)."""
    spatial_end = data.ndim - 1 if rgb and data.ndim >= 3 else data.ndim
    axes = (spatial_end - 2, spatial_end - 1)
    return np.ascontiguousarray(np.rot90(np.asarray(data), k=k, axes=axes))


def rotate_angle(
    data,
    angle: float,
    resize: bool = False,
    rgb: bool = False,
    progress=None,
) -> np.ndarray:
    """Rotate every 2D spatial plane by ``angle`` degrees (positive = clockwise like ImageJ)."""
    from scipy.ndimage import rotate as nd_rotate

    spatial_end = data.ndim - 1 if rgb and data.ndim >= 3 else data.ndim
    lead_ndim = spatial_end - 2
    # scipy.ndimage.rotate uses CCW for positive angle on (y, x); negate for ImageJ CW convention
    scipy_angle = -float(angle)

    def _rot_plane(plane: np.ndarray) -> np.ndarray:
        if rgb and plane.ndim == 3:
            chans = [nd_rotate(plane[..., c].astype(np.float32), scipy_angle, reshape=resize, order=1)
                     for c in range(plane.shape[-1])]
            return np.stack(chans, axis=-1)
        return nd_rotate(plane.astype(np.float32), scipy_angle, reshape=resize, order=1)

    if lead_ndim == 0:
        return _rot_plane(np.asarray(data))

    lead_shape = data.shape[:lead_ndim]
    combos = list(np.ndindex(*lead_shape))
    first = _rot_plane(np.asarray(data[combos[0]]))
    out = np.empty(lead_shape + first.shape, dtype=np.float32)
    out[combos[0]] = first
    for n, idx in enumerate(combos[1:], start=2):
        out[idx] = _rot_plane(np.asarray(data[idx]))
        if progress is not None and progress(n, len(combos)) is False:
            break
    return out


def convert_type(
    data,
    target_dtype: str,
    scale_range: tuple[float, float] | None = None,
    progress=None,
) -> np.ndarray:
    """Convert ``data`` to ``'uint8'``, ``'uint16'``, or ``'float32'``, plane by plane."""
    dtype_map = {'uint8': np.uint8, 'uint16': np.uint16, 'float32': np.float32}
    if target_dtype not in dtype_map:
        raise ValueError(f'unsupported target dtype {target_dtype!r}')
    dst = np.dtype(dtype_map[target_dtype])

    def _conv(plane: np.ndarray) -> np.ndarray:
        if dst == np.float32:
            return np.asarray(plane, dtype=np.float32)
        hi_dst = float(np.iinfo(dst).max)
        arr = np.asarray(plane, dtype=np.float64)
        if scale_range is not None:
            lo, hi = float(scale_range[0]), float(scale_range[1])
            span = hi - lo if hi > lo else 1.0
            arr = (arr - lo) / span * hi_dst
        return np.clip(np.round(arr), 0.0, hi_dst).astype(dst)

    if data.ndim <= 2:
        return _conv(np.asarray(data))

    lead = data.shape[:-2]
    combos = list(np.ndindex(*lead))
    out = np.empty(data.shape, dtype=dst)
    for n, idx in enumerate(combos):
        out[idx] = _conv(np.asarray(data[idx]))
        if progress is not None and progress(n + 1, len(combos)) is False:
            break
    return out


def composite_rgb(
    layers: list[tuple[object, object, tuple[float, float]]],
    progress=None,
) -> np.ndarray:
    """Flatten ``[(data, colormap, (lo, hi)), ...]`` additively into uint8 RGB ``(..., Y, X, 3)``."""
    if not layers:
        raise ValueError('no layers to composite')
    ref_shape = layers[0][0].shape
    for data, _, _ in layers:
        if data.shape != ref_shape:
            raise ValueError(f'layer shape {data.shape} does not match {ref_shape}')

    lead = ref_shape[:-2]
    combos = list(np.ndindex(*lead)) if lead else [()]
    out = np.empty(ref_shape + (3,), dtype=np.uint8)
    for n, idx in enumerate(combos):
        acc = np.zeros(ref_shape[-2:] + (3,), dtype=np.float32)
        for data, cmap, (lo, hi) in layers:
            plane = np.asarray(data[idx], dtype=np.float32)
            span = float(hi - lo) if hi > lo else 1.0
            norm = np.clip((plane - float(lo)) / span, 0.0, 1.0)
            rgb = np.asarray(cmap.map(norm.ravel()), dtype=np.float32)[:, :3].reshape(ref_shape[-2:] + (3,))
            acc += rgb
        out[idx] = np.clip(np.round(acc * 255.0), 0, 255).astype(np.uint8)
        if progress is not None and progress(n + 1, len(combos)) is False:
            break
    return out
