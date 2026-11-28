"""ROI Manager data: named ROIs, ImageJ .roi/.zip files and Labels conversion.

Coordinates are napari data coordinates of the image plane, ``(y, x)`` with
pixel centres on integers. ImageJ puts pixel *corners* on integers, so
converting adds or removes half a pixel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import roi as roi_mod

IJ_OFFSET = 0.5  # ImageJ coordinate = napari coordinate + 0.5


@dataclass
class Roi:
    name: str
    kind: str  # napari shape type: rectangle, ellipse, polygon, path, line, point
    yx: np.ndarray  # (N, 2) vertices in image-plane data coordinates
    position: tuple[int, ...] | None = None  # stack position, None = every slice
    meta: dict = field(default_factory=dict)

    def mask(self, shape2d: tuple[int, int]) -> np.ndarray:
        if self.kind == 'point':
            m = np.zeros(shape2d, bool)
            for y, x in np.round(self.yx).astype(int):
                if 0 <= y < shape2d[0] and 0 <= x < shape2d[1]:
                    m[y, x] = True
            return m
        return roi_mod.shape_mask(self.yx, self.kind, shape2d)

    @property
    def is_area(self) -> bool:
        return self.kind in ('rectangle', 'ellipse', 'polygon')

    def default_name(self, index: int) -> str:
        """ImageJ-style ``0001-0123-0456`` (index, centre y, centre x)."""
        if self.yx.size == 0:
            return f'{index:04d}-0000-0000'
        cy, cx = self.yx.mean(axis=0)
        if not (np.isfinite(cy) and np.isfinite(cx)):
            return f'{index:04d}-0000-0000'
        return f'{index:04d}-{int(round(cy)):04d}-{int(round(cx)):04d}'


# ------------------------------------------------------------------ ImageJ

def _axis_aligned(corners: np.ndarray) -> bool:
    c = np.asarray(corners)
    return len(c) == 4 and (np.allclose(c[0, 0], c[1, 0]) and np.allclose(c[1, 1], c[2, 1])
                            or np.allclose(c[0, 1], c[1, 1]) and np.allclose(c[1, 0], c[2, 0]))


def to_imagej(roi: Roi):
    from roifile import ROI_OPTIONS, ROI_TYPE, ImagejRoi

    xy = roi.yx[:, ::-1] + IJ_OFFSET
    position = roi.position[0] + 1 if roi.position and len(roi.position) == 1 else None
    if roi.kind in ('rectangle', 'ellipse') and _axis_aligned(roi.yx):
        fx0, fy0 = float(xy[:, 0].min()), float(xy[:, 1].min())
        fx1, fy1 = float(xy[:, 0].max()), float(xy[:, 1].max())
        bounds = np.array([fx0, fy0, fx1, fy1], dtype=float)
        if np.allclose(bounds, np.round(bounds), atol=1e-4):
            x0, y0, x1, y1 = np.round(bounds).astype(int)
            ij = ImagejRoi(
                roitype=ROI_TYPE.RECT if roi.kind == 'rectangle' else ROI_TYPE.OVAL,
                left=int(x0), top=int(y0), right=int(x1), bottom=int(y1),
            )
        else:
            ij = ImagejRoi(
                roitype=ROI_TYPE.RECT if roi.kind == 'rectangle' else ROI_TYPE.OVAL,
                options=ROI_OPTIONS.SUB_PIXEL_RESOLUTION,
                left=int(np.floor(fx0)), top=int(np.floor(fy0)),
                right=int(np.ceil(fx1)), bottom=int(np.ceil(fy1)),
                xd=fx0, yd=fy0, widthd=max(fx1 - fx0, 0.0), heightd=max(fy1 - fy0, 0.0),
            )
    elif roi.kind == 'line':
        (x1, y1), (x2, y2) = xy[0], xy[-1]
        ij = ImagejRoi(roitype=ROI_TYPE.LINE, x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                       left=int(min(x1, x2)), top=int(min(y1, y2)),
                       right=int(np.ceil(max(x1, x2))), bottom=int(np.ceil(max(y1, y2))))
    else:
        if roi.kind == 'ellipse':  # rotated: store as its outline
            xy = roi_mod._ellipse_polygon(roi.yx)[:, ::-1] + IJ_OFFSET
        ij = ImagejRoi.frompoints(xy)
        ij.roitype = {
            'polygon': ROI_TYPE.POLYGON,
            'rectangle': ROI_TYPE.POLYGON,
            'ellipse': ROI_TYPE.POLYGON,
            'path': ROI_TYPE.POLYLINE,
            'point': ROI_TYPE.POINT,
        }[roi.kind]
    ij.name = roi.name
    hpos = roi.meta.get('hyperstack_pos') if isinstance(roi.meta, dict) else None
    if isinstance(hpos, dict) and hpos:
        if 'C' in hpos:
            ij.c_position = int(hpos['C']) + 1
        if 'Z' in hpos:
            ij.z_position = int(hpos['Z']) + 1
        if 'T' in hpos:
            ij.t_position = int(hpos['T']) + 1
    elif position is not None:
        ij.position = position
    return ij


def from_imagej(ij, fallback_name: str = 'roi') -> Roi | None:
    """Convert a roifile ImagejRoi; returns None for unsupported types (e.g. composite)."""
    from roifile import ROI_OPTIONS, ROI_SUBTYPE, ROI_TYPE

    t = ij.roitype
    sub = getattr(ij, 'subtype', 0)
    non_geom_subs = tuple(
        s for s in (getattr(ROI_SUBTYPE, 'TEXT', None), getattr(ROI_SUBTYPE, 'IMAGE', None)) if s is not None
    )
    if sub in non_geom_subs:
        return None
    name = ij.name or fallback_name
    hpos: dict[str, int] = {}
    for role, val in (('T', getattr(ij, 't_position', 0)), ('Z', getattr(ij, 'z_position', 0)), ('C', getattr(ij, 'c_position', 0))):
        if val and int(val) > 0:
            hpos[role] = int(val) - 1
    meta: dict = {'hyperstack_pos': hpos} if hpos else {}
    if hpos:
        position: tuple[int, ...] | None = tuple(hpos[r] for r in ('T', 'Z', 'C') if r in hpos)
    elif ij.position:
        position = (int(ij.position) - 1,)
    else:
        position = None
    if t in (ROI_TYPE.RECT, ROI_TYPE.OVAL) and not ij.composite:
        has_subpx = bool(getattr(ij, 'options', 0) & ROI_OPTIONS.SUB_PIXEL_RESOLUTION) and (
            float(getattr(ij, 'widthd', 0.0) or 0.0) > 0 or float(getattr(ij, 'heightd', 0.0) or 0.0) > 0
        )
        if has_subpx:
            x0 = float(ij.xd)
            y0 = float(ij.yd)
            x1 = x0 + float(ij.widthd)
            y1 = y0 + float(ij.heightd)
        else:
            x0, y0, x1, y1 = ij.left, ij.top, ij.right, ij.bottom
        corners_xy = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
        yx = corners_xy[:, ::-1] - IJ_OFFSET
        return Roi(name, 'rectangle' if t == ROI_TYPE.RECT else 'ellipse', yx, position, meta=meta)
    if t == ROI_TYPE.LINE:
        yx = np.array([[ij.y1, ij.x1], [ij.y2, ij.x2]], float) - IJ_OFFSET
        return Roi(name, 'line', yx, position, meta=meta)
    kind = {
        ROI_TYPE.POLYGON: 'polygon',
        ROI_TYPE.FREEHAND: 'polygon',
        ROI_TYPE.TRACED: 'polygon',
        ROI_TYPE.POLYLINE: 'path',
        ROI_TYPE.FREELINE: 'path',
        ROI_TYPE.ANGLE: 'path',
        ROI_TYPE.POINT: 'point',
    }.get(t)
    if kind is None or ij.composite:
        return None
    xy = np.asarray(ij.coordinates(), float)
    if xy.ndim != 2 or len(xy) == 0:
        return None
    return Roi(name, kind, xy[:, ::-1] - IJ_OFFSET, position, meta=meta)


def save_imagej(path: str | Path, rois: list[Roi]) -> Path:
    """Write ``.roi`` (one ROI) or ``.zip`` (ROI set) readable by ImageJ/Fiji."""
    import os
    import tempfile

    from roifile import roiwrite

    path = Path(path)
    if path.suffix.lower() == '.roi' and len(rois) != 1:
        path = path.with_suffix('.zip')
    ij_rois = [to_imagej(r) for r in rois]
    if path.suffix.lower() == '.zip':
        seen: set[str] = set()
        for idx, ij in enumerate(ij_rois, 1):
            base = (ij.name or f'{idx:04d}').replace('/', '_').replace('\\', '_').strip() or f'{idx:04d}'
            cand = base
            k = 2
            while cand.lower() in seen:
                cand = f'{base}_{k}'
                k += 1
            seen.add(cand.lower())
            ij.name = cand
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.stem}_', suffix=path.suffix or '.zip')
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        tmp_path.unlink(missing_ok=True)  # roiwrite appends to existing zips
        payload = ij_rois[0] if (path.suffix.lower() == '.roi' and len(ij_rois) == 1) else ij_rois
        roiwrite(tmp_path, payload, mode='w')
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return path


def load_imagej(path: str | Path) -> tuple[list[Roi], int]:
    """Read ImageJ ROIs; returns ``(rois, n_skipped)``."""
    from roifile import roiread

    items = roiread(path)
    if not isinstance(items, list):
        items = [items]
    rois, skipped = [], 0
    for i, ij in enumerate(items):
        r = from_imagej(ij, fallback_name=f'{i + 1:04d}')
        if r is None:
            skipped += 1
        else:
            rois.append(r)
    return rois, skipped


# ------------------------------------------------------------------ Labels

def rois_from_labels(labels2d: np.ndarray, tolerance: float = 0.5, prefix: str = 'label') -> list[Roi]:
    """One polygon ROI per label id (outer boundary of its largest piece)."""
    from skimage.measure import approximate_polygon, find_contours

    labels2d = np.asarray(labels2d)
    rois = []
    for lid in np.unique(labels2d):
        if lid == 0:
            continue
        mask = np.pad(labels2d == lid, 1)
        contours = find_contours(mask.astype(np.uint8), 0.5)
        if not contours:
            continue
        contour = max(contours, key=len) - 1  # undo padding
        raw_contour = contour[:-1] if (contour.shape[0] > 1 and np.allclose(contour[0], contour[-1])) else contour
        if tolerance > 0 and len(contour) > 8:
            approx = approximate_polygon(contour, tolerance)
            if approx.shape[0] > 1 and np.allclose(approx[0], approx[-1]):
                approx = approx[:-1]
            contour = approx if len(approx) >= 3 else raw_contour
        else:
            contour = raw_contour
        rois.append(Roi(f'{prefix} {int(lid)}', 'polygon', contour, meta={'label': int(lid)}))
    return rois


def labels_from_rois(rois: list[Roi], shape2d: tuple[int, int]) -> np.ndarray:
    """Rasterise ROIs into a label image (ids 1..n in list order; later ROIs win)."""
    out = np.zeros(shape2d, dtype=np.int32)
    for i, roi in enumerate(rois, start=1):
        out[roi.mask(shape2d)] = i
    return out
