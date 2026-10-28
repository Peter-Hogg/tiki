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
        cy, cx = self.yx.mean(axis=0)
        return f'{index:04d}-{int(round(cy)):04d}-{int(round(cx)):04d}'


# ------------------------------------------------------------------ ImageJ

def _axis_aligned(corners: np.ndarray) -> bool:
    c = np.asarray(corners)
    return len(c) == 4 and (np.allclose(c[0, 0], c[1, 0]) and np.allclose(c[1, 1], c[2, 1])
                            or np.allclose(c[0, 1], c[1, 1]) and np.allclose(c[1, 0], c[2, 0]))


def to_imagej(roi: Roi):
    from roifile import ROI_TYPE, ImagejRoi

    xy = roi.yx[:, ::-1] + IJ_OFFSET
    position = roi.position[0] + 1 if roi.position and len(roi.position) == 1 else None
    if roi.kind in ('rectangle', 'ellipse') and _axis_aligned(roi.yx):
        x0, y0 = np.round(xy.min(axis=0)).astype(int)
        x1, y1 = np.round(xy.max(axis=0)).astype(int)
        ij = ImagejRoi(
            roitype=ROI_TYPE.RECT if roi.kind == 'rectangle' else ROI_TYPE.OVAL,
            left=int(x0), top=int(y0), right=int(x1), bottom=int(y1),
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
    if position is not None:
        ij.position = position
    return ij


def from_imagej(ij, fallback_name: str = 'roi') -> Roi | None:
    """Convert a roifile ImagejRoi; returns None for unsupported types (e.g. composite)."""
    from roifile import ROI_TYPE

    t = ij.roitype
    name = ij.name or fallback_name
    pos = ij.position or ij.t_position or ij.z_position
    position = (int(pos) - 1,) if pos else None
    if t in (ROI_TYPE.RECT, ROI_TYPE.OVAL) and not ij.composite:
        x0, y0, x1, y1 = ij.left, ij.top, ij.right, ij.bottom
        corners_xy = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
        yx = corners_xy[:, ::-1] - IJ_OFFSET
        return Roi(name, 'rectangle' if t == ROI_TYPE.RECT else 'ellipse', yx, position)
    if t == ROI_TYPE.LINE:
        yx = np.array([[ij.y1, ij.x1], [ij.y2, ij.x2]], float) - IJ_OFFSET
        return Roi(name, 'line', yx, position)
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
    return Roi(name, kind, xy[:, ::-1] - IJ_OFFSET, position)


def save_imagej(path: str | Path, rois: list[Roi]) -> None:
    """Write ``.roi`` (one ROI) or ``.zip`` (ROI set) readable by ImageJ/Fiji."""
    from roifile import roiwrite

    path = Path(path)
    if path.suffix.lower() == '.roi' and len(rois) != 1:
        path = path.with_suffix('.zip')
    if path.exists():
        path.unlink()  # roiwrite appends to existing zips
    roiwrite(path, [to_imagej(r) for r in rois], mode='w')


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
        if tolerance > 0 and len(contour) > 8:
            contour = approximate_polygon(contour, tolerance)
        if contour.shape[0] > 1 and np.allclose(contour[0], contour[-1]):
            contour = contour[:-1]
        rois.append(Roi(f'{prefix} {int(lid)}', 'polygon', contour, meta={'label': int(lid)}))
    return rois


def labels_from_rois(rois: list[Roi], shape2d: tuple[int, int]) -> np.ndarray:
    """Rasterise ROIs into a label image (ids 1..n in list order; later ROIs win)."""
    out = np.zeros(shape2d, dtype=np.int32)
    for i, roi in enumerate(rois, start=1):
        out[roi.mask(shape2d)] = i
    return out
