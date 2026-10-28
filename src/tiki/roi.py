"""ROI helpers: the per-window ``ROI`` shapes layer, masks and current position.

ImageJ has one "current selection" per image. TIKI keeps all ROIs of a window
in a Shapes layer named ``ROI``; the current selection is the selected shape,
or the most recently drawn one if nothing is selected.

ROI and point layers are 2D even on stacks, so (as in ImageJ) a selection
stays put while you scroll through slices or frames.
"""

from __future__ import annotations

import numpy as np
from napari.layers import Image, Points, Shapes
from skimage.draw import line as draw_line
from skimage.draw import polygon2mask

ROI_NAME = 'ROI'
POINTS_NAME = 'Points'
ROI_COLOR = 'yellow'
# temporary layers that must never count as "the image"
PREVIEW_NAMES = {'filter preview', 'threshold preview'}


def target_image(viewer) -> Image | None:
    """The image commands act on: the selected Image layer, else the topmost one."""
    active = viewer.layers.selection.active
    if isinstance(active, Image) and active.name not in PREVIEW_NAMES:
        return active
    images = [ly for ly in viewer.layers if isinstance(ly, Image) and ly.name not in PREVIEW_NAMES]
    return images[-1] if images else None


def plane_kwargs(image: Image | None) -> dict:
    """scale/translate/ndim for a 2D overlay layer that lines up with ``image``."""
    if image is None:
        return {'ndim': 2}
    return dict(scale=image.scale[-2:], translate=image.translate[-2:], ndim=2)


def roi_layer(viewer, create: bool = True) -> Shapes | None:
    if ROI_NAME in viewer.layers and isinstance(viewer.layers[ROI_NAME], Shapes):
        return viewer.layers[ROI_NAME]
    if not create:
        return None
    kwargs = plane_kwargs(target_image(viewer))
    layer = Shapes(
        name=ROI_NAME,
        edge_color=ROI_COLOR,
        face_color='transparent',
        edge_width=1.5,
        **kwargs,
    )
    viewer.add_layer(layer)
    return layer


def points_layer(viewer, create: bool = True) -> Points | None:
    if POINTS_NAME in viewer.layers and isinstance(viewer.layers[POINTS_NAME], Points):
        return viewer.layers[POINTS_NAME]
    if not create:
        return None
    kwargs = plane_kwargs(target_image(viewer))
    layer = Points(name=POINTS_NAME, face_color=ROI_COLOR, size=5, **kwargs)
    viewer.add_layer(layer)
    return layer


def current_shape(layer: Shapes | None) -> tuple[np.ndarray, str] | None:
    """``(vertices, shape_type)`` of the current selection, or None."""
    if layer is None or layer.nshapes == 0:
        return None
    selected = sorted(layer.selected_data)
    i = selected[-1] if selected else layer.nshapes - 1
    return np.asarray(layer.data[i]), layer.shape_type[i]


def _ellipse_polygon(corners: np.ndarray, n: int = 128) -> np.ndarray:
    """napari stores ellipses as their 4 bounding-box corners; trace the outline."""
    center = corners.mean(axis=0)
    a = (corners[1] - corners[0]) / 2
    b = (corners[3] - corners[0]) / 2
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return center + np.outer(np.cos(theta), a) + np.outer(np.sin(theta), b)


def shape_mask(vertices: np.ndarray, shape_type: str, shape2d: tuple[int, int]) -> np.ndarray:
    """Boolean mask of a shape in the last two (spatial) dims."""
    verts = np.asarray(vertices, dtype=float)[:, -2:]
    if shape_type == 'ellipse':
        verts = _ellipse_polygon(verts)
    if shape_type in ('line', 'path'):
        mask = np.zeros(shape2d, dtype=bool)
        for p, q in zip(verts[:-1], verts[1:]):
            rr, cc = draw_line(*np.round(p).astype(int), *np.round(q).astype(int))
            keep = (rr >= 0) & (rr < shape2d[0]) & (cc >= 0) & (cc < shape2d[1])
            mask[rr[keep], cc[keep]] = True
        return mask
    # Pixel centres are at integer coordinates in napari; a pixel is inside if
    # its centre is. The tiny shift makes edges half-open, so a rectangle from
    # 10 to 20 covers 10 pixels as in ImageJ (and agrees with shape_box).
    return polygon2mask(shape2d, verts - 1e-6)


def shape_box(vertices: np.ndarray, shape2d: tuple[int, int]) -> tuple[int, int, int, int]:
    """Integer ``(y0, y1, x0, x1)`` bounding box of a shape, clipped to the image."""
    verts = np.asarray(vertices, dtype=float)[:, -2:]
    y0, x0 = np.floor(verts.min(axis=0) + 0.5).astype(int)
    y1, x1 = np.floor(verts.max(axis=0) + 0.5).astype(int)
    y0, x0 = max(y0, 0), max(x0, 0)
    y1, x1 = min(y1, shape2d[0]), min(x1, shape2d[1])
    return int(y0), int(y1), int(x0), int(x1)


def spatial_shape(image: Image) -> tuple[int, int]:
    shape = image.data.shape[:-1] if image.rgb else image.data.shape
    return tuple(shape[-2:])


def current_index(viewer, image: Image) -> tuple[int, ...]:
    """Integer data index of the slice being shown (spatial entries are 0)."""
    ndim = image.ndim
    point = image.world_to_data(viewer.dims.point[-ndim:])
    shape = image.data.shape[:ndim]
    idx = []
    for d in range(ndim):
        if d >= ndim - 2:
            idx.append(0)
        else:
            idx.append(int(np.clip(np.round(point[d]), 0, shape[d] - 1)))
    return tuple(idx)


def current_plane(viewer, image: Image) -> np.ndarray:
    idx = current_index(viewer, image)
    plane = np.asarray(image.data[idx[:-2]])
    if image.rgb:
        plane = plane[..., :3].mean(axis=-1)
    return plane


def axis_name(viewer, image: Image, d: int) -> str:
    """ImageJ-style name for data axis ``d``; napari's defaults are just -3, -2..."""
    label = str(viewer.dims.axis_labels[-image.ndim:][d])
    if not label.lstrip('-').isdigit():
        return label
    n_stack = image.ndim - 2
    names = {1: ['z'], 2: ['t', 'z'], 3: ['t', 'z', 'c']}.get(n_stack)
    return names[d] if names and d < n_stack else f'dim{d}'


def pixel_area(image: Image) -> float:
    return float(image.scale[-1] * image.scale[-2])
