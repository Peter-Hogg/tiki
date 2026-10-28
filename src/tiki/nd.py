"""Which part of an nD image an operation works on.

Commands describe their scope as *blocks*: index tuples into the image data,
each selecting a 2D plane (Y, X) or a 3D volume (Z, Y, X). The same blocks
index a Labels layer of lower dimensionality, because napari aligns layers on
their trailing axes (see :func:`for_layer`).
"""

from __future__ import annotations

import itertools

from . import axes as axes_mod

PLANE, VOLUME, WHOLE = 'plane', 'volume', 'whole'


def scopes(info: axes_mod.AxesInfo) -> list[tuple[str, str]]:
    """``(key, label)`` choices that make sense for these axes."""
    n_lead = len(info.axes) - 2
    z = info.index('Z')
    out = [(PLANE, 'This plane')]
    if n_lead == 0:
        return out
    others = [a for i, a in enumerate(info.axes[:-2]) if i != z]
    if z is not None and others:
        names = ', '.join(f'{a.role.lower() if a.role != "?" else "dim"}' for a in others)
        out.append((VOLUME, f'This volume (all z at this {names})'))
    out.append((WHOLE, 'Whole stack (all z)' if z is not None and not others else 'Whole stack'))
    return out


def blocks(info: axes_mod.AxesInfo, shape: tuple, index: tuple, scope: str, by_volume: bool) -> list[tuple]:
    """Index tuples selecting planes or (``by_volume``) Z-volumes for ``scope``.

    ``index`` is the current position (only its stack entries are used).
    """
    lead = len(shape) - 2
    z = info.index('Z')
    spatial = (slice(None), slice(None))
    by_volume = by_volume and z is not None
    if scope == PLANE:
        return [tuple(int(index[d]) for d in range(lead)) + spatial]
    if scope == VOLUME:
        if z is None:
            raise ValueError('no Z axis for a volume')
        return [tuple(slice(None) if d == z else int(index[d]) for d in range(lead)) + spatial]
    free = [d for d in range(lead) if not (by_volume and d == z)]
    out = []
    for combo in itertools.product(*(range(shape[d]) for d in free)):
        pos = dict(zip(free, combo))
        out.append(tuple(slice(None) if d not in pos else pos[d] for d in range(lead)) + spatial)
    return out


def for_layer(block: tuple, layer_ndim: int) -> tuple:
    """The same block for a layer that has only the trailing ``layer_ndim`` axes."""
    return block[len(block) - layer_ndim:]


def block_label(info: axes_mod.AxesInfo, block: tuple) -> str:
    """Human-readable position of a block, 1-based like ImageJ (e.g. 't26 z3')."""
    parts = []
    for a, b in zip(info.axes[:-2], block[:-2]):
        if isinstance(b, int):
            parts.append(f'{a.role.lower() if a.role != "?" else "dim"}{b + 1}')
    return ' '.join(parts)
