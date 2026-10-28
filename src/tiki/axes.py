"""What each array axis means (T, Z, C, Y, X) and its physical step.

The data is never rearranged; axes are only *labelled*. Labels come from
file metadata when it is there (ImageJ hyperstacks, ScanImage, OME), else
from a remembered choice for that folder, else from a cautious guess. The
``source`` and ``confident`` fields say which, so the UI can point the user
to Image > Properties when the guess might be wrong.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROLES = ['T', 'Z', 'C', 'Y', 'X', '?']
ROLE_NAMES = {'T': 'time', 'Z': 'depth', 'C': 'channel', 'Y': 'rows', 'X': 'columns', '?': 'unknown'}
DEFAULT_UNITS = {'T': 's', 'Z': 'µm', 'Y': 'µm', 'X': 'µm'}
PINT_UNITS = {'µm': 'micrometer', 'nm': 'nanometer', 'mm': 'millimeter',
              's': 'second', 'ms': 'millisecond', 'min': 'minute'}
MEMORY_FILE = Path.home() / '.config' / 'tiki' / 'axes_memory.json'

# tifffile axis codes -> TIKI roles. 'I' (generic image sequence) and the
# rarer codes stay unknown: better to ask than to call a movie a z-stack.
TIFF_ROLES = {'T': 'T', 'Z': 'Z', 'C': 'C', 'Y': 'Y', 'X': 'X'}
UNIT_ALIASES = {'micron': 'µm', 'microns': 'µm', 'um': 'µm', 'µm': 'µm', '\\u00B5m': 'µm',
                'nm': 'nm', 'mm': 'mm', 'sec': 's', 's': 's', 'ms': 'ms', 'min': 'min'}


@dataclass
class Axis:
    role: str  # one of ROLES
    size: int
    step: float | None = None  # physical size of one step, None = uncalibrated
    unit: str | None = None

    @property
    def scale(self) -> float:
        return float(self.step) if self.step else 1.0


@dataclass
class AxesInfo:
    axes: list[Axis]
    source: str = 'guess'
    confident: bool = False
    notes: list[str] = field(default_factory=list)

    # -------------------------------------------------------------- queries

    @property
    def roles(self) -> str:
        return ''.join(a.role for a in self.axes)

    def index(self, role: str) -> int | None:
        for i, a in enumerate(self.axes):
            if a.role == role:
                return i
        return None

    @property
    def labels(self) -> list[str]:
        out = []
        for i, a in enumerate(self.axes):
            out.append(f'dim{i}' if a.role == '?' else a.role.lower())
        return out

    @property
    def z_assumed(self) -> bool:
        """Z has no step but X/Y are calibrated: Z is drawn as cubic voxels."""
        z, y = self.index('Z'), self.index('Y')
        return z is not None and y is not None and not self.axes[z].step and bool(self.axes[y].step)

    @property
    def scale(self) -> list[float]:
        out = [1.0 if a.role in ('C', '?') else a.scale for a in self.axes]
        if self.z_assumed:  # like ImageJ's 3D viewer: assume isotropic voxels
            out[self.index('Z')] = self.axes[self.index('Y')].step
        return out

    def display_unit(self, i: int) -> str | None:
        a = self.axes[i]
        if a.step:
            return a.unit
        if a.role == 'Z' and self.z_assumed:
            return self.axes[self.index('Y')].unit
        return None

    def describe(self) -> str:
        parts = []
        for a in self.axes:
            if a.step and a.role not in ('C', '?'):
                parts.append(f'{a.role}{a.size} ({a.step:.4g} {a.unit or ""})'.replace(' )', ')'))
            else:
                parts.append(f'{a.role}{a.size}')
        return ' × '.join(parts)

    def problems(self, shape: tuple | None = None) -> list[str]:
        """Why these axes cannot be used as is (empty list = fine)."""
        issues = []
        if shape is not None and len(shape) != len(self.axes):
            issues.append(f'{len(self.axes)} axes for {len(shape)}-D data')
        if shape is not None:
            for a, n in zip(self.axes, shape):
                if a.size != n:
                    issues.append(f'{a.role} has size {a.size}, data has {n}')
        for r in 'TZCYX':
            if self.roles.count(r) > 1:
                issues.append(f'more than one {r} axis')
        if len(self.axes) >= 2 and self.roles[-2:] != 'YX':
            issues.append('the last two axes must be Y and X')
        return issues

    # ------------------------------------------------------------ derive

    def without(self, i: int) -> AxesInfo:
        """Axes after removing axis ``i`` (projection, channel split...)."""
        return AxesInfo([a for k, a in enumerate(self.axes) if k != i], self.source, self.confident, list(self.notes))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> AxesInfo:
        return cls([Axis(**a) for a in d['axes']], d.get('source', 'guess'), d.get('confident', False),
                   list(d.get('notes', [])))


# ------------------------------------------------------------------ guess

def guess(shape: tuple, rgb: bool = False) -> AxesInfo:
    """Cautious default: last two axes are Y, X; anything before is unknown."""
    shape = tuple(shape[:-1]) if rgb else tuple(shape)
    n = len(shape)
    axes = [Axis('?', s) for s in shape]
    if n >= 2:
        axes[-2].role, axes[-1].role = 'Y', 'X'
    elif n == 1:
        axes[0].role = 'X'
    confident = n <= 2
    notes = [] if confident else ['axis meaning unknown (no metadata)']
    return AxesInfo(axes, 'guess', confident, notes)


def _unit(u) -> str | None:
    if u is None:
        return None
    u = str(u).strip()
    return UNIT_ALIASES.get(u.lower(), u) if u else None


# ------------------------------------------------------------------- TIFF

def _xy_step(tf, ij: dict) -> tuple[float | None, str | None, str | None]:
    """Pixel size from TIFF resolution tags; returns (step, unit, note)."""
    import tifffile

    page = tf.pages[0]
    xres = page.tags.get('XResolution')
    if xres is None:
        return None, None, None
    num, den = xres.value
    if not num or not den:
        return None, None, None
    step = den / num  # units per pixel
    resunit = page.tags.get('ResolutionUnit')
    resunit = resunit.value if resunit is not None else tifffile.RESUNIT.NONE
    ij_unit = _unit(ij.get('unit'))
    if ij_unit and ij_unit not in ('pixel', 'pixels'):
        unit = ij_unit
    elif resunit == tifffile.RESUNIT.CENTIMETER:
        step, unit = step * 1e4, 'µm'
    elif resunit == tifffile.RESUNIT.MILLIMETER:
        step, unit = step * 1e3, 'µm'
    elif resunit == tifffile.RESUNIT.MICROMETER:
        unit = 'µm'
    else:  # inches / none: usually a screen DPI, not a calibration
        return None, None, None
    if step == 1 and unit in (None, 'pixel'):
        return None, None, None
    if unit == 'µm' and not (0.01 <= step <= 1000):
        return None, None, f'ignored implausible pixel size {step:.3g} µm'
    return step, unit, None


def _ome_steps(tf) -> dict:
    """PhysicalSizeX/Y/Z and TimeIncrement from OME-XML (with units)."""
    xml = tf.ome_metadata or ''
    out = {}
    m = re.search(r'<Pixels\b[^>]*>', xml)
    if not m:
        return out
    tag = m.group(0)

    def attr(name):
        a = re.search(rf'\b{name}="([^"]+)"', tag)
        return a.group(1) if a else None

    for role, key in (('X', 'PhysicalSizeX'), ('Y', 'PhysicalSizeY'), ('Z', 'PhysicalSizeZ'), ('T', 'TimeIncrement')):
        val = attr(key)
        if val:
            try:
                out[role] = (float(val), _unit(attr(key + 'Unit')) or DEFAULT_UNITS[role])
            except ValueError:
                pass
    return out


def from_tiff(tf, data_shape: tuple) -> AxesInfo:
    """Axes from an open ``tifffile.TiffFile`` whose first series gave ``data_shape``."""
    series = tf.series[0]
    codes = series.axes
    notes = []
    if len(codes) != len(data_shape):
        # e.g. a truncated ScanImage file read with a different shape
        info = guess(data_shape)
        info.notes.append(f'file axes {codes} do not match data shape {data_shape}')
        return info
    if codes.endswith('S') and data_shape[-1] in (3, 4):  # RGB(A): napari shows it as colour
        codes, data_shape = codes[:-1], data_shape[:-1]
    axes = [Axis(TIFF_ROLES.get(c, '?'), int(n)) for c, n in zip(codes, data_shape)]
    info = AxesInfo(axes, 'TIFF', True, notes)
    ij = tf.imagej_metadata or {}

    if tf.is_ome:
        info.source = 'OME metadata'
        for role, (step, unit) in _ome_steps(tf).items():
            i = info.index(role)
            if i is not None:
                axes[i].step, axes[i].unit = step, unit
    else:
        if tf.is_imagej:
            info.source = 'ImageJ metadata'
        elif tf.is_scanimage:
            info.source = 'ScanImage metadata'
        if not tf.is_scanimage:  # ScanImage resolution tags are not a calibration
            step, unit, note = _xy_step(tf, ij)
            if note:
                notes.append(note)
            if step:
                for r in 'YX':
                    i = info.index(r)
                    if i is not None:
                        axes[i].step, axes[i].unit = step, unit
        if ij.get('spacing') and info.index('Z') is not None:
            axes[info.index('Z')].step = float(ij['spacing'])
            axes[info.index('Z')].unit = _unit(ij.get('unit')) or 'µm'
        if ij.get('finterval') and info.index('T') is not None:
            axes[info.index('T')].step = float(ij['finterval'])
            axes[info.index('T')].unit = _unit(ij.get('tunit')) or 's'
        if tf.is_scanimage and info.index('T') is not None:
            fd = (tf.scanimage_metadata or {}).get('FrameData', {})
            period = fd.get('SI.hRoiManager.scanFramePeriod')
            if period:
                axes[info.index('T')].step, axes[info.index('T')].unit = float(period), 's'

    if '?' in info.roles[:-2]:
        info.confident = False
        notes.append(f'file says {series.axes}: stack axis meaning unknown')
    if info.index('Z') is not None and axes[info.index('Z')].step is None:
        notes.append('z-step not in file')
    if info.problems(tuple(a.size for a in axes)):
        fallback = guess(data_shape)
        fallback.notes = notes + ['file axes unusable: ' + '; '.join(info.problems())]
        return fallback
    return info


# ----------------------------------------------------------- folder memory

def _memory_key(path: Path, shape: tuple) -> str:
    return f'{path.parent}|{len(shape)}'


def remember(path: str | Path, shape: tuple, info: AxesInfo) -> None:
    """Store roles/steps for files of this dimensionality in this folder."""
    try:
        MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        memory = json.loads(MEMORY_FILE.read_text()) if MEMORY_FILE.exists() else {}
        memory[_memory_key(Path(path), shape)] = [
            {'role': a.role, 'step': a.step, 'unit': a.unit} for a in info.axes
        ]
        MEMORY_FILE.write_text(json.dumps(memory, indent=1))
    except OSError:
        pass  # memory is a convenience; never fail an action over it


def recall(path: str | Path, shape: tuple) -> AxesInfo | None:
    try:
        memory = json.loads(MEMORY_FILE.read_text())
    except (OSError, ValueError):
        return None
    saved = memory.get(_memory_key(Path(path), shape))
    if not saved or len(saved) != len(shape):
        return None
    axes = [Axis(s['role'], int(n), s.get('step'), s.get('unit')) for s, n in zip(saved, shape)]
    info = AxesInfo(axes, 'remembered for this folder', True)
    return None if info.problems(shape) else info


# ------------------------------------------------------- layers & viewers

def get(image) -> AxesInfo:
    """Axes of a napari Image layer (falls back to a guess if missing or stale)."""
    shape = tuple(image.data.shape[: image.ndim])
    d = image.metadata.get('axes') if hasattr(image, 'metadata') else None
    if d:
        try:
            info = AxesInfo.from_dict(d)
            if not info.problems(shape):
                return info
        except (KeyError, TypeError):
            pass
    return guess(shape)


def fit(info: AxesInfo, shape: tuple) -> AxesInfo:
    """Reuse roles and steps for data of a new shape (crop, filter...)."""
    if len(info.axes) != len(shape):
        return guess(shape)
    out = AxesInfo([Axis(a.role, int(n), a.step, a.unit) for a, n in zip(info.axes, shape)],
                   info.source, info.confident, list(info.notes))
    return out if not out.problems(shape) else guess(shape)


def stack_axis(info: AxesInfo, prefer: str = 'TZ') -> int | None:
    """Axis to run along for profiles/traces: T, else Z, else the first unknown one."""
    for role in prefer:
        i = info.index(role)
        if i is not None:
            return i
    for i, a in enumerate(info.axes[:-2]):
        if a.role == '?':
            return i
    return None


def geometry(image, ndim: int | None = None, drop: int | None = None) -> dict:
    """scale/translate/units for a new layer lining up with ``image``.

    ``ndim=2`` gives a plane overlay (ROIs, previews); ``drop`` removes one
    axis (channel split). Passing units at construction matters: napari
    disables units in a window if layers disagree even momentarily.
    """
    keep = [i for i in range(image.ndim) if i != drop]
    if ndim is not None:
        keep = keep[-ndim:]
    out = {
        'scale': [image.scale[i] for i in keep],
        'translate': [image.translate[i] for i in keep],
    }
    units = getattr(image, 'units', None)
    if units is not None:
        out['units'] = tuple(str(units[i]) for i in keep)
    return out


def match_units(layer, image) -> None:
    """Give ``layer`` the image's units for the axes they share (napari ignores
    units altogether if layers in a window disagree)."""
    if layer is image or not hasattr(layer, 'units') or not hasattr(image, 'units'):
        return
    units = tuple(image.units)
    if layer.ndim > len(units):
        return
    try:
        layer.units = units[-layer.ndim:]
    except Exception:
        pass


def apply(viewer, image, info: AxesInfo) -> None:
    """Store ``info`` on the layer and update scale, slider labels and scale bar."""
    image.metadata['axes'] = info.to_dict()
    old_tail = tuple(image.scale[-2:])
    image.scale = info.scale
    new_tail = tuple(info.scale[-2:])
    for layer in viewer.layers:
        if layer is image:
            continue
        if layer.ndim == 2 and tuple(layer.scale) == old_tail:
            layer.scale = new_tail
        elif layer.ndim == image.ndim and layer.metadata.get('axes') is None:
            layer.scale = info.scale
    labels = list(viewer.dims.axis_labels)
    for k, label in enumerate(info.labels):
        labels[len(labels) - image.ndim + k] = label
    viewer.dims.axis_labels = labels
    try:  # napari >= 0.6 takes physical units per layer axis
        image.units = tuple(PINT_UNITS.get(info.display_unit(i), 'pixel') for i in range(len(info.axes)))
    except Exception:
        pass  # unknown unit string: keep the scale, skip the unit
    for layer in viewer.layers:
        match_units(layer, image)
    y = info.axes[info.index('Y')] if info.index('Y') is not None else None
    overlays = getattr(getattr(viewer, 'canvas', None), 'overlays', None)
    scale_bar = getattr(overlays, 'scale_bar', None) or viewer.scale_bar  # napari 0.9 moved it
    scale_bar.visible = bool(y is not None and y.step and y.unit)
