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
PINT_UNITS = {'µm': 'micrometer', 'nm': 'nanometer', 'mm': 'millimeter', 'cm': 'centimeter',
              'm': 'meter', 's': 'second', 'ms': 'millisecond', 'µs': 'microsecond',
              'ns': 'nanosecond', 'min': 'minute', 'h': 'hour'}
LENGTH_TO_M = {'nm': 1e-9, 'µm': 1e-6, 'mm': 1e-3, 'cm': 1e-2, 'm': 1.0}
TIME_TO_S = {'ns': 1e-9, 'µs': 1e-6, 'ms': 1e-3, 's': 1.0, 'min': 60.0, 'h': 3600.0}
UNIT_FACTORS = {**LENGTH_TO_M, **TIME_TO_S}
MEMORY_FILE = Path.home() / '.config' / 'tiki' / 'axes_memory.json'

# tifffile axis codes -> TIKI roles. 'I' (generic image sequence) and the
# rarer codes stay unknown: better to ask than to call a movie a z-stack.
TIFF_ROLES = {'T': 'T', 'Z': 'Z', 'C': 'C', 'Y': 'Y', 'X': 'X'}
UNIT_ALIASES = {'micron': 'µm', 'microns': 'µm', 'um': 'µm', 'µm': 'µm', 'μm': 'µm',
                '\\u00b5m': 'µm', '\\u00B5m': 'µm',
                'micrometer': 'µm', 'micrometers': 'µm', 'micrometre': 'µm', 'micrometres': 'µm',
                'nm': 'nm', 'nanometer': 'nm', 'nanometers': 'nm', 'nanometre': 'nm',
                'mm': 'mm', 'millimeter': 'mm', 'millimeters': 'mm', 'millimetre': 'mm',
                'cm': 'cm', 'centimeter': 'cm', 'centimeters': 'cm', 'centimetre': 'cm',
                'm': 'm', 'meter': 'm', 'meters': 'm', 'metre': 'm',
                'sec': 's', 's': 's', 'second': 's', 'seconds': 's',
                'ms': 'ms', 'millisecond': 'ms', 'milliseconds': 'ms',
                'us': 'µs', 'µs': 'µs', 'μs': 'µs', 'microsecond': 'µs', 'microseconds': 'µs',
                'ns': 'ns', 'nanosecond': 'ns', 'nanoseconds': 'ns',
                'min': 'min', 'minute': 'min', 'minutes': 'min',
                'h': 'h', 'hr': 'h', 'hour': 'h', 'hours': 'h'}


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


def convert_unit(value: float, src: str | None, dst: str | None) -> float:
    """Convert ``value`` between compatible metric length or time units."""
    s = _unit(src)
    d = _unit(dst)
    if s is None or d is None or s == d:
        return float(value)
    if s in LENGTH_TO_M and d in LENGTH_TO_M:
        return float(value) * (LENGTH_TO_M[s] / LENGTH_TO_M[d])
    if s in TIME_TO_S and d in TIME_TO_S:
        return float(value) * (TIME_TO_S[s] / TIME_TO_S[d])
    return float(value)


def spatial_unit(info: AxesInfo) -> str | None:
    """Physical spatial unit of Y/X if calibrated, else None."""
    for role in ('Y', 'X'):
        idx = info.index(role)
        if idx is not None and info.axes[idx].step and info.axes[idx].unit:
            return info.axes[idx].unit
    return None


def z_ratio(info: AxesInfo) -> float:
    """XY pixel size divided by Z step in matching physical units (1.0 if either is unknown)."""
    z, y = info.index('Z'), info.index('Y')
    if z is None or y is None:
        return 1.0
    az, ay = info.axes[z], info.axes[y]
    if not (az.step and ay.step):
        return 1.0
    sz_in_y = convert_unit(az.step, az.unit, ay.unit)
    return float(ay.step / sz_in_y) if sz_in_y > 0 else 1.0


def physical_scale_3d(info: AxesInfo) -> tuple[tuple[float, float, float], str | None, bool]:
    """Return ``((sz, sy, sx), unit, z_assumed)`` in consistent XY spatial units."""
    y_idx, x_idx, z_idx = info.index('Y'), info.index('X'), info.index('Z')
    y_ax = info.axes[y_idx] if y_idx is not None else None
    x_ax = info.axes[x_idx] if x_idx is not None else None
    z_ax = info.axes[z_idx] if z_idx is not None else None

    sy = float(y_ax.step) if (y_ax is not None and y_ax.step) else 1.0
    sx = float(x_ax.step) if (x_ax is not None and x_ax.step) else sy
    unit = spatial_unit(info)
    z_assumed = info.z_assumed

    if z_ax is not None and z_ax.step:
        sz = convert_unit(float(z_ax.step), z_ax.unit, unit) if unit else float(z_ax.step)
    elif z_assumed:
        sz = sy
    else:
        sz = 1.0
    return (sz, sy, sx), unit, z_assumed


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


def _xml_tag(el) -> str:
    return el.tag.split('}')[-1] if '}' in el.tag else el.tag


def _find_child(el, name: str):
    for child in el:
        if _xml_tag(child) == name:
            return child
    return None


def _find_children(el, name: str) -> list:
    return [child for child in el if _xml_tag(child) == name]


def _find_all(el, name: str) -> list:
    return [node for node in el.iter() if _xml_tag(node) == name]


def _float_attr(el, name: str) -> float | None:
    if el is None:
        return None
    val = el.get(name)
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def parse_ome_xml(xml: str, series_index: int = 0) -> dict:
    """Parse OME-XML calibration, channel names/colors, and acquisition metadata."""
    import xml.etree.ElementTree as ET

    empty = {
        'steps': {},
        'notes': [],
        'series_name': None,
        'channel_names': [],
        'channel_colors': [],
        'channels': [],
        'acquisition': {},
    }
    if not xml or not str(xml).strip():
        return empty
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return empty

    images = _find_children(root, 'Image') or _find_all(root, 'Image')
    if not images:
        return empty
    img_el = images[series_index] if 0 <= series_index < len(images) else images[0]
    series_name = (img_el.get('Name') or '').strip() or None

    steps: dict[str, tuple[float, str]] = {}
    notes: list[str] = []
    channel_names: list[str] = []
    channel_colors: list[tuple[int, int, int] | None] = []
    channels: list[dict] = []

    pixels_el = _find_child(img_el, 'Pixels')
    if pixels_el is not None:
        for role, key in (
            ('X', 'PhysicalSizeX'),
            ('Y', 'PhysicalSizeY'),
            ('Z', 'PhysicalSizeZ'),
            ('T', 'TimeIncrement'),
        ):
            fval = _float_attr(pixels_el, key)
            if fval is not None and fval > 0:
                u = _unit(pixels_el.get(key + 'Unit'))
                if u == 'm' and role in ('X', 'Y', 'Z'):
                    steps[role] = _normalize_czi_step(role, fval, u)
                else:
                    steps[role] = (fval, u or DEFAULT_UNITS[role])

        # Fallback: compute time interval from <Plane DeltaT=.../> when TimeIncrement is absent
        if 'T' not in steps:
            grouped: dict[tuple[str, str], list[tuple[int, float, str | None]]] = {}
            for pl in _find_children(pixels_el, 'Plane'):
                dt = _float_attr(pl, 'DeltaT')
                the_t = pl.get('TheT')
                if dt is None or the_t is None:
                    continue
                try:
                    t_idx = int(the_t)
                except ValueError:
                    continue
                key_zc = (pl.get('TheZ') or '0', pl.get('TheC') or '0')
                grouped.setdefault(key_zc, []).append((t_idx, dt, _unit(pl.get('DeltaTUnit')) or 's'))
            if grouped:
                best_key = max(grouped, key=lambda k: (len(grouped[k]), -int(k[0]) if k[0].isdigit() else 0))
                pts = sorted(grouped[best_key], key=lambda item: item[0])
                t_unit = pts[0][2] or 's'
                diffs = []
                for (t1, d1, u1), (t2, d2, u2) in zip(pts[:-1], pts[1:]):
                    if t2 > t1:
                        v1 = convert_unit(d1, u1, t_unit)
                        v2 = convert_unit(d2, u2, t_unit)
                        delta = (v2 - v1) / (t2 - t1)
                        if delta > 0:
                            diffs.append(delta)
                if diffs:
                    diffs.sort()
                    mid = len(diffs) // 2
                    med = diffs[mid] if len(diffs) % 2 else 0.5 * (diffs[mid - 1] + diffs[mid])
                    steps['T'] = (float(med), t_unit)
                    notes.append('time step from OME Plane DeltaT')

        for ch_el in _find_children(pixels_el, 'Channel'):
            cname = (ch_el.get('Name') or '').strip()
            fluor = (ch_el.get('Fluor') or '').strip()
            channel_names.append(cname or fluor)
            rgb = None
            raw_color = ch_el.get('Color')
            if raw_color is not None:
                try:
                    u32 = int(raw_color) & 0xFFFFFFFF
                    rgb_cand = ((u32 >> 24) & 0xFF, (u32 >> 16) & 0xFF, (u32 >> 8) & 0xFF)
                    if rgb_cand != (0, 0, 0):
                        rgb = rgb_cand
                except ValueError:
                    pass
            channel_colors.append(rgb)
            ch_info = {}
            if cname:
                ch_info['name'] = cname
            if fluor:
                ch_info['fluor'] = fluor
            if rgb is not None:
                ch_info['color_rgb'] = rgb
            for attr_name, key_out in (
                ('ExcitationWavelength', 'excitation_nm'),
                ('EmissionWavelength', 'emission_nm'),
                ('PinholeSize', 'pinhole_um'),
            ):
                fval = _float_attr(ch_el, attr_name)
                if fval is not None:
                    ch_info[key_out] = fval
            for attr_name, key_out in (
                ('AcquisitionMode', 'acquisition_mode'),
                ('IlluminationType', 'illumination_type'),
            ):
                sval = (ch_el.get(attr_name) or '').strip()
                if sval:
                    ch_info[key_out] = sval
            channels.append(ch_info)

    acquisition: dict = {}
    if series_name:
        acquisition['image_name'] = series_name
    acq_date = _find_child(img_el, 'AcquisitionDate')
    if acq_date is not None and acq_date.text and acq_date.text.strip():
        acquisition['acquisition_date'] = acq_date.text.strip()
    desc_el = _find_child(img_el, 'Description')
    if desc_el is not None and desc_el.text and desc_el.text.strip():
        acquisition['description'] = desc_el.text.strip()
    stage_el = _find_child(img_el, 'StageLabel')
    if stage_el is not None:
        stage = {}
        if stage_el.get('Name'):
            stage['name'] = stage_el.get('Name')
        for axis_key in ('X', 'Y', 'Z'):
            v = _float_attr(stage_el, axis_key)
            if v is not None:
                stage[axis_key.lower()] = v
        if stage:
            acquisition['stage_label'] = stage

    instruments = _find_children(root, 'Instrument') or _find_all(root, 'Instrument')
    inst_ref = _find_child(img_el, 'InstrumentRef')
    inst_id = inst_ref.get('ID') if inst_ref is not None else None
    inst_el = next((i for i in instruments if inst_id and i.get('ID') == inst_id), instruments[0] if instruments else None)
    if inst_el is not None:
        mic_el = _find_child(inst_el, 'Microscope')
        if mic_el is not None:
            mic = {
                k: v
                for k, attr in (('manufacturer', 'Manufacturer'), ('model', 'Model'), ('serial_number', 'SerialNumber'))
                if (v := (mic_el.get(attr) or '').strip())
            }
            if mic:
                acquisition['microscope'] = mic

        objs = _find_children(inst_el, 'Objective')
        obj_set = _find_child(img_el, 'ObjectiveSettings')
        obj_id = obj_set.get('ID') if obj_set is not None else None
        obj_el = next((o for o in objs if obj_id and o.get('ID') == obj_id), objs[0] if objs else None)
        if obj_el is not None:
            obj = {}
            for k, attr in (('manufacturer', 'Manufacturer'), ('model', 'Model'), ('immersion', 'Immersion')):
                v = (obj_el.get(attr) or '').strip()
                if v:
                    obj[k] = v
            for k, attr in (
                ('magnification', 'NominalMagnification'),
                ('na', 'LensNA'),
                ('working_distance', 'WorkingDistance'),
            ):
                v = _float_attr(obj_el, attr)
                if v is not None:
                    obj[k] = v
            if obj:
                acquisition['objective'] = obj

        dets = _find_children(inst_el, 'Detector')
        det_id = None
        if pixels_el is not None:
            for ch_el in _find_children(pixels_el, 'Channel'):
                ds = _find_child(ch_el, 'DetectorSettings')
                if ds is not None and ds.get('ID'):
                    det_id = ds.get('ID')
                    break
        det_el = next((d for d in dets if det_id and d.get('ID') == det_id), dets[0] if dets else None)
        if det_el is not None:
            det = {}
            for k, attr in (('manufacturer', 'Manufacturer'), ('model', 'Model'), ('type', 'Type')):
                v = (det_el.get(attr) or '').strip()
                if v:
                    det[k] = v
            gain = _float_attr(det_el, 'Gain')
            if gain is not None:
                det['gain'] = gain
            if det:
                acquisition['detector'] = det

    if channels:
        acquisition['channels'] = channels

    return {
        'steps': steps,
        'notes': notes,
        'series_name': series_name,
        'channel_names': channel_names,
        'channel_colors': channel_colors,
        'channels': channels,
        'acquisition': acquisition,
    }


def _ome_steps(tf, series_index: int = 0) -> dict:
    """PhysicalSizeX/Y/Z and TimeIncrement from OME-XML (with units)."""
    return parse_ome_xml(tf.ome_metadata or '', series_index=series_index)['steps']


def _scanimage_fov_steps(fov, ny: int, nx: int) -> tuple[float | None, float | None]:
    """Derive ``(step_y, step_x)`` in µm from ScanImage ``imagingFovUm`` corner coordinates."""
    if fov is None or ny <= 0 or nx <= 0:
        return None, None
    try:
        pts = [list(pt) for pt in fov]
        if len(pts) < 2 or any(len(pt) < 2 for pt in pts):
            return None, None
        xs = [float(pt[0]) for pt in pts]
        ys = [float(pt[1]) for pt in pts]
    except (TypeError, ValueError):
        return None, None
    fov_x = max(xs) - min(xs)
    fov_y = max(ys) - min(ys)
    sx = fov_x / nx if fov_x > 0 else None
    sy = fov_y / ny if fov_y > 0 else None
    if sx is not None and not (0.001 <= sx <= 1000):
        sx = None
    if sy is not None and not (0.001 <= sy <= 1000):
        sy = None
    return sy, sx


def from_tiff(tf, data_shape: tuple, series_index: int = 0) -> AxesInfo:
    """Axes from an open ``tifffile.TiffFile`` whose series ``series_index`` gave ``data_shape``."""
    series = tf.series[series_index]
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
        parsed = parse_ome_xml(tf.ome_metadata or '', series_index=series_index)
        for role, (step, unit) in parsed['steps'].items():
            i = info.index(role)
            if i is not None:
                axes[i].step, axes[i].unit = step, unit
        notes.extend(parsed['notes'])
    else:
        if tf.is_imagej:
            info.source = 'ImageJ metadata'
        elif tf.is_scanimage:
            info.source = 'ScanImage metadata'
        elif getattr(tf, 'is_micromanager', False):
            info.source = 'Micro-Manager metadata'

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
            sp = abs(float(ij['spacing']))
            if sp > 0:
                axes[info.index('Z')].step = sp
                axes[info.index('Z')].unit = _unit(ij.get('unit')) or 'µm'
        fps_val = float(ij['fps']) if ij.get('fps') else 0.0
        fint_val = float(ij['finterval']) if ij.get('finterval') else 0.0
        rate = (fps_val if fps_val > 0 else None) or (1.0 / fint_val if fint_val > 0 else None)
        if (info.index('T') is None and info.index('Z') is not None and rate
                and not ij.get('spacing') and not ij.get('frames')):
            # ImageJ often saves movies as plain "slices"; a frame rate without a
            # z-spacing says the stack axis is really time.
            axes[info.index('Z')].role = 'T'
            notes.append(f'ImageJ stored the frames as slices; fps = {float(rate):g} says this is time '
                         '(change in Image ▸ Properties if not)')
        if fps_val > 0 and fint_val <= 0 and info.index('T') is not None:
            axes[info.index('T')].step = 1.0 / fps_val
            axes[info.index('T')].unit = 's'
        if fint_val > 0 and info.index('T') is not None:
            axes[info.index('T')].step = fint_val
            axes[info.index('T')].unit = _unit(ij.get('tunit')) or 's'
        if tf.is_scanimage:
            fd = (tf.scanimage_metadata or {}).get('FrameData', {})
            if info.index('T') is not None:
                period = fd.get('SI.hRoiManager.scanFramePeriod')
                if not period and fd.get('SI.hRoiManager.scanFrameRate'):
                    try:
                        fr = float(fd['SI.hRoiManager.scanFrameRate'])
                        period = 1.0 / fr if fr > 0 else None
                    except (TypeError, ValueError):
                        period = None
                if period and float(period) > 0:
                    axes[info.index('T')].step, axes[info.index('T')].unit = float(period), 's'
            y_idx, x_idx = info.index('Y'), info.index('X')
            if y_idx is not None and x_idx is not None:
                sy, sx = _scanimage_fov_steps(
                    fd.get('SI.hRoiManager.imagingFovUm'), axes[y_idx].size, axes[x_idx].size
                )
                if sy:
                    axes[y_idx].step, axes[y_idx].unit = sy, 'µm'
                if sx:
                    axes[x_idx].step, axes[x_idx].unit = sx, 'µm'
            z_idx = info.index('Z')
            if z_idx is not None:
                z_step = fd.get('SI.hStackManager.actualStackZStepSize') or fd.get('SI.hStackManager.stackZStepSize')
                try:
                    z_val = abs(float(z_step)) if z_step is not None else 0.0
                except (TypeError, ValueError):
                    z_val = 0.0
                if z_val > 0:
                    axes[z_idx].step, axes[z_idx].unit = z_val, 'µm'

        if getattr(tf, 'is_micromanager', False):
            summary = (getattr(tf, 'micromanager_metadata', None) or {}).get('Summary', {})
            if isinstance(summary, dict):
                px_um = summary.get('PixelSize_um')
                try:
                    if px_um and float(px_um) > 0:
                        for r in 'YX':
                            idx = info.index(r)
                            if idx is not None and axes[idx].step is None:
                                axes[idx].step, axes[idx].unit = float(px_um), 'µm'
                except (TypeError, ValueError):
                    pass
                z_um = summary.get('z-step_um')
                try:
                    if z_um and float(z_um) > 0 and info.index('Z') is not None:
                        z_idx = info.index('Z')
                        if axes[z_idx].step is None:
                            axes[z_idx].step, axes[z_idx].unit = float(z_um), 'µm'
                except (TypeError, ValueError):
                    pass
                interval_ms = summary.get('Interval_ms')
                try:
                    if interval_ms and float(interval_ms) > 0 and info.index('T') is not None:
                        t_idx = info.index('T')
                        if axes[t_idx].step is None:
                            ms = float(interval_ms)
                            if ms >= 1000.0:
                                axes[t_idx].step, axes[t_idx].unit = ms / 1000.0, 's'
                            else:
                                axes[t_idx].step, axes[t_idx].unit = ms, 'ms'
                except (TypeError, ValueError):
                    pass

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


# -------------------------------------------------------------------- CZI

def _normalize_czi_step(role: str, step: float, unit: str | None) -> tuple[float, str | None]:
    """Convert CZI SI meter distances to microscopy units (µm / nm)."""
    u = _unit(unit)
    if role in ('X', 'Y', 'Z') and (u == 'm' or (u is None and 0 < step < 1e-2)):
        um = step * 1e6
        if 0 < um < 0.01:
            return um * 1e3, 'nm'
        return um, 'µm'
    return step, u or DEFAULT_UNITS.get(role)


def parse_czi_metadata(xml: str) -> dict:
    """Extract calibration steps, channel names/colors, and acquisition info from CZI XML."""
    import xml.etree.ElementTree as ET

    out: dict = {'steps': {}, 'colors': [], 'channel_names': [], 'acquisition': {}}
    if not xml:
        return out
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return out

    steps: dict[str, tuple[float, str]] = {}
    colors: list[str] = []
    channel_names: list[str] = []
    channels_info: list[dict] = []

    for dist in root.findall('.//Scaling/Items/Distance'):
        axis_id = (dist.get('Id') or '').upper()
        val_el = dist.find('Value')
        if axis_id in ('X', 'Y', 'Z') and val_el is not None and val_el.text:
            try:
                val = float(val_el.text)
                if val > 0:
                    steps[axis_id] = _normalize_czi_step(axis_id, val, 'm')
            except ValueError:
                pass

    for path in ('.//Dimensions/T/StartTime', './/Dimensions/T/Positions/Interval/Increment', './/TimeIncrement'):
        el = root.find(path)
        if path.endswith('StartTime') or el is None or not el.text:
            continue
        try:
            val = float(el.text)
            if val > 0:
                steps['T'] = (val, 's')
                break
        except ValueError:
            pass

    for ch in root.findall('.//Dimensions/Channels/Channel'):
        color_el = ch.find('Color')
        color = (color_el.text if color_el is not None and color_el.text else ch.get('Color') or '').strip()
        if color:
            colors.append(color)
        name = (ch.get('Name') or '').strip()
        fluor_el = ch.find('Fluor')
        fluor = (fluor_el.text if fluor_el is not None and fluor_el.text else ch.get('Fluor') or '').strip()
        channel_names.append(name or fluor)
        entry: dict = {}
        if name:
            entry['name'] = name
        if fluor:
            entry['fluor'] = fluor
        if color:
            entry['color'] = color
        for tag, key_out in (
            ('ExcitationWavelength', 'excitation_nm'),
            ('EmissionWavelength', 'emission_nm'),
            ('ExposureTime', 'exposure_ms'),
        ):
            sub = ch.find(tag)
            if sub is not None and sub.text:
                try:
                    entry[key_out] = float(sub.text)
                except ValueError:
                    pass
        if entry:
            channels_info.append(entry)

    if not colors:
        for ch in root.findall('.//DisplaySetting/Channels/Channel'):
            color_el = ch.find('Color')
            color = (color_el.text if color_el is not None and color_el.text else '').strip()
            if color:
                colors.append(color)

    acq: dict = {}
    date_el = root.find('.//Information/Image/AcquisitionDateAndTime')
    if date_el is not None and date_el.text and date_el.text.strip():
        acq['acquisition_date'] = date_el.text.strip()
    mic_el = root.find('.//Information/Instrument/Microscopes/Microscope')
    if mic_el is not None:
        mic_name = (mic_el.get('Name') or '').strip()
        sys_el = mic_el.find('System')
        sys_name = (sys_el.text or '').strip() if sys_el is not None and sys_el.text else ''
        mic = {k: v for k, v in (('name', mic_name), ('system', sys_name)) if v}
        if mic:
            acq['microscope'] = mic
    obj_el = root.find('.//Information/Instrument/Objectives/Objective')
    if obj_el is not None:
        obj: dict = {}
        if obj_el.get('Name'):
            obj['name'] = obj_el.get('Name').strip()
        for tag, key_out in (('NominalMagnification', 'magnification'), ('LensNA', 'na')):
            sub = obj_el.find(tag)
            if sub is not None and sub.text:
                try:
                    obj[key_out] = float(sub.text)
                except ValueError:
                    pass
        imm_el = obj_el.find('Immersion')
        if imm_el is not None and imm_el.text and imm_el.text.strip():
            obj['immersion'] = imm_el.text.strip()
        if obj:
            acq['objective'] = obj
    if channels_info:
        acq['channels'] = channels_info

    out['steps'] = steps
    out['colors'] = colors
    out['channel_names'] = channel_names
    out['acquisition'] = acq
    return out


def from_czi_xml(xml: str) -> tuple[dict[str, tuple[float, str]], list[str]]:
    """Extract ``({role: (step, unit)}, [channel_colors])`` from Zeiss CZI XML."""
    parsed = parse_czi_metadata(xml)
    return parsed['steps'], parsed['colors']


def from_czi(
    codes: str,
    data_shape: tuple,
    coord_scales: dict | None = None,
    coord_units: dict | None = None,
    xml: str = '',
) -> AxesInfo:
    """AxesInfo from a CZI scene's axis codes, coordinate scales/units and XML."""
    notes: list[str] = []
    if len(codes) != len(data_shape):
        info = guess(data_shape)
        info.notes.append(f'CZI axes {codes} do not match data shape {data_shape}')
        return info
    if codes and codes[-1] in ('S', '0') and data_shape[-1] in (3, 4):
        codes, data_shape = codes[:-1], data_shape[:-1]
    axes = [Axis(TIFF_ROLES.get(c, '?'), int(n)) for c, n in zip(codes, data_shape)]
    info = AxesInfo(axes, 'CZI metadata', True, notes)

    xml_steps, _ = from_czi_xml(xml)
    scales = coord_scales or {}
    units = coord_units or {}
    for role in ('X', 'Y', 'Z', 'T'):
        idx = info.index(role)
        if idx is None:
            continue
        if role in scales and scales[role] is not None and float(scales[role]) > 0:
            step, unit = _normalize_czi_step(role, float(scales[role]), units.get(role))
            axes[idx].step, axes[idx].unit = step, unit
        elif role in xml_steps:
            axes[idx].step, axes[idx].unit = xml_steps[role]

    if '?' in info.roles[:-2]:
        info.confident = False
        notes.append(f'CZI says {codes}: stack axis meaning unknown')
    if info.index('Z') is not None and axes[info.index('Z')].step is None:
        notes.append('z-step not in file')
    if info.problems(tuple(a.size for a in axes)):
        fallback = guess(data_shape)
        fallback.notes = notes + ['CZI axes unusable: ' + '; '.join(info.problems())]
        return fallback
    return info


# --------------------------------------------------------------- OME-Zarr

def from_ome_zarr(attrs: dict, data_shape: tuple) -> tuple[AxesInfo, list[str]]:
    """Parse OME-NGFF ``multiscales`` axes/scale and optional ``omero`` channel hex colors."""
    multiscales = attrs.get('multiscales') or []
    ms = multiscales[0] if multiscales and isinstance(multiscales[0], dict) else {}
    raw_axes = ms.get('axes') or []
    datasets = ms.get('datasets') or []
    scale_vals: list[float] | None = None
    if datasets and isinstance(datasets[0], dict):
        for tr in datasets[0].get('coordinateTransformations') or []:
            if isinstance(tr, dict) and tr.get('type') == 'scale' and isinstance(tr.get('scale'), list):
                scale_vals = [float(s) for s in tr['scale']]
                break

    if len(raw_axes) != len(data_shape):
        info = guess(data_shape)
        info.source = 'OME-Zarr'
        return info, []

    axes = []
    for i, (ax_def, size) in enumerate(zip(raw_axes, data_shape)):
        if isinstance(ax_def, str):
            name, unit = ax_def.upper(), None
        else:
            name = str(ax_def.get('name', '?')).upper()
            unit = _unit(ax_def.get('unit'))
        role = TIFF_ROLES.get(name, '?')
        step = None
        if scale_vals is not None and i < len(scale_vals) and role in ('T', 'Z', 'Y', 'X'):
            val = float(scale_vals[i])
            if val > 0 and not (val == 1.0 and unit is None):
                step, unit = _normalize_czi_step(role, val, unit) if unit == 'm' else (val, unit or DEFAULT_UNITS.get(role))
        axes.append(Axis(role, int(size), step, unit if step else None))

    confident = '?' not in ''.join(a.role for a in axes[:-2])
    info = AxesInfo(axes, 'OME-Zarr metadata', confident)
    if info.problems(data_shape):
        info = guess(data_shape)

    colors: list[str] = []
    omero = attrs.get('omero') or {}
    for ch in omero.get('channels') or []:
        if isinstance(ch, dict) and ch.get('color'):
            colors.append(str(ch['color']))
    return info, colors


# ----------------------------------------------------------- folder memory

def _memory_key(path: Path, shape: tuple) -> str:
    return f'{path.parent}|{len(shape)}'


def remember(path: str | Path, shape: tuple, info: AxesInfo) -> None:
    """Store roles/steps for files of this dimensionality in this folder."""
    try:
        MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        if MEMORY_FILE.exists():
            try:
                loaded = json.loads(MEMORY_FILE.read_text())
                memory = loaded if isinstance(loaded, dict) else {}
            except (OSError, ValueError):
                memory = {}
        else:
            memory = {}
        memory[_memory_key(Path(path), shape)] = [
            {'role': a.role, 'step': a.step, 'unit': a.unit} for a in info.axes
        ]
        MEMORY_FILE.write_text(json.dumps(memory, indent=1))
    except (OSError, TypeError, ValueError):
        pass  # memory is a convenience; never fail an action over it


def recall(path: str | Path, shape: tuple) -> AxesInfo | None:
    try:
        memory = json.loads(MEMORY_FILE.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(memory, dict):
        return None
    saved = memory.get(_memory_key(Path(path), shape))
    if not isinstance(saved, list) or len(saved) != len(shape):
        return None
    try:
        axes = [Axis(s['role'], int(n), s.get('step'), s.get('unit')) for s, n in zip(saved, shape)]
    except (TypeError, KeyError, ValueError):
        return None
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
