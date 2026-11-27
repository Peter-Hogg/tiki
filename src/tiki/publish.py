"""Publication & talk exports and Orthogonal Views.

- ``File ▸ Export ▸ Snapshot (PNG / SVG)…`` & ``Analyze ▸ Tools ▸ Scale Bar…``:
  flatten visible channels with current LUTs/contrast, optional ROI overlays,
  and a calibrated burned-in scale bar to PNG or vector SVG.
- ``File ▸ Export ▸ Export Movie (MP4 / GIF / AVI)…``: render a T or Z stack
  with current LUTs/contrast, optional scale bar, timestamp, and ROIs.
- ``Image ▸ Stacks ▸ Make Montage…``: tile slices/frames into a figure grid.
- ``Image ▸ Stacks ▸ Orthogonal Views`` (``Ctrl+Shift+H``): linked XY / XZ / YZ
  views with crosshair cursor and physical voxel aspect ratios.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Polygon, Rectangle
from napari.layers import Image, Shapes
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from . import axes as axes_mod
from . import nd, ops, style
from . import roi as roi_mod

CORNERS = [
    ('Lower Right', 'lower right'),
    ('Lower Left', 'lower left'),
    ('Upper Right', 'upper right'),
    ('Upper Left', 'upper left'),
]
BAR_COLORS = [
    ('White', 'white'),
    ('Black', 'black'),
    ('Yellow', 'yellow'),
    ('Cyan', 'cyan'),
    ('Magenta', 'magenta'),
]


def _error(message: str, fix: str | None = None):
    from .app import TikiError

    raise TikiError(message, fix=fix)


def _form(parent, title: str):
    dlg = QDialog(parent)
    style.apply(dlg)
    dlg.setWindowTitle(title)
    return dlg, QFormLayout(dlg)


def _buttons(dlg: QDialog, form: QFormLayout):
    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)
    form.addRow(buttons)
    return buttons


def _parse_indices(spec: str, n: int) -> list[int]:
    """Expand ``ops.parse_slice_spec`` (which may return a ``slice`` or ``list``) into integer indices."""
    parsed = ops.parse_slice_spec(spec, n)
    if isinstance(parsed, slice):
        return list(range(*parsed.indices(n)))
    return [int(i) for i in parsed]


# ------------------------------------------------------- rendering helpers

def visible_image_layers(viewer) -> list[Image]:
    """Visible non-preview Image layers in ``viewer``."""
    return [
        ly for ly in viewer.layers
        if isinstance(ly, Image) and ly.visible and ly.name not in roi_mod.PREVIEW_NAMES
    ]


def _step_offset(layer: Image, step_len: int) -> int:
    """Index in ``viewer.dims.current_step`` corresponding to axis 0 of ``layer``."""
    return step_len - layer.ndim


def _layer_plane(layer: Image, step: tuple[int, ...]) -> np.ndarray:
    """Extract the 2D (or 2D RGB) plane of ``layer`` at ``step``."""
    data = layer.data
    n_lead = data.ndim - (3 if layer.rgb else 2)
    if n_lead <= 0:
        return np.asarray(data)
    offset = _step_offset(layer, len(step))
    idx = []
    for d in range(n_lead):
        s = int(step[offset + d]) if 0 <= offset + d < len(step) else 0
        idx.append(int(np.clip(s, 0, data.shape[d] - 1)))
    return np.asarray(data[tuple(idx)])


def render_plane_rgb(viewer, step: tuple[int, ...] | None = None) -> np.ndarray:
    """Composite visible Image layers at ``step`` into uint8 RGB ``(H, W, 3)``."""
    layers = visible_image_layers(viewer)
    if not layers:
        target = roi_mod.target_image(viewer)
        if target is None:
            _error('No image open.')
        layers = [target]
    if step is None:
        step = tuple(viewer.dims.current_step)

    ref_shape = roi_mod.spatial_shape(layers[0])
    acc = np.zeros(ref_shape + (3,), dtype=np.float32)
    scalar_batch = []

    for ly in layers:
        if roi_mod.spatial_shape(ly) != ref_shape:
            continue
        plane = _layer_plane(ly, step)
        opacity = float(getattr(ly, 'opacity', 1.0))
        if ly.rgb:
            rgb = plane[..., :3].astype(np.float32)
            if np.issubdtype(plane.dtype, np.integer):
                hi = float(np.iinfo(plane.dtype).max)
                rgb = rgb / (hi if hi > 0 else 255.0)
            else:
                rgb = np.clip(np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
            acc += rgb * opacity
        else:
            lo, hi = ly.contrast_limits
            scalar_batch.append((plane, ly.colormap, (float(lo), float(hi)), opacity))

    if scalar_batch:
        for plane, cmap, (lo, hi), opacity in scalar_batch:
            comp = ops.composite_rgb([(plane, cmap, (lo, hi))]).astype(np.float32) / 255.0
            acc += comp * opacity

    return np.clip(np.round(acc * 255.0), 0, 255).astype(np.uint8)


def _xy_calibration(image: Image) -> tuple[float, str]:
    """Return ``(pixel_size_x, unit)`` for ``image`` (defaults to ``(1.0, 'px')``)."""
    info = axes_mod.get(image)
    ax = info.axes[-1]
    scale = float(ax.scale) if ax.scale and ax.scale > 0 else 1.0
    unit = ax.unit or 'px'
    return scale, unit


def auto_scale_bar_length(width_px: int, pixel_size: float) -> float:
    """Choose a clean round scale bar length (~20% of image width)."""
    target = max(width_px * pixel_size * 0.2, 1e-9)
    exp = np.floor(np.log10(target))
    base = 10.0 ** exp
    for mult in (1.0, 2.0, 5.0, 10.0):
        cand = mult * base
        if cand >= target * 0.75:
            return float(cand)
    return float(base)


def _format_length(length: float, unit: str) -> str:
    txt = f'{length:g}'
    return f'{txt} {unit}' if unit else txt


def draw_scale_bar(
    ax,
    width_px: int,
    height_px: int,
    pixel_size: float,
    unit: str,
    length: float,
    *,
    corner: str = 'lower right',
    color: str = 'white',
    thickness: float = 4.0,
    font_size: float = 10.0,
    show_label: bool = True,
) -> None:
    """Draw a calibrated scale bar onto a matplotlib ``Axes``."""
    bar_px = max(float(length) / max(float(pixel_size), 1e-12), 1.0)
    bar_px = min(bar_px, width_px * 0.9)
    pad_x = max(width_px * 0.04, 6.0)
    pad_y = max(height_px * 0.05, 8.0)
    bar_h = max(float(thickness), 1.0)

    is_right = 'right' in corner
    is_lower = 'lower' in corner

    x0 = (width_px - pad_x - bar_px) if is_right else pad_x
    y0 = (height_px - pad_y - bar_h) if is_lower else pad_y

    rect = Rectangle((x0, y0), bar_px, bar_h, facecolor=color, edgecolor='none', zorder=10)
    ax.add_patch(rect)

    if show_label:
        label = _format_length(length, unit)
        tx = x0 + bar_px / 2.0
        if is_lower:
            ty = y0 - max(bar_h * 0.8, 3.0)
            va = 'bottom'
        else:
            ty = y0 + bar_h + max(bar_h * 0.8, 3.0)
            va = 'top'
        ax.text(
            tx, ty, label,
            color=color, fontsize=font_size, fontweight='bold',
            ha='center', va=va, zorder=11,
        )


def draw_timestamp(
    ax,
    text: str,
    *,
    corner: str = 'upper left',
    color: str = 'white',
    font_size: float = 10.0,
) -> None:
    """Draw a compact timestamp / slice label in a corner of ``ax``."""
    if not text:
        return
    is_right = 'right' in corner
    is_lower = 'lower' in corner
    x = 0.97 if is_right else 0.03
    y = 0.04 if is_lower else 0.96
    ha = 'right' if is_right else 'left'
    va = 'bottom' if is_lower else 'top'
    bg = 'black' if color != 'black' else 'white'
    ax.text(
        x, y, text,
        transform=ax.transAxes,
        color=color, fontsize=font_size, fontweight='bold',
        ha=ha, va=va, zorder=12,
        bbox={'facecolor': bg, 'alpha': 0.45, 'edgecolor': 'none', 'pad': 2.0},
    )


def _draw_shape(ax, verts: np.ndarray, shape_type: str, color: str, lw: float = 1.5, label: str | None = None):
    pts = np.asarray(verts, dtype=float)[:, -2:]
    if shape_type == 'ellipse':
        pts = roi_mod._ellipse_polygon(pts)
    xy = pts[:, ::-1]  # (y, x) -> (x, y)
    if shape_type in ('line', 'path'):
        ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=lw, zorder=8)
    else:
        poly = Polygon(xy, closed=True, fill=False, edgecolor=color, linewidth=lw, zorder=8)
        ax.add_patch(poly)
    if label:
        cx, cy = float(xy[:, 0].mean()), float(xy[:, 1].mean())
        ax.text(
            cx, cy, label,
            color=color, fontsize=8, fontweight='bold',
            ha='center', va='center', zorder=9,
            bbox={'facecolor': 'black', 'alpha': 0.4, 'edgecolor': 'none', 'pad': 1.0},
        )


def draw_rois_on_axes(
    ax,
    viewer,
    *,
    include_selection: bool = True,
    include_manager: bool = True,
    roi_manager=None,
    show_labels: bool = False,
) -> None:
    """Overlay current ROI shapes and ROI Manager ROIs onto ``ax``."""
    if include_selection and roi_mod.ROI_NAME in viewer.layers:
        layer = viewer.layers[roi_mod.ROI_NAME]
        if isinstance(layer, Shapes) and layer.visible:
            for verts, stype in zip(layer.data, layer.shape_type):
                _draw_shape(ax, verts, stype, color='yellow', lw=1.5)
    if include_manager and roi_manager is not None and getattr(roi_manager, 'rois', None):
        for r in roi_manager.rois:
            _draw_shape(
                ax, r.yx, r.kind,
                color='cyan', lw=1.2,
                label=r.name if show_labels else None,
            )


def format_step_label(viewer, image: Image, axis: int, idx: int) -> str:
    """Format a human-readable slice/time label (e.g. ``t = 1.5 s`` or ``Z 12``)."""
    info = axes_mod.get(image)
    ax = info.axes[axis] if axis < len(info.axes) else None
    if ax is None:
        return f'frame {idx + 1}'
    val = idx * ax.scale if ax.scale else idx
    if ax.role == 'T' and ax.unit:
        return f't = {val:.3g} {ax.unit}'
    if ax.unit:
        return f'{ax.role or "Z"} = {val:.3g} {ax.unit}'
    return f'{ax.role or "frame"} {idx + 1}'


def _render_figure(
    rgb: np.ndarray,
    *,
    dpi: int = 100,
    viewer=None,
    pixel_size: float = 1.0,
    unit: str = 'px',
    scale_bar: bool = False,
    bar_length: float | None = None,
    bar_corner: str = 'lower right',
    bar_color: str = 'white',
    bar_thickness: float = 4.0,
    bar_font_size: float = 10.0,
    show_bar_label: bool = True,
    show_rois: bool = False,
    show_roi_labels: bool = False,
    roi_manager=None,
    timestamp: str | None = None,
    timestamp_corner: str = 'upper left',
) -> Figure:
    h, w = rgb.shape[:2]
    fig = Figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.imshow(rgb, interpolation='nearest')
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.axis('off')

    if show_rois and viewer is not None:
        draw_rois_on_axes(
            ax, viewer,
            include_selection=True,
            include_manager=True,
            roi_manager=roi_manager,
            show_labels=show_roi_labels,
        )
    if scale_bar:
        length = bar_length if bar_length and bar_length > 0 else auto_scale_bar_length(w, pixel_size)
        draw_scale_bar(
            ax, w, h, pixel_size, unit, length,
            corner=bar_corner, color=bar_color,
            thickness=bar_thickness, font_size=bar_font_size,
            show_label=show_bar_label,
        )
    if timestamp:
        draw_timestamp(ax, timestamp, corner=timestamp_corner, color=bar_color, font_size=bar_font_size)
    return fig


def _figure_to_rgb(fig: Figure) -> np.ndarray:
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())
    return np.ascontiguousarray(buf[..., :3])


def save_snapshot(
    viewer,
    path: str | Path,
    *,
    dpi: int = 300,
    scale_bar: bool = True,
    bar_length: float | None = None,
    bar_corner: str = 'lower right',
    bar_color: str = 'white',
    bar_thickness: float = 4.0,
    bar_font_size: float = 10.0,
    show_bar_label: bool = True,
    show_rois: bool = True,
    show_roi_labels: bool = False,
    roi_manager=None,
    timestamp: str | None = None,
    timestamp_corner: str = 'upper left',
) -> Path:
    """Save a publication-ready PNG or SVG snapshot of the current view."""
    out = Path(path)
    image = roi_mod.target_image(viewer)
    if image is None:
        _error('No image open.')
    rgb = render_plane_rgb(viewer)
    pixel_size, unit = _xy_calibration(image)
    h, w = rgb.shape[:2]
    # Keep physical figure size reasonable (~6 in wide) so fonts and vector SVGs look crisp
    fig_w = 6.0
    fig_h = fig_w * (h / max(w, 1))
    fig = Figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.imshow(rgb, interpolation='nearest')
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.axis('off')

    if show_rois:
        draw_rois_on_axes(
            ax, viewer,
            include_selection=True,
            include_manager=True,
            roi_manager=roi_manager,
            show_labels=show_roi_labels,
        )
    if scale_bar:
        length = bar_length if bar_length and bar_length > 0 else auto_scale_bar_length(w, pixel_size)
        draw_scale_bar(
            ax, w, h, pixel_size, unit, length,
            corner=bar_corner, color=bar_color,
            thickness=bar_thickness, font_size=bar_font_size,
            show_label=show_bar_label,
        )
    if timestamp:
        draw_timestamp(ax, timestamp, corner=timestamp_corner, color=bar_color, font_size=bar_font_size)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=dpi, bbox_inches='tight', pad_inches=0)
    return out


# --------------------------------------------------------- movie export

def find_ffmpeg() -> str | None:
    """Return path to ``ffmpeg`` binary (system PATH or ``imageio-ffmpeg``)."""
    exe = shutil.which('ffmpeg')
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def export_movie(
    viewer,
    path: str | Path,
    *,
    axis: int,
    indices: list[int],
    fps: float = 15.0,
    scale_bar: bool = False,
    bar_length: float | None = None,
    bar_corner: str = 'lower right',
    bar_color: str = 'white',
    bar_thickness: float = 4.0,
    bar_font_size: float = 10.0,
    show_timestamp: bool = True,
    timestamp_corner: str = 'upper left',
    show_rois: bool = False,
    show_roi_labels: bool = False,
    roi_manager=None,
    progress=None,
) -> Path:
    """Export a stack along ``axis`` as ``.gif``, ``.mp4``, or ``.avi``."""
    out = Path(path)
    ext = out.suffix.lower()
    if ext not in ('.gif', '.mp4', '.avi'):
        _error(f'Unsupported movie format {ext!r}. Use .gif, .mp4, or .avi.')
    if not indices:
        _error('No frames selected for movie export.')

    image = roi_mod.target_image(viewer)
    if image is None:
        _error('No image open.')
    pixel_size, unit = _xy_calibration(image)
    base_step = list(viewer.dims.current_step)
    step_axis = _step_offset(image, len(base_step)) + axis
    needs_overlay = bool(scale_bar or show_timestamp or show_rois)

    def _frame(frame_idx: int) -> np.ndarray:
        step = list(base_step)
        if 0 <= step_axis < len(step):
            step[step_axis] = int(frame_idx)
        rgb = render_plane_rgb(viewer, tuple(step))
        if not needs_overlay:
            return rgb
        ts = format_step_label(viewer, image, axis, frame_idx) if show_timestamp else None
        fig = _render_figure(
            rgb,
            dpi=100,
            viewer=viewer,
            pixel_size=pixel_size,
            unit=unit,
            scale_bar=scale_bar,
            bar_length=bar_length,
            bar_corner=bar_corner,
            bar_color=bar_color,
            bar_thickness=bar_thickness,
            bar_font_size=bar_font_size,
            show_rois=show_rois,
            show_roi_labels=show_roi_labels,
            roi_manager=roi_manager,
            timestamp=ts,
            timestamp_corner=timestamp_corner,
        )
        return _figure_to_rgb(fig)

    out.parent.mkdir(parents=True, exist_ok=True)
    fps = max(float(fps), 0.1)

    if ext == '.gif':
        from PIL import Image as PILImage

        pil_frames = []
        for n, idx in enumerate(indices, 1):
            pil_frames.append(PILImage.fromarray(_frame(idx), mode='RGB'))
            if progress is not None and progress(n, len(indices)) is False:
                return out
        duration_ms = max(int(round(1000.0 / fps)), 10)
        pil_frames[0].save(
            str(out),
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
        )
        return out

    # MP4 / AVI via ffmpeg subprocess pipe
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        _error(
            'MP4/AVI export requires ffmpeg.\n\n'
            'Install system ffmpeg (e.g. `sudo apt install ffmpeg`) or '
            '`pip install imageio-ffmpeg`.\n'
            'GIF export works without ffmpeg.'
        )

    first = _frame(indices[0])
    h, w = first.shape[:2]
    cmd = [
        ffmpeg, '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-s', f'{w}x{h}',
        '-r', f'{fps:g}',
        '-i', '-',
    ]
    if ext == '.mp4':
        cmd += [
            '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '18',
            '-movflags', '+faststart',
        ]
    else:  # .avi
        cmd += ['-c:v', 'mpeg4', '-q:v', '3']
    cmd.append(str(out))

    import threading

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    stderr_chunks: list[bytes] = []

    def _drain_stderr() -> None:
        if proc.stderr is None:
            return
        try:
            while True:
                chunk = proc.stderr.read(4096)
                if not chunk:
                    break
                stderr_chunks.append(chunk)
        finally:
            try:
                proc.stderr.close()
            except Exception:
                pass

    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_thread.start()
    cancelled = False
    broken_pipe = False
    try:
        try:
            proc.stdin.write(first.tobytes())
        except BrokenPipeError:
            broken_pipe = True
        if not broken_pipe:
            if progress is not None and progress(1, len(indices)) is False:
                cancelled = True
            else:
                for n, idx in enumerate(indices[1:], 2):
                    frame = _frame(idx)
                    try:
                        proc.stdin.write(frame.tobytes())
                    except BrokenPipeError:
                        broken_pipe = True
                        break
                    if progress is not None and progress(n, len(indices)) is False:
                        cancelled = True
                        break
        try:
            proc.stdin.close()
        except OSError:
            pass
        rc = proc.wait()
        stderr_thread.join(timeout=5.0)
        stderr = b''.join(stderr_chunks).decode('utf-8', errors='replace')
        if cancelled:
            return out
        if rc != 0 or broken_pipe:
            _error(f'ffmpeg failed (exit {rc}):\n{stderr[-400:]}')
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        stderr_thread.join(timeout=2.0)
    return out


# --------------------------------------------------------------- dialogs

def snapshot_dialog(tiki, *, focus_scale_bar: bool = False) -> None:
    """``File ▸ Export ▸ Snapshot (PNG / SVG)…`` & ``Analyze ▸ Tools ▸ Scale Bar…``."""
    viewer, image = tiki.require_image()
    pixel_size, unit = _xy_calibration(image)
    h, w = roi_mod.spatial_shape(image)
    default_len = auto_scale_bar_length(w, pixel_size)

    title = 'Scale Bar & Snapshot' if focus_scale_bar else 'Export Snapshot (PNG / SVG)'
    dlg, form = _form(tiki, title)

    fmt_box = QComboBox()
    fmt_box.addItem('PNG (raster)', '.png')
    fmt_box.addItem('SVG (vector)', '.svg')
    dpi_spin = QSpinBox(minimum=72, maximum=1200, value=300, suffix=' dpi')
    fmt_box.currentIndexChanged.connect(lambda _: dpi_spin.setEnabled(fmt_box.currentData() == '.png'))

    live_bar = QCheckBox('Show live napari canvas scale bar')
    live_bar.setChecked(bool(viewer.scale_bar.visible))

    burn_bar = QCheckBox('Burn in scale bar')
    burn_bar.setChecked(True)
    length_spin = QDoubleSpinBox(
        minimum=1e-6, maximum=1e9, decimals=4, value=default_len, suffix=f' {unit}',
    )
    corner_box = QComboBox()
    for label, val in CORNERS:
        corner_box.addItem(label, val)
    color_box = QComboBox()
    for label, val in BAR_COLORS:
        color_box.addItem(label, val)
    thick_spin = QSpinBox(minimum=1, maximum=40, value=max(int(round(h * 0.015)), 2), suffix=' px')
    font_spin = QSpinBox(minimum=6, maximum=36, value=11, suffix=' pt')
    show_label = QCheckBox('Show length label')
    show_label.setChecked(True)

    show_rois = QCheckBox('Include ROI overlays')
    show_rois.setChecked(True)
    show_roi_labels = QCheckBox('Show ROI Manager labels')
    show_roi_labels.setChecked(False)

    show_ts = QCheckBox('Show slice / timestamp')
    n_lead = image.data.ndim - (3 if image.rgb else 2)
    show_ts.setChecked(False)
    show_ts.setEnabled(n_lead > 0)

    def _toggle_bar(on: bool):
        for widget in (length_spin, corner_box, color_box, thick_spin, font_spin, show_label):
            widget.setEnabled(on)

    burn_bar.toggled.connect(_toggle_bar)
    _toggle_bar(True)

    form.addRow('Format:', fmt_box)
    form.addRow('Resolution:', dpi_spin)
    if focus_scale_bar:
        form.addRow(live_bar)
    form.addRow(burn_bar)
    form.addRow('Bar length:', length_spin)
    form.addRow('Position:', corner_box)
    form.addRow('Color:', color_box)
    form.addRow('Bar height:', thick_spin)
    form.addRow('Font size:', font_spin)
    form.addRow(show_label)
    form.addRow(show_rois)
    form.addRow(show_roi_labels)
    form.addRow(show_ts)
    _buttons(dlg, form)

    if not dlg.exec_():
        return

    if focus_scale_bar:
        viewer.scale_bar.visible = live_bar.isChecked()

    ext = fmt_box.currentData()
    filt = 'PNG (*.png);;SVG (*.svg)' if ext == '.png' else 'SVG (*.svg);;PNG (*.png)'
    default_name = f'{Path(image.name).stem}_snapshot{ext}'
    path, _ = QFileDialog.getSaveFileName(tiki, 'Save Snapshot', default_name, filt)
    if not path:
        return
    out_path = Path(path)
    if out_path.suffix.lower() not in ('.png', '.svg'):
        out_path = out_path.with_suffix(ext)

    ts = None
    if show_ts.isChecked() and n_lead > 0:
        info = axes_mod.get(image)
        ax_idx = axes_mod.stack_axis(info, 'TZ')
        if ax_idx is None:
            ax_idx = 0
        step_offset = _step_offset(image, len(viewer.dims.current_step))
        cur_idx = int(viewer.dims.current_step[step_offset + ax_idx])
        ts = format_step_label(viewer, image, ax_idx, cur_idx)

    save_snapshot(
        viewer,
        out_path,
        dpi=dpi_spin.value(),
        scale_bar=burn_bar.isChecked(),
        bar_length=length_spin.value(),
        bar_corner=corner_box.currentData(),
        bar_color=color_box.currentData(),
        bar_thickness=float(thick_spin.value()),
        bar_font_size=float(font_spin.value()),
        show_bar_label=show_label.isChecked(),
        show_rois=show_rois.isChecked(),
        show_roi_labels=show_roi_labels.isChecked(),
        roi_manager=tiki.roi_manager,
        timestamp=ts,
    )
    tiki.status(f'Saved snapshot: {out_path.name}')


def export_movie_dialog(tiki) -> None:
    """``File ▸ Export ▸ Export Movie (MP4 / GIF / AVI)…``."""
    viewer, image = tiki.require_image()
    n_lead = image.data.ndim - (3 if image.rgb else 2)
    if n_lead <= 0:
        _error('Need a stack (3D or higher) to export a movie.')

    info = axes_mod.get(image)
    pixel_size, unit = _xy_calibration(image)
    h, w = roi_mod.spatial_shape(image)
    default_len = auto_scale_bar_length(w, pixel_size)

    dlg, form = _form(tiki, 'Export Movie')

    axis_box = QComboBox()
    for d in range(n_lead):
        if info.axes[d].role != 'C':
            axis_box.addItem(f'{info.labels[d]} ({image.data.shape[d]})', d)
    if axis_box.count() == 0:
        for d in range(n_lead):
            axis_box.addItem(f'{info.labels[d]} ({image.data.shape[d]})', d)
    default_ax = axes_mod.stack_axis(info, 'TZ')
    if default_ax is not None and axis_box.findData(default_ax) >= 0:
        axis_box.setCurrentIndex(axis_box.findData(default_ax))

    slices_edit = QLineEdit()
    fps_spin = QDoubleSpinBox(minimum=0.5, maximum=120.0, decimals=1, value=15.0, suffix=' fps')

    def _sync_axis():
        ax_idx = axis_box.currentData()
        if ax_idx is None:
            return
        n_frames = image.data.shape[ax_idx]
        slices_edit.setText(f'1-{n_frames}')
        ax = info.axes[ax_idx]
        if ax.role == 'T' and ax.scale and ax.scale > 0 and ax.unit in ('s', 'sec', 'seconds'):
            fps_spin.setValue(float(np.clip(1.0 / ax.scale, 1.0, 60.0)))

    axis_box.currentIndexChanged.connect(lambda _: _sync_axis())
    _sync_axis()

    fmt_box = QComboBox()
    has_ffmpeg = find_ffmpeg() is not None
    if has_ffmpeg:
        fmt_box.addItem('MP4 (H.264 — PowerPoint / Keynote)', '.mp4')
    fmt_box.addItem('GIF (animated — universal)', '.gif')
    if has_ffmpeg:
        fmt_box.addItem('AVI (MPEG-4)', '.avi')
    else:
        fmt_box.addItem('MP4 (requires ffmpeg)', '.mp4')

    burn_bar = QCheckBox('Burn in scale bar')
    burn_bar.setChecked(True)
    length_spin = QDoubleSpinBox(
        minimum=1e-6, maximum=1e9, decimals=4, value=default_len, suffix=f' {unit}',
    )
    corner_box = QComboBox()
    for label, val in CORNERS:
        corner_box.addItem(label, val)
    color_box = QComboBox()
    for label, val in BAR_COLORS:
        color_box.addItem(label, val)

    show_ts = QCheckBox('Show timestamp / slice')
    show_ts.setChecked(True)
    show_rois = QCheckBox('Include ROI overlays')
    show_rois.setChecked(False)

    form.addRow('Stack axis:', axis_box)
    form.addRow('Frames (e.g. 1-100 or 1-100-2):', slices_edit)
    form.addRow('Format:', fmt_box)
    form.addRow('Playback speed:', fps_spin)
    form.addRow(burn_bar)
    form.addRow('Bar length:', length_spin)
    form.addRow('Bar position:', corner_box)
    form.addRow('Color:', color_box)
    form.addRow(show_ts)
    form.addRow(show_rois)
    _buttons(dlg, form)

    if not dlg.exec_():
        return

    ax_idx = axis_box.currentData()
    n_frames = image.data.shape[ax_idx]
    try:
        indices = _parse_indices(slices_edit.text(), n_frames)
    except ValueError as exc:
        _error(str(exc))

    ext = fmt_box.currentData()
    filters = {
        '.mp4': 'MP4 Video (*.mp4);;GIF (*.gif);;AVI (*.avi)',
        '.gif': 'GIF (*.gif);;MP4 Video (*.mp4);;AVI (*.avi)',
        '.avi': 'AVI Video (*.avi);;MP4 Video (*.mp4);;GIF (*.gif)',
    }
    default_name = f'{Path(image.name).stem}{ext}'
    path, _ = QFileDialog.getSaveFileName(tiki, 'Export Movie', default_name, filters[ext])
    if not path:
        return
    out_path = Path(path)
    if out_path.suffix.lower() not in ('.gif', '.mp4', '.avi'):
        out_path = out_path.with_suffix(ext)

    with tiki.progress('Export Movie', len(indices)) as tick:
        export_movie(
            viewer,
            out_path,
            axis=ax_idx,
            indices=indices,
            fps=fps_spin.value(),
            scale_bar=burn_bar.isChecked(),
            bar_length=length_spin.value(),
            bar_corner=corner_box.currentData(),
            bar_color=color_box.currentData(),
            bar_thickness=max(float(round(h * 0.015)), 2.0),
            show_timestamp=show_ts.isChecked(),
            show_rois=show_rois.isChecked(),
            roi_manager=tiki.roi_manager,
            progress=tick,
        )
    tiki.status(f'Exported movie: {out_path.name} ({len(indices)} frames @ {fps_spin.value():g} fps)')


def make_montage_dialog(tiki) -> None:
    """``Image ▸ Stacks ▸ Make Montage…``."""
    from .layer_select import LayerCombo

    viewer, image = tiki.require_image()
    n_lead = image.data.ndim - (3 if image.rgb else 2)
    if n_lead <= 0:
        _error('Need a stack (3D or higher) to make a montage.')

    dlg, form = _form(tiki, 'Make Montage')
    img_box = LayerCombo(viewer, Image, current=image)

    axis_box = QComboBox()
    slices_edit = QLineEdit()
    cols_spin = QSpinBox(minimum=1, maximum=200, value=1)
    rows_spin = QSpinBox(minimum=1, maximum=200, value=1)
    border_spin = QSpinBox(minimum=0, maximum=100, value=2, suffix=' px')
    flatten_box = QCheckBox('Flatten visible channels (RGB)')
    vis_imgs = visible_image_layers(viewer)
    flatten_box.setChecked(len(vis_imgs) > 1)
    flatten_box.setEnabled(len(vis_imgs) > 1 or image.rgb)

    def _populate_axes(layer):
        axis_box.clear()
        if layer is None:
            return
        nl = layer.data.ndim - (3 if layer.rgb else 2)
        info = axes_mod.get(layer)
        for d in range(nl):
            axis_box.addItem(f'{info.labels[d]} ({layer.data.shape[d]})', d)
        default = axes_mod.stack_axis(info, 'ZT')
        if default is not None and axis_box.findData(default) >= 0:
            axis_box.setCurrentIndex(axis_box.findData(default))

    def _update_grid():
        layer = img_box.layer()
        ax_idx = axis_box.currentData()
        if layer is None or ax_idx is None:
            return
        n_total = layer.data.shape[ax_idx]
        try:
            n = len(_parse_indices(slices_edit.text(), n_total))
        except ValueError:
            n = n_total
        c = max(int(np.ceil(np.sqrt(n))), 1)
        r = max(int(np.ceil(n / c)), 1)
        cols_spin.setValue(c)
        rows_spin.setValue(r)

    def _on_axis_change():
        layer = img_box.layer()
        ax_idx = axis_box.currentData()
        if layer is None or ax_idx is None:
            return
        slices_edit.setText(f'1-{layer.data.shape[ax_idx]}')
        _update_grid()

    img_box.layer_changed.connect(_populate_axes)
    axis_box.currentIndexChanged.connect(lambda _: _on_axis_change())
    slices_edit.editingFinished.connect(_update_grid)
    _populate_axes(image)
    _on_axis_change()

    form.addRow('Image:', img_box)
    form.addRow('Stack axis:', axis_box)
    form.addRow('Slices (e.g. 1-16 or 1-32-2):', slices_edit)
    form.addRow('Columns:', cols_spin)
    form.addRow('Rows:', rows_spin)
    form.addRow('Border width:', border_spin)
    form.addRow(flatten_box)
    _buttons(dlg, form)

    if not dlg.exec_():
        return

    v = viewer
    layer = img_box.layer()
    if layer is None:
        _error('No image selected.')
    ax_idx = axis_box.currentData()
    if ax_idx is None:
        _error('Selected image is not a stack.')
    n_total = layer.data.shape[ax_idx]
    try:
        indices = _parse_indices(slices_edit.text(), n_total)
    except ValueError as exc:
        _error(str(exc))

    n_lead_sel = layer.data.ndim - (3 if layer.rgb else 2)
    base_step = list(v.dims.current_step)
    step_offset = _step_offset(layer, len(base_step))
    is_rgb = bool(flatten_box.isChecked() or layer.rgb)

    if flatten_box.isChecked():
        frames = []
        for idx in indices:
            step = list(base_step)
            if 0 <= step_offset + ax_idx < len(step):
                step[step_offset + ax_idx] = int(idx)
            frames.append(render_plane_rgb(v, tuple(step)))
        stack = np.stack(frames, axis=0)
        montage = ops.make_montage(
            stack, axis=0, cols=cols_spin.value(), rows=rows_spin.value(),
            border=border_spin.value(), border_value=255, rgb=True,
        )
    else:
        # Slice along non-montage leading axes at current viewer step
        idx_tuple = []
        for d in range(n_lead_sel):
            if d == ax_idx:
                idx_tuple.append(slice(None))
            else:
                s = int(base_step[step_offset + d]) if step_offset + d >= 0 else 0
                idx_tuple.append(int(np.clip(s, 0, layer.data.shape[d] - 1)))
        sub = np.asarray(layer.data[tuple(idx_tuple)])
        montage = ops.make_montage(
            sub, axis=0, indices=indices,
            cols=cols_spin.value(), rows=rows_spin.value(),
            border=border_spin.value(), border_value=0, rgb=bool(layer.rgb),
        )

    info = axes_mod.get(layer)
    ay, ax_x = info.axes[-2], info.axes[-1]
    montage_axes = axes_mod.AxesInfo(
        axes=[
            axes_mod.Axis('Y', montage.shape[0], ay.step, ay.unit),
            axes_mod.Axis('X', montage.shape[1], ax_x.step, ax_x.unit),
        ],
        source=info.source,
        confident=True,
    )
    kw: dict = {'axes': montage_axes, 'rgb': is_rgb}
    if not is_rgb:
        kw['colormap'] = layer.colormap
        kw['contrast_limits'] = tuple(layer.contrast_limits)
    tiki.open_data(montage, f'Montage of {layer.name}', **kw)
    tiki.status(f'Created {cols_spin.value()}×{rows_spin.value()} montage ({len(indices)} slices)')


# ------------------------------------------------------- Orthogonal Views

class OrthogonalViewsWindow(QWidget):
    """ImageJ-style linked XY / XZ / YZ orthogonal slice viewer."""

    def __init__(self, viewer, image: Image, z_axis: int, parent=None):
        super().__init__(parent, Qt.Window)
        style.apply(self)
        self.viewer = viewer
        self.image = image
        self.z_axis = z_axis
        self._dragging_ax = None
        self._syncing_viewer = False

        self.setWindowTitle(f'{image.name} — Orthogonal Views')
        self.resize(680, 620)

        self.figure = Figure(figsize=(6.8, 5.8), layout='constrained')
        self.canvas = FigureCanvasQTAgg(self.figure)
        gs = self.figure.add_gridspec(2, 2, width_ratios=[3, 1.2], height_ratios=[3, 1.2])
        self.ax_xy = self.figure.add_subplot(gs[0, 0])
        self.ax_yz = self.figure.add_subplot(gs[0, 1])
        self.ax_xz = self.figure.add_subplot(gs[1, 0])
        self.ax_info = self.figure.add_subplot(gs[1, 1])
        self.ax_info.axis('off')

        for ax, title in ((self.ax_xy, 'XY'), (self.ax_xz, 'XZ'), (self.ax_yz, 'YZ')):
            ax.set_title(title, fontsize=9, pad=3)
            ax.set_xticks([])
            ax.set_yticks([])

        self.coord_label = QLabel()
        self.sync_box = QCheckBox('Sync Z with viewer')
        self.sync_box.setChecked(True)

        bar = QHBoxLayout()
        bar.addWidget(self.coord_label, 1)
        bar.addWidget(self.sync_box)

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas, 1)
        layout.addLayout(bar)

        nz, ny, nx = self._volume_shape()
        self.z = self._viewer_z()
        self.y = ny // 2
        self.x = nx // 2

        self._im_xy = None
        self._im_xz = None
        self._im_yz = None
        self._lines: list = []

        self._cid_press = self.canvas.mpl_connect('button_press_event', self._on_press)
        self._cid_move = self.canvas.mpl_connect('motion_notify_event', self._on_move)
        self._cid_release = self.canvas.mpl_connect('button_release_event', self._on_release)

        self.viewer.dims.events.current_step.connect(self._on_viewer_step)
        self.image.events.contrast_limits.connect(self._on_display_change)
        self.image.events.colormap.connect(self._on_display_change)

        self.refresh()

    def _n_lead(self) -> int:
        return self.image.data.ndim - (3 if self.image.rgb else 2)

    def _step_index_for(self, axis: int) -> int:
        return _step_offset(self.image, len(self.viewer.dims.current_step)) + axis

    def _viewer_z(self) -> int:
        idx = self._step_index_for(self.z_axis)
        nz = self.image.data.shape[self.z_axis]
        if 0 <= idx < len(self.viewer.dims.current_step):
            return int(np.clip(self.viewer.dims.current_step[idx], 0, nz - 1))
        return nz // 2

    def _volume_shape(self) -> tuple[int, int, int]:
        ny, nx = roi_mod.spatial_shape(self.image)
        return int(self.image.data.shape[self.z_axis]), ny, nx

    def _current_volume(self) -> np.ndarray:
        """Return 3D ``(Z, Y, X)`` (or 4D RGB ``(Z, Y, X, 3)``) volume at current non-Z step."""
        n_lead = self._n_lead()
        step = self.viewer.dims.current_step
        offset = _step_offset(self.image, len(step))
        sel = []
        for d in range(n_lead):
            if d == self.z_axis:
                sel.append(slice(None))
            else:
                s = int(step[offset + d]) if 0 <= offset + d < len(step) else 0
                sel.append(int(np.clip(s, 0, self.image.data.shape[d] - 1)))
        return np.asarray(self.image.data[tuple(sel)])

    def _voxel_aspects(self) -> tuple[float, float]:
        """Return ``(xz_aspect, yz_aspect)`` from calibrated voxel sizes in matching units."""
        info = axes_mod.get(self.image)
        az = info.axes[self.z_axis] if self.z_axis < len(info.axes) else None
        ay = info.axes[-2] if len(info.axes) >= 2 else None
        ax = info.axes[-1] if len(info.axes) >= 1 else None
        sy = float(ay.scale) if ay and ay.scale > 0 else 1.0
        sx = float(ax.scale) if ax and ax.scale > 0 else 1.0
        if az and az.step and ay and ay.step:
            sz_y = axes_mod.convert_unit(az.step, az.unit, ay.unit)
        else:
            sz_y = float(az.scale) if az and az.scale > 0 else 1.0
        if az and az.step and ax and ax.step:
            sz_x = axes_mod.convert_unit(az.step, az.unit, ax.unit)
        else:
            sz_x = float(az.scale) if az and az.scale > 0 else 1.0
        # Clamp extreme ratios so thin Z-stacks stay visible and clickable
        xz_aspect = float(np.clip(sz_x / sx, 0.1, 10.0))
        yz_aspect = float(np.clip(sy / sz_y, 0.1, 10.0))
        return xz_aspect, yz_aspect

    def _to_display(self, plane: np.ndarray) -> np.ndarray:
        if self.image.rgb:
            rgb = plane[..., :3]
            if np.issubdtype(rgb.dtype, np.integer):
                hi = float(np.iinfo(rgb.dtype).max)
                return np.clip(np.round(rgb.astype(np.float32) / (hi if hi > 0 else 255.0) * 255.0), 0, 255).astype(np.uint8)
            return np.clip(np.nan_to_num(rgb.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        lo, hi = self.image.contrast_limits
        return ops.composite_rgb([(plane, self.image.colormap, (float(lo), float(hi)))])

    def set_coords(self, *, z: int | None = None, y: int | None = None, x: int | None = None, sync: bool = True) -> None:
        nz, ny, nx = self._volume_shape()
        if z is not None:
            self.z = int(np.clip(z, 0, nz - 1))
        if y is not None:
            self.y = int(np.clip(y, 0, ny - 1))
        if x is not None:
            self.x = int(np.clip(x, 0, nx - 1))
        if sync and self.sync_box.isChecked():
            idx = self._step_index_for(self.z_axis)
            if 0 <= idx < len(self.viewer.dims.current_step) and self.viewer.dims.current_step[idx] != self.z:
                self._syncing_viewer = True
                try:
                    self.viewer.dims.set_current_step(idx, self.z)
                finally:
                    self._syncing_viewer = False
        self.refresh()

    def refresh(self) -> None:
        vol = self._current_volume()
        xy, xz, yz = ops.orthogonal_slices(vol, self.z, self.y, self.x, rgb=bool(self.image.rgb))
        d_xy = self._to_display(xy)
        d_xz = self._to_display(xz)
        d_yz = self._to_display(yz)
        xz_aspect, yz_aspect = self._voxel_aspects()

        if self._im_xy is None:
            self._im_xy = self.ax_xy.imshow(d_xy, interpolation='nearest', aspect='equal')
            self._im_xz = self.ax_xz.imshow(d_xz, interpolation='nearest', aspect=xz_aspect)
            self._im_yz = self.ax_yz.imshow(d_yz, interpolation='nearest', aspect=yz_aspect)
            kw = {'color': 'yellow', 'lw': 0.8, 'alpha': 0.85}
            self._lines = [
                self.ax_xy.axhline(self.y, **kw),
                self.ax_xy.axvline(self.x, **kw),
                self.ax_xz.axhline(self.z, **kw),
                self.ax_xz.axvline(self.x, **kw),
                self.ax_yz.axhline(self.y, **kw),
                self.ax_yz.axvline(self.z, **kw),
            ]
        else:
            self._im_xy.set_data(d_xy)
            self._im_xz.set_data(d_xz)
            self._im_yz.set_data(d_yz)
            self._ax_set_line(self._lines[0], self.y, horizontal=True)
            self._ax_set_line(self._lines[1], self.x, horizontal=False)
            self._ax_set_line(self._lines[2], self.z, horizontal=True)
            self._ax_set_line(self._lines[3], self.x, horizontal=False)
            self._ax_set_line(self._lines[4], self.y, horizontal=True)
            self._ax_set_line(self._lines[5], self.z, horizontal=False)

        info = axes_mod.get(self.image)
        sz = info.axes[self.z_axis]
        sy = info.axes[-2]
        sx = info.axes[-1]
        z_txt = f'Z={self.z}' + (f' ({self.z * sz.scale:.3g} {sz.unit})' if sz.unit else '')
        y_txt = f'Y={self.y}' + (f' ({self.y * sy.scale:.3g} {sy.unit})' if sy.unit else '')
        x_txt = f'X={self.x}' + (f' ({self.x * sx.scale:.3g} {sx.unit})' if sx.unit else '')
        self.coord_label.setText(f'{x_txt}   {y_txt}   {z_txt}')
        self.canvas.draw_idle()

    @staticmethod
    def _ax_set_line(line, pos: int, horizontal: bool) -> None:
        if horizontal:
            line.set_ydata([pos, pos])
        else:
            line.set_xdata([pos, pos])

    def _update_from_event(self, event) -> None:
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        if event.inaxes is self.ax_xy:
            self.set_coords(x=int(round(event.xdata)), y=int(round(event.ydata)))
        elif event.inaxes is self.ax_xz:
            self.set_coords(x=int(round(event.xdata)), z=int(round(event.ydata)))
        elif event.inaxes is self.ax_yz:
            self.set_coords(z=int(round(event.xdata)), y=int(round(event.ydata)))

    def _on_press(self, event) -> None:
        if event.button != 1 or event.inaxes not in (self.ax_xy, self.ax_xz, self.ax_yz):
            return
        self._dragging_ax = event.inaxes
        self._update_from_event(event)

    def _on_move(self, event) -> None:
        if self._dragging_ax is not None and event.inaxes is self._dragging_ax:
            self._update_from_event(event)

    def _on_release(self, _event) -> None:
        self._dragging_ax = None

    def _on_viewer_step(self, _event=None) -> None:
        if self._syncing_viewer:
            return
        self.z = self._viewer_z()
        self.refresh()

    def _on_display_change(self, _event=None) -> None:
        self.refresh()

    def closeEvent(self, event):
        try:
            self.viewer.dims.events.current_step.disconnect(self._on_viewer_step)
        except Exception:
            pass
        try:
            self.image.events.contrast_limits.disconnect(self._on_display_change)
            self.image.events.colormap.disconnect(self._on_display_change)
        except Exception:
            pass
        super().closeEvent(event)


def orthogonal_views(tiki) -> OrthogonalViewsWindow:
    """``Image ▸ Stacks ▸ Orthogonal Views`` (``Ctrl+Shift+H``)."""
    viewer, image = tiki.require_image()
    n_lead = image.data.ndim - (3 if image.rgb else 2)
    if n_lead <= 0:
        _error('Orthogonal Views requires a 3D stack (Z or T).')
    info = axes_mod.get(image)
    z_axis = axes_mod.stack_axis(info, 'ZT')
    if z_axis is None:
        z_axis = 0
    if image.data.shape[z_axis] < 2:
        _error('Need at least 2 slices along the stack axis for Orthogonal Views.')
    win = OrthogonalViewsWindow(viewer, image, z_axis)
    tiki.keep_window(win)
    win.show()
    tiki.status(f'Orthogonal Views: click or drag crosshairs on XY, XZ, or YZ')
    return win
