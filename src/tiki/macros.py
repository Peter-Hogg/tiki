"""Plugins > Macros: write and run plain Python against the open images.

A macro is ordinary Python run inside TIKI with a ``tiki`` object for talking
to the windows (see :class:`MacroAPI`), plus ``np``. Anything installed in
TIKI's environment can be imported: scikit-image, scipy, CaImAn, suite2p...
Macros saved in the macros folder appear in the Plugins > Macros menu.
"""

from __future__ import annotations

import io
import os
import re
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
from qtpy.QtCore import QEvent, Qt
from qtpy.QtGui import QColor, QFont, QKeySequence, QSyntaxHighlighter, QTextCharFormat, QTextCursor
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from . import axes as axes_mod
from . import style

MACROS_DIR = Path(os.environ.get('TIKI_MACROS', Path.home() / '.config' / 'tiki' / 'macros'))


class MacroCancelled(Exception):
    """Raised inside a macro when the user presses Cancel on a progress bar."""


# ---------------------------------------------------------------------- API

class MacroAPI:
    """The ``tiki`` object available in every macro.

    Choosing layers and windows
        Most calls take ``which`` = a layer name or index (0 = bottom, -1 = top,
        among that window's image or labels layers) and ``window`` = a window
        title or index. Leave them out for the current image of the current window.
        ``tiki.windows()``      titles of the open windows
        ``tiki.layers(kind='image', window=None)``   layer names, bottom to top
    Images
        ``tiki.image(which=None, window=None)``   data (memory-mapped when possible)
        ``tiki.layer(which=None, window=None)``   the napari Image layer
        ``tiki.viewer(window=None)``  the napari viewer (window)
        ``tiki.axes(which)``    axis roles, e.g. ``'TYX'``
        ``tiki.scale(which)``   physical step per axis (1.0 if uncalibrated)
        ``tiki.index()``        current slider position
        ``tiki.show(data, name, axes='TYX', scale=None)``   open data in a new window
        ``tiki.add_image(data, name, like=None, window=None)``   add an aligned layer
        ``tiki.add_labels(data, name, like=None, window=None)``  add a Labels layer
        ``tiki.labels(which=None, window=None)``  data of a Labels layer
    ROIs and measurements
        ``tiki.rois()``         ROI Manager entries (``.name``, ``.kind``, ``.yx``, ``.mask(shape)``)
        ``tiki.roi_masks()``    (n, Y, X) boolean masks of all manager ROIs
        ``tiki.add_roi(yx, kind='polygon', name=None)``   add an ROI to the manager
        ``tiki.rois_from_labels(labels)``   one ROI per label id
        ``tiki.traces(masks=None, which=None, window=None)``   (n, T) mean traces
        ``tiki.measure(**values)``    add a row to the Results table
    Running things
        ``tiki.run('Gaussian Blur…')``   run a menu command by name, as ImageJ's run()
        ``tiki.progress(iterable, 'label')``   progress bar with Cancel; keeps TIKI responsive
        ``tiki.status('message')``
        ``tiki.ops``            TIKI's numpy operations (z_project, dff_movie, correlation_image...)
    """

    def __init__(self, app):
        self._app = app
        from . import ops

        self.ops = ops

    # -- choosing windows and layers ------------------------------------------
    #
    # Most calls take ``which`` (a layer name, or an index among the window's
    # layers of that type: 0 = bottom, -1 = top) and ``window`` (a window title,
    # or an index into tiki.windows()). Leaving both out means "the current
    # image of the current window", as in ImageJ.

    def windows(self) -> list[str]:
        """Titles of the open image windows, in opening order."""
        return [v.title for v in self._app.windows.viewers]

    def viewer(self, window=None):
        if window is None:
            viewer = self._app.windows.active
            if viewer is None:
                raise RuntimeError('no image window is open')
            return viewer
        viewers = self._app.windows.viewers
        if isinstance(window, int):
            try:
                return viewers[window]
            except IndexError:
                raise IndexError(f'window index {window} out of range; open windows: {self.windows()}') from None
        for v in viewers:
            if v.title == window:
                return v
        raise KeyError(f'no window titled {window!r}; open windows: {self.windows()}')

    def _layers_of(self, kind, window=None) -> list:
        from . import roi as roi_mod

        return [ly for ly in self.viewer(window).layers
                if isinstance(ly, kind) and ly.name not in roi_mod.PREVIEW_NAMES]

    def layers(self, kind: str = 'image', window=None) -> list[str]:
        """Names of the window's layers of ``kind`` ('image', 'labels' or 'all'), bottom to top."""
        from napari.layers import Image, Labels, Layer

        cls = {'image': Image, 'labels': Labels, 'all': Layer}[kind]
        return [ly.name for ly in self._layers_of(cls, window)]

    def _pick(self, kind, which, window, what: str):
        layers = self._layers_of(kind, window)
        if not layers:
            raise RuntimeError(f'no {what} layer in window {self.viewer(window).title!r}')
        if isinstance(which, int):
            try:
                return layers[which]
            except IndexError:
                raise IndexError(f'{what} layer index {which} out of range; '
                                 f'layers: {[ly.name for ly in layers]}') from None
        for ly in layers:
            if ly.name == which:
                return ly
        raise KeyError(f'no {what} layer named {which!r}; layers: {[ly.name for ly in layers]}')

    # -- images ------------------------------------------------------------

    def layer(self, which=None, window=None):
        """An Image layer: the current one, or by name / index (see above)."""
        from napari.layers import Image

        from . import roi as roi_mod

        if which is None:
            image = roi_mod.target_image(self.viewer(window))
            if image is None:
                raise RuntimeError('no image is open')
            return image
        return self._pick(Image, which, window, 'image')

    def image(self, which=None, window=None):
        """Data of an image layer (memory-mapped when possible)."""
        return self.layer(which, window).data

    def axes(self, which=None, window=None) -> str:
        return axes_mod.get(self.layer(which, window)).roles

    def scale(self, which=None, window=None) -> list[float]:
        return [float(s) for s in self.layer(which, window).scale]

    def index(self, window=None) -> tuple:
        from . import roi as roi_mod

        return roi_mod.current_index(self.viewer(window), self.layer(window=window))

    def _axes_info(self, data, axes, scale):
        data = np.asarray(data) if not hasattr(data, 'shape') else data
        roles = axes or ('?' * (data.ndim - 2) + 'YX')
        roles = roles.upper()
        if len(roles) != data.ndim:
            raise ValueError(f'axes {roles!r} has {len(roles)} letters for {data.ndim}-D data')
        units = {'T': 's', 'Z': 'µm', 'Y': 'µm', 'X': 'µm'}
        steps = list(scale) if scale is not None else [None] * data.ndim
        info = axes_mod.AxesInfo(
            [axes_mod.Axis(r, int(n), (float(s) if s and s != 1 else None), units.get(r) if s and s != 1 else None)
             for r, n, s in zip(roles, data.shape, steps)],
            'macro', True)
        problems = info.problems(tuple(data.shape))
        if problems:
            raise ValueError('; '.join(problems))
        return info

    def show(self, data, name: str = 'macro result', axes: str | None = None, scale=None):
        """Open ``data`` in a new image window; returns the napari viewer."""
        return self._app.open_data(data, name, axes=self._axes_info(data, axes, scale))

    def add_image(self, data, name: str = 'macro result', like=None, window=None, **kwargs):
        """Add an image layer to a window, aligned with image ``like`` (default: current)."""
        from napari.layers import Image

        source = self.layer(like, window)
        geometry = axes_mod.geometry(source, ndim=np.ndim(data) if np.ndim(data) < source.ndim else None)
        layer = Image(data, name=name, **(geometry | kwargs))
        viewer = self.viewer(window)
        viewer.add_layer(layer)
        self._app.reapply_tool(viewer)
        return layer

    def add_labels(self, data, name: str = 'macro labels', like=None, window=None):
        """Add a Labels layer to a window, aligned with image ``like`` (default: current)."""
        from napari.layers import Labels

        source = self.layer(like, window)
        data = np.asarray(data)
        geometry = axes_mod.geometry(source, ndim=data.ndim if data.ndim < source.ndim else None)
        layer = Labels(data.astype(np.int32, copy=False), name=name, opacity=0.6, **geometry)
        viewer = self.viewer(window)
        viewer.add_layer(layer)
        self._app.reapply_tool(viewer)
        return layer

    def labels(self, which=None, window=None):
        """Data of a Labels layer: the selected/topmost one, or by name / index."""
        from napari.layers import Labels

        if which is None:
            return self._app.labels_layer(self.viewer(window)).data
        return self._pick(Labels, which, window, 'labels').data

    # -- ROIs --------------------------------------------------------------

    def _manager(self):
        return self._app.show_roi_manager()

    def rois(self):
        return list(self._manager().rois)

    def roi_masks(self, which=None, window=None) -> np.ndarray:
        shape = self.layer(which, window).data.shape[-2:]
        rois = self.rois()
        return np.stack([r.mask(shape) for r in rois]) if rois else np.zeros((0,) + shape, bool)

    def add_roi(self, yx, kind: str = 'polygon', name: str | None = None):
        from . import rois as R

        roi = R.Roi(name or '', kind, np.asarray(yx, dtype=float))
        self._manager().add([roi])
        return roi

    def rois_from_labels(self, labels):
        from . import rois as R

        rois = R.rois_from_labels(np.asarray(labels))
        self._manager().add(rois)
        return rois

    def traces(self, masks=None, which=None, window=None) -> np.ndarray:
        """Mean trace of each mask along T (else Z) of an image: (n, frames)."""
        from . import ops
        from . import roi as roi_mod

        viewer, image = self.viewer(window), self.layer(which, window)
        masks = self.roi_masks(which, window) if masks is None else np.asarray(masks, bool)
        if masks.ndim == 2:
            masks = masks[None]
        axis = self._app._stack_axis(viewer, image)
        return ops.roi_traces(image.data, axis, roi_mod.current_index(viewer, image), list(masks))

    def measure(self, **values) -> None:
        self._app.results.add_row(values)
        self._app.show_results()

    def open(self, path: str | Path):
        """Open an image file and return its napari viewer."""
        opened = self._app.open_paths([str(path)], raise_errors=True)
        if not opened:
            raise RuntimeError(f'failed to open {path}')
        return opened[-1]

    def save_tiff(self, path: str | Path, merge_split: bool = False):
        """Save the active image or labels layer as a calibrated TIFF."""
        return self._app.save_layer(path, merge_split=merge_split)

    def set_lut(self, name: str) -> None:
        """Apply a named LUT (e.g. 'Fire', 'Cyan Hot', 'Grays') to the active image."""
        self._app.set_lut(name)

    def set_scale(self, step_x: float, step_y: float | None = None, unit: str = 'µm',
                  global_scale: bool = False) -> None:
        """Set physical pixel spacing (X/Y) on the current image or all open windows."""
        from napari.layers import Image

        from . import roi as roi_mod

        sx = float(step_x)
        sy = float(step_x if step_y is None else step_y)
        targets = (
            [
                (v, ly)
                for v in self._app.windows.viewers
                for ly in v.layers
                if isinstance(ly, Image) and ly.name not in roi_mod.PREVIEW_NAMES
            ]
            if global_scale
            else [(self.viewer(), self.layer())]
        )
        for v, ly in targets:
            info = axes_mod.get(ly)
            for role, step in (('X', sx), ('Y', sy)):
                idx = info.index(role)
                if idx is not None:
                    info.axes[idx].step = step
                    info.axes[idx].unit = unit
            info.source, info.confident = 'macro', True
            axes_mod.apply(v, ly, info)

    def filter(self, key: str, whole_stack: bool = True, three_d: bool = False, **params):
        """Run a TIKI filter ('gaussian', 'mean', 'median', 'minimum', 'maximum', 'variance', 'tophat', 'unsharp', 'background')."""
        from . import filters

        viewer, image = self.viewer(), self.layer()
        return filters.run_filter(
            self._app, key, viewer, image, params, whole_stack=whole_stack, three_d=three_d
        )

    def z_project(self, method: str = 'max', axis: int | None = None, start: int = 0, stop: int | None = None):
        """Run Z/T Projection along ``axis`` (defaults to first Z or T axis)."""
        image = self.layer()
        if axis is None:
            axis = axes_mod.stack_axis(axes_mod.get(image), prefer='ZT')
            if axis is None:
                raise RuntimeError('active image has no Z or T stack axis')
        return self._app.z_project(image, method, axis=axis, start=start, stop=stop)

    def threshold(
        self,
        lower: float,
        upper: float,
        scope: str = 'plane',
        split: bool = True,
        connect_z: bool = False,
        min_size: int = 0,
    ):
        """Threshold the current image and add a Labels layer."""
        from napari.layers import Labels

        from . import nd, ops
        from . import roi as roi_mod

        viewer, image = self.viewer(), self.layer()
        info = axes_mod.get(image)
        shape = tuple(image.data.shape[: image.ndim])
        index = roi_mod.current_index(viewer, image)
        if image.ndim == 2:
            out = ops.segment(np.asarray(image.data), lower, upper, split=split, min_size=min_size)
            geom = axes_mod.geometry(image)
            meta = {}
        else:
            blocks = nd.blocks(info, shape, index, scope, bool(connect_z and scope != nd.PLANE))
            out = ops.threshold_blocks(image.data, lower, upper, blocks, split=split, min_size=min_size)
            if scope == nd.PLANE:
                out = out[blocks[0]]
                geom = axes_mod.geometry(image, ndim=2)
                meta = {}
            elif scope == nd.VOLUME:
                out = out[blocks[0]]
                geom = axes_mod.geometry(image, ndim=3)
                meta = {'axes': axes_mod.fit(info, out.shape).to_dict()}
            else:
                geom = axes_mod.geometry(image)
                meta = {'axes': info.to_dict()}
        layer = Labels(out, name=f'{image.name} labels', opacity=0.6, metadata=meta, **geom)
        viewer.add_layer(layer)
        self._app.reapply_tool(viewer)
        return layer

    def analyze_particles(
        self,
        min_size: int = 0,
        max_size: float | None = None,
        exclude_edges: bool = False,
        clear: bool = True,
        scope: str = 'plane',
        three_d: bool = True,
    ) -> int:
        """Run Analyze Particles on the active Labels layer."""
        viewer = self.viewer()
        return self._app.particles(
            viewer,
            self.layer(),
            self._app.labels_layer(viewer),
            min_size=min_size,
            exclude_edges=exclude_edges,
            clear=clear,
            scope=scope,
            three_d=three_d,
            max_size=max_size,
        )

    def math(self, op_name: str, value: float = 0.0, whole_stack: bool = True):
        """Run Process > Math on the active greyscale image."""
        from . import process

        return process.run_math(self._app, self.viewer(), self.layer(), op_name, value, whole_stack=whole_stack)

    def resize(self, height: int, width: int, order: int = 1):
        """Resize the active image to (height, width), updating pixel calibration."""
        from . import process

        return process.run_resize(self._app, self.viewer(), self.layer(), height, width, order=order)

    def morphology_labels(self, op: str = 'open', radius: float = 1.0, scope: str = 'whole', three_d: bool = True):
        """Apply morphological operation to the active Labels layer."""
        from . import process

        viewer = self.viewer()
        return process.run_morphology_labels(
            self._app, viewer, self._app.labels_layer(viewer), op=op, radius=radius, scope=scope, three_d=three_d
        )

    def voronoi(self, boundaries_only: bool = False, use_physical: bool = True):
        """Compute Voronoi territories or boundaries from the active Labels layer."""
        from . import process

        viewer = self.viewer()
        return process.run_voronoi(
            self._app, viewer, self._app.labels_layer(viewer), boundaries_only=boundaries_only, use_physical=use_physical
        )

    def convolve(self, kernel, normalize: bool = True, whole_stack: bool = True):
        """Convolve the active greyscale image (plane or stack) with a 2D kernel."""
        from . import process

        return process.run_convolve(self._app, self.viewer(), self.layer(), kernel, normalize=normalize, whole_stack=whole_stack)

    def find_maxima(
        self,
        prominence: float = 10.0,
        output: str = 'Point Selection',
        exclude_edges: bool = False,
        light_background: bool = False,
        strict: bool = False,
        whole_stack: bool = False,
        use_roi: bool = False,
    ):
        """Find local intensity maxima on the active greyscale image."""
        from . import process

        return process.run_find_maxima(
            self._app,
            self.viewer(),
            self.layer(),
            prominence=prominence,
            output=output,
            exclude_edges=exclude_edges,
            light_background=light_background,
            strict=strict,
            whole_stack=whole_stack,
            use_roi=use_roi,
        )

    def edge_detection(
        self,
        method: str = 'canny',
        sigma: float = 1.0,
        low_threshold: float = 0.1,
        high_threshold: float = 0.2,
        use_quantiles: bool = True,
        sigma_ratio: float = 1.6,
        whole_stack: bool = False,
        three_d: bool = False,
        as_labels: bool = False,
    ):
        """Run Process > Edge Detection on the active greyscale image."""
        from . import process

        return process.run_edge_detection(
            self._app,
            self.viewer(),
            self.layer(),
            method=method,
            sigma=sigma,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            use_quantiles=use_quantiles,
            sigma_ratio=sigma_ratio,
            whole_stack=whole_stack,
            three_d=three_d,
            as_labels=as_labels,
        )

    def add_noise(self, sigma: float = 25.0, whole_stack: bool = True):
        """Add zero-mean Gaussian noise with standard deviation ``sigma``."""
        from . import fft_noise

        return fft_noise.run_add_noise(self._app, self.viewer(), self.layer(), sigma=sigma, whole_stack=whole_stack)

    def remove_outliers(self, radius: float = 2.0, threshold: float = 50.0, which: str = 'bright', whole_stack: bool = True):
        """Remove bright, dark, or both intensity outliers relative to local median."""
        from . import fft_noise

        return fft_noise.run_remove_outliers(
            self._app, self.viewer(), self.layer(), radius=radius, threshold=threshold, which=which, whole_stack=whole_stack
        )

    def remove_nans(self, radius: float = 2.0, whole_stack: bool = True):
        """Replace non-finite pixels with the local finite median."""
        from . import fft_noise

        return fft_noise.run_remove_nans(self._app, self.viewer(), self.layer(), radius=radius, whole_stack=whole_stack)

    def bandpass_filter(
        self,
        filter_large: float = 40.0,
        filter_small: float = 3.0,
        suppress_stripes: str = 'none',
        tolerance: float = 5.0,
        autoscale: bool = False,
        whole_stack: bool = True,
    ):
        """Run 2D FFT Bandpass Filter on the active greyscale image or stack."""
        from . import fft_noise

        return fft_noise.run_bandpass(
            self._app,
            self.viewer(),
            self.layer(),
            filter_large=filter_large,
            filter_small=filter_small,
            suppress_stripes=suppress_stripes,
            tolerance=tolerance,
            autoscale=autoscale,
            whole_stack=whole_stack,
        )

    # -- running -------------------------------------------------------------

    def run(self, command: str):
        """Run a TIKI menu command by its name (e.g. 'Z Project…', 'Measure')."""
        return self._app.run_command(command, raise_errors=True)

    def status(self, message: str) -> None:
        self._app.status(str(message))
        QApplication.processEvents()

    def progress(self, iterable, label: str = 'Macro', total: int | None = None):
        """Wrap a loop: shows a progress bar, keeps TIKI responsive, Cancel stops the macro."""
        if total is None:
            try:
                total = len(iterable)
            except TypeError:
                total = 0
        last = 0.0
        with self._app.progress(label, total or 1) as tick:
            for n, item in enumerate(iterable, 1):
                yield item
                now = time.monotonic()
                if now - last > 0.05 or n == total:
                    last = now
                    if tick(n, total or n) is False:
                        raise MacroCancelled(f'{label} cancelled')


# -------------------------------------------------------------------- running

def run_code(app, code: str, filename: str = '<macro>', namespace: dict | None = None):
    """Execute macro ``code``; returns ``(ok, output_text, namespace)``.

    stdout/stderr are captured. Errors are reported as tracebacks trimmed to
    the macro's own lines, never raised into Qt.
    """
    ns = namespace if namespace is not None else {}
    ns.setdefault('__name__', '__macro__')
    ns['tiki'] = MacroAPI(app)
    ns.setdefault('np', np)
    buffer = io.StringIO()
    ok = True
    rec = getattr(app, 'recorder_window', None)
    was_paused = rec.paused if rec is not None else False
    if rec is not None:
        rec.paused = True
    try:
        compiled = compile(code, filename, 'exec')
        with redirect_stdout(buffer), redirect_stderr(buffer):
            exec(compiled, ns)
    except MacroCancelled as exc:
        buffer.write(f'\n{exc}\n')
        ok = False
    except Exception as exc:  # a macro error must never take TIKI down
        ok = False
        tb = traceback.extract_tb(exc.__traceback__)
        user = [f for f in tb if f.filename == filename]
        lines = traceback.format_list(user) if user else []
        if isinstance(exc, SyntaxError):
            buffer.write(''.join(traceback.format_exception_only(type(exc), exc)))
        else:
            buffer.write('Traceback (macro lines):\n' + ''.join(lines) + f'{type(exc).__name__}: {exc}\n')
    finally:
        if rec is not None:
            rec.paused = was_paused
    return ok, buffer.getvalue(), ns


def list_macros() -> list[Path]:
    if not MACROS_DIR.is_dir():
        return []
    return sorted(p for p in MACROS_DIR.glob('*.py') if not p.name.startswith('_'))


# ------------------------------------------------------------------ templates

TEMPLATES = {
    'Blank (API reference)': '''"""TIKI macro: plain Python with a `tiki` object and numpy as `np`.

tiki.windows()  tiki.layers()        # what is open: window titles, layer names
tiki.image(which=None, window=None)  # which = layer name or index (0 bottom, -1 top)
tiki.layer()  tiki.viewer()  tiki.axes()  tiki.scale()  tiki.index()
tiki.show(data, name, axes='TYX', scale=None)   tiki.add_image(data, name)
tiki.add_labels(data, name)   tiki.labels()
tiki.rois()  tiki.roi_masks()  tiki.add_roi(yx)  tiki.rois_from_labels(labels)
tiki.traces(masks=None)   tiki.measure(**values)
tiki.run('Command…')   tiki.progress(iterable, 'label')   tiki.status('msg')
tiki.ops  (z_project, dff_movie, correlation_image, segment, ...)
"""

print('windows:', tiki.windows())
print('layers in this window:', tiki.layers())
data = tiki.image()                  # the current image
print(tiki.layer().name, data.shape, data.dtype, tiki.axes())
# e.g. two channels after Image > Color > Split Channels:
# green, red = tiki.image('C1-movie.tif'), tiki.image('C2-movie.tif')
''',
    'Motion correction (rigid)': '''"""Rigid motion correction by phase correlation against a median template.

Works on a (T, Y, X) movie; runs here with scikit-image and scipy.
Frames are mean-subtracted and Hann-windowed, and the cross-power spectrum is
not whitened (normalization=None): scikit-image's default "phase" mode can
lock onto the image borders and report no shift at all.
"""
import numpy as np
from scipy.ndimage import shift as nd_shift
from skimage.filters import window
from skimage.registration import phase_cross_correlation

movie = tiki.image()
if tiki.axes() != 'TYX':
    print('note: expected T, Y, X; got', tiki.axes())

n_template = min(200, len(movie))
template = np.median(np.asarray(movie[:n_template], dtype=np.float32), axis=0)
hann = window('hann', template.shape).astype(np.float32)


def prepare(img):
    return (img - img.mean()) * hann


ref = prepare(template)
out = np.empty(movie.shape, np.float32)
shifts = np.zeros((len(movie), 2))
for t in tiki.progress(range(len(movie)), 'Motion correction'):
    frame = np.asarray(movie[t], dtype=np.float32)
    shifts[t], _, _ = phase_cross_correlation(ref, prepare(frame), upsample_factor=10, normalization=None)
    out[t] = nd_shift(frame, shifts[t], order=1, mode='nearest')

tiki.show(out, 'MC_' + tiki.layer().name, axes=tiki.axes(), scale=tiki.scale())
print('max |shift| (y, x) px:', np.abs(shifts).max(axis=0).round(2))
''',
    'CaImAn CNMF → ROIs': '''"""Run CaImAn's CNMF on the current (T, Y, X) movie and bring its
components back as Labels and ROI Manager entries.

Needs CaImAn installed in TIKI's environment (conda is the reliable route:
https://caiman.readthedocs.io). Tune the parameters for your data.
NOTE: this template has not been run against CaImAn from TIKI yet.
"""
import numpy as np

try:
    from caiman.source_extraction.cnmf import cnmf, params
except ImportError:
    raise RuntimeError("CaImAn is not installed in TIKI's Python environment") from None

movie = np.asarray(tiki.image(), dtype=np.float32)
dims = movie.shape[1:]
opts = params.CNMFParams(params_dict={
    'fr': 30,               # frame rate (Hz)
    'decay_time': 0.4,      # GCaMP6f-ish
    'p': 1, 'nb': 2,
    'rf': 15, 'stride': 6,  # patch half-size and overlap
    'K': 4, 'gSig': [4, 4], # components per patch, neuron radius
    'method_init': 'greedy_roi',
    'merge_thr': 0.85,
})
cnm = cnmf.CNMF(n_processes=1, params=opts, dview=None)
cnm = cnm.fit(movie)

A = cnm.estimates.A.toarray().reshape(dims + (-1,), order='F')  # (Y, X, K)
labels = np.zeros(dims, np.int32)
for k in range(A.shape[-1]):
    comp = A[..., k]
    labels[comp > 0.3 * comp.max()] = k + 1
tiki.add_labels(labels, 'CNMF components')
tiki.rois_from_labels(labels)
print(A.shape[-1], 'components; C (temporal traces):', cnm.estimates.C.shape)
''',
    'Measure ROIs over time → Results': '''"""Mean of every ROI Manager entry at every frame, into the Results table
and a cells x frames array."""
traces = tiki.traces()                      # (n_rois, n_frames)
names = [r.name for r in tiki.rois()]
for name, trace in zip(names, traces):
    tiki.measure(ROI=name, Mean=float(trace.mean()), Min=float(trace.min()), Max=float(trace.max()))
print(traces.shape)
''',
}


# ------------------------------------------------------------------- editor

class _PythonHighlighter(QSyntaxHighlighter):
    KEYWORDS = r'\b(and|as|assert|break|class|continue|def|del|elif|else|except|finally|for|from|global|if|import|' \
               r'in|is|lambda|not|or|pass|raise|return|try|while|with|yield|None|True|False)\b'

    def __init__(self, doc):
        super().__init__(doc)

        def fmt(color, bold=False, italic=False):
            f = QTextCharFormat()
            f.setForeground(QColor(color))
            if bold:
                f.setFontWeight(QFont.Bold)
            f.setFontItalic(italic)
            return f

        self.rules = [
            (re.compile(self.KEYWORDS), fmt('#0b3d91', bold=True)),
            (re.compile(r'\btiki\b'), fmt('#7a1fa2', bold=True)),
            (re.compile(r'\b[0-9]+(\.[0-9]*)?\b'), fmt('#a0522d')),
            (re.compile(r"('[^'\n]*'|\"[^\"\n]*\")"), fmt('#1b7a2e')),
            (re.compile(r'#[^\n]*'), fmt('#8a8a8a', italic=True)),
        ]
        self.docstring = fmt('#1b7a2e', italic=True)

    def highlightBlock(self, text):
        for pattern, f in self.rules:
            for m in pattern.finditer(text):
                self.setFormat(m.start(), m.end() - m.start(), f)
        # triple-quoted strings spanning lines
        self.setCurrentBlockState(0)
        start = 0 if self.previousBlockState() == 1 else -1
        pos = 0
        while True:
            if start < 0:
                start = text.find('"""', pos)
                if start < 0:
                    break
                pos = start + 3
            end = text.find('"""', pos)
            if end < 0:
                self.setFormat(start, len(text) - start, self.docstring)
                self.setCurrentBlockState(1)
                break
            self.setFormat(start, end + 3 - start, self.docstring)
            pos = end + 3
            start = -1


class _CodeEdit(QPlainTextEdit):
    """Editor that keeps its own shortcuts (TIKI's app-wide ones would steal Ctrl+S)."""

    OWN = ('Ctrl+R', 'Ctrl+Shift+R', 'Ctrl+S', 'Ctrl+O', 'Ctrl+N')

    def __init__(self, handlers: dict):
        super().__init__()
        self.handlers = handlers
        font = QFont('Monospace')
        font.setStyleHint(QFont.TypeWriter)
        self.setFont(font)
        self.setTabStopDistance(4 * self.fontMetrics().horizontalAdvance(' '))
        self.setLineWrapMode(QPlainTextEdit.NoWrap)

    def _key(self, event) -> str:
        return QKeySequence(event.keyCombination() if hasattr(event, 'keyCombination')
                            else int(event.modifiers()) | event.key()).toString()

    def event(self, event):
        if event.type() == QEvent.ShortcutOverride and self._key(event) in self.OWN:
            event.accept()
            return True
        return super().event(event)

    def keyPressEvent(self, event):
        key = self._key(event)
        if key in self.handlers:
            self.handlers[key]()
            return
        if event.key() == Qt.Key_Tab:
            self.insertPlainText('    ')
            return
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):  # keep indentation
            line = self.textCursor().block().text()
            indent = line[: len(line) - len(line.lstrip())]
            if line.rstrip().endswith(':'):
                indent += '    '
            super().keyPressEvent(event)
            self.insertPlainText(indent)
            return
        super().keyPressEvent(event)


class MacroEditor(QMainWindow):
    """Plugins > Macros > New Macro: a small Python editor with a Run button."""

    def __init__(self, app, path: str | None = None):
        super().__init__()
        style.apply(self)
        self.app = app
        self.path: Path | None = None
        self.namespace: dict = {}
        self.resize(760, 620)
        self.editor = _CodeEdit({
            'Ctrl+R': self.run, 'Ctrl+Shift+R': self.run_selection,
            'Ctrl+S': self.save, 'Ctrl+O': self.open_file, 'Ctrl+N': self.new,
        })
        self._highlighter = _PythonHighlighter(self.editor.document())
        self.editor.document().modificationChanged.connect(lambda _: self._title())
        self.output = QPlainTextEdit(readOnly=True)
        self.output.setFont(self.editor.font())
        self.output.setMaximumBlockCount(5000)

        run = QPushButton('Run (Ctrl+R)')
        run.clicked.connect(self.run)
        sel = QPushButton('Run selection')
        sel.clicked.connect(self.run_selection)
        clear = QPushButton('Clear output')
        clear.clicked.connect(self.output.clear)
        self.keep = QCheckBox('Keep variables between runs')
        bar = QHBoxLayout()
        for w in (run, sel, self.keep):
            bar.addWidget(w)
        bar.addStretch()
        bar.addWidget(clear)

        bottom = QWidget()
        lay = QVBoxLayout(bottom)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(bar)
        lay.addWidget(self.output)
        split = QSplitter(Qt.Vertical)
        split.addWidget(self.editor)
        split.addWidget(bottom)
        split.setSizes([420, 200])
        self.setCentralWidget(split)

        m = self.menuBar().addMenu('&File')
        m.addAction('New (Ctrl+N)', self.new)
        m.addAction('Open… (Ctrl+O)', self.open_file)
        m.addAction('Save (Ctrl+S)', self.save)
        m.addAction('Save As…', self.save_as)
        m.addSeparator()
        m.addAction('Close', self.close)
        t = self.menuBar().addMenu('&Templates')
        for name, code in TEMPLATES.items():
            t.addAction(name, lambda c=code: self._load_text(c, None))
        r = self.menuBar().addMenu('&Run')
        r.addAction('Run (Ctrl+R)', self.run)
        r.addAction('Run selection (Ctrl+Shift+R)', self.run_selection)
        h = self.menuBar().addMenu('&Help')
        h.addAction('Macro API', lambda: self._print(MacroAPI.__doc__))

        if path:
            self._load_file(Path(path))
        else:
            self._load_text(TEMPLATES['Blank (API reference)'], None)

    # -- files -----------------------------------------------------------------

    def _title(self) -> None:
        name = self.path.name if self.path else 'Untitled.py'
        self.setWindowTitle(f'{"*" if self.editor.document().isModified() else ""}{name} — TIKI Macro')

    def _load_text(self, text: str, path: Path | None) -> None:
        self.editor.setPlainText(text)
        self.path = path
        self.editor.document().setModified(False)
        self._title()

    def _load_file(self, path: Path) -> None:
        self._load_text(path.read_text(), path)

    def new(self) -> None:
        self._load_text(TEMPLATES['Blank (API reference)'], None)

    def open_file(self) -> None:
        MACROS_DIR.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getOpenFileName(self, 'Open macro', str(MACROS_DIR), 'Python (*.py)')
        if path:
            self._load_file(Path(path))

    def save(self) -> None:
        if self.path is None:
            self.save_as()
            return
        self.path.write_text(self.editor.toPlainText())
        self.editor.document().setModified(False)
        self._title()
        self.app.status(f'Saved macro {self.path}')

    def save_as(self) -> None:
        MACROS_DIR.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(self, 'Save macro', str(MACROS_DIR / 'my_macro.py'), 'Python (*.py)')
        if path:
            self.path = Path(path) if path.endswith('.py') else Path(path + '.py')
            self.save()

    # -- running -----------------------------------------------------------------

    def _print(self, text: str, color: str | None = None) -> None:
        cursor = self.output.textCursor()
        cursor.movePosition(QTextCursor.End)
        fmt = QTextCharFormat()
        if color:
            fmt.setForeground(QColor(color))
        cursor.insertText(text if text.endswith('\n') else text + '\n', fmt)
        self.output.setTextCursor(cursor)
        self.output.ensureCursorVisible()

    def _execute(self, code: str) -> bool:
        name = str(self.path) if self.path else '<macro>'
        self._print(f'── run {time.strftime("%H:%M:%S")} ──', '#8a8a8a')
        QApplication.setOverrideCursor(Qt.WaitCursor)
        t0 = time.perf_counter()
        try:
            ns = self.namespace if self.keep.isChecked() else {}
            ok, out, ns = run_code(self.app, code, name, ns)
        finally:
            QApplication.restoreOverrideCursor()
        if self.keep.isChecked():
            self.namespace = ns
        if out:
            self._print(out, None if ok else '#b00020')
        self._print(f'{"done" if ok else "stopped"} in {time.perf_counter() - t0:.2f} s',
                    '#1b7a2e' if ok else '#b00020')
        return ok

    def run(self) -> bool:
        return self._execute(self.editor.toPlainText())

    def run_selection(self) -> bool:
        text = self.editor.textCursor().selectedText().replace(' ', '\n')
        return self._execute(text or self.editor.textCursor().block().text())

    def closeEvent(self, event):
        if self.editor.document().isModified():
            from qtpy.QtWidgets import QMessageBox

            answer = QMessageBox.question(self, 'Macro', 'Save changes before closing?',
                                          QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
            if answer == QMessageBox.Cancel:
                event.ignore()
                return
            if answer == QMessageBox.Save:
                self.save()
        super().closeEvent(event)


class LogWindow(QMainWindow):
    """Window > Log: output of macros run from the Plugins menu (like ImageJ's Log)."""

    def __init__(self):
        super().__init__()
        style.apply(self)
        self.setWindowTitle('Log')
        self.resize(560, 300)
        self.text = QPlainTextEdit(readOnly=True)
        font = QFont('Monospace')
        font.setStyleHint(QFont.TypeWriter)
        self.text.setFont(font)
        self.setCentralWidget(self.text)
        self.menuBar().addAction('Clear', self.text.clear)

    def append(self, text: str) -> None:
        self.text.appendPlainText(text.rstrip('\n'))


class MacroRecorderWindow(QWidget):
    """Plugins > Macros > Record…: capture menu actions as editable Python code."""

    HEADER = '# TIKI Macro Recorder — run actions in TIKI, then click "Create Macro"\n'

    def __init__(self, app):
        super().__init__(None, Qt.Window)
        style.apply(self)
        self.app = app
        self.paused = False
        self.setWindowTitle('Macro Recorder')
        self.resize(620, 320)

        self.text = QPlainTextEdit(self.HEADER)
        font = QFont('Monospace')
        font.setStyleHint(QFont.TypeWriter)
        self.text.setFont(font)

        self.pause_btn = QPushButton('Pause')
        self.pause_btn.setCheckable(True)
        self.pause_btn.toggled.connect(self._toggle_pause)

        create_btn = QPushButton('Create Macro')
        create_btn.clicked.connect(self.create_macro)
        copy_btn = QPushButton('Copy')
        copy_btn.clicked.connect(self.copy)
        clear_btn = QPushButton('Clear')
        clear_btn.clicked.connect(self.clear)

        bar = QHBoxLayout()
        bar.addWidget(self.pause_btn)
        bar.addWidget(create_btn)
        bar.addStretch()
        bar.addWidget(copy_btn)
        bar.addWidget(clear_btn)

        layout = QVBoxLayout(self)
        layout.addLayout(bar)
        layout.addWidget(self.text, 1)

    def _toggle_pause(self, paused: bool) -> None:
        self.paused = bool(paused)
        self.pause_btn.setText('Resume' if self.paused else 'Pause')

    def record(self, code_line: str) -> None:
        if self.paused or not code_line.strip():
            return
        self.text.appendPlainText(code_line.rstrip())

    def code(self) -> str:
        return self.text.toPlainText()

    def create_macro(self):
        editor = MacroEditor(self.app)
        editor._load_text(self.code(), None)
        self.app.keep_window(editor)
        editor.show()
        editor.raise_()
        return editor

    def copy(self) -> None:
        QApplication.clipboard().setText(self.code())

    def clear(self) -> None:
        self.text.setPlainText(self.HEADER)
