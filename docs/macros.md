# Python macros

**Plugins → Macros → New Macro…** ({kbd}`Ctrl+Shift+M`) opens an editor for
plain Python. Macros run inside TIKI with a `tiki` object for the open windows
and `np` for numpy; anything installed in TIKI's environment can be imported
(scikit-image, scipy, CaImAn, suite2p, …).

- {kbd}`Ctrl+R` runs the macro, {kbd}`Ctrl+Shift+R` runs the selection (or the
  current line). Output and errors appear below the code; errors show only the
  macro's own lines.
- **Templates** include rigid motion correction, CaImAn CNMF → ROIs, and
  measuring ROIs over time.
- Save macros into the macros folder (**Plugins → Macros → Show Macros
  Folder**, default `~/.config/tiki/macros`, or set `TIKI_MACROS`). Every `.py`
  there appears in the Plugins → Macros menu; files starting with `_` are
  hidden. Output of menu-run macros goes to the Log window.
- **Plugins → Macros → Record…** opens the Macro Recorder, capturing menu
  operations (Open, Math, Resize, Convolve, Filters, Noise, FFT Bandpass, Edge
  Detection, Threshold, Z Project, Analyze Particles, Find Maxima, Set Scale,
  Labels Morphology/Voronoi, etc.) as runnable Python code with **Pause / Resume**
  and one-click **Create Macro**.
- **Process → Batch → Convert…** plans output destinations upfront with collision
  and source-overwrite protection (`rename`, `skip`, or `overwrite`).
- **Process → Batch → Run Macro…** runs a saved macro, recorded macro, or
  inline Python script across a folder of images (optional TIFF auto-save and
  combined `Results.csv` output). Failed commands or unreadable files fail fast
  and are recorded in `errors` rather than silently succeeding; multi-scene
  files can run on **Each scene separately**, **First scene only**, or **All
  scenes open, run once on active**.

## The `tiki` object

Without arguments, calls use the current image of the current window, as in
ImageJ. To pick another, pass `which`: a layer name or an index among that
window's image (or labels) layers, 0 = bottom and -1 = top, as listed in the
Layers panel. Pass `window` with a window title or index for another window.

```python
tiki.windows()                       # ['movie.tif', 'MAX_movie.tif']
tiki.layers()                        # ['C1-movie.tif', 'C2-movie.tif']
green = tiki.image('C1-movie.tif')
red = tiki.image(1)
avg = tiki.image(window='MAX_movie.tif')
```

A wrong name raises an error listing the names that exist.

| Call | Returns / does |
|---|---|
| `tiki.windows()`, `tiki.layers(kind='image')` | open window titles; layer names, bottom to top |
| `tiki.image(which=None, window=None)` | data of an image (memory-mapped when possible) |
| `tiki.layer(which, window)`, `tiki.viewer(window)` | the napari layer and window |
| `tiki.axes(which)`, `tiki.scale(which)`, `tiki.index()` | axis roles (`'TYX'`), physical steps, slider position |
| `tiki.show(data, name, axes='TYX', scale=None)` | open data in a new window |
| `tiki.add_image(data, name, like=None, window=None)`, `tiki.add_labels(...)` | add a layer aligned with image `like` |
| `tiki.labels(which=None, window=None)` | a Labels layer's data |
| `tiki.rois()`, `tiki.roi_masks()` | ROI Manager entries, (n, Y, X) masks |
| `tiki.add_roi(yx)`, `tiki.rois_from_labels(labels)` | add ROIs to the manager |
| `tiki.traces(masks=None, which=None, window=None)` | (n, frames) mean traces |
| `tiki.measure(**values)` | add a row to the Results table |
| `tiki.run('Measure')` | run any menu command by name (raises on errors in macros/batch) |
| `tiki.open(path)`, `tiki.save_tiff(path)` | open a file; save the active image/labels as calibrated TIFF |
| `tiki.math(op, value, whole_stack=True)` | scalar arithmetic or unary transform (32-bit float) |
| `tiki.resize(h, w, order=1)` | spatial resize preserving physical pixel calibration |
| `tiki.convolve(kernel, normalize=True, whole_stack=True)` | spatial 2D convolution across plane or stack |
| `tiki.edge_detection(method='canny', ...)` | Canny, Sobel, Scharr, Prewitt, Roberts, LoG, DoG (2D/3D) |
| `tiki.find_maxima(prominence=10.0, output='Point Selection', ...)` | local maxima detection across any of the 5 Fiji outputs |
| `tiki.filter(key, whole_stack=True, three_d=False, **params)` | run any registered filter (`'gaussian'`, `'variance'`, `'tophat'`, …) |
| `tiki.add_noise(sigma=25.0)`, `tiki.remove_outliers(...)`, `tiki.remove_nans(...)` | noise cleanup & synthetic noise generation |
| `tiki.bandpass_filter(...)` | 2D FFT bandpass filter with stripe suppression |
| `tiki.morphology_labels(op='open', ...)`, `tiki.voronoi(...)` | Labels morphology and Voronoi partitioning |
| `tiki.z_project(method='max')`, `tiki.threshold(lo, hi)` | Z/T projection; threshold to a Labels layer |
| `tiki.analyze_particles(...)`, `tiki.set_scale(sx, sy, unit)` | run Analyze Particles; set XY pixel calibration |
| `tiki.set_lut(name)` | apply a LUT (`'Fire'`, `'Cyan Hot'`, …) |
| `tiki.progress(iterable, 'label')` | progress bar with Cancel; keeps TIKI responsive |
| `tiki.status('text')`, `tiki.ops` | status bar message; TIKI's numpy operations |

## Example: rigid motion correction

```python
from scipy.ndimage import shift as nd_shift
from skimage.filters import window
from skimage.registration import phase_cross_correlation

movie = tiki.image()
template = np.median(np.asarray(movie[:200], np.float32), axis=0)
hann = window('hann', template.shape)
prep = lambda img: (img - img.mean()) * hann
out = np.empty(movie.shape, np.float32)
for t in tiki.progress(range(len(movie)), 'Motion correction'):
    frame = np.asarray(movie[t], np.float32)
    shift, _, _ = phase_cross_correlation(prep(template), prep(frame),
                                          upsample_factor=10, normalization=None)
    out[t] = nd_shift(frame, shift, order=1, mode='nearest')
tiki.show(out, 'MC_' + tiki.layer().name, axes=tiki.axes(), scale=tiki.scale())
```

Use `normalization=None` and a window as above: scikit-image's default
`'phase'` normalisation can lock onto image borders and report no shift.

## Limits

Macros run in TIKI's process and on its main thread: a long computation keeps
the windows busy unless the loop uses `tiki.progress(...)`. Packages must be
installed in TIKI's environment; CaImAn is easiest to install with conda.
