<p align="center">
  <img src="assets/tiki-logo-concept.png" alt="TIKI logo: a microscope containing a tropical island sunset" width="720">
</p>

<p align="center">
  <strong>An ImageJ-style toolbar for napari.</strong><br>
  Familiar tools and workflows, backed by Python image processing.
</p>

> [!NOTE]
> TIKI is an early prototype. The core workflow works, but interfaces and
> shortcuts may change while it is tested on real microscopy data.

## What is TIKI?

**TIKI Is Kinda ImageJ.** It is a small, persistent toolbar that opens and
controls napari image windows. Commands operate on the last selected window
and its active layer, giving ImageJ users a familiar “current image” workflow
without embedding ImageJ or Java.

TIKI is not intended to reproduce every Fiji plugin. Its goal is to make the
everyday microscopy loop—open, inspect, draw, measure, project and plot—fast
and approachable in Python.

## Quick start

TIKI currently runs from a source checkout and requires Python 3.10 or newer.

```bash
git clone https://github.com/Peter-Hogg/tiki.git
cd tiki

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .

tiki
```

Open an image directly from the command line:

```bash
tiki movie.tif
tiki recording.npy
```

For this development checkout, the launcher is also available directly:

```bash
~/Code/tiki/.venv/bin/tiki
```

To try TIKI without data, choose **File → Open Samples → Synthetic movie**.

## The basic workflow

1. Open a TIFF or NumPy array. Each image gets its own napari window.
2. Select a tool from the TIKI toolbar and draw on the active image.
3. Press <kbd>Ctrl</kbd>+<kbd>M</kbd> to measure the selection.
4. Use **Image → Stacks → Z Project…** to create a projection in a new window.
5. Press <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>Y</kbd> for a live intensity trace
   through the stack.
6. Click another image window; subsequent commands follow that window.

The status bar continuously reports the active image, cursor coordinates and
pixel value.

## Features

### ImageJ-like toolbar

- Rectangle, oval, polygon, freehand, straight-line and point selections
- Move/edit and pan tools
- New selections replace the previous selection; hold <kbd>Shift</kbd> to keep
  multiple selections
- Application-wide shortcuts that continue to work when a napari image window
  has focus

### Stacks and images

- Maximum, average, sum, standard-deviation, median and minimum projections
- Selectable projection range
- Chunked projection operations for memory-mapped movies
- Crop, duplicate, enhance contrast and reset contrast
- First-slice opening behavior, matching ImageJ rather than napari's default
  middle-slice position

### Measurement and plotting

- Results table with area, mean, standard deviation, minimum, maximum and
  integrated density
- Copy results or save them as CSV
- Straight-line intensity profiles
- Live ROI intensity profile through a stack, with current-frame indicator
- Copy profiles or save them as CSV

### Threshold → Labels

- ImageJ-style Threshold dialog: histogram, lower/upper sliders, Default,
  IsoData, Li, Mean, Minimum, Otsu, Percentile, Triangle and Yen methods,
  dark-background and stack-histogram options
- Live preview drawn as a translucent napari Labels layer on the current slice
- **Apply** adds a Labels layer to the same window, either a 0/1 mask or one id
  per connected object, with an optional minimum object size
- The result is selected with napari's label controls open, ready for touch-up
  with the brush, eraser and fill tools
- Brush and eraser tools on the TIKI toolbar paint into the selected Labels
  layer, or create an empty one for annotating from scratch

### Particles, ROI Manager and traces

- **Analyze Particles** measures every object in a Labels layer (Area, Mean,
  StdDev, Min, Max, X, Y, Perim., Circ., AR, Solidity, IntDen), on one slice or
  the whole stack, with min size and exclude-on-edges options
- **ROI Manager**: Add (<kbd>Ctrl</kbd>+<kbd>T</kbd>), Update, Delete, Rename,
  Measure, Multi Plot, Show All with names; it is global, so ROIs drawn on one
  window (e.g. an average projection) can be used on another (the movie)
- ImageJ/Fiji compatible: Open and Save `.roi` and `RoiSet.zip`
- **From Labels / To Labels** converts between objects and ROIs
- **Traces**: one mean trace per ROI or label through the stack, shown as
  stacked lines or a heatmap, optional ΔF/F (F0 = median), exported as a
  cells × frames `.npy` (plus names) or CSV
- ROI layers are 2D, so a selection stays in place while you scroll frames

### Filters

- Gaussian Blur, Median and rolling-ball Subtract Background with a live
  preview of the current slice
- Non-destructive: the result is added as a new layer above the original

### Send to Notebook

- Exports the current window to a folder: labels, other layers, ROIs
  (`RoiSet.zip`, `rois.json`, `roi_masks.npy`), traces, results and a manifest
- Writes `load_session.py` (numpy/tifffile only) and copies it to the clipboard;
  the source movie is referenced by path, not copied

### File handling

- TIFF/OME-TIFF via `tifffile`
- NumPy `.npy` arrays
- Memory mapping where the format permits it
- Graceful handling of incomplete ScanImage TIFFs whose header advertises more
  frames than the file contains

## Commands and shortcuts

| Menu | Command | Shortcut |
|---|---|---|
| File | Open… | <kbd>Ctrl</kbd>+<kbd>O</kbd> |
| File | Close / Close All | <kbd>Ctrl</kbd>+<kbd>W</kbd> / <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>W</kbd> |
| File | Save As TIFF… | <kbd>Ctrl</kbd>+<kbd>S</kbd> |
| Edit → Selection | Select All / Select None | <kbd>Ctrl</kbd>+<kbd>A</kbd> / <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>A</kbd> |
| Image → Adjust | Brightness/Contrast | <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>C</kbd> |
| Image → Adjust | Threshold… (preview and result are napari Labels) | <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>T</kbd> |
| Image → Adjust | Enhance Contrast | <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>E</kbd> |
| Image | Show Info | <kbd>Ctrl</kbd>+<kbd>I</kbd> |
| Image | Crop | <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>X</kbd> |
| Image | Duplicate | <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>D</kbd> |
| Process → Filters | Gaussian Blur…, Median… (live preview, result as new layer) | |
| Process | Subtract Background… (rolling ball, light background) | |
| Image → Stacks | Plot Z-axis Profile | <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>Y</kbd> |
| Analyze | Measure | <kbd>Ctrl</kbd>+<kbd>M</kbd> |
| Analyze | Analyze Particles… (per-object table from a Labels layer) | |
| Analyze → Traces | From ROI Manager / From Labels (cells × time) | |
| Analyze → Tools | ROI Manager… / Add to Manager | — / <kbd>Ctrl</kbd>+<kbd>T</kbd> |
| File → Export | Send to Notebook… | <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>N</kbd> |
| Analyze | Plot Profile | <kbd>Ctrl</kbd>+<kbd>K</kbd> |
| Help | Find Commands | <kbd>Ctrl</kbd>+<kbd>L</kbd> |

TIKI removes conflicting napari bindings from windows it manages. For
example, napari normally uses <kbd>Ctrl</kbd>+<kbd>M</kbd> to toggle the menu bar
on Linux; in TIKI it consistently means **Measure**.

## Development

Install the test dependencies and run the suite headlessly:

```bash
cd ~/Code/tiki
source .venv/bin/activate
python -m pip install -e '.[test]'
QT_QPA_PLATFORM=offscreen python -m pytest -q
```

The code is intentionally split so image processing can be tested without the
GUI:

```text
src/tiki/
├── app.py       toolbar, menus, shortcuts and commands
├── manager.py   napari window registry and active-window tracking
├── ops.py       NumPy image operations
├── roi.py       selection layers, masks and current-slice helpers
├── io.py        TIFF and NumPy loading
├── widgets.py   Results and profile windows
├── threshold.py Threshold dialog producing napari Labels
├── rois.py      ROI model, ImageJ .roi/.zip and Labels conversion
├── roi_manager.py ROI Manager window
├── filters.py   Process filters with live preview
├── export.py    Send to Notebook
├── style.py     fixed light theme for TIKI windows
└── icons.py     toolbar icon drawing
```

## Roadmap

- A dedicated Brightness/Contrast dialog
- Better OME and acquisition metadata handling
- Calibration-aware measurements in physical units
- More everyday ImageJ stack and image commands
- Packaging and installation documentation for non-developers

## Name

**TIKI** is a recursive acronym: **T**IKI **I**s **K**inda **I**mageJ. The name
also nods to Fiji and napari's island theme without depending on ImageJ or
shipping any Java code.

## License

A license has not been selected yet. Add a `LICENSE` file before distributing
or accepting external contributions.
