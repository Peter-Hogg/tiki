# Getting started

## Requirements

- Python 3.11 or newer
- A desktop with working OpenGL graphics
- Linux, macOS or Windows

## Install

With [uv](https://docs.astral.sh/uv/), one command installs TIKI with its own
Python, napari and Qt, and puts `tiki` on your PATH:

```bash
uv tool install git+https://github.com/Peter-Hogg/tiki
```

Install uv first if needed (`curl -LsSf https://astral.sh/uv/install.sh | sh`
on Linux/macOS; see the uv site for Windows). The first install downloads a
few hundred MB and takes a few minutes. Later: `uv tool upgrade napari-tiki`.

To use an existing environment instead (for example one with CaImAn, so
macros can import it):

```bash
pip install "napari-tiki @ git+https://github.com/Peter-Hogg/tiki"
```

### Add TIKI to the applications menu (Linux)

```bash
tiki --install-desktop
```

TIKI then appears in the applications menu (Activities on Ubuntu/GNOME) with
its icon: pin it to the dock, drop files onto the icon to open them, or
right-click a TIFF and choose **Open With → TIKI**. Undo with
`tiki --uninstall-desktop`. For a keyboard shortcut, add a custom shortcut in
your desktop's keyboard settings with the command `tiki`.

### From source

```bash
git clone https://github.com/Peter-Hogg/tiki.git
cd tiki
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

## Launch

Start with an empty toolbar:

```bash
tiki
```

Or open one or more images immediately:

```bash
tiki movie.tif second-image.ome.tif
```

You can also drag files onto the toolbar or any image window, as in ImageJ:
images and folders of images open in new windows, ImageJ ROI sets
(`.roi`, `.zip`) open in the ROI Manager and `.py` macros in the editor. To
explore without data, run `tiki --sample` or choose **File → Open Samples →
Synthetic movie**. `tiki --version` reports the installed version.

## First five minutes

1. Open a stack and move the frame slider.
2. Choose the rectangle tool and draw around a structure.
3. Press {kbd}`Ctrl+M` to add a measurement to the Results table.
4. Press {kbd}`Ctrl+Shift+Y` to plot the ROI intensity through time or Z.
5. Choose **Image → Stacks → Z Project…** and make a maximum projection.
6. Click between the source and projection windows. TIKI commands follow the
   most recently selected window.

## Updating a source checkout

```bash
cd tiki
git pull
source .venv/bin/activate
python -m pip install -e .
```
