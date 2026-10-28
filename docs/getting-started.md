# Getting started

## Requirements

- Python 3.10 or newer
- A desktop with working OpenGL graphics
- Linux, macOS or Windows

## Install from source

Until packaged releases are available, install TIKI from its GitHub checkout:

```bash
git clone https://github.com/Peter-Hogg/tiki.git
cd tiki
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

## Launch

Start with an empty toolbar:

```bash
tiki
```

Or open one or more images immediately:

```bash
tiki movie.tif second-image.ome.tif
```

You can also drop TIFF or NumPy files onto the toolbar. To explore without
data, select **File → Open Samples → Synthetic movie**.

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
