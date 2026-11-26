# Getting started

You can install TIKI either with a **one-click standalone installer** (no terminal or Python setup needed) or into an existing Python environment using `uv` or `pip`.

## Choose an installation method

| Method | Best for | Terminal needed? | Extra formats & `.nd2` | Updating |
|---|---|---|---|---|
| **Standalone Installer** | Everyday microscopy & ImageJ/Fiji users | No (double-click; one bash command on Linux) | Bundles `.tif`, `.czi`, `.lif`, `.zarr`, `.h5`, `.roi`, MP4/GIF (no `.nd2`) | Download newer installer |
| **`uv tool install`** | Quick CLI setup with isolated Python | 1 command | Use `--with nd2` or `.[all]` | `uv tool upgrade napari-tiki` |
| **`pip install`** | Existing Python / CaImAn / Suite2p environments | Yes | Choose `[formats]`, `[video]`, or `[all]` extras | `pip install --upgrade ...` |

## Standalone Installer (Recommended for non-developers)

Pre-built installers are attached to every release on the [**GitHub Releases**](https://github.com/Peter-Hogg/tiki/releases) page. Each bundles Python 3.12, napari, Qt 6 (`PySide6`), `.tif`/OME-TIFF, Zeiss `.czi`, Leica `.lif`, OME-Zarr, HDF5, ImageJ ROIs, and MP4/GIF movie export inside a self-contained folder so nothing on your system Python is modified.

::::{tab-set}

:::{tab-item} Windows
1. Download `TIKI-<version>-Windows-x86_64.exe` from [**GitHub Releases**](https://github.com/Peter-Hogg/tiki/releases).
2. Double-click the `.exe` installer. Because the installer is not signed with an expensive code-signing certificate yet, **Windows Defender SmartScreen** usually shows *"Windows protected your PC"*:
   - Click **More info**.
   - Click **Run anyway**.
3. Follow the wizard (default install location inside your user folder needs **no administrator privileges**).
4. Launch **TIKI** from the Start Menu or Desktop shortcut.
:::

:::{tab-item} macOS (Apple Silicon)
1. Download `TIKI-<version>-MacOSX-arm64.pkg` from [**GitHub Releases**](https://github.com/Peter-Hogg/tiki/releases).
2. Double-click the `.pkg` (or right-click **Open** if macOS warns about an unidentified developer) and complete the installer.
3. On your first launch of **TIKI** from Applications, **macOS Gatekeeper** may block an unsigned app:
   - Right-click (or {kbd}`Ctrl`-click) **TIKI** in **Applications** and choose **Open** → **Open**; **or**
   - Attempt to open TIKI once, then open **System Settings → Privacy & Security**, scroll down to **Security**, and click **Open Anyway**.
:::

:::{tab-item} Linux
1. Download `TIKI-<version>-Linux-x86_64.sh` from [**GitHub Releases**](https://github.com/Peter-Hogg/tiki/releases).
2. Run the installer in a terminal and accept the prompts:

   ```bash
   bash TIKI-*-Linux-x86_64.sh
   ```

3. Launch **TIKI** from your desktop applications menu (or run `~/tiki/bin/tiki` depending on your chosen install prefix).
:::

::::

```{note}
**Nikon `.nd2` users:** The `nd2` reader library is currently excluded from the standalone installer. To open `.nd2` files natively, install TIKI via `uv` or `pip` with `[formats]` / `[all]` below (or save as OME-TIFF in NIS-Elements).
```

## Command-line install (`uv` or `pip`)

Requirements: **Python 3.11 or newer**, working **OpenGL graphics**, and **Linux, macOS, or Windows**.

### Isolated install with `uv` (recommended CLI method)

With [uv](https://docs.astral.sh/uv/), one command installs TIKI with its own Python, napari, and Qt, and puts `tiki` on your `PATH`:

```bash
uv tool install "napari-tiki[all] @ git+https://github.com/Peter-Hogg/tiki"
```

*(Omit `[all]` if you only need TIFF, OME-TIFF, CZI, and NumPy.)* Install `uv` first if needed (`curl -LsSf https://astral.sh/uv/install.sh | sh` on Linux/macOS; see the [uv installation docs](https://docs.astral.sh/uv/getting-started/installation/) for Windows). The first install downloads a few hundred MB and takes a few minutes.

To try TIKI without installing it permanently:

```bash
uvx --from "git+https://github.com/Peter-Hogg/tiki" tiki --sample
```

### Existing Python environment (`pip`)

To install into an existing environment (for example one that already has CaImAn or Suite2p, so TIKI Python macros can import them):

```bash
pip install "napari-tiki[all] @ git+https://github.com/Peter-Hogg/tiki"
```

Available optional extras in `pyproject.toml`:

- `[formats]`: Nikon `.nd2` (`nd2`), Leica `.lif` (`readlif`), OME-Zarr `.zarr` (`zarr`), HDF5 / MATLAB v7.3 `.h5`/`.mat` (`h5py`)
- `[video]`: MP4/AVI video encoding (`imageio-ffmpeg`)
- `[all]`: all of `[formats]` and `[video]`

### Add TIKI to the Linux applications menu (`uv`/`pip` installs)

```bash
tiki --install-desktop
```

TIKI appears in your applications menu (Activities on GNOME) with its icon so you can pin it to the dock, drop files onto it, or right-click images and choose **Open With → TIKI**. Undo with `tiki --uninstall-desktop`.

### From source (for development)

```bash
git clone https://github.com/Peter-Hogg/tiki.git
cd tiki
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[all,test]'
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

## Launch

Start TIKI from your Start Menu / Applications shortcut, or in a terminal:

```bash
tiki
```

Open one or more images immediately:

```bash
tiki movie.tif second-image.ome.tif
```

You can also drag files onto the toolbar or any image window, as in ImageJ: images, multi-series OME-TIFFs, and folders of images open in new windows, ImageJ ROI sets (`.roi`, `.zip`) open in the ROI Manager, and `.py` macros open in the macro editor. To explore without your own data, choose **File → Open Samples → Synthetic movie** (or **Cells 3D** / **Mouse Kidney**), or run `tiki --sample`. Useful CLI flags: `--no-splash` skips the splash screen, and `tiki --version` prints the installed version.

## First five minutes

1. Choose **File → Open Samples → Cells 3D (2-ch Z-stack)** — TIKI opens the calibrated stack in `µm`, splits its channels, and labels `z` in physical coordinates.
2. Scroll through `z` with the slider or open **Image → Show Info…** ({kbd}`Ctrl+I`) to inspect pixel size, axes, and acquisition metadata.
3. Choose the rectangle tool and draw around a cell, then press {kbd}`Ctrl+M` to measure **Area**, centroid, and intensity directly in `µm²` and `µm`.
4. Click **Summarize** in the Results window to see Mean / SD / Min / Max across rows, or press {kbd}`Ctrl+Shift+Y` to plot the Z-axis intensity profile in `µm`.
5. Choose **Image → Stacks → Z Project…** and make a maximum projection; click between windows to target commands at the active window.

## Updating TIKI

- **Standalone Installer:** Use **Help → Check for Updates…** in the toolbar to check GitHub Releases, or download and run the latest installer over your existing installation.
- **`uv` install:** Run `uv tool upgrade napari-tiki`.
- **`pip` / source checkout:**

  ```bash
  cd tiki
  git pull
  source .venv/bin/activate
  python -m pip install -e .
  ```
