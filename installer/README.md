# TIKI Standalone Installers

This directory builds the standalone graphical installers for **Windows**, **macOS**, and **Linux** using [conda `constructor`](https://conda.github.io/constructor/). End users get an isolated Python + napari + TIKI environment with Start Menu / Applications / `.desktop` shortcuts and do not need to manage Python or virtual environments.

## Files in this directory

- `construct.yaml`: Main `constructor` recipe pinning `conda-forge` packages (Python 3.12, `napari >=0.9`, `pyside6`, scientific Python stack, `tifffile`, `imagecodecs`, `dask-core`, `imageio-ffmpeg`, `h5py`, `zarr`).
- `build_icons.py`: Generates `tiki.png`, `tiki.ico`, and `tiki.icns` from `src/tiki/icons.py` (`Pillow` required).
- `menu.json`: `menuinst >=2.1` shortcut specification registered during installation.
- `post_install.sh` / `post_install.bat`: Unpacks `wheels.zip`, installs `napari-tiki` and pure-Python/pip microscopy readers offline (`--no-deps --no-index`), removes the temporary archive, and runs `menuinst.api.install`.

> [!NOTE]
> The standalone installer includes `.tif`/OME-TIFF (`tifffile` + `imagecodecs`), Zeiss `.czi` (`czifile`), Leica `.lif` (`readlif`), OME-Zarr (`zarr`), HDF5/MATLAB v7.3 (`h5py`), ImageJ `.roi`/`.zip` (`roifile`), and MP4/GIF export (`imageio-ffmpeg`). Nikon `.nd2` (`nd2`) is **not** bundled in the offline installer; `.nd2` users should install TIKI via `uv tool install` or `pip install "napari-tiki[formats] @ ..."`.

## Automated releases (GitHub Actions)

Tagging a release (e.g. `git tag v0.1.0 && git push --tags`) triggers [`.github/workflows/release.yml`](../.github/workflows/release.yml), which builds on `windows-latest`, `macos-latest` (`arm64`), and `ubuntu-latest` (`x86_64`) and attaches:

- `TIKI-<version>-Windows-x86_64.exe`
- `TIKI-<version>-MacOSX-arm64.pkg`
- `TIKI-<version>-Linux-x86_64.sh`

Because the installers are not signed with commercial Apple/Microsoft developer certificates, Windows SmartScreen and macOS Gatekeeper show an unverified-publisher prompt on first launch; see the [Getting Started guide](../docs/getting-started.md) for click-through steps.

## Building locally

Run inside a clean base conda/mamba environment that has `constructor >=3.9`, `pillow`, and `pip`:

```bash
python installer/build_icons.py
mkdir -p installer/wheels dist
python -m pip wheel --no-deps . -w installer/wheels
python -m pip wheel --no-deps czifile roifile readlif -w installer/wheels
python -c "import glob, os, zipfile; zf = zipfile.ZipFile('installer/wheels.zip', 'w'); [zf.write(p, os.path.basename(p)) for p in sorted(glob.glob('installer/wheels/*.whl'))]; zf.close()"

# Replace __VERSION__ with the target version, then run constructor:
sed -i.bak 's/__VERSION__/0.1.0/g' installer/construct.yaml && rm -f installer/construct.yaml.bak
constructor installer/ --output-dir dist/
```

Remember not to commit `installer/wheels/`, `installer/wheels.zip`, `installer/tiki.*`, or version-stamped edits to `installer/construct.yaml`.
