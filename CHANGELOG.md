# Changelog

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and intends to use [Semantic Versioning](https://semver.org/) for releases.

## [Unreleased]

### Added

- ImageJ-style global toolbar controlling multiple napari windows.
- TIFF, OME-TIFF, ScanImage, Zeiss CZI and NumPy loading with cautious axis handling.
- ROI Manager with ImageJ ROI-set interoperability and multi-ROI traces.
- Thresholding, particle analysis, projections and common filters.
- Results tables, line profiles, stack traces and notebook export.
- Dedicated Brightness/Contrast dialog (`Ctrl+Shift+C`), Lookup Tables menu (`Fire`, `Cyan Hot`, `HiLo`, etc.) and per-channel contrast sliders in the Channels tool.
- `Duplicate…` / `Make Substack…` slice-range dialog, `Bin…` stack/spatial downsampling, lazy Dask `Image Sequence…` import (`Ctrl+Shift+I`), and calibrated ImageJ hyperstack / Labels TIFF saving.
- First-class rigid `Motion Correction…` (with dual-channel shift application and drift plot) and `Labels ▸ Watershed…` / `Fill Holes`.
- Live polyline `Plot Profile` (`Ctrl+K`) with adjustable width, Magic Wand & segmented-line toolbar tools, `Restore Selection` (`Ctrl+Shift+R`), `Enlarge / Shrink…`, `Make Band…`, `Clear / Clear Outside`, `Image ▸ Transform`, `Image ▸ Type` (8/16/32-bit & RGB flatten), `Set Measurements…`, and calibrated units in Results headers and status bar.
- Optional `[formats]` readers for Nikon `.nd2`, Leica `.lif`, lazy OME-Zarr `.zarr` (`File ▸ Import ▸ OME-Zarr…`), and HDF5 / MATLAB v7.3 `.h5`/`.hdf5`/`.mat`.
- Publication & talk exports: `File ▸ Export ▸ Snapshot (PNG / SVG)…` & `Analyze ▸ Tools ▸ Scale Bar…` (composite channels + ROI overlays + calibrated burned-in scale bar), `File ▸ Export ▸ Export Movie (MP4 / GIF / AVI)…` (with FPS, scale bar & timestamps), `Image ▸ Stacks ▸ Make Montage…`, and linked `Image ▸ Stacks ▸ Orthogonal Views` (`Ctrl+Shift+H`).
- Sphinx documentation and GitHub Pages deployment.

## [0.0.1] - 2026-10-27

### Added

- First working TIKI prototype.

[Unreleased]: https://github.com/Peter-Hogg/tiki/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/Peter-Hogg/tiki/releases/tag/v0.0.1
