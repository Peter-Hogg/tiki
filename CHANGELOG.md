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
- Sphinx documentation and GitHub Pages deployment.

## [0.0.1] - 2026-10-27

### Added

- First working TIKI prototype.

[Unreleased]: https://github.com/Peter-Hogg/tiki/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/Peter-Hogg/tiki/releases/tag/v0.0.1
