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
- Standalone app polish: lightweight startup splash screen (`--no-splash` to skip), `Help ▸ Check for Updates…`, GUI error dialogs with collapsible tracebacks & "Copy Error Details" button, `[all]` / `[video]` extras (`imageio-ffmpeg` + microscopy formats), and `Plugins ▸ Python Console…` (`Ctrl+Shift+J`).
- Sphinx documentation, GitHub Pages deployment, and non-developer standalone installer guides (`installer/README.md`, Windows SmartScreen / macOS Gatekeeper / Linux walkthroughs, dynamic version in `docs/conf.py`).
- Better OME and acquisition metadata handling: namespace-agnostic OME-XML parser (`parse_ome_xml` with Greek `μm`, `<Plane DeltaT>` fallback, channel names/colors/wavelengths, `<Instrument>`/`<Objective>`/`<Detector>`/`<StageLabel>`), multi-series OME-TIFF reading (`read_tiff_scenes`), ScanImage & Micro-Manager TIFF metadata, extended CZI (`parse_czi_metadata`) / ND2 / LIF / OME-Zarr acquisition metadata, OME channel name/color roundtrip in `save_tiff`, channel-named `Split Channels`, and scrollable `Image ▸ Show Info…` window with raw metadata toggle & `.txt` export.
- Calibration-aware measurements in physical units: `convert_unit` / `spatial_unit` / `physical_scale_3d`, anisotropic perimeter & `Feret` metrics in `mask_stats` and 2D `particle_table`, calibrated point measurements, ROI Manager `Measure` calibration & line `Length`, 3D `Analyze Particles…` physical Min/Max size filters and mixed-unit guards, Results table `Summarize` (Mean/SD/Min/Max) & CSV unit headers, calibrated `ProfileWindow` / `TracesWindow` / Bleach / Motion Correction / Kymograph axes, physical kernel readouts in filters/Watershed/Enlarge/Make Band, `Set Scale…` pixel aspect ratio / Global / Remove Scale, and extended `Send to Notebook` `session.json` metadata.
- Everyday ImageJ stack and image commands: `Edit ▸ Invert`, `Image ▸ Adjust ▸ Size…`, `Process` (`Smooth` `Ctrl+Shift+S`, `Sharpen`, `Find Edges`, `Edge Detection…` [Canny, Sobel, Scharr, Prewitt, Roberts, LoG, DoG], `Find Maxima…`, `Math…`), `Process ▸ Filters` (`Convolve…` with presets, `Mean…`, `Minimum…`, `Maximum…`, `Variance…`, `Top Hat…`, `Unsharp Mask…` 2D/3D), `Process ▸ Labels ▸ Morphology…` (binary and per-label 2D/3D erode/dilate/open/close), `Image ▸ Stacks` (`Concatenate…`, `Images to Stack…`, `Hyperstack to Stack`, `Reverse`, `Remove Slices…`), and multi-window `Merge Channels`.
- Python Macro Recorder (`Plugins ▸ Macros ▸ Record…`), programmatic `MacroAPI` helpers (`open`, `save_tiff`, `filter`, `z_project`, `threshold`, `analyze_particles`, `set_scale`, `set_lut`), and folder `Process ▸ Batch ▸ Convert…` / `Run Macro…`.
- Zero-dependency Noise & FFT cleanup (`Process ▸ Noise`: `Despeckle`, `Remove Outliers…`, `Remove NaNs…`, `Add Noise`, `Add Specified Noise…`, `Salt and Pepper`; `Process ▸ FFT`: `FFT`, `Inverse FFT`, `Bandpass Filter…` with stripe suppression).
- Colocalization analysis (`Analyze ▸ Colocalization…`: Pearson's r, Spearman's ρ, Overlap R, Manders' M1/M2, Costes auto-thresholds, and cytofluorogram), Labels morphometry (`Process ▸ Labels`: calibrated `Distance Map`, 2D/3D `Skeletonize`, `Outline`, `Voronoi…`), `Edit ▸ Undo` (`Ctrl+Z`), and `Image ▸ Adjust ▸ Sync Display Across Windows` (plus `Channels Tool ▸ Sync to All Windows`).

## [0.0.1] - 2026-10-27

### Added

- First working TIKI prototype.

[Unreleased]: https://github.com/Peter-Hogg/tiki/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/Peter-Hogg/tiki/releases/tag/v0.0.1
