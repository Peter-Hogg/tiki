# Multidimensional data

Scientific image arrays need axis meanings, not just shapes. A `(100, 512,
512)` array might be time, depth, or an unlabelled sequence.

## Axis roles

TIKI uses five explicit roles:

| Role | Meaning | Typical unit |
|---|---|---|
| T | time | seconds (`s`) |
| Z | depth | micrometres (`µm`) |
| C | channel | none |
| Y | row | micrometres (`µm`) |
| X | column | micrometres (`µm`) |

Physical calibration, channel names/colors, and acquisition metadata are read from:

- **OME-TIFF / OME-XML**: `<Pixels>` physical sizes and units (including Greek `μm`), `<Plane DeltaT>` timestamps when `TimeIncrement` is absent, multi-series `Image` datasets (each series opens in its own window), channel names/colors/wavelengths, and `<Instrument>`/`<Objective>`/`<Detector>`/`<StageLabel>` tags.
- **ImageJ hyperstacks**: `frames`, `slices`, `channels`, `finterval`, `spacing`, `unit`, and channel LUTs.
- **ScanImage** (`.tif`) and **Micro-Manager** metadata: FOV pixel size, Z step, frame rate/interval, and acquisition settings.
- **Zeiss `.czi`**, **Nikon `.nd2`**, **Leica `.lif`**, and **OME-Zarr `.zarr`**: axis order, physical steps, channel names, and acquisition metadata.

Generic TIFF sequences and `.npy` arrays remain `dim0`, `dim1`, and so on when their meaning is ambiguous. Choose **Image → Show Info…** ({kbd}`Ctrl+I`) at any time to inspect dimensions, calibration, acquisition metadata, and toggleable raw headers/XML.

## Correcting axes and calibration

Choose **Image → Properties…** ({kbd}`Ctrl+Shift+P`) to assign roles, pixel size, Z step and frame interval. TIKI validates that roles are unique and that the final two axes are Y and X. You can remember a choice for files with the same dimensionality in a folder.

To calibrate from an image feature, draw a line of known physical length and choose **Analyze → Set Scale…** (supports pixel aspect ratio `Y/X`, **Global** application across open windows, and **Remove Scale**).

## Channels

**Image → Color → Split Channels** exposes a C axis as independently coloured,
additively blended layers without copying data, keeping parsed channel names
(`C1 (DAPI)-…`) and file LUTs/colors when present. Running **Merge Channels** on
split layers reconstructs the original multidimensional image; running **Merge
Channels** when no channels are split lets you combine greyscale layers or open
images into a composite stack.

## Hyperstacks, concatenation & stacks

- **Image → Stacks → Stack to Hyperstack…** and **Hyperstack to Stack** reshape between interleaved stacks and explicit `C × Z × T` hyperstacks.
- **Image → Stacks → Concatenate…** and **Images to Stack…** join multiple open images or stacks into one stack.
- **Image → Stacks → Reverse** and **Remove Slices…** reorder or trim slices/frames along any stack axis (`1-5, 10, 12-20-2`).

## 3D display and anisotropic Z

In 3D, drawing tools are disabled so dragging rotates the scene. Calibrated Z
steps set the voxel aspect ratio (normalizing mixed units such as `nm` in Z and
`µm` in XY automatically). If Z calibration is absent while XY is calibrated,
TIKI assumes cubic voxels (`sz = sy`) for display and 3D measurement, and
reports that assumption in the status bar and **Show Info…**.
