# Multidimensional data

Scientific image arrays need axis meanings, not just shapes. A `(100, 512,
512)` array might be time, depth, or an unlabelled sequence.

## Axis roles

TIKI uses five explicit roles:

| Role | Meaning | Typical unit |
|---|---|---|
| T | time | seconds |
| Z | depth | micrometres |
| C | channel | none |
| Y | row | micrometres |
| X | column | micrometres |

ImageJ hyperstack, OME and ScanImage metadata are used when available. Generic
TIFF sequences and `.npy` arrays remain `dim0`, `dim1`, and so on when their
meaning is ambiguous.

## Correcting axes and calibration

Choose **Image → Properties…** to assign roles, pixel size, Z step and frame
interval. TIKI validates that roles are unique and that the final two axes are
Y and X. You can remember a choice for files with the same dimensionality in a
folder.

## Channels

**Image → Color → Split Channels** exposes a C axis as independently coloured,
additively blended layers without copying data. **Merge Channels** reconstructs
the multidimensional image.

## 3D display

In 3D, drawing tools are disabled so dragging rotates the scene. Calibrated Z
steps set the voxel aspect ratio. If Z calibration is absent, TIKI assumes
cubic voxels for display and reports that assumption in the status bar.
