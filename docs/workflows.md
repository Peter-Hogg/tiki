# Everyday workflows

## Measure an ROI

1. Select an image window and an area-selection tool.
2. Draw the ROI. Hold {kbd}`Shift` while drawing to retain previous ROIs.
3. Press {kbd}`Ctrl+M`.
4. Copy the Results table or save it as CSV.

Measurements include area, mean, standard deviation, minimum, maximum and
integrated density. Calibrated images report spatial measurements in physical
units.

## Project a stack

Choose **Image → Stacks → Z Project…**, select the axis and slice range, then
choose maximum, average, sum, standard deviation, median or minimum. T and C
axes are retained; TIKI never averages channels implicitly.

## Threshold and analyze particles

1. Choose **Image → Adjust → Threshold…**.
2. Select an image layer and whether to process a plane, volume or whole stack.
3. Adjust the limits or choose an automatic method.
4. Enable 3D connectivity when objects should connect through Z.
5. Apply to create a Labels layer.
6. Choose **Analyze → Analyze Particles…** to produce an object table.

The threshold preview is temporary. Applying creates a new Labels layer and
does not alter the image.

## Filter an image

Gaussian blur, median filtering and rolling-ball background subtraction show a
live preview. Applying adds a new image layer. For a calibrated Z stack, 3D
Gaussian and median kernels account for anisotropic voxel dimensions.

## Send a session to a notebook

Choose **File → Export → Send to Notebook…**. TIKI writes labels, ROIs, traces,
results and a manifest into a folder, plus a lightweight `load_session.py`.
The source movie is referenced by path rather than duplicated.

## ΔF/F and bleaching

**Process → ΔF/F…** computes a ΔF/F movie along T. Choose F0 as the median,
mean, a percentile, the mean of the first frames, or a sliding percentile for
slow drift, and optionally subtract a camera offset first. The result uses a
diverging colormap centred on zero.

**Process → Bleach Correction…** fits a single exponential to the frame means
and divides by it, reporting τ. If the means are not an exponential decay (for
example a dip caused by a stimulus or motion), TIKI shows the plot and corrects
nothing. The simple-ratio method removes every change in whole-frame
brightness, including real global signals.

## Kymographs

Draw a straight line or a path with the line tool, then choose **Image → Stacks
→ Kymograph / Reslice…**. Rows are time (or Z), columns are distance along the
line; a line width greater than one averages across the line.

## Calibrate from a known distance

Draw a line across a feature of known length and choose **Analyze → Set
Scale…**. The pixel length is filled in; enter the known distance and unit.

## Find active cells in a calcium movie

1. Open the movie and check that its stack axis is T (Image → Properties).
2. Choose **Image → Stacks → Correlation Image**. Each pixel shows how strongly
   it co-fluctuates with its neighbours, so firing cells are bright even where
   the illumination is dim, and bright but inactive regions stay dark.
3. Threshold the correlation image (Otsu works well), Apply, then use the ROI
   Manager's **From Labels** and switch back to the movie for **Multi Plot**.

This is a quick first pass. It does not demix overlapping cells or subtract
neuropil the way CNMF (CaImAn) or Suite2p do.

If the data has a negative offset (common for resonant-scanner recordings),
ΔF/F needs that offset removed first: TIKI pre-sets it to the data minimum and
says so. ΔF/F amplitudes depend on this choice, so enter the true dark level if
you know it.
