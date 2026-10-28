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
