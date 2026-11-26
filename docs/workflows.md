# Everyday workflows

## Measure an ROI

1. Select an image window and an area-selection tool.
2. Draw the ROI. Hold {kbd}`Shift` while drawing to retain previous ROIs.
3. Press {kbd}`Ctrl+M`.
4. Copy the Results table or save it as CSV.

Measurements include area, mean, standard deviation, minimum, maximum,
integrated density, centroid, perimeter, circularity, Feret diameter, aspect ratio,
and solidity (selectable via **Analyze → Set Measurements…**). Calibrated images
report spatial measurements in physical units (`µm²`, `µm`, etc.) directly in
the column headers and exported CSVs; click **Summarize** in the Results window
for Mean / SD / Min / Max across rows. Point measurements also record
calibrated `(X, Y)` coordinates.

## Project or reorder a stack

- **Image → Stacks → Z Project…**: choose the axis and slice range, then choose
  maximum, average, sum, standard deviation, median or minimum. T and C axes are
  retained; TIKI never averages channels implicitly.
- **Image → Stacks → Stack to Hyperstack… / Hyperstack to Stack**: assign or
  flatten `C × Z × T` hyperstack structure.
- **Image → Stacks → Concatenate… / Images to Stack…**: join compatible stacks
  or single planes into one stack.
- **Image → Stacks → Reverse / Remove Slices…**: reverse slice order or drop
  slices/frames (e.g. `1-5, 10, 12-20-2`) along any stack axis.

## Threshold and analyze particles

1. Choose **Image → Adjust → Threshold…**.
2. Select an image layer and whether to process a plane, volume or whole stack.
3. Adjust the limits or choose an automatic method.
4. Enable 3D connectivity when objects should connect through Z.
5. Apply to create a Labels layer.
6. Choose **Analyze → Analyze Particles…** to produce an object table (2D with
   Area, centroid, Perim., Circ., Feret, AR, Solidity; or 3D Volume and Z using
   voxel size), optionally filtered by physical Min/Max size.

The threshold preview is temporary. Applying creates a new Labels layer and
does not alter the image.

Use **Process → Labels → Watershed…**, **Fill Holes**, **Morphology…**
(`Erode`, `Dilate`, `Open`, `Close`), **Distance Map** (calibrated Euclidean
distance map), **Skeletonize** (2D & 3D preserving label IDs), **Outline**, or
**Voronoi…** to clean up and measure segmented labels. Press {kbd}`Ctrl+Z`
(**Edit → Undo**) to undo Labels painting strokes, `Crop`, or added layers.

## Filter, Noise, FFT, maxima, resize, and math

Gaussian blur, Mean, Median, Minimum, Maximum, Variance, Top Hat, Unsharp Mask,
Convolve (with custom 2D kernels and presets), **Remove Outliers…**, **Bandpass
Filter…**, and rolling-ball Subtract Background show a live preview and display
kernel sizes in physical units alongside pixels when calibrated:

- **Process → Noise**: **Despeckle**, **Remove Outliers…** (`Bright` / `Dark` /
  `Both` hot pixels), **Remove NaNs…**, **Add Noise**, **Add Specified Noise…**,
  and **Salt and Pepper**.
- **Process → FFT**: **FFT** (log-power spectrum), **Inverse FFT**, and
  **Bandpass Filter…** (large/small feature filtering with horizontal or
  vertical stripe suppression).
- Top-level **Process** commands include **Smooth** ({kbd}`Ctrl+Shift+S`),
  **Sharpen**, **Find Edges** (one-click Sobel), **Edge Detection…** (Canny,
  Sobel, Scharr, Prewitt, Roberts, LoG, and DoG with live preview and
  anisotropic 3D support), and **Find Maxima…** (h-maxima prominence detection
  with live point preview and Point Selection, Single Points, Segmented
  Particles, List, and Count outputs).
- **Edit → Invert** inverts intensity non-destructively, **Process → Math…**
  applies scalar/unary math (`Add`, `Subtract`, `Multiply`, `Divide`, `Gamma`,
  `Log`, `Sqrt`, etc.), **Image → Adjust → Size…** resizes stacks/planes while
  updating physical pixel spacing, and **Image → Adjust → Sync Display Across
  Windows** copies LUTs and contrast ranges across open windows.

For a calibrated Z stack, 3D Gaussian, Mean, Median, Minimum, Maximum, Variance,
Top Hat, and Unsharp kernels account for anisotropic voxel dimensions.

## Colocalization

Choose **Analyze → Colocalization…**, select two greyscale channels or open
images (current plane or whole stack, optional ROI mask), and choose **Costes
automatic**, **Manual**, or **None** thresholding. TIKI displays a 2D
log-density cytofluorogram, Pearson's *r*, Spearman's ρ, Overlap R, Manders'
M1 / M2, Costes tM1 / tM2, and **Add to Results**.

## Batch conversion and macros

- **Process → Batch → Convert…**: convert a folder of images (including
  subfolders and filename globs) to calibrated `.tif`, `.png`, `.jpg`, or `.npy`.
- **Process → Batch → Run Macro…**: run a saved, recorded, or inline Python
  macro across matching files in a folder and save optional outputs plus
  `Results.csv`.

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
Scale…**. The pixel length is filled in; enter the known distance, optional
pixel aspect ratio (`Y/X`), and unit, check **Global** to calibrate all open
windows, or click **Remove Scale** to clear spatial calibration.

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
