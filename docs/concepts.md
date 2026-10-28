# How TIKI works

## Toolbar and image windows

TIKI runs one Qt application with a small global toolbar and one napari viewer
per image. The toolbar remembers the last activated viewer. Commands use that
viewer's selected image layer, reproducing ImageJ's active-image model while
retaining napari's layers and multidimensional rendering.

## Non-destructive operations

Processing commands normally create a new image or label layer. Display
contrast changes do not alter pixel values. This makes it easy to compare a
result with its source and reduces accidental data loss.

## ROIs and labels

ROIs are vector selections used for measurement and traces. Labels are raster
images in which each object has an integer ID. TIKI can convert between the two
and can read and write ImageJ `.roi` and `RoiSet.zip` files.

## Memory and large data

TIFF and NumPy arrays are memory-mapped when possible. Operations such as
projections and traces work in chunks rather than eagerly copying an entire
movie into RAM. Some algorithms still require an in-memory result; TIKI adds
that result as a new layer and leaves the source untouched.

## Errors

Expected problems are shown as actionable dialogs. Unexpected exceptions are
printed to the terminal and reported in the status bar rather than escaping a
Qt callback and terminating the process.
