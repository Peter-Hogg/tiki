# ROI Manager and traces

Open **Analyze → Tools → ROI Manager…**. The manager is global: an ROI drawn on
a projection can be applied to its source movie when the image dimensions
match.

## Managing ROIs

- **Add** or {kbd}`Ctrl+T`: store the active selection.
- **Update**: replace the selected stored ROI with the current selection.
- **Delete** and **Rename**: organize the ROI set.
- **Show All**: overlay stored ROIs and names.
- **Measure**: add measurements for selected ROIs.
- **Multi Plot**: plot one trace per ROI through T, or through Z if no T axis
  exists.

## ImageJ interoperability

TIKI reads and writes individual ImageJ `.roi` files and `RoiSet.zip`
archives. Pixel coordinates are preserved for round trips between Fiji and
TIKI.

## Labels and traces

Use **From Labels** to create one ROI per labelled object, or **To Labels** to
rasterize ROIs. Traces can be exported as CSV or as a cells × frames NumPy
array. Optional ΔF/F uses each trace's median as F0.
