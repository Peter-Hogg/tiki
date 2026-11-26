# Troubleshooting

## Windows SmartScreen or macOS Gatekeeper warns on the standalone installer

Because the standalone installers do not yet carry commercial EV/Apple Developer
certificates, the OS shows an unverified-publisher prompt the first time:

- **Windows (`.exe`):** Click **More info → Run anyway** on the SmartScreen prompt.
- **macOS (`.pkg` / `TIKI.app`):** Right-click (**Ctrl**-click) the `.pkg` (or
  **TIKI.app** in **Applications** on first launch) and choose **Open → Open**,
  or click **Open Anyway** under **System Settings → Privacy & Security**.

## A Nikon `.nd2` file says `nd2` is missing

The `nd2` library is excluded from the standalone installer to avoid native build
issues. Install via `uv` or `pip` with `[formats]` / `[all]` (for example
`uv tool install "napari-tiki[all] @ git+https://github.com/Peter-Hogg/tiki"`),
or export from NIS-Elements as OME-TIFF.

## A command says it needs a different selection

Commands validate their input. For example, Plot Profile requires a straight
or segmented line and Crop requires an area ROI. Select the appropriate toolbar
tool and draw a new selection; the warning does not damage the image.

## A stack axis is unknown

Open **Image → Properties…** ({kbd}`Ctrl+Shift+P`) and identify the axis as T, Z
or C. TIKI avoids guessing because applying a Z operation to time—or combining
channels—can silently produce scientifically incorrect results.

## 3D view says Z-step is unknown

When X and Y are calibrated but Z has no step in the file metadata, TIKI assumes
cubic voxels (`sz = sy`) so 3D view and 3D particle analysis can run instead of
flattening Z, and warns you in the status bar and **Image → Show Info…**. Set the
true Z step in **Image → Properties…**.

## Mixing calibrated and pixel measurements in Results

If you measure a calibrated image (`µm`) and then measure an uncalibrated image
(or one with a different unit such as `nm`), TIKI clears the table's shared unit
column headers rather than mislabelling mixed rows. Press **Clear Results** to
start a fresh table for the new calibration.

## A TIFF header reports more frames than are present

This can happen when acquisition stopped early or a ScanImage recording was
split. TIKI opens the frames actually present and reports the mismatch. Verify
the acquisition before quantitative analysis.

## The viewer is blank or OpenGL fails

Update the graphics driver and test ordinary napari in the same environment.
Remote desktop and virtual machines may need software OpenGL. Include terminal
output and system details in a bug report.

## TIKI closes unexpectedly

Launch `tiki` from a terminal (or use the **Copy Error Details** button on TIKI's
error dialog) and include the full traceback in an issue. Do not share private
paths or experimental data. See the project's security policy for
vulnerabilities.
