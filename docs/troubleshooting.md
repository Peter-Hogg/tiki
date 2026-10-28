# Troubleshooting

## A command says it needs a different selection

Commands validate their input. For example, Plot Profile requires a straight
line and Crop requires an area ROI. Select the appropriate toolbar tool and
draw a new selection; the warning does not damage the image.

## A stack axis is unknown

Open **Image → Properties…** and identify the axis as T, Z or C. TIKI avoids
guessing because applying a Z operation to time—or combining channels—can
silently produce scientifically incorrect results.

## A TIFF header reports more frames than are present

This can happen when acquisition stopped early or a ScanImage recording was
split. TIKI opens the frames actually present and reports the mismatch. Verify
the acquisition before quantitative analysis.

## The viewer is blank or OpenGL fails

Update the graphics driver and test ordinary napari in the same environment.
Remote desktop and virtual machines may need software OpenGL. Include terminal
output and system details in a bug report.

## TIKI closes unexpectedly

Launch `tiki` from a terminal and copy the full traceback into an issue. Do not
share private paths or experimental data. See the project's security policy
for vulnerabilities.
