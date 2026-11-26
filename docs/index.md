# TIKI

```{image} _static/tiki-logo.png
:alt: TIKI logo
:width: 620px
:align: center
```

**TIKI Is Kinda ImageJ**—a small, persistent ImageJ-style toolbar that opens
and controls napari image windows.

TIKI brings the familiar *current image* workflow to Python: open a TIFF, draw
an ROI, measure it, inspect a trace, make a projection, and move to the next
image without assembling a collection of unrelated widgets.

::::{grid} 1 1 1 1
:gutter: 2

:::{grid-item-card} Cells 3D — Orthogonal Views & Calibrated ROI Measurement
:text-align: center
```{image} _static/demo-cells3d.gif
:alt: Using TIKI on Cells 3D: Orthogonal Views and calibrated ROI measurements
:width: 720px
```
Open a calibrated 2-channel Z-stack, inspect in **Orthogonal Views** (`Ctrl+Shift+H`), draw ROIs, and **Measure** (`Ctrl+M`) in physical units.
:::

:::{grid-item-card} Mouse Kidney — Multi-Channel Z-Stack & Channels Tool
:text-align: center
```{image} _static/demo-kidney.gif
:alt: Using TIKI on Mouse Kidney: 3-channel Z-stack with Channels Tool
:width: 720px
```
Scroll through a 3-channel Z-stack and toggle channels, LUTs, and contrast live in the **Channels Tool** (`Ctrl+Shift+Z`).
:::

::::

```{admonition} Project status
:class: caution
TIKI is an early prototype. It is useful today, but interfaces and shortcuts
may change as it is tested across more microscopes and operating systems.
```

## Start here

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} Install and launch
:link: getting-started
:link-type: doc
Create an environment, start TIKI, and open your first image.
:::

:::{grid-item-card} Everyday workflows
:link: workflows
:link-type: doc
ROIs, measurements, projections, thresholds, particles and traces.
:::

:::{grid-item-card} Multidimensional data
:link: multidimensional-data
:link-type: doc
Understand T, Z, C, calibration and how TIKI avoids unsafe guesses.
:::

:::{grid-item-card} Contribute
:link: contributing
:link-type: doc
Run the tests, build these docs, and propose changes.
:::

::::

```{toctree}
:maxdepth: 2
:caption: User guide

getting-started
concepts
workflows
multidimensional-data
roi-manager
macros
troubleshooting
```

```{toctree}
:maxdepth: 2
:caption: Project

api
contributing
changelog
license
```
