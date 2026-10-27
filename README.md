# TIKI — TIKI Is Kinda ImageJ

An ImageJ-style floating toolbar for [napari](https://napari.org). Every image
opens in its own napari window; menu commands and tools act on the window you
clicked last, just like ImageJ.

```bash
tiki                 # empty toolbar
tiki movie.tif       # open images on launch (.tif memory-mapped, .npy mmap)
```

## What works

| Menu | Command | Shortcut |
|---|---|---|
| File | Open…, Open Samples, Close, Close All, Save As TIFF, Quit | Ctrl+O / W / Shift+W / S / Q |
| Edit ▸ Selection | Select All, Select None | Ctrl+A, Ctrl+Shift+A |
| Image ▸ Adjust | Brightness/Contrast (napari layer controls), Enhance Contrast (0.35 %), Reset | Ctrl+Shift+C, Ctrl+Shift+E |
| Image | Show Info, Crop, Duplicate (crops to ROI) | Ctrl+I, Ctrl+Shift+X, Ctrl+Shift+D |
| Image ▸ Stacks | Z Project (max/avg/sum/sd/median/min, slice range), Plot Z-axis Profile (live) | —, Ctrl+Shift+Y |
| Analyze | Measure → Results table, Plot Profile (line) | Ctrl+M, Ctrl+K |
| Window | List / raise image windows, show layer list | |
| Help | Find Commands | Ctrl+L |

Tools: rectangle, oval, polygon, freehand, straight line, point, move/edit,
pan. A new ROI replaces the previous one (hold Shift to keep both). The status
bar shows `x, y, z, value` under the cursor in whichever window it is over.

Shortcuts are application-wide, so they work while an image window has focus.
napari's own bindings for the same keys (e.g. Ctrl+M = toggle menubar on
Linux) are removed from TIKI windows.

## Layout

- `ops.py` — pure numpy operations (chunked; safe on memory-mapped movies)
- `roi.py` — ROI layer, masks, current slice
- `manager.py` — window registry and active-window tracking
- `widgets.py` — Results table, live profile plot
- `app.py` — the toolbar, tools and menu commands
