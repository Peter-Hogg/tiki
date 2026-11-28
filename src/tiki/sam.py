"""Process ▸ Labels ▸ Interactive SAM… (floating promptable SAM / SAM 2 window).

Runs a persistent isolated ``uv run --no-project --with ultralytics`` sidecar
(or local ``ultralytics`` when installed) so the model stays warm in memory and
each positive/negative click or bounding box returns a live mask preview
without bloating Tiki's base installation.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
from pathlib import Path

import numpy as np
from napari.layers import Image, Labels, Points, Shapes
from napari.utils.colormaps import DirectLabelColormap
from qtpy.QtCore import Qt, QThread, Signal
from qtpy.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from . import axes as axes_mod
from . import roi as roi_mod
from . import style
from .layer_select import LayerCombo

PREVIEW_NAME = 'sam preview'
PROMPTS_NAME = 'sam prompts'
BOX_NAME = 'sam box'

SAM_MODELS: dict[str, str] = {
    'SAM 2.1 Tiny (fast)': 'sam2.1_t.pt',
    'SAM 2.1 Small': 'sam2.1_s.pt',
    'SAM 2.1 Base (accurate)': 'sam2.1_b.pt',
    'SAM 2.1 Large': 'sam2.1_l.pt',
    'SAM 1 Base': 'sam_b.pt',
}


def _to_rgb_uint8(plane: np.ndarray) -> np.ndarray:
    """Convert any 2D greyscale or RGB(A) microscopy plane to uint8 RGB for SAM."""
    arr = np.asarray(plane)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        rgb = arr[..., :3].astype(np.float32, copy=False)
        lo, hi = float(np.nanpercentile(rgb, 0.5)), float(np.nanpercentile(rgb, 99.5))
        if hi <= lo:
            lo, hi = float(np.nanmin(rgb)), float(np.nanmax(rgb))
        scale = 255.0 / (hi - lo) if hi > lo else 1.0
        out = np.clip((rgb - lo) * scale, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(out)
    g = arr.astype(np.float32, copy=False)
    finite = g[np.isfinite(g)]
    if finite.size == 0:
        return np.zeros(arr.shape[:2] + (3,), dtype=np.uint8)
    lo, hi = float(np.percentile(finite, 0.5)), float(np.percentile(finite, 99.5))
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    scale = 255.0 / (hi - lo) if hi > lo else 1.0
    u8 = np.clip((np.nan_to_num(g, nan=lo) - lo) * scale, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(np.repeat(u8[..., None], 3, axis=-1))


_SAM_WORKER_SCRIPT = textwrap.dedent(
    """\
    import json, sys
    import numpy as np

    # Keep stdout strictly reserved for JSON protocol lines.
    _real_out = sys.stdout
    sys.stdout = sys.stderr

    def _send(obj):
        _real_out.write(json.dumps(obj) + '\\n')
        _real_out.flush()

    from ultralytics import SAM
    import torch

    model = None
    model_name = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def _load(name):
        global model, model_name
        if model is None or model_name != name:
            model = SAM(name)
            model_name = name
        return {'ok': True, 'device': device, 'model': model_name}

    def _extract_best_mask(results, shape_hw):
        if not results or results[0].masks is None or results[0].masks.data is None:
            return np.zeros(shape_hw, dtype=np.uint8)
        m = results[0].masks.data.detach().cpu().numpy()
        if m.ndim == 2:
            return (m > 0.5).astype(np.uint8)
        if m.shape[0] == 0:
            return np.zeros(shape_hw, dtype=np.uint8)
        # Union across returned prompted masks (one per box or best point mask).
        return (m.max(axis=0) > 0.5).astype(np.uint8)

    def _predict_plane(img_rgb, req):
        kw = {'verbose': False}
        pts = req.get('points') or []
        lbls = req.get('labels') or []
        bboxes = req.get('bboxes') or []
        if pts:
            # Wrap in outer list so all points refine ONE object (batch size 1),
            # matching a single bounding box when both points and box are given.
            kw['points'] = [pts]
            kw['labels'] = [lbls]
        if bboxes:
            kw['bboxes'] = bboxes
        results = model.predict(img_rgb, **kw)
        return _extract_best_mask(results, img_rgb.shape[:2])

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            cmd = req.get('cmd')
            if cmd == 'quit':
                _send({'ok': True})
                break
            elif cmd == 'load':
                _send(_load(req['model']))
            elif cmd == 'predict':
                _load(req['model'])
                img = np.load(req['image_path'])
                mask = _predict_plane(img, req)
                np.save(req['out_path'], mask)
                _send({'ok': True, 'pixels': int(mask.sum())})
            elif cmd == 'auto':
                _load(req['model'])
                img = np.load(req['image_path'])
                results = model.predict(img, verbose=False)
                out = np.zeros(img.shape[:2], dtype=np.int32)
                n_obj = 0
                if results and results[0].masks is not None and results[0].masks.data is not None:
                    ms = results[0].masks.data.detach().cpu().numpy()
                    # Paint largest masks first so smaller nested objects stay visible
                    areas = [int((m > 0.5).sum()) for m in ms]
                    for idx in np.argsort(areas)[::-1]:
                        if areas[idx] <= 0:
                            continue
                        n_obj += 1
                        out[ms[idx] > 0.5] = n_obj
                np.save(req['out_path'], out)
                _send({'ok': True, 'n_objects': n_obj})
            elif cmd == 'propagate':
                _load(req['model'])
                stack = np.load(req['stack_path'])  # (N, H, W, 3) uint8
                start = int(req.get('start_frame', 0))
                n_frames, h, w = stack.shape[:3]
                out = np.zeros((n_frames, h, w), dtype=np.uint8)
                seed = _predict_plane(stack[start], req)
                out[start] = seed

                def _mask_bbox(m):
                    ys, xs = np.nonzero(m)
                    if ys.size == 0:
                        return None
                    pad = 4
                    return [
                        max(0, int(xs.min()) - pad),
                        max(0, int(ys.min()) - pad),
                        min(w - 1, int(xs.max()) + pad),
                        min(h - 1, int(ys.max()) + pad),
                    ]

                # Propagate forward & backward using the previous frame's mask bounding box
                for direction in (1, -1):
                    prev_box = _mask_bbox(seed)
                    t = start + direction
                    while 0 <= t < n_frames and prev_box is not None:
                        m = _predict_plane(stack[t], {'bboxes': [prev_box]})
                        out[t] = m
                        prev_box = _mask_bbox(m)
                        t += direction

                np.save(req['out_path'], out)
                _send({'ok': True, 'frames': int((out.reshape(n_frames, -1).sum(axis=1) > 0).sum())})
            else:
                _send({'ok': False, 'error': f'unknown command: {cmd}'})
        except Exception as exc:
            import traceback
            traceback.print_exc(file=sys.stderr)
            _send({'ok': False, 'error': str(exc)})
    """
)


def detect_sam_mode() -> tuple[str, str]:
    """Return ``('local' | 'uv' | 'missing', human_message)``."""
    if importlib.util.find_spec('ultralytics') is not None:
        return 'local', 'ultralytics installed locally'
    if shutil.which('uv'):
        return 'uv', 'Isolated uv environment (ultralytics, cached after first run)'
    return 'missing', 'Install uv (recommended) or run: pip install ultralytics'


class _SamSidecar:
    """Persistent subprocess running SAM / SAM 2 with JSON-line IPC."""

    def __init__(self, on_status=None):
        self._on_status = on_status
        self._proc: subprocess.Popen | None = None
        self._tmpdir: tempfile.TemporaryDirectory | None = None
        self._lock = threading.Lock()
        self._stderr_tail: list[str] = []

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _start(self) -> None:
        if self.alive:
            return
        mode, msg = detect_sam_mode()
        if mode == 'missing':
            raise RuntimeError(msg)
        self._tmpdir = tempfile.TemporaryDirectory(prefix='tiki_sam_')
        script = Path(self._tmpdir.name) / 'sam_worker.py'
        script.write_text(_SAM_WORKER_SCRIPT, encoding='utf-8')
        if mode == 'local':
            cmd = [sys.executable, str(script)]
        else:
            uv_bin = shutil.which('uv')
            cmd = [uv_bin, 'run', '--no-project', '--with', 'ultralytics', '--with', 'numpy', 'python', str(script)]
        if self._on_status:
            self._on_status('Starting SAM worker (first run downloads model/PyTorch)…')
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=self._drain_stderr, daemon=True).start()

    def _drain_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        for raw in proc.stderr:
            line = raw.strip()
            if not line:
                continue
            self._stderr_tail.append(line)
            if len(self._stderr_tail) > 100:
                del self._stderr_tail[:50]
            if self._on_status:
                short = line if len(line) <= 68 else line[:65] + '…'
                self._on_status(short)

    def request(self, payload: dict) -> dict:
        with self._lock:
            self._start()
            assert self._proc is not None and self._proc.stdin is not None and self._proc.stdout is not None
            self._proc.stdin.write(json.dumps(payload) + '\n')
            self._proc.stdin.flush()
            resp_line = self._proc.stdout.readline()
            if not resp_line:
                tail = '\n'.join(self._stderr_tail[-10:])
                self.close()
                raise RuntimeError(f'SAM worker exited unexpectedly:\n{tail}')
            resp = json.loads(resp_line)
            if not resp.get('ok'):
                raise RuntimeError(resp.get('error', 'SAM worker error'))
            return resp

    @property
    def workdir(self) -> Path:
        if self._tmpdir is None:
            self._start()
        assert self._tmpdir is not None
        return Path(self._tmpdir.name)

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is not None and proc.poll() is None:
            try:
                if proc.stdin:
                    proc.stdin.write(json.dumps({'cmd': 'quit'}) + '\n')
                    proc.stdin.flush()
                proc.wait(timeout=2)
            except Exception:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except Exception:
                    proc.kill()
        if self._tmpdir is not None:
            try:
                self._tmpdir.cleanup()
            except Exception:
                pass
            self._tmpdir = None


class _SamCallThread(QThread):
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self) -> None:
        try:
            self.finished_ok.emit(self._fn())
        except Exception as exc:
            self.failed.emit(str(exc))


class SamWindow(QWidget):
    """Floating ImageJ-style interactive SAM / SAM 2 prompting window."""

    _status_sig = Signal(str)

    def __init__(self, tiki):
        super().__init__(None, Qt.Window)
        style.apply(self)
        self.tiki = tiki
        self.viewer = None
        self.image: Image | None = None
        self.preview: Labels | None = None
        self.prompts_layer: Points | None = None
        self.box_layer: Shapes | None = None
        self._points: list[tuple[float, float]] = []  # (x, y) in image pixel coords
        self._point_labels: list[int] = []  # 1 = positive, 0 = negative
        self._bboxes: list[list[float]] = []  # [[x1, y1, x2, y2], ...]
        self._last_mask: np.ndarray | None = None
        self._last_mask_context: tuple | None = None  # (viewer, image, step_tuple)
        self._req_gen = 0
        self._cached_plane_key: tuple | None = None
        self._worker_thread: _SamCallThread | None = None
        self._busy = False
        self._sidecar = _SamSidecar(on_status=lambda s: self._status_sig.emit(s))
        self._status_sig.connect(self._set_status)
        self.setWindowTitle('Interactive SAM')

        # Image & model selectors
        self.image_box = LayerCombo(None, Image)
        self.image_box.layer_changed.connect(self._set_image)
        self.model_box = QComboBox()
        self.model_box.addItems(list(SAM_MODELS))

        top = QHBoxLayout()
        top.addWidget(QLabel('Image:'))
        top.addWidget(self.image_box, 1)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel('Model:'))
        model_row.addWidget(self.model_box, 1)

        # Prompt mode buttons
        self.pos_btn = QPushButton('+ Include')
        self.neg_btn = QPushButton('− Exclude')
        self.box_btn = QPushButton('▭ Draw Box')
        for b in (self.pos_btn, self.neg_btn, self.box_btn):
            b.setCheckable(True)
        self.pos_btn.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self.pos_btn)
        grp.addButton(self.neg_btn)
        grp.addButton(self.box_btn)

        self.use_roi_btn = QPushButton('From ROI')
        self.use_roi_btn.setToolTip('Use the active Tiki Rectangle ROI as a SAM bounding-box prompt')
        self.use_roi_btn.clicked.connect(self._use_box_roi)

        prompt_row = QHBoxLayout()
        prompt_row.addWidget(self.pos_btn)
        prompt_row.addWidget(self.neg_btn)
        prompt_row.addWidget(self.box_btn)
        prompt_row.addWidget(self.use_roi_btn)

        hint = QLabel('Points mode: Left-click = +, Right/Shift-click = −  |  Draw Box: click & drag')
        hint.setWordWrap(True)

        # Object actions
        self.accept_btn = QPushButton('Accept Object')
        self.accept_btn.clicked.connect(self._accept_object)
        self.undo_btn = QPushButton('Undo Point')
        self.undo_btn.clicked.connect(self._undo_point)
        self.clear_btn = QPushButton('Clear Prompts')
        self.clear_btn.clicked.connect(self._clear_prompts)

        act_row = QHBoxLayout()
        act_row.addWidget(self.accept_btn)
        act_row.addWidget(self.undo_btn)
        act_row.addWidget(self.clear_btn)

        # Whole-plane / stack actions
        self.auto_btn = QPushButton('Auto-Segment Plane')
        self.auto_btn.clicked.connect(self._auto_segment)
        self.prop_btn = QPushButton('Propagate Stack')
        self.prop_btn.setToolTip('Propagate the prompted object through the stack (T or Z)')
        self.prop_btn.clicked.connect(self._propagate_stack)

        stack_row = QHBoxLayout()
        stack_row.addWidget(self.auto_btn)
        stack_row.addWidget(self.prop_btn)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        mode, msg = detect_sam_mode()
        self.status_label.setText(msg)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addLayout(model_row)
        layout.addLayout(prompt_row)
        layout.addWidget(hint)
        layout.addLayout(act_row)
        layout.addLayout(stack_row)
        layout.addWidget(self.status_label)
        self.setMinimumWidth(360)

        self.tiki.windows.active_changed.connect(self.attach)
        self.attach(self.tiki.windows.active)

    # --------------------------------------------------------- viewer wiring

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        has_img = self.image is not None and self.viewer is not None
        for btn in (self.use_roi_btn, self.accept_btn, self.undo_btn, self.clear_btn, self.auto_btn):
            btn.setEnabled(has_img and not busy)
        self.prop_btn.setEnabled(has_img and not busy and self.image is not None and self.image.ndim > 2)

    def attach(self, viewer) -> None:
        if viewer is self.viewer:
            return
        self._detach()
        self.viewer = viewer
        self.image_box.set_viewer(viewer)
        if viewer is not None:
            viewer.mouse_drag_callbacks.append(self._on_canvas_click)
            viewer.dims.events.current_step.connect(self._on_slice_changed)
            try:
                cam = getattr(getattr(viewer, 'scene', None), 'camera', None) or viewer.camera
                cam.events.zoom.connect(self._update_prompt_size)
            except Exception:
                pass
        self._set_image(self.image_box.layer())

    def _detach(self) -> None:
        if self.viewer is not None:
            if self._on_canvas_click in self.viewer.mouse_drag_callbacks:
                self.viewer.mouse_drag_callbacks.remove(self._on_canvas_click)
            try:
                self.viewer.dims.events.current_step.disconnect(self._on_slice_changed)
            except Exception:
                pass
            try:
                cam = getattr(getattr(self.viewer, 'scene', None), 'camera', None) or self.viewer.camera
                cam.events.zoom.disconnect(self._update_prompt_size)
            except Exception:
                pass
        self._remove_overlays()
        self.viewer = None
        self.image = None

    def _set_image(self, image: Image | None) -> None:
        self._req_gen += 1
        self._remove_overlays()
        self.image = image
        self._points.clear()
        self._point_labels.clear()
        self._bboxes.clear()
        self._last_mask = None
        self._last_mask_context = None
        self._cached_plane_key = None
        self.setWindowTitle(f'Interactive SAM — {image.name}' if image is not None else 'Interactive SAM')
        self._set_busy(self._busy)

    def _on_slice_changed(self, _event=None) -> None:
        self._req_gen += 1
        self._cached_plane_key = None
        if self._last_mask is not None or self._points or self._bboxes:
            self._points.clear()
            self._point_labels.clear()
            self._bboxes.clear()
            self._last_mask = None
            self._last_mask_context = None
            self._remove_overlays()
            self._set_status('Slice changed — prompts and preview cleared')

    def _remove_overlays(self) -> None:
        if self.viewer is None:
            self.preview = None
            self.prompts_layer = None
            self.box_layer = None
            return
        for name in (PREVIEW_NAME, PROMPTS_NAME, BOX_NAME):
            if name in self.viewer.layers:
                try:
                    self.viewer.layers.remove(name)
                except Exception:
                    pass
        self.preview = None
        self.prompts_layer = None
        self.box_layer = None

    # -------------------------------------------------------- canvas prompts

    def _point_data_size(self) -> float:
        """Keep prompt dots ~10 screen pixels across regardless of zoom level."""
        if self.viewer is None:
            return 8.0
        cam = getattr(getattr(self.viewer, 'scene', None), 'camera', None) or self.viewer.camera
        zoom = float(getattr(cam, 'zoom', 1.0) or 1.0)
        sx = abs(float(self.image.scale[-1])) if self.image is not None else 1.0
        px_per_screen = 1.0 / max(zoom * max(sx, 1e-9), 1e-9)
        return float(np.clip(10.0 * px_per_screen, 1.5, 40.0))

    def _update_prompt_size(self, _event=None) -> None:
        if self.prompts_layer is not None and self.viewer is not None and self.prompts_layer in self.viewer.layers:
            if len(self.prompts_layer.data):
                self.prompts_layer.size = self._point_data_size()

    def _update_box_layer(self, box: list[float] | None) -> None:
        if self.viewer is None or self.image is None:
            return
        if box is None:
            if self.box_layer is not None and self.box_layer in self.viewer.layers:
                self.box_layer.data = []
            return
        x1, y1, x2, y2 = box
        rect = np.array([[y1, x1], [y1, x2], [y2, x2], [y2, x1]], dtype=float)
        if self.box_layer is not None and self.box_layer in self.viewer.layers:
            self.box_layer.data = [rect]
        else:
            kw = roi_mod.plane_kwargs(self.image)
            self.box_layer = Shapes(
                [rect],
                shape_type='rectangle',
                name=BOX_NAME,
                edge_color='yellow',
                face_color='transparent',
                edge_width=2,
                **kw,
            )
            self.viewer.add_layer(self.box_layer)
            self.viewer.layers.selection.active = self.image

    def _on_canvas_click(self, viewer, event):
        if not self.isVisible() or self._busy or self.image is None or viewer is not self.viewer:
            return
        if viewer.dims.ndisplay != 2:
            return
        if self.tiki.tool in ('rect', 'oval', 'polygon', 'freehand', 'line', 'polyline', 'brush', 'eraser'):
            return
        h, w = self.image.data.shape[-3:-1] if self.image.rgb else self.image.data.shape[-2:]

        def _img_xy(pos):
            c = self.image.world_to_data(pos)
            return float(c[-1]), float(c[-2])

        # Drag-to-draw bounding box mode (right-click still adds an exclude point!)
        if self.box_btn.isChecked() and event.button == 1 and 'Shift' not in event.modifiers:
            x0, y0 = _img_xy(event.position)
            if not (0 <= y0 < h and 0 <= x0 < w):
                return
            x1, y1 = x0, y0
            self._update_box_layer([x0, y0, x1, y1])
            cam = getattr(getattr(viewer, 'scene', None), 'camera', None) or viewer.camera
            was_pan = getattr(cam, 'mouse_pan', True)
            try:
                cam.mouse_pan = False
                yield
                while event.type == 'mouse_move':
                    x1, y1 = _img_xy(event.position)
                    bx1, bx2 = sorted((float(np.clip(x0, 0, w - 1)), float(np.clip(x1, 0, w - 1))))
                    by1, by2 = sorted((float(np.clip(y0, 0, h - 1)), float(np.clip(y1, 0, h - 1))))
                    self._update_box_layer([bx1, by1, bx2, by2])
                    yield
            finally:
                cam.mouse_pan = was_pan
            bx1, bx2 = sorted((float(np.clip(x0, 0, w - 1)), float(np.clip(x1, 0, w - 1))))
            by1, by2 = sorted((float(np.clip(y0, 0, h - 1)), float(np.clip(y1, 0, h - 1))))
            if (bx2 - bx1) >= 2.0 and (by2 - by1) >= 2.0:
                self._bboxes = [[bx1, by1, bx2, by2]]
                self._update_box_layer(self._bboxes[0])
                self._run_predict()
            return

        # Point mode: wait for release so panning drags don't drop stray points
        start_pos = tuple(event.pos) if hasattr(event, 'pos') else None
        x, y = _img_xy(event.position)
        if not (0 <= y < h and 0 <= x < w):
            return
        is_neg = (event.button == 2) or ('Shift' in event.modifiers) or self.neg_btn.isChecked()
        moved = False
        yield
        while event.type == 'mouse_move':
            if start_pos is not None and hasattr(event, 'pos'):
                dx = event.pos[0] - start_pos[0]
                dy = event.pos[1] - start_pos[1]
                if dx * dx + dy * dy > 16:
                    moved = True
            yield
        if moved:
            return
        self._points.append((x, y))
        self._point_labels.append(0 if is_neg else 1)
        self._update_prompts_layer()
        self._run_predict()

    def _update_prompts_layer(self) -> None:
        if self.viewer is None or self.image is None:
            return
        pts = np.array([[y, x] for x, y in self._points], dtype=float) if self._points else np.empty((0, 2))
        colors = ['lime' if lbl == 1 else 'red' for lbl in self._point_labels]
        sz = self._point_data_size()
        if self.prompts_layer is not None and self.prompts_layer in self.viewer.layers:
            self.prompts_layer.data = pts
            self.prompts_layer.size = sz
            if len(colors):
                self.prompts_layer.face_color = colors
        else:
            kw = roi_mod.plane_kwargs(self.image)
            self.prompts_layer = Points(
                pts,
                name=PROMPTS_NAME,
                size=sz,
                face_color=colors if colors else 'lime',
                border_color='white',
                **kw,
            )
            self.viewer.add_layer(self.prompts_layer)
            self.viewer.layers.selection.active = self.image

    def _show_preview_mask(self, mask: np.ndarray, context: tuple | None = None) -> None:
        if self.viewer is None or self.image is None:
            return
        self._last_mask = mask.astype(np.uint8, copy=False)
        self._last_mask_context = context if context is not None else (
            self.viewer,
            self.image,
            tuple(self.viewer.dims.current_step),
        )
        cmap = DirectLabelColormap(color_dict={None: 'transparent', 0: 'transparent', 1: 'cyan'})
        if self.preview is not None and self.preview in self.viewer.layers:
            self.preview.data = self._last_mask
            self.preview.colormap = cmap
        else:
            kw = axes_mod.geometry(self.image, ndim=2)
            self.preview = Labels(self._last_mask, name=PREVIEW_NAME, opacity=0.45, colormap=cmap, **kw)
            self.viewer.add_layer(self.preview)
            self.viewer.layers.selection.active = self.image

    # --------------------------------------------------------- sidecar calls

    def _ensure_plane_file(self) -> Path:
        assert self.viewer is not None and self.image is not None
        step = tuple(self.viewer.dims.current_step)
        key = (id(self.image), step)
        img_path = self._sidecar.workdir / 'plane.npy'
        if self._cached_plane_key != key or not img_path.exists():
            plane = roi_mod.current_plane(self.viewer, self.image)
            np.save(img_path, _to_rgb_uint8(plane))
            self._cached_plane_key = key
        return img_path

    def _run_async(self, fn, on_ok) -> None:
        if self._busy:
            return
        self._set_busy(True)
        t = _SamCallThread(fn, parent=self)
        self._worker_thread = t

        def _ok(res):
            self._set_busy(False)
            on_ok(res)

        def _err(msg):
            self._set_busy(False)
            self._set_status(f'Error: {msg}')
            self.tiki.report('Interactive SAM', msg)

        t.finished_ok.connect(_ok)
        t.failed.connect(_err)
        t.finished.connect(t.deleteLater)
        t.start()

    def _run_predict(self) -> None:
        if self.viewer is None or self.image is None or (not self._points and not self._bboxes):
            return
        req_viewer = self.viewer
        req_image = self.image
        req_step = tuple(req_viewer.dims.current_step)
        self._req_gen += 1
        gen = self._req_gen
        model_file = SAM_MODELS[self.model_box.currentText()]
        pts = [[x, y] for x, y in self._points]
        lbls = list(self._point_labels)
        bboxes = list(self._bboxes)
        self._set_status(f'Predicting ({self.model_box.currentText()})…')

        def _work():
            img_path = self._ensure_plane_file()
            out_path = self._sidecar.workdir / 'mask.npy'
            resp = self._sidecar.request({
                'cmd': 'predict',
                'model': model_file,
                'image_path': str(img_path),
                'out_path': str(out_path),
                'points': pts,
                'labels': lbls,
                'bboxes': bboxes,
            })
            return np.load(out_path), resp

        def _done(res):
            if (
                gen != self._req_gen
                or self.viewer is not req_viewer
                or self.image is not req_image
                or req_image not in req_viewer.layers
                or tuple(req_viewer.dims.current_step) != req_step
            ):
                return
            mask, resp = res
            self._show_preview_mask(mask, context=(req_viewer, req_image, req_step))
            self._set_status(f'Preview ready: {resp.get("pixels", 0):,} px — click Accept Object or refine')

        self._run_async(_work, _done)

    def _use_box_roi(self) -> None:
        if self.viewer is None or self.image is None:
            return
        cur = roi_mod.current_shape(roi_mod.roi_layer(self.viewer, create=False))
        if cur is None:
            self._set_status('Draw a Rectangle ROI on the image first, then click Use Box ROI.')
            return
        verts, _kind = cur
        v2 = np.asarray(verts, dtype=float)[:, -2:]
        h, w = self.image.data.shape[-3:-1] if self.image.rgb else self.image.data.shape[-2:]
        y1 = float(np.clip(v2[:, 0].min(), 0, h - 1))
        x1 = float(np.clip(v2[:, 1].min(), 0, w - 1))
        y2 = float(np.clip(v2[:, 0].max(), 0, h - 1))
        x2 = float(np.clip(v2[:, 1].max(), 0, w - 1))
        self._bboxes = [[x1, y1, x2, y2]]
        self._run_predict()

    def _undo_point(self) -> None:
        if self._points:
            self._points.pop()
            self._point_labels.pop()
            self._update_prompts_layer()
            if self._points or self._bboxes:
                self._run_predict()
            else:
                self._clear_prompts()

    def _clear_prompts(self) -> None:
        self._req_gen += 1
        self._points.clear()
        self._point_labels.clear()
        self._bboxes.clear()
        self._last_mask = None
        self._last_mask_context = None
        self._remove_overlays()
        self._set_status('Prompts cleared — click image to segment next object')

    def _target_labels_layer(self, ndim: int = 2) -> Labels:
        assert self.viewer is not None and self.image is not None
        name = f'{self.image.name} sam'
        if name in self.viewer.layers and isinstance(self.viewer.layers[name], Labels):
            ly = self.viewer.layers[name]
            if ly.ndim == ndim:
                return ly
        shape = self.image.data.shape[: self.image.ndim] if ndim == self.image.ndim else (
            self.image.data.shape[-3:-1] if self.image.rgb else self.image.data.shape[-2:]
        )
        geom = axes_mod.geometry(self.image) if ndim == self.image.ndim else axes_mod.geometry(self.image, ndim=2)
        ly = Labels(np.zeros(shape, dtype=np.int32), name=name, opacity=0.6, **geom)
        self.viewer.add_layer(ly)
        self.tiki.push_undo_layer('Interactive SAM', self.viewer, ly)
        return ly

    def _accept_object(self) -> None:
        if self.viewer is None or self.image is None or self._last_mask is None or not self._last_mask.any():
            self._set_status('No preview mask to accept yet.')
            return
        cur_ctx = (self.viewer, self.image, tuple(self.viewer.dims.current_step))
        if self._last_mask_context != cur_ctx:
            self._clear_prompts()
            self._set_status('Preview was from a different image or slice and was cleared.')
            return
        ly = self._target_labels_layer(ndim=self.image.ndim)
        arr = np.asarray(ly.data)
        next_id = int(arr.max()) + 1
        if ly.ndim == 2:
            arr[self._last_mask > 0] = next_id
        else:
            idx = roi_mod.current_index(self.viewer, self.image)[:-2]
            arr[idx][self._last_mask > 0] = next_id
        ly.data = arr
        ly.refresh()
        self._clear_prompts()
        self._set_status(f'Committed object #{next_id} → "{ly.name}"')

    def _auto_segment(self) -> None:
        if self.viewer is None or self.image is None:
            return
        req_viewer = self.viewer
        req_image = self.image
        req_step = tuple(req_viewer.dims.current_step)
        self._req_gen += 1
        gen = self._req_gen
        model_file = SAM_MODELS[self.model_box.currentText()]
        self._set_status(f'Auto-segmenting plane ({self.model_box.currentText()})…')

        def _work():
            img_path = self._ensure_plane_file()
            out_path = self._sidecar.workdir / 'auto_labels.npy'
            resp = self._sidecar.request({
                'cmd': 'auto',
                'model': model_file,
                'image_path': str(img_path),
                'out_path': str(out_path),
            })
            return np.load(out_path), resp

        def _done(res):
            labels, resp = res
            if (
                gen != self._req_gen
                or self.viewer is not req_viewer
                or self.image is not req_image
                or req_image not in req_viewer.layers
                or tuple(req_viewer.dims.current_step) != req_step
            ):
                return
            geom = axes_mod.geometry(req_image, ndim=2)
            ly = Labels(labels, name=f'{req_image.name} sam auto', opacity=0.6, **geom)
            req_viewer.add_layer(ly)
            self.tiki.push_undo_layer('SAM Auto-Segment', req_viewer, ly)
            self._set_status(f'Auto-segmented {resp.get("n_objects", 0)} objects → "{ly.name}"')

        self._run_async(_work, _done)

    def _propagate_stack(self) -> None:
        if self.viewer is None or self.image is None or self.image.ndim <= 2:
            return
        if not self._points and not self._bboxes:
            self._set_status('Click at least one point or use a Box ROI on the current slice first.')
            return
        req_viewer = self.viewer
        req_image = self.image
        self._req_gen += 1
        gen = self._req_gen
        model_file = SAM_MODELS[self.model_box.currentText()]
        axis = self.tiki._stack_axis(req_viewer, req_image)
        index = roi_mod.current_index(req_viewer, req_image)
        start_frame = int(index[axis])
        n_frames = req_image.data.shape[axis]
        pts = [[x, y] for x, y in self._points]
        lbls = list(self._point_labels)
        bboxes = list(self._bboxes)
        data = req_image.data
        lead = req_image.ndim - 2
        self._set_status(f'Propagating across {n_frames} slices ({self.model_box.currentText()})…')

        def _work():
            frames = []
            for t in range(n_frames):
                sl = tuple(t if d == axis else int(index[d]) for d in range(lead))
                frames.append(_to_rgb_uint8(np.asarray(data[sl])))
            stack_path = self._sidecar.workdir / 'stack.npy'
            out_path = self._sidecar.workdir / 'stack_masks.npy'
            np.save(stack_path, np.stack(frames, axis=0))
            resp = self._sidecar.request({
                'cmd': 'propagate',
                'model': model_file,
                'stack_path': str(stack_path),
                'out_path': str(out_path),
                'start_frame': start_frame,
                'points': pts,
                'labels': lbls,
                'bboxes': bboxes,
            })
            return np.load(out_path), resp

        def _done(res):
            masks, resp = res
            if (
                gen != self._req_gen
                or self.viewer is not req_viewer
                or self.image is not req_image
                or req_image not in req_viewer.layers
            ):
                return
            ly = self._target_labels_layer(ndim=req_image.ndim)
            arr = np.asarray(ly.data)
            next_id = int(arr.max()) + 1
            for t in range(n_frames):
                sl = tuple(t if d == axis else int(index[d]) for d in range(lead))
                arr[sl][masks[t] > 0] = next_id
            ly.data = arr
            ly.refresh()
            self._clear_prompts()
            self._set_status(
                f'Propagated object #{next_id} across {resp.get("frames", 0)}/{n_frames} slices → "{ly.name}"'
            )

        self._run_async(_work, _done)

    def closeEvent(self, event) -> None:
        self._detach()
        self._sidecar.close()
        super().closeEvent(event)
