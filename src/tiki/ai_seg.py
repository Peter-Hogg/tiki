"""Process ▸ Labels ▸ AI Segmentation… (zero-bloat Cellpose / StarDist bridge).

Runs installed models directly in-process when present in Tiki's Python
environment, or transparently offloads to an isolated ``uv run --with ...``
sidecar so Tiki's own install stays tiny and free of PyTorch/TensorFlow
dependency conflicts.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import textwrap
import time
from pathlib import Path

import numpy as np
from napari.layers import Image, Labels
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QMessageBox,
)

from . import axes as axes_mod
from . import nd, style
from . import roi as roi_mod
from .layer_select import LayerCombo

BACKENDS: dict[str, dict] = {
    'Cellpose': {
        'package': 'cellpose',
        'uv_spec': 'cellpose',
        'models': ['cyto3', 'nuclei', 'cyto2', 'cpsam'],
        'supports_3d': True,
        'description': 'Cellpose generalist cell & nucleus segmentation (PyTorch)',
    },
    'StarDist': {
        'package': 'stardist',
        'uv_spec': 'stardist',
        'models': ['2D_versatile_fluo', '2D_versatile_he', '2D_paper_dsb2018'],
        'supports_3d': False,
        'description': 'StarDist star-convex nucleus segmentation (TensorFlow)',
    },
}


def detect_backend(name: str) -> tuple[str, str]:
    """Return ``(mode, detail)`` where ``mode`` is ``'local'``, ``'uv'``, or ``'missing'``."""
    spec = BACKENDS[name]
    pkg = spec['package']
    if importlib.util.find_spec(pkg) is not None:
        return 'local', f'{pkg} is installed in this Python environment (runs directly)'
    uv_bin = shutil.which('uv')
    if uv_bin:
        return 'uv', f'Runs in an isolated uv environment via {pkg} (downloads once on first run, keeps Tiki light)'
    return 'missing', f'Install uv (recommended) or run: pip install {pkg}'


def _run_cellpose_local(arr: np.ndarray, params: dict) -> np.ndarray:
    """Run Cellpose directly in the current Python environment."""
    from cellpose import models

    model_name = str(params.get('model') or 'cyto3')
    diameter = float(params.get('diameter') or 0.0)
    diam_arg = diameter if diameter > 0 else None
    do_3d = bool(params.get('do_3d', False)) and arr.ndim == 3
    use_gpu = bool(params.get('use_gpu', True))
    anisotropy = params.get('anisotropy')

    # Support both Cellpose v4 (CellposeModel) and v3 (Cellpose / CellposeModel)
    try:
        model = models.CellposeModel(gpu=use_gpu, pretrained_model=model_name)
    except Exception:
        model = models.CellposeModel(gpu=use_gpu, model_type=model_name)

    eval_kw: dict = {'diameter': diam_arg, 'do_3D': do_3d}
    if do_3d and anisotropy and float(anisotropy) > 0:
        eval_kw['anisotropy'] = float(anisotropy)
    try:
        res = model.eval(arr, **eval_kw)
    except TypeError:
        # Cellpose v3 sometimes requires channels=[0, 0]
        eval_kw['channels'] = [0, 0]
        res = model.eval(arr, **eval_kw)
    masks = res[0] if isinstance(res, tuple) else res
    return np.asarray(masks, dtype=np.int32)


def _run_stardist_local(arr: np.ndarray, params: dict) -> np.ndarray:
    """Run StarDist 2D directly in the current Python environment."""
    from csbdeep.utils import normalize
    from stardist.models import StarDist2D

    model_name = str(params.get('model') or '2D_versatile_fluo')
    prob_thresh = params.get('prob_thresh')
    nms_thresh = params.get('nms_thresh')
    model = StarDist2D.from_pretrained(model_name)
    norm = normalize(np.asarray(arr, dtype=np.float32), 1.0, 99.8)
    kw: dict = {}
    if prob_thresh is not None and float(prob_thresh) > 0:
        kw['prob_thresh'] = float(prob_thresh)
    if nms_thresh is not None and float(nms_thresh) > 0:
        kw['nms_thresh'] = float(nms_thresh)
    labels, _ = model.predict_instances(norm, **kw)
    return np.asarray(labels, dtype=np.int32)


_UV_RUNNER_SCRIPT = textwrap.dedent(
    """\
    import json, sys
    import numpy as np

    backend, in_path, out_path, params_path = sys.argv[1:5]
    arr = np.load(in_path)
    with open(params_path, 'r', encoding='utf-8') as f:
        params = json.load(f)

    if backend == 'Cellpose':
        from cellpose import models
        model_name = str(params.get('model') or 'cyto3')
        diameter = float(params.get('diameter') or 0.0)
        diam_arg = diameter if diameter > 0 else None
        do_3d = bool(params.get('do_3d', False)) and arr.ndim == 3
        use_gpu = bool(params.get('use_gpu', True))
        anisotropy = params.get('anisotropy')
        try:
            model = models.CellposeModel(gpu=use_gpu, pretrained_model=model_name)
        except Exception:
            model = models.CellposeModel(gpu=use_gpu, model_type=model_name)
        eval_kw = {'diameter': diam_arg, 'do_3D': do_3d}
        if do_3d and anisotropy and float(anisotropy) > 0:
            eval_kw['anisotropy'] = float(anisotropy)
        try:
            res = model.eval(arr, **eval_kw)
        except TypeError:
            eval_kw['channels'] = [0, 0]
            res = model.eval(arr, **eval_kw)
        masks = res[0] if isinstance(res, tuple) else res
        out = np.asarray(masks, dtype=np.int32)
    elif backend == 'StarDist':
        from csbdeep.utils import normalize
        from stardist.models import StarDist2D
        model_name = str(params.get('model') or '2D_versatile_fluo')
        model = StarDist2D.from_pretrained(model_name)
        norm = normalize(np.asarray(arr, dtype=np.float32), 1.0, 99.8)
        out, _ = model.predict_instances(norm)
        out = np.asarray(out, dtype=np.int32)
    else:
        raise ValueError(f'Unknown backend: {backend}')

    np.save(out_path, out)
    """
)


def _run_via_uv(backend: str, arr: np.ndarray, params: dict, progress=None) -> np.ndarray | None:
    """Execute segmentation in an isolated ``uv run --with <pkg>`` subprocess."""
    import threading

    uv_bin = shutil.which('uv')
    if not uv_bin:
        raise RuntimeError(f'uv is not on PATH and {BACKENDS[backend]["package"]} is not installed.')

    pkg_spec = BACKENDS[backend]['uv_spec']
    with tempfile.TemporaryDirectory(prefix='tiki_ai_') as td:
        tmp = Path(td)
        in_path = tmp / 'input.npy'
        out_path = tmp / 'output.npy'
        params_path = tmp / 'params.json'
        script_path = tmp / 'runner.py'
        np.save(in_path, np.asarray(arr))
        params_path.write_text(json.dumps(params), encoding='utf-8')
        script_path.write_text(_UV_RUNNER_SCRIPT, encoding='utf-8')

        cmd = [
            uv_bin,
            'run',
            '--no-project',
            '--with',
            pkg_spec,
            '--with',
            'numpy',
            'python',
            str(script_path),
            backend,
            str(in_path),
            str(out_path),
            str(params_path),
        ]
        set_label = getattr(progress, 'set_label', None)
        if set_label is not None:
            set_label(f'Starting {backend} in isolated uv environment…\n(First run downloads PyTorch/weights)')

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        stderr_lines: list[str] = []

        def _drain_stderr() -> None:
            assert proc.stderr is not None
            for raw_line in proc.stderr:
                line = raw_line.strip()
                if not line:
                    continue
                stderr_lines.append(line)
                if len(stderr_lines) > 200:
                    del stderr_lines[:100]
                if set_label is not None:
                    short = line if len(line) <= 72 else line[:69] + '…'
                    set_label(f'{backend}: {short}')

        drainer = threading.Thread(target=_drain_stderr, daemon=True)
        drainer.start()

        while proc.poll() is None:
            if progress is not None and progress(0, 0) is False:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                drainer.join(timeout=1)
                return None
            time.sleep(0.1)

        drainer.join(timeout=2)
        if proc.returncode != 0:
            tail = '\n'.join(stderr_lines[-12:])
            raise RuntimeError(f'{backend} (uv) failed (exit {proc.returncode}):\n{tail}')
        if not out_path.exists():
            raise RuntimeError(f'{backend} finished without writing output labels.')
        return np.asarray(np.load(out_path), dtype=np.int32)


def run_segmentation(
    backend: str,
    data: np.ndarray,
    params: dict,
    *,
    blocks: list[tuple] | None = None,
    progress=None,
) -> np.ndarray | None:
    """Run AI segmentation on a single plane/volume or block-by-block over a stack."""
    mode, detail = detect_backend(backend)
    if mode == 'missing':
        raise RuntimeError(detail)

    def _run_one(arr: np.ndarray) -> np.ndarray | None:
        if mode == 'local':
            return _run_cellpose_local(arr, params) if backend == 'Cellpose' else _run_stardist_local(arr, params)
        return _run_via_uv(backend, arr, params, progress=progress)

    if blocks is None:
        res = _run_one(np.asarray(data))
        if progress is not None:
            progress(1, 1)
        return res

    out = np.zeros(data.shape, dtype=np.int32)
    next_id = 0
    for n, blk in enumerate(blocks, 1):
        res = _run_one(np.asarray(data[blk]))
        if res is None:
            return None
        if res.max():
            res[res > 0] += next_id
            next_id = int(res.max())
        out[blk] = res
        if progress is not None and progress(n, len(blocks)) is False:
            return None
    return out


def ai_segmentation_dialog(tiki) -> Labels | None:
    """``Process ▸ Labels ▸ AI Segmentation…`` dialog."""
    viewer, image = tiki.require_image()
    if image.rgb and image.ndim > 2:
        from .app import TikiError

        raise TikiError('AI segmentation on RGB stacks is not supported yet; split channels or use a 2D image')

    dlg = QDialog(tiki)
    style.apply(dlg)
    dlg.setWindowTitle('AI Segmentation')
    form = QFormLayout(dlg)

    img_box = LayerCombo(viewer, Image, current=image)
    backend_box = QComboBox()
    backend_box.addItems(list(BACKENDS))

    model_box = QComboBox()
    diameter = QDoubleSpinBox(minimum=0.0, maximum=1000.0, singleStep=5.0, decimals=1, value=0.0)
    diameter.setSpecialValueText('Auto (0)')
    diameter.setSuffix(' px')

    scope_box = QComboBox()
    gpu_box = QCheckBox('Use GPU when available', checked=True)
    status_label = QLabel()
    status_label.setWordWrap(True)

    form.addRow('Image:', img_box)
    form.addRow('Backend:', backend_box)
    form.addRow('Model:', model_box)
    form.addRow('Cell diameter:', diameter)
    form.addRow('Apply to:', scope_box)
    form.addRow(gpu_box)
    form.addRow('Environment:', status_label)

    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    ok_btn = buttons.button(QDialogButtonBox.Ok)
    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)
    form.addRow(buttons)

    def _refresh() -> None:
        b = backend_box.currentText()
        spec = BACKENDS[b]
        model_box.clear()
        model_box.addItems(spec['models'])
        diameter.setEnabled(b == 'Cellpose')

        cur_img = img_box.layer() or image
        info = axes_mod.get(cur_img)
        scope_box.clear()
        scope_box.addItem('Current plane (2D)', nd.PLANE)
        if cur_img.ndim > 2:
            if spec['supports_3d'] and info.index('Z') is not None:
                scope_box.addItem('Current Z volume (3D)', nd.VOLUME)
            scope_box.addItem('Whole stack (plane-by-plane)', nd.WHOLE)

        mode, detail = detect_backend(b)
        status_label.setText(detail)
        ok_btn.setEnabled(mode != 'missing')

    backend_box.currentTextChanged.connect(_refresh)
    img_box.layer_changed.connect(lambda _ly: _refresh())
    _refresh()

    if dlg.exec_() != QDialog.Accepted:
        return None

    target = img_box.layer()
    if target is None:
        return None
    backend = backend_box.currentText()
    mode, _ = detect_backend(backend)
    if mode == 'missing':
        QMessageBox.information(tiki, 'AI Segmentation', detect_backend(backend)[1])
        return None

    info = axes_mod.get(target)
    shape = tuple(target.data.shape[: target.ndim])
    index = roi_mod.current_index(viewer, target)
    scope = scope_box.currentData() or nd.PLANE

    params: dict = {
        'model': model_box.currentText(),
        'diameter': diameter.value(),
        'use_gpu': gpu_box.isChecked(),
        'do_3d': scope == nd.VOLUME,
    }
    if scope == nd.VOLUME:
        zr = axes_mod.z_ratio(info)
        if zr and zr > 0:
            params['anisotropy'] = float(1.0 / zr)

    if scope == nd.PLANE or target.ndim == 2:
        src_arr = roi_mod.current_plane(viewer, target)
        blocks = None
        total = 0
        geom = axes_mod.geometry(target, ndim=2)
        suffix = f' {backend.lower()}'
    elif scope == nd.VOLUME:
        (vol_blk,) = nd.blocks(info, shape, index, nd.VOLUME, False)
        src_arr = np.asarray(target.data[vol_blk])
        blocks = None
        total = 0
        geom = axes_mod.geometry(target, ndim=3)
        suffix = f' {backend.lower()} 3D'
    else:
        src_arr = target.data
        blocks = nd.blocks(info, shape, index, nd.WHOLE, False)
        total = len(blocks)
        geom = axes_mod.geometry(target)
        suffix = f' {backend.lower()}'

    title = f'{backend} ({params["model"]})'
    out, cancelled = tiki.run_task(
        title,
        total,
        lambda tick: run_segmentation(backend, src_arr, params, blocks=blocks, progress=tick),
    )
    if cancelled or out is None:
        tiki.status(f'{title} cancelled')
        return None

    out_info = info if out.ndim == target.ndim else axes_mod.fit(
        axes_mod.AxesInfo(info.axes[-out.ndim :], info.source, info.confident), out.shape
    )
    layer = Labels(
        out,
        name=f'{target.name}{suffix}',
        opacity=0.6,
        metadata={'axes': out_info.to_dict()} if out.ndim > 2 else {},
        **geom,
    )
    viewer.add_layer(layer)
    viewer.layers.selection.active = layer
    tiki.push_undo_layer(title, viewer, layer)
    roi = roi_mod.roi_layer(viewer, create=False)
    if roi is not None:
        viewer.layers.move(viewer.layers.index(roi), len(viewer.layers))
    n_objs = int(len(np.unique(out)) - (1 if (out == 0).any() else 0))
    tiki.status(f'{title}: {n_objs} objects → "{layer.name}"')
    return layer
