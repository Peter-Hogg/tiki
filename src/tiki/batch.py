"""Process > Batch: batch format conversion and batch Python macro execution."""

from __future__ import annotations

import csv
import fnmatch
import re
from pathlib import Path

import numpy as np
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
)

from . import axes as axes_mod
from . import io, macros, ops, process, style


CONFLICT_POLICIES = ('rename', 'skip', 'overwrite', 'error')
SCENE_POLICIES = ('all', 'first', 'active')


def _natkey(path: Path) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', path.name)]


def _stem_collisions(files: list[Path], in_root: Path, out_root: Path) -> set[tuple[Path, str]]:
    """Return ``(resolved_dst_dir, lowercase_stem)`` keys shared by more than one input file."""
    counts: dict[tuple[Path, str], int] = {}
    for src in files:
        rel_parent = src.relative_to(in_root).parent
        key = ((out_root / rel_parent).resolve(strict=False), src.stem.lower())
        counts[key] = counts.get(key, 0) + 1
    return {k for k, n in counts.items() if n > 1}


def _collision_stem(src: Path) -> str:
    ext_tag = re.sub(r'[^A-Za-z0-9]+', '_', src.suffix.lstrip('.').lower()).strip('_')
    return f'{src.stem}_{ext_tag}' if ext_tag else src.stem


def _unique_dest(
    dst: Path,
    source_resolved: set[Path],
    used_resolved: set[Path],
    on_conflict: str,
) -> Path | None:
    """Resolve a safe output path avoiding source overwrites and intra-batch collisions."""
    res = dst.resolve(strict=False)
    hits_source = res in source_resolved
    hits_used = res in used_resolved
    exists_on_disk = dst.exists()

    if not hits_source and not hits_used and not (exists_on_disk and on_conflict != 'overwrite'):
        used_resolved.add(res)
        return dst

    if hits_source:
        if on_conflict in ('error', 'overwrite'):
            raise ValueError(f'Refusing to overwrite input source file: {dst}')
        if on_conflict == 'skip':
            return None
    elif hits_used:
        if on_conflict == 'error':
            raise ValueError(f'Output destination collision in batch: {dst}')
        if on_conflict == 'skip':
            return None
    else:
        if on_conflict == 'error':
            raise ValueError(f'Output file already exists: {dst}')
        if on_conflict == 'skip':
            return None
        if on_conflict == 'overwrite':
            used_resolved.add(res)
            return dst

    tag = '_out' if hits_source else ''
    n = 1
    while True:
        if tag:
            suffix = tag if n == 1 else f'{tag}_{n}'
        else:
            suffix = f'_{n + 1}'
        cand = dst.with_name(f'{dst.stem}{suffix}{dst.suffix}')
        cand_res = cand.resolve(strict=False)
        if cand_res not in source_resolved and cand_res not in used_resolved and not cand.exists():
            used_resolved.add(cand_res)
            return cand
        n += 1


def plan_batch_convert(
    files: list[Path],
    in_root: str | Path,
    out_root: str | Path,
    ext: str,
    on_conflict: str = 'rename',
) -> tuple[list[tuple[Path, Path, str, Path | None]], set[Path], set[Path]]:
    """Pre-compute output paths for ``files`` and enforce collision/source safety before writing."""
    if on_conflict not in CONFLICT_POLICIES:
        raise ValueError(f'unknown on_conflict policy {on_conflict!r}; expected one of {CONFLICT_POLICIES}')
    in_p = Path(in_root)
    out_p = Path(out_root)
    source_resolved = {p.resolve(strict=False) for p in files}
    used_resolved: set[Path] = set()
    collisions = _stem_collisions(files, in_p, out_p)

    plan: list[tuple[Path, Path, str, Path | None]] = []
    for src in files:
        rel_parent = src.relative_to(in_p).parent
        dst_dir = out_p / rel_parent
        key = (dst_dir.resolve(strict=False), src.stem.lower())
        if key in collisions:
            if on_conflict == 'error':
                raise ValueError(f'Filename collision in batch output for stem {src.stem!r} ({src.name})')
            base_stem = _collision_stem(src)
        else:
            base_stem = src.stem
        planned = _unique_dest(dst_dir / f'{base_stem}{ext}', source_resolved, used_resolved, on_conflict)
        plan.append((src, dst_dir, base_stem, planned))
    return plan, source_resolved, used_resolved


def find_batch_files(folder: str | Path, pattern: str = '*', recursive: bool = False) -> list[Path]:
    """Return naturally sorted readable image files inside ``folder`` matching ``pattern``."""
    root = Path(folder)
    if not root.is_dir():
        return []
    pat = (pattern or '*').strip() or '*'
    iterator = root.rglob('*') if recursive else root.iterdir()
    files = [
        p
        for p in iterator
        if p.is_file()
        and p.suffix.lower() in io.IMAGE_SUFFIXES
        and fnmatch.fnmatch(p.name, pat)
    ]
    return sorted(files, key=_natkey)


def _to_uint8_display(plane: np.ndarray, rgb: bool) -> np.ndarray:
    arr = np.asarray(plane)
    if rgb and arr.dtype == np.uint8:
        return arr[..., :3]
    if arr.dtype == np.uint8:
        return arr
    if rgb:
        arr = np.asarray(arr, dtype=np.float64)
        lo, hi = ops.saturated_limits(arr)
        scaled = np.clip((arr - lo) / max(hi - lo, 1e-12) * 255.0, 0, 255).astype(np.uint8)
        return scaled[..., :3]
    lo, hi = ops.saturated_limits(arr)
    scaled = np.clip((np.asarray(arr, dtype=np.float64) - lo) / max(hi - lo, 1e-12) * 255.0, 0, 255)
    return scaled.astype(np.uint8)


def _flatten_to_2d(data: np.ndarray, rgb: bool, stack_mode: str = 'max') -> np.ndarray:
    arr = np.asarray(data)
    target_ndim = 3 if rgb else 2
    while arr.ndim > target_ndim:
        if stack_mode == 'first':
            arr = arr[0]
        else:
            arr = ops.z_project(arr, 'max', axis=0) if not rgb else arr.max(axis=0)
    return arr


def batch_convert(
    input_dir: str | Path,
    output_dir: str | Path,
    fmt: str = 'tiff',
    pattern: str = '*',
    recursive: bool = False,
    stack_mode: str = 'max',
    on_conflict: str = 'rename',
    progress=None,
) -> tuple[int, list[tuple[Path, str]]]:
    """Convert matching images in ``input_dir`` into ``output_dir`` as tiff/png/jpeg/npy."""
    in_root = Path(input_dir)
    out_root = Path(output_dir)
    files = find_batch_files(in_root, pattern=pattern, recursive=recursive)
    fmt_l = fmt.lower()
    ext = {'tiff': '.tif', 'png': '.png', 'jpeg': '.jpg', 'jpg': '.jpg', 'npy': '.npy'}[fmt_l]
    plan, source_resolved, used_resolved = plan_batch_convert(
        files, in_root, out_root, ext, on_conflict=on_conflict
    )
    out_root.mkdir(parents=True, exist_ok=True)

    ok_count = 0
    errors: list[tuple[Path, str]] = []
    for n, (src, dst_dir, base_stem, planned_dst) in enumerate(plan, 1):
        try:
            scenes = io.read_multi(src)
            if len(scenes) == 1:
                if planned_dst is None:
                    if progress is not None and progress(n, len(plan)) is False:
                        break
                    continue
                targets = [planned_dst]
            else:
                if planned_dst is not None:
                    used_resolved.discard(planned_dst.resolve(strict=False))
                targets = []
                for idx in range(len(scenes)):
                    t = _unique_dest(
                        dst_dir / f'{base_stem}_s{idx + 1}{ext}',
                        source_resolved,
                        used_resolved,
                        on_conflict,
                    )
                    targets.append(t)

            wrote_any = False
            for (data, meta), dst in zip(scenes, targets):
                if dst is None:
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                info = meta['axes']
                rgb = data.ndim == len(info.axes) + 1 and data.shape[-1] in (3, 4)
                if fmt_l == 'tiff':
                    io.save_tiff(
                        dst,
                        data,
                        axes=info,
                        luts=meta.get('imagej_luts'),
                        channel_names=meta.get('channel_names'),
                    )
                elif fmt_l == 'npy':
                    np.save(dst, np.asarray(data))
                else:
                    from PIL import Image as PilImage

                    c_idx = info.index('C') if not rgb else None
                    if c_idx is not None and 1 < data.shape[c_idx] <= 8:
                        from . import channels, contrast

                        arr_c = np.moveaxis(np.asarray(data), c_idx, 0)
                        while arr_c.ndim > 3:
                            arr_c = arr_c[:, 0] if stack_mode == 'first' else arr_c.max(axis=1)
                        n_c = arr_c.shape[0]
                        cmaps = channels._imagej_colormaps(meta.get('imagej_luts'), n_c) or [
                            contrast.LUTS[name] for name in channels.DEFAULT_COLORS.get(n_c, ['gray'] * n_c)
                        ]
                        specs = [(arr_c[k], cmaps[k], ops.saturated_limits(arr_c[k])) for k in range(n_c)]
                        u8 = ops.composite_rgb(specs)
                    else:
                        plane = _flatten_to_2d(data, rgb=rgb, stack_mode=stack_mode)
                        u8 = _to_uint8_display(plane, rgb=rgb)
                    PilImage.fromarray(u8).save(dst)
                wrote_any = True
            if wrote_any:
                ok_count += 1
        except Exception as exc:
            errors.append((src, str(exc)))
        if progress is not None and progress(n, len(plan)) is False:
            break
    return ok_count, errors


def batch_macro(
    tiki,
    input_dir: str | Path,
    code: str,
    output_dir: str | Path | None = None,
    pattern: str = '*',
    recursive: bool = False,
    auto_save: bool = False,
    save_results_csv: bool = True,
    close_after: bool = True,
    scenes: str = 'all',
    on_conflict: str = 'rename',
    progress=None,
) -> tuple[int, list[tuple[Path, str]]]:
    """Run Python macro ``code`` across matching files and scenes in ``input_dir``."""
    if scenes not in SCENE_POLICIES:
        raise ValueError(f'unknown scenes policy {scenes!r}; expected one of {SCENE_POLICIES}')
    if on_conflict not in CONFLICT_POLICIES:
        raise ValueError(f'unknown on_conflict policy {on_conflict!r}; expected one of {CONFLICT_POLICIES}')

    in_root = Path(input_dir)
    out_root = Path(output_dir) if output_dir else None
    files = find_batch_files(in_root, pattern=pattern, recursive=recursive)

    if auto_save and out_root is not None:
        save_plan, source_resolved, used_resolved = plan_batch_convert(
            files, in_root, out_root, '.tif', on_conflict=on_conflict
        )
    else:
        save_plan = [(src, in_root, src.stem, None) for src in files]
        source_resolved, used_resolved = set(), set()

    if out_root is not None:
        out_root.mkdir(parents=True, exist_ok=True)

    rec = getattr(tiki, 'recorder_window', None)
    was_paused = rec.paused if rec is not None else False
    if rec is not None:
        rec.paused = True
    ok_count = 0
    errors: list[tuple[Path, str]] = []
    try:
        for n, (src, dst_dir, base_stem, planned_dst) in enumerate(save_plan, 1):
            try:
                loaded = io.read_multi(src)
            except Exception as exc:
                errors.append((src, str(exc)))
                if progress is not None and progress(n, len(save_plan)) is False:
                    break
                continue

            multi = len(loaded) > 1
            if scenes == 'active':
                before = set(tiki.windows.viewers)
                try:
                    for idx, (data, meta) in enumerate(loaded):
                        tiki.open_scene_entry(data, meta, scene_idx=idx if multi else None, raise_errors=True)
                    ok, log_text, _ns = macros.run_code(
                        tiki,
                        code,
                        filename=src.name,
                        namespace={
                            'input_path': src,
                            'output_dir': out_root,
                            'scene_index': None,
                            'scene_count': len(loaded),
                            'scene_name': src.name,
                        },
                    )
                    if not ok:
                        errors.append((src, log_text.strip() or 'macro stopped'))
                    else:
                        if auto_save and out_root is not None and tiki.windows.active is not None:
                            dst = planned_dst
                            if dst is not None:
                                dst.parent.mkdir(parents=True, exist_ok=True)
                                tiki.save_layer(dst)
                        ok_count += 1
                except Exception as exc:
                    errors.append((src, str(exc)))
                finally:
                    if close_after:
                        for v in list(tiki.windows.viewers):
                            if v not in before:
                                tiki.windows.close(v)
            else:
                scene_items = list(enumerate(loaded)) if scenes == 'all' else [(0, loaded[0])]
                if multi and len(scene_items) > 1 and auto_save and out_root is not None and planned_dst is not None:
                    used_resolved.discard(planned_dst.resolve(strict=False))
                file_ok = True
                for scene_idx, (data, meta) in scene_items:
                    before = set(tiki.windows.viewers)
                    try:
                        tiki.open_scene_entry(
                            data,
                            meta,
                            scene_idx=scene_idx if multi else None,
                            raise_errors=True,
                        )
                        scene_name = str(meta.get('name') or src.name)
                        fname = f'{src.name}[{scene_idx}]' if multi else src.name
                        ok, log_text, _ns = macros.run_code(
                            tiki,
                            code,
                            filename=fname,
                            namespace={
                                'input_path': src,
                                'output_dir': out_root,
                                'scene_index': scene_idx,
                                'scene_count': len(loaded),
                                'scene_name': scene_name,
                            },
                        )
                        if not ok:
                            file_ok = False
                            msg = log_text.strip() or 'macro stopped'
                            errors.append((src, f'scene {scene_idx + 1}: {msg}' if multi else msg))
                        elif auto_save and out_root is not None and tiki.windows.active is not None:
                            if len(scene_items) > 1:
                                dst = _unique_dest(
                                    dst_dir / f'{base_stem}_s{scene_idx + 1}.tif',
                                    source_resolved,
                                    used_resolved,
                                    on_conflict,
                                )
                            else:
                                dst = planned_dst
                            if dst is not None:
                                dst.parent.mkdir(parents=True, exist_ok=True)
                                tiki.save_layer(dst)
                    except Exception as exc:
                        file_ok = False
                        errors.append((src, f'scene {scene_idx + 1}: {exc}' if multi else str(exc)))
                    finally:
                        if close_after:
                            for v in list(tiki.windows.viewers):
                                if v not in before:
                                    tiki.windows.close(v)
                if file_ok:
                    ok_count += 1

            if progress is not None and progress(n, len(save_plan)) is False:
                break
    finally:
        if rec is not None:
            rec.paused = was_paused

    if out_root is not None and save_results_csv and tiki.results.rows:
        csv_path = out_root / 'Results.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(tiki.results._header_labels())
            for row in tiki.results.rows:
                writer.writerow([row.get(k, '') for k in tiki.results.columns])

    return ok_count, errors


def _dir_row(parent, title: str, initial: str = ''):
    edit = QLineEdit(initial)
    btn = QPushButton('Browse…')

    def pick():
        chosen = QFileDialog.getExistingDirectory(parent, title, edit.text() or str(Path.home()))
        if chosen:
            edit.setText(chosen)

    btn.clicked.connect(pick)
    row = QHBoxLayout()
    row.addWidget(edit, 1)
    row.addWidget(btn)
    return edit, row


def batch_convert_dialog(tiki):
    """``Process ▸ Batch ▸ Convert…``: convert a folder of images to TIFF, PNG, JPEG, or NPY."""
    dlg, form = process._form(tiki, 'Batch Convert')
    in_edit, in_row = _dir_row(dlg, 'Select Input Folder')
    out_edit, out_row = _dir_row(dlg, 'Select Output Folder')
    pat_edit = QLineEdit('*')
    rec_box = QCheckBox('Include subfolders')
    fmt_box = QComboBox()
    fmt_box.addItem('TIFF (.tif, calibrated / OME channels)', 'tiff')
    fmt_box.addItem('PNG (.png, 8-bit)', 'png')
    fmt_box.addItem('JPEG (.jpg, 8-bit)', 'jpeg')
    fmt_box.addItem('NumPy (.npy)', 'npy')
    stack_box = QComboBox()
    stack_box.addItem('Max Z/T projection (for 2D PNG/JPEG)', 'max')
    stack_box.addItem('First plane (for 2D PNG/JPEG)', 'first')
    conflict_box = QComboBox()
    conflict_box.addItem('Rename automatically (_2, _out)', 'rename')
    conflict_box.addItem('Skip existing / conflicting', 'skip')
    conflict_box.addItem('Overwrite non-source outputs', 'overwrite')
    count_label = QLabel('Choose an input folder')

    form.addRow('Input folder:', in_row)
    form.addRow('Output folder:', out_row)
    form.addRow('Filename pattern:', pat_edit)
    form.addRow(rec_box)
    form.addRow('Output format:', fmt_box)
    form.addRow('Stacks in 2D formats:', stack_box)
    form.addRow('Existing / colliding files:', conflict_box)
    form.addRow('', count_label)
    buttons = process._buttons(dlg, form)

    def refresh(_=None):
        in_dir = Path(in_edit.text().strip())
        out_txt = out_edit.text().strip()
        files = find_batch_files(in_dir, pat_edit.text(), rec_box.isChecked()) if in_edit.text().strip() else []
        ok = bool(files) and bool(out_txt)
        msg = f'{len(files)} matching file(s)' if in_edit.text().strip() else 'Choose an input folder'
        if files and out_txt and in_dir.resolve(strict=False) == Path(out_txt).resolve(strict=False):
            msg += ' (same folder as input: source files will not be overwritten)'
        count_label.setText(msg)
        buttons.button(buttons.Ok).setEnabled(ok)
        stack_box.setEnabled(fmt_box.currentData() in ('png', 'jpeg'))

    in_edit.textChanged.connect(refresh)
    out_edit.textChanged.connect(refresh)
    pat_edit.textChanged.connect(refresh)
    rec_box.toggled.connect(refresh)
    fmt_box.currentIndexChanged.connect(refresh)
    refresh()

    if dlg.exec_() != QDialog.Accepted:
        return None
    files = find_batch_files(in_edit.text().strip(), pat_edit.text(), rec_box.isChecked())
    with tiki.progress('Batch Convert', max(len(files), 1)) as tick:
        n_ok, errs = batch_convert(
            in_edit.text().strip(),
            out_edit.text().strip(),
            fmt=fmt_box.currentData(),
            pattern=pat_edit.text(),
            recursive=rec_box.isChecked(),
            stack_mode=stack_box.currentData(),
            on_conflict=conflict_box.currentData(),
            progress=tick,
        )
    if errs:
        log = tiki.show_log()
        for p, msg in errs:
            log.append(f'[Batch Convert ERROR] {p}: {msg}')
    tiki.status(f'Batch Convert: {n_ok} file(s) converted' + (f', {len(errs)} error(s)' if errs else ''))
    return n_ok, errs


def batch_macro_dialog(tiki):
    """``Process ▸ Batch ▸ Run Macro…``: run Python macro code over a folder of images."""
    dlg, form = process._form(tiki, 'Batch Process (Python Macro)')
    dlg.resize(600, 520)
    in_edit, in_row = _dir_row(dlg, 'Select Input Folder')
    out_edit, out_row = _dir_row(dlg, 'Select Output Folder (optional)')
    pat_edit = QLineEdit('*')
    rec_box = QCheckBox('Include subfolders')
    preset_box = QComboBox()
    preset_box.addItem('(Custom / current editor)', '')
    for p in macros.list_macros():
        preset_box.addItem(p.name, str(p))
    if tiki.recorder_window is not None and tiki.recorder_window.text.toPlainText().strip():
        preset_box.addItem('(From Macro Recorder)', '__recorder__')

    code_edit = QPlainTextEdit(
        '# Available variables: tiki, np, input_path, output_dir\n'
        '# TIKI opens each file before running this code.\n'
        "tiki.run('Measure')\n"
    )
    font = QFont('Monospace')
    font.setStyleHint(QFont.TypeWriter)
    code_edit.setFont(font)
    code_edit.setMinimumHeight(140)
    auto_save = QCheckBox('Auto-save active image as TIFF to output folder')
    scenes_box = QComboBox()
    scenes_box.addItem('Each scene separately', 'all')
    scenes_box.addItem('First scene only', 'first')
    scenes_box.addItem('All scenes open, run once on active', 'active')
    conflict_box = QComboBox()
    conflict_box.addItem('Rename automatically (_2, _out)', 'rename')
    conflict_box.addItem('Skip existing / conflicting', 'skip')
    conflict_box.addItem('Overwrite non-source outputs', 'overwrite')
    conflict_box.setEnabled(False)
    save_csv = QCheckBox('Write Results.csv to output folder', checked=True)
    close_box = QCheckBox('Close image windows after each file', checked=True)
    count_label = QLabel('Choose an input folder')

    def load_preset(_=None):
        val = preset_box.currentData()
        if not val:
            return
        if val == '__recorder__' and tiki.recorder_window is not None:
            code_edit.setPlainText(tiki.recorder_window.code())
        elif Path(val).is_file():
            code_edit.setPlainText(Path(val).read_text())

    preset_box.currentIndexChanged.connect(load_preset)

    form.addRow('Input folder:', in_row)
    form.addRow('Output folder:', out_row)
    form.addRow('Filename pattern:', pat_edit)
    form.addRow(rec_box)
    form.addRow('Multi-scene files:', scenes_box)
    form.addRow('Load macro:', preset_box)
    form.addRow('Python code:', code_edit)
    form.addRow(auto_save)
    form.addRow('Auto-save conflicts:', conflict_box)
    form.addRow(save_csv)
    form.addRow(close_box)
    form.addRow('', count_label)
    buttons = process._buttons(dlg, form)

    def refresh(_=None):
        in_dir = Path(in_edit.text().strip())
        files = find_batch_files(in_dir, pat_edit.text(), rec_box.isChecked()) if in_edit.text().strip() else []
        need_out = auto_save.isChecked() or save_csv.isChecked()
        has_out = bool(out_edit.text().strip())
        ok = bool(files) and bool(code_edit.toPlainText().strip()) and (not need_out or has_out)
        if not in_edit.text().strip():
            count_label.setText('Choose an input folder')
        elif need_out and not has_out:
            count_label.setText(f'{len(files)} matching file(s) — choose an Output folder for saving')
        else:
            count_label.setText(f'{len(files)} matching file(s)')
        buttons.button(buttons.Ok).setEnabled(ok)
        conflict_box.setEnabled(auto_save.isChecked())

    in_edit.textChanged.connect(refresh)
    out_edit.textChanged.connect(refresh)
    pat_edit.textChanged.connect(refresh)
    rec_box.toggled.connect(refresh)
    code_edit.textChanged.connect(refresh)
    auto_save.toggled.connect(refresh)
    save_csv.toggled.connect(refresh)
    refresh()

    if dlg.exec_() != QDialog.Accepted:
        return None
    files = find_batch_files(in_edit.text().strip(), pat_edit.text(), rec_box.isChecked())
    with tiki.progress('Batch Macro', max(len(files), 1)) as tick:
        n_ok, errs = batch_macro(
            tiki,
            in_edit.text().strip(),
            code_edit.toPlainText(),
            output_dir=out_edit.text().strip() or None,
            pattern=pat_edit.text(),
            recursive=rec_box.isChecked(),
            auto_save=auto_save.isChecked(),
            save_results_csv=save_csv.isChecked(),
            close_after=close_box.isChecked(),
            scenes=scenes_box.currentData(),
            on_conflict=conflict_box.currentData(),
            progress=tick,
        )
    if errs:
        log = tiki.show_log()
        for p, msg in errs:
            log.append(f'[Batch Macro ERROR] {p}: {msg}')
        QMessageBox.warning(tiki, 'Batch Macro', f'Finished {n_ok} file(s) with {len(errs)} error(s); see Log.')
    tiki.status(f'Batch Macro: {n_ok} file(s) done' + (f', {len(errs)} error(s)' if errs else ''))
    return n_ok, errs
