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


def _natkey(path: Path) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', path.name)]


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
    progress=None,
) -> tuple[int, list[tuple[Path, str]]]:
    """Convert matching images in ``input_dir`` into ``output_dir`` as tiff/png/jpeg/npy."""
    in_root = Path(input_dir)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    files = find_batch_files(in_root, pattern=pattern, recursive=recursive)
    fmt_l = fmt.lower()
    ext = {'tiff': '.tif', 'png': '.png', 'jpeg': '.jpg', 'jpg': '.jpg', 'npy': '.npy'}[fmt_l]

    ok_count = 0
    errors: list[tuple[Path, str]] = []
    for n, src in enumerate(files, 1):
        try:
            rel_parent = src.relative_to(in_root).parent
            dst_dir = out_root / rel_parent
            dst_dir.mkdir(parents=True, exist_ok=True)
            scenes = io.read_multi(src)
            for idx, (data, meta) in enumerate(scenes):
                suffix = f'_s{idx + 1}' if len(scenes) > 1 else ''
                dst = dst_dir / f'{src.stem}{suffix}{ext}'
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

                    plane = _flatten_to_2d(data, rgb=rgb, stack_mode=stack_mode)
                    u8 = _to_uint8_display(plane, rgb=rgb)
                    PilImage.fromarray(u8).save(dst)
            ok_count += 1
        except Exception as exc:
            errors.append((src, str(exc)))
        if progress is not None and progress(n, len(files)) is False:
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
    progress=None,
) -> tuple[int, list[tuple[Path, str]]]:
    """Run Python macro ``code`` across matching files in ``input_dir``."""
    in_root = Path(input_dir)
    out_root = Path(output_dir) if output_dir else None
    if out_root is not None:
        out_root.mkdir(parents=True, exist_ok=True)
    files = find_batch_files(in_root, pattern=pattern, recursive=recursive)

    rec = getattr(tiki, 'recorder_window', None)
    was_paused = rec.paused if rec is not None else False
    if rec is not None:
        rec.paused = True
    ok_count = 0
    errors: list[tuple[Path, str]] = []
    try:
        for n, src in enumerate(files, 1):
            before = set(tiki.windows.viewers)
            try:
                tiki.open_paths([str(src)])
                ok, log_text, _ns = macros.run_code(
                    tiki,
                    code,
                    filename=str(src.name),
                    namespace={'input_path': src, 'output_dir': out_root},
                )
                if not ok:
                    errors.append((src, log_text.strip() or 'macro stopped'))
                else:
                    if auto_save and out_root is not None and tiki.windows.active is not None:
                        rel_parent = src.relative_to(in_root).parent
                        dst_dir = out_root / rel_parent
                        dst_dir.mkdir(parents=True, exist_ok=True)
                        tiki.save_layer(dst_dir / f'{src.stem}.tif')
                    ok_count += 1
            except Exception as exc:
                errors.append((src, str(exc)))
            finally:
                if close_after:
                    for v in list(tiki.windows.viewers):
                        if v not in before:
                            tiki.windows.close(v)
            if progress is not None and progress(n, len(files)) is False:
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
    count_label = QLabel('Choose an input folder')

    form.addRow('Input folder:', in_row)
    form.addRow('Output folder:', out_row)
    form.addRow('Filename pattern:', pat_edit)
    form.addRow(rec_box)
    form.addRow('Output format:', fmt_box)
    form.addRow('Stacks in 2D formats:', stack_box)
    form.addRow('', count_label)
    buttons = process._buttons(dlg, form)

    def refresh(_=None):
        in_dir = Path(in_edit.text().strip())
        files = find_batch_files(in_dir, pat_edit.text(), rec_box.isChecked()) if in_edit.text().strip() else []
        ok = bool(files) and bool(out_edit.text().strip())
        count_label.setText(f'{len(files)} matching file(s)' if in_edit.text().strip() else 'Choose an input folder')
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
    form.addRow('Load macro:', preset_box)
    form.addRow('Python code:', code_edit)
    form.addRow(auto_save)
    form.addRow(save_csv)
    form.addRow(close_box)
    form.addRow('', count_label)
    buttons = process._buttons(dlg, form)

    def refresh(_=None):
        in_dir = Path(in_edit.text().strip())
        files = find_batch_files(in_dir, pat_edit.text(), rec_box.isChecked()) if in_edit.text().strip() else []
        ok = bool(files) and bool(code_edit.toPlainText().strip())
        count_label.setText(f'{len(files)} matching file(s)' if in_edit.text().strip() else 'Choose an input folder')
        buttons.button(buttons.Ok).setEnabled(ok)

    in_edit.textChanged.connect(refresh)
    pat_edit.textChanged.connect(refresh)
    rec_box.toggled.connect(refresh)
    code_edit.textChanged.connect(refresh)
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
            progress=tick,
        )
    if errs:
        log = tiki.show_log()
        for p, msg in errs:
            log.append(f'[Batch Macro ERROR] {p}: {msg}')
        QMessageBox.warning(tiki, 'Batch Macro', f'Finished {n_ok} file(s) with {len(errs)} error(s); see Log.')
    tiki.status(f'Batch Macro: {n_ok} file(s) done' + (f', {len(errs)} error(s)' if errs else ''))
    return n_ok, errs
