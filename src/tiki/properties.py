"""Image > Properties: see and correct what each axis means.

Changing a role only relabels an axis (the data is never moved), so a
wrong choice is always harmless and can be corrected again. The dialog
refuses combinations TIKI cannot work with and says why.
"""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QVBoxLayout,
)

from . import axes as axes_mod
from . import style

UNITS = {'T': ['s', 'ms', 'min'], 'Z': ['µm', 'nm', 'mm'], 'Y': ['µm', 'nm', 'mm'], 'X': ['µm', 'nm', 'mm']}


class PropertiesDialog(QDialog):
    def __init__(self, parent, viewer, image):
        super().__init__(parent)
        style.apply(self)
        self.viewer, self.image = viewer, image
        self.info = axes_mod.get(image)
        self.shape = tuple(image.data.shape[: image.ndim])
        self.setWindowTitle(f'Properties — {image.name}')

        grid = QGridLayout()
        for c, text in enumerate(['Axis', 'Size', 'Meaning', 'Step', 'Unit']):
            head = QLabel(f'<b>{text}</b>')
            grid.addWidget(head, 0, c)
        self.rows = []
        for i, a in enumerate(self.info.axes):
            role = QComboBox()
            for r in axes_mod.ROLES:
                role.addItem(f'{r}  ({axes_mod.ROLE_NAMES[r]})', r)
            role.setCurrentIndex(axes_mod.ROLES.index(a.role))
            step = QDoubleSpinBox()
            step.setDecimals(6)
            step.setRange(0, 1e9)
            step.setSpecialValueText('uncalibrated')
            step.setValue(a.step or 0)
            unit = QComboBox()
            unit.setEditable(True)
            role.currentIndexChanged.connect(lambda _=0, k=i: self._role_changed(k))
            grid.addWidget(QLabel(str(i)), i + 1, 0)
            grid.addWidget(QLabel(str(a.size)), i + 1, 1, alignment=Qt.AlignRight)
            grid.addWidget(role, i + 1, 2)
            grid.addWidget(step, i + 1, 3)
            grid.addWidget(unit, i + 1, 4)
            self.rows.append((role, step, unit))
            self._fill_units(i, a.unit)

        source = self.info.source + ('' if self.info.confident else ' (not confident)')
        notes = '; '.join(self.info.notes)
        self.source = QLabel(f'Read from: {source}' + (f'<br><i>{notes}</i>' if notes else ''))
        self.source.setWordWrap(True)
        self.problems = QLabel('')
        self.problems.setStyleSheet('color: #b00020;')
        self.problems.setWordWrap(True)

        path = image.metadata.get('path')
        self.remember = QCheckBox('Remember for files with this many axes in this folder')
        self.remember.setVisible(bool(path))
        self.remember.setChecked(bool(path) and not self.info.confident)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel('Relabelling never moves data, so any choice can be undone here.'))
        layout.addLayout(grid)
        layout.addWidget(self.source)
        layout.addWidget(self.problems)
        layout.addWidget(self.remember)
        layout.addWidget(self.buttons)
        self._validate()

    def _fill_units(self, i: int, current: str | None = None) -> None:
        role_box, step, unit = self.rows[i]
        role = role_box.currentData()
        unit.blockSignals(True)
        unit.clear()
        options = UNITS.get(role, [])
        unit.addItems(options)
        unit.setEnabled(bool(options))
        step.setEnabled(bool(options))
        if current:
            unit.setCurrentText(current)
        unit.blockSignals(False)

    def _role_changed(self, i: int) -> None:
        self._fill_units(i)
        self._validate()

    def result_info(self) -> axes_mod.AxesInfo:
        axes = []
        for (role_box, step, unit), a in zip(self.rows, self.info.axes):
            role = role_box.currentData()
            has_units = role in UNITS
            axes.append(axes_mod.Axis(
                role, a.size,
                step.value() or None if has_units else None,
                unit.currentText() or None if has_units and step.value() else None,
            ))
        return axes_mod.AxesInfo(axes, 'set by you', True)

    def _validate(self) -> None:
        issues = self.result_info().problems(self.shape)
        self.problems.setText('' if not issues else 'Cannot apply: ' + '; '.join(issues))
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(not issues)

    def apply(self) -> axes_mod.AxesInfo:
        info = self.result_info()
        axes_mod.apply(self.viewer, self.image, info)
        path = self.image.metadata.get('path')
        if path and self.remember.isChecked():
            axes_mod.remember(Path(path), self.shape, info)
        return info
