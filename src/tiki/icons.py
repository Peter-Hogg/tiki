"""Tool icons drawn with QPainter, so TIKI ships without image assets."""

from __future__ import annotations

from qtpy.QtCore import QPointF, QRectF, Qt
from qtpy.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap, QPolygonF

SIZE = 24


def _icon(draw) -> QIcon:
    pix = QPixmap(SIZE, SIZE)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(QPen(QColor(30, 30, 30), 1.6))
    draw(p)
    p.end()
    return QIcon(pix)


def _poly(points) -> QPolygonF:
    return QPolygonF([QPointF(x, y) for x, y in points])


def rectangle(p):
    p.drawRect(QRectF(4, 6, 16, 12))


def oval(p):
    p.drawEllipse(QRectF(3, 6, 18, 12))


def polygon(p):
    p.drawPolygon(_poly([(4, 18), (7, 5), (14, 9), (20, 4), (19, 19)]))


def freehand(p):
    path = QPainterPath(QPointF(5, 16))
    path.cubicTo(1, 6, 10, 2, 13, 7)
    path.cubicTo(16, 12, 22, 6, 20, 15)
    path.cubicTo(18, 21, 9, 21, 5, 16)
    p.drawPath(path)


def line(p):
    p.drawLine(QPointF(4, 19), QPointF(20, 5))


def point(p):
    p.drawLine(QPointF(12, 4), QPointF(12, 20))
    p.drawLine(QPointF(4, 12), QPointF(20, 12))
    p.setBrush(QColor(230, 200, 0))
    p.drawEllipse(QPointF(12, 12), 3, 3)


def select(p):
    p.setBrush(QColor(255, 255, 255))
    p.drawPolygon(_poly([(7, 3), (7, 19), (11, 15), (14, 21), (16, 20), (13, 14), (18, 14)]))


def hand(p):
    # four-way arrow: ImageJ's scrolling tool
    p.drawLine(QPointF(12, 3), QPointF(12, 21))
    p.drawLine(QPointF(3, 12), QPointF(21, 12))
    for a, b, c in (
        ((9, 6), (12, 3), (15, 6)),
        ((9, 18), (12, 21), (15, 18)),
        ((6, 9), (3, 12), (6, 15)),
        ((18, 9), (21, 12), (18, 15)),
    ):
        p.drawPolyline(_poly([a, b, c]))


ICONS = {
    'rectangle': rectangle,
    'oval': oval,
    'polygon': polygon,
    'freehand': freehand,
    'line': line,
    'point': point,
    'select': select,
    'hand': hand,
}


def tool_icon(name: str) -> QIcon:
    return _icon(ICONS[name])
