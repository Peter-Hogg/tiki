"""Launch TIKI: ``tiki [image ...]`` or ``python -m tiki [image ...]``."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    from napari.qt import get_qapp

    from .app import Tiki

    argv = sys.argv[1:] if argv is None else argv
    app = get_qapp()
    tiki = Tiki()
    tiki.show()
    if argv:
        tiki.open_paths(argv)
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
