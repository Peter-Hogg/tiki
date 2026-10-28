"""Launch TIKI: ``tiki [image ...]`` or ``python -m tiki [image ...]``."""

from __future__ import annotations

import argparse
import sys


def _version() -> str:
    try:
        from importlib.metadata import version

        return version('napari-tiki')
    except Exception:
        from . import __version__

        return __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog='tiki',
        description='TIKI Is Kinda ImageJ: an ImageJ-style toolbar for napari image windows.',
        epilog='Files can also be dragged onto the toolbar or any image window.',
    )
    parser.add_argument('files', nargs='*', help='images (.tif, .npy, ...), folders, ImageJ ROIs (.roi, .zip) '
                                                 'or macros (.py) to open')
    parser.add_argument('--sample', action='store_true', help='open a small synthetic movie to try TIKI')
    parser.add_argument('--version', action='version', version=f'TIKI {_version()}')
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    from napari.qt import get_qapp

    from .app import Tiki, install_excepthook

    app = get_qapp()
    tiki = Tiki()
    install_excepthook(tiki)
    tiki.show()
    if args.files:
        tiki.open_dropped(args.files)
    if args.sample:
        tiki.open_sample_movie()
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
