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
    parser.add_argument('--install-desktop', action='store_true',
                        help='add TIKI to the application menu (Linux): pin it to the dock, drop files on it')
    parser.add_argument('--uninstall-desktop', action='store_true', help='remove the application-menu launcher')
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    if args.install_desktop or args.uninstall_desktop:
        from . import desktop

        try:
            paths = desktop.install() if args.install_desktop else desktop.uninstall()
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            return 1
        verb = 'Installed' if args.install_desktop else 'Removed'
        for p in paths:
            print(f'{verb} {p}')
        if args.install_desktop:
            print('TIKI is now in your applications menu (it may take a moment to appear).')
        return 0

    from napari.qt import get_qapp
    from qtpy.QtGui import QIcon

    from . import desktop
    from .app import Tiki, install_excepthook

    app = get_qapp()
    # match the launcher (StartupWMClass / desktop file) so the dock groups
    # TIKI's windows under its icon instead of an anonymous "napari" entry
    app.setApplicationName(desktop.APP_ID)
    app.setDesktopFileName(desktop.APP_ID)
    app.setWindowIcon(QIcon(str(desktop.ICON)))
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
