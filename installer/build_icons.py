"""Generate Windows (.ico) and macOS (.icns) icons from src/tiki/resources/tiki.png."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'src' / 'tiki' / 'resources' / 'tiki.png'
OUT = Path(__file__).resolve().parent


def main() -> None:
    img = Image.open(SRC).convert('RGBA')
    img.save(OUT / 'tiki.png')
    img.save(OUT / 'tiki.ico', format='ICO', sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    try:
        img.save(OUT / 'tiki.icns', format='ICNS')
    except Exception:
        # Pillow ICNS writer is available on all platforms in modern Pillow; ignore if unsupported
        pass
    print(f'Generated icons in {OUT}')


if __name__ == '__main__':
    main()
