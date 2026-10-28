from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

project = "TIKI"
author = "Peter Hogg"
copyright = "2026, Peter Hogg"
release = "0.0.1"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
myst_enable_extensions = ["colon_fence", "deflist", "fieldlist"]
autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {"members": True, "show-inheritance": True}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "TIKI documentation"
html_logo = "_static/tiki-logo.png"
html_favicon = "_static/favicon.png"
html_static_path = ["_static"]
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#008f9c",
        "color-brand-content": "#007984",
    },
    "dark_css_variables": {
        "color-brand-primary": "#33d6d1",
        "color-brand-content": "#63e1dc",
    },
    "source_repository": "https://github.com/Peter-Hogg/tiki/",
    "source_branch": "main",
    "source_directory": "docs/",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "napari": ("https://napari.org/stable/", None),
}

nitpicky = False
