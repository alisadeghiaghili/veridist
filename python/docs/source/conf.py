from __future__ import annotations

import sys
from pathlib import Path

DOCS_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DOCS_ROOT))

from toolchain import postprocess_rendered_html  # noqa: E402

project = "veridist"
author = "veridist contributors"
version = "0.0"
release = "1.0.1"

extensions = ["myst_parser"]
source_suffix = {".md": "markdown"}
root_doc = "index"
language = "en"
locale_dirs = ["../locales"]
gettext_compact = False
gettext_uuid = True
gettext_location = True

nitpicky = True
exclude_patterns = [
    "api.fa.md",
    "api.de.md",
    "exponential-right-censoring.fa.md",
    "exponential-right-censoring.de.md",
]
templates_path = []
html_theme = "alabaster"
html_static_path = ["_static"]
html_css_files = ["rtl.css"]
# Math markup stays in the static Sphinx HTML node.  A local no-op asset avoids
# an external MathJax fetch during offline documentation viewing and browser QC.
mathjax_path = "mathjax-static.js"


def setup(app):
    app.connect("build-finished", postprocess_rendered_html)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
