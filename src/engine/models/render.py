"""Write a portable snapshot with embedded data and no external assets."""

import json
import os
from pathlib import Path


def write_catalog(catalog: dict, output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    document = json.dumps(catalog, indent=2, allow_nan=False)
    (output / "catalog.json").write_text(document + "\n")
    template = Path(__file__).with_name("browser.html").read_text()
    base = os.path.relpath(catalog["root"], output.resolve()).replace(os.sep, "/") + "/"
    page = template.replace("__CATALOG__", document.replace("<", "\\u003c"))
    page = page.replace("__BASE__", json.dumps(base))
    destination = output / "index.html"
    destination.write_text(page)
    return destination
