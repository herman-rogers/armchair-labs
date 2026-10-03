"""Remove embedded notebook results before source versioning; never execute notebooks."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def clean(root: Path, *, check: bool = False) -> list[str]:
    names = (
        subprocess.check_output(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=root
        )
        .decode()
        .split("\0")
    )
    dirty = []
    for name in sorted(set(names)):
        path = root / name
        if not name or not path.is_file():
            continue
        if "__marimo__" in path.parts or ".ipynb_checkpoints" in path.parts:
            raise ValueError(f"Notebook session/output must not be versioned: {name}")
        if path.suffix != ".ipynb":
            continue
        notebook = json.loads(path.read_text())
        original = json.dumps(notebook, sort_keys=True)
        notebook.get("metadata", {}).pop("widgets", None)
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") == "code":
                cell["outputs"] = []
                cell["execution_count"] = None
            metadata = cell.get("metadata", {})
            metadata.pop("execution", None)
            metadata.pop("ExecuteTime", None)
        if json.dumps(notebook, sort_keys=True) != original:
            dirty.append(name)
            if not check:
                path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    return dirty


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    dirty = clean(args.root, check=args.check)
    print(json.dumps({"action": "check" if args.check else "clean", "notebooks": dirty}))
    if args.check and dirty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
