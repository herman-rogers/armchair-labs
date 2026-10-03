"""The supported interface for accepted, queryable analytical datasets.

registry.yaml declares analytical recipes; products.py adds selected application
outputs, and league.py captures mutable league observations. build.py validates complete
replacements; workflow.py connects builds to transport.py publication. query.py
opens verified native DuckDB tables from portable Parquet releases. Use this path
for new tables instead of writing separate loaders, caches, or GCS upload code.
See docs/operations/tables.md for generation, publication, and consumer contracts.
"""

from engine.tables.query import connect

__all__ = ["connect"]
