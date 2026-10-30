"""Serve the built frontend from the API process when deployed (ARMCHAIR_WEB_DIST).

Locally Vite serves the frontend and proxies /api; in the container there is no Vite, so the
API serves web/dist itself. Unknown paths fall back to index.html for client-side routes.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse


def mount_web(app: FastAPI) -> None:
    """Register last: the catch-all must not shadow API routes."""
    dist = os.environ.get("ARMCHAIR_WEB_DIST")
    if not dist:
        return
    root = Path(dist).resolve()
    index = root / "index.html"

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str) -> FileResponse:
        if path == "api" or path.startswith("api/"):
            raise HTTPException(status_code=404)
        file = (root / path).resolve()
        if path and file.is_file() and file.is_relative_to(root):
            # Vite fingerprints everything under assets/; index.html must revalidate.
            cache = (
                "public, max-age=31536000, immutable" if path.startswith("assets/") else "no-cache"
            )
            return FileResponse(file, headers={"Cache-Control": cache})
        return FileResponse(index, headers={"Cache-Control": "no-cache"})
