"""Resumable capture and strict parsing of public, dated NFL evidence.

The source files are retrospective captures, not original publication vintages.
Keep both the source date and collection timestamp, and never substitute one for
the other. Empty, redirected, and duplicate-year archives are coverage failures.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import threading
import time
from datetime import UTC, date, datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import polars as pl
import requests

from patron.data.nfl_transactions import _OfficialTransactionParser, parse_official_transactions

BACKFILL_FILE = "historical_transaction_backfill.parquet"
BACKFILL_MANIFEST = "historical_backfill_manifest.json"


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def expand_dated_revisions(frame: pl.DataFrame) -> pl.DataFrame:
    """Separate explicitly dated later activations embedded in an older archive row."""
    extra = []
    pattern = re.compile(
        r"placed ([^.;]+?) on [^.;]+?, then reverted to "
        r"(?:the )?active roster on (\d{1,2})/(\d{1,2})",
        re.I,
    )
    for row in frame.to_dicts():
        for match in pattern.finditer(row["description"]):
            effective = date(row["transaction_year"], int(match[2]), int(match[3]))
            extra.append(
                {
                    **row,
                    "transaction_date": max(row["transaction_date"], effective),
                    "description": f"Activated {match[1]} to the active roster "
                    f"on {match[2]}/{match[3]}.",
                    "date_basis": "max_archive_and_embedded_event_date",
                }
            )
    if not extra:
        return frame
    return pl.concat([frame, pl.DataFrame(extra, schema=frame.schema)], how="vertical")


def load_transaction_backfill(data_dir: Path) -> pl.DataFrame | None:
    """Verify the export and every contributing raw capture before using evidence."""
    path = data_dir / "static" / BACKFILL_FILE
    if not path.exists():
        return None
    manifest = json.loads((data_dir / "static" / BACKFILL_MANIFEST).read_text())
    if manifest.get("schema_version") != 1 or sha256(path.read_bytes()) != manifest["sha256"]:
        raise ValueError("Historical transaction backfill manifest mismatch")
    captures = {}
    for filename, expected in manifest["captures"].items():
        if Path(filename).name != filename:
            raise ValueError("Invalid evidence capture path")
        raw = gzip.decompress((data_dir / "cache/historical_evidence/v1" / filename).read_bytes())
        if sha256(raw) != expected:
            raise ValueError(f"Historical evidence capture changed: {filename}")
        captures[filename] = expected
    frame = pl.read_parquet(path)
    for row in frame.select("source_capture_file", "source_sha256").unique().to_dicts():
        if captures.get(row["source_capture_file"]) != row["source_sha256"]:
            raise ValueError("Historical transaction has unbound source provenance")
    return frame


class EvidenceCapture:
    """Immutable per-URL captures with bounded concurrency and host throttling."""

    def __init__(self, root: Path, delay: float = 0.5):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self.lock = threading.Lock()
        self.host_next: dict[str, float] = {}

    def fetch(self, url: str) -> tuple[dict, str]:
        key = sha256(url.encode())
        metadata_path = self.root / f"{key}.json"
        body_path = self.root / f"{key}.html.gz"
        if metadata_path.exists():
            record = json.loads(metadata_path.read_text())
            raw = gzip.decompress(body_path.read_bytes()) if body_path.exists() else b""
            if record["sha256"] != sha256(raw):
                raise ValueError(f"Captured evidence changed: {url}")
            return record, raw.decode("utf-8", errors="replace")
        host = urlparse(url).netloc
        with self.lock:
            now = time.monotonic()
            slot = max(now, self.host_next.get(host, now))
            self.host_next[host] = slot + self.delay
        time.sleep(max(0, slot - time.monotonic()))
        record = {"requested_url": url, "retrieved_at": datetime.now(UTC).isoformat()}
        raw = b""
        try:
            response = requests.get(url, timeout=30, headers={"User-Agent": "Patron/1.0"})
            raw = response.content
            record.update(status_code=response.status_code, final_url=response.url)
        except requests.RequestException as error:
            record.update(status_code=0, final_url=url, error=str(error))
        record.update(sha256=sha256(raw), capture_file=body_path.name)
        body_path.write_bytes(gzip.compress(raw, mtime=0))
        metadata_path.write_text(json.dumps(record, indent=2) + "\n")
        return record, raw.decode("utf-8", errors="replace")


def archive_rows(html: str, record: dict, year: int, team: str) -> tuple[list[dict], str, str]:
    """Require the exact year route and reject a selected different year."""
    expected = f"/team/transactions/{year}"
    if record["status_code"] != 200:
        return [], "fetch_failed", ""
    if urlparse(record["final_url"]).path.rstrip("/") != expected:
        return [], "redirected_year", ""
    canonical = re.search(r"<link\b[^>]*rel=[\"\']canonical[\"\'][^>]*href=[\"\']([^\"\']+)", html)
    if not canonical or urlparse(unescape(canonical[1])).path.rstrip("/") != expected:
        return [], "unverified_year", ""
    selected = re.findall(r"<option\b[^>]*\bselected\b[^>]*>\s*(\d{4})\s*</option>", html)
    if selected and str(year) not in selected:
        return [], "wrong_selected_year", ""
    parsed = _OfficialTransactionParser()
    parsed.feed(html)
    fingerprint = sha256(json.dumps(parsed.rows, ensure_ascii=False).encode())
    try:
        rows = parse_official_transactions(
            html,
            year=year,
            team=team,
            source_url=record["final_url"],
            cutoff_date=date(year, 12, 31),
        )
    except ValueError:
        return [], "invalid_archive_date", fingerprint
    for row in rows:
        row.update(
            source_sha256=record["sha256"],
            source_capture_file=record["capture_file"],
            date_basis="official_archive_event_date",
        )
    return rows, "captured" if rows else "empty_archive", fingerprint


class ArticleIndexParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_cell = False
        self.parts: list[str] = []
        self.row: list[str] = []
        self.url: str | None = None
        self.records: list[dict] = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "tr":
            self.row, self.url = [], None
        if tag == "td":
            self.in_cell, self.parts = True, []
        if tag == "a" and self.in_cell and "/news/" in attrs.get("href", ""):
            self.url = urljoin("https://www.nfl.com", attrs["href"])

    def handle_data(self, data):
        if self.in_cell:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag == "td" and self.in_cell:
            self.row.append(" ".join(" ".join(self.parts).split()))
            self.in_cell = False
        if tag == "tr" and len(self.row) >= 2 and self.url:
            try:
                published = date.fromisoformat(self.row[0])
            except ValueError:
                return
            self.records.append({"published_on": published, "title": self.row[1], "url": self.url})


class ArticleParser(HTMLParser):
    """Capture article text only; related headlines and navigation are excluded."""

    def __init__(self) -> None:
        super().__init__()
        self.depth = 0
        self.article_depth: int | None = None
        self.parts: list[str] = []
        self.paragraphs: list[str] = []
        self.paragraph: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in {"meta", "link", "img", "input", "br", "hr", "source", "wbr"}:
            return
        self.depth += 1
        if self.article_depth is None and {
            "nfl-c-body-part",
            "d3-o-article__body",
            "story-part-rich-text-editor-wrapper",
        }.intersection(attrs.get("class", "").split()):
            self.article_depth = self.depth
        if self.article_depth is not None and tag == "p":
            self.paragraph = []

    def handle_data(self, data):
        if self.article_depth is not None:
            self.parts.append(data)
            if self.paragraph is not None:
                self.paragraph.append(data)

    def handle_endtag(self, tag):
        if tag in {"meta", "link", "img", "input", "br", "hr", "source", "wbr"}:
            return
        if tag == "p" and self.paragraph is not None:
            self.paragraphs.append(" ".join("".join(self.paragraph).split()))
            self.paragraph = None
        if self.depth == self.article_depth:
            self.article_depth = None
        self.depth = max(0, self.depth - 1)


def article_content(html: str) -> dict:
    dates = re.findall(r'"datePublished"\s*:\s*"([^"]+)"', html)
    modified = re.findall(r'"dateModified"\s*:\s*"([^"]+)"', html)
    parser = ArticleParser()
    parser.feed(html)
    return {
        "published_at": dates[0] if dates else None,
        "modified_at": modified[0] if modified else None,
        "paragraphs": parser.paragraphs,
        "text": " ".join(" ".join(parser.parts).split()),
    }
