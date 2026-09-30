"""Check the extracted answers against the pages as a visitor sees them, and fingerprint the snapshot.

The REST API gives the stored content; the displayed page is what a customer reads. Every
extracted answer must appear, whole, in the displayed text of its page. A mismatch means the
extraction lost something (a list item, a qualification) before any evaluation starts.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

from .corpus import UA, load_entries


def norm(s: str) -> str:
    s = html.unescape(s).replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip().lower()


def visible_text(page_html: str) -> str:
    page_html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", page_html, flags=re.S | re.I)
    return norm(re.sub(r"<[^>]+>", " ", page_html))


def check(entries, fetch=None) -> dict:
    fetch = fetch or _fetch
    by_url: dict[str, list] = {}
    for e in entries:
        by_url.setdefault(e.url, []).append(e)
    report = {}
    for url, es in by_url.items():
        text = visible_text(fetch(url))
        missing = [e.id for e in es if norm(e.answer) not in text]
        report[url] = {"entries": len(es), "missing": missing}
    return report


def _fetch(url: str) -> str:
    time.sleep(1.0)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", "replace")


def fingerprint(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == "__main__":
    pages = Path(sys.argv[1])
    exclude = set(sys.argv[2:])
    entries = [e for e in load_entries(pages) if e.url not in exclude]
    rep = check(entries)
    print(json.dumps({"snapshot_sha256": fingerprint(pages), "pages": rep}, indent=1))
