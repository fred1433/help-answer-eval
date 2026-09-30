"""Fetch a WordPress site's public pages once, and extract its FAQ blocks.

Only the Yoast FAQ block markup is parsed (`schema-faq-section`, `schema-faq-question`,
`schema-faq-answer`). Every entry keeps its page URL, the page's last-modified date and
the block id, so any answer can be traced back to where it was published.
"""
from __future__ import annotations

import html
import json
import re
import time
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36"


def fetch_pages(site: str, out: Path, per_page: int = 100, pause: float = 1.0) -> Path:
    """Download /wp-json/wp/v2/pages once (paginated) into a local cache file."""
    if out.exists():
        return out
    pages, n = [], 1
    while True:
        url = f"{site.rstrip('/')}/wp-json/wp/v2/pages?per_page={per_page}&page={n}"
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                batch = json.load(r)
        except urllib.error.HTTPError as e:  # WordPress answers 400 past the last page
            if e.code == 400 and pages:
                break
            raise
        if not batch:
            break
        pages.extend(batch)
        if len(batch) < per_page:
            break
        n += 1
        time.sleep(pause)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pages))
    return out


@dataclass
class Entry:
    id: str          # "<page id>:<block id>"
    url: str
    modified: str    # page last-modified date (ISO)
    block: str       # Yoast block id, e.g. faq-question-1659637025185
    question: str
    answer: str

    def to_dict(self):
        return asdict(self)


_SECTION = re.compile(r'<div class="schema-faq-section" id="([^"]+)">(.*?)</div>', re.S)
_Q = re.compile(r'<strong class="schema-faq-question">(.*?)</strong>', re.S)
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t ]+")


def clean(fragment: str) -> str:
    text = re.sub(r"</p>|<br\s*/?>", "\n", fragment)
    text = html.unescape(_TAG.sub("", text))
    lines = [_WS.sub(" ", ln).strip() for ln in text.split("\n")]
    return "\n".join(ln for ln in lines if ln).strip()


def extract_faq(pages: list[dict]) -> list[Entry]:
    entries: list[Entry] = []
    for p in pages:
        body = p.get("content", {}).get("rendered", "")
        if "schema-faq-question" not in body:
            continue
        seen: dict[str, int] = {}
        for block, inner in _SECTION.findall(body):
            m = _Q.search(inner)
            if not m:
                continue
            q = clean(m.group(1))
            a = clean(inner[m.end():])
            if not q or not a:
                continue
            seen[block] = seen.get(block, 0) + 1
            eid = f"{p['id']}:{block}" + (f"#{seen[block]}" if seen[block] > 1 else "")
            entries.append(Entry(eid, p["link"], p["modified"], block, q, a))
    return entries


def load_entries(pages_file: Path) -> list[Entry]:
    return extract_faq(json.loads(Path(pages_file).read_text()))
