"""Hold-out hygiene: when a question is held out, every copy of its answer leaves the corpus too.

Help centers repeat themselves: an older page keeps a near copy, a topic page pastes the
same paragraph. If any copy stays, a system "answers" by finding the duplicate, and the
bench measures duplicate search instead of judgment. Three independent signals mark a
variant; any one is enough.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

_WORD = re.compile(r"[a-z0-9]+")


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def jaccard(a: str, b: str) -> float:
    sa, sb = set(words(a)), set(words(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def shingles(text: str, n: int = 8) -> set[tuple[str, ...]]:
    w = words(text)
    return {tuple(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def containment(a: str, b: str, n: int = 8) -> float:
    """Share of a's n-word shingles that also appear in b."""
    sa = shingles(a, n)
    if not sa:
        return 0.0
    return len(sa & shingles(b, n)) / len(sa)


def is_variant(held, other, *, q_sim: float = 0.6, j_min: float = 0.3, c_min: float = 0.2) -> bool:
    if held.id == other.id:
        return True
    if held.block == other.block and held.url != other.url:  # same block copied to another page
        return True
    if SequenceMatcher(None, held.question.lower(), other.question.lower()).ratio() >= q_sim:
        return True
    if jaccard(held.question + " " + held.answer, other.question + " " + other.answer) >= j_min:
        return True
    if containment(held.answer, other.answer) >= c_min or containment(other.answer, held.answer) >= c_min:
        return True
    return False


def without_variants(entries, held):
    """The corpus a system may see while answering `held`."""
    kept, removed = [], []
    for e in entries:
        (removed if is_variant(held, e) else kept).append(e)
    return kept, removed
