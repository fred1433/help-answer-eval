"""Three readers, one model, one instruction, one set of source notes. Only the evidence differs.

A. full corpus     every entry of every page, with page URL and last-updated date.
B. passages        sentence windows carrying their page and entry title, hybrid retrieval
                   (BM25 + local embeddings, reciprocal rank fusion), local cross-encoder
                   reranking, each hit widened by its neighbouring window, filled to a word budget.
C. whole answers   complete entries, same hybrid retrieval and reranking, same word budget.

B and C spend the same evidence budget. A reads everything and is reported apart.
The operator's source notes (which page is the older version, which conflicts are unresolved)
are given to all three, word for word.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .retrieval import BM25, Dense, rrf, top_k

INSTRUCTION = """You answer customer emails for {business}.
Use only the reference material in the message. Be specific and brief (under 150 words), plain text.
If the answer depends on something the customer did not say, say what it depends on or ask.
If the reference material does not settle the question, do not guess: say you will check with {escalate_to}."""


def instruction(escalate_to: str, business: str = "a small business") -> str:
    return INSTRUCTION.format(escalate_to=escalate_to, business=business)


def header(e) -> str:
    return f"[{e.url} | page updated {e.modified[:10]} | entry: {e.question}]"


def fmt_entry(e) -> str:
    return f"{header(e)}\n{e.answer}"


def context_full(entries) -> str:
    """Pages oldest first, entries in page order."""
    pages: dict[str, list] = {}
    for e in entries:
        pages.setdefault(e.url, []).append(e)
    ordered = sorted(pages.items(), key=lambda kv: (kv[1][0].modified, kv[0]))
    return "\n\n".join(fmt_entry(e) for _, es in ordered for e in es)


_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def sentences(text: str) -> list[str]:
    out = []
    for para in text.split("\n"):
        out.extend(s.strip() for s in _SENT.split(para) if s.strip())
    return out


def windows(text: str, size: int = 3, stride: int = 2) -> list[str]:
    s = sentences(text)
    if len(s) <= size:
        return [" ".join(s)] if s else []
    return [" ".join(s[i:i + size]) for i in range(0, len(s) - size + stride, stride) if s[i:i + size]]


def nwords(t: str) -> int:
    return len(t.split())


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", scorer=None):
        self.model_name = model_name
        self._scorer = scorer

    def scores(self, query: str, docs: list[str]) -> list[float]:
        if self._scorer is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            enc = TextCrossEncoder(self.model_name)
            self._scorer = lambda q, ds: list(enc.rerank(q, ds))
        return [float(x) for x in self._scorer(query, docs)]


RERANK_SYSTEM = "You rank passages by how useful they are for answering a customer email. Reply with JSON only."
RERANK_PROMPT = """Customer email:
{query}

Candidate passages:
{docs}

Return JSON {{"ranking": [candidate numbers, most useful first, at most 15]}}. Leave out passages that do not help."""


class LLMReranker(Reranker):
    """Listwise reranking by a model through the same cached runner as everything else."""

    def __init__(self, model: str, cache_dir, runner=None):
        super().__init__(model_name=model)
        from . import llm
        self._llm, self.cache_dir, self.runner = llm, cache_dir, runner

    def scores(self, query: str, docs: list[str]) -> list[float]:
        import json as _json
        listing = "\n\n".join(f"[{i + 1}] {d}" for i, d in enumerate(docs))
        rec = self._llm.call(self.model_name, RERANK_SYSTEM, RERANK_PROMPT.format(query=query, docs=listing),
                             self.cache_dir, runner=self.runner)
        m = re.search(r"\{.*\}", rec["result"], re.S)
        order = _json.loads(m.group(0)).get("ranking", []) if m else []
        score = [-1000.0] * len(docs)
        for rank, n in enumerate(order):
            if isinstance(n, int) and 1 <= n <= len(docs) and score[n - 1] == -1000.0:
                score[n - 1] = -float(rank)
        return score


@dataclass
class Retriever:
    """Hybrid retrieval + rerank over a list of units (text, entry, position)."""
    units: list
    dense: Dense
    reranker: Reranker
    candidates: int = 40

    def __post_init__(self):
        texts = [u[0] for u in self.units]
        self.bm25 = BM25(texts)
        self.vecs = self.dense.embed(texts) if texts else None

    def ranked(self, question: str) -> list[int]:
        n = len(self.units)
        fused = rrf([top_k(self.bm25.scores(question), n)[:self.candidates],
                     top_k(self.dense.scores(question, self.vecs), n)[:self.candidates]])[:self.candidates]
        rs = self.reranker.scores(question, [self.units[i][0] for i in fused])
        return [fused[j] for j in sorted(range(len(fused)), key=lambda j: (-rs[j], j))]


class PassageReader:
    def __init__(self, entries, dense, reranker, budget: int = 1000):
        self.budget = budget
        units = []
        for e in entries:
            for k, w in enumerate(windows(e.answer)):
                units.append((f"{header(e)} {w}", e, k))
        self.by_entry = {}
        for i, (_, e, k) in enumerate(units):
            self.by_entry.setdefault(e.id, {})[k] = i
        self.units = units
        self.r = Retriever(units, dense, reranker)

    def context(self, question: str) -> str:
        chosen: list[int] = []
        used = 0
        for i in self.r.ranked(question):
            _, e, k = self.units[i]
            group = [self.by_entry[e.id][j] for j in (k - 1, k, k + 1) if j in self.by_entry[e.id]]
            new = [g for g in group if g not in chosen]
            cost = sum(nwords(self.units[g][0]) for g in new)
            if not new or used + cost > self.budget:
                continue
            chosen.extend(new)
            used += cost
        # print passages grouped by entry, in window order, so a condition split across windows reads in order
        blocks: dict[str, list] = {}
        for g in chosen:
            _, e, k = self.units[g]
            blocks.setdefault(e.id, [e, []])[1].append(k)
        out = []
        for e, ks in blocks.values():
            ws = windows(e.answer)
            out.append(header(e) + "\n" + " [...] ".join(ws[k] for k in sorted(set(ks))))
        return "\n\n".join(out)


class AnswerReader:
    def __init__(self, entries, dense, reranker, budget: int = 1000):
        self.budget = budget
        self.units = [(fmt_entry(e), e, 0) for e in entries]
        self.r = Retriever(self.units, dense, reranker)

    def context(self, question: str) -> str:
        out, used = [], 0
        for i in self.r.ranked(question):
            text = self.units[i][0]
            if used + nwords(text) > self.budget:
                continue
            out.append(text)
            used += nwords(text)
        return "\n\n".join(out)


def prompt(notes: str, context: str, question: str) -> str:
    return (f"<source_notes>\n{notes}\n</source_notes>\n\n<reference>\n{context}\n</reference>\n\n"
            f"Customer email:\n{question}")
