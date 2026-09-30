"""Three experiments, three readers, one blind grader.

  apply     the sources stay in the corpus, conflicting versions included; only the question is new.
  missing   the email leaves out a detail the answer depends on; asking or answering
            conditionally is the right move.
  transfer  an entry is held out together with every near copy; the reader must carry what the
            other entries teach over to an unseen case (or say it cannot).

Readers: A full corpus, B passages, C whole answers (see systems.py); O is the oracle control,
the same reader given the hand-picked source entries, to separate "not found" from "found but misused".

Usage:
  python -m hae.bench fetch    CONFIG
  python -m hae.bench prepare  CONFIG   # transfer set: pick, reword, label answerability
  python -m hae.bench run      CONFIG   # answers + grades, every call cached on disk
  python -m hae.bench report   CONFIG
"""
from __future__ import annotations

import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import grade, llm, systems
from .corpus import fetch_pages, load_entries
from .leak import without_variants
from .retrieval import Dense

READERS = ("A", "B", "C")


class Cfg(dict):
    def __init__(self, path):
        super().__init__(json.loads(Path(path).read_text()))
        self.root = Path(path).resolve().parent

    def p(self, key) -> Path:
        v = Path(self[key])
        return v if v.is_absolute() else self.root / v


def corpus(cfg):
    ex = set(cfg.get("exclude_urls", []))
    return [e for e in load_entries(cfg.p("pages")) if e.url not in ex]


def _json(text):
    return json.loads(re.search(r"\{.*\}", text, re.S).group(0))


def _norm(s):
    return re.sub(r"\s+", " ", s.replace("’", "'")).strip().lower()


# ---------- prepare the transfer set ----------

REWORD_SYSTEM = "You turn help-page entries into the email a real customer would send. Reply with JSON only, in English."
REWORD_PROMPT = """For each entry below, write the short email (1 to 3 sentences, first person, plain words) a customer
with this problem would send BEFORE reading the answer. Do not include any fact, number or fix that only the
answer contains. Return a JSON object mapping each id to its email.

{items}"""

LABEL_SYSTEM = "You check whether a reference text contains what is needed to answer a question. Reply with JSON only."
LABEL_PROMPT = """<reference>
{context}
</reference>

Customer email:
{question}

The source answer (NOT in the reference, for your comparison only):
{expert}

Does the reference contain the facts needed to give this answer's main action to this customer?
Return JSON: {{"answerable": true or false, "quotes": [up to 3 verbatim sentences copied exactly from the reference that support it]}}"""


def prepare(cfg, runner=None):
    entries = corpus(cfg)
    b = cfg["transfer"]
    pool = [e for e in entries if e.url in set(b["pool_urls"]) and len(e.answer.split()) >= b.get("min_words", 25)]
    pool.sort(key=lambda e: e.id)
    held = random.Random(b["seed"]).sample(pool, b["n"])
    cache = cfg.p("out") / "calls"
    items = "\n\n".join(f"id: {e.id}\ntitle: {e.question}\nanswer: {e.answer}" for e in held)
    emails = _json(llm.call(cfg["grader_model"], REWORD_SYSTEM, REWORD_PROMPT.format(items=items), cache, runner=runner)["result"])

    def label(e):
        kept, removed = without_variants(entries, e)
        ctx = systems.context_full(kept)
        lab = _json(llm.call(cfg["grader_model"], LABEL_SYSTEM,
                             LABEL_PROMPT.format(context=ctx, question=emails[e.id], expert=e.answer), cache, runner=runner)["result"])
        hay = _norm(ctx)
        verified = [q for q in lab.get("quotes", []) if len(q) > 20 and _norm(q) in hay]
        return {"id": e.id, "title": e.question, "question": emails[e.id], "question_source": "reworded by us from the entry title",
                "expert": e.answer, "url": e.url, "modified": e.modified, "removed_variants": [x.id for x in removed],
                "answerable": bool(lab.get("answerable")) and bool(verified),
                "answerable_claimed": bool(lab.get("answerable")), "support": verified}

    with ThreadPoolExecutor(cfg.get("workers", 6)) as ex:
        rows = list(ex.map(label, held))
    (cfg.p("out") / "transfer_questions.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    print(f"{len(rows)} held-out questions, {sum(r['answerable'] for r in rows)} answerable from the rest")
    return rows


# ---------- run ----------

def load_cases(cfg):
    spec = json.loads(cfg.p("cases").read_text())
    tq = cfg.p("out") / "transfer_questions.json"
    transfer = json.loads(tq.read_text()) if tq.exists() else []
    return spec, transfer


def build_jobs(cfg, dense=None, reranker=None, only=None, notes_append=""):
    entries = corpus(cfg)
    by_id = {e.id: e for e in entries}
    dense = dense or Dense(cache_file=str(cfg.p("out") / "embeddings.pkl"))
    if reranker is None:
        name = cfg.get("reranker", "BAAI/bge-reranker-base")
        reranker = (systems.LLMReranker(name[4:], cfg.p("out") / "calls") if name.startswith("llm:")
                    else systems.Reranker(name))
    spec, transfer = load_cases(cfg)
    notes = spec["notes"] + (("\n" + notes_append) if notes_append else "")
    if only is not None:
        spec = {**spec, "cases": [c for c in spec["cases"] if c["id"] in only]}
        transfer = [q for q in transfer if q["id"] in only]
    sysmsg = systems.instruction(cfg["escalate_to"], cfg.get("business", "a small business"))
    budget = cfg.get("budget_words", 1000)
    items = [(c["experiment"], c["id"], c["question"], None, c["expect"], c["sources"], True) for c in spec["cases"]]
    for q in transfer:
        expect = {"does": "gives the main action of the source answer below to this customer",
                  "must_not": "", "conditions": "", "clarify_ok": not q["answerable"]}
        items.append(("transfer", q["id"], q["question"], q["id"], expect, [q["id"]], False))
    dense.embed([it[2] for it in items])  # every query embedded up front: worker threads only read the memo
    dense.save()

    def build(it):
        experiment, case_id, question, held, expect, source_ids, oracle = it
        visible = entries if held is None else without_variants(entries, by_id[held])[0]
        ctx = {"A": systems.context_full(visible),
               "B": systems.PassageReader(visible, dense, reranker, budget).context(question),
               "C": systems.AnswerReader(visible, dense, reranker, budget).context(question)}
        if oracle:
            ctx["O"] = "\n\n".join(systems.fmt_entry(by_id[i]) for i in source_ids)
        sources = "\n\n".join(systems.fmt_entry(by_id[i]) if i in by_id else i for i in source_ids)
        print(f"context ready: {case_id}", flush=True)
        out = []
        for r, c in ctx.items():
            received = [] if r == "A" else re.findall(r"^\[(\S+) \| page updated (\S+) \| entry: (.*)\]$", c, re.M)
            out.append({"experiment": experiment, "case": case_id, "reader": r, "system_prompt": sysmsg,
                        "prompt": systems.prompt(notes, c, question), "context_words": len(c.split()),
                        "received": received, "question": question, "expect": expect, "sources": sources})
        return out

    with ThreadPoolExecutor(cfg.get("workers", 6)) as ex:
        jobs = [j for group in ex.map(build, items) for j in group]
    dense.save()
    return jobs


def run(cfg, runner=None, dense=None, reranker=None):
    jobs = build_jobs(cfg, dense, reranker)
    cache = cfg.p("out") / "calls"

    def one(job):
        ans = llm.call(cfg["answer_model"], job["system_prompt"], job["prompt"], cache, runner=runner)
        gp = grade.grader_prompt(job["question"], job["expect"], job["sources"], ans["result"])
        g = llm.call(cfg["grader_model"], grade.GRADER_SYSTEM, gp, cache, runner=runner)
        parsed = grade.parse(g["result"])
        return {"experiment": job["experiment"], "case": job["case"], "reader": job["reader"],
                "context_words": job["context_words"], "received": job["received"], "answer": ans["result"], "usage": ans.get("usage", {}),
                "answer_call": ans["key"], "grade_call": g["key"], **parsed,
                "success": grade.success(parsed["outcome"], bool(job["expect"].get("clarify_ok")))}

    with ThreadPoolExecutor(cfg.get("workers", 6)) as ex:
        rows = list(ex.map(one, jobs))
    (cfg.p("out") / "graded.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    return rows


def regrade(cfg, src="graded.json", dst="graded_site.json", runner=None, dense=None, reranker=None):
    """Grade the same replies again, the grader now also seeing the whole site. Replies are not rerun."""
    jobs = {(j["experiment"], j["case"], j["reader"]): j for j in build_jobs(cfg, dense, reranker)}
    site = systems.context_full(corpus(cfg))
    notes = load_cases(cfg)[0]["notes"]
    cache = cfg.p("out") / "calls"
    rows = json.loads((cfg.p("out") / src).read_text())

    def one(r):
        job = jobs[(r["experiment"], r["case"], r["reader"])]
        gp = grade.grader_prompt(job["question"], job["expect"], job["sources"], r["answer"], site=site, notes=notes)
        g = llm.call(cfg["grader_model"], grade.GRADER_SYSTEM, gp, cache, runner=runner)
        parsed = grade.parse(g["result"])
        return {**r, **parsed, "grade_call": g["key"], "graded_with": "whole site",
                "success": grade.success(parsed["outcome"], bool(job["expect"].get("clarify_ok")))}

    with ThreadPoolExecutor(cfg.get("workers", 6)) as ex:
        out = list(ex.map(one, rows))
    (cfg.p("out") / dst).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out


def intervene(cfg, runner=None, dense=None, reranker=None):
    """Rerun only the listed cases after one declared change (here: one line added to the source notes).
    Answers and grades go to graded_intervention.json; the main results are untouched."""
    iv = cfg["intervention"]
    jobs = build_jobs(cfg, dense, reranker, only=set(iv["cases"]), notes_append=iv["notes_append"])
    site = systems.context_full(corpus(cfg))
    notes = load_cases(cfg)[0]["notes"] + "\n" + iv["notes_append"]
    cache = cfg.p("out") / "calls"

    def one(job):
        ans = llm.call(cfg["answer_model"], job["system_prompt"], job["prompt"], cache, runner=runner)
        gp = grade.grader_prompt(job["question"], job["expect"], job["sources"], ans["result"], site=site, notes=notes)
        g = llm.call(cfg["grader_model"], grade.GRADER_SYSTEM, gp, cache, runner=runner)
        parsed = grade.parse(g["result"])
        return {"experiment": job["experiment"], "case": job["case"], "reader": job["reader"],
                "context_words": job["context_words"], "received": job["received"], "answer": ans["result"],
                "answer_call": ans["key"], "grade_call": g["key"], **parsed, "intervention": iv["notes_append"],
                "success": grade.success(parsed["outcome"], bool(job["expect"].get("clarify_ok")))}

    with ThreadPoolExecutor(cfg.get("workers", 6)) as ex:
        rows = list(ex.map(one, jobs))
    (cfg.p("out") / "graded_intervention.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    for r in rows:
        print(r["case"], r["reader"], r["outcome"])
    return rows


# ---------- report ----------

def report(cfg, src="graded.json"):
    rows = json.loads((cfg.p("out") / src).read_text())
    summary: dict = {}
    by_case: dict = {}
    for r in rows:
        s = summary.setdefault(r["experiment"], {}).setdefault(r["reader"], {"n": 0, "success": 0})
        s["n"] += 1
        s["success"] += int(r["success"])
        s[r["outcome"]] = s.get(r["outcome"], 0) + 1
        by_case.setdefault(r["experiment"], {}).setdefault(r["case"], {})[r["reader"]] = r["success"]
    pairs = {exp: {f"{a} vs {b}": grade.paired(cases, a, b) for a, b in (("A", "B"), ("A", "C"), ("B", "C"))}
             for exp, cases in by_case.items()}
    out = {"summary": summary, "paired": pairs}
    (cfg.p("out") / src.replace("graded", "results")).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1))
    return out


def main(argv):
    cmd, path = argv[1], argv[2]
    cfg = Cfg(path)
    if cmd == "fetch":
        print(fetch_pages(cfg["site"], cfg.p("pages")))
    elif cmd == "prepare":
        prepare(cfg)
    elif cmd == "run":
        run(cfg)
    elif cmd == "intervene":
        intervene(cfg)
    elif cmd == "regrade":
        regrade(cfg)
    elif cmd == "report":
        report(cfg, argv[3] if len(argv) > 3 else "graded.json")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
