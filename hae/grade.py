"""Blind grading against a written, source-supported expectation. Frozen before any run.

The grader sees the customer email, the expectation written from the sources (what a correct
reply does, what it must not do, the conditions it must keep), the source passages, and ONE
draft. It never sees which reader wrote the draft, nor the other drafts. Tone is not scored.

One outcome per draft:
  correct            does the expected action and keeps its conditions
  unsupported        states an action, promise or fact the sources do not support or contradict
  lost condition     gives advice that holds in some situations but drops a condition that matters here
  clarified          asks for the missing detail, answers conditionally, or escalates, when the email
                     lacks a detail or the sources leave the point unresolved
  needless escalation defers or asks although the sources settle the question
  incomplete         nothing wrong, but misses the expected main action
"""
from __future__ import annotations

import json
import re

OUTCOMES = ("correct", "unsupported", "lost condition", "clarified", "needless escalation", "incomplete")

GRADER_SYSTEM = """You check draft customer replies against a written expectation derived from published sources.
You are strict and literal. Tone and style do not matter. Reply with one JSON object and nothing else."""

GRADER_TEMPLATE = """Customer email:
{question}

Expectation (written from the sources before any draft was produced; provisional, not the shop's own verdict):
Does: {does}
Must not: {must_not}
Conditions to keep: {conditions}
Clarifying or escalating counts as success here: {clarify_ok}

Source passages for this question:
{sources}
{site_block}
Draft reply:
{candidate}

Classify the draft with exactly one outcome:
- "correct": does the expected action and keeps its conditions.
- "unsupported": states an action, promise or fact that the published pages do not support or that they contradict (including anything listed under Must not). {site_rule}
- "lost condition": gives advice that holds in some situation but drops a condition listed above that matters for this customer.
- "clarified": asks for the missing detail, answers conditionally, or says it will check, when the email lacks a needed detail or the sources leave the point unresolved.
- "needless escalation": asks or defers although the sources settle the question.
- "incomplete": says nothing wrong but misses the expected main action.
If several apply, pick the first that applies in this order: unsupported, lost condition, correct, clarified, needless escalation, incomplete.

Return JSON with keys:
"outcome": one of the six strings above,
"evidence": the phrase from the draft (max 20 words) that decided the outcome,
"keeps": up to 2 short phrases (max 6 words each) of what the draft gets right,
"loses": up to 2 short phrases (max 6 words each) of what it gets wrong or leaves out."""


SITE_BLOCK = """
Notes on the sources (written by the tester):
{notes}

Everything else the site publishes (a fact found here is supported, even if it is not in the passages above):
<site>
{site}
</site>
"""
SITE_RULE = ("A fact published anywhere in <site> counts as supported, except anything listed under Must not "
             "and anything the notes mark as an older version that the current page changed.")


def grader_prompt(question: str, expect: dict, sources: str, candidate: str, site: str | None = None,
                  notes: str = "") -> str:
    return GRADER_TEMPLATE.format(
        site_block=SITE_BLOCK.format(site=site, notes=notes or "none") if site else "",
        site_rule=SITE_RULE if site else "",
        question=question, does=expect["does"], must_not=expect.get("must_not") or "nothing listed",
        conditions=expect.get("conditions") or "none listed",
        clarify_ok="yes" if expect.get("clarify_ok") else "no", sources=sources, candidate=candidate)


def parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON in grader output: {text[:200]}")
    data = json.loads(m.group(0))
    if data.get("outcome") not in OUTCOMES:
        raise ValueError(f"bad outcome {data.get('outcome')!r}")
    data["evidence"] = str(data.get("evidence", ""))
    data["keeps"] = [str(x) for x in data.get("keeps", [])][:2]
    data["loses"] = [str(x) for x in data.get("loses", [])][:2]
    return data


def success(outcome: str, clarify_ok: bool) -> bool:
    return outcome == "correct" or (clarify_ok and outcome == "clarified")


def paired(results: dict[str, dict[str, bool]], a: str, b: str) -> dict:
    """results[case][reader] -> success. Wins/losses/ties of reader a against reader b."""
    w = l = t = 0
    for r in results.values():
        if a in r and b in r:
            if r[a] and not r[b]:
                w += 1
            elif r[b] and not r[a]:
                l += 1
            else:
                t += 1
    return {"wins": w, "losses": l, "ties": t, "n": w + l + t}
