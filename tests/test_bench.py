"""Offline tests: synthetic pages, a fake embedder and a fake model. No network, no key."""
import json
import re

import numpy as np
import pytest

from hae import bench, grade, systems
from hae.corpus import clean, extract_faq
from hae.leak import containment, is_variant, without_variants
from hae.retrieval import BM25, Dense, chunk_words, rrf, top_k


def faq_page(pid, link, modified, items):
    blocks = "".join(
        f'<div class="schema-faq-section" id="{b}"><strong class="schema-faq-question">{q}</strong> </p>'
        f'<p class="schema-faq-answer">{a}</p></p></div>' for b, q, a in items)
    return {"id": pid, "link": link, "modified": modified,
            "content": {"rendered": f'<div class="schema-faq wp-block-yoast-faq-block">{blocks}</div>'}}


OLD = "https://bakery.example/help-old/"
NEW = "https://bakery.example/help/"
TOPIC = "https://bakery.example/care/"

PAGES = [
    faq_page(1, OLD, "2023-01-12T00:00:00", [
        ("faq-1", "When will my cake be ready?", "Custom cake orders are ready within 24 hours after the baker's final check."),
        ("faq-2", "How do I clean the fretboard?", "Use a soft dry cloth and a drop of lemon oil once a year, never water."),
    ]),
    faq_page(2, NEW, "2026-08-14T00:00:00", [
        ("faq-1", "How custom orders work", "Custom cake orders are ready within 48 hours after the baker's final check."),
        ("faq-3", "Delivery area", "We deliver within ten miles of the bakery; farther orders are picked up at the counter."),
        ("faq-4", "Storing bread", "Keep the loaf in a paper bag at room temperature and freeze sliced bread for longer storage."),
    ]),
    faq_page(3, TOPIC, "2026-01-06T00:00:00", [
        ("faq-9", "Keeping bread fresh", "Keep the loaf in a paper bag at room temperature and freeze sliced bread for longer storage in summer."),
        ("faq-9", "A different entry sharing a block id", "A cracked crust on sourdough is normal and means a good bake."),
    ]),
]


class HashEmbedder:
    """Deterministic bag-of-words vectors, enough to exercise ranking code."""

    def embed(self, texts):
        for t in texts:
            v = np.zeros(64, dtype="float32")
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                v[hash(w) % 64] += 1
            yield v


@pytest.fixture
def entries():
    return extract_faq(PAGES)


def test_extract_keeps_url_date_and_full_answer(entries):
    assert len(entries) == 7
    e = entries[0]
    assert e.url == OLD and e.modified.startswith("2023") and e.answer.endswith("check.")
    ids = [x.id for x in entries]
    assert len(set(ids)) == len(ids), "duplicate block ids inside one page must still get unique ids"


def test_clean_unescapes_and_strips_tags():
    assert clean("It&#8217;s <a href='x'>here</a><br>next") == "It’s here\nnext"


def test_variant_detection_removes_copies_and_old_versions(entries):
    held = next(e for e in entries if e.question == "Storing bread")
    kept, removed = without_variants(entries, held)
    removed_q = {e.question for e in removed}
    assert "Keeping bread fresh" in removed_q, "near copy on another page must leave with the held-out entry"
    assert "Delivery area" not in removed_q
    held_ship = next(e for e in entries if e.question == "How custom orders work")
    _, removed = without_variants(entries, held_ship)
    assert {e.url for e in removed} == {OLD, NEW}, "same block id on an older page is a variant"


def test_same_block_id_inside_one_page_is_not_a_variant(entries):
    a, b = [e for e in entries if e.url == TOPIC]
    assert not is_variant(b, a)


def test_containment():
    assert containment("a b c d e f g h i j", "x a b c d e f g h i j y") == 1.0
    assert containment("short", "short") == 0.0


def test_bm25_and_fusion():
    docs = ["ready within 48 hours", "paper bag room temperature", "delivery ten miles"]
    s = BM25(docs).scores("when is it ready")
    assert top_k(s, 1) == [0]
    assert rrf([[0, 1, 2], [1, 0, 2]])[:2] in ([0, 1], [1, 0])
    assert rrf([[2], [2, 0]])[0] == 2


def test_chunking_overlaps():
    text = " ".join(str(i) for i in range(500))
    chunks = chunk_words(text, 200, 40)
    assert chunks[0].split()[-40:] == chunks[1].split()[:40]
    assert chunks[-1].split()[-1] == "499"


class FakeReranker(systems.Reranker):
    def __init__(self):
        super().__init__(scorer=lambda q, ds: [len(set(re.findall(r"[a-z0-9]+", q.lower())) & set(re.findall(r"[a-z0-9]+", d.lower()))) for d in ds])


def test_three_readers(entries):
    dense, rr = Dense(embedder=HashEmbedder()), FakeReranker()
    a = systems.context_full(entries)
    assert a.index(OLD) < a.index(NEW), "full corpus lists pages oldest first"
    assert "page updated 2023-01-12" in a
    b = systems.PassageReader(entries, dense, rr, budget=40).context("when will my cake be ready")
    c = systems.AnswerReader(entries, dense, rr, budget=40).context("when will my cake be ready")
    assert "ready" in b and "ready" in c
    assert len(b.split()) <= 40 + 40 and len(c.split()) <= 40, "both retrievers respect the evidence budget"
    assert "page updated" in b and "page updated" in c, "both carry the same source metadata"


def test_windows_keep_every_sentence():
    text = "One. Two is here. Three? Four! Five."
    ws = systems.windows(text, 3, 2)
    joined = " ".join(ws)
    for s in ("One.", "Two is here.", "Three?", "Four!", "Five."):
        assert s in joined


def test_grade_parse_success_and_pairs():
    g = grade.parse('x {"outcome": "clarified", "evidence": "it depends", "keeps": ["a","b","c"], "loses": []} y')
    assert g["keeps"] == ["a", "b"]
    assert grade.success("clarified", True) and not grade.success("clarified", False)
    assert grade.success("correct", False) and not grade.success("lost condition", True)
    with pytest.raises(ValueError):
        grade.parse('{"outcome": "great"}')
    p = grade.paired({"c1": {"A": True, "B": False}, "c2": {"A": True, "B": True}, "c3": {"A": False, "B": True}}, "A", "B")
    assert p == {"wins": 1, "losses": 1, "ties": 1, "n": 3}


def test_grader_prompt_is_blind():
    p = grade.grader_prompt("q", {"does": "x"}, "src", "draft")
    for word in ("full corpus", "passage reader", "whole answer", "reader a", "oracle"):
        assert word not in p.lower()


def test_end_to_end_with_fake_model(tmp_path):
    (tmp_path / "pages.json").write_text(json.dumps(PAGES))
    (tmp_path / "cases.json").write_text(json.dumps({"notes": "old page is older", "cases": [{
        "id": "ready", "experiment": "apply", "pair": None, "question": "When will my custom cake be ready?",
        "expect": {"does": "48 hours", "must_not": "24 hours", "conditions": "", "clarify_ok": False},
        "sources": ["2:faq-1", "1:faq-1"]}]}))
    cfg_data = {"site": "https://bakery.example", "pages": "pages.json", "escalate_to": "the owner",
                "answer_model": "fake-model", "grader_model": "fake-grader", "cases": "cases.json",
                "out": "runs", "workers": 2, "budget_words": 60,
                "transfer": {"n": 2, "seed": 1, "min_words": 5, "pool_urls": [NEW]}}
    (tmp_path / "cfg.json").write_text(json.dumps(cfg_data))
    cfg = bench.Cfg(tmp_path / "cfg.json")
    seen = []

    def fake(model, system, prompt):
        seen.append((model, prompt))
        if model == "fake-grader" and "map" in prompt and "id:" in prompt:
            ids = re.findall(r"^id: (\S+)", prompt, re.M)
            return json.dumps({i: f"Customer asks about {i}?" for i in ids})
        if model == "fake-grader" and "Does the reference contain" in prompt:
            quote = re.search(r"\]\n([^\n]{30,})", prompt).group(1)
            return json.dumps({"answerable": True, "quotes": [quote]})
        if model == "fake-grader":
            draft = prompt.split("Draft reply:")[1]
            return json.dumps({"outcome": "unsupported" if "24 hours" in draft else "correct", "evidence": "", "keeps": [], "loses": []})
        ref = prompt.split("<reference>")[1].split("</reference>")[0]
        return "It is ready within 48 hours." if "48 hours" in ref else "It is ready within 24 hours."

    rows = bench.prepare(cfg, runner=fake)
    assert len(rows) == 2 and all(r["removed_variants"] for r in rows)
    graded = bench.run(cfg, runner=fake, dense=Dense(embedder=HashEmbedder()), reranker=FakeReranker())
    assert len(graded) == 4 + 2 * 3, "apply case: A, B, C and the oracle; transfer: A, B, C"
    for r in rows:  # the held-out answer never reaches a reader answering its own question
        for model, prompt in seen:
            if model == "fake-model" and r["question"] in prompt:
                assert r["expert"] not in prompt
    out = bench.report(cfg)
    assert out["summary"]["apply"]["O"]["success"] == 1
    assert out["paired"]["apply"]["A vs B"]["n"] == 1


def test_regrade_and_intervention_only_touch_what_they_declare(tmp_path):
    test_end_to_end_with_fake_model(tmp_path)
    cfg_path = tmp_path / "cfg.json"
    data = json.loads(cfg_path.read_text())
    data["intervention"] = {"cases": ["ready"], "notes_append": "EXTRA NOTE"}
    cfg_path.write_text(json.dumps(data))
    cfg = bench.Cfg(cfg_path)
    prompts = []

    def fake(model, system, prompt):
        prompts.append(prompt)
        if model == "fake-grader":
            return json.dumps({"outcome": "correct", "evidence": "", "keeps": [], "loses": []})
        return "It is ready within 48 hours."

    before = (tmp_path / "runs" / "graded.json").read_text()
    kw = dict(runner=fake, dense=Dense(embedder=HashEmbedder()), reranker=FakeReranker())
    out = bench.regrade(cfg, **kw)
    assert len(out) == 10 and all(r["graded_with"] == "whole site" for r in out)
    assert (tmp_path / "runs" / "graded.json").read_text() == before, "regrade never rewrites the first grading"
    assert any("<site>" in p for p in prompts)
    rows = bench.intervene(cfg, **kw)
    assert {r["case"] for r in rows} == {"ready"} and len(rows) == 4
    assert any("EXTRA NOTE" in p and "Customer email" in p for p in prompts)
