# help-answer-eval

A diagnostic for support answers built from a help center. For a few decisions the published pages require, it shows where an answering setup loses a condition, whether the setup failed to find the advice or found it and misread it, and whether a small change restores it without breaking a paired case.

It reads the FAQ blocks of a WordPress site (Yoast FAQ markup) through the public REST API, once, into a local snapshot, then checks every extracted answer against the text of the page as a visitor sees it. Nothing from any site is stored in this repository; the tests run on synthetic pages.

## Three setups, one model, one instruction

Every setup gets the same model, the same instruction and the same operator notes about the sources (which page is an older version, which conflicts are unresolved). Only the evidence differs.

- **A, whole site**: every entry of every page, with page URL and last-updated date.
- **B, passage search**: sentence windows tagged with their page and entry, hybrid retrieval (BM25 written here plus local embeddings `BAAI/bge-small-en-v1.5`, reciprocal rank fusion), listwise reranking of the top 40 candidates by a small model (`claude-haiku-4-5` in the published run; a local cross-encoder is also supported), each hit widened by its neighbouring window, filled to a word budget.
- **C, whole-answer search**: complete entries, same retrieval and reranking, same word budget as B.
- **O, control**: the same reader given the hand-picked source entries. It separates "not found" from "found but misused".

B and C spend the same evidence budget, counted on the text actually rendered. A reads everything and is reported apart. `comparator_version: 1` in the config reproduces an earlier implementation (reranker exclusions not enforced, passage headers charged per window) for replaying recorded runs.

## Three experiments

- **apply**: the sources stay in the corpus, conflicting versions included; only the customer email is new.
- **missing**: the email leaves out a detail the answer depends on; asking or answering conditionally is the right move.
- **transfer**: an entry is held out together with every near copy (same block on another page, similar question, overlapping text); each held-out email is labeled answerable or not from what is left, and the label is kept only when its supporting quote is found verbatim in the remaining corpus.

## Scoring, frozen before any run

Each case carries a written expectation: what a correct reply does, what it must not do, the conditions it must keep, and whether asking counts as success. A separate model grades each reply blind (it never sees which setup wrote it) into one outcome: correct, unsupported, lost condition, clarified, needless escalation, incomplete. Tone is not scored. Results are raw counts and paired wins, losses and ties per case, never percentages.

## Run

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q          # offline: synthetic pages, fake embedder, fake reranker, fake model
```

A real run needs a config (see `example.config.json`), a cases file you write, and the `claude` CLI logged in. Each model call is a `claude -p` run with no tools, no settings files and no project files, cached on disk by the hash of its inputs, so every raw answer and grade stays inspectable.

```
.venv/bin/python -m hae.bench fetch   my.config.json
.venv/bin/python -m hae.verify <snapshot.json>     # extracted answers against the displayed pages
.venv/bin/python -m hae.bench prepare my.config.json
.venv/bin/python -m hae.bench run     my.config.json
.venv/bin/python -m hae.bench report  my.config.json
```

## What it does not show

- A help center of a few hundred entries fits in any current context window. Nothing here says how the setups compare on a larger archive of real support conversations.
- The expectations are the tester's reading of the published pages, not the business owner's verdict.
- The grader is a model. Focal cases are meant to be inspected by a person; model review of a model is not human calibration.
- The CLI exposes no temperature setting. `llm.call(..., sample=n)` draws an independent answer to the same inputs when repeated samples are needed.
