"""One model call = one `claude -p` run with no tools, no project files, no memory.

Every call is cached on disk by the hash of (model, system, prompt), so a rerun only pays
for what changed, and the raw output of every answer stays inspectable.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import threading
from pathlib import Path

CLAUDE = os.environ.get("HAE_CLAUDE_BIN", "claude")


def key(model: str, system: str, prompt: str) -> str:
    return hashlib.sha256(json.dumps([model, system, prompt]).encode()).hexdigest()[:20]


def call(model: str, system: str, prompt: str, cache_dir: Path, *, timeout: int = 600, runner=None) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    k = key(model, system, prompt)
    path = cache_dir / f"{k}.json"
    if path.exists():
        return json.loads(path.read_text())
    if runner is not None:  # tests inject a fake model here
        text = runner(model, system, prompt)
        rec = {"model": model, "result": text, "usage": {}, "cost_usd_list": 0.0}
    else:
        cmd = [CLAUDE, "-p", "--safe-mode", "--setting-sources", "local", "--tools", "", "--strict-mcp-config",
               "--no-session-persistence", "--model", model, "--system-prompt", system,
               "--output-format", "json"]
        with tempfile.TemporaryDirectory() as cwd:  # no project instructions can be picked up
            out = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout, cwd=cwd)
        if out.returncode != 0:
            raise RuntimeError(f"claude -p failed ({out.returncode}): {out.stderr[-500:]}")
        data = json.loads(out.stdout)
        if data.get("is_error"):
            raise RuntimeError(f"claude -p error: {str(data)[:500]}")
        rec = {"model": model, "result": data.get("result", ""), "usage": data.get("usage", {}),
               "models_used": list((data.get("modelUsage") or {}).keys()),
               "cost_usd_list": data.get("total_cost_usd")}
    rec["key"] = k
    tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    os.replace(tmp, path)  # atomic: a parallel reader never sees a half-written file
    return rec
