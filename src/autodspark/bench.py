"""Measure single-stream speed with and without the drafter.

Speculative decoding helps most when a GPU serves few requests at once, so
the benchmark sends one request at a time and reports output tokens/s.
Acceptance numbers come from vLLM's Prometheus counters.
"""

from __future__ import annotations

import contextlib
import json
import re
import time
from importlib import resources
from pathlib import Path

import httpx

_SPEC_COUNTERS = {
    "drafts": "vllm:spec_decode_num_drafts",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens",
    "accepted": "vllm:spec_decode_num_accepted_tokens",
}
_METRIC_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+eE0-9.]+|NaN)")


def load_prompts(path: str | None, limit: int) -> list[dict]:
    """Rows with `messages` (OpenAI format) or `prompt`, plus an optional `category`."""
    if path:
        text = Path(path).read_text(encoding="utf-8")
    else:
        text = resources.files("autodspark").joinpath("eval_prompts.jsonl").read_text("utf-8")
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "messages" not in row:
            content = row.get("prompt") or row.get("question") or row.get("input")
            if not content:
                continue
            row["messages"] = [{"role": "user", "content": content}]
        # Drop any assistant answer at the end so the model writes its own.
        msgs = list(row["messages"])
        while msgs and msgs[-1].get("role") == "assistant":
            msgs.pop()
        if not msgs:
            continue
        rows.append({"messages": msgs, "category": row.get("category", "general")})
        if len(rows) >= limit:
            break
    return rows


def parse_spec_metrics(text: str) -> dict[str, float]:
    """Sum vLLM speculative-decoding counters across label sets."""
    totals = dict.fromkeys(_SPEC_COUNTERS, 0.0)
    for line in text.splitlines():
        m = _METRIC_LINE.match(line)
        if not m:
            continue
        name, value = m.group(1), m.group(3)
        for key, prefix in _SPEC_COUNTERS.items():
            if name in (prefix, f"{prefix}_total"):
                with contextlib.suppress(ValueError):
                    totals[key] += float(value)
    return totals


def _fetch_metrics(client: httpx.Client, url: str) -> dict[str, float]:
    try:
        return parse_spec_metrics(client.get(f"{url}/metrics", timeout=10).text)
    except httpx.HTTPError:
        return dict.fromkeys(_SPEC_COUNTERS, 0.0)


class BenchError(RuntimeError):
    pass


def run_bench(
    url: str,
    prompts: list[dict],
    *,
    max_tokens: int,
    temperature: float,
    model_name: str = "target",
    warmup: int = 2,
    timeout: float = 600,
    transport: httpx.BaseTransport | None = None,  # for tests
) -> dict:
    failed: list[str] = []
    # trust_env=False: requests to the local server must never go through a proxy.
    with httpx.Client(timeout=timeout, trust_env=False, transport=transport) as client:

        def ask(messages: list[dict]) -> tuple[int, float] | None:
            """None when the server rejects this prompt (e.g. longer than the context)."""
            t0 = time.perf_counter()
            r = client.post(
                f"{url}/v1/chat/completions",
                json={
                    "model": model_name,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
            if 400 <= r.status_code < 500:
                failed.append(r.text[:200])
                return None
            r.raise_for_status()
            elapsed = time.perf_counter() - t0
            return int(r.json()["usage"]["completion_tokens"]), elapsed

        for p in prompts[:warmup]:
            ask(p["messages"])
        failed.clear()

        before = _fetch_metrics(client, url)
        per_request, by_cat = [], {}
        for p in prompts:
            answer = ask(p["messages"])
            if answer is None:
                continue
            tokens, secs = answer
            per_request.append({"category": p["category"], "tokens": tokens, "seconds": secs})
            cat = by_cat.setdefault(p["category"], {"tokens": 0, "seconds": 0.0})
            cat["tokens"] += tokens
            cat["seconds"] += secs
        after = _fetch_metrics(client, url)

    if not per_request:
        raise BenchError(f"every prompt was rejected by the server; first error: {failed[:1]}")
    total_tokens = sum(r["tokens"] for r in per_request)
    total_secs = sum(r["seconds"] for r in per_request)
    delta = {k: after[k] - before[k] for k in after}
    result = {
        "num_requests": len(per_request),
        "failed": len(failed),
        "output_tokens": total_tokens,
        "seconds": round(total_secs, 2),
        "tokens_per_second": round(total_tokens / total_secs, 2) if total_secs else 0.0,
        "by_category": {
            c: round(v["tokens"] / v["seconds"], 2) if v["seconds"] else 0.0
            for c, v in by_cat.items()
        },
        # Raw sums, so an overall speed for any subset of categories can be
        # recomputed later (sum of tokens / sum of seconds, not a mean of speeds).
        "category_totals": {
            c: {"tokens": v["tokens"], "seconds": round(v["seconds"], 3)} for c, v in by_cat.items()
        },
    }
    if delta["drafts"] > 0:
        result["acceptance_length"] = round(1 + delta["accepted"] / delta["drafts"], 3)
        result["acceptance_rate"] = (
            round(delta["accepted"] / delta["draft_tokens"], 3) if delta["draft_tokens"] else None
        )
    return result


def compare(baseline: dict, dspark: dict) -> dict:
    base_tps = baseline.get("tokens_per_second") or 0
    spec_tps = dspark.get("tokens_per_second") or 0
    cats = sorted(set(baseline.get("by_category", {})) | set(dspark.get("by_category", {})))
    return {
        "speedup": round(spec_tps / base_tps, 3) if base_tps else None,
        "acceptance_length": dspark.get("acceptance_length"),
        "acceptance_rate": dspark.get("acceptance_rate"),
        "by_category": {
            c: {
                "baseline_tps": baseline.get("by_category", {}).get(c),
                "dspark_tps": dspark.get("by_category", {}).get(c),
            }
            for c in cats
        },
    }
