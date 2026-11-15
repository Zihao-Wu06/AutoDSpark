import json

from autodspark import bench

METRICS = """# HELP vllm:spec_decode_num_drafts_total Number of spec decoding drafts.
# TYPE vllm:spec_decode_num_drafts_total counter
vllm:spec_decode_num_drafts_total{engine="0",model_name="target"} 100.0
vllm:spec_decode_num_drafts_total{engine="1",model_name="target"} 50.0
vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="target"} 1200.0
vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="target"} 360.0
vllm:num_requests_running{engine="0"} 1.0
"""


def test_parse_spec_metrics():
    m = bench.parse_spec_metrics(METRICS)
    assert m == {"drafts": 150.0, "draft_tokens": 1200.0, "accepted": 360.0}


def test_parse_spec_metrics_absent():
    assert bench.parse_spec_metrics("vllm:num_requests_running 0\n") == {
        "drafts": 0.0,
        "draft_tokens": 0.0,
        "accepted": 0.0,
    }


def test_builtin_prompts():
    rows = bench.load_prompts(None, 100)
    assert len(rows) == 40
    assert {r["category"] for r in rows} >= {"code", "math", "zh", "chat"}
    assert all(r["messages"][-1]["role"] == "user" for r in rows)


def test_user_prompts_strip_trailing_answer(tmp_path):
    f = tmp_path / "p.jsonl"
    f.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ]
            }
        )
        + "\n"
        + json.dumps({"prompt": "second", "category": "x"})
        + "\n"
        + json.dumps({"nothing": True})
        + "\n"
    )
    rows = bench.load_prompts(str(f), 10)
    assert len(rows) == 2
    assert rows[0]["messages"] == [{"role": "user", "content": "hi"}]
    assert rows[1]["category"] == "x"


def test_compare():
    base = {"tokens_per_second": 50.0, "by_category": {"code": 50.0}}
    spec = {
        "tokens_per_second": 120.0,
        "acceptance_length": 3.4,
        "acceptance_rate": 0.3,
        "by_category": {"code": 130.0},
    }
    c = bench.compare(base, spec)
    assert c["speedup"] == 2.4
    assert c["acceptance_length"] == 3.4
    assert c["by_category"]["code"] == {"baseline_tps": 50.0, "dspark_tps": 130.0}


def _fake_server(reject_if=lambda body: False, drafts=(0, 0, 0)):
    import httpx

    calls = {"metrics": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/metrics":
            calls["metrics"] += 1
            n = calls["metrics"] - 1  # before = 0, after = drafts
            d, t, a = (x * n for x in drafts)
            return httpx.Response(
                200,
                text=f"vllm:spec_decode_num_drafts_total {d}\n"
                f"vllm:spec_decode_num_draft_tokens_total {t}\n"
                f"vllm:spec_decode_num_accepted_tokens_total {a}\n",
            )
        body = json.loads(request.content)
        if reject_if(body):
            return httpx.Response(400, text="prompt too long")
        return httpx.Response(200, json={"usage": {"completion_tokens": 100}})

    return httpx.MockTransport(handler)


def test_run_bench_skips_rejected_prompts():
    prompts = [
        {"messages": [{"role": "user", "content": "ok"}], "category": "a"},
        {"messages": [{"role": "user", "content": "x" * 50}], "category": "b"},
        {"messages": [{"role": "user", "content": "fine"}], "category": "a"},
    ]
    long = lambda body: len(body["messages"][0]["content"]) > 20  # noqa: E731
    r = bench.run_bench(
        "http://127.0.0.1:1",
        prompts,
        max_tokens=64,
        temperature=0,
        warmup=1,
        transport=_fake_server(long, drafts=(10, 80, 30)),
    )
    assert r["num_requests"] == 2
    assert r["failed"] == 1
    assert r["output_tokens"] == 200
    assert r["acceptance_length"] == 4.0  # 1 + 30 / 10
    assert set(r["by_category"]) == {"a"}
    assert r["category_totals"]["a"]["tokens"] == 200
    assert r["category_totals"]["a"]["seconds"] >= 0


def test_run_bench_all_rejected():
    import pytest

    prompts = [{"messages": [{"role": "user", "content": "x"}], "category": "a"}]
    with pytest.raises(bench.BenchError):
        bench.run_bench(
            "http://127.0.0.1:1",
            prompts,
            max_tokens=8,
            temperature=0,
            transport=_fake_server(lambda body: True),
        )
