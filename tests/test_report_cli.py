from pathlib import Path

from typer.testing import CliRunner

from autodspark import report
from autodspark.cli import app
from autodspark.hardware import simulated
from autodspark.model_info import inspect_model
from autodspark.planner import Plan, make_plan

runner = CliRunner()

EVAL = {
    "baseline": {
        "num_requests": 40,
        "tokens_per_second": 52.1,
        "by_category": {"code": 50.0, "chat": 54.0},
    },
    "dspark": {
        "num_requests": 40,
        "tokens_per_second": 118.4,
        "acceptance_length": 3.41,
        "acceptance_rate": 0.34,
        "by_category": {"code": 131.0, "chat": 109.0},
    },
    "compare": {
        "speedup": 2.27,
        "acceptance_length": 3.41,
        "acceptance_rate": 0.34,
        "by_category": {
            "code": {"baseline_tps": 50.0, "dspark_tps": 131.0},
            "chat": {"baseline_tps": 54.0, "dspark_tps": 109.0},
        },
    },
}


def test_report_render(model_dir):
    plan = make_plan(inspect_model(str(model_dir("qwen3-0.6b"))), simulated("1x24"), "quick")
    text = report.render(plan, Path("/runs/x/checkpoints/checkpoint_best"), EVAL)
    assert "**2.27x faster**" in text
    assert "| code | 50.0 | 131.0 | 2.62x |" in text
    assert '"model": "/runs/x/checkpoints/checkpoint_best"' in text
    assert "target layers [2, 14, 25]" in text


def test_report_without_eval(model_dir):
    plan = make_plan(inspect_model(str(model_dir("qwen3-0.6b"))), simulated("1x24"), "quick")
    text = report.render(plan, None, None)
    assert "## Training setup" in text
    assert "## Result" not in text


def test_cli_plan_and_status(model_dir, tmp_path):
    model = model_dir("llama-3.2-1b")
    workdir = tmp_path / "run"
    r = runner.invoke(
        app, ["plan", "--model", str(model), "--gpus", "2x24", "--workdir", str(workdir)]
    )
    assert r.exit_code == 0, r.output
    assert (workdir / "plan.yaml").exists()
    assert Plan.load(workdir / "plan.yaml").layout.train_gpus == [1]

    r = runner.invoke(app, ["status", "--workdir", str(workdir)])
    assert r.exit_code == 0, r.output
    assert "pending" in r.output


def test_cli_dry_run_from_workdir(model_dir, tmp_path):
    model = model_dir("qwen3-0.6b")
    workdir = tmp_path / "run"
    runner.invoke(app, ["plan", "--model", str(model), "--gpus", "1x80", "--workdir", str(workdir)])
    r = runner.invoke(app, ["run", "--workdir", str(workdir), "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "speculator-type dspark" in r.output.replace("\n", " ")


def test_cli_gpus_flag_needs_dry_run(model_dir, tmp_path):
    r = runner.invoke(app, ["run", "--model", str(model_dir("qwen3-0.6b")), "--gpus", "1x24"])
    assert r.exit_code == 2


def test_cli_bad_model(tmp_path):
    (tmp_path / "empty").mkdir()
    r = runner.invoke(app, ["plan", "--model", str(tmp_path / "empty"), "--gpus", "1x24"])
    assert r.exit_code == 2
    assert "config.json" in r.output


def test_cli_serve_without_drafter(model_dir, tmp_path):
    workdir = tmp_path / "run"
    runner.invoke(
        app,
        [
            "plan",
            "--model",
            str(model_dir("qwen3-0.6b")),
            "--gpus",
            "1x24",
            "--workdir",
            str(workdir),
        ],
    )
    r = runner.invoke(app, ["serve", "--workdir", str(workdir)])
    assert r.exit_code == 2
    assert "No trained drafter" in r.output


def test_cli_run_keeps_edited_plan(model_dir, tmp_path):
    model = model_dir("qwen3-0.6b")
    workdir = tmp_path / "run"
    runner.invoke(app, ["plan", "--model", str(model), "--gpus", "1x80", "--workdir", str(workdir)])
    p = Plan.load(workdir / "plan.yaml")
    p.train.epochs = 9
    p.save(workdir / "plan.yaml")
    r = runner.invoke(
        app,
        ["run", "--model", str(model), "--workdir", str(workdir), "--dry-run", "--samples", "5"],
    )
    assert r.exit_code == 0, r.output
    assert "ignored" in r.output
    assert "--epochs 9" in r.output.replace("\n", " ")
    assert Plan.load(workdir / "plan.yaml").train.epochs == 9


def test_cli_rejects_broken_plan_edit(model_dir, tmp_path):
    model = model_dir("qwen3-0.6b")
    workdir = tmp_path / "run"
    runner.invoke(app, ["plan", "--model", str(model), "--gpus", "1x80", "--workdir", str(workdir)])
    p = Plan.load(workdir / "plan.yaml")
    p.draft.target_layer_ids = [2, 28]
    p.save(workdir / "plan.yaml")
    r = runner.invoke(app, ["run", "--workdir", str(workdir), "--yes"])
    assert r.exit_code == 2
    assert "target_layer_ids" in r.output


def test_report_explains_a_slow_quick_run(model_dir):
    plan = make_plan(inspect_model(str(model_dir("qwen3-0.6b"))), simulated("1x24"), "quick")
    slow = {**EVAL, "compare": {**EVAL["compare"], "speedup": 0.88}}
    text = report.render(plan, Path("/ckpt"), slow)
    assert "This was a `quick` run" in text
    assert "doesn't speed generation up yet" in text
    assert "Small target models" in text


def test_report_headline_for_slowdown(model_dir):
    plan = make_plan(inspect_model(str(model_dir("qwen3-0.6b"))), simulated("1x24"), "quick")
    slow = {**EVAL, "compare": {**EVAL["compare"], "speedup": 0.876}}
    text = report.render(plan, Path("/ckpt"), slow)
    assert "**0.88x the baseline speed**" in text
    assert "faster**" not in text
