import hashlib
import json
import socket
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from automl.cli import main

FAMILY = "hist_gradient_boosting"


def _toml(tmp_path: Path, extra: str = "") -> str:
    return f"""\
version = 1
seed = 7

[data]
train = "train.csv"
test = "test.csv"
sample_submission = "sample_submission.csv"
target = "y"
id_columns = ["id"]

[task]
task = "binary"
metric = "roc_auc"

[validation]
n_folds = 3
{extra}"""


@pytest.fixture
def project(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    n, m = 90, 30
    a = rng.normal(size=n + m)
    b = rng.choice(["x", "y", "z"], n + m)
    y = ((a + 0.8 * (b == "x") + rng.normal(scale=0.5, size=n + m)) > 0.4).astype(int)
    frame = pd.DataFrame({"id": np.arange(n + m), "a": a, "b": b, "y": y})
    frame.iloc[:n].to_csv(tmp_path / "train.csv", index=False)
    frame.iloc[n:].drop(columns="y").to_csv(tmp_path / "test.csv", index=False)
    pd.DataFrame({"id": np.arange(n, n + m), "y": 0.0}).to_csv(tmp_path / "sample_submission.csv", index=False)
    (tmp_path / "run.toml").write_text(_toml(tmp_path))
    return tmp_path


def cli(argv, capsys):
    code = main([str(a) for a in argv])
    out, err = capsys.readouterr()
    return code, (json.loads(out) if out.strip() else None), err


def _digest(directory: Path, names=("train.csv", "test.csv", "sample_submission.csv", "run.toml")) -> dict[str, str]:
    return {n: hashlib.sha256((directory / n).read_bytes()).hexdigest() for n in names}


@pytest.fixture
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


def _config(project: Path) -> str:
    return str(project / "run.toml")


# ---- baseline ----


def test_help_lists_every_command(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    out = capsys.readouterr().out

    assert exc.value.code == 0
    for name in ("validate", "split", "eda", "report", "baseline", "tune", "model-report", "refit", "submit", "run"):
        assert name in out


def test_baseline_writes_result_and_refuses_overwrite(project, capsys, tmp_path_factory):
    out = project / "runs" / "baseline"
    code, payload, _ = cli(["baseline", _config(project), "--output-dir", out], capsys)

    assert code == 0
    assert payload["status"] == "created"
    assert payload["splits"]["source"] == "config"
    assert np.isfinite(payload["mean"])
    assert (out / "baseline.json").is_file()
    assert payload["data"]["train"] == "train.csv"

    code, payload, err = cli(["baseline", _config(project), "--output-dir", out], capsys)
    assert code == 1 and payload is None
    assert "already exists" in err


def test_saved_splits_give_the_same_scores_as_config_splits(project, capsys):
    cli(["split", _config(project), "--output-dir", project / "splits"], capsys)
    _, pinned, _ = cli(
        ["baseline", _config(project), "--output-dir", project / "a", "--splits", project / "splits" / "splits.json"],
        capsys,
    )
    _, built, _ = cli(["baseline", _config(project), "--output-dir", project / "b"], capsys)

    assert pinned["splits"]["source"].endswith("splits.json")
    assert pinned["mean"] == built["mean"]


def test_bad_splits_file_names_the_split_command(project, capsys):
    code, _, err = cli(
        ["baseline", _config(project), "--output-dir", project / "a", "--splits", project / "missing.json"], capsys
    )

    assert code == 1
    assert "automl split" in err
    assert not (project / "a").exists()


def test_splits_for_other_data_are_rejected(project, capsys):
    cli(["split", _config(project), "--output-dir", project / "splits"], capsys)
    train = pd.read_csv(project / "train.csv")
    train.iloc[:60].to_csv(project / "train.csv", index=False)

    code, _, err = cli(
        ["baseline", _config(project), "--output-dir", project / "a", "--splits", project / "splits" / "splits.json"],
        capsys,
    )

    assert code == 1
    assert "covers 90 rows but the train data has 60" in err


def test_unknown_estimator_is_an_error(project, capsys):
    code, _, err = cli(["baseline", _config(project), "--output-dir", project / "a", "--estimator", "nope"], capsys)

    assert code == 1
    assert "unsupported value" in err


# ---- tuning ----


def test_tune_needs_a_budget_and_writes_nothing(project, capsys):
    pytest.importorskip("optuna")
    code, _, err = cli(["tune", _config(project), "--output-dir", project / "t", "--families", FAMILY], capsys)

    assert code == 1
    assert "budget" in err
    assert not (project / "t").exists()


def test_tune_families_then_resume(project, capsys):
    pytest.importorskip("optuna")
    out = project / "t"
    code, payload, err = cli(
        ["tune", _config(project), "--output-dir", out, "--families", FAMILY, "--n-trials", 2], capsys
    )

    assert code == 0
    assert payload["kind"] == "families" and payload["best_family"] == FAMILY
    assert (out / "tuning_families.json").is_file()
    assert "n_trials=2" in err

    code, payload, _ = cli(
        ["tune", _config(project), "--output-dir", out, "--n-trials", 1, "--resume"], capsys
    )
    assert code == 0 and payload["status"] == "resumed"

    code, _, err = cli(["tune", _config(project), "--output-dir", out, "--n-trials", 1], capsys)
    assert code == 1 and "--resume" in err


def test_tune_single_estimator(project, capsys):
    pytest.importorskip("optuna")
    out = project / "t"
    code, payload, _ = cli(
        ["tune", _config(project), "--output-dir", out, "--estimator", FAMILY, "--n-trials", 2], capsys
    )

    assert code == 0
    assert payload["kind"] == "tuning" and payload["n_trials"] == 2
    assert payload["best"]["score"] is not None
    assert (out / "tuning.json").is_file()


def test_tune_rejects_estimator_with_families(project, capsys):
    pytest.importorskip("optuna")
    code, _, err = cli(
        ["tune", _config(project), "--output-dir", project / "t", "--estimator", FAMILY, "--families", FAMILY,
         "--n-trials", 1],
        capsys,
    )

    assert code == 1 and "not both" in err


def test_resume_needs_an_existing_study(project, capsys):
    pytest.importorskip("optuna")
    (project / "t").mkdir()
    code, _, err = cli(["tune", _config(project), "--output-dir", project / "t", "--resume", "--n-trials", 1], capsys)

    assert code == 1 and "--resume needs an existing tuning directory" in err


# ---- full path ----


def _require_extras():
    pytest.importorskip("optuna")
    pytest.importorskip("matplotlib")


def test_stages_chain_from_csv_to_submission_without_network(project, capsys, no_network):
    _require_extras()
    config, runs = _config(project), project / "runs"
    before = _digest(project)

    steps = [
        ["validate", config],
        ["split", config, "--output-dir", runs / "splits"],
        ["eda", config, "--output-dir", runs / "eda"],
        ["report", config, runs / "eda" / "eda.json", "--output-dir", runs / "reports"],
        ["baseline", config, "--output-dir", runs / "baseline", "--splits", runs / "splits" / "splits.json"],
        ["tune", config, "--output-dir", runs / "tuning", "--splits", runs / "splits" / "splits.json",
         "--families", FAMILY, "--n-trials", 2],
        ["model-report", config, runs / "baseline", runs / "tuning", "--output-dir", runs / "reports"],
        ["refit", config, runs / "tuning", "--output-dir", runs / "refit"],
        ["submit", config, runs / "refit", "--output-dir", runs / "submission"],
    ]
    results = []
    for step in steps:
        code, payload, err = cli(step, capsys)
        assert code == 0, f"{step[0]} failed: {err}"
        results.append(payload)

    assert (runs / "reports" / "eda_report.pdf").stat().st_size > 0
    assert (runs / "reports" / "model_report.pdf").stat().st_size > 0
    assert {c["label"] for c in results[6]["candidates"]} == {f"baseline:{FAMILY}", f"family:{FAMILY}"}
    submission = pd.read_csv(runs / "submission" / "submission.csv")
    sample = pd.read_csv(project / "sample_submission.csv")
    test = pd.read_csv(project / "test.csv")
    assert list(submission.columns) == list(sample.columns)
    assert submission["id"].tolist() == test["id"].tolist()
    assert submission["y"].between(0, 1).all() and submission["y"].notna().all()
    assert results[-1]["schema_source"] == "sample"
    assert _digest(project) == before


def test_refit_label_and_rank_are_exclusive_and_overwrite_needs_a_flag(project, capsys):
    base = project / "baseline"
    cli(["baseline", _config(project), "--output-dir", base], capsys)
    out = project / "refit"

    code, payload, _ = cli(["refit", _config(project), base, "--output-dir", out], capsys)
    assert code == 0
    assert payload["label"] == f"baseline:{FAMILY}" and payload["n_test_rows"] == 30

    code, _, err = cli(["refit", _config(project), base, "--output-dir", out], capsys)
    assert code == 1 and "--overwrite" in err

    code, _, _ = cli(["refit", _config(project), base, "--output-dir", out, "--overwrite"], capsys)
    assert code == 0

    with pytest.raises(SystemExit) as exc:
        main(["refit", _config(project), str(base), "--output-dir", str(out), "--label", "a", "--rank", "1"])
    assert exc.value.code == 2


def test_submit_never_overwrites_without_flag(project, capsys):
    cli(["baseline", _config(project), "--output-dir", project / "base"], capsys)
    cli(["refit", _config(project), project / "base", "--output-dir", project / "refit"], capsys)
    args = ["submit", _config(project), project / "refit", "--output-dir", project / "sub"]

    assert cli(args, capsys)[0] == 0
    code, _, err = cli(args, capsys)
    assert code == 1 and "--overwrite" in err
    assert cli([*args, "--overwrite"], capsys)[0] == 0


def test_model_report_overwrite_needs_a_flag(project, capsys):
    pytest.importorskip("matplotlib")
    cli(["baseline", _config(project), "--output-dir", project / "base"], capsys)
    args = ["model-report", _config(project), project / "base", "--output-dir", project / "rep"]

    code, payload, _ = cli(args, capsys)
    assert code == 0 and payload["candidates"][0]["rank"] == 1
    assert (project / "rep" / "model_report.pdf").stat().st_size > 0

    code, _, err = cli(args, capsys)
    assert code == 1 and "overwrite" in err
    assert cli([*args, "--overwrite"], capsys)[0] == 0


def test_run_with_tracking_logs_each_stage(project, capsys):
    mlflow = pytest.importorskip("mlflow")
    pytest.importorskip("optuna")
    (project / "run.toml").write_text(_toml(project, f'\n[tracking]\ndirectory = "{project / "ml"}"\n'))

    code, payload, err = cli(
        ["run", _config(project), "--output-dir", project / "run", "--skip-reports", "--families", FAMILY,
         "--n-trials", 2, "--track"],
        capsys,
    )

    assert code == 0, err
    info = payload["stages"]["baseline"]["tracking"]
    assert info["run_id"]
    assert payload["stages"]["refit"]["tracking"]["run_id"]
    assert payload["stages"]["submission"]["tracking"]["run_id"]
    mlflow.set_tracking_uri(info["tracking_uri"])
    assert mlflow.get_run(info["run_id"]).info.status == "FINISHED"


# ---- run ----


def test_run_matches_the_individual_stages(project, capsys, no_network):
    _require_extras()
    config = _config(project)
    before = _digest(project)
    code, payload, err = cli(
        ["run", config, "--output-dir", project / "run", "--families", FAMILY, "--n-trials", 2], capsys
    )

    assert code == 0, err
    assert payload["status"] == "completed"
    assert payload["selected"]["source"] == "tuning"
    for name in ("splits", "eda", "baseline", "tuning", "reports", "refit", "submission"):
        assert (project / "run" / name).is_dir()
    assert (project / "run" / "reports" / "eda_report.pdf").is_file()
    assert (project / "run" / "reports" / "model_report.pdf").is_file()

    chain = project / "chain"
    cli(["split", config, "--output-dir", chain / "splits"], capsys)
    cli(["tune", config, "--output-dir", chain / "tuning", "--splits", chain / "splits" / "splits.json",
         "--families", FAMILY, "--n-trials", 2], capsys)
    cli(["refit", config, chain / "tuning", "--output-dir", chain / "refit"], capsys)
    cli(["submit", config, chain / "refit", "--output-dir", chain / "submission"], capsys)
    assert (chain / "submission" / "submission.csv").read_bytes() == (
        project / "run" / "submission" / "submission.csv"
    ).read_bytes()
    assert _digest(project) == before


def test_run_skip_flags_use_the_baseline(project, capsys):
    code, payload, _ = cli(
        ["run", _config(project), "--output-dir", project / "run", "--skip-tuning", "--skip-reports"], capsys
    )

    assert code == 0
    assert payload["selected"]["source"] == "baseline"
    assert set(payload["skipped"]) == {"eda_report", "model_report", "tuning"}
    assert not (project / "run" / "tuning").exists() and not (project / "run" / "reports").exists()
    assert (project / "run" / "submission" / "submission.csv").is_file()


def test_run_refuses_a_non_empty_directory(project, capsys):
    (project / "run").mkdir()
    (project / "run" / "keep.txt").write_text("x")

    code, _, err = cli(["run", _config(project), "--output-dir", project / "run", "--skip-tuning"], capsys)

    assert code == 1 and "not empty" in err
    assert (project / "run" / "keep.txt").read_text() == "x"


def test_run_stops_at_the_failing_stage_and_keeps_artifacts(project, capsys, monkeypatch):
    from automl import SubmissionError, workflow

    def fail(*args, **kwargs):
        raise SubmissionError("boom")

    monkeypatch.setattr(workflow, "submit", fail)

    code, _, err = cli(
        ["run", _config(project), "--output-dir", project / "run", "--skip-tuning", "--skip-reports"], capsys
    )

    assert code == 1
    assert "stage 'submission' failed after completing" in err and "refit" in err and "boom" in err
    assert (project / "run" / "refit" / "predictions.csv").is_file()
    assert not (project / "run" / "submission").exists()


def test_run_needs_a_tuning_budget_before_writing(project, capsys):
    _require_extras()
    code, _, err = cli(["run", _config(project), "--output-dir", project / "run", "--families", FAMILY], capsys)

    assert code == 1 and "budget" in err
    assert not (project / "run").exists()


# ---- optional extras ----


def test_missing_optuna_gives_an_install_hint_and_no_files(project, capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "optuna", None)

    code, _, err = cli(["tune", _config(project), "--output-dir", project / "t", "--n-trials", 1], capsys)

    assert code == 1 and "automl[tuning]" in err
    assert not (project / "t").exists()


def test_missing_matplotlib_stops_run_before_writing(project, capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.backends.backend_pdf", None)

    code, _, err = cli(["run", _config(project), "--output-dir", project / "run", "--skip-tuning"], capsys)

    assert code == 1 and "matplotlib" in err.lower()
    assert not (project / "run").exists()


def test_missing_boosting_library_names_the_extra(project, capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "lightgbm", None)

    code, _, err = cli(
        ["baseline", _config(project), "--output-dir", project / "b", "--estimator", "lightgbm"], capsys
    )

    assert code == 1 and "automl[lightgbm]" in err
    assert not (project / "b").exists()


# ---- tracking ----


def test_track_needs_a_tracking_table(project, capsys):
    code, _, err = cli(["baseline", _config(project), "--output-dir", project / "b", "--track"], capsys)

    assert code == 1 and "[tracking]" in err
    assert not (project / "b").exists()


def test_missing_mlflow_is_reported_before_work(project, capsys, monkeypatch):
    (project / "run.toml").write_text(_toml(project, f'\n[tracking]\ndirectory = "{project / "ml"}"\n'))
    monkeypatch.setitem(sys.modules, "mlflow", None)

    code, _, err = cli(["baseline", _config(project), "--output-dir", project / "b", "--track"], capsys)

    assert code == 1 and "automl[mlflow]" in err
    assert not (project / "b").exists()


def test_track_logs_to_a_local_store_and_is_off_by_default(project, capsys):
    mlflow = pytest.importorskip("mlflow")
    store = project / "ml"
    (project / "run.toml").write_text(_toml(project, f'\n[tracking]\ndirectory = "{store}"\n'))

    code, payload, _ = cli(["baseline", _config(project), "--output-dir", project / "off"], capsys)
    assert code == 0 and payload["tracking"] is None
    assert not store.exists()

    code, payload, err = cli(["baseline", _config(project), "--output-dir", project / "on", "--track"], capsys)
    assert code == 0, err
    assert payload["tracking"]["run_id"]
    mlflow.set_tracking_uri(payload["tracking"]["tracking_uri"])
    run = mlflow.get_run(payload["tracking"]["run_id"])
    assert run.info.status == "FINISHED"
