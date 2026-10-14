import json
import sys

import numpy as np
import pandas as pd
import pytest

from automl import (
    TuningConfig,
    TuningError,
    ValidationConfig,
    estimator_available,
    load_tuning,
    load_tuning_families,
    make_splits,
    resume_tuning_families,
    tune,
    tune_families,
)
from automl.estimators import ESTIMATOR_NAMES
from automl.metrics import Task, get_metric
from automl.task import ResolvedTask

pytest.importorskip("optuna")

INSTALLED = [f for f in ESTIMATOR_NAMES if estimator_available(f)]
DEFAULT_METRIC = {"binary": "roc_auc", "multiclass": "log_loss", "regression": "rmse"}
HGB = "hist_gradient_boosting"


def make_frame(task, n=90, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {"id": np.arange(n), "x1": rng.normal(size=n), "x2": rng.normal(size=n), "cat": rng.choice(["a", "b", "c"], n)}
    )
    df.loc[:4, "x1"] = np.nan
    signal = df["x2"].fillna(0) + (df["cat"] == "a")
    if task == "regression":
        df["y"] = signal * 2 + rng.normal(0, 0.1, n) + 5
    elif task == "binary":
        df["y"] = (signal + rng.normal(0, 0.5, n) > 0.5).astype(int)
        df.loc[:1, "y"] = [0, 1]
    else:
        df["y"] = pd.cut(signal, [-9, -0.3, 0.8, 9], labels=["lo", "mid", "hi"]).astype(str)
    return df


def setup(task="binary", metric=None):
    df = make_frame(task)
    resolved = ResolvedTask(Task(task), get_metric(metric or DEFAULT_METRIC[task]))
    return df, resolved, make_splits(df["y"], resolved, ValidationConfig(n_folds=3))


def run(task="binary", metric=None, cfg=None, **kw):
    df, resolved, splits = setup(task, metric)
    return df, resolved, splits, tune_families(df, "y", ("id",), resolved, splits, cfg or TuningConfig(n_trials=2), **kw)


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
def test_all_installed_families_each_task(task):
    *_, res = run(task)
    assert [o.family for o in res.outcomes] == INSTALLED
    for o in res.outcomes:
        assert o.status == "complete" and len(o.result.trials) == 2
        assert all(np.isfinite(t.score) for t in o.result.trials)
        assert o.result.space["kind"] == "builtin"
        # sampled parameters are exactly the documented space
        assert set(o.result.trials[0].params) == set(o.result.space["spec"])
    assert res.best_family == res.comparison[0]["family"] and res.comparison[0]["rank"] == 1


@pytest.mark.parametrize("metric, direction", [("roc_auc", "maximize"), ("log_loss", "minimize")])
def test_comparison_ranked_by_direction(metric, direction):
    *_, res = run("binary", metric, families=[HGB, *INSTALLED[1:2]])
    assert res.direction == direction
    scores = [r["best_score"] for r in res.comparison]
    assert scores == sorted(scores, reverse=(direction == "maximize"))
    assert [r["rank"] for r in res.comparison] == list(range(1, len(scores) + 1))
    assert "not leaderboard" in res.notes[0]
    row = res.comparison[0]
    assert {"best_fold_std", "trial_counts", "duration_seconds", "best_params", "failure_categories"} <= set(row)


def test_missing_requested_family_gives_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "lightgbm", None)
    df, resolved, splits = setup()
    with pytest.raises(Exception, match=r"pip install 'automl\[lightgbm\]'"):
        tune_families(df, "y", ("id",), resolved, splits, TuningConfig(n_trials=1), families=[HGB, "lightgbm"])


def test_unknown_family_rejected():
    df, resolved, splits = setup()
    with pytest.raises(TuningError, match="unique names"):
        tune_families(df, "y", ("id",), resolved, splits, TuningConfig(n_trials=1), families=["nope"])


def test_budgets_per_family_shared_and_missing():
    df, resolved, splits = setup()
    cfg = TuningConfig(n_trials=1, families=(HGB,), budgets={HGB: {"n_trials": 3}})
    res = tune_families(df, "y", ("id",), resolved, splits, cfg)
    assert len(res.outcome(HGB).result.trials) == 3 and res.outcome(HGB).budget["n_trials"] == 3
    res = tune_families(df, "y", ("id",), resolved, splits, cfg, budgets={HGB: {"n_trials": 2}})
    assert len(res.outcome(HGB).result.trials) == 2
    res = tune_families(df, "y", ("id",), resolved, splits, TuningConfig(n_trials=2), families=[HGB])
    assert len(res.outcome(HGB).result.trials) == 2
    only = TuningConfig(families=(HGB,), budgets={HGB: {"n_trials": 1}})
    with pytest.raises(TuningError, match="family 'lightgbm' has no budget"):
        tune_families(df, "y", ("id",), resolved, splits, only, families=[HGB, "lightgbm"])


def test_timeout_budget_per_family():
    df, resolved, splits = setup()
    cfg = TuningConfig(n_trials=500, timeout_seconds=0.5)
    res = tune_families(df, "y", ("id",), resolved, splits, cfg, families=[HGB])
    assert 1 <= len(res.outcome(HGB).result.trials) < 500
    assert res.outcome(HGB).result.budgets[0]["elapsed_seconds"] < 5.5


def test_identical_splits_and_repeatable():
    *_, a = run(cfg=TuningConfig(n_trials=3, seed=3), families=INSTALLED[:2])
    *_, b = run(cfg=TuningConfig(n_trials=3, seed=3), families=INSTALLED[:2])
    hashes = {o.result.fingerprint["splits_sha256"] for o in a.outcomes}
    assert len(hashes) == 1
    for x, y in zip(a.outcomes, b.outcomes):
        assert [(t.params, t.score) for t in x.result.trials] == [(t.params, t.score) for t in y.result.trials]


def test_override_respected_and_conflicts_with_supplied_space():
    cfg = TuningConfig(
        n_trials=4,
        spaces={HGB: {"learning_rate": {"low": 0.1, "high": 0.2}, "max_depth": {"enabled": False}, "max_iter": {"fixed": 10}}},
    )
    *_, res = run(cfg=cfg, families=[HGB])
    trials = res.outcome(HGB).result.trials
    assert all(0.1 <= t.params["learning_rate"] <= 0.2 and t.params["max_iter"] == 10 for t in trials)
    assert all("max_depth" not in t.params for t in trials)
    df, resolved, splits = setup()
    with pytest.raises(TuningError, match="cannot be combined"):
        tune_families(df, "y", ("id",), resolved, splits, cfg, families=[HGB], spaces={HGB: {"max_iter": {"type": "int", "low": 5, "high": 9}}})


def test_failure_categories_recorded_without_corrupting_results():
    def space(trial):
        n = trial.number
        if n == 1:
            return {"bogus": 1}
        if n == 2:
            raise TimeoutError("took too long")
        if n == 3:
            return {"learning_rate": -1.0}
        return {"max_iter": 5}

    *_, res = run(cfg=TuningConfig(n_trials=5), families=[HGB], spaces={HGB: space})
    trials = res.outcome(HGB).result.trials
    assert [t.state for t in trials] == ["complete", "failed", "failed", "failed", "complete"]
    assert [t.failure_category for t in trials] == [None, "invalid_parameter", "timeout", "invalid_parameter", None]
    assert all(t.failure_reason for t in trials if t.state == "failed")
    assert res.outcome(HGB).result.best.state == "complete"
    assert res.comparison[0]["failure_categories"] == {"invalid_parameter": 2, "timeout": 1}


def test_failed_family_is_isolated_and_all_failed_raises(tmp_path):
    fams = [HGB, *INSTALLED[1:2]]
    *_, res = run(families=fams, spaces={fams[1]: lambda t: {"bogus": 1}}, directory=tmp_path / "r")
    assert res.outcome(fams[1]).status == "failed" and "no trial completed" in res.outcome(fams[1]).error
    assert res.best_family == HGB and res.comparison[-1]["status"] == "failed" and res.comparison[-1]["rank"] is None
    loaded = load_tuning_families(tmp_path / "r")
    assert loaded.outcome(fams[1]).status == "failed" and loaded.outcome(fams[1]).result.best is None
    df, resolved, splits = setup()
    with pytest.raises(TuningError, match="no family produced a completed trial"):
        tune_families(df, "y", ("id",), resolved, splits, TuningConfig(n_trials=1), families=[HGB], spaces={HGB: lambda t: {"bogus": 1}})


def test_persistence_layout_roundtrip_and_no_overwrite(tmp_path, monkeypatch):
    d = tmp_path / "run"
    fams = INSTALLED[:2]
    df, resolved, splits, res = run(families=fams, directory=d)
    assert sorted(p.name for p in d.iterdir()) == sorted([*fams, "tuning_families.json"])
    for f in fams:
        assert (d / f / "study.db").is_file() and (d / f / "tuning.json").is_file()
        assert load_tuning(d / f).config.estimator == f
    raw = json.loads((d / "tuning_families.json").read_text())
    assert raw["schema_version"] == 1 and set(raw["budget_plan"]) == set(fams) and raw["best_family"] == res.best_family
    monkeypatch.setitem(sys.modules, "optuna", None)  # loading needs no optuna
    loaded = load_tuning_families(d)
    assert loaded.best_family == res.best_family and loaded.comparison == tuple(json.loads(json.dumps(list(res.comparison))))
    assert [t.score for t in loaded.outcome(fams[0]).result.trials] == [t.score for t in res.outcome(fams[0]).result.trials]
    monkeypatch.undo()
    with pytest.raises(TuningError, match="already exists"):
        tune_families(df, "y", ("id",), resolved, splits, TuningConfig(n_trials=1), families=fams, directory=d)


def test_load_errors(tmp_path):
    with pytest.raises(TuningError, match="not found"):
        load_tuning_families(tmp_path)
    (tmp_path / "tuning_families.json").write_text("{bad")
    with pytest.raises(TuningError, match="not valid JSON"):
        load_tuning_families(tmp_path)
    (tmp_path / "tuning_families.json").write_text('{"schema_version": 9}')
    with pytest.raises(TuningError, match="schema_version"):
        load_tuning_families(tmp_path)


def test_resume_per_family(tmp_path):
    d = tmp_path / "run"
    fams = INSTALLED[:2]
    df, resolved, splits, first = run(families=fams, directory=d, cfg=TuningConfig(n_trials=2))
    res = resume_tuning_families(d, df, "y", ("id",), resolved, splits, budgets={fams[0]: {"n_trials": 2}}, n_trials=1, families=fams)
    assert len(res.outcome(fams[0]).result.trials) == 4 and len(res.outcome(fams[1]).result.trials) == 3
    assert [t.score for t in res.outcome(fams[0]).result.trials[:2]] == [t.score for t in first.outcome(fams[0]).result.trials]
    assert len(load_tuning(d / fams[0]).trials) == 4
    with pytest.raises(TuningError, match="no saved study"):
        resume_tuning_families(d, df, "y", ("id",), resolved, splits, n_trials=1, families=["nope"])
    with pytest.raises(TuningError, match="budget"):
        resume_tuning_families(d, df, "y", ("id",), resolved, splits)


def test_single_study_tune_unchanged_and_supplied_space_wins():
    df, resolved, splits = setup()
    space = {"max_iter": {"type": "int", "low": 5, "high": 9}}
    res = tune(df, "y", ("id",), resolved, splits, TuningConfig(n_trials=2), space=space)
    assert res.space["kind"] == "mapping" and all(5 <= t.params["max_iter"] <= 9 for t in res.trials)
    with pytest.raises(TuningError, match="needs a budget|budget"):
        tune(df, "y", ("id",), resolved, splits, TuningConfig(families=(HGB,), budgets={HGB: {"n_trials": 1}}))


def test_input_not_mutated():
    df, resolved, splits = setup()
    before = df.copy(deep=True)
    tune_families(df, "y", ("id",), resolved, splits, TuningConfig(n_trials=1), families=[HGB])
    pd.testing.assert_frame_equal(df, before)
