import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from automl import (
    BaselineConfig,
    TuningConfig,
    TuningError,
    TuningUnavailableError,
    ValidationConfig,
    load_config,
    load_tuning,
    make_splits,
    resume_tuning,
    run_baseline,
    tune,
)
from automl.metrics import Task, get_metric
from automl.task import ResolvedTask


def make_frame(task, n=90, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "id": np.arange(n),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "cat": rng.choice(["a", "b", "c"], n),
        }
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


DEFAULT_METRIC = {"binary": "roc_auc", "multiclass": "log_loss", "regression": "rmse"}
SMALL = {"learning_rate": {"type": "float", "low": 0.05, "high": 0.3, "log": True}, "max_iter": {"type": "int", "low": 5, "high": 20}}


def setup(task="binary", metric=None, n_folds=3, df=None):
    df = make_frame(task) if df is None else df
    resolved = ResolvedTask(Task(task), get_metric(metric or DEFAULT_METRIC[task]))
    splits = make_splits(df["y"], resolved, ValidationConfig(n_folds=n_folds))
    return df, resolved, splits


def do_tune(task="binary", metric=None, cfg=None, space=SMALL, **kw):
    df, resolved, splits = setup(task, metric)
    cfg = cfg or TuningConfig(n_trials=4)
    return df, resolved, splits, tune(df, "y", ("id",), resolved, splits, cfg, space=space, **kw)


# ---- studies (need optuna) ----

pytest.importorskip("optuna")


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
def test_each_task(task):
    _, _, _, res = do_tune(task)
    assert len(res.trials) == 4 and all(t.state == "complete" for t in res.trials)
    assert all(np.isfinite(t.score) and len(t.fold_scores) == 3 for t in res.trials)
    assert res.best is not None and res.direction == get_metric(DEFAULT_METRIC[task]).direction
    pick = max if res.direction == "maximize" else min
    assert res.best.score == pick(t.score for t in res.trials)


def test_direction_both_ways():
    assert do_tune("binary", "roc_auc")[3].direction == "maximize"
    assert do_tune("binary", "log_loss")[3].direction == "minimize"


def test_builtin_space_and_registry_space():
    df, resolved, splits = setup()
    res = tune(df, "y", ("id",), resolved, splits, TuningConfig(n_trials=2))
    assert res.space["kind"] == "builtin" and set(res.trials[0].params) == set(res.space["spec"])
    pytest.importorskip("lightgbm")
    res = tune(df, "y", ("id",), resolved, splits, TuningConfig(n_trials=1, estimator="lightgbm"))
    assert res.space["kind"] == "builtin" and "num_leaves" in res.space["spec"]


def test_budgets():
    assert len(do_tune(cfg=TuningConfig(n_trials=3))[3].trials) == 3
    _, _, _, res = do_tune(cfg=TuningConfig(timeout_seconds=0.5, n_trials=500))
    assert 1 <= len(res.trials) < 500
    assert res.budgets[0]["elapsed_seconds"] < 0.5 + 5
    _, _, _, res = do_tune(cfg=TuningConfig(timeout_seconds=60, n_trials=2))
    assert len(res.trials) == 2


def test_repeatable_and_seed_sensitive():
    a = do_tune(cfg=TuningConfig(n_trials=4, seed=1))[3]
    b = do_tune(cfg=TuningConfig(n_trials=4, seed=1))[3]
    c = do_tune(cfg=TuningConfig(n_trials=4, seed=2))[3]
    assert [(t.params, t.score) for t in a.trials] == [(t.params, t.score) for t in b.trials]
    assert [t.params for t in a.trials] != [t.params for t in c.trials]


def test_matches_run_baseline_for_fixed_params():
    df, resolved, splits = setup()
    fixed = {"learning_rate": 0.1, "max_iter": 10}
    space = {k: {"type": "categorical", "choices": [v]} for k, v in fixed.items()}
    res = tune(df, "y", ("id",), resolved, splits, TuningConfig(n_trials=1), space=space)
    base = run_baseline(df, "y", ("id",), resolved, splits, BaselineConfig(estimator_params=fixed))
    assert list(res.best.fold_scores) == [f.score for f in base.folds]


def test_callable_space():
    _, _, _, res = do_tune(space=lambda t: {"max_iter": t.suggest_int("max_iter", 5, 15)})
    assert res.space["kind"] == "callable" and all(5 <= t.params["max_iter"] <= 15 for t in res.trials)


def test_mapping_samples_within_bounds():
    space = {
        "learning_rate": {"type": "float", "low": 0.1, "high": 0.2},
        "max_iter": {"type": "int", "low": 5, "high": 9, "step": 2},
        "loss": {"type": "categorical", "choices": ["log_loss"]},
    }
    for t in do_tune(space=space)[3].trials:
        assert 0.1 <= t.params["learning_rate"] <= 0.2
        assert t.params["max_iter"] in (5, 7, 9) and t.params["loss"] == "log_loss"


@pytest.mark.parametrize(
    "space, text",
    [
        ({}, "at least one"),
        ({"a": {"type": "str"}}, "unsupported type"),
        ({"a": {"type": "float", "low": 1, "high": 1}}, "low < high"),
        ({"a": {"type": "float", "low": 0, "high": 1, "log": True}}, "low > 0"),
        ({"a": {"type": "int", "low": 0.5, "high": 3}}, "integer"),
        ({"a": {"type": "categorical", "choices": []}}, "choices"),
        ({"random_state": {"type": "int", "low": 0, "high": 3}}, "'seed' setting"),
    ],
)
def test_invalid_spaces(space, text):
    df, resolved, splits = setup()
    with pytest.raises(TuningError):
        tune(df, "y", ("id",), resolved, splits, TuningConfig(n_trials=1), space=space)


def test_duplicate_fixed_key_rejected():
    df, resolved, splits = setup()
    cfg = TuningConfig(n_trials=1, estimator_params={"max_iter": 5})
    with pytest.raises(TuningError, match="also fixed"):
        tune(df, "y", ("id",), resolved, splits, cfg, space=SMALL)


def test_failed_trials_recorded_and_never_best():
    def space(trial):
        bad = trial.number % 2 == 1
        return {"max_iter": 5, "learning_rate": -1.0 if bad else 0.1}

    _, _, _, res = do_tune(space=space, cfg=TuningConfig(n_trials=4))
    states = [t.state for t in res.trials]
    assert states == ["complete", "failed", "complete", "failed"]
    assert all(t.failure_reason for t in res.trials if t.state == "failed")
    assert res.best.state == "complete"


def test_all_failed_raises(tmp_path):
    df, resolved, splits = setup()
    with pytest.raises(TuningError, match="no trial completed") as err:
        tune(df, "y", ("id",), resolved, splits, TuningConfig(n_trials=2), space=lambda t: {"bogus": 1}, directory=tmp_path / "t")
    assert "does not accept" in str(err.value)
    assert load_tuning(tmp_path / "t").best is None


def test_pruning_default_off_and_prunes_when_on():
    def space(trial):
        return {"max_iter": 20, "learning_rate": 0.1 if trial.number < 3 else 1e-9}

    off = do_tune(space=space, cfg=TuningConfig(n_trials=5))[3]
    assert all(t.state == "complete" for t in off.trials)
    on = do_tune(space=space, cfg=TuningConfig(n_trials=5, pruning=True))[3]
    pruned = [t for t in on.trials if t.state == "pruned"]
    assert pruned and all(0 < len(t.fold_scores) < 3 and t.score is None for t in pruned)
    assert on.best.state == "complete"


def test_leakage_and_no_mutation(monkeypatch):
    df, resolved, splits = setup()
    before = df.copy()
    seen = []
    import automl.tuning as tuning

    real = tuning._run_fold

    def spy(pipeline, X, y, split, *a):
        seen.append(set(X.columns))
        return real(pipeline, X, y, split, *a)

    monkeypatch.setattr(tuning, "_run_fold", spy)
    tune(df, "y", ("id",), resolved, splits, TuningConfig(n_trials=1), space=SMALL)
    assert seen and all("y" not in cols and "id" not in cols for cols in seen)
    pd.testing.assert_frame_equal(df, before)


def test_holdout_strategy():
    df = make_frame("binary")
    resolved = ResolvedTask(Task.BINARY, get_metric("roc_auc"))
    splits = make_splits(df["y"], resolved, ValidationConfig(strategy="holdout"))
    res = tune(df, "y", ("id",), resolved, splits, TuningConfig(n_trials=2, pruning=True), space=SMALL)
    assert all(len(t.fold_scores) == 1 for t in res.trials)


def test_persistence_roundtrip_and_no_overwrite(tmp_path, monkeypatch):
    d = tmp_path / "run"
    _, _, _, res = do_tune(directory=d)
    assert sorted(p.name for p in d.iterdir()) == ["study.db", "tuning.json"]
    json.loads((d / "tuning.json").read_text())
    monkeypatch.setitem(sys.modules, "optuna", None)  # loading must not need optuna
    loaded = load_tuning(d)
    assert loaded.best.number == res.best.number
    assert [t.score for t in loaded.trials] == [t.score for t in res.trials]
    assert loaded.config == res.config and loaded.space == res.space
    monkeypatch.undo()
    df, resolved, splits = setup()
    with pytest.raises(TuningError, match="already exists"):
        tune(df, "y", ("id",), resolved, splits, TuningConfig(n_trials=1), space=SMALL, directory=d)


def test_load_errors(tmp_path):
    with pytest.raises(TuningError, match="not found"):
        load_tuning(tmp_path)
    (tmp_path / "tuning.json").write_text("{bad")
    with pytest.raises(TuningError, match="not valid JSON"):
        load_tuning(tmp_path)
    (tmp_path / "tuning.json").write_text('{"schema_version": 9}')
    with pytest.raises(TuningError, match="schema_version"):
        load_tuning(tmp_path)


def test_resume_adds_trials(tmp_path):
    d = tmp_path / "run"
    df, resolved, splits, first = do_tune(directory=d, cfg=TuningConfig(n_trials=3))
    res = resume_tuning(d, df, "y", ("id",), resolved, splits, n_trials=2)
    assert len(res.trials) == 5 and len(res.budgets) == 2
    assert [t.score for t in res.trials[:3]] == [t.score for t in first.trials]
    assert len(load_tuning(d).trials) == 5
    assert res.best.score >= first.best.score


def test_resume_mismatch_and_requirements(tmp_path):
    d = tmp_path / "run"
    df, resolved, splits, _ = do_tune(directory=d)
    with pytest.raises(TuningError, match="budget"):
        resume_tuning(d, df, "y", ("id",), resolved, splits)
    other = df.iloc[:-5].reset_index(drop=True)
    s2 = make_splits(other["y"], resolved, ValidationConfig(n_folds=3))
    with pytest.raises(TuningError, match="does not match"):
        resume_tuning(d, other, "y", ("id",), resolved, s2, n_trials=1)
    s3 = make_splits(df["y"], resolved, ValidationConfig(n_folds=3, seed=7))
    with pytest.raises(TuningError, match="splits_sha256"):
        resume_tuning(d, df, "y", ("id",), resolved, s3, n_trials=1)


def test_resume_callable_space_must_be_resupplied(tmp_path):
    d = tmp_path / "run"
    space = lambda t: {"max_iter": t.suggest_int("max_iter", 5, 10)}  # noqa: E731
    df, resolved, splits, _ = do_tune(directory=d, space=space)
    with pytest.raises(TuningError, match="callable"):
        resume_tuning(d, df, "y", ("id",), resolved, splits, n_trials=1)
    assert len(resume_tuning(d, df, "y", ("id",), resolved, splits, n_trials=1, space=space).trials) == 5


@pytest.mark.parametrize("name", ["lightgbm", "xgboost", "catboost"])
def test_adapter_two_trials(name):
    pytest.importorskip(name)
    spaces = {
        "lightgbm": {"n_estimators": {"type": "int", "low": 2, "high": 6}},
        "xgboost": {"n_estimators": {"type": "int", "low": 2, "high": 6}},
        "catboost": {"iterations": {"type": "int", "low": 2, "high": 6}},
    }
    df, resolved, splits = setup("binary")
    res = tune(df, "y", ("id",), resolved, splits, TuningConfig(n_trials=2, estimator=name), space=spaces[name])
    assert res.best is not None and np.isfinite(res.best.score)
