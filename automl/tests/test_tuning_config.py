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


def test_config_validation():
    for kwargs in (
        {},
        {"n_trials": 0},
        {"n_trials": True},
        {"timeout_seconds": -1},
        {"n_trials": 2, "pruning": "yes"},
        {"n_trials": 2, "estimator": "nope"},
        {"n_trials": 2, "estimator_params": {"random_state": 1}},
        {"n_trials": 2, "device": "tpu"},
    ):
        with pytest.raises(TuningError):
            TuningConfig(**kwargs)
    assert TuningConfig(timeout_seconds=1.5).n_trials is None


def test_toml_tuning_table(tmp_path):
    base = '''version = 1
[data]
train = "t.csv"
test = "s.csv"
target = "y"
'''
    p = tmp_path / "c.toml"
    p.write_text(base + "[tuning]\nn_trials = 3\npruning = true\n")
    cfg = load_config(p)
    assert cfg.tuning_config.n_trials == 3 and cfg.tuning_config.pruning
    p.write_text(base)
    assert load_config(p).tuning_config is None
    p.write_text(base + "[tuning]\nbogus = 1\n")
    with pytest.raises(Exception, match="unknown keys in \\[tuning\\]"):
        load_config(p)
    p.write_text(base + "[tuning]\npruning = true\n")
    with pytest.raises(Exception, match="budget"):
        load_config(p)


def test_import_does_not_load_optuna():
    code = "import sys, automl; sys.exit(1 if 'optuna' in sys.modules else 0)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_missing_optuna(monkeypatch):
    df, resolved, splits = setup()
    monkeypatch.setitem(sys.modules, "optuna", None)
    with pytest.raises(TuningUnavailableError, match=r"pip install 'automl\[tuning\]'"):
        tune(df, "y", ("id",), resolved, splits, TuningConfig(n_trials=1))
    run_baseline(df, "y", ("id",), resolved, splits)  # baseline path still works


