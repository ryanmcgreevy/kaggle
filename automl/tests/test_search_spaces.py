import json
import subprocess
import sys

import numpy as np
import pytest

from automl import (
    SPACE_FAMILIES,
    TuningConfig,
    TuningError,
    categorize_failure,
    describe_search_space,
    estimator_available,
    get_estimator_spec,
    get_search_space,
    load_config,
    load_tuning,
)
from automl.errors import BaselineError
from automl.estimators import CONTROLLED_PARAMS, FeatureLayout
from automl.metrics import Task
from automl.tuning import _validate_mapping

TASKS = list(Task)


@pytest.mark.parametrize("family", SPACE_FAMILIES)
@pytest.mark.parametrize("task", TASKS)
def test_space_is_valid_and_has_no_controlled_keys(family, task):
    space = get_search_space(family, task)
    assert _validate_mapping(space) == space
    assert not set(space) & set(CONTROLLED_PARAMS)
    assert [d["name"] for d in describe_search_space(family, task)] == list(space)


def test_space_is_task_aware_and_copy_safe():
    reg = get_search_space("lightgbm", Task.REGRESSION)
    multi = get_search_space("lightgbm", Task.MULTICLASS)
    assert reg["n_estimators"]["high"] > multi["n_estimators"]["high"]
    reg["learning_rate"]["low"] = 99
    assert get_search_space("lightgbm", Task.REGRESSION)["learning_rate"]["low"] == 0.01


def test_unknown_family():
    with pytest.raises(TuningError, match="no search space for estimator family 'nope'"):
        get_search_space("nope", Task.BINARY)


def _midpoint(spec):
    if spec["type"] == "categorical":
        return spec["choices"][0]
    value = (spec["low"] + spec["high"]) / 2
    return int(round(value)) if spec["type"] == "int" else value


@pytest.mark.parametrize("family", SPACE_FAMILIES)
@pytest.mark.parametrize("task", TASKS)
def test_sampled_parameters_are_accepted_by_estimator(family, task):
    if not estimator_available(family):
        pytest.skip(f"{family} not installed")
    params = {k: _midpoint(v) for k, v in get_search_space(family, task).items()}
    est = get_estimator_spec(family).make(task, params, 0, 1, "cpu", FeatureLayout(2, 0))
    X = np.random.default_rng(0).normal(size=(60, 2))
    y = np.arange(60) % (3 if task is Task.MULTICLASS else 2) if task is not Task.REGRESSION else X[:, 0]
    est.fit(X, y)
    assert len(est.predict(X)) == 60


def test_overrides_narrow_drop_and_fix():
    cfg = TuningConfig(
        n_trials=1,
        spaces={
            "lightgbm": {
                "learning_rate": {"low": 0.1, "high": 0.2},
                "reg_alpha": {"enabled": False},
                "num_leaves": {"fixed": 31},
            }
        },
    )
    space = get_search_space("lightgbm", Task.BINARY, cfg.spaces["lightgbm"])
    assert space["learning_rate"]["low"] == 0.1 and space["learning_rate"]["log"] is True
    assert "reg_alpha" not in space
    assert space["num_leaves"] == {"type": "categorical", "choices": [31]}


@pytest.mark.parametrize(
    "spaces, text",
    [
        ({"nope": {}}, "unknown estimator 'nope'"),
        ({"lightgbm": {"bogus": {"low": 1}}}, "'lightgbm.bogus': unknown parameter"),
        ({"lightgbm": {"learning_rate": {"min": 1}}}, "'lightgbm.learning_rate' has unknown keys"),
        ({"lightgbm": {"learning_rate": {"low": 1.0, "high": 0.5}}}, "needs low < high"),
        ({"lightgbm": {"learning_rate": {"low": 0.0}}}, "needs low > 0"),
        ({"lightgbm": {"learning_rate": {"choices": [0.1]}}}, "unknown keys"),
        ({"lightgbm": {"subsample_freq": {"low": 1, "high": 2}}}, "unknown keys"),
        ({"lightgbm": {"learning_rate": {"enabled": "no"}}}, "'enabled' must be true or false"),
        ({"lightgbm": {"learning_rate": {"fixed": 0.1, "low": 0.01}}}, "'fixed' cannot be combined"),
        ({"lightgbm": "x"}, "must be a table"),
        ({"xgboost": {k: {"enabled": False} for k in get_search_space("xgboost", Task.BINARY)}}, "at least one parameter"),
    ],
)
def test_invalid_overrides(spaces, text):
    with pytest.raises(TuningError, match=text):
        TuningConfig(n_trials=1, spaces=spaces)


def test_family_config_validation():
    for kwargs in (
        {"families": []},
        {"families": ["nope"]},
        {"families": ["lightgbm", "lightgbm"]},
        {"budgets": {"nope": {"n_trials": 1}}},
        {"budgets": {"lightgbm": {"n_trials": 0}}},
        {"budgets": {"lightgbm": {}}},
        {"budgets": {"lightgbm": {"bogus": 1}}},
    ):
        with pytest.raises(TuningError):
            TuningConfig(n_trials=1, **kwargs)
    with pytest.raises(TuningError, match="needs a budget"):
        TuningConfig(families=["lightgbm", "xgboost"], budgets={"lightgbm": {"n_trials": 1}})
    ok = TuningConfig(families=["lightgbm"], budgets={"lightgbm": {"timeout_seconds": 5}})
    assert ok.families == ("lightgbm",) and ok.n_trials is None


def test_toml_families_budgets_and_spaces(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text(
        """version = 1
[data]
train = "t.csv"
test = "s.csv"
target = "y"
[tuning]
n_trials = 5
families = ["lightgbm", "xgboost"]
[tuning.budgets.xgboost]
timeout_seconds = 30
[tuning.spaces.lightgbm.learning_rate]
low = 0.05
high = 0.2
[tuning.spaces.lightgbm.reg_alpha]
enabled = false
"""
    )
    cfg = load_config(p).tuning_config
    assert cfg.families == ("lightgbm", "xgboost") and cfg.budgets == {"xgboost": {"timeout_seconds": 30}}
    assert cfg.spaces["lightgbm"]["reg_alpha"] == {"enabled": False}
    p.write_text(p.read_text().replace("low = 0.05", "low = 0.5"))
    with pytest.raises(Exception, match="needs low < high"):
        load_config(p)


@pytest.mark.parametrize(
    "exc, category",
    [
        (BaselineError("setting 'estimator_params' has parameters ['x'] that A does not accept"), "invalid_parameter"),
        (BaselineError("fold 0: metric 'm' returned a non-finite score"), "non_finite_score"),
        (BaselineError("fold 0: class sets differ"), "other"),
        (TimeoutError("slow"), "timeout"),
        (TuningError("bad"), "invalid_parameter"),
        (ValueError("Invalid parameter 'x'"), "invalid_parameter"),
        (RuntimeError("boom"), "other"),
    ],
)
def test_categorize_failure(exc, category):
    assert categorize_failure(exc) == category


def test_categorize_library_error():
    LightGBMError = type("LightGBMError", (Exception,), {"__module__": "lightgbm.basic"})
    assert categorize_failure(LightGBMError("x")) == "library_error"


def test_schema_1_tuning_json_still_loads(tmp_path):
    trial = {
        "number": 0, "state": "failed", "params": {}, "score": None, "fold_scores": [],
        "fold_std": None, "duration_seconds": 0.1, "failure_reason": "RuntimeError: x",
    }
    ok = {**trial, "number": 1, "state": "complete", "score": 0.5, "fold_scores": [0.5], "fold_std": 0.0, "failure_reason": None}
    payload = {
        "schema_version": 1, "task": "binary", "metric": {"name": "roc_auc", "direction": "maximize"},
        "config": {"n_trials": 2}, "space": {"kind": "builtin", "spec": {}}, "fingerprint": {}, "budgets": [],
        "best_trial": 1, "trials": [trial, ok], "versions": {}, "notes": [], "files": {},
    }
    (tmp_path / "tuning.json").write_text(json.dumps(payload))
    res = load_tuning(tmp_path)
    assert res.trials[0].failure_category == "other" and res.trials[1].failure_category is None
    assert res.config.families is None and res.config.spaces == {}


def test_import_loads_no_optional_libraries():
    code = (
        "import sys, automl; "
        "bad = [m for m in ('optuna', 'lightgbm', 'xgboost', 'catboost') if m in sys.modules]; "
        "sys.exit(1 if bad else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
