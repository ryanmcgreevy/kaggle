import numpy as np
import pandas as pd
import pytest

from automl import (
    Task,
    TaskConfig,
    TaskMetricError,
    DataContractError,
    get_metric,
    list_metrics,
    load_config,
    load_data,
    register_metric,
    resolve_task_metric,
    suggest_metric,
    suggest_task,
)

BINARY = pd.Series([0, 1, 0, 1, 1, 0])
MULTI = pd.Series(["a", "b", "c", "a", "b", "c"])
REG = pd.Series([0.5, 1.7, 2.2, 3.9, 4.1, 5.3])

SAMPLES = {
    Task.BINARY: (
        np.array([0, 1, 0, 1, 1, 0]),
        {"labels": np.array([0, 1, 1, 1, 0, 0]), "proba": np.array([0.2, 0.8, 0.4, 0.7, 0.6, 0.1])},
    ),
    Task.MULTICLASS: (
        np.array([0, 1, 2, 0, 1, 2]),
        {
            "labels": np.array([0, 1, 2, 1, 1, 2]),
            "proba": np.array(
                [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.3, 0.4, 0.3], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]
            ),
        },
    ),
    Task.REGRESSION: (REG.to_numpy(), {"values": REG.to_numpy() + 0.1}),
}


def _series_for(task: Task) -> pd.Series:
    return {Task.BINARY: BINARY, Task.MULTICLASS: MULTI, Task.REGRESSION: REG}[task]


@pytest.mark.parametrize("task", list(Task))
def test_every_builtin_combination_resolves_and_scores(task):
    names = list_metrics(task)
    assert names
    for name in names:
        resolved = resolve_task_metric(_series_for(task), TaskConfig(task.value, name))
        y_true, preds = SAMPLES[task]
        value = resolved.metric(y_true, preds[resolved.metric.needs])
        assert np.isfinite(value)
        assert resolved.direction in ("maximize", "minimize")


def test_builtin_metric_sets():
    assert list_metrics("binary") == ["accuracy", "f1", "log_loss", "roc_auc"]
    assert list_metrics("multiclass") == ["accuracy", "balanced_accuracy", "log_loss", "macro_f1"]
    assert list_metrics("regression") == ["mae", "r2", "rmse", "rmsle"]


def test_unset_task_never_defaulted_and_includes_suggestion():
    with pytest.raises(TaskMetricError) as exc:
        resolve_task_metric(BINARY, TaskConfig(metric="roc_auc"))
    assert "setting 'task'" in str(exc.value) and "Suggestion: 'binary'" in str(exc.value)


def test_unset_metric_never_defaulted_and_includes_suggestion():
    with pytest.raises(TaskMetricError) as exc:
        resolve_task_metric(BINARY, TaskConfig(task="binary"))
    msg = str(exc.value)
    assert "setting 'metric'" in msg and "'roc_auc'" in msg and "f1" in msg


def test_both_unset_reports_both():
    with pytest.raises(TaskMetricError) as exc:
        resolve_task_metric(REG, TaskConfig())
    assert len(exc.value.problems) == 2


def test_unknown_metric_and_mismatch():
    with pytest.raises(TaskMetricError, match="unknown metric 'nope'"):
        resolve_task_metric(BINARY, TaskConfig("binary", "nope"))
    with pytest.raises(TaskMetricError, match="metric 'rmse' does not support task 'binary'"):
        resolve_task_metric(BINARY, TaskConfig("binary", "rmse"))


def test_unsupported_task_value():
    with pytest.raises(TaskMetricError, match="setting 'task' has unsupported value 'ranking'"):
        resolve_task_metric(BINARY, TaskConfig("ranking", "accuracy"))


@pytest.mark.parametrize(
    ("task", "metric", "target", "fragment"),
    [
        ("binary", "accuracy", MULTI, "task 'binary' requires exactly 2"),
        ("multiclass", "accuracy", BINARY, "task 'multiclass' requires 3 or more"),
        ("regression", "rmse", MULTI, "task 'regression' requires a numeric target"),
        ("binary", "accuracy", pd.Series([0, 1, None, 1]), "target has 1 null"),
    ],
)
def test_task_inconsistent_with_target(task, metric, target, fragment):
    with pytest.raises(TaskMetricError, match=fragment):
        resolve_task_metric(target, TaskConfig(task, metric))


def test_suggest_task():
    assert suggest_task(BINARY).value == "binary"
    assert suggest_task(BINARY).ambiguous
    assert suggest_task(pd.Series(["x", "y", "x"])).value == "binary"
    assert not suggest_task(pd.Series(["x", "y", "x"])).ambiguous
    assert suggest_task(MULTI).value == "multiclass"
    s = suggest_task(REG)
    assert s.value == "regression" and not s.ambiguous and s.evidence["unique"] == 6


def test_ambiguous_integer_target_is_flagged_not_selected():
    s = suggest_task(pd.Series([1, 2, 3, 4, 1, 2]))
    assert s.value == "multiclass" and s.ambiguous
    with pytest.raises(TaskMetricError):
        resolve_task_metric(pd.Series([1, 2, 3, 4]), TaskConfig(metric="accuracy"))


def test_suggest_metric_defaults():
    assert suggest_metric("binary").value == "roc_auc"
    assert suggest_metric("multiclass").value == "log_loss"
    assert suggest_metric("regression").value == "rmse"


def test_rmsle_rejects_negatives():
    with pytest.raises(TaskMetricError, match="rmsle"):
        get_metric("rmsle")([1.0, -1.0], [1.0, 1.0])


def test_custom_metric_registration():
    name = "custom_abs_gap"
    register_metric(name, lambda t, p: float(np.max(np.abs(np.asarray(t) - np.asarray(p)))), ["regression"], "minimize")
    assert name in list_metrics(Task.REGRESSION)
    resolved = resolve_task_metric(REG, TaskConfig("regression", name))
    assert resolved.metric(REG.to_numpy(), REG.to_numpy() + 1) == pytest.approx(1.0)
    with pytest.raises(TaskMetricError, match="already registered"):
        register_metric(name, lambda t, p: 0.0, ["regression"], "minimize")
    register_metric(name, lambda t, p: 0.0, ["regression"], "minimize", overwrite=True)


def test_register_metric_validation():
    with pytest.raises(TaskMetricError, match="built in"):
        register_metric("rmse", lambda t, p: 0.0, ["regression"], "minimize", overwrite=True)
    with pytest.raises(TaskMetricError) as exc:
        register_metric("bad_one", lambda t, p: 0.0, ["ranking"], "up")
    assert len(exc.value.problems) == 2


def test_task_toml_and_overrides(data_dir):
    plain = load_config(data_dir / "run.toml")
    assert plain.task_config == TaskConfig()
    (data_dir / "task.toml").write_text(
        (data_dir / "run.toml").read_text() + '\n[task]\ntask = "binary"\nmetric = "roc_auc"\n'
    )
    cfg = load_config(data_dir / "task.toml")
    assert cfg.task_config == TaskConfig("binary", "roc_auc")
    assert load_config(data_dir / "task.toml", metric="f1").task_config == TaskConfig("binary", "f1")
    bundle = load_data(cfg)
    resolved = resolve_task_metric(bundle)
    assert resolved.task is Task.BINARY and resolved.metric.name == "roc_auc"


def test_task_toml_errors(data_dir):
    base = (data_dir / "run.toml").read_text()
    (data_dir / "bad.toml").write_text(base + '\n[task]\nkind = "x"\nmetric = 3\n')
    with pytest.raises(DataContractError) as exc:
        load_config(data_dir / "bad.toml")
    msg = str(exc.value)
    assert "unknown keys in [task]" in msg and "[task] 'metric' must be a string" in msg


def test_resolution_does_not_mutate_target():
    target = pd.Series([1, 2, 3, 4, 1, 2])
    before = target.copy()
    suggest_task(target)
    resolve_task_metric(target, TaskConfig("multiclass", "accuracy"))
    pd.testing.assert_series_equal(target, before)
