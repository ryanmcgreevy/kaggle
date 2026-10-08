import numpy as np
import pandas as pd
import pytest

from automl import (
    DataContractError,
    ResolvedTask,
    Task,
    ValidationConfig,
    ValidationSplitError,
    get_metric,
    load_config,
    load_splits,
    make_splits,
    save_splits,
)


def _binary(n=100):
    return pd.Series([0] * (n * 7 // 10) + [1] * (n - n * 7 // 10))


def _multiclass(n=90):
    return pd.Series(np.repeat([0, 1, 2], n // 3))


def _regression(n=50):
    return pd.Series(np.linspace(0.0, 10.0, n))


def test_repeatable_and_seed_sensitive():
    y = _binary()
    a = make_splits(y, Task.BINARY, ValidationConfig(seed=1))
    b = make_splits(y, Task.BINARY, ValidationConfig(seed=1))
    c = make_splits(y, Task.BINARY, ValidationConfig(seed=2))
    assert a.splits == b.splits
    assert a.splits != c.splits


@pytest.mark.parametrize("y,task", [(_binary(), Task.BINARY), (_multiclass(), Task.MULTICLASS), (_regression(), Task.REGRESSION)])
def test_kfold_no_overlap_and_partition(y, task):
    res = make_splits(y, task, ValidationConfig(n_folds=4))
    assert len(res.splits) == 4
    for s in res.splits:
        assert not set(s.train_idx) & set(s.valid_idx)
        assert len(s.train_idx) + len(s.valid_idx) == len(y)
    all_valid = np.concatenate([s.valid_idx for s in res.splits])
    assert sorted(all_valid) == list(range(len(y)))


@pytest.mark.parametrize("strategy", ["kfold", "holdout"])
@pytest.mark.parametrize("y,task", [(_binary(), Task.BINARY), (_multiclass(), Task.MULTICLASS)])
def test_stratification_preserves_proportions(y, task, strategy):
    res = make_splits(y, task, ValidationConfig(strategy=strategy))
    assert res.stratified
    overall = y.value_counts(normalize=True)
    for s in res.splits:
        valid = y.iloc[s.valid_idx].value_counts(normalize=True)
        assert (valid - overall).abs().max() < 0.05


def test_regression_not_stratified_and_accepts_resolved_task():
    resolved = ResolvedTask(Task.REGRESSION, get_metric("rmse"))
    res = make_splits(_regression(), resolved)
    assert not res.stratified and res.task is Task.REGRESSION


def test_holdout_fraction():
    res = make_splits(_regression(100), Task.REGRESSION, ValidationConfig(strategy="holdout", holdout_fraction=0.25))
    (s,) = res.splits
    assert len(s.valid_idx) == 25 and len(s.train_idx) == 75


@pytest.mark.parametrize(
    "kwargs,needle",
    [
        ({"strategy": "loo"}, "'strategy'"),
        ({"n_folds": 1}, "'n_folds'"),
        ({"n_folds": "5"}, "'n_folds'"),
        ({"holdout_fraction": 1.5}, "'holdout_fraction'"),
        ({"holdout_fraction": 0}, "'holdout_fraction'"),
        ({"seed": 1.5}, "'seed'"),
    ],
)
def test_config_errors(kwargs, needle):
    with pytest.raises(ValidationSplitError, match=needle):
        ValidationConfig(**kwargs)


def test_config_collects_all_problems():
    with pytest.raises(ValidationSplitError) as exc:
        ValidationConfig(strategy="x", n_folds=1)
    assert len(exc.value.problems) == 2


def test_rare_class_no_silent_fallback():
    y = pd.Series([0] * 20 + [1] * 3)
    with pytest.raises(ValidationSplitError) as exc:
        make_splits(y, Task.BINARY, ValidationConfig(n_folds=5))
    msg = str(exc.value)
    assert "class 1" in msg and "3 member" in msg and "n_folds" in msg
    # Regression with same sizes is unaffected.
    make_splits(pd.Series(np.arange(23.0)), Task.REGRESSION, ValidationConfig(n_folds=5))


def test_holdout_rare_class_and_class_count_errors():
    with pytest.raises(ValidationSplitError, match="class 1 has 1 member"):
        make_splits(pd.Series([0] * 20 + [1]), Task.BINARY, ValidationConfig(strategy="holdout"))
    y = pd.Series(np.repeat(np.arange(10), 2))
    with pytest.raises(ValidationSplitError, match="per class"):
        make_splits(y, Task.MULTICLASS, ValidationConfig(strategy="holdout", holdout_fraction=0.2))


def test_null_target_and_too_few_rows():
    with pytest.raises(ValidationSplitError, match="2 null"):
        make_splits(pd.Series([1.0, None, None, 4.0, 5.0, 6.0]), Task.REGRESSION)
    with pytest.raises(ValidationSplitError, match=r"'n_folds' \(5\) exceeds"):
        make_splits(pd.Series([1.0, 2.0, 3.0]), Task.REGRESSION)
    with pytest.raises(ValidationSplitError, match="'holdout_fraction'"):
        make_splits(pd.Series([1.0]), Task.REGRESSION, ValidationConfig(strategy="holdout"))


def test_target_not_mutated():
    y = _binary()
    before = y.copy()
    make_splits(y, Task.BINARY)
    pd.testing.assert_series_equal(y, before)


def test_validation_toml_and_overrides(data_dir):
    assert load_config(data_dir / "run.toml").validation_config == ValidationConfig()
    (data_dir / "val.toml").write_text(
        (data_dir / "run.toml").read_text() + "\n[validation]\nstrategy = \"holdout\"\nseed = 3\nholdout_fraction = 0.3\n"
    )
    cfg = load_config(data_dir / "val.toml")
    assert cfg.validation_config == ValidationConfig("holdout", 3, 5, 0.3)
    assert cfg.seed == 7
    over = load_config(data_dir / "val.toml", strategy="kfold", n_folds=3)
    assert over.validation_config == ValidationConfig("kfold", 3, 3, 0.3)


def test_validation_toml_errors(data_dir):
    base = (data_dir / "run.toml").read_text()
    (data_dir / "bad.toml").write_text(base + "\n[validation]\nfolds = 3\n")
    with pytest.raises(DataContractError, match=r"unknown keys in \[validation\]"):
        load_config(data_dir / "bad.toml")
    (data_dir / "bad2.toml").write_text(base + "\n[validation]\nn_folds = 1\n")
    with pytest.raises(DataContractError, match="'n_folds'"):
        load_config(data_dir / "bad2.toml")


def test_artifact_round_trip_and_stage_boundary(tmp_path):
    y = _multiclass()
    res = make_splits(y, Task.MULTICLASS, ValidationConfig(n_folds=3, seed=9))
    path = tmp_path / "splits.json"
    save_splits(res, path)
    loaded = load_splits(path)  # no in-memory state from make_splits
    assert loaded == res
    with pytest.raises(ValidationSplitError, match="already exists"):
        save_splits(res, path)


def test_artifact_rejects_bad_files(tmp_path):
    import json

    res = make_splits(_regression(), Task.REGRESSION, ValidationConfig(n_folds=2))
    good = tmp_path / "good.json"
    save_splits(res, good)
    raw = json.loads(good.read_text())

    def write(name, mutate):
        data = json.loads(json.dumps(raw))
        mutate(data)
        p = tmp_path / name
        p.write_text(json.dumps(data))
        return p

    with pytest.raises(ValidationSplitError, match="not found"):
        load_splits(tmp_path / "missing.json")
    (tmp_path / "junk.json").write_text("{")
    with pytest.raises(ValidationSplitError, match="not valid JSON"):
        load_splits(tmp_path / "junk.json")
    with pytest.raises(ValidationSplitError, match="schema_version"):
        load_splits(write("v.json", lambda d: d.update(schema_version=99)))
    with pytest.raises(ValidationSplitError, match="out of range|in \\[0"):
        load_splits(write("oor.json", lambda d: d["splits"][0]["valid_idx"].append(10_000)))
    with pytest.raises(ValidationSplitError, match="overlapping"):
        load_splits(write("ov.json", lambda d: d["splits"][0]["train_idx"].append(d["splits"][0]["valid_idx"][0])))
    with pytest.raises(ValidationSplitError, match="duplicate"):
        load_splits(write("dup.json", lambda d: d["splits"][0]["train_idx"].append(d["splits"][0]["train_idx"][0])))
    with pytest.raises(ValidationSplitError, match="covers"):
        load_splits(write("cov.json", lambda d: d["splits"][0]["train_idx"].pop()))
    with pytest.raises(ValidationSplitError, match="n_train"):
        load_splits(write("cnt.json", lambda d: d["splits"][0].update(n_train=1)))
    with pytest.raises(ValidationSplitError, match="exactly once"):
        load_splits(write("part.json", lambda d: d["splits"][1].update(valid_idx=d["splits"][0]["valid_idx"], train_idx=d["splits"][0]["train_idx"])))
    with pytest.raises(ValidationSplitError, match="expected 2 split"):
        load_splits(write("n.json", lambda d: d["splits"].pop()))
    with pytest.raises(ValidationSplitError, match="missing or invalid"):
        load_splits(write("t.json", lambda d: d.update(task="ranking")))


def test_readme_example(data_dir):
    from automl import load_data, resolve_task_metric

    cfg = load_config(data_dir / "run.toml", task="binary", metric="roc_auc", n_folds=2)
    bundle = load_data(cfg)
    resolved = resolve_task_metric(bundle)
    y = pd.concat([bundle.train["y"]] * 3, ignore_index=True)
    result = make_splits(y, resolved, cfg.validation_config)
    save_splits(result, data_dir / "splits.json")
    assert load_splits(data_dir / "splits.json") == result
