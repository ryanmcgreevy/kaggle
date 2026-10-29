import json

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from automl import (
    BaselineConfig,
    BaselineError,
    PreprocessingConfig,
    ValidationConfig,
    list_metrics,
    load_baseline,
    load_config,
    load_splits,
    make_splits,
    run_baseline,
    save_baseline,
    save_splits,
)
from automl.errors import DataContractError
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
            "const": 1.0,
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


def run(task="binary", metric=None, strategy="kfold", n_folds=3, df=None, **kw):
    df = make_frame(task) if df is None else df
    metric = metric or {"binary": "roc_auc", "multiclass": "log_loss", "regression": "rmse"}[task]
    resolved = ResolvedTask(Task(task), get_metric(metric))
    splits = make_splits(df["y"], resolved, ValidationConfig(strategy=strategy, n_folds=n_folds))
    return df, resolved, splits, run_baseline(df, "y", ("id",), resolved, splits, **kw)


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
def test_each_task_default_metric(task):
    df, _, _, res = run(task)
    assert len(res.folds) == 3
    assert np.isfinite([f.score for f in res.folds]).all() and np.isfinite(res.mean) and res.std >= 0
    assert res.estimator.startswith("HistGradientBoosting")
    assert res.estimator_params["random_state"] == 42 and res.estimator_params["early_stopping"] is False
    assert res.preprocessing_config.encoder == "ordinal"
    assert res.split["strategy"] == "kfold" and res.split["n_folds"] == 3
    assert not res.oof.drop(columns=["row", "fold"]).isna().any().any()


def test_progress_is_opt_in_for_python_api(capsys, monkeypatch):
    df = make_frame("binary")
    resolved = ResolvedTask(Task.BINARY, get_metric("roc_auc"))
    splits = make_splits(df["y"], resolved, ValidationConfig(n_folds=3))
    events = []
    from sklearn.ensemble import HistGradientBoostingClassifier

    original_fit = HistGradientBoostingClassifier.fit
    fitted = []

    def checked_fit(model, X, y, **kwargs):
        assert any(event.event == "fold.start" for event in events)
        fitted.append(True)
        return original_fit(model, X, y, **kwargs)

    monkeypatch.setattr(HistGradientBoostingClassifier, "fit", checked_fit)
    result = run_baseline(df, "y", ("id",), resolved, splits, progress=events.append)
    captured = capsys.readouterr()

    assert captured.out == captured.err == ""
    assert result.mean == pytest.approx(events[-1].details["score"])
    assert [event.event for event in events].count("fold.start") == len(splits.splits)
    assert [event.event for event in events].count("fold.complete") == len(splits.splits)
    assert events[0].event == "stage.start" and events[-1].event == "stage.complete"
    assert all(event.elapsed_seconds >= 0 for event in events)
    assert len(fitted) == len(splits.splits) + 1
    assert any(event.event == "full_fit.complete" for event in events)


def test_progress_does_not_change_parallel_baseline_scores():
    df = make_frame("binary")
    resolved = ResolvedTask(Task.BINARY, get_metric("roc_auc"))
    splits = make_splits(df["y"], resolved, ValidationConfig(n_folds=3))
    config = BaselineConfig(n_jobs=2)
    expected = run_baseline(df, "y", ("id",), resolved, splits, config)
    observed = run_baseline(df, "y", ("id",), resolved, splits, config, progress=lambda event: None)

    assert [fold.score for fold in observed.folds] == [fold.score for fold in expected.folds]
    pd.testing.assert_frame_equal(observed.oof, expected.oof)


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
def test_every_builtin_metric(task):
    for name in list_metrics(task):
        _, _, _, res = run(task, name)
        assert np.isfinite(res.mean), name
        assert res.metric == name and res.direction == get_metric(name).direction


def test_prediction_kinds():
    assert run("binary", "roc_auc")[3].prediction_kind == "proba_positive"
    assert run("binary", "accuracy")[3].prediction_kind == "labels"
    multi = run("multiclass", "log_loss")[3]
    assert multi.prediction_kind == "proba"
    assert {"proba_lo", "proba_mid", "proba_hi"} <= set(multi.oof.columns)
    assert run("regression")[3].prediction_kind == "values"
    pos = run("binary", "roc_auc")[3].oof["prediction"]
    assert pos.between(0, 1).all()


def test_kfold_oof_covers_every_row_once():
    df, _, _, res = run("binary")
    assert sorted(res.oof["row"]) == list(range(len(df)))


def test_holdout_oof_only_validation_rows():
    df, _, splits, res = run("regression", strategy="holdout")
    assert len(res.folds) == 1 and res.std == 0.0
    assert set(res.oof["row"]) == set(splits.splits[0].valid_idx.tolist())


def test_scores_use_validation_rows_only():
    df, resolved, splits, res = run("regression")
    for f, s in zip(res.folds, splits.splits):
        rows = res.oof[res.oof["fold"] == s.fold].sort_values("row")
        expected = resolved.metric(df["y"].iloc[rows["row"]], rows["prediction"])
        assert f.score == pytest.approx(expected)
        assert f.n_valid == len(s.valid_idx) and f.n_train == len(s.train_idx)


def test_repeatable_and_split_seed_varies():
    df = make_frame("binary")
    a = run("binary", df=df)[3]
    b = run("binary", df=df)[3]
    assert a.mean == b.mean and a.oof.equals(b.oof)
    resolved = ResolvedTask(Task.BINARY, get_metric("roc_auc"))
    other = make_splits(df["y"], resolved, ValidationConfig(n_folds=3, seed=1))
    assert run_baseline(df, "y", ("id",), resolved, other).mean != a.mean


def test_target_and_id_excluded():
    _, _, _, res = run("binary")
    pre = res.models[0].named_steps["pre"]
    assert list(pre.feature_names_in_) == ["x1", "x2", "cat"]


def test_no_leakage_from_validation_rows():
    df, resolved, splits, a = run("binary", strategy="holdout")
    s = splits.splits[0]
    mutated = df.copy()
    mutated.loc[s.valid_idx, "x2"] = 1e6
    mutated.loc[s.valid_idx, "cat"] = "novel"
    mutated.loc[s.valid_idx, "y"] = 1 - mutated.loc[s.valid_idx, "y"]
    b = run_baseline(mutated, "y", ("id",), resolved, splits)
    rows = df.iloc[s.train_idx][["x1", "x2", "cat"]]
    np.testing.assert_allclose(a.models[0].predict_proba(rows), b.models[0].predict_proba(rows))
    # Preprocessor fit equals fitting on train rows alone.
    alone = clone(a.models[0].named_steps["pre"]).fit(rows, df["y"].iloc[s.train_idx])
    np.testing.assert_allclose(alone.transform(rows), a.models[0].named_steps["pre"].transform(rows))


def test_inputs_not_mutated():
    df = make_frame("binary")
    before = df.copy()
    run("binary", df=df)
    pd.testing.assert_frame_equal(df, before)


def test_custom_preprocessing_and_estimator_params():
    cfg = BaselineConfig(estimator_params={"max_iter": 20})
    _, _, _, res = run("binary", config=cfg, preprocessing_config=PreprocessingConfig(encoder="onehot"))
    assert res.estimator_params["max_iter"] == 20 and res.preprocessing_config.encoder == "onehot"


def test_parallel_matches_serial():
    df = make_frame("binary")
    assert run("binary", df=df, config=BaselineConfig(n_jobs=2))[3].mean == run("binary", df=df)[3].mean


def test_input_errors():
    df, resolved, splits, _ = run("binary")
    with pytest.raises(BaselineError, match="'nope' is not in the train data"):
        run_baseline(df, "nope", ("id",), resolved, splits)
    with pytest.raises(BaselineError, match="splits were built for 90 rows but train has 10"):
        run_baseline(df.iloc[:10], "y", ("id",), resolved, splits)
    reg = ResolvedTask(Task.REGRESSION, get_metric("rmse"))
    with pytest.raises(BaselineError, match="task 'binary' but the resolved task is 'regression'"):
        run_baseline(df, "y", ("id",), reg, splits)
    with pytest.raises(BaselineError, match="'bogus'"):
        run_baseline(df, "y", ("id",), resolved, splits, BaselineConfig(estimator_params={"bogus": 1}))


def test_class_set_mismatch_names_fold():
    df, resolved, splits, _ = run("multiclass")
    bad = df.copy()
    s = splits.splits[0]
    # A class present only in the validation rows of fold 0.
    bad["y"] = bad["y"].astype(object)
    bad.loc[s.valid_idx[0], "y"] = "only-here"
    with pytest.raises(BaselineError, match="fold 0: class sets differ"):
        run_baseline(bad, "y", ("id",), resolved, splits)


def test_nonfinite_metric_or_prediction_errors():
    df = make_frame("regression")
    df["y"] = df["y"] - 100  # negative targets break rmsle
    resolved = ResolvedTask(Task.REGRESSION, get_metric("rmsle"))
    splits = make_splits(df["y"], resolved, ValidationConfig(n_folds=3))
    with pytest.raises(BaselineError, match="fold 0: metric 'rmsle' failed"):
        run_baseline(df, "y", ("id",), resolved, splits)


@pytest.mark.parametrize(
    "kwargs,fragment",
    [
        ({"seed": True}, "'seed'"),
        ({"n_jobs": 0}, "'n_jobs'"),
        ({"save_models": "yes"}, "'save_models'"),
        ({"estimator_params": {"random_state": 1}}, "cannot set 'random_state'"),
        ({"estimator_params": [1]}, "'estimator_params'"),
    ],
)
def test_config_errors(kwargs, fragment):
    with pytest.raises(BaselineError, match=fragment):
        BaselineConfig(**kwargs)


def test_toml_parsing_and_overrides(data_dir):
    run_toml = data_dir / "run.toml"
    base = run_toml.read_text()
    assert load_config(run_toml).baseline_config == BaselineConfig()
    run_toml.write_text(base + "[baseline]\nn_jobs = 2\nsave_models = false\n[baseline.estimator_params]\nmax_iter = 10\n")
    cfg = load_config(run_toml).baseline_config
    assert (cfg.n_jobs, cfg.save_models, cfg.estimator_params) == (2, False, {"max_iter": 10})
    assert load_config(run_toml, n_jobs=3).baseline_config.n_jobs == 3
    run_toml.write_text(base + "[baseline]\nbogus = 1\n")
    with pytest.raises(DataContractError, match="unknown keys in \\[baseline\\]"):
        load_config(run_toml)
    run_toml.write_text(base + "[baseline]\nn_jobs = 0\n")
    with pytest.raises(DataContractError, match="n_jobs"):
        load_config(run_toml)


def test_artifact_round_trip_and_stage_boundary(tmp_path):
    df, resolved, splits, res = run("multiclass")
    save_splits(splits, tmp_path / "splits.json")
    out = tmp_path / "run"
    save_baseline(res, out)
    assert sorted(p.name for p in out.iterdir()) == [
        "baseline.json",
        "fitted-model",
        "models.joblib",
        "oof_predictions.csv",
    ]
    with pytest.raises(BaselineError, match="already exists"):
        save_baseline(res, out)

    loaded = load_baseline(out)
    assert loaded.models == ()
    assert [f.score for f in loaded.folds] == [f.score for f in res.folds]
    assert (loaded.mean, loaded.std, loaded.plan, loaded.preprocessing_config, loaded.config) == (
        res.mean, res.std, res.plan, res.preprocessing_config, res.config
    )
    pd.testing.assert_frame_equal(loaded.oof, res.oof.astype(loaded.oof.dtypes.to_dict()))

    # Reloaded fold models reproduce the saved OOF predictions on validation rows.
    with_models = load_baseline(out, load_models=True)
    for model, s in zip(with_models.models, load_splits(tmp_path / "splits.json").splits):
        rows = loaded.oof[loaded.oof["fold"] == s.fold].sort_values("row")
        proba = model.predict_proba(df.iloc[rows["row"]][["x1", "x2", "cat"]])
        np.testing.assert_allclose(proba, rows[[f"proba_{c}" for c in model.named_steps["model"].classes_]], atol=1e-9)


def test_artifact_has_no_absolute_paths_and_save_models_false(tmp_path):
    _, _, _, res = run("binary", config=BaselineConfig(save_models=False))
    out = tmp_path / "run"
    save_baseline(res, out)
    assert not (out / "models.joblib").exists()
    text = (out / "baseline.json").read_text()
    assert str(tmp_path) not in text and json.loads(text)["files"]["models"] is None
    with pytest.raises(BaselineError, match="no models file was saved"):
        load_baseline(out, load_models=True)


def test_save_models_without_models(tmp_path):
    _, _, _, res = run("binary")
    saved = load_baseline_dir(res, tmp_path)
    with pytest.raises(BaselineError, match="holds no fitted models"):
        save_baseline(saved, tmp_path / "second")


def load_baseline_dir(res, tmp_path):
    save_baseline(res, tmp_path / "first")
    return load_baseline(tmp_path / "first")


def test_load_rejects_malformed(tmp_path):
    _, _, _, res = run("binary")
    out = tmp_path / "run"
    save_baseline(res, out)
    good = json.loads((out / "baseline.json").read_text())

    def variant(name, mutate=None, oof=None):
        d = tmp_path / name
        d.mkdir()
        data = json.loads(json.dumps(good))
        if mutate:
            mutate(data)
        (d / "baseline.json").write_text(json.dumps(data))
        csv = (out / "oof_predictions.csv").read_text() if oof is None else oof
        (d / "oof_predictions.csv").write_text(csv)
        return d

    with pytest.raises(BaselineError, match="baseline.json not found"):
        load_baseline(tmp_path / "missing")
    (tmp_path / "nj").mkdir()
    (tmp_path / "nj" / "baseline.json").write_text("{")
    with pytest.raises(BaselineError, match="not valid JSON"):
        load_baseline(tmp_path / "nj")
    with pytest.raises(BaselineError, match="schema_version"):
        load_baseline(variant("v", lambda d: d.update(schema_version=9)))
    with pytest.raises(BaselineError, match="missing or invalid field"):
        load_baseline(variant("m", lambda d: d.pop("folds")))
    with pytest.raises(BaselineError, match="summary mean"):
        load_baseline(variant("s", lambda d: d["summary"].update(mean=123.0)))
    with pytest.raises(BaselineError, match="prediction count"):
        load_baseline(variant("c", oof="row,fold,prediction\n0,0,0.5\n"))
    with pytest.raises(BaselineError, match="'row' and 'fold'"):
        load_baseline(variant("h", oof="a,b\n1,2\n"))
    with pytest.raises(BaselineError, match="preprocessing"):
        load_baseline(variant("p", lambda d: d["preprocessing"].update(schema_version=9)))
    with pytest.raises(BaselineError, match="out-of-fold predictions file is missing"):
        load_baseline(variant("f", lambda d: d["files"].update(oof_predictions="gone.csv")))


def test_readme_example(tmp_path):
    df = make_frame("binary")
    resolved = ResolvedTask(Task.BINARY, get_metric("roc_auc"))
    splits = make_splits(df["y"], resolved, ValidationConfig(n_folds=3))
    result = run_baseline(df, "y", ("id",), resolved, splits)
    save_baseline(result, tmp_path / "baseline")
    assert load_baseline(tmp_path / "baseline").mean == pytest.approx(result.mean)
