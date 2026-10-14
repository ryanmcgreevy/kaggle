import hashlib
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from automl import (
    BaselineConfig,
    RefitConfig,
    RefitError,
    TrackingConfig,
    TrackingUnavailableError,
    TuningConfig,
    ValidationConfig,
    estimator_available,
    load_config,
    load_refit,
    make_splits,
    predict,
    refit,
    refit_and_predict,
    run_baseline,
    save_baseline,
    save_refit,
    select_pipeline,
    tune,
    tune_families,
)
from automl.metrics import Task, get_metric
from automl.refit import MODEL_FILE, PREDICTIONS_FILE, METADATA_FILE
from automl.task import ResolvedTask

HGB = "hist_gradient_boosting"
SMALL = {"learning_rate": {"type": "float", "low": 0.05, "high": 0.3, "log": True}, "max_iter": {"type": "int", "low": 5, "high": 20}}
FAST = BaselineConfig(estimator_params={"max_iter": 10})
METRIC = {Task.BINARY: "roc_auc", Task.MULTICLASS: "accuracy", Task.REGRESSION: "rmse"}


def make(task=Task.BINARY, n=90, m=25, seed=0):
    rng = np.random.default_rng(seed)

    def frame(rows, offset):
        df = pd.DataFrame(
            {"id": np.arange(rows) + offset, "x1": rng.normal(size=rows), "x2": rng.normal(size=rows), "cat": rng.choice(["a", "b", "c"], rows)}
        )
        df.loc[:2, "x1"] = np.nan
        return df

    train, test = frame(n, 0), frame(m, 1000)
    test.loc[0, "cat"] = "unseen"
    signal = train["x2"].fillna(0) + (train["cat"] == "a")
    if task is Task.BINARY:
        train["y"] = (signal + rng.normal(0, 0.5, n) > 0.5).astype(int)
        train.loc[:1, "y"] = [0, 1]
    elif task is Task.MULTICLASS:
        train["y"] = np.array(["lo", "mid", "hi"])[np.digitize(signal, [0, 1.0])]
        train.loc[:2, "y"] = ["lo", "mid", "hi"]
    else:
        train["y"] = signal + rng.normal(0, 0.1, n)
    resolved = ResolvedTask(task, get_metric(METRIC[task]))
    return train, test, resolved


def baseline_dir(tmp_path, task=Task.BINARY, config=FAST, name="b"):
    train, test, resolved = make(task)
    splits = make_splits(train["y"], resolved, ValidationConfig(n_folds=3))
    res = run_baseline(train, "y", ("id",), resolved, splits, config)
    save_baseline(res, tmp_path / name)
    return train, test, resolved, res, tmp_path / name


def digest(path):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(path.iterdir()) if p.is_file()}


# ---- config ----


def test_config_defaults_and_errors():
    cfg = RefitConfig()
    assert cfg.save_model and cfg.probabilities and cfg.max_reported_rows == 10
    with pytest.raises(RefitError) as exc:
        RefitConfig(save_model="yes", proba_tolerance=2, max_reported_rows=0)
    assert len(exc.value.problems) == 3


def test_toml_parsing_and_overrides(data_dir):
    run_toml = data_dir / "run.toml"
    base = run_toml.read_text()
    assert load_config(run_toml).refit_config == RefitConfig()
    run_toml.write_text(base + "[refit]\nsave_model = false\nmax_reported_rows = 3\n")
    cfg = load_config(run_toml).refit_config
    assert (cfg.save_model, cfg.max_reported_rows) == (False, 3)
    assert load_config(run_toml, max_reported_rows=5).refit_config.max_reported_rows == 5
    run_toml.write_text(base + "[refit]\nbogus = 1\n")
    with pytest.raises(Exception, match="unknown keys in \\[refit\\]"):
        load_config(run_toml)
    run_toml.write_text(base + "[refit]\nproba_tolerance = 5\n")
    with pytest.raises(Exception, match="proba_tolerance"):
        load_config(run_toml)


def test_import_without_optional_libraries():
    code = "import sys; import automl; sys.exit(1 if {'mlflow', 'optuna', 'matplotlib'} & set(sys.modules) else 0)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


# ---- selection ----


def test_select_from_baseline_directory_and_loaded_result(tmp_path):
    _, _, _, res, d = baseline_dir(tmp_path)
    sel = select_pipeline(d)
    assert sel.source_kind == "baseline" and sel.estimator == HGB and sel.task == "binary" and sel.metric == "roc_auc"
    assert sel.params == {"max_iter": 10} and sel.seed == 42
    assert sel.validation_mean == pytest.approx(res.mean) and sel.features == ("x1", "x2", "cat")
    from_obj = select_pipeline(res)
    assert from_obj.params == sel.params and from_obj.source_path == "<in-memory>"
    assert json.dumps(sel.to_dict(), allow_nan=False)


def test_select_rank_label_and_errors(tmp_path):
    _, _, _, _, d1 = baseline_dir(tmp_path, name="b1")
    _, _, _, _, d2 = baseline_dir(tmp_path, config=BaselineConfig(estimator_params={"max_iter": 5}), name="b2")
    sel = select_pipeline([d1, d2])
    assert "selected rank 1" in sel.notes[0]
    second = select_pipeline([d1, d2], rank=2)
    assert second.label.startswith("baseline:") and not second.notes
    assert select_pipeline([d1, d2], label=sel.label).label == sel.label
    with pytest.raises(RefitError, match="no candidate labelled"):
        select_pipeline(d1, label="nope")
    with pytest.raises(RefitError, match="rank"):
        select_pipeline(d1, rank=3)
    with pytest.raises(RefitError, match="not both"):
        select_pipeline(d1, label="x", rank=1)
    with pytest.raises(RefitError, match="not both or neither"):
        select_pipeline()
    with pytest.raises(RefitError, match="is not a directory"):
        select_pipeline(tmp_path / "missing")


def test_explicit_selection_and_errors():
    sel = select_pipeline(estimator=HGB, params={"max_iter": 5}, task="binary", metric="roc_auc")
    assert sel.source_kind == "explicit" and sel.direction == "maximize" and sel.features is None
    with pytest.raises(RefitError) as exc:
        select_pipeline(estimator="nope", task="weird", device="tpu", params={"random_state": 1})
    assert len(exc.value.problems) >= 4
    with pytest.raises(RefitError, match="does not accept|rejected"):
        select_pipeline(estimator=HGB, params={"not_a_param": 1}, task="binary")
    with pytest.raises(RefitError, match="only to a saved result"):
        select_pipeline(estimator=HGB, task="binary", rank=1)


def test_missing_library_hint(monkeypatch):
    from automl import estimators

    spec = estimators.get_estimator_spec("lightgbm")
    monkeypatch.setitem(sys.modules, spec.package, None)
    with pytest.raises(Exception, match=r"pip install 'automl\[lightgbm\]'"):
        select_pipeline(estimator="lightgbm", task="binary")


# ---- refit ----


@pytest.mark.parametrize("task", [Task.BINARY, Task.MULTICLASS, Task.REGRESSION])
def test_refit_predict_all_tasks(task):
    train, test, resolved = make(task)
    sel = select_pipeline(estimator=HGB, params={"max_iter": 10}, task=task.value, metric=METRIC[task])
    fitted = refit(sel, train, target="y", id_columns=("id",), resolved_task=resolved)
    assert fitted.n_train == len(train) and fitted.features == ["x1", "x2", "cat"]
    assert list(fitted.pipeline.named_steps) == ["pre", "model"]
    preds = predict(fitted, test)
    assert len(preds.frame) == len(test) == preds.n_rows
    assert list(preds.frame["id"]) == list(test["id"])
    if task is Task.REGRESSION:
        assert preds.prediction_kind == "values" and np.isfinite(preds.frame["prediction"]).all()
    else:
        classes = sorted(train["y"].unique())
        assert list(fitted.classes) == classes
        assert set(preds.frame["prediction"]) <= set(classes)
        assert preds.prediction_kind == "labels+proba"
        proba = preds.frame[list(preds.proba_columns)]
        assert list(preds.proba_columns) == [f"proba_{c}" for c in classes]
        np.testing.assert_allclose(proba.sum(axis=1), 1, atol=1e-5)
    assert json.dumps(preds.to_dict(), allow_nan=False)


def test_refit_matches_manual_fit_and_ignores_extra_columns(tmp_path):
    train, test, resolved, res, d = baseline_dir(tmp_path)
    fitted = refit(select_pipeline(d), train, target="y", id_columns=("id",), resolved_task=resolved)
    from automl.refit import build_pipeline

    pipe, plan, fit_params, _, _ = build_pipeline(
        train, "y", ("id",), Task.BINARY, HGB, {"max_iter": 10}, 42, 1, "cpu", None, fitted.selection.preprocessing_config
    )
    cols = [*plan.numeric, *plan.categorical]
    pipe.fit(train[cols], train["y"], **fit_params)
    np.testing.assert_allclose(pipe.predict_proba(test[cols]), fitted.pipeline.predict_proba(test[cols]))
    wide = test.assign(junk=1.0)
    np.testing.assert_array_equal(predict(fitted, wide).frame["prediction"], predict(fitted, test).frame["prediction"])
    assert fitted.pipeline.named_steps["model"].get_params()["max_iter"] == 10


def test_refit_uses_all_rows_and_preprocessing_statistics_from_train_only():
    train, test, resolved = make()
    fitted = refit(select_pipeline(estimator=HGB, params={"max_iter": 5}, task="binary"), train, target="y", id_columns=("id",), resolved_task=resolved)
    pre = fitted.pipeline.named_steps["pre"]
    before = np.asarray(pre.transform(train[fitted.features]), dtype=float)
    shifted = test.assign(x1=test["x1"] + 1e6, x2=test["x2"] * 1e3)
    predict(fitted, shifted)
    after = np.asarray(pre.transform(train[fitted.features]), dtype=float)
    np.testing.assert_array_equal(before, after)
    assert fitted.n_train == len(train)


def test_refit_input_errors():
    train, _, resolved = make()
    sel = select_pipeline(estimator=HGB, task="binary", metric="roc_auc")
    with pytest.raises(RefitError, match="not in the train data"):
        refit(sel, train.drop(columns="y"), target="y", resolved_task=resolved)
    with pytest.raises(RefitError, match="null values"):
        refit(sel, train.assign(y=train["y"].where(train.index > 0)), target="y", resolved_task=resolved)
    with pytest.raises(RefitError, match="task 'binary'"):
        refit(sel, train, target="y", resolved_task=ResolvedTask(Task.REGRESSION, get_metric("rmse")))
    with pytest.raises(RefitError, match="metric 'roc_auc'"):
        refit(sel, train, target="y", resolved_task=ResolvedTask(Task.BINARY, get_metric("accuracy")))
    with pytest.raises(RefitError, match="fewer than two classes"):
        refit(sel, train.assign(y=1), target="y", resolved_task=resolved)


def test_refit_rejects_changed_features(tmp_path):
    train, _, resolved, _, d = baseline_dir(tmp_path)
    with pytest.raises(RefitError, match="unexpected"):
        refit(select_pipeline(d), train.assign(x3=np.random.default_rng(1).normal(size=len(train))), target="y", id_columns=("id",), resolved_task=resolved)


def test_cpu_repeatable():
    train, test, resolved = make()
    sel = select_pipeline(estimator=HGB, params={"max_iter": 10}, task="binary")
    a = predict(refit(sel, train, target="y", id_columns=("id",), resolved_task=resolved), test)
    b = predict(refit(sel, train, target="y", id_columns=("id",), resolved_task=resolved), test)
    pd.testing.assert_frame_equal(a.frame, b.frame)


@pytest.mark.parametrize("lib", ["lightgbm", "xgboost", "catboost"])
def test_boosting_adapters_predict(lib):
    pytest.importorskip(lib)
    for task in (Task.BINARY, Task.MULTICLASS, Task.REGRESSION):
        train, test, resolved = make(task)
        params = {"lightgbm": {"n_estimators": 5, "min_child_samples": 2}, "xgboost": {"n_estimators": 5}, "catboost": {"iterations": 5}}[lib]
        sel = select_pipeline(estimator=lib, params=params, task=task.value)
        fitted = refit(sel, train, target="y", id_columns=("id",), resolved_task=resolved)
        preds = predict(fitted, test)
        assert len(preds.frame) == len(test)
        assert set(preds.frame["prediction"]) <= set(train["y"]) or task is Task.REGRESSION


# ---- prediction validation ----


class _Stub:
    def __init__(self, values, proba=None):
        self.values, self.proba = values, proba

    def predict(self, X):
        return np.asarray(self.values)

    def predict_proba(self, X):
        return np.asarray(self.proba)


def _stubbed(task, values, proba=None, classes=(0, 1)):
    train, test, resolved = make(task)
    sel = select_pipeline(estimator=HGB, params={"max_iter": 3}, task=task.value)
    fitted = refit(sel, train, target="y", id_columns=("id",), resolved_task=resolved)
    object.__setattr__(fitted, "pipeline", _Stub(values, proba))
    object.__setattr__(fitted, "classes", tuple(classes))
    return fitted, test


def test_validity_rules():
    n = 25
    good = np.tile([0.3, 0.7], (n, 1))
    fitted, test = _stubbed(Task.BINARY, np.zeros(n, dtype=int), good)
    assert len(predict(fitted, test).frame) == n
    cases = {
        "expected 25 predictions": (np.zeros(n - 1, dtype=int), good),
        "outside the training classes": (np.full(n, 7), good),
        "not finite": (np.zeros(n, dtype=int), np.where(np.arange(n)[:, None] == 2, np.nan, good)),
        "outside \\[0, 1\\]": (np.zeros(n, dtype=int), np.tile([-0.5, 1.5], (n, 1))),
        "do not sum to 1": (np.zeros(n, dtype=int), np.tile([0.3, 0.3], (n, 1))),
    }
    for message, (values, proba) in cases.items():
        fitted, test = _stubbed(Task.BINARY, values, proba)
        with pytest.raises(RefitError, match=message):
            predict(fitted, test)
    for bad in (np.nan, np.inf):
        values = np.ones(n)
        values[3] = bad
        fitted, test = _stubbed(Task.REGRESSION, values)
        with pytest.raises(RefitError, match="not finite.*positions: 3"):
            predict(fitted, test)


def test_reported_rows_are_capped():
    n = 25
    fitted, test = _stubbed(Task.REGRESSION, np.full(n, np.nan))
    with pytest.raises(RefitError, match=r"\(\+22 more\)"):
        predict(fitted, test, config=RefitConfig(max_reported_rows=3))


def test_test_data_errors():
    train, test, resolved = make()
    fitted = refit(select_pipeline(estimator=HGB, params={"max_iter": 3}, task="binary"), train, target="y", id_columns=("id",), resolved_task=resolved)
    with pytest.raises(RefitError, match=r"missing feature columns \['x2'\]"):
        predict(fitted, test.drop(columns="x2"))
    with pytest.raises(RefitError, match="missing id columns"):
        predict(fitted, test.drop(columns="id"))
    with pytest.raises(RefitError, match="no rows"):
        predict(fitted, test.iloc[:0])
    with pytest.raises(RefitError, match="collide"):
        predict(fitted, test.assign(prediction=1), id_columns=("prediction",))


def test_no_probabilities_by_config_or_estimator():
    train, test, resolved = make()
    fitted = refit(select_pipeline(estimator=HGB, params={"max_iter": 3}, task="binary"), train, target="y", id_columns=("id",), resolved_task=resolved)
    preds = predict(fitted, test, config=RefitConfig(probabilities=False))
    assert preds.prediction_kind == "labels" and list(preds.frame.columns) == ["id", "prediction"]
    assert "probabilities" in preds.notes[0]
    fitted, test = _stubbed(Task.BINARY, np.zeros(25, dtype=int))
    object.__setattr__(fitted, "pipeline", type("P", (), {"predict": lambda self, X: np.zeros(len(X), dtype=int)})())
    preds = predict(fitted, test)
    assert preds.prediction_kind == "labels" and "no predict_proba" in preds.notes[0]


def test_string_and_integer_labels_preserved():
    for task, dtype in ((Task.BINARY, "int"), (Task.MULTICLASS, "object")):
        train, test, resolved = make(task)
        fitted = refit(select_pipeline(estimator=HGB, params={"max_iter": 3}, task=task.value), train, target="y", id_columns=("id",), resolved_task=resolved)
        col = predict(fitted, test).frame["prediction"]
        assert col.dtype.kind == ("i" if dtype == "int" else "O") and set(col) <= set(train["y"])


# ---- persistence ----


def test_save_load_round_trip(tmp_path):
    train, test, resolved, _, d = baseline_dir(tmp_path)
    before = digest(d)
    train_before, test_before = train.copy(), test.copy()
    out = tmp_path / "refit"
    res = refit_and_predict(d, train, test, target="y", id_columns=("id",), resolved_task=resolved, directory=out)
    assert digest(d) == before
    pd.testing.assert_frame_equal(train, train_before)
    pd.testing.assert_frame_equal(test, test_before)
    assert sorted(p.name for p in out.iterdir()) == sorted([METADATA_FILE, PREDICTIONS_FILE, MODEL_FILE])
    loaded = load_refit(out, load_model=True)
    pd.testing.assert_frame_equal(loaded.predictions, res.predictions.frame, check_dtype=False)
    meta = loaded.metadata
    assert meta["schema_version"] == 1 and meta["data"]["n_train_rows"] == len(train) and meta["data"]["n_test_rows"] == len(test)
    assert meta["selection"]["source_kind"] == "baseline" and meta["files"]["model"] == MODEL_FILE
    reproduced = loaded.pipeline.predict_proba(test[["x1", "x2", "cat"]])
    np.testing.assert_allclose(reproduced, res.predictions.frame[list(res.predictions.proba_columns)].to_numpy())


def test_metadata_loads_without_estimator_libraries(tmp_path):
    train, test, resolved = make()
    out = tmp_path / "r"
    refit_and_predict(select_pipeline(estimator=HGB, params={"max_iter": 3}, task="binary"), train, test, target="y", id_columns=("id",), resolved_task=resolved, directory=out)
    code = (
        "import sys; sys.modules['lightgbm']=None; sys.modules['xgboost']=None; sys.modules['catboost']=None\n"
        f"from automl import load_refit; r = load_refit({str(out)!r}); print(r.metadata['task'], len(r.predictions))"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0 and done.stdout.split()[0] == "binary"


def test_overwrite_and_no_model(tmp_path):
    train, test, resolved = make()
    sel = select_pipeline(estimator=HGB, params={"max_iter": 3}, task="binary")
    out = tmp_path / "r"
    kw = dict(target="y", id_columns=("id",), resolved_task=resolved)
    refit_and_predict(sel, train, test, directory=out, **kw)
    with pytest.raises(RefitError, match="already contains"):
        refit_and_predict(sel, train, test, directory=out, **kw)
    refit_and_predict(sel, train, test, directory=out, overwrite=True, config=RefitConfig(save_model=False), **kw)
    assert not (out / MODEL_FILE).exists() and load_refit(out).metadata["files"]["model"] is None
    with pytest.raises(RefitError, match="save_model=false"):
        load_refit(out, load_model=True)
    assert [p.name for p in out.iterdir() if p.name.startswith(".")] == []


def test_failed_save_leaves_no_partial_files(tmp_path, monkeypatch):
    train, test, resolved = make()
    fitted = refit(select_pipeline(estimator=HGB, params={"max_iter": 3}, task="binary"), train, target="y", id_columns=("id",), resolved_task=resolved)
    preds = predict(fitted, test)
    import joblib

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(joblib, "dump", boom)
    out = tmp_path / "r"
    with pytest.raises(OSError):
        save_refit(fitted, preds, out)
    assert list(out.iterdir()) == []


def test_load_errors(tmp_path):
    with pytest.raises(RefitError, match="not found"):
        load_refit(tmp_path)
    (tmp_path / METADATA_FILE).write_text("{bad")
    with pytest.raises(RefitError, match="not valid JSON"):
        load_refit(tmp_path)
    (tmp_path / METADATA_FILE).write_text('{"schema_version": 9}')
    with pytest.raises(RefitError, match="schema_version"):
        load_refit(tmp_path)
    (tmp_path / METADATA_FILE).write_text('{"schema_version": 1}')
    with pytest.raises(RefitError, match="missing keys"):
        load_refit(tmp_path)


def test_restart_reproduces_saved_predictions(tmp_path):
    train, test, resolved, _, d = baseline_dir(tmp_path)
    out = tmp_path / "r"
    res = refit_and_predict(d, train, test, target="y", id_columns=("id",), resolved_task=resolved, directory=out)
    code = (
        "import pandas as pd, numpy as np; from automl import load_refit\n"
        f"r = load_refit({str(out)!r}, load_model=True)\n"
        f"t = pd.read_pickle({str(tmp_path / 't.pkl')!r})\n"
        "p = r.pipeline.predict_proba(t[['x1','x2','cat']])\n"
        "print(float(np.abs(p[:,1] - r.predictions['proba_1']).max()))"
    )
    test.to_pickle(tmp_path / "t.pkl")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0 and float(done.stdout) < 1e-9
    assert res.predictions.n_rows == len(test)


# ---- other sources ----


def test_select_from_tuning_and_families(tmp_path):
    pytest.importorskip("optuna")
    train, test, resolved = make()
    splits = make_splits(train["y"], resolved, ValidationConfig(n_folds=3))
    tres = tune(train, "y", ("id",), resolved, splits, TuningConfig(n_trials=3), space=SMALL, directory=tmp_path / "t")
    sel = select_pipeline(tmp_path / "t")
    assert sel.source_kind == "tuning" and sel.trial_number == tres.best.number
    assert sel.params["max_iter"] == tres.best.params["max_iter"]
    fres = tune_families(train, "y", ("id",), resolved, splits, TuningConfig(n_trials=2), families=[HGB], directory=tmp_path / "f")
    fsel = select_pipeline(tmp_path / "f")
    assert fsel.source_kind == "family" and fsel.label == f"family:{HGB}"
    out = refit_and_predict(tmp_path / "f", train, test, target="y", id_columns=("id",), resolved_task=resolved)
    assert out.predictions.n_rows == len(test) and out.directory is None and fres.best_family == HGB
    both = select_pipeline([tmp_path / "t", tmp_path / "f"], rank=2)
    assert both.source_kind in ("tuning", "family")


# ---- MLflow ----


def test_absence_raises_before_fitting(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "mlflow", None)
    train, test, resolved = make()
    sel = select_pipeline(estimator=HGB, params={"max_iter": 3}, task="binary")
    with pytest.raises(TrackingUnavailableError, match=r"pip install 'automl\[mlflow\]'"):
        refit_and_predict(sel, train, test, target="y", id_columns=("id",), resolved_task=resolved, directory=tmp_path / "r", tracking=TrackingConfig())
    assert not (tmp_path / "r").exists()


def test_mlflow_logging(tmp_path):
    ml = pytest.importorskip("mlflow")
    previous = ml.get_tracking_uri()
    try:
        train, test, resolved, _, d = baseline_dir(tmp_path)
        out = tmp_path / "r"
        res = refit_and_predict(
            d, train, test, target="y", id_columns=("id",), resolved_task=resolved, directory=out,
            tracking=TrackingConfig(tags={"owner": "me"}),
        )
        client = ml.MlflowClient(tracking_uri=f"sqlite:///{out.resolve() / 'mlflow.db'}")
        run = client.get_run(res.tracking.run_id)
        tags = run.data.tags
        assert tags["automl.kind"] == "refit" and tags["automl.source_kind"] == "baseline" and tags["owner"] == "me"
        assert run.data.params["selection.estimator"] == HGB and run.data.params["task"] == "binary"
        assert run.data.metrics["validation_mean_score"] == pytest.approx(res.refit.selection.validation_mean)
        artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
        assert "refit.json" in artifacts and not any(a.endswith(".csv") for a in artifacts)
        local = json.loads((out / METADATA_FILE).read_text())
        logged = json.loads(open(client.download_artifacts(run.info.run_id, "refit.json")).read())
        assert logged == local
        assert res.tracking.model_uri
        ml.set_tracking_uri(f"sqlite:///{out.resolve() / 'mlflow.db'}")
        model = ml.sklearn.load_model(res.tracking.model_uri)
        np.testing.assert_allclose(
            model.predict_proba(test[["x1", "x2", "cat"]]), res.predictions.frame[list(res.predictions.proba_columns)].to_numpy()
        )
        assert ml.active_run() is None
    finally:
        ml.set_tracking_uri(previous)


def test_mlflow_failure_keeps_local_results(tmp_path):
    pytest.importorskip("mlflow")
    train, test, resolved = make()
    sel = select_pipeline(estimator=HGB, params={"max_iter": 3}, task="binary")
    out = tmp_path / "r"
    blocker = tmp_path / "blocker"
    blocker.write_text("file")
    with pytest.raises(Exception, match="local results are intact"):
        refit_and_predict(
            sel, train, test, target="y", id_columns=("id",), resolved_task=resolved, directory=out,
            tracking=TrackingConfig(tracking_uri=f"sqlite:///{blocker}/sub/mlflow.db"),
        )
    assert (out / PREDICTIONS_FILE).is_file() and (out / METADATA_FILE).is_file()
