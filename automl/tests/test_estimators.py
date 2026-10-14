import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from automl import (
    BaselineConfig,
    BaselineError,
    EstimatorUnavailableError,
    PreprocessingConfig,
    ValidationConfig,
    build_preprocessor,
    estimator_available,
    get_estimator_spec,
    infer_columns,
    list_estimators,
    load_baseline,
    load_config,
    make_splits,
    resolve_device,
    run_baseline,
    save_baseline,
)
from automl import estimators as est_mod
from automl.errors import DataContractError
from automl.estimators import FeatureLayout
from automl.metrics import Task, get_metric
from automl.task import ResolvedTask
from test_baseline import make_frame

ADAPTERS = ["lightgbm", "xgboost", "catboost"]
DEFAULT_METRIC = {"binary": "roc_auc", "multiclass": "log_loss", "regression": "rmse"}


def need(name):
    spec = get_estimator_spec(name)
    if spec.package:
        pytest.importorskip(spec.package)
    return spec


def run(estimator, task="binary", metric=None, df=None, n_folds=3, strategy="kfold", **cfg):
    df = make_frame(task) if df is None else df
    resolved = ResolvedTask(Task(task), get_metric(metric or DEFAULT_METRIC[task]))
    splits = make_splits(df["y"], resolved, ValidationConfig(strategy=strategy, n_folds=n_folds))
    result = run_baseline(df, "y", ("id",), resolved, splits, BaselineConfig(estimator=estimator, **cfg))
    return df, resolved, splits, result


def test_registry():
    assert list_estimators() == ["hist_gradient_boosting", "lightgbm", "xgboost", "catboost"]
    for name in list_estimators():
        spec = get_estimator_spec(name)
        assert spec.tasks == frozenset(Task) and spec.supports_native_categoricals
    assert estimator_available("hist_gradient_boosting")
    assert get_estimator_spec("hist_gradient_boosting").native_default is False
    assert all(get_estimator_spec(n).native_default for n in ADAPTERS)
    with pytest.raises(BaselineError, match="unknown estimator 'nope'.*lightgbm"):
        get_estimator_spec("nope")


def test_import_does_not_load_adapter_libraries():
    code = "import sys, automl; bad=[m for m in ('lightgbm','xgboost','catboost') if m in sys.modules]; assert not bad, bad"
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("name", ADAPTERS)
def test_absent_library_gives_install_hint(name, monkeypatch):
    monkeypatch.setitem(sys.modules, name, None)
    assert not estimator_available(name)
    df = make_frame("binary")
    with pytest.raises(EstimatorUnavailableError, match=rf"'{name}'.*pip install 'automl\[{name}\]'"):
        run(name, df=df)
    # The default estimator is unaffected.
    assert np.isfinite(run("hist_gradient_boosting", df=df)[3].mean)


@pytest.mark.parametrize(
    "kwargs,fragment",
    [
        ({"estimator": "nope"}, "'estimator'"),
        ({"native_categoricals": "yes"}, "'native_categoricals'"),
        ({"device": "tpu"}, "'device'"),
        ({"estimator_params": {"random_seed": 1}}, "cannot set 'random_seed'; use the 'seed' setting"),
        ({"estimator_params": {"thread_count": 2}}, "use the 'n_jobs' setting"),
        ({"estimator_params": {"device_type": "gpu"}}, "use the 'device' setting"),
        ({"estimator_params": {"cat_features": [0]}}, "use the 'native_categoricals' setting"),
        ({"estimator_params": {"verbose": 1}}, "'verbose'; it is managed by automl"),
        ({"estimator_params": {"early_stopping_rounds": 5}}, "managed by automl"),
        ({"estimator_params": {"allow_writing_files": True}}, "managed by automl"),
    ],
)
def test_config_errors(kwargs, fragment):
    with pytest.raises(BaselineError, match=fragment):
        BaselineConfig(**kwargs)


def test_toml_parsing(data_dir):
    path = data_dir / "run.toml"
    base = path.read_text()
    assert load_config(path).baseline_config.estimator == "hist_gradient_boosting"
    path.write_text(base + '[baseline]\nestimator = "lightgbm"\ndevice = "auto"\nnative_categoricals = false\n')
    cfg = load_config(path).baseline_config
    assert (cfg.estimator, cfg.device, cfg.native_categoricals) == ("lightgbm", "auto", False)
    assert load_config(path, estimator="xgboost").baseline_config.estimator == "xgboost"
    path.write_text(base + '[baseline]\nestimator = "nope"\n')
    with pytest.raises(DataContractError, match="estimator"):
        load_config(path)


def test_native_preprocessing_layout_and_leakage():
    df = make_frame("binary")
    plan = infer_columns(df, "y", ("id",))
    pre = build_preprocessor(plan, PreprocessingConfig(encoder="onehot"), "binary", native_categoricals=True)
    train, valid = df.iloc[:60][["x1", "x2", "cat"]].copy(), df.iloc[60:][["x1", "x2", "cat"]].copy()
    train["cat"] = train["cat"].replace("c", "a")  # category 'c' only appears outside the train rows
    valid.iloc[0, 2] = "c"
    valid.iloc[1, 2] = None
    pre.fit(train)
    out = pre.transform(valid)
    assert out.shape[1] == 3 and set(pre.get_feature_names_out()) == {"num__x1", "num__x2", "cat__cat"}
    assert np.isnan(out[0, 2]) and np.isnan(out[1, 2])
    assert set(out[2:, 2][~np.isnan(out[2:, 2])]) <= {0.0, 1.0}
    layout = FeatureLayout(len(plan.numeric), len(plan.categorical), True)
    assert layout.categorical_idx == (2,) and FeatureLayout(2, 1, False).categorical_idx == ()


@pytest.mark.parametrize("name", ADAPTERS + ["hist_gradient_boosting"])
@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
def test_smoke_per_adapter(name, task):
    need(name)
    _, _, _, res = run(name, task)
    assert np.isfinite([f.score for f in res.folds]).all() and np.isfinite(res.mean)
    assert res.estimator_key == name and res.device["effective"] == "cpu"
    assert res.categorical_handling == ("encoded" if name == "hist_gradient_boosting" else "native")
    if name != "hist_gradient_boosting":
        assert res.versions[name]


@pytest.mark.parametrize("name", ADAPTERS)
def test_metric_inputs_and_class_consistency(name):
    need(name)
    df = make_frame("multiclass")
    _, _, _, res = run(name, "multiclass", "log_loss", df=df)
    assert {"proba_hi", "proba_lo", "proba_mid"} <= set(res.oof.columns)
    probs = res.oof[["proba_hi", "proba_lo", "proba_mid"]]
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    _, _, _, labels = run(name, "multiclass", "macro_f1", df=df)
    assert set(labels.oof["prediction"]) <= {"hi", "lo", "mid"}
    ints = df.copy()
    ints["y"] = ints["y"].map({"lo": 0, "mid": 1, "hi": 2})
    assert np.isfinite(run(name, "multiclass", "log_loss", df=ints)[3].mean)
    _, _, _, binary = run(name, "binary", "roc_auc")
    assert binary.oof["prediction"].between(0, 1).all()
    assert set(run(name, "binary", "accuracy")[3].oof["prediction"]) <= {0, 1}


@pytest.mark.parametrize("name", ADAPTERS + ["hist_gradient_boosting"])
def test_native_categoricals_handle_unseen_and_missing(name):
    need(name)
    df = make_frame("binary")
    df["cat"] = df["cat"].astype(object)
    df.loc[10:14, "cat"] = None
    df.loc[85, "cat"] = "rare-only-here"
    resolved = ResolvedTask(Task.BINARY, get_metric("roc_auc"))
    splits = make_splits(df["y"], resolved, ValidationConfig(strategy="holdout"))
    idx = splits.splits[0]
    df.loc[idx.valid_idx[0], "cat"] = "never-seen-in-train"
    res = run_baseline(
        df, "y", ("id",), resolved, splits, BaselineConfig(estimator=name, native_categoricals=True)
    )
    assert res.categorical_handling == "native" and np.isfinite(res.mean)


def test_native_ignores_encoder_setting_and_is_recorded():
    need("lightgbm")
    df = make_frame("binary")
    resolved = ResolvedTask(Task.BINARY, get_metric("roc_auc"))
    splits = make_splits(df["y"], resolved, ValidationConfig(n_folds=3))
    cfg = BaselineConfig(estimator="lightgbm")
    a = run_baseline(df, "y", ("id",), resolved, splits, cfg, PreprocessingConfig(encoder="onehot"))
    b = run_baseline(df, "y", ("id",), resolved, splits, cfg, PreprocessingConfig(encoder="ordinal"))
    assert a.mean == b.mean and a.categorical_handling == "native"
    enc = run_baseline(df, "y", ("id",), resolved, splits, BaselineConfig(estimator="lightgbm", native_categoricals=False),
                       PreprocessingConfig(encoder="onehot"))
    assert enc.categorical_handling == "encoded"
    assert enc.models[0].named_steps["pre"].get_feature_names_out().size > 3


@pytest.mark.parametrize("name", ADAPTERS)
def test_estimator_params_take_effect_and_unknown_rejected(name):
    need(name)
    key = "iterations" if name == "catboost" else "n_estimators"
    _, _, _, res = run(name, estimator_params={key: 7})
    assert res.estimator_params[key] == 7 and res.config.estimator_params == {key: 7}
    assert res.estimator_params.get("random_state", res.estimator_params.get("random_seed")) == 42
    with pytest.raises(BaselineError, match="'bogus'"):
        run(name, estimator_params={"bogus": 1})


@pytest.mark.parametrize("name", ADAPTERS)
def test_cpu_repeatable_and_leakage_safe(name):
    need(name)
    df = make_frame("binary")
    a, b = run(name, df=df)[3], run(name, df=df)[3]
    assert a.mean == b.mean and a.oof.equals(b.oof)
    resolved = ResolvedTask(Task.BINARY, get_metric("roc_auc"))
    _, _, splits, base = run(name, df=df, strategy="holdout")
    s = splits.splits[0]
    mutated = df.copy()
    mutated.loc[s.valid_idx, "x2"] = 1e6
    mutated.loc[s.valid_idx, "cat"] = "novel"
    mutated.loc[s.valid_idx, "y"] = 1 - mutated.loc[s.valid_idx, "y"]
    other = run_baseline(mutated, "y", ("id",), resolved, splits, BaselineConfig(estimator=name))
    rows = df.iloc[s.train_idx][["x1", "x2", "cat"]]
    np.testing.assert_allclose(base.models[0].predict_proba(rows), other.models[0].predict_proba(rows))
    assert not {"id", "y", "const"} & set(base.models[0].named_steps["pre"].feature_names_in_)


@pytest.mark.parametrize("name", ADAPTERS)
def test_no_files_written(name, tmp_path, monkeypatch):
    need(name)
    monkeypatch.chdir(tmp_path)
    run(name)
    assert list(tmp_path.iterdir()) == []


@pytest.fixture
def probe_cache(monkeypatch):
    monkeypatch.setattr(est_mod, "_PROBES", {})
    return est_mod._PROBES


def test_device_cpu_and_unsupported_estimator(probe_cache):
    hgb = get_estimator_spec("hist_gradient_boosting")
    assert resolve_device(hgb, "cpu").effective == "cpu"
    auto = resolve_device(hgb, "auto")
    assert auto.effective == "cpu" and "does not support GPU" in auto.reason
    with pytest.raises(EstimatorUnavailableError, match="does not support GPU training"):
        resolve_device(hgb, "gpu")
    _, _, _, res = run("hist_gradient_boosting", device="auto")
    assert res.device["requested"] == "auto" and res.device["effective"] == "cpu" and res.device["note"] is None


@pytest.mark.parametrize("name", ADAPTERS)
def test_device_fallback_and_error_with_failed_probe(name, probe_cache):
    spec = need(name)
    probe_cache[name] = (False, "GPU probe fit failed for 'x': no device")
    choice = resolve_device(spec, "auto")
    assert choice.effective == "cpu" and "fell back to cpu" in choice.reason and "no device" in choice.reason
    with pytest.raises(EstimatorUnavailableError, match="no device"):
        resolve_device(spec, "gpu")
    _, _, _, res = run(name, device="auto")
    assert res.device["effective"] == "cpu"


@pytest.mark.parametrize(
    "name,key,value",
    [("lightgbm", "device_type", "gpu"), ("xgboost", "device", "cuda"), ("catboost", "task_type", "GPU")],
)
def test_gpu_parameter_mapping_with_mocked_probe(name, key, value, probe_cache):
    spec = need(name)
    probe_cache[name] = (True, "GPU probe fit succeeded")
    assert resolve_device(spec, "gpu").effective == "gpu"
    est = spec.make(Task.REGRESSION, {}, 1, 1, "gpu", FeatureLayout(2, 0))
    _, params = est_mod.describe_estimator(est)
    assert params[key] == value
    cpu = est_mod.describe_estimator(spec.make(Task.REGRESSION, {}, 1, 1, "cpu", FeatureLayout(2, 0)))[1]
    assert cpu.get(key) != value


@pytest.mark.parametrize("name", ADAPTERS)
def test_real_gpu_smoke(name):
    spec = need(name)
    if resolve_device(spec, "auto").effective != "gpu":
        pytest.skip("no usable GPU for this library")
    _, _, _, res = run(name, device="gpu")
    assert res.device["effective"] == "gpu" and "bit-reproducible" in res.device["note"]
    assert np.isfinite(res.mean)


@pytest.mark.parametrize("name", ADAPTERS)
def test_artifact_round_trip_and_reloaded_models(name, tmp_path):
    need(name)
    df, _, splits, res = run(name, "multiclass")
    out = tmp_path / "run"
    save_baseline(res, out)
    payload = json.loads((out / "baseline.json").read_text())
    assert payload["schema_version"] == 2 and payload["estimator"]["key"] == name
    assert payload["categorical_handling"] == "native" and payload["device"]["effective"] == "cpu"
    loaded = load_baseline(out, load_models=True)
    assert loaded.estimator_key == name and loaded.config == res.config and loaded.device == res.device
    for model, s in zip(loaded.models, splits.splits):
        rows = loaded.oof[loaded.oof["fold"] == s.fold].sort_values("row")
        cols = [f"proba_{c}" for c in model.named_steps["model"].classes_]
        proba = model.predict_proba(df.iloc[rows["row"]][["x1", "x2", "cat"]])
        np.testing.assert_allclose(proba, rows[cols], atol=1e-9)


def test_version_1_artifact_still_loads(tmp_path):
    _, _, _, res = run("hist_gradient_boosting")
    out = tmp_path / "run"
    save_baseline(res, out)
    path = out / "baseline.json"
    data = json.loads(path.read_text())
    data["schema_version"] = 1
    for key in ("categorical_handling", "device"):
        data.pop(key)
    data["estimator"].pop("key")
    for key in ("estimator", "native_categoricals", "device"):
        data["config"].pop(key)
    path.write_text(json.dumps(data))
    loaded = load_baseline(out)
    assert loaded.estimator_key == "hist_gradient_boosting" and loaded.categorical_handling == "encoded"
    assert loaded.device == {} and loaded.config == BaselineConfig()


def test_readme_example(tmp_path):
    need("lightgbm")
    df = make_frame("binary")
    resolved = ResolvedTask(Task.BINARY, get_metric("roc_auc"))
    splits = make_splits(df["y"], resolved, ValidationConfig(n_folds=3))
    cfg = BaselineConfig(estimator="lightgbm", native_categoricals=True, device="auto", estimator_params={"n_estimators": 50})
    result = run_baseline(df, "y", ("id",), resolved, splits, cfg)
    save_baseline(result, tmp_path / "lgbm")
    assert load_baseline(tmp_path / "lgbm").estimator_key == "lightgbm"
