import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from automl import (
    BaselineConfig,
    ModelData,
    TrackingConfig,
    TrackingError,
    TrackingUnavailableError,
    TuningConfig,
    ValidationConfig,
    load_baseline,
    load_config,
    load_tuning,
    make_splits,
    run_baseline,
    save_baseline,
    track_baseline,
    track_families,
    track_tuning,
    tune,
    tune_families,
)
from automl.metrics import Task, get_metric
from automl.task import ResolvedTask
from automl.tracking import _destination, redact_uri

HGB = "hist_gradient_boosting"
SMALL = {"learning_rate": {"type": "float", "low": 0.05, "high": 0.3, "log": True}, "max_iter": {"type": "int", "low": 5, "high": 20}}


def make_frame(n=90, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {"id": np.arange(n), "x1": rng.normal(size=n), "x2": rng.normal(size=n), "cat": rng.choice(["a", "b", "c"], n)}
    )
    df.loc[:4, "x1"] = np.nan
    signal = df["x2"].fillna(0) + (df["cat"] == "a")
    df["y"] = (signal + rng.normal(0, 0.5, n) > 0.5).astype(int)
    df.loc[:1, "y"] = [0, 1]
    return df


def setup():
    df = make_frame()
    resolved = ResolvedTask(Task.BINARY, get_metric("roc_auc"))
    return df, resolved, make_splits(df["y"], resolved, ValidationConfig(n_folds=3))


@pytest.fixture
def mlflow():
    ml = pytest.importorskip("mlflow")
    previous = ml.get_tracking_uri()
    yield ml
    ml.set_tracking_uri(previous)


def query(tmp_path, res):
    """Open the local store in a fresh client and return (client, run)."""
    import mlflow as ml

    client = ml.MlflowClient(tracking_uri=f"sqlite:///{tmp_path.resolve() / 'mlflow.db'}")
    return client, client.get_run(res.run_id)


def children(client, res):
    return client.search_runs([res.experiment_id], filter_string=f"tags.mlflow.parentRunId = '{res.run_id}'")


# ---- config and helpers (no MLflow needed) ----


def test_config_defaults_and_errors():
    cfg = TrackingConfig()
    assert cfg.enabled and cfg.experiment_name == "automl" and cfg.max_trials_logged == 200 and cfg.log_model
    with pytest.raises(TrackingError, match="max_trials_logged"):
        TrackingConfig(max_trials_logged=0)
    with pytest.raises(TrackingError, match="credential-like"):
        TrackingConfig(tags={"api_key": "x"})
    with pytest.raises(TrackingError, match="tags"):
        TrackingConfig(tags={"a": 1})
    with pytest.raises(TrackingError, match="experiment_name"):
        TrackingConfig(experiment_name=" ")
    with pytest.raises(TrackingError, match="log_model"):
        TrackingConfig(log_model="yes")


def test_redact_uri():
    assert redact_uri("postgresql://bob:hunter2@host/db") == "postgresql://bob:***@host/db"
    assert redact_uri("sqlite:////tmp/x.db") == "sqlite:////tmp/x.db"
    assert "hunter2" not in redact_uri("failed: http://u:hunter2@h:5000/x")


def test_destination_resolution(tmp_path):
    uri, loc = _destination(TrackingConfig(), tmp_path / "run")
    assert uri.startswith("sqlite:///") and uri.endswith("mlflow.db") and loc.endswith("mlartifacts")
    assert (tmp_path / "run").is_dir()
    assert _destination(TrackingConfig(tracking_uri="http://h:5000"), None) == ("http://h:5000", None)
    assert _destination(TrackingConfig(directory=str(tmp_path / "d")), None)[0].endswith("d/mlflow.db")
    with pytest.raises(TrackingError, match="no tracking destination"):
        _destination(TrackingConfig(), None)


def test_toml_tracking_section(tmp_path):
    (tmp_path / "run.toml").write_text(
        'version = 1\n[data]\ntrain = "t.csv"\ntest = "s.csv"\ntarget = "y"\n'
        '[tracking]\nexperiment_name = "e"\nmax_trials_logged = 5\n[tracking.tags]\nowner = "me"\n'
    )
    cfg = load_config(tmp_path / "run.toml").tracking_config
    assert cfg.experiment_name == "e" and cfg.max_trials_logged == 5 and cfg.tags == {"owner": "me"}
    (tmp_path / "bad.toml").write_text(
        'version = 1\n[data]\ntrain = "t.csv"\ntest = "s.csv"\ntarget = "y"\n[tracking]\nbogus = 1\n'
    )
    with pytest.raises(Exception, match=r"unknown keys in \[tracking\]: \['bogus'\]"):
        load_config(tmp_path / "bad.toml")
    (tmp_path / "bad2.toml").write_text(
        'version = 1\n[data]\ntrain = "t.csv"\ntest = "s.csv"\ntarget = "y"\n[tracking]\nmax_trials_logged = 0\n'
    )
    with pytest.raises(Exception, match="max_trials_logged"):
        load_config(tmp_path / "bad2.toml")
    assert load_config(tmp_path / "run.toml", ).tuning_config is None


def test_import_does_not_load_mlflow():
    code = "import sys, automl; sys.exit(1 if 'mlflow' in sys.modules else 0)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_absence_raises_before_work(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "mlflow", None)
    df, resolved, splits = setup()
    # the baseline path works without mlflow when tracking is not requested
    res = run_baseline(df, "y", ("id",), resolved, splits)
    with pytest.raises(TrackingUnavailableError, match=r"mlflow.*pip install 'automl\[mlflow\]'"):
        run_baseline(df, "y", ("id",), resolved, splits, tracking=TrackingConfig(directory=str(tmp_path / "t")))
    with pytest.raises(TrackingUnavailableError):
        track_baseline(res, TrackingConfig(directory=str(tmp_path / "t")))
    assert not (tmp_path / "t").exists()


# ---- baseline ----


def test_baseline_logged_and_queried(mlflow, tmp_path):
    df, resolved, splits = setup()
    before = df.copy()
    tdir = tmp_path / "run"
    cfg = TrackingConfig(directory=str(tdir), tags={"owner": "me"})
    res = run_baseline(df, "y", ("id",), resolved, splits, BaselineConfig(save_models=False), tracking=cfg)
    pd.testing.assert_frame_equal(df, before)
    assert (tdir / "mlflow.db").is_file()

    import mlflow as ml

    client = ml.MlflowClient(tracking_uri=f"sqlite:///{tdir.resolve() / 'mlflow.db'}")
    exp = client.get_experiment_by_name("automl")
    (run,) = client.search_runs([exp.experiment_id])
    assert run.data.metrics["mean_score"] == pytest.approx(res.mean)
    assert run.data.metrics["std_score"] == pytest.approx(res.std)
    hist = client.get_metric_history(run.info.run_id, "fold_score")
    assert [h.value for h in sorted(hist, key=lambda h: h.step)] == pytest.approx([f.score for f in res.folds])
    assert run.data.params["config.seed"] == "42" and run.data.params["metric"] == "roc_auc"
    assert run.data.tags["automl.kind"] == "baseline" and run.data.tags["owner"] == "me"
    assert "automl.version.scikit-learn" in run.data.tags and "not leaderboard" in run.data.tags["automl.notes"]
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert artifacts == ["baseline.json"] or "baseline.json" in artifacts
    # only summaries are logged: no predictions or data files
    assert not any(a.endswith(".csv") for a in artifacts)
    assert run.data.tags["automl.model_uri"]


def test_baseline_model_reproduces_predictions(mlflow, tmp_path):
    import mlflow as ml

    from automl.tracking import _fit_pipeline

    df, resolved, splits = setup()
    cfg = BaselineConfig(save_models=False)
    res = run_baseline(df, "y", ("id",), resolved, splits, cfg)
    out = track_baseline(res, TrackingConfig(directory=str(tmp_path)), data=ModelData(df, "y", ("id",), resolved))
    assert out.model_uri
    ml.set_tracking_uri(f"sqlite:///{tmp_path.resolve() / 'mlflow.db'}")  # model URIs resolve through the store
    loaded = ml.sklearn.load_model(out.model_uri)
    pipeline, X = _fit_pipeline(
        ModelData(df, "y", ("id",), resolved), res.estimator_key, cfg.estimator_params, cfg.seed, cfg.n_jobs,
        cfg.device, cfg.native_categoricals, res.preprocessing_config, None,
    )
    np.testing.assert_array_equal(loaded.predict(X), pipeline.predict(X))
    # the logged model holds preprocessing and estimator together and accepts raw feature rows
    assert [n for n, _ in loaded.steps] == ["pre", "model"]


def test_posthoc_from_directory_skips_model_with_note(mlflow, tmp_path):
    df, resolved, splits = setup()
    res = run_baseline(df, "y", ("id",), resolved, splits)
    save_baseline(res, tmp_path / "b")
    out = track_baseline(tmp_path / "b", TrackingConfig())
    assert out.model_uri is None and any("no training data" in n for n in out.notes)
    client, run = query(tmp_path / "b", out)
    assert run.data.metrics["mean_score"] == pytest.approx(load_baseline(tmp_path / "b").mean)
    assert "no training data" in run.data.tags["automl.notes"]
    assert [a.path for a in client.list_artifacts(out.run_id)] == ["baseline.json"]
    saved = json.loads((tmp_path / "b" / "baseline.json").read_text())
    local = client.download_artifacts(out.run_id, "baseline.json")
    assert json.loads(open(local).read()) == saved


def test_log_model_false_and_mismatched_data(mlflow, tmp_path):
    df, resolved, splits = setup()
    res = run_baseline(df, "y", ("id",), resolved, splits)
    out = track_baseline(res, TrackingConfig(directory=str(tmp_path), log_model=False), data=ModelData(df, "y", ("id",), resolved))
    assert out.model_uri is None and any("'log_model' is false" in n for n in out.notes)
    with pytest.raises(TrackingError, match="rows"):
        track_baseline(res, TrackingConfig(directory=str(tmp_path)), data=ModelData(df.iloc[:50], "y", ("id",), resolved))


# ---- tuning ----


def test_tuning_parent_and_nested_trials(mlflow, tmp_path):
    df, resolved, splits = setup()
    tdir = tmp_path / "t"
    res = tune(df, "y", ("id",), resolved, splits, TuningConfig(n_trials=3), space=SMALL, directory=tdir, tracking=TrackingConfig())
    import mlflow as ml

    client = ml.MlflowClient(tracking_uri=f"sqlite:///{tdir.resolve() / 'mlflow.db'}")
    exp = client.get_experiment_by_name("automl")
    runs = client.search_runs([exp.experiment_id])
    (parent,) = [r for r in runs if r.data.tags.get("automl.kind") == "tuning"]
    kids = client.search_runs([exp.experiment_id], filter_string=f"tags.mlflow.parentRunId = '{parent.info.run_id}'")
    assert len(kids) == 3
    saved = load_tuning(tdir)
    by_no = {int(k.info.run_name.split("-")[1]): k for k in kids}
    for t in saved.trials:
        k = by_no[t.number]
        assert k.data.metrics["score"] == pytest.approx(t.score)
        assert k.data.tags["automl.trial_state"] == t.state
        assert float(k.data.params["param.learning_rate"]) == pytest.approx(t.params["learning_rate"])
    assert parent.data.metrics["best_score"] == pytest.approx(res.best.score)
    assert parent.data.tags["automl.best_trial"] == str(res.best.number)
    assert [by_no[res.best.number].data.tags["automl.is_best"]] == ["true"]
    assert "tuning.json" in [a.path for a in client.list_artifacts(parent.info.run_id)]
    assert parent.data.tags["automl.model_uri"]
    ml.set_tracking_uri(f"sqlite:///{tdir.resolve() / 'mlflow.db'}")
    loaded = ml.sklearn.load_model(parent.data.tags["automl.model_uri"])
    assert loaded.predict(df[["x1", "x2", "cat"]]).shape == (len(df),)


def test_trial_cap_keeps_best_and_records_truncation(mlflow, tmp_path):
    df, resolved, splits = setup()
    res = tune(df, "y", ("id",), resolved, splits, TuningConfig(n_trials=5), space=SMALL, directory=tmp_path / "t")
    out = track_tuning(res, TrackingConfig(max_trials_logged=2, log_model=False))
    client, run = query(tmp_path / "t", out)
    assert len(out.child_run_ids) == 2
    assert run.data.tags["automl.trials_not_logged"] == "3" and run.data.tags["automl.trials_logged"] == "2"
    logged = {int(children_run.info.run_name.split("-")[1]) for children_run in children(client, out)}
    assert res.best.number in logged
    assert json.loads(open(client.download_artifacts(out.run_id, "tuning.json")).read())["trials"].__len__() == 5


def test_failed_trials_tagged_never_best(mlflow, tmp_path):
    df, resolved, splits = setup()

    def space(trial):
        lr = trial.suggest_float("learning_rate", 0.05, 0.3)
        if trial.number == 0:
            raise ValueError("bad learning rate parameter")
        return {"learning_rate": lr, "max_iter": 5}

    res = tune(df, "y", ("id",), resolved, splits, TuningConfig(n_trials=3), space=space, directory=tmp_path / "t")
    out = track_tuning(res, TrackingConfig(log_model=False))
    client, run = query(tmp_path / "t", out)
    by_no = {int(k.info.run_name.split("-")[1]): k for k in children(client, out)}
    assert by_no[0].data.tags["automl.trial_state"] == "failed"
    assert "bad learning rate" in by_no[0].data.tags["automl.failure_reason"]
    assert by_no[0].data.tags["automl.is_best"] == "false" and "score" not in by_no[0].data.metrics


def test_posthoc_tuning_directory_and_resume_logs_new_trials(mlflow, tmp_path):
    from automl import resume_tuning

    df, resolved, splits = setup()
    tune(df, "y", ("id",), resolved, splits, TuningConfig(n_trials=2), space=SMALL, directory=tmp_path / "t")
    out = track_tuning(tmp_path / "t", TrackingConfig())
    assert len(out.child_run_ids) == 2 and out.model_uri is None
    res = resume_tuning(
        tmp_path / "t", df, "y", ("id",), resolved, splits, n_trials=2, space=SMALL,
        tracking=TrackingConfig(experiment_name="resumed", log_model=False),
    )
    assert len(res.trials) == 4
    import mlflow as ml

    client = ml.MlflowClient(tracking_uri=f"sqlite:///{(tmp_path / 't').resolve() / 'mlflow.db'}")
    exp = client.get_experiment_by_name("resumed")
    runs = client.search_runs([exp.experiment_id])
    parent = next(r for r in runs if r.data.tags.get("automl.kind") == "tuning")
    assert parent.data.tags["automl.since_trial"] == "2"
    assert len([r for r in runs if r.data.tags.get("automl.kind") == "trial"]) == 2


# ---- families ----


def test_families_nested_runs(mlflow, tmp_path):
    df, resolved, splits = setup()
    fdir = tmp_path / "f"
    res = tune_families(
        df, "y", ("id",), resolved, splits, TuningConfig(n_trials=2), families=[HGB], directory=fdir, tracking=TrackingConfig()
    )
    import mlflow as ml

    client = ml.MlflowClient(tracking_uri=f"sqlite:///{fdir.resolve() / 'mlflow.db'}")
    exp = client.get_experiment_by_name("automl")
    runs = client.search_runs([exp.experiment_id])
    top = next(r for r in runs if r.data.tags.get("automl.kind") == "tuning_families")
    assert top.data.tags["automl.best_family"] == res.best_family
    fam = [r for r in runs if r.data.tags.get("automl.family") == HGB]
    assert len(fam) == 1 and fam[0].data.tags["mlflow.parentRunId"] == top.info.run_id
    trials = [r for r in runs if r.data.tags.get("automl.kind") == "trial"]
    assert len(trials) == 2 and all(t.data.tags["mlflow.parentRunId"] == fam[0].info.run_id for t in trials)
    assert top.data.params["rank.hist_gradient_boosting"] == "1"
    assert top.data.tags["automl.model_uri"]
    assert "tuning_families.json" in [a.path for a in client.list_artifacts(top.info.run_id)]
    again = track_families(fdir, TrackingConfig(experiment_name="posthoc"))
    assert again.model_uri is None and len(again.child_run_ids) == 3


# ---- privacy and failures ----


def test_no_raw_data_in_logged_content(mlflow, tmp_path):
    df, resolved, splits = setup()
    df["secret_marker"] = ["SENTINEL_VALUE"] * len(df)
    res = run_baseline(df, "y", ("id",), resolved, splits)
    out = track_baseline(res, TrackingConfig(directory=str(tmp_path)), data=ModelData(df, "y", ("id",), resolved))
    client, run = query(tmp_path, out)
    blob = json.dumps({"p": run.data.params, "t": run.data.tags, "m": run.data.metrics})
    assert "SENTINEL_VALUE" not in blob
    local = client.download_artifacts(out.run_id, "baseline.json")
    assert "SENTINEL_VALUE" not in open(local).read()
    assert [a.path for a in client.list_artifacts(out.run_id)] == ["baseline.json", "model"] or "oof" not in str(
        [a.path for a in client.list_artifacts(out.run_id)]
    )
    assert not list((tmp_path).rglob("input_example*"))


def test_unreachable_store_is_wrapped_and_leaves_local_results(mlflow, tmp_path):
    df, resolved, splits = setup()
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    tdir = tmp_path / "t"
    cfg = TrackingConfig(tracking_uri=f"sqlite:///{blocker}/sub/mlflow.db")
    with pytest.raises(TrackingError, match="local results are intact"):
        tune(df, "y", ("id",), resolved, splits, TuningConfig(n_trials=2), space=SMALL, directory=tdir, tracking=cfg)
    saved = load_tuning(tdir)  # local results exist and load without MLflow
    assert len(saved.trials) == 2
    before = (tdir / "tuning.json").read_text()
    with pytest.raises(TrackingError):
        track_tuning(tdir, cfg)
    assert (tdir / "tuning.json").read_text() == before


def test_error_message_redacts_credentials(mlflow, tmp_path):
    df, resolved, splits = setup()
    res = run_baseline(df, "y", ("id",), resolved, splits)
    cfg = TrackingConfig(tracking_uri="postgresql://bob:hunter2@127.0.0.1:1/none")
    with pytest.raises(TrackingError) as err:
        track_baseline(res, cfg)
    assert "hunter2" not in str(err.value) and "***" in str(err.value)


def test_failure_mid_run_leaves_no_running_run(mlflow, tmp_path, monkeypatch):
    import automl.tracking as tr

    df, resolved, splits = setup()
    res = run_baseline(df, "y", ("id",), resolved, splits)

    def boom(*a, **k):
        raise RuntimeError("write failed")

    monkeypatch.setattr(tr, "save_baseline", boom)
    out_dir = tmp_path / "t"
    with pytest.raises(TrackingError, match="write failed"):
        track_baseline(res, TrackingConfig(directory=str(out_dir)))
    import mlflow as ml

    client = ml.MlflowClient(tracking_uri=f"sqlite:///{out_dir.resolve() / 'mlflow.db'}")
    exp = client.get_experiment_by_name("automl")
    runs = client.search_runs([exp.experiment_id])
    assert runs and all(r.info.status != "RUNNING" for r in runs)
    assert ml.active_run() is None
