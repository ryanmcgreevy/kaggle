import hashlib
import json
import socket
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from automl import (
    LoadedRefit,
    SubmissionConfig,
    SubmissionError,
    TrackingConfig,
    TrackingUnavailableError,
    build_submission,
    load_config,
    load_submission,
    refit_and_predict,
    save_submission,
    select_pipeline,
)
from automl.metrics import Task, get_metric
from automl.submission import CSV_FILE, METADATA_FILE
from automl.task import ResolvedTask

HGB = "hist_gradient_boosting"
METRIC = {Task.BINARY: "roc_auc", Task.MULTICLASS: "accuracy", Task.REGRESSION: "rmse"}


def make(task=Task.BINARY, n=80, m=20, seed=0, metric=None):
    rng = np.random.default_rng(seed)

    def frame(rows, offset):
        return pd.DataFrame({"id": np.arange(rows) + offset, "x1": rng.normal(size=rows), "x2": rng.normal(size=rows)})

    train, test = frame(n, 0), frame(m, 1000)
    signal = train["x2"] + train["x1"]
    if task is Task.BINARY:
        train["y"] = (signal > 0).astype(int)
    elif task is Task.MULTICLASS:
        train["y"] = np.array(["lo", "mid", "hi"])[np.digitize(signal, [-0.5, 0.5])]
    else:
        train["y"] = signal
    return train, test, ResolvedTask(task, get_metric(metric or METRIC[task]))


def refit_result(task=Task.BINARY, metric=None):
    train, test, resolved = make(task, metric=metric)
    sel = select_pipeline(estimator=HGB, params={"max_iter": 3}, task=task.value, metric=metric or METRIC[task])
    res = refit_and_predict(sel, train, test, target="y", id_columns=("id",), resolved_task=resolved)
    return res, test


def loaded(res, frame=None, metric="roc_auc"):
    meta = {"selection": {"metric": metric, "label": "x", "source_kind": "explicit"}, "data": {"target": "y", "id_columns": ["id"]}}
    return LoadedRefit(meta, (res.predictions.frame if frame is None else frame).copy())


def digest(path):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(path.iterdir()) if p.is_file()}


# ---- config ----


def test_config_defaults_and_errors():
    cfg = SubmissionConfig()
    assert cfg.column_map is None and cfg.max_reported_rows == 10 and not cfg.allow_unverified_ids
    with pytest.raises(SubmissionError) as exc:
        SubmissionConfig(column_map={"a": 1}, max_reported_rows=0, float_format="", allow_unverified_ids="no")
    assert len(exc.value.problems) == 4


def test_toml_parsing_and_overrides(data_dir):
    run_toml = data_dir / "run.toml"
    base = run_toml.read_text()
    assert load_config(run_toml).submission_config == SubmissionConfig()
    run_toml.write_text(base + '[submission]\nfloat_format = "%.4f"\nallow_unverified_ids = true\n[submission.column_map]\ny = "prediction"\n')
    cfg = load_config(run_toml).submission_config
    assert cfg.float_format == "%.4f" and cfg.allow_unverified_ids and cfg.column_map == {"y": "prediction"}
    assert load_config(run_toml, float_format="%.2f").submission_config.float_format == "%.2f"
    run_toml.write_text(base + "[submission]\nbogus = 1\n")
    with pytest.raises(Exception, match="unknown keys in \\[submission\\]"):
        load_config(run_toml)
    run_toml.write_text(base + "[submission]\nmax_reported_rows = 0\n")
    with pytest.raises(Exception, match="max_reported_rows"):
        load_config(run_toml)


def test_import_without_optional_libraries():
    code = "import sys; import automl; sys.exit(1 if {'mlflow', 'optuna', 'matplotlib'} & set(sys.modules) else 0)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


# ---- schema and ordering ----


def test_binary_probability_sample_in_test_order():
    res, test = refit_result(Task.BINARY)
    sample = pd.DataFrame({"id": test["id"].iloc[::-1].to_numpy(), "y": 0.5})
    test_before, sample_before = test.copy(), sample.copy()
    sub = build_submission(res, test, sample=sample)
    assert list(sub.frame.columns) == ["id", "y"] and list(sub.frame["id"]) == list(test["id"])
    assert sub.frame["y"].dtype == sample["y"].dtype
    np.testing.assert_allclose(sub.frame["y"], res.predictions.frame["proba_1"])
    assert sub.column_mapping == {"id": "test:id", "y": "proba_1"}
    assert sub.id_check["sample_order_matches_test"] is False
    assert any("different order" in n for n in sub.notes)
    pd.testing.assert_frame_equal(test, test_before)
    pd.testing.assert_frame_equal(sample, sample_before)


def test_binary_label_sample_for_label_metric():
    res, test = refit_result(Task.BINARY, metric="accuracy")
    sample = pd.DataFrame({"id": test["id"], "y": 0})
    sub = build_submission(res, test, sample=sample)
    assert sub.column_mapping["y"] == "prediction" and sub.frame["y"].dtype == sample["y"].dtype
    assert list(sub.frame["y"]) == list(res.predictions.frame["prediction"])


def test_multiclass_probability_columns_follow_sample_order():
    res, test = refit_result(Task.MULTICLASS)
    sample = pd.DataFrame({"id": test["id"], "mid": 0.0, "hi": 0.0, "lo": 0.0})
    sub = build_submission(res, test, sample=sample)
    assert list(sub.frame.columns) == ["id", "mid", "hi", "lo"]
    np.testing.assert_allclose(sub.frame["hi"], res.predictions.frame["proba_hi"])
    assert sub.column_mapping["mid"] == "proba_mid"


def test_multiclass_label_sample_and_regression():
    res, test = refit_result(Task.MULTICLASS)
    sample = pd.DataFrame({"id": test["id"], "y": "lo"})
    sub = build_submission(res, test, sample=sample)
    assert list(sub.frame["y"]) == list(res.predictions.frame["prediction"])
    res, test = refit_result(Task.REGRESSION)
    sub = build_submission(res, test, sample=pd.DataFrame({"id": test["id"], "target": 0.0}))
    assert list(sub.columns) == ["id", "target"] and np.isfinite(sub.frame["target"]).all()


def test_loaded_refit_accepted():
    res, test = refit_result(Task.BINARY)
    sample = pd.DataFrame({"id": test["id"], "y": 0.5})
    sub = build_submission(loaded(res), test, sample=sample)
    np.testing.assert_allclose(sub.frame["y"], res.predictions.frame["proba_1"])


def test_refit_out_of_test_order_is_rejected():
    res, test = refit_result()
    shuffled = res.predictions.frame.iloc[::-1].reset_index(drop=True)
    sample = pd.DataFrame({"id": test["id"], "y": 0.5})
    with pytest.raises(SubmissionError, match="do not match the test rows in order"):
        build_submission(loaded(res, shuffled), test, sample=sample)


def test_schema_errors_and_column_map():
    res, test = refit_result(Task.MULTICLASS)
    sample = pd.DataFrame({"id": test["id"], "a": 0.0, "b": 0.0, "c": 0.0})
    with pytest.raises(SubmissionError, match=r"cannot map sample columns \['a', 'b', 'c'\]"):
        build_submission(res, test, sample=sample)
    cmap = {"a": "proba_hi", "b": "proba_lo", "c": "proba_mid"}
    sub = build_submission(res, test, sample=sample, config=SubmissionConfig(column_map=cmap))
    np.testing.assert_allclose(sub.frame["a"], res.predictions.frame["proba_hi"])
    for bad, text in (
        ({"a": "proba_hi", "b": "proba_lo"}, "does not cover"),
        ({**cmap, "zz": "proba_hi"}, "not in the sample submission"),
        ({**cmap, "c": "nope"}, "do not exist"),
    ):
        with pytest.raises(SubmissionError, match=text):
            build_submission(res, test, sample=sample, config=SubmissionConfig(column_map=bad))
    with pytest.raises(SubmissionError, match="no columns besides"):
        build_submission(res, test, sample=pd.DataFrame({"id": test["id"]}))


# ---- identifiers ----


def test_identifier_errors():
    res, test = refit_result()
    base = pd.DataFrame({"id": test["id"], "y": 0.5})
    dup = base.copy()
    dup.loc[1, "id"] = dup.loc[0, "id"]
    with pytest.raises(SubmissionError, match="duplicate identifiers"):
        build_submission(res, test, sample=dup)
    nul = base.copy()
    nul["id"] = nul["id"].astype(float)
    nul.loc[2, "id"] = np.nan
    with pytest.raises(SubmissionError, match="null identifiers"):
        build_submission(res, test, sample=nul)
    other = base.copy()
    other.loc[0, "id"] = 99999
    with pytest.raises(SubmissionError, match=r"only in test \[.*\]; only in sample \['99999'\]"):
        build_submission(res, test, sample=other)
    with pytest.raises(SubmissionError, match="sample submission has 19 rows"):
        build_submission(res, test, sample=base.iloc[:-1])
    with pytest.raises(SubmissionError, match="missing id columns"):
        build_submission(res, test, sample=base.rename(columns={"id": "ident"}))


def test_no_id_columns_requires_opt_out():
    res, test = refit_result()
    frame = res.predictions.frame.drop(columns="id")
    sample = pd.DataFrame({"y": 0.5}, index=range(len(test)))
    with pytest.raises(SubmissionError, match="allow_unverified_ids"):
        build_submission(loaded(res, frame), test, sample=sample, id_columns=())
    sub = build_submission(loaded(res, frame), test, sample=sample, id_columns=(), config=SubmissionConfig(allow_unverified_ids=True))
    assert sub.id_check["verified"] is False and any("unverified" in n for n in sub.notes)


# ---- values ----


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda f: f.assign(proba_1=f["proba_1"].where(f.index != 3)), "null values in 1 rows, first positions: 3"),
        (lambda f: f.assign(proba_1=f["proba_1"].where(f.index != 3, np.inf)), "non-finite values"),
        (lambda f: f.assign(proba_1=f["proba_1"].where(f.index != 4, 1.5)), r"outside \[0, 1\]"),
        (lambda f: f.iloc[:-1], "refit predictions have 19 rows"),
    ],
)
def test_value_errors_write_nothing(tmp_path, mutate, message):
    res, test = refit_result()
    sample = pd.DataFrame({"id": test["id"], "y": 0.5})
    with pytest.raises(SubmissionError, match=message):
        build_submission(loaded(res, mutate(res.predictions.frame)), test, sample=sample, directory=tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_dtype_checks_and_integer_labels():
    res, test = refit_result(Task.BINARY, metric="accuracy")
    sample = pd.DataFrame({"id": test["id"], "y": 0.0})
    assert build_submission(res, test, sample=sample).frame["y"].dtype == np.float64
    lossy = pd.DataFrame({"id": test["id"], "y": 0})
    res2, test2 = refit_result(Task.REGRESSION)
    with pytest.raises(SubmissionError, match="non-integer values"):
        build_submission(res2, test2, sample=pd.DataFrame({"id": test2["id"], "y": 0}))
    assert build_submission(res, test, sample=lossy).frame["y"].dtype == lossy["y"].dtype
    as_text = pd.DataFrame({"id": test["id"], "y": "a"})
    assert set(build_submission(res, test, sample=as_text).frame["y"]) <= {"0", "1"}
    with pytest.raises(SubmissionError, match="non-numeric"):
        build_submission(*refit_result(Task.MULTICLASS), sample=pd.DataFrame({"id": test["id"], "y": 0.0}))


def test_reported_rows_are_capped():
    res, test = refit_result()
    frame = res.predictions.frame.assign(proba_1=2.0)
    sample = pd.DataFrame({"id": test["id"], "y": 0.5})
    with pytest.raises(SubmissionError, match=r"\(\+17 more\)"):
        build_submission(loaded(res, frame), test, sample=sample, config=SubmissionConfig(max_reported_rows=3))


# ---- fallback ----


def test_no_sample_fallback():
    res, test = refit_result(Task.MULTICLASS)
    sub = build_submission(res, test)
    assert sub.columns == ("id", "y") and sub.schema_source == "fallback"
    assert list(sub.frame["y"]) == list(res.predictions.frame["prediction"])
    assert any("unverified" in n for n in sub.notes)
    assert build_submission(res, test, target="label").columns == ("id", "label")


# ---- persistence ----


def test_save_load_round_trip_and_no_network(tmp_path, monkeypatch):
    res, test = refit_result()
    sample = pd.DataFrame({"id": test["id"].iloc[::-1].to_numpy(), "y": 0.5})
    out = tmp_path / "sub"

    def blocked(*a, **k):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    sub = build_submission(res, test, sample=sample, sample_name="sample_submission.csv", directory=out, config=SubmissionConfig(float_format="%.6f"))
    assert sorted(p.name for p in out.iterdir()) == sorted([CSV_FILE, METADATA_FILE]) and sub.directory == out
    loaded_sub = load_submission(out)
    assert list(loaded_sub.frame.columns) == ["id", "y"] and list(loaded_sub.frame["id"]) == list(test["id"])
    np.testing.assert_allclose(loaded_sub.frame["y"], sub.frame["y"], atol=1e-6)
    meta = loaded_sub.metadata
    assert meta["schema_version"] == 1 and meta["sample_file"] == "sample_submission.csv" and meta["n_rows"] == len(test)
    assert "Nothing was submitted" in meta["statement"] and meta["column_mapping"]["y"] == "proba_1"
    assert out.joinpath(CSV_FILE).read_text().splitlines()[0] == "id,y"


def test_overwrite_and_partial_files(tmp_path, monkeypatch):
    res, test = refit_result()
    sample = pd.DataFrame({"id": test["id"], "y": 0.5})
    out = tmp_path / "sub"
    sub = build_submission(res, test, sample=sample, directory=out)
    before = digest(out)
    with pytest.raises(SubmissionError, match="already contains"):
        save_submission(sub, out)
    assert digest(out) == before
    save_submission(sub, out, overwrite=True)

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_csv", boom)
    empty = tmp_path / "empty"
    with pytest.raises(OSError):
        save_submission(sub, empty)
    assert list(empty.iterdir()) == []


def test_load_errors(tmp_path):
    with pytest.raises(SubmissionError, match="not found"):
        load_submission(tmp_path)
    (tmp_path / METADATA_FILE).write_text("{bad")
    with pytest.raises(SubmissionError, match="not valid JSON"):
        load_submission(tmp_path)
    (tmp_path / METADATA_FILE).write_text('{"schema_version": 9}')
    with pytest.raises(SubmissionError, match="schema_version"):
        load_submission(tmp_path)
    (tmp_path / METADATA_FILE).write_text('{"schema_version": 1}')
    with pytest.raises(SubmissionError, match="missing keys"):
        load_submission(tmp_path)


def test_saved_refit_inputs_not_mutated(tmp_path):
    from automl import load_refit

    train, test, resolved = make()
    sel = select_pipeline(estimator=HGB, params={"max_iter": 3}, task="binary", metric="roc_auc")
    rdir = tmp_path / "refit"
    refit_and_predict(sel, train, test, target="y", id_columns=("id",), resolved_task=resolved, directory=rdir)
    before = digest(rdir)
    sub = build_submission(load_refit(rdir), test, sample=pd.DataFrame({"id": test["id"], "y": 0.5}))
    assert digest(rdir) == before and sub.n_rows == len(test)
    assert sub.source["label"] == sel.label


# ---- MLflow ----


def test_absence_raises_before_writing(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "mlflow", None)
    res, test = refit_result()
    with pytest.raises(TrackingUnavailableError, match=r"pip install 'automl\[mlflow\]'"):
        build_submission(res, test, directory=tmp_path / "s", tracking=TrackingConfig())
    assert not (tmp_path / "s").exists()
    with pytest.raises(SubmissionError, match="pass a directory"):
        build_submission(res, test, tracking=TrackingConfig())


def test_mlflow_logging(tmp_path):
    ml = pytest.importorskip("mlflow")
    previous = ml.get_tracking_uri()
    try:
        res, test = refit_result()
        out = tmp_path / "s"
        sub = build_submission(res, test, sample=pd.DataFrame({"id": test["id"], "y": 0.5}), directory=out, tracking=TrackingConfig(tags={"owner": "me"}))
        client = ml.MlflowClient(tracking_uri=f"sqlite:///{out.resolve() / 'mlflow.db'}")
        run = client.get_run(sub.tracking.run_id)
        assert run.data.tags["automl.kind"] == "submission" and run.data.tags["owner"] == "me"
        artifacts = {a.path for a in client.list_artifacts(run.info.run_id)}
        assert {"submission.json", "submission.csv"} <= artifacts
        logged = json.loads(open(client.download_artifacts(run.info.run_id, "submission.json")).read())
        assert logged == json.loads((out / METADATA_FILE).read_text())
        assert ml.active_run() is None
    finally:
        ml.set_tracking_uri(previous)


def test_mlflow_failure_keeps_local_files(tmp_path):
    pytest.importorskip("mlflow")
    res, test = refit_result()
    out = tmp_path / "s"
    blocker = tmp_path / "blocker"
    blocker.write_text("file")
    with pytest.raises(Exception, match="local results are intact"):
        build_submission(res, test, directory=out, tracking=TrackingConfig(tracking_uri=f"sqlite:///{blocker}/sub/mlflow.db"))
    assert (out / CSV_FILE).is_file() and (out / METADATA_FILE).is_file()
