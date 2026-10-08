import json

import numpy as np
import pandas as pd
import pytest

from automl import (
    EdaConfig,
    EdaError,
    Task,
    DataContractError,
    load_config,
    load_data,
    load_eda,
    resolve_task_metric,
    save_eda,
    summarize,
    summarize_bundle,
)

IDS = ("row_id",)


def _run(eda_frames, **kw):
    train, test = eda_frames
    return summarize(train, test, target="y", id_columns=IDS, **kw)


def _col(result, name):
    return next(c for c in result.columns if c.name == name)


def _cmp(result, name):
    return next(c for c in result.train_test.columns if c.name == name)


def test_dataset_and_complete_sections(eda_frames):
    res = _run(eda_frames, task=Task.BINARY)
    train, _ = eda_frames
    assert res.dataset.n_rows == 102 and res.dataset.n_columns == train.shape[1]
    assert res.dataset.dtypes["num"] == "float64"
    assert res.dataset.duplicate_rows == 2 and res.dataset.duplicate_ids == 2
    assert res.dataset.memory_bytes > 0
    assert [c.name for c in res.columns] == ["serial", "num", "cat", "flag", "const", "allnull"]
    assert res.target is not None and res.train_test is not None


def test_missingness_and_flags(eda_frames):
    res = _run(eda_frames)
    num = _col(res, "num")
    assert num.null_count == 4 and num.null_fraction == pytest.approx(4 / 102)
    const, allnull = _col(res, "const"), _col(res, "allnull")
    assert const.constant and not const.all_null
    assert allnull.all_null and not allnull.constant
    assert _col(res, "serial").id_like and not num.id_like and not _col(res, "cat").id_like


def test_numeric_stats_and_histogram(eda_frames):
    res = _run(eda_frames, config=EdaConfig(histogram_bins=7))
    train, _ = eda_frames
    s = train["num"].dropna()
    n = _col(res, "num").numeric
    assert n.count == len(s) and n.mean == pytest.approx(s.mean()) and n.std == pytest.approx(s.std())
    assert n.min == s.min() and n.max == s.max() and n.p50 == pytest.approx(s.median())
    assert n.skew == pytest.approx(s.skew())
    assert sum(n.histogram["counts"]) == len(s) and len(n.histogram["edges"]) == 8


def test_outlier_methods_hand_computed():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100.0]})
    both = summarize(df, config=EdaConfig(zscore_threshold=3.0)).columns[0].numeric.outliers
    # IQR fence is 14.5; the single extreme value masks itself under a z-score of 3 (z ~ 2.84).
    assert both["iqr"]["count"] == 1 and both["iqr"]["upper"] == pytest.approx(14.5)
    assert both["zscore"]["count"] == 0
    z2 = summarize(df, config=EdaConfig(outlier_method="zscore", zscore_threshold=2.0)).columns[0].numeric.outliers
    assert set(z2) == {"zscore"} and z2["zscore"]["count"] == 1
    iqr = summarize(df, config=EdaConfig(outlier_method="iqr")).columns[0].numeric.outliers
    assert set(iqr) == {"iqr"} and iqr["iqr"]["fraction"] == pytest.approx(0.1)


def test_categorical_cardinality_top_n(eda_frames):
    res = _run(eda_frames, config=EdaConfig(top_n=2))
    cat = _col(res, "cat")
    assert cat.unique == 3 and len(cat.categorical.top_values) == 2
    top = cat.categorical.top_values
    assert top[0]["count"] >= top[1]["count"] and 0 < top[0]["fraction"] <= 1
    assert _col(res, "flag").kind == "boolean" and _col(res, "flag").categorical is not None


def test_degenerate_columns_are_json_safe():
    df = pd.DataFrame({"c": [3.0] * 5, "n": [np.nan] * 5, "inf": [1.0, np.inf, 2.0, 3.0, 4.0]})
    res = summarize(df)
    c = res.columns[0].numeric
    assert c.std == 0 and c.histogram is not None and c.outliers["zscore"]["count"] == 0
    n = res.columns[1].numeric
    assert n.count == 0 and n.mean is None and n.histogram is None
    assert res.columns[2].numeric.non_finite_count == 1
    json.dumps(res.to_dict(), allow_nan=False)


@pytest.mark.parametrize(
    "task,y",
    [
        (Task.BINARY, [0] * 7 + [1] * 3),
        (Task.MULTICLASS, [0] * 6 + [1] * 3 + [2]),
        (Task.REGRESSION, [float(i) for i in range(10)]),
        (None, [0] * 7 + [1] * 3),
    ],
)
def test_target_summary(task, y):
    t = summarize(pd.DataFrame({"y": y, "x": range(10)}), target="y", task=task).target
    assert t.unique == len(set(y)) and t.null_count == 0
    if task is None:
        assert t.task is None and t.class_counts is None and t.numeric is None
    elif task is Task.REGRESSION:
        assert t.numeric.count == 10 and t.class_counts is None
    else:
        assert sum(t.class_counts.values()) == 10
        assert t.imbalance_ratio == pytest.approx(max(t.class_counts.values()) / min(t.class_counts.values()))
        assert sum(t.class_fractions.values()) == pytest.approx(1.0)


def test_task_accepts_resolved_task(data_dir):
    cfg = load_config(data_dir / "run.toml", task="binary", metric="roc_auc")
    bundle = load_data(cfg)
    res = summarize_bundle(bundle, task=resolve_task_metric(bundle))
    assert res.target.task == "binary" and res.dataset.duplicate_ids == 0
    assert res.train_test.test_rows == 2 and [c.name for c in res.columns] == ["a", "b"]


def test_test_without_target_and_one_sided_columns(eda_frames):
    train, test = eda_frames
    res = summarize(train, test.assign(extra=1), target="y", id_columns=IDS)
    assert res.train_test.only_in_test == ("extra",) and res.train_test.only_in_train == ()
    res = summarize(train, test.drop(columns="cat"), target="y", id_columns=IDS)
    assert res.train_test.only_in_train == ("cat",)
    assert "y" not in {c.name for c in res.train_test.columns}


def test_train_test_deltas_and_unseen_categories(eda_frames):
    res = _run(eda_frames)
    tt = res.train_test
    assert tt.train_rows == 102 and tt.test_rows == 60
    num = _cmp(res, "num")
    assert num.test_null_fraction == 0 and num.null_delta == pytest.approx(-4 / 102)
    cat = _cmp(res, "cat")
    assert cat.test_only_examples == ("d",) and cat.train_only_examples == ("c",)
    assert cat.test_only_categories == 1 and cat.train_only_categories == 1


def test_drift_flagged_for_shift_not_for_identical(eda_frames):
    res = _run(eda_frames)
    num = _cmp(res, "num")
    assert num.test_name == "ks" and num.drift is True and num.mean_shift > 0.3  # train std is inflated by planted outliers
    cat = _cmp(res, "cat")
    assert cat.test_name == "chi2" and cat.drift is True
    assert _cmp(res, "const").skipped_reason == "constant column"
    assert _cmp(res, "serial").skipped_reason == "id-like column"
    assert "fewer than 2" in _cmp(res, "allnull").skipped_reason or "kind" in _cmp(res, "allnull").skipped_reason

    train, _ = eda_frames
    same = summarize(train, train.drop(columns="y"), target="y", id_columns=IDS).train_test
    for c in same.columns:
        assert not c.drift


def test_kind_mismatch_is_reported():
    res = summarize(pd.DataFrame({"x": [1.0, 2.0, 3.0]}), pd.DataFrame({"x": ["a", "b", "c"]}))
    assert "dtype kind differs" in res.train_test.columns[0].skipped_reason


def test_inputs_not_mutated(eda_frames):
    train, test = eda_frames
    t0, s0 = train.copy(), test.copy()
    _run(eda_frames, task=Task.BINARY)
    pd.testing.assert_frame_equal(train, t0)
    pd.testing.assert_frame_equal(test, s0)


def test_input_errors(eda_frames):
    train, _ = eda_frames
    with pytest.raises(EdaError) as exc:
        summarize(train, target="nope", id_columns=("zip",))
    assert len(exc.value.problems) == 2 and "'nope'" in str(exc.value) and "'zip'" in str(exc.value)


@pytest.mark.parametrize(
    "kwargs,needle",
    [
        ({"outlier_method": "mad"}, "'outlier_method'"),
        ({"iqr_multiplier": 0}, "'iqr_multiplier'"),
        ({"zscore_threshold": "3"}, "'zscore_threshold'"),
        ({"histogram_bins": 0}, "'histogram_bins'"),
        ({"top_n": 1.5}, "'top_n'"),
        ({"id_like_threshold": 1.5}, "'id_like_threshold'"),
        ({"drift_alpha": 1}, "'drift_alpha'"),
    ],
)
def test_config_errors(kwargs, needle):
    with pytest.raises(EdaError, match=needle):
        EdaConfig(**kwargs)


def test_eda_toml_and_overrides(data_dir):
    assert load_config(data_dir / "run.toml").eda_config == EdaConfig()
    (data_dir / "eda.toml").write_text(
        (data_dir / "run.toml").read_text() + '\n[eda]\noutlier_method = "iqr"\ntop_n = 3\ndrift_alpha = 0.01\n'
    )
    cfg = load_config(data_dir / "eda.toml")
    assert cfg.eda_config == EdaConfig(outlier_method="iqr", top_n=3, drift_alpha=0.01)
    assert load_config(data_dir / "eda.toml", top_n=5).eda_config.top_n == 5
    base = (data_dir / "run.toml").read_text()
    (data_dir / "bad.toml").write_text(base + "\n[eda]\nbins = 3\n")
    with pytest.raises(DataContractError, match=r"unknown keys in \[eda\]"):
        load_config(data_dir / "bad.toml")
    (data_dir / "bad2.toml").write_text(base + "\n[eda]\ntop_n = 0\n")
    with pytest.raises(DataContractError, match="'top_n'"):
        load_config(data_dir / "bad2.toml")


def test_json_safe_and_artifact_round_trip(eda_frames, tmp_path):
    res = _run(eda_frames, task=Task.BINARY)
    d = res.to_dict()
    json.dumps(d, allow_nan=False)
    path = tmp_path / "eda.json"
    save_eda(res, path)
    loaded = load_eda(path)  # fresh call, no in-memory state
    assert loaded == json.loads(json.dumps(d))
    assert loaded["schema_version"] == 1 and loaded["dataset"]["n_rows"] == 102
    assert loaded["target"]["class_counts"] == {"0": 72, "1": 30}
    with pytest.raises(EdaError, match="already exists"):
        save_eda(res, path)


def test_artifact_rejects_bad_files(eda_frames, tmp_path):
    good = tmp_path / "good.json"
    save_eda(_run(eda_frames), good)
    raw = json.loads(good.read_text())

    def write(name, mutate):
        data = json.loads(json.dumps(raw))
        mutate(data)
        p = tmp_path / name
        p.write_text(json.dumps(data))
        return p

    with pytest.raises(EdaError, match="not found"):
        load_eda(tmp_path / "missing.json")
    (tmp_path / "junk.json").write_text("{")
    with pytest.raises(EdaError, match="not valid JSON"):
        load_eda(tmp_path / "junk.json")
    with pytest.raises(EdaError, match="schema_version"):
        load_eda(write("v.json", lambda d: d.update(schema_version=9)))
    with pytest.raises(EdaError, match="missing section 'columns'"):
        load_eda(write("c.json", lambda d: d.pop("columns")))
    with pytest.raises(EdaError, match="'columns' must be"):
        load_eda(write("k.json", lambda d: d.update(columns=[{"name": "x"}])))
    with pytest.raises(EdaError, match="n_rows"):
        load_eda(write("n.json", lambda d: d.update(dataset={})))


def test_readme_example(data_dir):
    cfg = load_config(data_dir / "run.toml", task="binary", metric="roc_auc")
    bundle = load_data(cfg)
    result = summarize_bundle(bundle, task=resolve_task_metric(bundle), config=EdaConfig(top_n=5))
    save_eda(result, data_dir / "eda.json")
    assert load_eda(data_dir / "eda.json")["target"]["task"] == "binary"
