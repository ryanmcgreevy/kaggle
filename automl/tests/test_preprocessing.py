import itertools
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

from automl import (
    ColumnPlan,
    PreprocessingConfig,
    PreprocessingError,
    ValidationConfig,
    build_preprocessor,
    fit_transform_fold,
    infer_columns,
    load_config,
    load_plan,
    make_splits,
    save_plan,
)
from automl.errors import DataContractError


@pytest.fixture
def frame():
    rng = np.random.default_rng(1)
    n = 80
    df = pd.DataFrame(
        {
            "id": np.arange(n),
            "x": rng.normal(size=n),
            "count": rng.integers(0, 1000, n),
            "level": rng.integers(0, 3, n),
            "cat": rng.choice(["a", "b", "c"], n),
            "flag": rng.random(n) > 0.5,
            "const": 1.0,
            "allnull": np.nan,
            "y": np.array([0, 1] * (n // 2)),
        }
    )
    df.loc[:4, "x"] = np.nan
    df.loc[5:9, "cat"] = None
    return df


def plan_for(df, config=None):
    return infer_columns(df, "y", ("id",), config)


def test_inference(frame):
    plan = plan_for(frame)
    assert plan.numeric == ("x", "count")
    assert plan.categorical == ("level", "cat", "flag")
    assert plan.dropped == ("const", "allnull")
    assert plan.excluded == ("id", "y")
    assert "all values are null" in plan.reasons["allnull"]
    assert set(plan.reasons) == {"x", "count", "level", "cat", "flag", "const", "allnull"}


def test_overrides(frame):
    cfg = PreprocessingConfig(
        numeric_columns=("level",), categorical_columns=("count",), drop_columns=("cat",)
    )
    plan = plan_for(frame, cfg)
    assert plan.numeric == ("x", "level")
    assert "count" in plan.categorical and "cat" in plan.dropped


@pytest.mark.parametrize("bad", ["y", "id"])
def test_target_and_id_in_overrides_rejected(frame, bad):
    with pytest.raises(PreprocessingError, match=f"'{bad}'"):
        plan_for(frame, PreprocessingConfig(numeric_columns=(bad,)))


def test_unknown_override_column(frame):
    with pytest.raises(PreprocessingError, match="'nope'"):
        plan_for(frame, PreprocessingConfig(drop_columns=("nope",)))


def test_missing_target(frame):
    with pytest.raises(PreprocessingError, match="not in the train data"):
        infer_columns(frame.drop(columns="y"), "y", ("id",))


def test_unsupported_dtype_dropped(frame):
    frame["when"] = pd.date_range("2020-01-01", periods=len(frame))
    assert "when" in plan_for(frame).dropped


@pytest.mark.parametrize(
    "num,cat,enc,scaler",
    list(
        itertools.product(
            ["mean", "median", "constant"], ["most_frequent", "constant"], ["onehot", "ordinal", "target"],
            ["none", "standard", "robust"],
        )
    ),
)
def test_all_combinations(frame, num, cat, enc, scaler):
    cfg = PreprocessingConfig(numeric_imputer=num, categorical_imputer=cat, encoder=enc, scaler=scaler)
    pre = build_preprocessor(plan_for(frame, cfg), cfg, "binary")
    out = pre.fit_transform(frame, frame["y"])
    assert out.shape[0] == len(frame)
    assert np.isfinite(out).all()
    names = pre.get_feature_names_out()
    assert len(names) == out.shape[1]
    assert not any(n.split("__", 1)[-1] in ("id", "y") for n in names)


def test_regression_and_multiclass_target_encoding(frame):
    cfg = PreprocessingConfig(encoder="target")
    plan = plan_for(frame, cfg)
    reg = build_preprocessor(plan, cfg, "regression").fit_transform(frame, frame["x"].fillna(0))
    assert np.isfinite(reg).all()
    multi = frame["level"]
    assert np.isfinite(build_preprocessor(plan, cfg, "multiclass").fit_transform(frame, multi)).all()


@pytest.mark.parametrize("enc", ["onehot", "ordinal", "target"])
def test_unknown_category_at_transform(frame, enc):
    cfg = PreprocessingConfig(encoder=enc)
    pre = build_preprocessor(plan_for(frame, cfg), cfg, "binary").fit(frame, frame["y"])
    new = frame.iloc[:3].copy()
    new["cat"] = "never-seen"
    new["x"] = np.inf
    assert np.isfinite(pre.transform(new)).all()


def test_unsupported_task_and_no_columns(frame):
    plan = plan_for(frame)
    with pytest.raises(PreprocessingError, match="unsupported task"):
        build_preprocessor(plan, PreprocessingConfig(), "ranking")
    empty = ColumnPlan((), (), ("const",), ("id", "y"))
    with pytest.raises(PreprocessingError, match="no usable feature columns"):
        build_preprocessor(empty, PreprocessingConfig(), "binary")


def test_clone_and_cross_val(frame):
    pre = build_preprocessor(plan_for(frame), PreprocessingConfig(scaler="standard"), "binary")
    cloned = clone(pre)
    assert not hasattr(cloned, "transformers_")
    model = make_pipeline(pre, LogisticRegression(max_iter=200))
    scores = cross_val_score(model, frame, frame["y"], cv=3)
    assert len(scores) == 3


@pytest.mark.parametrize("num", ["mean", "median"])
@pytest.mark.parametrize("enc", ["onehot", "ordinal", "target"])
@pytest.mark.parametrize("scaler", ["none", "standard", "robust"])
def test_no_leakage_from_validation_rows(frame, num, enc, scaler):
    cfg = PreprocessingConfig(numeric_imputer=num, encoder=enc, scaler=scaler)
    pre = build_preprocessor(plan_for(frame, cfg), cfg, "binary")
    split = make_splits(frame["y"], "binary", ValidationConfig(n_folds=4)).splits[0]
    y = frame["y"]
    train_a, valid_a = fit_transform_fold(pre, frame, y, split)

    # Fitting on the train subset alone gives the same train output.
    alone = clone(pre).fit_transform(frame.iloc[split.train_idx], y.iloc[split.train_idx])
    np.testing.assert_allclose(train_a, alone)

    # Rewriting validation rows (values, missingness, new categories, labels) changes nothing in train output.
    mutated = frame.copy()
    v = split.valid_idx
    mutated.loc[v, "x"] = 1e9
    mutated.loc[v[:3], "x"] = np.nan
    mutated.loc[v, "cat"] = "novel"
    mutated.loc[v, "count"] = 10**9
    y_mut = y.copy()
    y_mut.iloc[v] = 1 - y_mut.iloc[v]
    train_b, _ = fit_transform_fold(pre, mutated, y_mut, split)
    np.testing.assert_allclose(train_a, train_b)

    fit_a = clone(pre).fit(frame.iloc[split.train_idx], y.iloc[split.train_idx])
    fit_b = clone(pre).fit(mutated.iloc[split.train_idx], y_mut.iloc[split.train_idx])
    np.testing.assert_allclose(fit_a.transform(frame.iloc[v]), fit_b.transform(frame.iloc[v]))


def test_validation_statistics_not_used(frame):
    cfg = PreprocessingConfig(numeric_imputer="mean")
    pre = build_preprocessor(plan_for(frame, cfg), cfg, "binary")
    split = make_splits(frame["y"], "binary", ValidationConfig(n_folds=4)).splits[0]
    fitted = clone(pre).fit(frame.iloc[split.train_idx], frame["y"].iloc[split.train_idx])
    stats = fitted.named_transformers_["num"].named_steps["impute"].statistics_
    expected = frame.iloc[split.train_idx][["x", "count"]].mean().to_numpy()
    np.testing.assert_allclose(stats, expected)


def test_frame_not_mutated(frame):
    before = frame.copy()
    cfg = PreprocessingConfig(encoder="target")
    plan = plan_for(frame, cfg)
    pre = build_preprocessor(plan, cfg, "binary")
    split = make_splits(frame["y"], "binary").splits[0]
    fit_transform_fold(pre, frame, frame["y"], split)
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize(
    "kwargs,fragment",
    [
        ({"numeric_imputer": "mode"}, "'numeric_imputer'"),
        ({"categorical_imputer": "mean"}, "'categorical_imputer'"),
        ({"encoder": "hash"}, "'encoder'"),
        ({"scaler": "minmax"}, "'scaler'"),
        ({"max_cardinality": 0}, "'max_cardinality'"),
        ({"numeric_fill_value": "a"}, "'numeric_fill_value'"),
        ({"categorical_fill_value": 3}, "'categorical_fill_value'"),
        ({"seed": True}, "'seed'"),
        ({"numeric_columns": "x"}, "'numeric_columns'"),
        ({"numeric_columns": ("x",), "drop_columns": ("x",)}, "column 'x' is listed in both"),
    ],
)
def test_config_errors(kwargs, fragment):
    with pytest.raises(PreprocessingError, match=fragment):
        PreprocessingConfig(**kwargs)


def test_config_collects_problems():
    with pytest.raises(PreprocessingError) as exc:
        PreprocessingConfig(encoder="x", scaler="y")
    assert len(exc.value.problems) == 2


PREP_TOML = """
[preprocessing]
encoder = "ordinal"
scaler = "robust"
max_cardinality = 5
numeric_columns = ["a"]
"""


def test_toml_parsing_and_overrides(data_dir):
    run = data_dir / "run.toml"
    base = run.read_text()
    assert load_config(run).preprocessing_config == PreprocessingConfig()
    run.write_text(base + PREP_TOML)
    cfg = load_config(run).preprocessing_config
    assert (cfg.encoder, cfg.scaler, cfg.max_cardinality, cfg.numeric_columns) == ("ordinal", "robust", 5, ("a",))
    assert load_config(run, encoder="onehot").preprocessing_config.encoder == "onehot"
    assert load_config(run, seed=9).seed == 9


def test_toml_errors(data_dir):
    run = data_dir / "run.toml"
    base = run.read_text()
    run.write_text(base + "[preprocessing]\nbogus = 1\n")
    with pytest.raises(DataContractError, match="unknown keys in \\[preprocessing\\]"):
        load_config(run)
    run.write_text(base + '[preprocessing]\nencoder = "hash"\n')
    with pytest.raises(DataContractError, match="encoder"):
        load_config(run)


def test_plan_round_trip_and_stage_boundary(frame, tmp_path):
    cfg = PreprocessingConfig(encoder="ordinal", scaler="standard", drop_columns=("flag",))
    plan = plan_for(frame, cfg)
    path = tmp_path / "plan.json"
    save_plan(plan, cfg, path)
    with pytest.raises(PreprocessingError, match="already exists"):
        save_plan(plan, cfg, path)
    plan2, cfg2 = load_plan(path)
    assert (plan2, cfg2) == (plan, cfg)

    split = make_splits(frame["y"], "binary").splits[0]
    a = fit_transform_fold(build_preprocessor(plan, cfg, "binary"), frame, frame["y"], split)
    b = fit_transform_fold(build_preprocessor(plan2, cfg2, "binary"), frame, frame["y"], split)
    np.testing.assert_allclose(a[0], b[0])
    np.testing.assert_allclose(a[1], b[1])


def test_plan_rejects_malformed(frame, tmp_path):
    cfg = PreprocessingConfig()
    path = tmp_path / "plan.json"
    save_plan(plan_for(frame), cfg, path)
    good = json.loads(path.read_text())

    def write(**changes):
        p = tmp_path / f"bad{len(list(tmp_path.iterdir()))}.json"
        p.write_text(json.dumps({**good, **changes}))
        return p

    with pytest.raises(PreprocessingError, match="not found"):
        load_plan(tmp_path / "missing.json")
    notjson = tmp_path / "nj.json"
    notjson.write_text("{")
    with pytest.raises(PreprocessingError, match="not valid JSON"):
        load_plan(notjson)
    with pytest.raises(PreprocessingError, match="schema_version"):
        load_plan(write(schema_version=99))
    with pytest.raises(PreprocessingError, match="'numeric' must be a list"):
        load_plan(write(numeric="x"))
    with pytest.raises(PreprocessingError, match="more than one group"):
        load_plan(write(dropped=good["numeric"]))
    with pytest.raises(PreprocessingError, match="known preprocessing settings"):
        load_plan(write(config={"bogus": 1}))
    with pytest.raises(PreprocessingError, match="encoder"):
        load_plan(write(config={**good["config"], "encoder": "hash"}))
    broken = {k: v for k, v in good.items() if k != "reasons"}
    p = tmp_path / "nr.json"
    p.write_text(json.dumps(broken))
    with pytest.raises(PreprocessingError, match="missing field"):
        load_plan(p)


def test_readme_example(frame):
    cfg = PreprocessingConfig(encoder="onehot", scaler="standard")
    plan = infer_columns(frame, "y", ("id",), cfg)
    pre = build_preprocessor(plan, cfg, "binary")
    split = make_splits(frame["y"], "binary").splits[0]
    train_t, valid_t = fit_transform_fold(pre, frame, frame["y"], split)
    assert train_t.shape[1] == valid_t.shape[1]
