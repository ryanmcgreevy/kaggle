import json

import numpy as np
import pandas as pd
import pytest

from automl import (
    BaselineConfig,
    EnsembleConfig,
    EnsembleError,
    RefitError,
    TuningConfig,
    RefitConfig,
    ValidationConfig,
    load_ensemble,
    make_splits,
    refit_and_predict,
    run_baseline,
    run_ensemble,
    save_baseline,
    tune,
    tune_families,
)
from automl.metrics import Task, get_metric
from automl.task import ResolvedTask


def frames(task: Task, n: int = 90, metric_override: str | None = None):
    rng = np.random.default_rng(19)
    x = rng.normal(size=n)
    category = rng.choice(["a", "b", "c"], size=n)
    if task is Task.BINARY:
        y = ((x + (category == "a") + rng.normal(scale=0.4, size=n)) > 0.3).astype(int)
        metric = "roc_auc"
    elif task is Task.MULTICLASS:
        y = np.asarray(["low", "mid", "high"])[np.digitize(x, [-0.5, 0.5])]
        metric = "accuracy"
    else:
        y = x + (category == "a") + rng.normal(scale=0.2, size=n)
        metric = "rmse"
    train = pd.DataFrame({"id": np.arange(n), "x": x, "category": category, "target": y})
    test = pd.DataFrame({"id": np.arange(1000, 1010), "x": rng.normal(size=10), "category": rng.choice(["a", "b"], 10)})
    return train, test, ResolvedTask(task, get_metric(metric_override or metric))


def saved_candidates(tmp_path, task=Task.BINARY, metric_override=None):
    train, test, resolved = frames(task, metric_override=metric_override)
    splits = make_splits(train["target"], resolved, ValidationConfig(n_folds=3, seed=11))
    results = []
    for name, iterations in (("one", 6), ("two", 11)):
        result = run_baseline(
            train,
            "target",
            ("id",),
            resolved,
            splits,
            BaselineConfig(seed=11, estimator_params={"max_iter": iterations}),
        )
        directory = tmp_path / f"{task.value}-{name}"
        save_baseline(result, directory)
        results.append(directory)
    return train, test, resolved, splits, results


@pytest.mark.parametrize("method", ["voting", "stacking"])
def test_classification_ensembles_evaluate_and_persist(tmp_path, method):
    train, test, resolved, splits, sources = saved_candidates(tmp_path)
    output = tmp_path / "ensemble"
    result = run_ensemble(
        train,
        "target",
        ("id",),
        resolved,
        splits,
        sources,
        EnsembleConfig(method=method, seed=11, stack_cv=3),
        directory=output,
    )

    assert len(result.folds) == 3
    assert len(result.candidates) == 2
    assert all(label.startswith("baseline:hist_gradient_boosting#") for label in result.candidates)
    assert np.isfinite(result.mean)
    assert result.directory == output
    loaded = load_ensemble(output, load_model=True)
    assert loaded.metadata["method"] == method
    assert loaded.model is not None
    assert loaded.metadata["splits"]["sha256"]
    assert set(loaded.metadata["candidate_details"]) == set(result.candidates)
    assert json.loads((output / "ensemble.json").read_text())["summary"]["mean"] == result.mean

    refitted = refit_and_predict(
        output,
        train,
        test,
        target="target",
        id_columns=("id",),
        resolved_task=resolved,
        config=RefitConfig(),
    )
    assert refitted.predictions.n_rows == len(test)
    assert list(refitted.predictions.frame["id"]) == list(test["id"])
    assert refitted.refit.selection.source_kind == "ensemble"
    different_train = train.copy()
    different_train.loc[0, "x"] += 1
    with pytest.raises(RefitError, match="different training data"):
        refit_and_predict(
            output,
            different_train,
            test,
            target="target",
            id_columns=("id",),
            resolved_task=resolved,
        )


def test_regression_stacking_and_configured_meta_estimator(tmp_path):
    train, _, resolved, splits, sources = saved_candidates(tmp_path, Task.REGRESSION)
    result = run_ensemble(
        train,
        "target",
        ("id",),
        resolved,
        splits,
        sources,
        EnsembleConfig(
            method="stacking",
            seed=11,
            stack_cv=3,
            meta_estimator="decision_tree_regressor",
            meta_estimator_params={"max_depth": 3},
        ),
    )
    assert result.meta_estimator == "DecisionTreeRegressor"
    assert np.isfinite(result.mean)


@pytest.mark.parametrize(
    ("task", "method"),
    [
        (Task.BINARY, "voting"),
        (Task.BINARY, "stacking"),
        (Task.REGRESSION, "stacking"),
    ],
)
def test_fit_mode_skips_outer_validation_and_persists_prediction_model(tmp_path, monkeypatch, task, method):
    import automl.ensemble as ensemble_module

    train, test, resolved, _, sources = saved_candidates(tmp_path, task)

    def unexpected_score(*args, **kwargs):
        raise AssertionError("fit mode must not compute outer validation scores")

    monkeypatch.setattr(ensemble_module, "_score", unexpected_score)
    output = tmp_path / f"{task.value}-{method}-fit"
    result = run_ensemble(
        train,
        "target",
        ("id",),
        resolved,
        sources=sources,
        config=EnsembleConfig(method=method, seed=11, stack_cv=3, mode="fit"),
        directory=output,
    )

    assert result.mean is None and result.std is None
    assert result.folds == ()
    assert result.candidate_fold_scores == {}
    assert result.validation_status == "not_evaluated"
    assert result.validation_limitation
    assert result.to_dict()["score_statement"] == "No validation was run in fit mode."
    assert result.splits is None
    assert all(
        "validation_mean" not in candidate and "source_validation_mean" in candidate
        for candidate in result.candidate_details.values()
    )
    loaded = load_ensemble(output, load_model=True)
    assert loaded.metadata["config"]["mode"] == "fit"
    assert loaded.metadata["validation_status"] == "not_evaluated"
    assert loaded.model is not None

    refitted = refit_and_predict(
        output,
        train,
        test,
        target="target",
        id_columns=("id",),
        resolved_task=resolved,
    )
    assert refitted.predictions.n_rows == len(test)
    assert list(refitted.predictions.frame["id"]) == list(test["id"])


def test_evaluate_mode_requires_splits_and_fit_mode_rejects_them(tmp_path):
    train, _, resolved, splits, sources = saved_candidates(tmp_path)
    with pytest.raises(EnsembleError, match="requires validation splits"):
        run_ensemble(train, "target", ("id",), resolved, sources=sources)
    with pytest.raises(EnsembleError, match="does not accept outer validation splits"):
        run_ensemble(
            train,
            "target",
            ("id",),
            resolved,
            splits,
            sources,
            EnsembleConfig(mode="fit"),
        )


@pytest.mark.parametrize("source_kind", ["tuning", "families"])
def test_ensembles_consume_saved_tuning_candidates(tmp_path, source_kind):
    pytest.importorskip("optuna")
    train, _, resolved, splits, sources = saved_candidates(tmp_path)
    if source_kind == "tuning":
        tuned = tune(
            train,
            "target",
            ("id",),
            resolved,
            splits,
            TuningConfig(n_trials=1, estimator="hist_gradient_boosting", seed=11),
            directory=tmp_path / "tuning",
        )
        candidate_source = tuned.directory
    else:
        families = tune_families(
            train,
            "target",
            ("id",),
            resolved,
            splits,
            TuningConfig(n_trials=1, families=("hist_gradient_boosting",), seed=11),
            directory=tmp_path / "families",
        )
        candidate_source = families.directory

    result = run_ensemble(
        train,
        "target",
        ("id",),
        resolved,
        splits,
        [sources[0], candidate_source],
        EnsembleConfig(method="stacking", seed=11, stack_cv=3),
    )
    assert np.isfinite(result.mean)
    assert len(result.candidates) == 2


def test_stacking_fits_native_lightgbm_candidate(tmp_path):
    pytest.importorskip("lightgbm")
    train, _, resolved, splits, sources = saved_candidates(tmp_path)
    tuned = tune(
        train,
        "target",
        ("id",),
        resolved,
        splits,
        TuningConfig(n_trials=1, estimator="lightgbm", seed=11),
        directory=tmp_path / "lightgbm",
    )

    result = run_ensemble(
        train,
        "target",
        ("id",),
        resolved,
        splits,
        [sources[0], tuned.directory],
        EnsembleConfig(method="stacking", seed=11, stack_cv=3),
    )

    assert np.isfinite(result.mean)
    assert any(label.startswith("tuned:lightgbm") for label in result.candidates)


def test_voting_rejects_regression_and_probability_only_metric(tmp_path):
    train, _, resolved, splits, sources = saved_candidates(tmp_path, Task.REGRESSION)
    with pytest.raises(EnsembleError, match="classification-only"):
        run_ensemble(
            train, "target", ("id",), resolved, splits, sources, EnsembleConfig(method="voting")
        )

    train, _, resolved, splits, sources = saved_candidates(
        tmp_path, Task.MULTICLASS, metric_override="log_loss"
    )
    with pytest.raises(EnsembleError, match="cannot be scored"):
        run_ensemble(
            train, "target", ("id",), resolved, splits, sources, EnsembleConfig(method="voting")
        )


def test_invalid_candidate_selection_and_overwrite(tmp_path):
    train, _, resolved, splits, sources = saved_candidates(tmp_path)
    with pytest.raises(EnsembleError, match="duplicates"):
        run_ensemble(
            train, "target", ("id",), resolved, splits, sources, labels=["baseline:hist_gradient_boosting"] * 2
        )
    with pytest.raises(EnsembleError, match="duplicate model results"):
        run_ensemble(train, "target", ("id",), resolved, splits, [sources[0], sources[0]])
    output = tmp_path / "ensemble"
    run_ensemble(train, "target", ("id",), resolved, splits, sources, directory=output)
    with pytest.raises(EnsembleError, match="already exists"):
        run_ensemble(train, "target", ("id",), resolved, splits, sources, directory=output)


def test_validation_rows_do_not_affect_fold_preprocessing(tmp_path, monkeypatch):
    import automl.ensemble as ensemble_module

    train, _, resolved, splits, sources = saved_candidates(tmp_path)
    captured = []
    make_model = ensemble_module._make_model

    def record_model(*args):
        model = make_model(*args)
        captured.append(model)
        return model

    monkeypatch.setattr(ensemble_module, "_make_model", record_model)
    run_ensemble(
        train, "target", ("id",), resolved, splits, sources,
        EnsembleConfig(method="stacking", seed=11, stack_cv=3),
    )
    first_fold_stats = captured[0].estimators_[0].estimator_.named_steps["pre"].named_transformers_["num"]
    observed = first_fold_stats.named_steps["impute"].statistics_[0]
    expected = train.iloc[splits.splits[0].train_idx]["x"].median()
    assert observed == pytest.approx(expected)

    changed = train.copy()
    changed.iloc[splits.splits[0].valid_idx, changed.columns.get_loc("x")] = 1_000_000
    captured.clear()
    run_ensemble(
        changed, "target", ("id",), resolved, splits, sources,
        EnsembleConfig(method="stacking", seed=11, stack_cv=3),
    )
    changed_stats = captured[0].estimators_[0].estimator_.named_steps["pre"].named_transformers_["num"]
    changed_observed = changed_stats.named_steps["impute"].statistics_[0]
    assert changed_observed == pytest.approx(observed)
