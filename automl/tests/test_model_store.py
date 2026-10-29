import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from automl import (
    BaselineConfig,
    EnsembleConfig,
    EnsembleError,
    ModelStoreError,
    RefitConfig,
    TrackingConfig,
    ValidationConfig,
    build_submission,
    load_ensemble,
    load_fitted_model,
    make_splits,
    refit_and_predict,
    run_baseline,
    run_saved_ensemble,
    save_baseline,
    track_fitted_model,
)
from automl.metrics import Task, get_metric
from automl.task import ResolvedTask


def _data(n=72):
    rng = np.random.default_rng(907)
    x = rng.normal(size=n)
    cat = rng.choice(["a", "b", "c"], size=n)
    target = ((x + (cat == "a") + rng.normal(scale=0.4, size=n)) > 0.3).astype(int)
    train = pd.DataFrame({"id": np.arange(n), "x": x, "cat": cat, "target": target})
    test = pd.DataFrame(
        {"id": np.arange(1000, 1008), "x": rng.normal(size=8), "cat": rng.choice(["a", "b"], size=8)}
    )
    return train, test, ResolvedTask(Task.BINARY, get_metric("accuracy"))


def _saved_members(tmp_path, *, method="voting"):
    train, test, resolved = _data()
    splits = make_splits(train["target"], resolved, ValidationConfig(n_folds=3, seed=19))
    refs = []
    for index, max_iter in enumerate((5, 9)):
        result = run_baseline(
            train,
            "target",
            ("id",),
            resolved,
            splits,
            BaselineConfig(seed=19, estimator_params={"max_iter": max_iter}),
        )
        output = tmp_path / f"member-{index}"
        save_baseline(result, output)
        refs.append(output / "fitted-model")
    return train, test, resolved, refs


def test_baseline_model_round_trip_and_restart(tmp_path):
    train, _, resolved = _data()
    splits = make_splits(train["target"], resolved, ValidationConfig(n_folds=3, seed=13))
    result = run_baseline(
        train,
        "target",
        ("id",),
        resolved,
        splits,
        BaselineConfig(seed=13, estimator_params={"max_iter": 5}),
    )
    output = tmp_path / "baseline"
    save_baseline(result, output)

    loaded = load_fitted_model(output / "fitted-model")
    metadata = loaded.metadata
    assert metadata["fit_scope"] == "full_data"
    assert metadata["training_data_sha256"]
    assert metadata["feature_columns"] == ["x", "cat"]
    assert metadata["validation_splits"]["sha256"]
    assert metadata["oof"] == {"available": True, "file": "oof_predictions.csv", "coverage_rows": len(train), "complete": True}
    pd.testing.assert_frame_equal(
        loaded.oof_predictions[["row", "fold"]],
        result.model_oof_predictions[["row", "fold"]],
    )
    np.testing.assert_array_equal(
        loaded.pipeline.predict(train[metadata["feature_columns"]]),
        result.full_model.predict(train[metadata["feature_columns"]]),
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from automl import load_fitted_model; import sys; "
            "print(load_fitted_model(sys.argv[1]).metadata['model_id'])",
            str(output / "fitted-model"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == metadata["model_id"]


def test_model_loader_rejects_bad_schema_and_credentials(tmp_path):
    train, _, resolved = _data()
    splits = make_splits(train["target"], resolved, ValidationConfig(n_folds=3))
    result = run_baseline(train, "target", ("id",), resolved, splits, BaselineConfig(estimator_params={"max_iter": 4}))
    output = tmp_path / "baseline"
    save_baseline(result, output)
    metadata_path = output / "fitted-model" / "model.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["schema_version"] = 99
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ModelStoreError, match="schema_version"):
        load_fitted_model(output / "fitted-model")


@pytest.mark.parametrize(
    ("method", "expected_meta_fits", "expected_mean"),
    [("voting", 0, True), ("stacking", 1, False)],
)
def test_saved_ensemble_does_not_refit_members(tmp_path, monkeypatch, method, expected_meta_fits, expected_mean):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    train, test, resolved, refs = _saved_members(tmp_path)
    member_fits = []
    meta_fits = []
    original_member_fit = HistGradientBoostingClassifier.fit
    original_meta_fit = LogisticRegression.fit

    def count_member_fit(estimator, X, y, **kwargs):
        member_fits.append(True)
        return original_member_fit(estimator, X, y, **kwargs)

    def count_meta_fit(estimator, X, y, **kwargs):
        meta_fits.append(True)
        return original_meta_fit(estimator, X, y, **kwargs)

    monkeypatch.setattr(HistGradientBoostingClassifier, "fit", count_member_fit)
    monkeypatch.setattr(LogisticRegression, "fit", count_meta_fit)
    output = tmp_path / f"saved-{method}"
    result = run_saved_ensemble(
        train,
        "target",
        refs,
        resolved,
        EnsembleConfig(method=method, seed=19, stack_cv=3),
        directory=output,
    )

    assert member_fits == []
    assert len(meta_fits) == expected_meta_fits
    assert (result.mean is not None) is expected_mean
    loaded = load_ensemble(output, load_model=True)
    assert loaded.model is not None
    if method == "stacking":
        assert result.validation_status == "unavailable"
        assert result.validation_limitation
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
    submission = build_submission(refitted, test, target="target")
    assert len(submission.frame) == len(test)
    assert member_fits == []


def test_saved_models_reject_different_training_data(tmp_path):
    train, _, resolved, refs = _saved_members(tmp_path)
    train.loc[0, "x"] += 1
    with pytest.raises(EnsembleError, match="different training data"):
        run_saved_ensemble(
            train,
            "target",
            refs,
            resolved,
            EnsembleConfig(method="voting"),
        )


def test_local_mlflow_reference_round_trip(tmp_path):
    pytest.importorskip("mlflow")
    train, _, resolved, refs = _saved_members(tmp_path)
    local = load_fitted_model(refs[0])
    reference = track_fitted_model(
        local.pipeline,
        local.metadata,
        TrackingConfig(directory=str(tmp_path / "mlflow")),
        oof_predictions=local.oof_predictions,
    )
    restored = load_fitted_model(
        reference,
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow' / 'mlflow.db'}",
    )

    assert reference.uri.startswith("runs:/")
    assert restored.metadata["model_id"] == local.metadata["model_id"]
    pd.testing.assert_frame_equal(restored.oof_predictions, local.oof_predictions)
    np.testing.assert_array_equal(
        restored.pipeline.predict(train[local.metadata["feature_columns"]]),
        local.pipeline.predict(train[local.metadata["feature_columns"]]),
    )
