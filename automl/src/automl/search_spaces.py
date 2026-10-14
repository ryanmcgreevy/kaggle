"""Per-family Optuna search spaces, task-aware, with TOML-style range overrides."""

from __future__ import annotations

from typing import Any, Callable

from automl.errors import TuningError
from automl.metrics import Task

Space = dict[str, dict[str, Any]]
Overrides = dict[str, dict[str, dict[str, Any]]]

OVERRIDE_KEYS = {"low", "high", "log", "step", "choices", "enabled", "fixed"}


def _f(low: float, high: float, log: bool = False) -> dict[str, Any]:
    return {"type": "float", "low": low, "high": high, **({"log": True} if log else {})}


def _i(low: int, high: int) -> dict[str, Any]:
    return {"type": "int", "low": low, "high": high}


def _rounds(task: Task) -> tuple[int, int]:
    # Multiclass fits one tree per class per round, so its ceiling is lower to keep trials comparable in cost.
    return (20, 200) if task is Task.MULTICLASS else (20, 300)


def _hist_gradient_boosting(task: Task) -> Space:
    lo, hi = _rounds(task)
    return {
        "learning_rate": _f(0.01, 0.3, log=True),
        "max_iter": _i(lo, hi),
        "max_leaf_nodes": _i(8, 63),
        "max_depth": _i(3, 12),
        "min_samples_leaf": _i(5, 40),
        "l2_regularization": _f(1e-8, 10.0, log=True),
    }


def _lightgbm(task: Task) -> Space:
    lo, hi = _rounds(task)
    return {
        "learning_rate": _f(0.01, 0.3, log=True),
        "n_estimators": _i(lo, hi),
        "num_leaves": _i(8, 128),
        "max_depth": _i(3, 12),
        "min_child_samples": _i(5, 50),
        "subsample": _f(0.5, 1.0),
        # LightGBM ignores 'subsample' unless bagging frequency is positive.
        "subsample_freq": {"type": "categorical", "choices": [1]},
        "colsample_bytree": _f(0.5, 1.0),
        "reg_alpha": _f(1e-8, 10.0, log=True),
        "reg_lambda": _f(1e-8, 10.0, log=True),
    }


def _xgboost(task: Task) -> Space:
    lo, hi = _rounds(task)
    return {
        "learning_rate": _f(0.01, 0.3, log=True),
        "n_estimators": _i(lo, hi),
        "max_depth": _i(2, 10),
        "min_child_weight": _f(1.0, 10.0),
        "subsample": _f(0.5, 1.0),
        "colsample_bytree": _f(0.5, 1.0),
        "gamma": _f(1e-8, 5.0, log=True),
        "reg_alpha": _f(1e-8, 10.0, log=True),
        "reg_lambda": _f(1e-8, 10.0, log=True),
    }


def _catboost(task: Task) -> Space:
    lo, hi = _rounds(task)
    return {
        "learning_rate": _f(0.01, 0.3, log=True),
        "iterations": _i(lo, hi),
        "depth": _i(3, 8),
        "l2_leaf_reg": _f(1.0, 10.0, log=True),
        "random_strength": _f(1e-3, 10.0, log=True),
    }


_SPACES: dict[str, Callable[[Task], Space]] = {
    "hist_gradient_boosting": _hist_gradient_boosting,
    "lightgbm": _lightgbm,
    "xgboost": _xgboost,
    "catboost": _catboost,
}
SPACE_FAMILIES = tuple(_SPACES)


def _builder(family: str) -> Callable[[Task], Space]:
    try:
        return _SPACES[family]
    except KeyError:
        raise TuningError(f"no search space for estimator family {family!r}; valid families: {list(SPACE_FAMILIES)}") from None


def apply_overrides(family: str, base: Space, overrides: dict[str, dict[str, Any]] | None) -> Space:
    """Return `base` with range overrides applied; bounds are validated later with the whole space."""
    out = {k: dict(v) for k, v in base.items()}
    for param, ov in (overrides or {}).items():
        where = f"search space override '{family}.{param}'"
        if param not in out:
            raise TuningError(f"{where}: unknown parameter; the {family} space has {sorted(base)}")
        if not isinstance(ov, dict):
            raise TuningError(f"{where} must be a table of keys {sorted(OVERRIDE_KEYS)}")
        unknown = set(ov) - OVERRIDE_KEYS
        if unknown:
            raise TuningError(f"{where} has unknown keys {sorted(unknown)}; valid keys: {sorted(OVERRIDE_KEYS)}")
        ov = dict(ov)
        enabled = ov.pop("enabled", True)
        if not isinstance(enabled, bool):
            raise TuningError(f"{where}: 'enabled' must be true or false")
        if not enabled:
            del out[param]
        elif "fixed" in ov:
            if len(ov) > 1:
                raise TuningError(f"{where}: 'fixed' cannot be combined with {sorted(set(ov) - {'fixed'})}")
            out[param] = {"type": "categorical", "choices": [ov["fixed"]]}
        else:
            out[param] = {**out[param], **ov}
    return out


def get_search_space(family: str, task: Task, overrides: dict[str, dict[str, Any]] | None = None) -> Space:
    """The documented search space for `family` and `task`, with optional per-parameter overrides."""
    return apply_overrides(family, _builder(family)(Task(task)), overrides)


def describe_search_space(family: str, task: Task) -> list[dict[str, Any]]:
    """Parameter names with type, bounds, log flag, step, or choices; needs no optional library."""
    return [{"name": name, **spec} for name, spec in get_search_space(family, task).items()]
