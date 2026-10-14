"""Tune several estimator families with one study each and compare their best validated scores."""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from automl.errors import TuningError
from automl.estimators import ESTIMATOR_NAMES, estimator_available, get_estimator_spec
from automl.preprocessing import PreprocessingConfig
from automl.tuning import (
    RESULT_FILE,
    STUDY_FILE,
    NoCompletedTrials,
    SearchSpace,
    TuningConfig,
    TuningResult,
    _budget_problems,
    _require_optuna,
    _resolve_space,
    load_tuning,
    resume_tuning,
    tune,
)
from automl.validation import SplitResult

FAMILIES_FILE = "tuning_families.json"
FAMILIES_SCHEMA_VERSION = 1

SELECTION_NOTE = (
    "Family scores are mean validation scores of the best trial per family. They are optimistically "
    "biased by the search and are not leaderboard estimates."
)


@dataclass(frozen=True, eq=False)
class FamilyOutcome:
    family: str
    status: str  # "complete" or "failed"
    budget: dict[str, Any]
    result: TuningResult | None
    error: str | None = None


@dataclass(frozen=True, eq=False)
class FamiliesResult:
    task: str
    metric: str
    direction: str
    config: TuningConfig
    outcomes: tuple[FamilyOutcome, ...]
    comparison: tuple[dict[str, Any], ...]
    best_family: str | None
    notes: tuple[str, ...]
    directory: Path | None = None

    def outcome(self, family: str) -> FamilyOutcome:
        for o in self.outcomes:
            if o.family == family:
                return o
        raise KeyError(family)


def _family_config(config: TuningConfig, family: str, budget: dict[str, Any]) -> TuningConfig:
    return replace(
        config,
        estimator=family,
        n_trials=budget.get("n_trials"),
        timeout_seconds=budget.get("timeout_seconds"),
        families=None,
        budgets={},
        spaces={family: config.spaces[family]} if family in config.spaces else {},
        estimator_params=config.estimator_params if family == config.estimator else {},
    )


def _budget_plan(
    n_trials: int | None,
    timeout_seconds: float | None,
    config_budgets: dict[str, dict[str, Any]],
    families: tuple[str, ...],
    budgets: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    shared = {"n_trials": n_trials, "timeout_seconds": timeout_seconds}
    plan: dict[str, dict[str, Any]] = {}
    missing = []
    for fam in families:
        chosen = (budgets or {}).get(fam) or config_budgets.get(fam)
        if chosen is None and (n_trials is not None or timeout_seconds is not None):
            chosen = shared
        if chosen is None:
            missing.append(fam)
            continue
        entry = {"n_trials": chosen.get("n_trials"), "timeout_seconds": chosen.get("timeout_seconds")}
        bad = _budget_problems(entry["n_trials"], entry["timeout_seconds"], f"budgets.{fam}.")
        if bad:
            raise TuningError(bad)
        plan[fam] = entry
    if missing:
        raise TuningError(
            [f"family '{f}' has no budget; set 'n_trials' and/or 'timeout_seconds' or a per-family budget" for f in missing]
        )
    return plan


def _rank(outcomes: tuple[FamilyOutcome, ...], direction: str) -> tuple[tuple[dict[str, Any], ...], str | None]:
    sign = 1 if direction == "maximize" else -1
    rows: list[dict[str, Any]] = []
    for o in outcomes:
        res = o.result
        row: dict[str, Any] = {"family": o.family, "status": o.status, "rank": None, "error": o.error}
        if res is not None:
            states = Counter(t.state for t in res.trials)
            row["trial_counts"] = dict(states)
            row["failure_categories"] = dict(Counter(t.failure_category for t in res.trials if t.state == "failed"))
            row["duration_seconds"] = sum(b.get("elapsed_seconds", 0.0) for b in res.budgets)
            row["budget"] = o.budget
            if res.best is not None:
                row.update(
                    best_score=res.best.score,
                    best_fold_std=res.best.fold_std,
                    best_trial=res.best.number,
                    best_params=res.best.params,
                )
        rows.append(row)
    done = sorted((r for r in rows if "best_score" in r), key=lambda r: -sign * r["best_score"])
    for i, r in enumerate(done, 1):
        r["rank"] = i
    ordered = done + [r for r in rows if "best_score" not in r]
    return tuple(ordered), (done[0]["family"] if done else None)


def _write(directory: Path, config: TuningConfig, task: str, metric: str, direction: str, plan, entries, comparison, best, notes) -> None:
    payload = {
        "schema_version": FAMILIES_SCHEMA_VERSION,
        "task": task,
        "metric": {"name": metric, "direction": direction},
        "config": asdict(config),
        "budget_plan": plan,
        "families": entries,
        "comparison": list(comparison),
        "best_family": best,
        "notes": list(notes),
    }
    final = directory / FAMILIES_FILE
    tmp = final.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False))
    os.replace(tmp, final)


def _entries(outcomes: tuple[FamilyOutcome, ...], pending: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    return [{"family": f, "status": "pending", "directory": f, "error": None} for f in pending] + [
        {"family": o.family, "status": o.status, "directory": o.family, "error": o.error} for o in outcomes
    ]


def _finalize(config, task, metric, direction, plan, outcomes, directory) -> FamiliesResult:
    comparison, best = _rank(outcomes, direction)
    notes = (SELECTION_NOTE,)
    result = FamiliesResult(task, metric, direction, config, outcomes, comparison, best, notes, directory)
    if directory is not None:
        _write(directory, config, task, metric, direction, plan, _entries(outcomes), comparison, best, notes)
    if best is None:
        raise TuningError(
            ["no family produced a completed trial"] + [f"{o.family}: {o.error}" for o in outcomes if o.error]
        )
    return result


def tune_families(
    train: pd.DataFrame,
    target: str,
    id_columns: tuple[str, ...],
    resolved_task: Any,
    splits: SplitResult,
    config: TuningConfig,
    families: list[str] | tuple[str, ...] | None = None,
    budgets: dict[str, dict[str, Any]] | None = None,
    spaces: dict[str, SearchSpace] | None = None,
    preprocessing_config: PreprocessingConfig | None = None,
    directory: str | Path | None = None,
) -> FamiliesResult:
    """Run one seeded study per family on identical splits and rank the best trials.

    `families` defaults to `config.families`, then to every installed family. A requested family that is
    not installed raises its install hint. Per-family settings come from `budgets`, then `config.budgets`,
    then the shared `n_trials`/`timeout_seconds`. `config.estimator_params` apply to `config.estimator` only.
    """
    _require_optuna()
    requested = tuple(families) if families is not None else config.families
    if requested is None:
        chosen = tuple(f for f in ESTIMATOR_NAMES if estimator_available(f))
    else:
        unknown = [f for f in requested if f not in ESTIMATOR_NAMES]
        if unknown or len(set(requested)) != len(requested) or not requested:
            raise TuningError(f"'families' must be unique names from {list(ESTIMATOR_NAMES)}, got {list(requested)}")
        chosen = requested
        for f in chosen:
            get_estimator_spec(f).require()
    plan = _budget_plan(config.n_trials, config.timeout_seconds, config.budgets, chosen, budgets)
    configs = {f: _family_config(config, f, plan[f]) for f in chosen}
    for f in chosen:  # fail on a bad space override before any family runs
        _resolve_space((spaces or {}).get(f), configs[f], resolved_task.task)
    parent = None
    if directory is not None:
        parent = Path(directory)
        try:
            parent.mkdir(parents=True)
        except FileExistsError:
            raise TuningError(f"tuning directory {parent} already exists; choose a new path") from None
        _write(
            parent, config, resolved_task.task.value, resolved_task.metric.name, resolved_task.metric.direction,
            plan, _entries((), chosen), (), None, (SELECTION_NOTE,),
        )
    outcomes: list[FamilyOutcome] = []
    for f in chosen:
        try:
            res = tune(
                train, target, id_columns, resolved_task, splits, configs[f],
                space=(spaces or {}).get(f), preprocessing_config=preprocessing_config,
                directory=parent / f if parent else None,
            )
            outcomes.append(FamilyOutcome(f, "complete", plan[f], res))
        except NoCompletedTrials as exc:
            res = load_tuning(parent / f) if parent and (parent / f / RESULT_FILE).is_file() else None
            outcomes.append(FamilyOutcome(f, "failed", plan[f], res, "; ".join(exc.problems)))
    return _finalize(
        config, resolved_task.task.value, resolved_task.metric.name, resolved_task.metric.direction,
        plan, tuple(outcomes), parent,
    )


def load_tuning_families(directory: str | Path) -> FamiliesResult:
    """Read `tuning_families.json` and each family's `tuning.json`; does not need Optuna."""
    directory = Path(directory)
    label = f"tuning families result {directory}"
    try:
        raw = json.loads((directory / FAMILIES_FILE).read_text())
    except FileNotFoundError:
        raise TuningError(f"{label}: {FAMILIES_FILE} not found") from None
    except json.JSONDecodeError as exc:
        raise TuningError(f"{label}: {FAMILIES_FILE} is not valid JSON: {exc}") from None
    if not isinstance(raw, dict) or raw.get("schema_version") != FAMILIES_SCHEMA_VERSION:
        raise TuningError(f"{label}: unsupported or missing 'schema_version'; expected {FAMILIES_SCHEMA_VERSION}")
    try:
        outcomes = []
        for e in raw["families"]:
            if e["status"] == "pending":
                continue
            fam_dir = directory / e["directory"]
            res = load_tuning(fam_dir) if (fam_dir / RESULT_FILE).is_file() else None
            outcomes.append(FamilyOutcome(e["family"], e["status"], raw["budget_plan"][e["family"]], res, e["error"]))
        return FamiliesResult(
            task=raw["task"],
            metric=raw["metric"]["name"],
            direction=raw["metric"]["direction"],
            config=TuningConfig(**raw["config"]),
            outcomes=tuple(outcomes),
            comparison=tuple(raw["comparison"]),
            best_family=raw["best_family"],
            notes=tuple(raw["notes"]),
            directory=directory,
        )
    except (KeyError, TypeError) as exc:
        raise TuningError(f"{label}: missing or invalid field ({exc})") from None


def resume_tuning_families(
    directory: str | Path,
    train: pd.DataFrame,
    target: str,
    id_columns: tuple[str, ...],
    resolved_task: Any,
    splits: SplitResult,
    n_trials: int | None = None,
    timeout_seconds: float | None = None,
    budgets: dict[str, dict[str, Any]] | None = None,
    families: list[str] | tuple[str, ...] | None = None,
    spaces: dict[str, SearchSpace] | None = None,
    preprocessing_config: PreprocessingConfig | None = None,
) -> FamiliesResult:
    """Add trials to the saved per-family studies; budgets apply to the new trials only."""
    _require_optuna()
    directory = Path(directory)
    saved = load_tuning_families(directory)
    available = [o.family for o in saved.outcomes if (directory / o.family / STUDY_FILE).is_file()]
    chosen = tuple(families) if families is not None else tuple(available)
    missing = [f for f in chosen if f not in available]
    if missing or not chosen:
        raise TuningError(f"families {missing or list(chosen)} have no saved study in {directory}; saved: {available}")
    plan = _budget_plan(n_trials, timeout_seconds, {}, chosen, budgets)
    done: dict[str, FamilyOutcome] = {o.family: o for o in saved.outcomes}
    for f in chosen:
        try:
            res = resume_tuning(
                directory / f, train, target, id_columns, resolved_task, splits,
                n_trials=plan[f]["n_trials"], timeout_seconds=plan[f]["timeout_seconds"],
                space=(spaces or {}).get(f), preprocessing_config=preprocessing_config,
            )
            done[f] = FamilyOutcome(f, "complete", plan[f], res)
        except NoCompletedTrials as exc:
            done[f] = FamilyOutcome(f, "failed", plan[f], load_tuning(directory / f), "; ".join(exc.problems))
    ordered = tuple(done[o.family] for o in saved.outcomes)
    stored_plan = {o.family: o.budget for o in ordered}
    return _finalize(saved.config, saved.task, saved.metric, saved.direction, stored_plan, ordered, directory)
