"""Comparison PDF for the best validated models, built only from saved baseline and tuning results."""

from __future__ import annotations

import json
import math
import os
import tempfile
import textwrap
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from automl.baseline import RESULT_FILE as BASELINE_FILE
from automl.baseline import BaselineResult, load_baseline
from automl.errors import DataContractError, ReportError
from automl.families import FAMILIES_FILE, FamiliesResult, load_tuning_families
from automl.report import _Skip, _require_matplotlib, _text_pages
from automl.tuning import RESULT_FILE as TUNING_FILE
from automl.tuning import TrialRecord, TuningResult, load_tuning

MODEL_REPORT_SCHEMA_VERSION = 1
IN_MEMORY = "<in-memory>"
SCORE_STATEMENT = (
    "Scores are validation scores (cross-validation or holdout on the training data); they are optimistic "
    "after selection and are not leaderboard predictions."
)


@dataclass(frozen=True)
class ModelReportConfig:
    top_n: int = 3
    max_trials_plotted: int = 200
    min_trials_for_importance: int = 10
    max_params_importance: int = 10
    title: str = "Model Comparison Report"

    def __post_init__(self) -> None:
        problems: list[str] = []
        for name in ("top_n", "max_trials_plotted", "min_trials_for_importance", "max_params_importance"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int):
                problems.append(f"setting '{name}' must be an integer")
            elif v < 1:
                problems.append(f"setting '{name}' must be at least 1, got {v}")
        if not isinstance(self.title, str) or not self.title.strip():
            problems.append("setting 'title' must be a non-empty string")
        if problems:
            raise ReportError(problems)


MODEL_REPORT_KEYS = {f.name for f in fields(ModelReportConfig)}


@dataclass(frozen=True, eq=False)
class Candidate:
    label: str
    source_kind: str  # "baseline", "tuning", or "family"
    source_path: str
    estimator_key: str
    estimator: str
    params: dict[str, Any]
    mean: float
    std: float | None
    fold_scores: tuple[float, ...]
    n_folds: int
    n_rows: int
    splits_sha256: str | None
    trial_counts: dict[str, int]
    best_trial: int | None
    details: dict[str, Any]
    versions: dict[str, str]
    rank: int | None = None
    has_oof: bool = False
    prediction_kind: str | None = None
    trials: tuple[TrialRecord, ...] = field(default=(), repr=False)
    oof: Any = field(default=None, repr=False)
    build: dict[str, Any] = field(default_factory=dict, repr=False)  # what refit needs to rebuild the pipeline

    def to_dict(self) -> dict[str, Any]:
        scores = self.fold_scores
        return {
            "rank": self.rank,
            "label": self.label,
            "source_kind": self.source_kind,
            "source_path": self.source_path,
            "estimator_key": self.estimator_key,
            "estimator": self.estimator,
            "params": self.params,
            "mean": self.mean,
            "std": self.std,
            "min": min(scores) if scores else None,
            "max": max(scores) if scores else None,
            "fold_scores": list(scores),
            "n_folds": self.n_folds,
            "n_rows": self.n_rows,
            "splits_sha256": self.splits_sha256,
            "has_oof_predictions": self.has_oof,
            "trial_counts": self.trial_counts,
            "best_trial": self.best_trial,
            "details": self.details,
            "versions": self.versions,
        }


@dataclass(frozen=True)
class ExcludedEntry:
    label: str
    source_path: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, eq=False)
class CandidateSet:
    task: str
    metric: str
    direction: str
    candidates: tuple[Candidate, ...]
    excluded: tuple[ExcludedEntry, ...]
    warnings: tuple[str, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class ModelReportResult:
    path: Path
    summary_path: Path
    pages: int
    task: str
    metric: str
    direction: str
    candidates: tuple[Candidate, ...]
    excluded: tuple[ExcludedEntry, ...]
    warnings: tuple[str, ...]
    charts: tuple[str, ...]
    charts_skipped: dict[str, str]
    schema_version: int = MODEL_REPORT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "path": str(self.path),
            "summary_path": str(self.summary_path),
            "pages": self.pages,
            "task": self.task,
            "metric": {"name": self.metric, "direction": self.direction},
            "candidates": [c.to_dict() for c in self.candidates],
            "excluded": [e.to_dict() for e in self.excluded],
            "warnings": list(self.warnings),
            "charts": list(self.charts),
            "charts_skipped": dict(self.charts_skipped),
            "score_statement": SCORE_STATEMENT,
        }


# ---- candidate collection ----


def _plain(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str, allow_nan=False))


def _kind_of(path: Path) -> str:
    if not path.is_dir():
        raise ReportError(f"source {path} is not a directory")
    found = [k for k, f in (("baseline", BASELINE_FILE), ("tuning", TUNING_FILE), ("families", FAMILIES_FILE)) if (path / f).is_file()]
    if not found:
        raise ReportError(
            f"source {path} holds none of {BASELINE_FILE}, {TUNING_FILE}, or {FAMILIES_FILE}; "
            "pass a baseline, tuning, or tuning-families result directory"
        )
    if len(found) > 1:
        raise ReportError(f"source {path} is ambiguous: it holds result files for {found}; pass one result per directory")
    return found[0]


def _load(path: Path, kind: str):
    loader = {"baseline": load_baseline, "tuning": load_tuning, "families": load_tuning_families}[kind]
    try:
        return loader(path)
    except ReportError:
        raise
    except DataContractError as exc:
        raise ReportError(exc.problems) from None


def _from_baseline(res: BaselineResult, path: str) -> Candidate:
    scores = tuple(f.score for f in res.folds)
    return Candidate(
        label=f"baseline:{res.estimator_key}",
        source_kind="baseline",
        source_path=path,
        estimator_key=res.estimator_key,
        estimator=res.estimator,
        params=_plain(res.config.estimator_params),
        mean=res.mean,
        std=res.std,
        fold_scores=scores,
        n_folds=len(res.folds),
        n_rows=res.n_rows,
        splits_sha256=None,
        trial_counts={},
        best_trial=None,
        details={
            "seed": res.config.seed,
            "split": _plain(res.split),
            "preprocessing": _plain(asdict(res.preprocessing_config)),
            "categorical_handling": res.categorical_handling,
            "device": _plain(res.device),
            "n_features": len(res.plan.numeric) + len(res.plan.categorical),
        },
        versions=dict(res.versions),
        has_oof=res.oof is not None and len(res.oof) > 0,
        prediction_kind=res.prediction_kind,
        oof=res.oof,
        build={
            "estimator_params": _plain(res.config.estimator_params),
            "seed": res.config.seed,
            "n_jobs": res.config.n_jobs,
            "device": res.config.device,
            "native_categoricals": res.categorical_handling == "native",
            "preprocessing": _plain(asdict(res.preprocessing_config)),
            "features": [*res.plan.numeric, *res.plan.categorical],
            "task": res.task.value,
            "metric": res.metric,
            "direction": res.direction,
        },
    )


def _from_tuning(res: TuningResult, path: str, label: str, kind: str) -> Candidate | None:
    best = res.best
    if best is None:
        return None
    fp = res.fingerprint
    scores = tuple(best.fold_scores)
    return Candidate(
        label=label,
        source_kind=kind,
        source_path=path,
        estimator_key=res.config.estimator,
        estimator=res.config.estimator,
        params=_plain(best.params),
        mean=float(best.score),
        std=best.fold_std,
        fold_scores=scores,
        n_folds=int(fp.get("n_folds", len(scores))),
        n_rows=int(fp.get("n_rows", 0)),
        splits_sha256=fp.get("splits_sha256"),
        trial_counts=dict(Counter(t.state for t in res.trials)),
        best_trial=best.number,
        details={
            "seed": fp.get("seed"),
            "fixed_estimator_params": _plain(fp.get("estimator_params", {})),
            "preprocessing": _plain(fp.get("preprocessing", {})),
            "categorical_handling": "native" if fp.get("native_categoricals") else "encoded",
            "device": fp.get("device"),
            "n_features": len(fp.get("features", [])),
            "budgets": _plain(list(res.budgets)),
        },
        versions=dict(res.versions),
        trials=tuple(res.trials),
        build={
            "estimator_params": _plain({**fp.get("estimator_params", {}), **best.params}),
            "seed": res.config.seed,
            "n_jobs": res.config.n_jobs,
            "device": res.config.device,
            "native_categoricals": bool(fp.get("native_categoricals")),
            "preprocessing": _plain(fp.get("preprocessing", {})),
            "features": list(fp.get("features", [])),
            "task": res.task,
            "metric": res.metric,
            "direction": res.direction,
        },
    )


def _is_sequence(sources: Any) -> bool:
    return isinstance(sources, Sequence) and not isinstance(sources, (str, bytes, Path))


def collect_candidates(sources: Any) -> CandidateSet:
    """Load saved results and rank their best models; needs no Matplotlib, Optuna, or fitted models.

    Each source is a baseline, tuning, or tuning-families result directory (or the loaded result).
    """
    items = list(sources) if _is_sequence(sources) else [sources]
    if not items:
        raise ReportError("sources must contain at least one result directory or loaded result")
    raw: list[Candidate] = []
    metas: list[tuple[str, str, str]] = []
    excluded: list[ExcludedEntry] = []
    notes: list[str] = []
    for src in items:
        if isinstance(src, (str, Path)):
            path = Path(src)
            kind = _kind_of(path)
            loaded = _load(path, kind)
            where = str(path)
        elif isinstance(src, BaselineResult):
            kind, loaded, where = "baseline", src, IN_MEMORY
        elif isinstance(src, TuningResult):
            kind, loaded, where = "tuning", src, str(src.directory) if src.directory else IN_MEMORY
        elif isinstance(src, FamiliesResult):
            kind, loaded, where = "families", src, str(src.directory) if src.directory else IN_MEMORY
        else:
            raise ReportError(f"unsupported source type {type(src).__name__}; pass a directory or a loaded result")
        if kind == "baseline":
            raw.append(_from_baseline(loaded, where))
            metas.append((loaded.task.value, loaded.metric, loaded.direction))
        elif kind == "tuning":
            label = f"tuned:{loaded.config.estimator}"
            cand = _from_tuning(loaded, where, label, "tuning")
            if cand is None:
                excluded.append(ExcludedEntry(label, where, "no completed trial"))
            else:
                raw.append(cand)
                metas.append((loaded.task, loaded.metric, loaded.direction))
        else:
            notes.extend(n for n in loaded.notes if n not in notes)
            for o in loaded.outcomes:
                label = f"family:{o.family}"
                fam_path = str(Path(where) / o.family) if where != IN_MEMORY else IN_MEMORY
                cand = _from_tuning(o.result, fam_path, label, "family") if o.result is not None else None
                if cand is None:
                    excluded.append(ExcludedEntry(label, fam_path, o.error or "no completed trial"))
                else:
                    raw.append(cand)
                    metas.append((o.result.task, o.result.metric, o.result.direction))
    if not raw:
        raise ReportError(["no candidate has a validation score"] + [f"{e.label}: {e.reason}" for e in excluded])

    task, metric, direction = metas[0]
    problems = [
        f"{c.label} ({c.source_path}) has task/metric/direction {m}, expected {(task, metric, direction)}"
        for c, m in zip(raw, metas)
        if m != (task, metric, direction)
    ]
    if problems:
        raise ReportError(["candidates are not comparable:"] + problems)

    sign = 1 if direction == "maximize" else -1
    order = sorted(range(len(raw)), key=lambda i: (-sign * raw[i].mean, math.inf if raw[i].std is None else raw[i].std, i))
    ranked = _unique_labels([raw[i] for i in order])
    ranked = tuple(replace(c, rank=i) for i, c in enumerate(ranked, 1))
    return CandidateSet(task, metric, direction, ranked, tuple(excluded), _split_warnings(ranked), tuple(notes))


def _unique_labels(cands: list[Candidate]) -> list[Candidate]:
    counts = Counter(c.label for c in cands)
    seen: Counter = Counter()
    out = []
    for c in cands:
        if counts[c.label] > 1:
            seen[c.label] += 1
            c = replace(c, label=f"{c.label}#{seen[c.label]}")
        out.append(c)
    return out


def _split_warnings(cands: tuple[Candidate, ...]) -> tuple[str, ...]:
    out: list[str] = []
    if len({c.n_rows for c in cands}) > 1:
        out.append("Candidates were scored on different row counts; their scores are not strictly comparable.")
    if len({c.n_folds for c in cands}) > 1:
        out.append("Candidates used different fold counts; their scores are not strictly comparable.")
    hashes = {c.splits_sha256 for c in cands if c.splits_sha256}
    if len(hashes) > 1:
        out.append("Tuning candidates used different validation splits; their scores are not strictly comparable.")
    return tuple(out)


# ---- diagnostics from saved artifacts ----


def _comparable(a: Candidate, b: Candidate) -> bool:
    if a.n_rows != b.n_rows or a.n_folds != b.n_folds:
        return False
    return a.splits_sha256 is None or b.splits_sha256 is None or a.splits_sha256 == b.splits_sha256


def improvements(cset: CandidateSet) -> list[dict[str, Any]]:
    """Tuned score minus baseline score, signed so positive is better; only for comparable splits."""
    sign = 1 if cset.direction == "maximize" else -1
    baselines = [c for c in cset.candidates if c.source_kind == "baseline"]
    rows = []
    for c in cset.candidates:
        if c.source_kind == "baseline":
            continue
        comparable = [b for b in baselines if _comparable(b, c)]
        if not comparable:
            continue
        base = next((b for b in comparable if b.estimator_key == c.estimator_key), comparable[0])
        rows.append(
            {
                "candidate": c.label,
                "baseline": base.label,
                "baseline_mean": base.mean,
                "candidate_mean": c.mean,
                "improvement": sign * (c.mean - base.mean),
            }
        )
    return rows


def _importance(trials: Sequence[TrialRecord], cfg: ModelReportConfig) -> list[tuple[str, float]]:
    from scipy.stats import rankdata, spearmanr

    done = [t for t in trials if t.state == "complete" and t.score is not None]
    if len(done) < cfg.min_trials_for_importance:
        raise _Skip(f"{len(done)} completed trial(s); at least {cfg.min_trials_for_importance} are needed")
    out: list[tuple[str, float]] = []
    for name in sorted({k for t in done for k in t.params}):
        pts = [(t.params[name], t.score) for t in done if t.params.get(name) is not None]
        if len(pts) < cfg.min_trials_for_importance:
            continue
        vals = [p[0] for p in pts]
        score = np.array([p[1] for p in pts], dtype=float)
        if np.ptp(score) == 0:
            continue
        if all(isinstance(v, Real) and not isinstance(v, bool) for v in vals):
            x = np.asarray(vals, dtype=float)
            if np.ptp(x) == 0:
                continue
            value = abs(float(spearmanr(x, score).statistic))
        else:
            ranks = rankdata(score)
            keys = np.array([str(v) for v in vals])
            sst = float(((ranks - ranks.mean()) ** 2).sum())
            ssb = sum(float((keys == k).sum() * (ranks[keys == k].mean() - ranks.mean()) ** 2) for k in set(keys))
            value = math.sqrt(ssb / sst) if sst else float("nan")
        if math.isfinite(value):
            out.append((name, value))
    if not out:
        raise _Skip("no parameter varied across the completed trials")
    out.sort(key=lambda r: (-r[1], r[0]))
    return out[: cfg.max_params_importance]


# ---- charts ----


@dataclass(frozen=True)
class _Ctx:
    cset: CandidateSet
    top: tuple[Candidate, ...]
    cfg: ModelReportConfig
    target: np.ndarray | None


def _footer(fig) -> None:
    fig.text(0.5, 0.008, "\n".join(textwrap.wrap(SCORE_STATEMENT, 125)), ha="center", va="bottom", fontsize=6.5, color="dimgray")


def _note(fig, text: str | None) -> None:
    if text:
        fig.text(0.5, 0.045, text, ha="center", fontsize=8, style="italic")


def _finish(fig, note: str | None = None):
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _note(fig, note)
    return fig


def _chart_scores(Figure, ctx: _Ctx):
    cands = ctx.cset.candidates[:30]
    fig = Figure(figsize=(8.5, max(3.5, 0.45 * len(cands) + 2.2)))
    ax = fig.add_subplot(111)
    ax.barh(range(len(cands)), [c.mean for c in cands], xerr=[c.std or 0.0 for c in cands], capsize=3)
    ax.set_yticks(range(len(cands)), [c.label for c in cands])
    ax.invert_yaxis()
    ax.set_xlabel(f"mean validation {ctx.cset.metric} ({ctx.cset.direction}); bars show fold std")
    ax.set_title("Validation score by candidate")
    note = f"Showing 30 of {len(ctx.cset.candidates)} candidates." if len(ctx.cset.candidates) > 30 else None
    return [_finish(fig, note)]


def _chart_fold_scores(Figure, ctx: _Ctx):
    cands = [c for c in ctx.top if c.n_folds >= 2 and c.fold_scores]
    if not cands:
        raise _Skip("no top candidate has more than one validation fold")
    fig = Figure(figsize=(8.5, max(3.5, 0.6 * len(cands) + 2.2)))
    ax = fig.add_subplot(111)
    for i, c in enumerate(cands):
        ax.scatter(c.fold_scores, [i] * len(c.fold_scores), alpha=0.7, label="fold score" if i == 0 else None)
        ax.scatter([c.mean], [i], marker="|", s=400, color="red", label="mean" if i == 0 else None)
    ax.set_yticks(range(len(cands)), [c.label for c in cands])
    ax.invert_yaxis()
    ax.set_xlabel(f"validation {ctx.cset.metric}")
    ax.set_title("Per-fold score variability")
    ax.legend()
    return [_finish(fig)]


def _chart_improvement(Figure, ctx: _Ctx):
    rows = improvements(ctx.cset)
    if not rows:
        raise _Skip("no tuned candidate has a baseline on matching splits")
    fig = Figure(figsize=(8.5, max(3.5, 0.45 * len(rows) + 2.2)))
    ax = fig.add_subplot(111)
    ax.barh(range(len(rows)), [r["improvement"] for r in rows])
    ax.set_yticks(range(len(rows)), [f"{r['candidate']} vs {r['baseline']}" for r in rows])
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"validation {ctx.cset.metric} change (positive is better)")
    ax.set_title("Tuned vs baseline improvement (validation only)")
    return [_finish(fig)]


def _chart_trials(Figure, cand: Candidate, ctx: _Ctx):
    if not cand.trials:
        raise _Skip("candidate has no trial history")
    done = [t for t in cand.trials if t.state == "complete" and t.score is not None]
    if not done:
        raise _Skip("no completed trials")
    sign = 1 if ctx.cset.direction == "maximize" else -1
    best_so_far, cur = [], None
    for t in done:
        cur = t.score if cur is None or sign * t.score > sign * cur else cur
        best_so_far.append(cur)
    keep = list(range(len(done)))
    if len(keep) > ctx.cfg.max_trials_plotted:
        best_i = next(i for i, t in enumerate(done) if t.number == cand.best_trial) if cand.best_trial is not None else 0
        keep = sorted(set([i for i in keep if i != best_i][: ctx.cfg.max_trials_plotted - 1]) | {best_i})
    fig = Figure(figsize=(8.5, 5))
    ax = fig.add_subplot(111)
    ax.scatter([done[i].number for i in keep], [done[i].score for i in keep], alpha=0.6, label="trial score")
    ax.plot([done[i].number for i in keep], [best_so_far[i] for i in keep], color="red", label="best so far")
    ax.set_xlabel("trial number")
    ax.set_ylabel(f"validation {ctx.cset.metric}")
    ax.set_title(f"Optimization history: {cand.label}")
    ax.legend()
    counts = ", ".join(f"{k}={v}" for k, v in sorted(cand.trial_counts.items()))
    note = f"Trials: {counts}." + (f" Showing {len(keep)} of {len(done)} completed trials." if len(keep) < len(done) else "")
    return [_finish(fig, note)]


def _chart_importance(Figure, cand: Candidate, ctx: _Ctx):
    if not cand.trials:
        raise _Skip("candidate has no trial history")
    rows = _importance(cand.trials, ctx.cfg)
    fig = Figure(figsize=(8.5, max(3.0, 0.4 * len(rows) + 2.2)))
    ax = fig.add_subplot(111)
    ax.barh(range(len(rows)), [r[1] for r in rows])
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("association with score (0 to 1)")
    ax.set_title(f"Parameter importance: {cand.label}")
    return [_finish(fig, "Descriptive heuristic (rank correlation with score); not causal.")]


def _oof_labels(cand: Candidate, y: np.ndarray) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    oof = cand.oof.sort_values("row")
    rows = oof["row"].to_numpy()
    if rows.max() >= len(y):
        raise _Skip(f"target has {len(y)} rows but predictions reference row {int(rows.max())}")
    return np.asarray(y)[rows], sorted({str(v) for v in y}), oof


def _chart_oof(Figure, cand: Candidate, ctx: _Ctx):
    if cand.source_kind != "baseline" or cand.oof is None or not cand.has_oof:
        raise _Skip("tuning results store no out-of-fold predictions; use a baseline directory")
    if ctx.target is None:
        raise _Skip("pass target= (the training target column) to include out-of-fold diagnostics")
    if len(ctx.target) != cand.n_rows:
        raise _Skip(f"target has {len(ctx.target)} rows but the run used {cand.n_rows}")
    truth, classes, oof = _oof_labels(cand, ctx.target)
    kind = cand.prediction_kind
    fig = Figure(figsize=(8.5, 4.5))
    if ctx.cset.task == "regression":
        pred = oof["prediction"].to_numpy(dtype=float)
        truth = truth.astype(float)
        ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122)
        ax1.scatter(truth, pred, s=8, alpha=0.5)
        lo, hi = float(min(truth.min(), pred.min())), float(max(truth.max(), pred.max()))
        ax1.plot([lo, hi], [lo, hi], color="red", linewidth=0.8)
        ax1.set_xlabel("actual")
        ax1.set_ylabel("predicted (out-of-fold)")
        ax1.set_title("Predicted vs actual")
        ax2.hist(truth - pred, bins=30)
        ax2.set_xlabel("residual (actual - predicted)")
        ax2.set_title("Residuals")
        return [_finish(fig)]
    truth_s = np.array([str(v) for v in truth])
    if kind == "labels":
        pred_s = oof["prediction"].astype(str).to_numpy()
    elif kind == "proba":
        cols = [c for c in oof.columns if c.startswith("proba_")]
        pred_s = np.array([c[len("proba_"):] for c in cols])[oof[cols].to_numpy().argmax(axis=1)]
    elif kind == "proba_positive":
        if len(classes) != 2:
            raise _Skip(f"binary probabilities need two target classes, found {len(classes)}")
        pred_s = np.where(oof["prediction"].to_numpy(dtype=float) >= 0.5, classes[1], classes[0])
    else:
        raise _Skip("stored predictions are not class labels or probabilities")
    labels = sorted(set(classes) | set(pred_s))
    matrix = pd.crosstab(pd.Categorical(truth_s, labels), pd.Categorical(pred_s, labels), dropna=False).to_numpy()
    has_cal = kind == "proba_positive"
    ax1 = fig.add_subplot(121 if has_cal else 111)
    ax1.imshow(matrix, cmap="Blues")
    for (i, j), v in np.ndenumerate(matrix):
        ax1.text(j, i, int(v), ha="center", va="center", fontsize=8, color="white" if v > matrix.max() / 2 else "black")
    ax1.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax1.set_yticks(range(len(labels)), labels)
    ax1.set_xlabel("predicted" + (" (threshold 0.5)" if has_cal else ""))
    ax1.set_ylabel("actual")
    ax1.set_title("Confusion matrix (out-of-fold)")
    if has_cal:
        p = oof["prediction"].to_numpy(dtype=float)
        hit = (truth_s == classes[1]).astype(float)
        edges = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(p, edges) - 1, 0, 9)
        xs = [p[idx == b].mean() for b in range(10) if (idx == b).any()]
        ys = [hit[idx == b].mean() for b in range(10) if (idx == b).any()]
        ax2 = fig.add_subplot(122)
        ax2.plot([0, 1], [0, 1], color="gray", linestyle="--")
        ax2.plot(xs, ys, marker="o")
        ax2.set_xlabel(f"mean predicted probability of '{classes[1]}'")
        ax2.set_ylabel("observed fraction")
        ax2.set_title("Calibration")
    return [_finish(fig)]


_GLOBAL_CHARTS = (
    ("score_comparison", _chart_scores),
    ("fold_variability", _chart_fold_scores),
    ("improvement", _chart_improvement),
)
_CANDIDATE_CHARTS = (
    ("trial_history", _chart_trials),
    ("parameter_importance", _chart_importance),
    ("oof_diagnostics", _chart_oof),
)


# ---- text pages ----


def _fmt(v: float | None) -> str:
    return "-" if v is None else f"{v:.6g}"


def _overview_lines(cset: CandidateSet, cfg: ModelReportConfig) -> list[str]:
    cands = cset.candidates
    lines = [
        f"Task: {cset.task}    metric: {cset.metric} ({cset.direction})",
        f"Candidates ranked: {len(cands)}    detailed pages: {min(cfg.top_n, len(cands))}    excluded: {len(cset.excluded)}",
        "",
        *textwrap.wrap(SCORE_STATEMENT, 95),
    ]
    for w in cset.warnings:
        lines += textwrap.wrap(f"WARNING: {w}", 95)
    for n in cset.notes:
        lines += textwrap.wrap(f"Note: {n}", 95)
    lines += ["", f"{'rank':>4}  {'candidate':<34} {'mean':>12} {'std':>10} {'folds':>5} {'trials':>6}"]
    for c in cands:
        trials = sum(c.trial_counts.values()) or "-"
        lines.append(f"{c.rank:>4}  {c.label[:34]:<34} {_fmt(c.mean):>12} {_fmt(c.std):>10} {c.n_folds:>5} {trials!s:>6}")
    if cset.excluded:
        lines += ["", "Excluded (no validation score):"]
        for e in cset.excluded:
            lines += textwrap.wrap(f"{e.label}: {e.reason}", 95, initial_indent="  ", subsequent_indent="    ")
    return lines


def _candidate_lines(c: Candidate, cset: CandidateSet) -> list[str]:
    d = c.details
    scores = list(c.fold_scores)
    lines = [
        f"Source: {c.source_kind} ({c.source_path})",
        f"Estimator: {c.estimator_key} ({c.estimator})",
        f"Validation {cset.metric}: mean {_fmt(c.mean)}  std {_fmt(c.std)}  "
        f"min {_fmt(min(scores) if scores else None)}  max {_fmt(max(scores) if scores else None)}",
    ]
    if c.n_folds < 2:
        lines.append("Variability is not estimable from a single validation fold.")
    lines += textwrap.wrap("Fold scores: " + ", ".join(f"{s:.6g}" for s in scores), 95)
    lines.append(f"Rows: {c.n_rows}    folds: {c.n_folds}    features: {d.get('n_features')}    seed: {d.get('seed')}")
    if "split" in d:
        lines += textwrap.wrap("Split: " + ", ".join(f"{k}={v}" for k, v in d["split"].items()), 95)
    lines.append(f"Categorical handling: {d.get('categorical_handling')}    device: {_device(d.get('device'))}")
    prep = d.get("preprocessing") or {}
    if prep:
        lines += textwrap.wrap("Preprocessing: " + ", ".join(f"{k}={v}" for k, v in prep.items()), 95)
    if c.trial_counts:
        counts = ", ".join(f"{k}={v}" for k, v in sorted(c.trial_counts.items()))
        lines.append(f"Trials: {counts}    best trial: {c.best_trial}")
    if d.get("fixed_estimator_params"):
        lines += textwrap.wrap(f"Fixed parameters: {d['fixed_estimator_params']}", 95)
    lines.append("Parameters:" if c.params else "Parameters: none sampled")
    for k, v in sorted(c.params.items()):
        lines += textwrap.wrap(f"{k} = {v}", 90, initial_indent="  ", subsequent_indent="      ")
    if c.versions:
        lines += textwrap.wrap("Versions: " + ", ".join(f"{k} {v}" for k, v in sorted(c.versions.items())), 95)
    if len(lines) > 42:
        lines = lines[:41] + ["... truncated; see the summary JSON for the full configuration."]
    return lines


def _device(device: Any) -> str:
    if isinstance(device, dict):
        return str(device.get("effective") or device.get("requested"))
    return str(device)


def _candidate_page(Figure, c: Candidate, cset: CandidateSet):
    fig = Figure(figsize=(8.5, 11))
    fig.text(0.07, 0.95, f"#{c.rank} {c.label}", fontsize=14, weight="bold", va="top")
    fig.text(0.07, 0.91, "\n".join(_candidate_lines(c, cset)), fontsize=8, family="monospace", va="top")
    if c.fold_scores:
        ax = fig.add_axes((0.12, 0.09, 0.8, 0.2))
        ax.bar(range(1, len(c.fold_scores) + 1), c.fold_scores)
        ax.axhline(c.mean, color="red", linestyle="--", label="mean")
        ax.set_xlabel("fold")
        ax.set_ylabel(cset.metric)
        ax.set_title("Per-fold validation score", fontsize=9)
        ax.legend(fontsize=7)
    return fig


# ---- writer ----


def write_model_report(
    sources: Any,
    output_dir: str | Path,
    *,
    filename: str = "model_report.pdf",
    config: ModelReportConfig | None = None,
    overwrite: bool = False,
    target: Any = None,
) -> ModelReportResult:
    """Write the model comparison PDF and summary JSON into `output_dir` from saved results.

    `target` is the optional training target column; it enables confusion, calibration, and residual
    plots for baseline candidates, whose saved predictions do not include the true values.
    """
    cfg = config or ModelReportConfig()
    Figure, PdfPages = _require_matplotlib()
    if Path(filename).name != filename or not filename.lower().endswith(".pdf"):
        raise ReportError(f"setting 'filename' must be a plain file name ending in .pdf, got {filename!r}")
    y = None
    if target is not None:
        y = np.asarray(target)
        if y.ndim != 1:
            raise ReportError(f"setting 'target' must be one-dimensional, got shape {y.shape}")
    out_dir = Path(output_dir)
    pdf_path = out_dir / filename
    summary_path = out_dir / (Path(filename).stem + "_summary.json")
    if not overwrite:
        existing = [str(p) for p in (pdf_path, summary_path) if p.exists()]
        if existing:
            raise ReportError(f"report output already exists: {existing}; pass overwrite=True to replace")
    cset = collect_candidates(sources)
    out_dir.mkdir(parents=True, exist_ok=True)

    top = cset.candidates[: cfg.top_n]
    ctx = _Ctx(cset, top, cfg, y)
    included: list[str] = []
    skipped: dict[str, str] = {}
    tmp_pdf = tmp_json = None
    try:
        fd, tmp_pdf = tempfile.mkstemp(suffix=".pdf", dir=out_dir)
        os.close(fd)
        with PdfPages(tmp_pdf) as pdf:

            def save(figs) -> None:
                for fig in figs:
                    _footer(fig)
                    pdf.savefig(fig)

            save(_text_pages(Figure, cfg.title, _overview_lines(cset, cfg)))
            for name, fn in _GLOBAL_CHARTS:
                try:
                    figs = fn(Figure, ctx)
                except _Skip as skip:
                    skipped[name] = str(skip)
                    continue
                included.append(name)
                save(figs)
            for c in top:
                save([_candidate_page(Figure, c, cset)])
                for name, fn in _CANDIDATE_CHARTS:
                    key = f"{name}:{c.label}"
                    if name in ("trial_history", "parameter_importance") and c.source_kind == "baseline":
                        skipped[key] = "baseline runs have no trials"
                        continue
                    try:
                        figs = fn(Figure, c, ctx)
                    except _Skip as skip:
                        skipped[key] = str(skip)
                        continue
                    included.append(key)
                    save(figs)
            pages = pdf.get_pagecount()
        result = ModelReportResult(
            pdf_path, summary_path, pages, cset.task, cset.metric, cset.direction,
            cset.candidates, cset.excluded, cset.warnings, tuple(included), skipped,
        )
        payload = {**result.to_dict(), "config": asdict(cfg), "notes": list(cset.notes)}
        fd, tmp_json = tempfile.mkstemp(suffix=".json", dir=out_dir)
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(payload, allow_nan=False))
        os.replace(tmp_pdf, pdf_path)
        tmp_pdf = None
        os.replace(tmp_json, summary_path)
        tmp_json = None
    finally:
        for tmp in (tmp_pdf, tmp_json):
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)
    return result
