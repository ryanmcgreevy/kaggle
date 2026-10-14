import hashlib
import json
import subprocess
import sys
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from automl import (
    BaselineConfig,
    ModelReportConfig,
    ReportError,
    TuningConfig,
    ValidationConfig,
    collect_candidates,
    load_baseline,
    load_config,
    make_splits,
    run_baseline,
    save_baseline,
    write_model_report,
)
from automl import model_report
from automl.metrics import Task, get_metric
from automl.task import ResolvedTask
from automl.tuning import SCHEMA_VERSION as TUNING_SCHEMA
from automl.families import FAMILIES_SCHEMA_VERSION

DEFAULT_METRIC = {"binary": "roc_auc", "multiclass": "log_loss", "regression": "rmse"}


def make_frame(task, n=90, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"id": np.arange(n), "x1": rng.normal(size=n), "x2": rng.normal(size=n), "cat": rng.choice(["a", "b", "c"], n)})
    signal = df["x2"] + (df["cat"] == "a")
    if task == "regression":
        df["y"] = signal * 2 + rng.normal(0, 0.1, n) + 5
    elif task == "binary":
        df["y"] = (signal + rng.normal(0, 0.5, n) > 0.5).astype(int)
        df.loc[:1, "y"] = [0, 1]
    else:
        df["y"] = pd.cut(signal, [-9, -0.3, 0.8, 9], labels=["lo", "mid", "hi"]).astype(str)
    return df


def run_saved(tmp_path, name, task="binary", metric=None, strategy="kfold", df=None):
    df = make_frame(task) if df is None else df
    resolved = ResolvedTask(Task(task), get_metric(metric or DEFAULT_METRIC[task]))
    splits = make_splits(df["y"], resolved, ValidationConfig(strategy=strategy, n_folds=3))
    result = run_baseline(df, "y", ("id",), resolved, splits, BaselineConfig(save_models=False))
    save_baseline(result, tmp_path / name)
    return df, tmp_path / name


def trial(number, score, params=None, state="complete", folds=None, reason=None):
    folds = [score - 0.01, score, score + 0.01] if folds is None and state == "complete" else (folds or [])
    return {
        "number": number, "state": state, "params": params or {}, "score": score if state == "complete" else None,
        "fold_scores": folds, "fold_std": float(np.std(folds)) if state == "complete" else None,
        "duration_seconds": 0.1, "failure_reason": reason, "failure_category": "other" if state == "failed" else None,
    }


def write_tuning(path, trials, best="auto", direction="maximize", task="binary", metric="roc_auc",
                 estimator="hist_gradient_boosting", n_rows=90, n_folds=3, sha="s1"):
    path.mkdir(parents=True)
    done = [t for t in trials if t["state"] == "complete"]
    if best == "auto":
        pick = max if direction == "maximize" else min
        best = pick(done, key=lambda t: t["score"])["number"] if done else None
    payload = {
        "schema_version": TUNING_SCHEMA, "task": task, "metric": {"name": metric, "direction": direction},
        "config": asdict(TuningConfig(n_trials=len(trials), estimator=estimator)), "space": {},
        "fingerprint": {"n_rows": n_rows, "n_folds": n_folds, "splits_sha256": sha, "seed": 42, "estimator_params": {},
                        "native_categoricals": False, "device": "cpu", "features": ["a", "b"], "preprocessing": {"encoder": "ordinal"}},
        "budgets": [], "best_trial": best, "trials": trials, "versions": {"python": "3"}, "notes": [], "files": {"study": "study.db"},
    }
    (path / "tuning.json").write_text(json.dumps(payload))
    return path


def write_families(path, members, task="binary", metric="roc_auc", direction="maximize"):
    """members: {family: (status, trials-or-None, error)}."""
    path.mkdir(parents=True)
    entries = []
    for fam, (status, trials, error) in members.items():
        if trials is not None:
            write_tuning(path / fam, trials, direction=direction, task=task, metric=metric, estimator=fam)
        entries.append({"family": fam, "status": status, "directory": fam, "error": error})
    payload = {
        "schema_version": FAMILIES_SCHEMA_VERSION, "task": task, "metric": {"name": metric, "direction": direction},
        "config": asdict(TuningConfig(n_trials=3)), "budget_plan": {f: {"n_trials": 3, "timeout_seconds": None} for f in members},
        "families": entries, "comparison": [], "best_family": None, "notes": ["Family note."],
    }
    (path / "tuning_families.json").write_text(json.dumps(payload))
    return path


GOOD = [trial(0, 0.70, {"learning_rate": 0.1}), trial(1, 0.80, {"learning_rate": 0.2}), trial(2, 0.75, {"learning_rate": 0.3})]


@pytest.fixture
def mpl():
    return pytest.importorskip("matplotlib")


@pytest.fixture(scope="module")
def base_dir(tmp_path_factory):
    return run_saved(tmp_path_factory.mktemp("base"), "baseline")


def hash_tree(root):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


# ---- collection ----


def test_ranking_both_directions_and_ties(tmp_path):
    a = write_tuning(tmp_path / "a", [trial(0, 0.9)])
    b = write_tuning(tmp_path / "b", [trial(0, 0.8)], estimator="lightgbm")
    cs = collect_candidates([b, a])
    assert [c.label for c in cs.candidates] == ["tuned:hist_gradient_boosting", "tuned:lightgbm"]
    assert [c.rank for c in cs.candidates] == [1, 2]
    lo = write_tuning(tmp_path / "lo", [trial(0, 0.2)], direction="minimize", metric="log_loss")
    hi = write_tuning(tmp_path / "hi", [trial(0, 0.5)], direction="minimize", metric="log_loss", estimator="xgboost")
    cs = collect_candidates([hi, lo])
    assert cs.direction == "minimize" and cs.candidates[0].mean == 0.2
    tie1 = write_tuning(tmp_path / "t1", [trial(0, 0.5, folds=[0.1, 0.9, 0.5])], estimator="xgboost")
    tie2 = write_tuning(tmp_path / "t2", [trial(0, 0.5, folds=[0.49, 0.5, 0.51])], estimator="lightgbm")
    assert collect_candidates([tie1, tie2]).candidates[0].label == "tuned:lightgbm"


def test_duplicate_labels_are_disambiguated(tmp_path):
    a = write_tuning(tmp_path / "a", [trial(0, 0.9)])
    b = write_tuning(tmp_path / "b", [trial(0, 0.8)])
    assert [c.label for c in collect_candidates([a, b]).candidates] == [
        "tuned:hist_gradient_boosting#1", "tuned:hist_gradient_boosting#2",
    ]


def test_families_expand_and_exclusions(tmp_path):
    fam = write_families(tmp_path / "fam", {
        "hist_gradient_boosting": ("complete", GOOD, None),
        "lightgbm": ("failed", [trial(0, None, state="failed", reason="boom")], "no completed trial: boom"),
        "xgboost": ("failed", None, "library missing"),
    })
    cs = collect_candidates(fam)
    assert [c.label for c in cs.candidates] == ["family:hist_gradient_boosting"]
    assert cs.candidates[0].mean == 0.8 and cs.candidates[0].best_trial == 1
    assert {e.label: e.reason for e in cs.excluded} == {
        "family:lightgbm": "no completed trial: boom", "family:xgboost": "library missing",
    }
    assert cs.notes == ("Family note.",)


def test_tuning_without_best_is_excluded(tmp_path):
    ok = write_tuning(tmp_path / "ok", GOOD)
    none = write_tuning(tmp_path / "none", [trial(0, None, state="failed", reason="x")], estimator="xgboost")
    cs = collect_candidates([ok, none])
    assert len(cs.candidates) == 1 and cs.excluded[0].label == "tuned:xgboost"


def test_only_excluded_is_an_error(tmp_path):
    none = write_tuning(tmp_path / "none", [trial(0, None, state="failed", reason="x")])
    with pytest.raises(ReportError, match="no candidate has a validation score"):
        collect_candidates(none)


def test_mismatched_candidates_name_each_path(tmp_path):
    a = write_tuning(tmp_path / "a", GOOD)
    b = write_tuning(tmp_path / "b", GOOD, metric="log_loss", direction="minimize")
    c = write_tuning(tmp_path / "c", GOOD, task="regression", metric="rmse", direction="minimize")
    with pytest.raises(ReportError) as exc:
        collect_candidates([a, b, c])
    text = str(exc.value)
    assert str(b) in text and str(c) in text
    assert f"{a} " not in text


@pytest.mark.parametrize("make, match", [
    (lambda p: p / "missing", "is not a directory"),
    (lambda p: (p / "empty").mkdir() or p / "empty", "holds none of"),
])
def test_bad_directories(tmp_path, make, match):
    with pytest.raises(ReportError, match=match):
        collect_candidates(make(tmp_path))


def test_ambiguous_directory(tmp_path):
    d = write_tuning(tmp_path / "d", GOOD)
    (d / "tuning_families.json").write_text("{}")
    with pytest.raises(ReportError, match="ambiguous"):
        collect_candidates(d)


def test_corrupt_artifact_is_report_error(tmp_path):
    d = write_tuning(tmp_path / "d", GOOD)
    (d / "tuning.json").write_text("{not json")
    with pytest.raises(ReportError, match="not valid JSON"):
        collect_candidates(d)


def test_bad_source_inputs():
    with pytest.raises(ReportError, match="at least one"):
        collect_candidates([])
    with pytest.raises(ReportError, match="unsupported source type"):
        collect_candidates([42])


def test_split_warnings(tmp_path):
    a = write_tuning(tmp_path / "a", GOOD, sha="s1")
    b = write_tuning(tmp_path / "b", GOOD, sha="s2", estimator="xgboost")
    c = write_tuning(tmp_path / "c", GOOD, n_rows=50, n_folds=5, estimator="lightgbm")
    assert collect_candidates([a]).warnings == ()
    text = " ".join(collect_candidates([a, b, c]).warnings)
    assert "row counts" in text and "fold counts" in text and "different validation splits" in text


def test_baseline_candidate_and_loaded_results(base_dir):
    _, d = base_dir
    cs = collect_candidates([d, load_baseline(d)])
    c = cs.candidates[0]
    assert c.source_kind == "baseline" and c.has_oof and c.n_folds == 3 and len(c.fold_scores) == 3
    assert np.isclose(np.mean(c.fold_scores), c.mean)
    assert {x.source_path for x in cs.candidates} == {str(d), model_report.IN_MEMORY}
    json.dumps(c.to_dict(), allow_nan=False)


# ---- diagnostics ----


@pytest.mark.parametrize("direction, metric, base, tuned", [("maximize", "roc_auc", 0.7, 0.8), ("minimize", "log_loss", 0.7, 0.6)])
def test_improvement_sign_matches_direction(tmp_path, direction, metric, base, tuned):
    t = write_tuning(tmp_path / "t", [trial(0, tuned)], direction=direction, metric=metric)
    cs = collect_candidates(t)
    cs_with = model_report.CandidateSet(cs.task, cs.metric, cs.direction, (
        model_report.replace(cs.candidates[0], mean=tuned),
        model_report.replace(cs.candidates[0], source_kind="baseline", label="baseline:hist_gradient_boosting", mean=base, splits_sha256=None),
    ), (), (), ())
    (row,) = model_report.improvements(cs_with)
    assert row["improvement"] == pytest.approx(0.1)


def test_improvement_skipped_when_splits_differ(tmp_path, base_dir):
    _, bd = base_dir
    same = write_tuning(tmp_path / "same", GOOD)
    other = write_tuning(tmp_path / "other", GOOD, n_folds=5, estimator="xgboost")
    rows = model_report.improvements(collect_candidates([bd, same, other]))
    assert [r["candidate"] for r in rows] == ["tuned:hist_gradient_boosting"] and rows[0]["baseline"].startswith("baseline:")


def test_importance_order_threshold_and_categorical():
    rng = np.random.default_rng(0)
    trials = []
    for i in range(30):
        lr = float(rng.uniform(0, 1))
        trials.append(model_report.TrialRecord(
            i, "complete", {"lr": lr, "noise": float(rng.uniform()), "const": 1, "kind": "a" if lr > 0.5 else "b", "none": None},
            2 * lr + rng.normal(0, 0.05), (), None, None,
        ))
    cfg = ModelReportConfig(min_trials_for_importance=10)
    rows = model_report._importance(trials, cfg)
    names = [r[0] for r in rows]
    assert names[0] in {"lr", "kind"} and "const" not in names and "none" not in names
    assert dict(rows)["lr"] > dict(rows)["noise"]
    assert len(model_report._importance(trials, ModelReportConfig(max_params_importance=1))) == 1
    with pytest.raises(model_report._Skip, match="at least 10"):
        model_report._importance(trials[:5], cfg)


# ---- config ----


def test_config_validation_collects_problems():
    with pytest.raises(ReportError) as exc:
        ModelReportConfig(top_n=0, max_trials_plotted="x", title=" ")
    text = str(exc.value)
    assert "'top_n'" in text and "'max_trials_plotted'" in text and "'title'" in text


def test_toml_table_and_overrides(data_dir):
    path = data_dir / "run.toml"
    base = path.read_text()
    path.write_text(base + "\n[model_report]\ntop_n = 5\ntitle = \"Mine\"\n")
    cfg = load_config(path).model_report_config
    assert cfg.top_n == 5 and cfg.title == "Mine"
    assert load_config(path, max_trials_plotted=2).model_report_config.max_trials_plotted == 2
    assert load_config(path, model_report_config=ModelReportConfig(top_n=9)).model_report_config.top_n == 9
    assert load_config(data_dir / "run.toml", title="T").report_config.title == "T"
    path.write_text(base + "\n[model_report]\nbogus = 1\n")
    with pytest.raises(Exception, match=r"unknown keys in \[model_report\]"):
        load_config(path)
    path.write_text(base + "\n[model_report]\ntop_n = 0\n")
    with pytest.raises(Exception, match="'top_n'"):
        load_config(path)


def test_default_config_without_table(data_dir):
    assert load_config(data_dir / "run.toml").model_report_config == ModelReportConfig()


# ---- PDF ----


def _mixed(tmp_path, base_dir):
    _, bd = base_dir
    t = write_tuning(tmp_path / "t", [trial(i, 0.6 + 0.02 * i, {"learning_rate": 0.01 * i, "kind": "a" if i % 2 else "b"}) for i in range(12)])
    fam = write_families(tmp_path / "fam", {"xgboost": ("complete", GOOD, None), "lightgbm": ("failed", None, "not installed")})
    return [bd, t, fam]


def _page_count(path):
    import re

    return len(re.findall(rb"/Type /Page\b", path.read_bytes()))


def test_pdf_from_mixed_sources(tmp_path, base_dir, mpl):
    res = write_model_report(_mixed(tmp_path, base_dir), tmp_path / "out", config=ModelReportConfig(top_n=2))
    assert res.path.read_bytes().startswith(b"%PDF") and res.path.stat().st_size > 0
    assert res.pages == _page_count(res.path)
    assert {"score_comparison", "fold_variability", "improvement"} <= set(res.charts)
    assert any(c.startswith("trial_history:") for c in res.charts)
    assert any(c.startswith("parameter_importance:") for c in res.charts)
    assert len(res.candidates) == 3 and res.excluded[0].label == "family:lightgbm"
    summary = json.loads(res.summary_path.read_text())
    assert summary["config"]["top_n"] == 2 and len(summary["candidates"]) == 3
    assert summary["score_statement"] == model_report.SCORE_STATEMENT
    json.dumps(res.to_dict(), allow_nan=False)


def test_top_n_caps_detail_pages_but_not_table(tmp_path, base_dir, mpl):
    srcs = _mixed(tmp_path, base_dir)
    small = write_model_report(srcs, tmp_path / "a", config=ModelReportConfig(top_n=1))
    big = write_model_report(srcs, tmp_path / "b", config=ModelReportConfig(top_n=3))
    assert big.pages > small.pages
    assert len(small.candidates) == len(big.candidates) == 3


def test_loaded_results_are_accepted(tmp_path, base_dir, mpl):
    _, bd = base_dir
    res = write_model_report(load_baseline(bd), tmp_path)
    assert res.candidates[0].source_path == model_report.IN_MEMORY


def test_score_statement_on_every_page_without_leaderboard_claims(tmp_path, base_dir, mpl, monkeypatch):
    seen = []
    real = model_report._footer

    def spy(fig):
        real(fig)
        seen.append(" ".join(" ".join(t.get_text() for t in [*fig.texts, *[a.title for a in fig.axes]]).split()))

    monkeypatch.setattr(model_report, "_footer", spy)
    res = write_model_report(_mixed(tmp_path, base_dir), tmp_path / "out")
    assert len(seen) == res.pages
    for text in seen:
        assert "not leaderboard predictions" in text
        assert text.lower().count("leaderboard") == text.count("not leaderboard predictions")


def test_readable_sections(tmp_path, base_dir, mpl, monkeypatch):
    texts = []
    real = model_report._footer
    monkeypatch.setattr(model_report, "_footer", lambda fig: (real(fig), texts.append("\n".join(t.get_text() for t in fig.texts)))[0])
    write_model_report(_mixed(tmp_path, base_dir), tmp_path / "out", config=ModelReportConfig(title="My Models"))
    joined = "\n".join(texts)
    assert "My Models" in joined and "candidate" in joined and "#1 " in joined
    assert "Source: baseline" in joined and "Parameters:" in joined and "Excluded" in joined


def test_single_fold_variability_message(tmp_path, mpl, monkeypatch):
    _, d = run_saved(tmp_path, "holdout", strategy="holdout")
    texts = []
    real = model_report._footer
    monkeypatch.setattr(model_report, "_footer", lambda fig: (real(fig), texts.append("\n".join(t.get_text() for t in fig.texts)))[0])
    res = write_model_report(d, tmp_path / "out")
    assert res.candidates[0].n_folds == 1
    assert "not estimable" in "\n".join(texts) and "fold_variability" in res.charts_skipped


def test_parameter_importance_skip_reason(tmp_path, mpl):
    t = write_tuning(tmp_path / "t", GOOD)
    res = write_model_report(t, tmp_path / "out")
    assert "at least 10" in res.charts_skipped["parameter_importance:tuned:hist_gradient_boosting"]
    assert "trial_history:tuned:hist_gradient_boosting" in res.charts


def test_trial_history_cap_keeps_best(tmp_path, mpl):
    from matplotlib.figure import Figure

    trials = [trial(i, 0.5 + 0.001 * i) for i in range(40)]
    trials[5] = trial(5, 0.99)
    t = write_tuning(tmp_path / "t", trials)
    cs = collect_candidates(t)
    ctx = model_report._Ctx(cs, cs.candidates, ModelReportConfig(max_trials_plotted=10), None)
    (fig,) = model_report._chart_trials(Figure, cs.candidates[0], ctx)
    xs = fig.axes[0].collections[0].get_offsets()[:, 0]
    assert len(xs) == 10 and 5 in xs
    assert any("Showing 10 of 40" in x.get_text() for x in fig.texts)


@pytest.mark.parametrize("task, metric, expect", [
    ("binary", "roc_auc", {"Confusion matrix (out-of-fold)", "Calibration"}),
    ("binary", "accuracy", {"Confusion matrix (out-of-fold)"}),
    ("multiclass", "log_loss", {"Confusion matrix (out-of-fold)"}),
    ("regression", "rmse", {"Predicted vs actual", "Residuals"}),
])
def test_oof_diagnostics_with_target(tmp_path, mpl, task, metric, expect):
    from matplotlib.figure import Figure

    df, d = run_saved(tmp_path, "b", task=task, metric=metric)
    cs = collect_candidates(d)
    ctx = model_report._Ctx(cs, cs.candidates, ModelReportConfig(), df["y"].to_numpy())
    (fig,) = model_report._chart_oof(Figure, cs.candidates[0], ctx)
    assert {a.get_title() for a in fig.axes} == expect
    res = write_model_report(d, tmp_path / "out", target=df["y"])
    assert any(c.startswith("oof_diagnostics:") for c in res.charts)


def test_oof_skip_reasons(tmp_path, base_dir, mpl):
    df, bd = base_dir
    res = write_model_report(bd, tmp_path / "a")
    assert "pass target=" in res.charts_skipped["oof_diagnostics:baseline:hist_gradient_boosting"]
    res = write_model_report(bd, tmp_path / "b", target=df["y"].iloc[:10])
    assert "10 rows" in res.charts_skipped["oof_diagnostics:baseline:hist_gradient_boosting"]
    t = write_tuning(tmp_path / "t", GOOD)
    res = write_model_report(t, tmp_path / "c", target=df["y"])
    assert "no out-of-fold predictions" in res.charts_skipped["oof_diagnostics:tuned:hist_gradient_boosting"]
    with pytest.raises(ReportError, match="one-dimensional"):
        write_model_report(bd, tmp_path / "d", target=df)


def test_real_tuning_and_families(tmp_path, base_dir, mpl):
    pytest.importorskip("optuna")
    from automl import tune, tune_families

    df, bd = base_dir
    resolved = ResolvedTask(Task.BINARY, get_metric("roc_auc"))
    splits = make_splits(df["y"], resolved, ValidationConfig(n_folds=3))
    space = {"learning_rate": {"type": "float", "low": 0.05, "high": 0.3, "log": True}, "max_iter": {"type": "int", "low": 5, "high": 20}}
    tune(df, "y", ("id",), resolved, splits, TuningConfig(n_trials=12), space=space, directory=tmp_path / "tuned")
    tune_families(df, "y", ("id",), resolved, splits, TuningConfig(n_trials=3), families=["hist_gradient_boosting"], directory=tmp_path / "fams")
    res = write_model_report([bd, tmp_path / "tuned", tmp_path / "fams"], tmp_path / "out", target=df["y"])
    kinds = {c.source_kind for c in res.candidates}
    assert kinds == {"baseline", "tuning", "family"}
    assert "improvement" in res.charts and any(c.startswith("parameter_importance:tuned") for c in res.charts)


def test_accurate_after_restart(tmp_path, base_dir, mpl):
    srcs = _mixed(tmp_path, base_dir)
    first = write_model_report(srcs, tmp_path / "first")
    code = (
        "import sys\nfrom automl import write_model_report\n"
        "write_model_report(sys.argv[2:], sys.argv[1])\n"
    )
    subprocess.run([sys.executable, "-c", code, str(tmp_path / "second"), *map(str, srcs)], check=True)

    def norm(path):
        d = json.loads(path.read_text())
        d.pop("path"), d.pop("summary_path")
        return d

    assert norm(first.summary_path) == norm(tmp_path / "second" / "model_report_summary.json")


def test_inputs_not_mutated_and_output_location(tmp_path, base_dir, mpl):
    srcs = _mixed(tmp_path, base_dir)
    roots = [p for p in srcs]
    before = [hash_tree(p) for p in roots]
    out = tmp_path / "nested" / "out"
    res = write_model_report(srcs, out)
    assert [hash_tree(p) for p in roots] == before
    assert sorted(p.name for p in out.iterdir()) == ["model_report.pdf", "model_report_summary.json"]
    assert res.path.parent == out


def test_overwrite_rules(tmp_path, base_dir, mpl):
    _, bd = base_dir
    write_model_report(bd, tmp_path)
    before = (tmp_path / "model_report.pdf").read_bytes()
    with pytest.raises(ReportError, match="overwrite=True"):
        write_model_report(bd, tmp_path)
    assert (tmp_path / "model_report.pdf").read_bytes() == before
    res = write_model_report(bd, tmp_path, overwrite=True, config=ModelReportConfig(title="Second"))
    assert json.loads(res.summary_path.read_text())["config"]["title"] == "Second"


def test_failure_leaves_no_files_and_keeps_existing(tmp_path, base_dir, mpl, monkeypatch):
    _, bd = base_dir
    monkeypatch.setattr(model_report, "_GLOBAL_CHARTS", (("bad", lambda *a: 1 / 0),))
    with pytest.raises(ZeroDivisionError):
        write_model_report(bd, tmp_path / "out")
    assert list((tmp_path / "out").iterdir()) == []
    monkeypatch.undo()
    write_model_report(bd, tmp_path)
    before = (tmp_path / "model_report.pdf").read_bytes()
    monkeypatch.setattr(model_report, "_GLOBAL_CHARTS", (("bad", lambda *a: 1 / 0),))
    with pytest.raises(ZeroDivisionError):
        write_model_report(bd, tmp_path, overwrite=True)
    assert (tmp_path / "model_report.pdf").read_bytes() == before


def test_filename_validation(tmp_path, base_dir, mpl):
    for bad in ("../x.pdf", "report.txt"):
        with pytest.raises(ReportError, match="'filename'"):
            write_model_report(base_dir[1], tmp_path, filename=bad)


def test_missing_matplotlib_message(tmp_path, base_dir, monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib.backends.backend_pdf", None)
    with pytest.raises(ReportError, match=r"pip install -e '\.\[reports\]'"):
        write_model_report(base_dir[1], tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_invalid_sources_name_the_path(tmp_path, mpl):
    with pytest.raises(ReportError, match="nowhere"):
        write_model_report(tmp_path / "nowhere", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_import_and_collect_without_matplotlib_or_optuna(tmp_path):
    d = write_tuning(tmp_path / "t", GOOD)
    code = (
        "import sys, automl\n"
        "cs = automl.collect_candidates(sys.argv[1])\n"
        "assert cs.candidates[0].mean == 0.8\n"
        "assert 'matplotlib' not in sys.modules and 'optuna' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code, str(d)], check=True)
