import json
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from automl import (
    DataContractError,
    EdaResult,
    Finding,
    ReportConfig,
    ReportError,
    Task,
    build_findings,
    load_config,
    load_data,
    resolve_task_metric,
    save_eda,
    summarize,
    summarize_bundle,
    write_eda_report,
)

IDS = ("row_id",)


def _eda(eda_frames, **kw) -> EdaResult:
    train, test = eda_frames
    return summarize(train, test, target="y", id_columns=IDS, task=Task.BINARY, **kw)


def _codes(findings, code):
    return [f for f in findings if f.code == code]


def _page_count(path) -> int:
    return len(re.findall(rb"/Type /Page\b", path.read_bytes()))


@pytest.fixture
def mpl():
    return pytest.importorskip("matplotlib")


def test_findings_on_mixed_fixture(eda_frames):
    f = build_findings(_eda(eda_frames), ReportConfig(outlier_fraction_warn=0.01))
    assert all(isinstance(x, Finding) for x in f)
    codes = {x.code for x in f}
    assert {"duplicate_rows", "duplicate_ids", "constant", "all_null", "id_like", "high_outliers",
            "drift", "unseen_test_categories"} <= codes
    assert {x.column for x in _codes(f, "drift")} == {"num", "cat"}
    assert _codes(f, "all_null")[0].column == "allnull"
    assert not _codes(f, "constant")[0].column == "allnull"
    json.dumps([x.to_dict() for x in f], allow_nan=False)


def test_findings_order_is_deterministic(eda_frames):
    eda = _eda(eda_frames)
    a, b = build_findings(eda), build_findings(eda.to_dict())
    assert a == b
    keys = [({"warning": 0, "info": 1}[x.severity], x.scope) for x in a]
    assert [k[0] for k in keys] == sorted(k[0] for k in keys)


def test_threshold_boundaries():
    def codes(df, cfg, **kw):
        return {x.code for x in build_findings(summarize(df, **kw), cfg)}

    df = pd.DataFrame({"x": [1.0, np.nan, 3.0, 4.0, 5.0]})  # 20% missing
    assert "high_missing" in codes(df, ReportConfig(high_missing_fraction=0.2))
    assert "high_missing" not in codes(df, ReportConfig(high_missing_fraction=0.21))

    y = pd.DataFrame({"y": [0] * 10 + [1], "x": range(11)})
    assert "class_imbalance" in codes(y, ReportConfig(imbalance_ratio_warn=10), target="y", task=Task.BINARY)
    assert "class_imbalance" not in codes(y, ReportConfig(imbalance_ratio_warn=10.5), target="y", task=Task.BINARY)

    dup = pd.DataFrame({"x": [1, 1, 2, 3, 4]})  # 1 of 5 duplicated = 0.2
    assert "duplicate_rows" in codes(dup, ReportConfig(duplicate_fraction_warn=0.2))
    assert "duplicate_rows" not in codes(dup, ReportConfig(duplicate_fraction_warn=0.25))


def test_one_sided_columns_and_null_delta(eda_frames):
    train, test = eda_frames
    t = test.assign(extra=1).assign(num=np.nan)
    f = build_findings(summarize(train, t, target="y", id_columns=IDS))
    assert _codes(f, "one_sided_columns")[0].scope == "train_test"
    assert [x.column for x in _codes(f, "null_delta")] == ["num"]


def test_clean_data_has_no_findings():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
    assert build_findings(summarize(df)) == ()


def test_findings_from_saved_artifact_and_bad_input(eda_frames, tmp_path):
    eda = _eda(eda_frames)
    save_eda(eda, tmp_path / "eda.json")
    assert build_findings(tmp_path / "eda.json") == build_findings(eda)
    with pytest.raises(ReportError, match="schema_version"):
        build_findings({"schema_version": 9})
    with pytest.raises(ReportError, match="EdaResult"):
        build_findings(42)


def test_pdf_is_valid_and_matches_result(eda_frames, tmp_path, mpl):
    res = write_eda_report(_eda(eda_frames), tmp_path / "out")
    assert res.path == tmp_path / "out" / "eda_report.pdf"
    data = res.path.read_bytes()
    assert data.startswith(b"%PDF") and len(data) > 1000
    assert res.pages == _page_count(res.path) >= 4
    assert set(res.charts) == {"missingness", "numeric_histograms", "target_distribution",
                               "categorical_top_values", "train_test_drift", "outlier_fractions"}
    assert res.charts_skipped == {}
    payload = json.loads(res.findings_path.read_text())
    assert payload["schema_version"] == 1 and len(payload["findings"]) == len(res.findings)
    assert payload["config"]["title"] == "EDA Report"
    json.dumps(res.to_dict(), allow_nan=False)
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["eda_report.pdf", "eda_report_findings.json"]


def test_pdf_from_saved_artifact_in_fresh_call(eda_frames, tmp_path, mpl):
    eda = _eda(eda_frames)
    save_eda(eda, tmp_path / "eda.json")
    from_file = write_eda_report(tmp_path / "eda.json", tmp_path / "a")
    direct = write_eda_report(eda, tmp_path / "b")
    assert from_file.pages == direct.pages and from_file.findings == direct.findings


def test_missing_optional_sections_are_skipped(tmp_path, mpl):
    df = pd.DataFrame({"c": ["x", "y", "z", "x"], "k": ["a", "a", "a", "a"]})
    res = write_eda_report(summarize(df), tmp_path)
    assert "numeric_histograms" in res.charts_skipped and "target_distribution" in res.charts_skipped
    assert "train_test_drift" in res.charts_skipped and "missingness" in res.charts_skipped
    assert res.charts == ("categorical_top_values",)
    assert res.pages == _page_count(res.path)


def test_degenerate_columns(tmp_path, mpl):
    df = pd.DataFrame({"const": [1.0] * 6, "allnull": [np.nan] * 6, "x": [1.0, 2, 3, 4, 5, 60]})
    res = write_eda_report(summarize(df, df.copy()), tmp_path)
    assert res.path.stat().st_size > 0 and "numeric_histograms" in res.charts


def test_regression_target_and_no_task_target(tmp_path, mpl):
    df = pd.DataFrame({"y": np.arange(30.0), "x": np.arange(30.0) ** 2})
    reg = write_eda_report(summarize(df, target="y", task=Task.REGRESSION), tmp_path / "r")
    assert "target_distribution" in reg.charts
    none = write_eda_report(summarize(df, target="y"), tmp_path / "n")
    assert "no task supplied" in none.charts_skipped["target_distribution"]


def test_chart_caps_and_omission_notice(mpl):
    from matplotlib.figure import Figure

    from automl import report

    cols = {f"c{i}": np.where(np.arange(20) < i + 1, np.nan, 1.0) for i in range(30)}
    d = summarize(pd.DataFrame(cols)).to_dict()
    cfg = ReportConfig(max_columns_per_chart=5, max_histograms=4)
    fig = report._chart_missingness(Figure, d, cfg)[0]
    assert any("Showing 5 of 30 columns" in t.get_text() for t in fig.texts)
    assert len(fig.axes[0].patches) == 5
    figs = report._chart_histograms(Figure, d, cfg)
    assert any("Showing 4 of" in t.get_text() for t in figs[0].texts)
    assert sum(len(f.axes) for f in figs) == 4


def test_pdf_text_contains_title_and_findings(eda_frames, mpl):
    from matplotlib.figure import Figure

    from automl import report

    d = _eda(eda_frames).to_dict()
    findings = build_findings(d)
    pages = report._text_pages(Figure, "My Title", report._overview_lines(d, findings))
    pages += report._text_pages(Figure, "Data-quality findings", report._findings_lines(findings))
    text = "\n".join(t.get_text() for p in pages for t in p.texts)
    assert "My Title" in text and "Train rows: 102" in text and "[WARNING] drift (num)" in text
    assert "heuristic" in text


def test_overwrite_rules(eda_frames, tmp_path, mpl):
    eda = _eda(eda_frames)
    write_eda_report(eda, tmp_path)
    before = (tmp_path / "eda_report.pdf").read_bytes()
    with pytest.raises(ReportError, match="overwrite=True"):
        write_eda_report(eda, tmp_path)
    assert (tmp_path / "eda_report.pdf").read_bytes() == before
    res = write_eda_report(eda, tmp_path, overwrite=True, config=ReportConfig(title="Second"))
    assert res.path.exists() and json.loads(res.findings_path.read_text())["config"]["title"] == "Second"


def test_failure_leaves_no_files(eda_frames, tmp_path, monkeypatch, mpl):
    from automl import report

    def boom(Figure, d, cfg):
        raise RuntimeError("boom")

    monkeypatch.setattr(report, "_CHARTS", (("bad", boom),))
    with pytest.raises(RuntimeError):
        write_eda_report(_eda(eda_frames), tmp_path / "out")
    assert list((tmp_path / "out").iterdir()) == []


def test_failure_keeps_existing_report(eda_frames, tmp_path, monkeypatch, mpl):
    from automl import report

    eda = _eda(eda_frames)
    write_eda_report(eda, tmp_path)
    before = (tmp_path / "eda_report.pdf").read_bytes()
    monkeypatch.setattr(report, "_CHARTS", (("bad", lambda *a: 1 / 0),))
    with pytest.raises(ZeroDivisionError):
        write_eda_report(eda, tmp_path, overwrite=True)
    assert (tmp_path / "eda_report.pdf").read_bytes() == before


def test_filename_validation(eda_frames, tmp_path, mpl):
    for bad in ("../x.pdf", "report.txt"):
        with pytest.raises(ReportError, match="'filename'"):
            write_eda_report(_eda(eda_frames), tmp_path, filename=bad)


def test_missing_matplotlib_message(eda_frames, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib.backends.backend_pdf", None)
    with pytest.raises(ReportError, match=r"pip install -e '\.\[reports\]'"):
        write_eda_report(_eda(eda_frames), tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_import_and_findings_without_matplotlib(tmp_path):
    code = (
        "import sys, pandas as pd, automl\n"
        "f = automl.build_findings(automl.summarize(pd.DataFrame({'a': [1.0, 1.0, 1.0]})))\n"
        "assert f and 'matplotlib' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_inputs_not_mutated(eda_frames, tmp_path, mpl):
    train, test = eda_frames
    t0, s0 = train.copy(), test.copy()
    eda = _eda(eda_frames)
    before = json.dumps(eda.to_dict())
    d = eda.to_dict()
    d_before = json.dumps(d)
    write_eda_report(eda, tmp_path / "a")
    write_eda_report(d, tmp_path / "b")
    assert json.dumps(eda.to_dict()) == before and json.dumps(d) == d_before
    pd.testing.assert_frame_equal(train, t0)
    pd.testing.assert_frame_equal(test, s0)


@pytest.mark.parametrize(
    "kwargs,needle",
    [
        ({"high_missing_fraction": 0}, "'high_missing_fraction'"),
        ({"outlier_fraction_warn": 1.5}, "'outlier_fraction_warn'"),
        ({"duplicate_fraction_warn": "x"}, "'duplicate_fraction_warn'"),
        ({"imbalance_ratio_warn": 0.5}, "'imbalance_ratio_warn'"),
        ({"max_columns_per_chart": 0}, "'max_columns_per_chart'"),
        ({"max_histograms": 1.5}, "'max_histograms'"),
        ({"title": " "}, "'title'"),
    ],
)
def test_config_errors(kwargs, needle):
    with pytest.raises(ReportError, match=needle):
        ReportConfig(**kwargs)


def test_report_toml_and_overrides(data_dir):
    assert load_config(data_dir / "run.toml").report_config == ReportConfig()
    base = (data_dir / "run.toml").read_text()
    (data_dir / "rep.toml").write_text(base + '\n[report]\ntitle = "Mine"\nmax_histograms = 3\n')
    assert load_config(data_dir / "rep.toml").report_config == ReportConfig(title="Mine", max_histograms=3)
    assert load_config(data_dir / "rep.toml", max_histograms=5).report_config.max_histograms == 5
    (data_dir / "bad.toml").write_text(base + "\n[report]\nbins = 3\n")
    with pytest.raises(DataContractError, match=r"unknown keys in \[report\]"):
        load_config(data_dir / "bad.toml")
    (data_dir / "bad2.toml").write_text(base + "\n[report]\nmax_histograms = 0\n")
    with pytest.raises(DataContractError, match="'max_histograms'"):
        load_config(data_dir / "bad2.toml")


def test_readme_example(data_dir, tmp_path, mpl):
    cfg = load_config(data_dir / "run.toml", task="binary", metric="roc_auc")
    bundle = load_data(cfg)
    save_eda(summarize_bundle(bundle, task=resolve_task_metric(bundle)), data_dir / "eda.json")
    result = write_eda_report(data_dir / "eda.json", tmp_path / "report", config=cfg.report_config)
    assert result.path.exists() and result.pages == _page_count(result.path)
