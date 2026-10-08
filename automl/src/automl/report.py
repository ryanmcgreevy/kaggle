"""EDA findings and PDF report generated from a structured EDA result."""

from __future__ import annotations

import json
import math
import os
import tempfile
import textwrap
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from numbers import Real
from pathlib import Path
from typing import Any

from automl.eda import SCHEMA_VERSION, EdaResult, load_eda
from automl.errors import ReportError

REPORT_SCHEMA_VERSION = 1
_SEVERITY_ORDER = {"warning": 0, "info": 1}
_SCOPE_ORDER = {"dataset": 0, "target": 1, "column": 2, "train_test": 3}
_LINES_PER_PAGE = 52
_PANELS_PER_PAGE = 6
_HEURISTIC_NOTE = "Findings are heuristic flags for review; they are not causal conclusions or removal advice."


@dataclass(frozen=True)
class ReportConfig:
    high_missing_fraction: float = 0.2
    outlier_fraction_warn: float = 0.05
    imbalance_ratio_warn: float = 10.0
    duplicate_fraction_warn: float = 0.01
    max_columns_per_chart: int = 20
    max_histograms: int = 12
    title: str = "EDA Report"

    def __post_init__(self) -> None:
        problems: list[str] = []
        for name in ("high_missing_fraction", "outlier_fraction_warn", "duplicate_fraction_warn"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, Real):
                problems.append(f"setting '{name}' must be a number")
            elif not 0 < v <= 1:
                problems.append(f"setting '{name}' must be in (0, 1], got {v}")
        v = self.imbalance_ratio_warn
        if isinstance(v, bool) or not isinstance(v, Real):
            problems.append("setting 'imbalance_ratio_warn' must be a number")
        elif v < 1:
            problems.append(f"setting 'imbalance_ratio_warn' must be at least 1, got {v}")
        for name in ("max_columns_per_chart", "max_histograms"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int):
                problems.append(f"setting '{name}' must be an integer")
            elif v < 1:
                problems.append(f"setting '{name}' must be at least 1, got {v}")
        if not isinstance(self.title, str) or not self.title.strip():
            problems.append("setting 'title' must be a non-empty string")
        if problems:
            raise ReportError(problems)


REPORT_KEYS = {f.name for f in fields(ReportConfig)}


@dataclass(frozen=True)
class Finding:
    code: str
    severity: str
    scope: str
    message: str
    column: str | None = None
    value: float | None = None
    threshold: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReportResult:
    path: Path
    findings_path: Path
    pages: int
    charts: tuple[str, ...]
    charts_skipped: dict[str, str]
    findings: tuple[Finding, ...]
    schema_version: int = field(default=REPORT_SCHEMA_VERSION)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "path": str(self.path),
            "findings_path": str(self.findings_path),
            "pages": self.pages,
            "charts": list(self.charts),
            "charts_skipped": dict(self.charts_skipped),
            "findings": [f.to_dict() for f in self.findings],
        }


def _as_dict(eda: EdaResult | Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(eda, EdaResult):
        return eda.to_dict()
    if isinstance(eda, (str, Path)):
        return load_eda(eda)
    if isinstance(eda, Mapping):
        missing = [k for k in ("config", "dataset", "columns", "target", "train_test") if k not in eda]
        if eda.get("schema_version") != SCHEMA_VERSION or missing:
            raise ReportError(
                f"EDA input must be an EDA result with schema_version {SCHEMA_VERSION}; missing sections: {missing}"
            )
        return dict(eda)
    raise ReportError(f"EDA input must be an EdaResult, a saved EDA path, or a dict, got {type(eda).__name__}")


def build_findings(eda: EdaResult | Mapping[str, Any] | str | Path, config: ReportConfig | None = None) -> tuple[Finding, ...]:
    """Derive data-quality findings from stored EDA values only."""
    cfg = config or ReportConfig()
    d = _as_dict(eda)
    out: list[Finding] = []

    def add(code: str, severity: str, scope: str, message: str, column=None, value=None, threshold=None) -> None:
        out.append(Finding(code, severity, scope, message, column, value, threshold))

    ds = d["dataset"]
    frac = ds.get("duplicate_row_fraction")
    if frac is not None and ds.get("duplicate_rows", 0) > 0 and frac >= cfg.duplicate_fraction_warn:
        add("duplicate_rows", "warning", "dataset", f"{ds['duplicate_rows']} duplicate rows ({frac:.1%} of train)",
            value=frac, threshold=cfg.duplicate_fraction_warn)
    if ds.get("duplicate_ids"):
        add("duplicate_ids", "warning", "dataset", f"{ds['duplicate_ids']} rows repeat an ID in train",
            value=float(ds["duplicate_ids"]))

    tgt = d.get("target")
    if tgt:
        ratio = tgt.get("imbalance_ratio")
        if ratio is not None and ratio >= cfg.imbalance_ratio_warn:
            add("class_imbalance", "warning", "target",
                f"largest class is {ratio:.1f}x the smallest; consider stratified validation and a suitable metric",
                column=tgt["name"], value=ratio, threshold=cfg.imbalance_ratio_warn)

    for col in d["columns"]:
        name = col["name"]
        nf = col.get("null_fraction")
        if col.get("all_null"):
            add("all_null", "warning", "column", "column has no non-null values", name)
        elif col.get("constant"):
            add("constant", "warning", "column", "column has a single value", name)
        if nf is not None and not col.get("all_null") and nf >= cfg.high_missing_fraction:
            add("high_missing", "warning", "column", f"{nf:.1%} of values are missing", name, nf, cfg.high_missing_fraction)
        if col.get("id_like"):
            add("id_like", "info", "column", "nearly every value is unique; may be an identifier", name)
        num = col.get("numeric")
        if num:
            fracs = [o["fraction"] for o in num.get("outliers", {}).values() if o.get("fraction") is not None]
            if fracs and max(fracs) >= cfg.outlier_fraction_warn:
                add("high_outliers", "info", "column", f"{max(fracs):.1%} of values are statistical outliers",
                    name, max(fracs), cfg.outlier_fraction_warn)

    tt = d.get("train_test")
    if tt:
        for side, cols in (("train", tt.get("only_in_train", [])), ("test", tt.get("only_in_test", []))):
            if cols:
                add("one_sided_columns", "warning", "train_test",
                    f"columns present only in {side}: {', '.join(cols)}")
        for c in tt.get("columns", []):
            name = c["name"]
            if c.get("drift"):
                add("drift", "warning", "train_test",
                    f"train/test distributions differ ({c['test_name']} p={c['pvalue']:.3g}); heuristic, sensitive to large samples",
                    name, c["pvalue"])
            if c.get("test_only_categories"):
                add("unseen_test_categories", "info", "train_test",
                    f"{c['test_only_categories']} categories appear only in test", name, float(c["test_only_categories"]))
            delta = c.get("null_delta")
            if delta is not None and abs(delta) >= cfg.high_missing_fraction:
                add("null_delta", "warning", "train_test",
                    f"missing fraction differs by {delta:+.1%} between train and test", name, delta, cfg.high_missing_fraction)

    out.sort(key=lambda f: (_SEVERITY_ORDER[f.severity], _SCOPE_ORDER[f.scope], f.column or "", f.code))
    return tuple(out)


class _Skip(Exception):
    pass


def _require_matplotlib():
    try:
        from matplotlib.backends.backend_pdf import PdfPages
        from matplotlib.figure import Figure
    except ImportError:
        raise ReportError(
            "PDF reports require matplotlib; install it with: pip install -e '.[reports]'"
        ) from None
    return Figure, PdfPages


def _cap(items: list, limit: int, what: str) -> tuple[list, str | None]:
    if len(items) <= limit:
        return items, None
    return items[:limit], f"Showing {limit} of {len(items)} {what}; others omitted."


def _note(fig, text: str | None) -> None:
    if text:
        fig.text(0.5, 0.01, text, ha="center", fontsize=8, style="italic")


def _bar_figure(Figure, title: str, labels: list[str], values: list[float], xlabel: str, note: str | None = None,
                vline: tuple[float, str] | None = None):
    fig = Figure(figsize=(8.5, max(3.0, 0.35 * len(labels) + 1.8)))
    ax = fig.add_subplot(111)
    ax.barh(range(len(labels)), values)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    if vline:
        ax.axvline(vline[0], color="red", linestyle="--", label=vline[1])
        ax.legend()
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _note(fig, note)
    return fig


def _chart_missingness(Figure, d, cfg):
    rows = [(c["name"], c["null_fraction"]) for c in d["columns"] if c.get("null_fraction")]
    if not rows:
        raise _Skip("no missing values in feature columns")
    rows.sort(key=lambda r: (-r[1], r[0]))
    rows, note = _cap(rows, cfg.max_columns_per_chart, "columns")
    return [_bar_figure(Figure, "Missing values by column", [r[0] for r in rows], [r[1] for r in rows],
                        "fraction missing", note)]


def _chart_outliers(Figure, d, cfg):
    rows = []
    for c in d["columns"]:
        num = c.get("numeric")
        fracs = [o["fraction"] for o in (num or {}).get("outliers", {}).values() if o.get("fraction") is not None]
        if fracs and max(fracs) > 0:
            rows.append((c["name"], max(fracs)))
    if not rows:
        raise _Skip("no numeric columns with outliers")
    rows.sort(key=lambda r: (-r[1], r[0]))
    rows, note = _cap(rows, cfg.max_columns_per_chart, "columns")
    return [_bar_figure(Figure, "Outlier fraction by numeric column", [r[0] for r in rows], [r[1] for r in rows],
                        "fraction of values flagged", note)]


def _hist_axes(ax, hist, title):
    edges, counts = hist["edges"], hist["counts"]
    widths = [b - a for a, b in zip(edges[:-1], edges[1:])]
    ax.bar(edges[:-1], counts, width=widths, align="edge")
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)


def _panels(Figure, title, items, draw, note):
    figs = []
    for start in range(0, len(items), _PANELS_PER_PAGE):
        fig = Figure(figsize=(8.5, 11))
        fig.suptitle(title)
        chunk = items[start : start + _PANELS_PER_PAGE]
        for i, item in enumerate(chunk):
            draw(fig.add_subplot(3, 2, i + 1), item)
        fig.tight_layout(rect=(0, 0.04, 1, 0.96))
        _note(fig, note)
        figs.append(fig)
    return figs


def _chart_histograms(Figure, d, cfg):
    cols = [c for c in d["columns"] if (c.get("numeric") or {}).get("histogram")]
    if not cols:
        raise _Skip("no numeric columns with histograms")

    def severity(c):
        fr = [o["fraction"] for o in c["numeric"]["outliers"].values() if o.get("fraction") is not None]
        return -(max(fr) if fr else 0.0), c["name"]

    cols.sort(key=severity)
    cols, note = _cap(cols, cfg.max_histograms, "numeric columns")
    return _panels(Figure, "Numeric distributions", cols,
                   lambda ax, c: _hist_axes(ax, c["numeric"]["histogram"], c["name"]), note)


def _chart_categorical(Figure, d, cfg):
    cols = [c for c in d["columns"] if (c.get("categorical") or {}).get("top_values")]
    if not cols:
        raise _Skip("no categorical or boolean columns")
    cols.sort(key=lambda c: (-c["unique"], c["name"]))
    cols, note = _cap(cols, cfg.max_histograms, "categorical columns")

    def draw(ax, c):
        top = c["categorical"]["top_values"]
        ax.barh(range(len(top)), [t["fraction"] for t in top])
        ax.set_yticks(range(len(top)), [str(t["value"])[:18] for t in top], fontsize=7)
        ax.invert_yaxis()
        ax.set_title(f"{c['name']} ({c['unique']} unique)", fontsize=9)
        ax.tick_params(labelsize=7)

    return _panels(Figure, "Categorical top values (fraction of non-null)", cols, draw, note)


def _chart_target(Figure, d, cfg):
    t = d.get("target")
    if not t:
        raise _Skip("no target in EDA result")
    if t.get("class_counts"):
        items = sorted(t["class_counts"].items(), key=lambda kv: (-kv[1], kv[0]))
        items, note = _cap(items, cfg.max_columns_per_chart, "classes")
        return [_bar_figure(Figure, f"Target distribution: {t['name']}", [str(k) for k, _ in items],
                            [v for _, v in items], "rows", note)]
    hist = (t.get("numeric") or {}).get("histogram")
    if hist:
        fig = Figure(figsize=(8.5, 4))
        ax = fig.add_subplot(111)
        _hist_axes(ax, hist, f"Target distribution: {t['name']}")
        fig.tight_layout()
        return [fig]
    raise _Skip("target has no distribution (no task supplied or no numeric values)")


def _chart_drift(Figure, d, cfg):
    tt = d.get("train_test")
    if not tt:
        raise _Skip("no test data in EDA result")
    rows = [(c["name"], -math.log10(max(c["pvalue"], 1e-300))) for c in tt["columns"] if c.get("pvalue") is not None]
    if not rows:
        raise _Skip("no train/test comparisons with a p-value")
    rows.sort(key=lambda r: (-r[1], r[0]))
    rows, note = _cap(rows, cfg.max_columns_per_chart, "columns")
    alpha = d["config"]["drift_alpha"]
    return [_bar_figure(Figure, "Train/test drift (higher = stronger evidence of difference)",
                        [r[0] for r in rows], [r[1] for r in rows], "-log10(p-value)", note,
                        vline=(-math.log10(alpha), f"alpha = {alpha}"))]


_CHARTS = (
    ("missingness", _chart_missingness),
    ("numeric_histograms", _chart_histograms),
    ("target_distribution", _chart_target),
    ("categorical_top_values", _chart_categorical),
    ("train_test_drift", _chart_drift),
    ("outlier_fractions", _chart_outliers),
)


def _text_pages(Figure, title: str, lines: list[str]):
    figs = []
    chunks = [lines[i : i + _LINES_PER_PAGE] for i in range(0, len(lines), _LINES_PER_PAGE)] or [[]]
    for chunk in chunks:
        fig = Figure(figsize=(8.5, 11))
        fig.text(0.07, 0.95, title, fontsize=14, weight="bold", va="top")
        fig.text(0.07, 0.91, "\n".join(chunk), fontsize=8, family="monospace", va="top")
        figs.append(fig)
    return figs


def _overview_lines(d: dict[str, Any], findings: tuple[Finding, ...]) -> list[str]:
    ds = d["dataset"]
    lines = [
        f"Train rows: {ds['n_rows']}    columns: {ds['n_columns']}    memory: {ds['memory_bytes'] / 1e6:.2f} MB",
        f"Duplicate rows: {ds['duplicate_rows']}",
    ]
    if ds.get("duplicate_ids") is not None:
        lines.append(f"Duplicate IDs: {ds['duplicate_ids']}")
    if d.get("train_test"):
        lines.append(f"Test rows: {d['train_test']['test_rows']}")
    t = d.get("target")
    if t:
        lines.append(f"Target: {t['name']} ({t['dtype']}), task: {t.get('task') or 'not supplied'}")
    kinds: dict[str, int] = {}
    for c in d["columns"]:
        kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
    lines.append("Feature columns by kind: " + (", ".join(f"{k}={v}" for k, v in sorted(kinds.items())) or "none"))
    warnings = sum(f.severity == "warning" for f in findings)
    lines += ["", f"Findings: {len(findings)} ({warnings} warnings, {len(findings) - warnings} info)", ""]
    lines += textwrap.wrap(_HEURISTIC_NOTE, 95)
    return lines


def _findings_lines(findings: tuple[Finding, ...]) -> list[str]:
    if not findings:
        return ["No findings."]
    lines: list[str] = []
    for f in findings:
        head = f"[{f.severity.upper()}] {f.code}" + (f" ({f.column})" if f.column else "")
        lines.append(head)
        lines += textwrap.wrap(f.message, 90, initial_indent="    ", subsequent_indent="    ")
    return lines


def write_eda_report(
    eda: EdaResult | Mapping[str, Any] | str | Path,
    output_dir: str | Path,
    *,
    filename: str = "eda_report.pdf",
    config: ReportConfig | None = None,
    overwrite: bool = False,
) -> ReportResult:
    """Write the EDA PDF and findings JSON into `output_dir` from a stored EDA result."""
    cfg = config or ReportConfig()
    Figure, PdfPages = _require_matplotlib()
    d = _as_dict(eda)
    if Path(filename).name != filename or not filename.lower().endswith(".pdf"):
        raise ReportError(f"setting 'filename' must be a plain file name ending in .pdf, got {filename!r}")
    out_dir = Path(output_dir)
    pdf_path = out_dir / filename
    findings_path = out_dir / (Path(filename).stem + "_findings.json")
    if not overwrite:
        existing = [str(p) for p in (pdf_path, findings_path) if p.exists()]
        if existing:
            raise ReportError(f"report output already exists: {existing}; pass overwrite=True to replace")
    out_dir.mkdir(parents=True, exist_ok=True)

    findings = build_findings(d, cfg)
    included: list[str] = []
    skipped: dict[str, str] = {}
    tmp_pdf = tmp_json = None
    try:
        fd, tmp_pdf = tempfile.mkstemp(suffix=".pdf", dir=out_dir)
        os.close(fd)
        with PdfPages(tmp_pdf) as pdf:
            for fig in _text_pages(Figure, cfg.title, _overview_lines(d, findings)):
                pdf.savefig(fig)
            for fig in _text_pages(Figure, "Data-quality findings", _findings_lines(findings)):
                pdf.savefig(fig)
            for name, fn in _CHARTS:
                try:
                    figs = fn(Figure, d, cfg)
                except _Skip as skip:
                    skipped[name] = str(skip)
                    continue
                included.append(name)
                for fig in figs:
                    pdf.savefig(fig)
            pages = pdf.get_pagecount()
        result = ReportResult(pdf_path, findings_path, pages, tuple(included), skipped, findings)
        payload = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "config": asdict(cfg),
            "findings": [f.to_dict() for f in findings],
            "charts": included,
            "charts_skipped": skipped,
        }
        fd, tmp_json = tempfile.mkstemp(suffix=".json", dir=out_dir)
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(payload, allow_nan=False))
        os.replace(tmp_pdf, pdf_path)
        tmp_pdf = None
        os.replace(tmp_json, findings_path)
        tmp_json = None
    finally:
        for tmp in (tmp_pdf, tmp_json):
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)
    return result
