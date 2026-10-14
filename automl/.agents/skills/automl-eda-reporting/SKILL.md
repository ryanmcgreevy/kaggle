---
name: automl-eda-reporting
description: "Use when producing or inspecting AutoML EDA summaries, eda.json artifacts, data-quality findings, EDA PDF reports, or the optional Matplotlib reports dependency."
---

# AutoML EDA and Reporting

## Use when

The user wants a data summary, data-quality findings, or the EDA PDF. EDA does not gate modeling; it can run before or alongside `automl-baseline`.

## Procedure

1. Run the `automl-data-intake` workflow first and confirm explicit task/metric choices.
2. Choose a new output directory and run `automl eda run.toml --output-dir runs/experiment-001`.
3. Check the JSON result and validate `runs/experiment-001/eda.json` with `automl.load_eda`. Review its dataset, columns, target, and train/test sections; statistical flags are review signals, not causal conclusions.
4. For a PDF and findings JSON, run `automl report run.toml runs/experiment-001/eda.json --output-dir runs/experiment-001/reports`. Verify both returned paths. If Matplotlib is unavailable, report the `.[reports]` install command; do not make PDF generation a prerequisite for EDA JSON.
5. Never replace existing report files without user direction; use `--overwrite` only after approval. Keep all data and artifacts local.

## Inputs

Run TOML and CSVs for `eda`; the same TOML plus `eda.json` for `report`.

## Outputs

`eda.json` (version 1), then `eda_report.pdf` (or `--filename`) and `<stem>_findings.json`.

## Assumptions and limits

`eda` never overwrites; `report` outputs are also non-overwriting unless `--overwrite`. `report` uses `[report]` settings and needs `pip install -e '.[reports]'`. Outlier flags and correlations are review signals, not causal conclusions.

## Check

Exit code 0, `eda.json` loads with `load_eda`, and the PDF is non-empty. Equivalent API functions are `summarize_bundle`, `save_eda`, `load_eda`, `build_findings`, and `write_eda_report`. See EDA Summary and EDA PDF in `automl/README.md`.