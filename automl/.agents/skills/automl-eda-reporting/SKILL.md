---
name: automl-eda-reporting
description: "Use when producing or inspecting AutoML EDA summaries, eda.json artifacts, data-quality findings, EDA PDF reports, or the optional Matplotlib reports dependency."
---

# AutoML EDA and Reporting

## Procedure

1. Run the `automl-data-intake` workflow first and confirm explicit task/metric choices.
2. Choose a new output directory and run `automl eda run.toml --output-dir runs/experiment-001`.
3. Check the JSON result and validate `runs/experiment-001/eda.json` with `automl.load_eda`. Review its dataset, columns, target, and train/test sections; statistical flags are review signals, not causal conclusions.
4. For a PDF and findings JSON, run `automl report run.toml runs/experiment-001/eda.json --output-dir runs/experiment-001/reports`. Verify both returned paths. If Matplotlib is unavailable, report the `.[reports]` install command; do not make PDF generation a prerequisite for EDA JSON.
5. Never replace existing report files without user direction; use `--overwrite` only after approval. Keep all data and artifacts local.

## Contract

`eda` writes the existing version-1 EDA JSON without overwriting. `report` uses `[report]` settings from the TOML and writes a PDF plus `<stem>_findings.json`; report outputs are also non-overwriting by default. Equivalent API functions are `summarize_bundle`, `save_eda`, `load_eda`, `build_findings`, and `write_eda_report`. See EDA Summary and EDA PDF in `automl/README.md` for schemas, optional dependencies, and limits.