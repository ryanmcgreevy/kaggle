# Requirements: Boosting Adapters (Phase 9)

## Scope

**Deliverable (from roadmap):** Add supported boosting estimators as optional, individually selectable adapters, beginning with the libraries chosen for the first release.

In scope:

- Adapters for LightGBM, XGBoost, and CatBoost, selectable one at a time through an `estimator` setting alongside the existing HistGradientBoosting baseline.
- A small registry describing each estimator: name, supported tasks, required package and install extra, native categorical and GPU support, defaults, and factory.
- Reuse of the Phase 8 run path, result, and artifact with the estimator generalized.
- Optional dependency groups: one extra per library plus a `boosting` bundle.
- Native categorical handling for adapters that support it, using a native-mode preprocessing path.
- Opt-in GPU use (`device = "auto"` or `"gpu"`) only when a GPU is detected and the installed library supports it; CPU by default.
- Clear, actionable errors when a library is missing or a device is unavailable.
- Synthetic-fixture tests and README documentation.

Out of scope:

- CLI command (deferred to Phase 16; Python API and TOML only).
- Hyperparameter search spaces and tuning (Phases 10 and 11); running several estimators in one call or ensembling.
- Third-party estimator registration (`register_estimator`).
- Early stopping, sample weights, custom objectives, and monotonic or interaction constraints beyond what `estimator_params` passes through.
- Multi-GPU, distributed, or cloud execution.
- Reports and MLflow tracking (Phases 12 and 13).
- Agent skill updates (consolidated in Phase 17; README documents invocation now).

## Decisions

| Topic | Decision |
|---|---|
| Libraries | LightGBM, XGBoost, CatBoost, each an optional extra. HistGradientBoosting stays the core default. |
| Selection | `estimator` key in `[baseline]` (default `"hist_gradient_boosting"`); one estimator per run. Unknown names raise `BaselineError` listing valid names. |
| Run path | `run_baseline` is generalized, not duplicated. Splitting, per-fold fitting, metric prediction inputs, out-of-fold handling, and artifact layout are unchanged. |
| Extras | `lightgbm`, `xgboost`, `catboost`, and `boosting` (all three). Core dependencies unchanged. Minimum versions set from compatibility testing and documented. |
| Missing library | `import automl` never imports an adapter library. Selecting an unavailable estimator raises `EstimatorUnavailableError` before any fitting, naming the package and the exact install command. The default path keeps working. |
| Defaults | Seeded from `seed`, CPU, threads from `n_jobs`, early stopping off, logging silenced, no files written by the library (CatBoost `allow_writing_files=False`). |
| Controlled parameters | `estimator_params` may not set seed, thread count, device, verbosity, or early stopping; the error names the controlling setting. Other parameters are validated against the estimator. |
| Labels | Every adapter works with string and integer targets and exposes `classes_` consistent with `predict_proba`, using a thin label-encoding wrapper only where the library needs it. |
| Native categoricals | `native_categoricals` defaults to the adapter default: true for LightGBM, XGBoost, and CatBoost; false for HistGradientBoosting to preserve Phase 8 results. Native mode uses an ordinal block with missing and unseen categories kept as missing, fit on train rows only, and ignores the preprocessing `encoder` setting. The result records the effective handling. |
| Column layout | Transformed columns are the numeric block followed by the categorical block, so categorical indices are derived from the plan, not guessed. |
| Device | `device` is `"cpu"` (default), `"auto"`, or `"gpu"`. `"auto"` uses the GPU only if the estimator supports it, the library has GPU support, and a tiny probe fit succeeds, otherwise falls back to CPU and records why. `"gpu"` raises if unavailable. HistGradientBoosting is CPU only. |
| Reproducibility | CPU runs repeat for fixed seed, splits, and library versions. GPU runs may not be bit-reproducible; this is recorded in the result and README. |
| Artifact | Schema version 2 adds `estimator_key`, `categorical_handling`, `device` (requested, effective, reason), and the chosen library's version. Version 1 artifacts still load with defaults for the new fields. |
| Interface | Python API and TOML config only. |

## Context

- `specs/mission.md`: CPU-capable and local by default; GPUs and other hardware are optional extensions; resource-aware; reproducible with explicit seeds and reported nondeterminism; inspectable.
- `specs/tech-stack.md`: optional adapters for LightGBM, CatBoost, and XGBoost where dependencies and task compatibility are available; missing packages produce a clear installation message and do not break core imports; named optional groups with documented install commands and tests that core still works without them; scikit-learn-compatible interfaces.
- Phase 8 provides `run_baseline`, `BaselineConfig`, `BaselineResult`, and the result directory. Phase 7 provides the preprocessing plan and `build_preprocessor`. Phases 10 and 11 will reuse the registry and defaults to build search spaces.

## Assumptions

- Boosting libraries are not installed in the project `.venv` today; verification installs the extras there, and the core suite is also run without them.
- Library versions supporting Python 3.12 and the native categorical APIs are available; the minimum versions in the extras reflect what was tested.
- The environment may have no GPU. GPU code paths are covered by mocked probes and skipped real runs unless a GPU and a GPU-capable build are present.
- Native categorical support differs per library; differences in scores between libraries are expected and are not treated as errors.
- Dense preprocessed output is acceptable at the data sizes this toolkit targets.
