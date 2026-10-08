# Requirements: Validation Strategy (Phase 3)

## Scope

**Deliverable (from roadmap):** Reproducible holdout and cross-validation split construction appropriate to the selected task, with configurable seed and fold count.

In scope:

- Two strategies: `holdout` and `kfold`.
- Optional `[validation]` TOML table (`strategy`, `seed`, `n_folds`, `holdout_fraction`) in config version 1, with Python overrides.
- `make_splits` producing positional index splits from the train target and the resolved task from Phase 2.
- Stratification for `binary` and `multiclass`; plain shuffled splitting for `regression`.
- A persisted JSON split artifact with save/load and validation.
- Actionable errors for invalid fold/class/size combinations.
- Synthetic-fixture tests and README documentation.

Out of scope:

- CLI command (deferred to Phase 15; Python API only).
- Repeated k-fold, group k-fold, time-based splits, and binned-target stratification for regression.
- Fitting models, preprocessing, or scoring (Phases 6+).
- Splitting the test data.

## Decisions

| Topic | Decision |
|---|---|
| Strategies | `holdout` and `kfold` only. |
| Defaults | `strategy="kfold"`, `seed=42`, `n_folds=5`, `holdout_fraction=0.2`. Defaults apply only to these split parameters; they are recorded in the artifact. |
| Classification | Stratified (`StratifiedKFold` / `StratifiedShuffleSplit`), shuffled and seeded. |
| Regression | Plain shuffled `KFold` / `ShuffleSplit`; no binned-target stratification. |
| Invalid class/fold combinations | Raise `ValidationSplitError` naming the class, its count, and the requested folds or fraction, with a suggested fix. No silent fallback to non-stratified splitting. |
| Config location | Optional `[validation]` table in config version 1 (no version bump). Python overrides take precedence. Files without the table load unchanged. |
| Result shape | Frozen `SplitResult` holding config, task, `stratified`, `n_rows`, and `Split` objects with sorted positional index arrays. |
| Artifact | JSON with schema version, strategy, seed, parameters, task, `stratified`, `n_rows`, per-fold sizes, and full index lists. Saving never overwrites an existing file. Loading validates structure and index consistency. |
| Interface | Python API only. |
| Dependencies | None new; uses scikit-learn and NumPy already in core. |

## Context

- `specs/mission.md`: reproducible with explicit seeds and persisted effective configuration; leakage-aware; conservative about inference; resource-aware.
- `specs/tech-stack.md`: stratified splitters for classification and non-stratified for regression unless the data contract specifies otherwise; seed from a shared run configuration; machine-readable artifacts; do not overwrite prior outputs implicitly; actionable errors naming the setting.
- Phase 1 provides `load_config` and the config/error style; Phase 2 provides `ResolvedTask` and `Task`. This phase consumes both and adds nothing to the data loader.

## Assumptions

- Indices are positional row indices into the train frame (`DataBundle.train`), not index labels.
- The split depends only on the target, task, and config, so it can be saved once and reused by later stages (baseline, tuning) without rerunning.
- Seeded splits are repeatable for a fixed scikit-learn version; the artifact is the durable record for exact reproduction.
- Holdout validation means a single split with `holdout_fraction` of rows (rounded by scikit-learn) in the validation set.
