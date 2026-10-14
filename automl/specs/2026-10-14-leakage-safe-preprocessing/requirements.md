# Requirements: Leakage-Safe Preprocessing (Phase 7)

## Scope

**Deliverable (from roadmap):** Reusable numeric and categorical preprocessing pipelines with configurable imputation and encoding, compatible with scikit-learn CV.

In scope:

- Optional `[preprocessing]` TOML table in config version 1, with Python overrides.
- Column typing inferred from dtype and cardinality, with explicit overrides.
- Numeric imputation (mean, median, constant) and categorical imputation (most frequent, constant).
- Categorical encoding: one-hot, ordinal, and target encoding, all unknown-category safe.
- Optional numeric scaling (none, standard, robust); default none.
- Guaranteed exclusion of target and ID columns from all processing.
- An unfitted, clonable `ColumnTransformer` usable in scikit-learn `Pipeline` and CV.
- A fold helper that fits only on a `Split`'s train indices.
- A persisted JSON preprocessing-plan artifact with save/load and validation.
- Synthetic-fixture tests (including explicit leakage tests) and README documentation.

Out of scope:

- CLI command (deferred to Phase 16; Python API only).
- Feature engineering, feature selection, binning, polynomial or interaction features.
- Text, datetime, and high-cardinality hashing encoders.
- Model fitting, scoring, and baselines (Phase 8+).
- Persisting fitted preprocessors (they live inside fitted pipelines in later phases).
- Agent skill updates (consolidated in Phase 17; README documents invocation now).

## Decisions

| Topic | Decision |
|---|---|
| Imputers | Numeric: `mean`, `median` (default), `constant`. Categorical: `most_frequent` (default), `constant`. Constant fill values configurable. Missing-indicator columns are not included. |
| Encoders | `onehot` (default; `handle_unknown="ignore"`, dense output), `ordinal` (unknown encoded as -1), `target` (scikit-learn `TargetEncoder`, seeded; internal cross-fitting applies during `fit_transform`). |
| Scaling | `none` (default), `standard`, `robust`. Applied to numeric columns only. |
| Column typing | Infer from dtype and cardinality; overrides via `numeric_columns`, `categorical_columns`, `drop_columns` in `[preprocessing]`. Integer columns are numeric only if unique count exceeds `max_cardinality` (default 20), else categorical. Constant and all-null columns are dropped and reported. |
| ID and target | Always excluded and never transformed or passed through. Naming either in an override is an error rather than silently ignored. |
| Unlisted columns | `remainder="drop"`; nothing reaches the model unless typed. |
| Task awareness | Target encoding is supported for binary, multiclass, and regression; unsupported task or encoder combinations raise an error. |
| Config location | Optional `[preprocessing]` table in config version 1 (no version bump). Python overrides take precedence. Files without the table load unchanged. |
| Leakage guarantee | Learned state (imputation statistics, category sets, scale, target means) comes only from train-fold rows. Preprocessors are returned unfitted and are cloned per fold. |
| Artifact | JSON plan: schema version, effective config, typed columns with reasons, excluded columns. Saving never overwrites an existing file. |
| Interface | Python API and TOML config only. |
| Dependencies | None new. |

## Context

- `specs/mission.md`: leakage-aware (fit learned steps only on the training fold; keep validation data out of target encoding and feature selection); inspectable by default; conservative about inference.
- `specs/tech-stack.md`: `Pipeline`/`ColumnTransformer`, imputers, and encoders from scikit-learn; learned transforms contained in the fitted pipeline used by CV; stage contracts with documented inputs, outputs, assumptions, and limits; actionable errors naming the setting or column.
- Phase 1 provides the config loader, ID and target columns, and the error style. Phase 2 provides `ResolvedTask`. Phase 3 provides `Split` and `SplitResult`. Phase 4 EDA already summarizes cardinality and constant/all-null columns; typing here is computed independently from the train frame so the stages remain decoupled.

## Assumptions

- Input is the train frame from the data contract; test data is only transformed by a preprocessor fit on training rows.
- Positional `Split` indices refer to rows of the train frame.
- Dense output is acceptable at this scale; sparse output is out of scope.
- Inference heuristics are suggestions; users override them in config and the plan artifact records the outcome and reasons.
- Target encoding relies on the installed scikit-learn version providing `TargetEncoder`; the minimum version is declared in package metadata if it is not already.
