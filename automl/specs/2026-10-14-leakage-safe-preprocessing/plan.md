# Plan: Leakage-Safe Preprocessing (Phase 7)

## 1. Module layout and errors

1. Add `src/automl/preprocessing.py` (config, column typing, pipeline builder, plan artifact I/O).
2. Add `PreprocessingError` to `errors.py` as a `DataContractError` subclass with the same multi-problem message style.
3. Confirm `import automl`, `automl --help`, and existing tests are unaffected. No new dependencies (scikit-learn and pandas are already core).

## 2. Configuration

1. Define a frozen `PreprocessingConfig`:
   - `numeric_imputer`: `"mean"`, `"median"`, or `"constant"` (default `"median"`); `numeric_fill_value` (default 0).
   - `categorical_imputer`: `"most_frequent"` or `"constant"` (default `"most_frequent"`); `categorical_fill_value` (default `"missing"`).
   - `encoder`: `"onehot"`, `"ordinal"`, or `"target"` (default `"onehot"`).
   - `scaler`: `"none"`, `"standard"`, or `"robust"` (default `"none"`).
   - `max_cardinality` for inferring categoricals from integer columns (default 20).
   - `numeric_columns` / `categorical_columns` / `drop_columns` explicit overrides (default empty).
2. Validate in `__post_init__`: known choices, types, no column in more than one override list, collecting all problems in one pass.
3. Extend `load_config` to accept an optional `[preprocessing]` table (unknown keys and wrong types rejected; Python overrides win). Keep config version 1 and existing behavior for files without the table.
4. Expose `preprocessing_config` on the loaded config.

## 3. Column typing

1. Implement `infer_columns(train, target, id_columns=(), config=None) -> ColumnPlan` (frozen: `numeric`, `categorical`, `dropped`, plus a reason per column).
2. Always exclude the target and all ID columns; never place them in any list, even if listed in an override. An override naming one raises `PreprocessingError`.
3. Infer: float and integer columns with more than `max_cardinality` unique values are numeric; object, string, category, bool, and low-cardinality integer columns are categorical; constant and all-null columns are dropped and reported.
4. Apply explicit overrides after inference; unknown override columns raise an error naming them.
5. Use only the train frame; do not mutate it.

## 4. Pipeline construction

1. Implement `build_preprocessor(plan, config, task) -> ColumnTransformer` using `SimpleImputer`, the chosen scaler, `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` or `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)`, and `TargetEncoder` (seeded; supported for binary, multiclass, and regression).
2. Return an unfitted, clonable transformer whose learned state is fit only inside `fit`; set `remainder="drop"` so excluded columns never reach a model.
3. Reject target encoding with a clear error when the resolved task does not support it.
4. Expose `get_feature_names_out` so output columns are inspectable.

## 5. Leakage-safe fitting helper

1. Implement `fit_transform_fold(preprocessor, X, y, split) -> (X_train_t, X_valid_t)` that clones the preprocessor, fits on the train indices only, and transforms the validation indices, consuming a `Split` from Phase 3.
2. Do not use `y` for anything except `TargetEncoder`, and only the train-fold `y`.

## 6. Plan artifact

1. Implement `save_plan(plan, config, path)` writing JSON: schema version, effective config, numeric/categorical/dropped columns with reasons, and excluded ID/target columns. Refuse to overwrite an existing file.
2. Implement `load_plan(path)` with schema/version validation and actionable errors.
3. Document the JSON schema.

## 7. Tests

1. Add a mixed-type synthetic frame with missing values, an ID column, and a target to `tests/conftest.py` (reuse existing fixtures where present).
2. Cover: mixed numeric/categorical/missing handling for every imputer, encoder, and scaler combination; ID and target exclusion including when named in overrides; low-cardinality integer and bool inference; constant and all-null dropping; unknown categories at transform time; `sklearn.base.clone` round trip; use inside `cross_val_score` with a simple estimator; config validation errors and `[preprocessing]` TOML parsing, unknown key, override precedence; target-encoding task restriction; plan artifact round trip, overwrite refusal, and malformed-file rejection; train frame not mutated.
3. Leakage tests: validation-only values and statistics (mean, median, category set, scale, target means) do not change the fitted preprocessor's learned state or train-fold output; fitting on train indices equals fitting on the train subset alone.
4. Assert error messages name the offending setting or column.

## 8. Public API export

1. Export `PreprocessingConfig`, `ColumnPlan`, `infer_columns`, `build_preprocessor`, `fit_transform_fold`, `save_plan`, `load_plan`, and `PreprocessingError` from `automl`.

## 9. Documentation and roadmap

1. Update `automl/README.md` with the `[preprocessing]` schema, defaults, column-typing rules, ID/target exclusion, leakage guarantees, the plan JSON schema, and a Python example for people and agents (no CLI this phase).
2. Mark Phase 7 complete in `specs/roadmap.md` with a verification note once validation passes.
