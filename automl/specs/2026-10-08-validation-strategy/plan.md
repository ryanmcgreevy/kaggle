# Plan: Validation Strategy (Phase 3)

## 1. Module layout and errors

1. Add `src/automl/validation.py` (config, split construction, result, artifact I/O).
2. Add `ValidationSplitError` to `errors.py` as a `DataContractError` subclass with the same multi-problem message style.
3. Confirm `import automl`, `automl --help`, and existing tests are unaffected. No new dependencies (scikit-learn is already core).

## 2. Configuration

1. Define a frozen `ValidationConfig`: `strategy` (`"holdout"` or `"kfold"`, default `"kfold"`), `seed` (default 42), `n_folds` (default 5), `holdout_fraction` (default 0.2).
2. Validate in `__post_init__`: known strategy, int `seed`, `n_folds >= 2`, `0 < holdout_fraction < 1`, correct types; collect all problems in one pass.
3. Extend `load_config` to accept an optional `[validation]` table (unknown keys and wrong types rejected; Python overrides win). Keep config version 1 and existing behavior for files without the table.
4. Expose `validation_config` on the loaded config.

## 3. Split construction

1. Define `Split` (frozen): `fold`, `train_idx`, `valid_idx` as sorted integer NumPy arrays of positional row indices.
2. Define `SplitResult` (frozen): `config`, `task`, `stratified`, `n_rows`, `splits` (one for holdout, `n_folds` for kfold).
3. Implement `make_splits(target, resolved_task, config) -> SplitResult` using `StratifiedShuffleSplit`/`StratifiedKFold` for `binary`/`multiclass` and `ShuffleSplit`/`KFold` (shuffled) for `regression`; seed via `random_state`. No binned-target stratification.
4. Use only the train target; do not mutate it.

## 4. Invalid-input handling

1. Before splitting, check: target length, null targets, `n_folds <= n_rows`, holdout produces at least one row on each side.
2. For classification, check every class has at least `n_folds` members (kfold) or at least 2 members (holdout, so each side can receive one); also check the holdout fraction yields at least one validation row per class.
3. Raise `ValidationSplitError` naming the setting, class, and counts with a suggested fix (fewer folds, larger fraction, or merging/removing rare classes). Never fall back to non-stratified splitting.

## 5. Artifact

1. Implement `save_splits(result, path)` writing JSON: schema version, strategy, seed, `n_folds`/`holdout_fraction`, task, `stratified`, `n_rows`, per-fold sizes, and index lists. Refuse to overwrite an existing file.
2. Implement `load_splits(path) -> SplitResult` with schema/version validation and actionable errors for malformed or inconsistent files (indices out of range, overlap, non-integer).
3. Document the JSON schema.

## 6. Tests

1. Add synthetic binary, multiclass, and regression targets to `tests/conftest.py` (reuse existing ones where present).
2. Cover: repeatability for a fixed seed and difference for another seed; no train/validation overlap per fold; k-fold validation sets partition all rows exactly once; stratification preserves class proportions within tolerance; regression is non-stratified; holdout fraction respected; config validation errors (strategy, folds, fraction, types); `[validation]` TOML parsing, unknown key, override precedence; class with too few members for kfold and holdout raises with the class name and counts, with no silent fallback; null target and too-few-rows errors; target not mutated; artifact round-trip equality, overwrite refusal, and malformed-file rejection.
3. Assert error messages name the offending setting, class, or count.

## 7. Public API export

1. Export `ValidationConfig`, `Split`, `SplitResult`, `make_splits`, `save_splits`, `load_splits`, and `ValidationSplitError` from `automl`.

## 8. Documentation and roadmap

1. Update `automl/README.md` with the `[validation]` schema, defaults, stratification rules, error behavior, the JSON artifact schema, and a Python example for people and agents (no CLI this phase).
2. Mark Phase 3 complete in `specs/roadmap.md` with a verification note once validation passes.
