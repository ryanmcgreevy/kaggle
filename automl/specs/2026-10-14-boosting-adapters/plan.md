# Plan: Boosting Adapters (Phase 9)

## 1. Module layout, errors, and optional dependencies

1. Add `src/automl/estimators.py` (estimator registry, specs, adapters, device selection). Keep `baseline.py` as the run path.
2. Add `EstimatorUnavailableError` to `errors.py` as a `BaselineError` subclass for a missing library or unsupported device; its message names the estimator, the missing package, and the exact install command (`pip install 'automl[lightgbm]'` or similar).
3. Add optional extras to `pyproject.toml`: `lightgbm`, `xgboost`, `catboost`, and a `boosting` bundle of all three, with minimum versions chosen from compatibility testing. Core `dependencies` are unchanged.
4. Confirm `import automl`, `automl --help`, and existing tests are unaffected when none of the extras are installed.

## 2. Estimator registry

1. Define a frozen `EstimatorSpec`: `name`, `tasks`, `package` (import name) and `extra` (install extra), `supports_native_categoricals`, `supports_gpu`, `default_params(task)`, and a `make(task, params, seed, n_jobs, device, categorical_idx)` factory.
2. Register four estimators: `hist_gradient_boosting` (existing, scikit-learn, no extra), `lightgbm`, `xgboost`, `catboost`. All support `binary`, `multiclass`, and `regression`.
3. Implement `list_estimators()`, `get_estimator_spec(name)` (unknown name raises `BaselineError` listing valid names), and `estimator_available(name)` (import check only, no model construction).
4. Import adapter libraries lazily inside the factory and availability check, never at `import automl`.

## 3. Adapter defaults and behavior

1. Every adapter is seeded from `BaselineConfig.seed`, runs on CPU by default, takes its thread count from `n_jobs`, turns early stopping off, and silences library logging so the CLI and tests stay quiet.
2. Per library: LightGBM (`LGBMClassifier`/`LGBMRegressor`), XGBoost (`XGBClassifier`/`XGBRegressor`, `tree_method="hist"`), CatBoost (`CatBoostClassifier`/`CatBoostRegressor`, `allow_writing_files=False`, no `catboost_info` directory).
3. Classification adapters must expose `classes_`, `predict`, and `predict_proba` consistent with the other estimators. Add a thin label-encoding wrapper only where a library requires numeric labels (XGBoost) or cannot take string labels, so string targets work for every adapter.
4. Validate `estimator_params` against the chosen estimator's accepted parameters, and reject keys the adapter controls (`random_state`/`seed`, thread count, device, verbosity, early stopping) with a message naming the setting that controls them.

## 4. Native categorical handling

1. Add `native_categoricals` to `BaselineConfig` (default `None` meaning the adapter default: true for `lightgbm`, `xgboost`, and `catboost`; false for `hist_gradient_boosting`, preserving Phase 8 behavior). Reject `true` for an estimator whose spec does not support it.
2. Extend `build_preprocessor` with a `native_categoricals` option. In native mode the categorical block is an ordinal encoding that keeps missing values and unseen categories as missing (no imputation, no one-hot or target encoding), fit on the train fold only; numeric columns are unchanged. The categorical block follows the numeric block, so categorical indices are `range(n_numeric, n_numeric + n_categorical)`.
3. Pass categorical indices to each adapter in its own way: LightGBM `categorical_feature`, XGBoost `feature_types` with `enable_categorical=True`, CatBoost `cat_features` (with the wrapper converting codes to the integer or string form it requires), and HistGradientBoosting `categorical_features`.
4. The preprocessing `encoder` setting does not apply in native mode; record `categorical_handling` (`"native"` or `"encoded"`) and the effective encoder in the result.

## 5. Device selection

1. Add `device` to `BaselineConfig`: `"cpu"` (default), `"auto"`, or `"gpu"`.
2. Implement `resolve_device(spec, requested)` returning the effective device and a reason string. `"auto"` uses the GPU only when the estimator supports it, the installed library was built with GPU support, and a tiny probe fit succeeds on it; otherwise it falls back to CPU and records why. `"gpu"` raises `EstimatorUnavailableError` with the reason if any of those fail. `hist_gradient_boosting` is CPU only.
3. Map the effective device to each library's parameters (`device="cuda"`/`tree_method="hist"` for XGBoost, `device_type="gpu"` for LightGBM, `task_type="GPU"` for CatBoost). Cache the probe result per estimator within the process.
4. Record the requested and effective device, the reason, and a note that GPU training is not guaranteed bit-reproducible.

## 6. Generalize the run path

1. Add `estimator` (default `"hist_gradient_boosting"`) to `BaselineConfig` and a `[baseline]` key; validate against the registry. Missing libraries raise `EstimatorUnavailableError` at the start of `run_baseline`, before any fitting.
2. Build the estimator through `make_estimator` from the registry, thread `categorical_idx` and device through, and keep fold fitting, scoring, and out-of-fold handling unchanged.
3. Keep `make_estimator(task, config)` backward compatible for the default estimator.
4. Extend `BaselineResult` and the artifact with `estimator_key`, `categorical_handling`, `device` (requested, effective, reason), and library versions for the chosen estimator.
5. Bump the artifact `schema_version` to 2; `load_baseline` accepts versions 1 and 2, defaulting the new fields for version 1.

## 7. Tests

1. Keep existing Phase 8 tests passing unchanged for the default estimator.
2. Registry tests that need no optional library: names, task support, unknown-name error, `estimator_available` for each, install-hint messages, `native_categoricals` and `device` config validation, and that `import automl` does not import any adapter library.
3. Simulate an absent library (for example by blocking its import) and assert `run_baseline` raises `EstimatorUnavailableError` naming the package and the install command, while the default estimator still runs.
4. Per adapter, guarded by `pytest.importorskip`: fit/predict/metric smoke test for binary, multiclass, and regression with string and integer targets; finite scores; classes and probability columns consistent with the target labels.
5. Native categorical tests per adapter: results differ from the encoded path only through categorical handling, unseen validation categories do not error, and categorical indices match the transformed column layout; the preprocessor is still fit on train rows only.
6. Parameter tests: accepted `estimator_params` take effect, controlled keys are rejected with the controlling setting named.
7. Device tests: `cpu` always works; `auto` falls back to CPU without a GPU and records the reason; `gpu` without GPU raises with the reason; a mocked probe verifies the parameter mapping for each library. GPU runs are skipped unless a GPU is detected.
8. Artifact tests: round trip with an adapter estimator, version 1 artifact still loads, and a reloaded adapter model reproduces saved out-of-fold predictions.

## 8. Documentation and roadmap

1. Update `automl/README.md` with the new `[baseline]` keys (`estimator`, `native_categoricals`, `device`), the estimator table (name, extra, install command, native categorical and GPU support), adapter defaults and controlled parameters, the native categorical column layout, the device rules and reproducibility note, and the artifact schema version 2 changes, with a Python example for people and agents (no CLI this phase).
2. Mark Phase 9 complete in `specs/roadmap.md` with a verification note once validation passes, including which extras were installed for verification.
