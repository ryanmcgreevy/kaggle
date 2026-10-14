# Validation: Boosting Adapters (Phase 9)

The feature is ready to merge when all checks pass on Python 3.12 or newer from the `automl/` project root, without competition data, Kaggle credentials, or a GPU. Network access is needed only to install the optional extras.

## Required Checks

1. **Core install, no extras**

   ```bash
   python -m pip install -e '.[test]'
   python -c 'import automl' && automl --help && python -m automl --help
   python -m pytest
   ```

   Pass when install succeeds with no new core dependency, all commands exit 0, and the suite passes with LightGBM, XGBoost, and CatBoost absent (adapter smoke tests skipped, absence tests passing).

2. **Extras install**

   ```bash
   python -m pip install -e '.[test,boosting]'
   python -m pip check
   ```

   Pass when each of `.[lightgbm]`, `.[xgboost]`, `.[catboost]`, and `.[boosting]` installs cleanly and `pip check` reports no broken requirements.

3. **Test suite with extras**

   ```bash
   python -m pytest
   ```

   Pass when all tests succeed, including Phase 0 to 8 tests and new tests covering:

   - the registry (names, tasks, availability checks, unknown-name error) and config validation for `estimator`, `native_categoricals`, and `device`;
   - a fit/predict/metric smoke test per installed adapter for binary, multiclass, and regression, with string and integer targets, finite scores, and probability columns consistent with `classes_`;
   - parameter handling: accepted `estimator_params` take effect and controlled keys are rejected naming the controlling setting;
   - native categoricals per adapter: unseen validation categories and missing values do not error, indices match the transformed column layout, and the encoder setting is ignored and recorded;
   - device resolution: `cpu`, `auto` fallback with a recorded reason, `gpu` raising when unavailable, and library-specific parameter mapping via a mocked probe;
   - artifact schema version 2 round trip with an adapter estimator, version 1 artifacts still loading, and reloaded adapter models reproducing saved out-of-fold predictions.

4. **Optional-dependency isolation**

   Pass when tests confirm `import automl` does not import `lightgbm`, `xgboost`, or `catboost`, and that, with each library's import blocked, selecting it raises `EstimatorUnavailableError` naming the package and the exact `pip install 'automl[...]'` command, before any fitting, while the default estimator still runs.

5. **Phase 8 behavior unchanged**

   Pass when all Phase 8 tests pass unmodified for the default estimator, including default preprocessing and encoded categorical handling.

6. **Leakage safety**

   Pass when tests confirm native categorical encoding and all preprocessing are fit on train rows only for every adapter, that scores use validation rows only, and that target and ID columns never reach any adapter.

7. **Reproducibility and defaults**

   Pass when tests confirm CPU runs repeat for a fixed seed and splits for each installed adapter, that early stopping is off, and that no adapter writes files (for example a `catboost_info` directory) into the working directory.

8. **GPU opt-in**

   Pass when the default is CPU, `device="auto"` never fails on a machine without GPU support, and the result records requested device, effective device, and reason. If a GPU and a GPU-capable library are available, a real GPU smoke run passes; otherwise it is skipped and noted in the roadmap verification.

9. **Error quality**

   Pass when each invalid-input test asserts the message names the offending setting, estimator, parameter, or package and suggests a fix.

10. **Documentation**

    Pass when `README.md` documents the new `[baseline]` keys, the estimator table with extras and install commands, adapter defaults and controlled parameters, the native categorical layout, device rules and the GPU reproducibility note, artifact schema version 2, and a Python example exercised by a test.

## Merge Criteria

- All required checks pass, with the core-only run and the full-extras run both recorded.
- No competition data, credentials, or local absolute paths are committed or written into artifacts.
- `specs/roadmap.md` marks Phase 9 complete with a verification note that lists the library versions tested and whether a GPU run occurred.
- No CLI behavior beyond Phase 6 was added.
- Existing config files without the new keys still load unchanged and produce the Phase 8 result.
