# Validation: Quick Baseline (Phase 8)

The feature is ready to merge when all checks pass on Python 3.12 or newer from the `automl/` project root, without competition data, Kaggle credentials, network access (after install), or optional ML dependencies.

## Required Checks

1. **Install**

   ```bash
   python -m pip install -e '.[test]'
   ```

   Pass when install succeeds and no new runtime dependency was added.

2. **Existing behavior intact**

   ```bash
   python -c 'import automl' && automl --help && python -m automl --help
   ```

   Pass when all exit 0.

3. **Test suite**

   ```bash
   python -m pytest
   ```

   Pass when all tests succeed, including Phase 0 to 7 tests and new tests covering:

   - synthetic binary, multiclass, and regression runs returning finite metrics, predictions, effective configuration, and a machine-readable result;
   - every built-in metric needing `labels`, `proba`, and `values` for the tasks it supports;
   - k-fold out-of-fold predictions covering every row once, and holdout covering only validation rows;
   - repeatability for a fixed seed, and different splits for a different split seed;
   - invalid config values and `[baseline]` TOML parsing, unknown key rejection, and override precedence;
   - invalid `estimator_params` and `random_state` inside them;
   - unseen validation classes and non-finite predictions raising errors;
   - artifact save/load round trip, overwrite refusal, `save_models=False`, and rejection of malformed or inconsistent files;
   - reloaded fold models reproducing the saved out-of-fold predictions.

4. **Leakage safety**

   Pass when tests confirm scores are computed on validation rows only, that changing validation-row features or labels does not change the fold model's training-row predictions, and that preprocessing is fit inside each fold (a fold's fitted preprocessor matches fitting on its train rows alone).

5. **Target and ID exclusion**

   Pass when tests confirm the target and ID columns are not among the fitted model's input features.

6. **Metric direction and inputs**

   Pass when tests confirm `proba` metrics receive probabilities (positive class for binary), `labels` and `values` metrics receive predictions, and the result records the metric's direction without declaring a best model.

7. **Error quality**

   Pass when each invalid-input test asserts the message names the offending setting, fold, or class and suggests a fix.

8. **Non-mutation**

   Pass when a test confirms the train frame and target are unchanged after `run_baseline`.

9. **Stage boundary**

   Pass when a test loads a saved result directory in a fresh call (no in-memory state) and obtains identical per-fold scores, summary, plan, and OOF predictions, using a saved split artifact as the split input.

10. **No optional dependencies**

    Pass when `import automl` and `run_baseline` work with only core dependencies installed (no Matplotlib, boosting libraries, Optuna, MLflow, or PyTorch).

11. **Documentation**

    Pass when `README.md` documents the `[baseline]` schema, defaults, prediction/metric rules, the artifact layout and JSON schema, the model-file trust and version caveats, and a Python example exercised by a test.

12. **Dependency health**

    ```bash
    python -m pip check
    ```

    Pass with no broken requirements.

## Merge Criteria

- All required checks pass.
- No competition data, credentials, or local absolute paths are committed or written into artifacts.
- `specs/roadmap.md` marks Phase 8 complete with a verification note.
- No CLI behavior beyond Phase 6 was added.
- Existing config files without a `[baseline]` table still load unchanged.
