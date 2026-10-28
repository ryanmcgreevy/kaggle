# Training and Tuning Progress Output Validation

## Acceptance Checks

### Timeliness and progress content

- A local synthetic test uses a deliberately long-running operation to prove the first progress message is flushed and observable before that operation completes.
- Baseline and ensemble tests verify applicable stage/model starts, fold starts and completions, and elapsed-time fields.
- Tuning tests verify completed-trial validation scores and best-so-far updates for both minimized and maximized metrics.
- Failed trials are identified as failures and do not produce success-shaped score updates.
- Resumed-study tests verify progress describes the new work accurately and the study remains resumable.

### CLI output and modes

- For baseline, tuning, ensembling, refitting, and `run`, captured stdout parses as valid result JSON in default, verbose, and quiet modes.
- Default progress is on stderr and is immediately flushed; verbose mode keeps progress on stderr and routes supported native estimator output away from stdout.
- Optional estimator adapters are covered when installed, including native output where supported.
- `--verbose` and `--quiet` are mutually exclusive and behave consistently across standalone commands and `run`.
- Quiet mode emits no progress but retains actionable errors and a non-zero status on failure.

### Python API

- Python API calls produce no terminal progress by default.
- Opting in uses the documented progress mechanism and emits the same relevant event information as the CLI without requiring library callers to use terminal output.

### Compatibility and documentation

- Tests verify progress does not change estimator parameters unrelated to logging, seeds, splits, validation scores, artifact schemas, or tuning resume compatibility.
- Existing baseline, tuning, ensemble, refit, and end-to-end workflow behavior remains usable.
- Documentation explains supported native verbosity, that training loss may differ from the selected validation metric, and that iteration-level output is estimator-dependent.
- No new core dependency, competition data, network access, or cloud resource is required.

## Suggested Validation Sequence

1. Run focused progress tests for the shared event mechanism and baseline/tuning output.
2. Run focused ensembling, refitting, and end-to-end CLI output tests.
3. Run optional-adapter tests for dependencies available in the environment.
4. Run the existing CLI, tuning/resume, baseline, ensemble, refit, and workflow regression tests.
5. Run the full AutoML test suite and verify stdout/stderr capture for every supported mode.

## Execution Record

- Focused progress, baseline, tuning, CLI stage, and agent-guidance tests: **133 passed**.
- Full suite from `automl/`: **625 passed, 1 skipped** (GPU-dependent test; no usable GPU).
- Verbose CLI adapter check ran with LightGBM, XGBoost, and CatBoost installed; stdout remained parseable result JSON and native output was routed to stderr.
- `git diff --check` and Python compilation passed.
- One unrelated MLflow SQLAlchemy deprecation warning was emitted during tracking tests.
