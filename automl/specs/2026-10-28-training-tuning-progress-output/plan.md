# Training and Tuning Progress Output Plan

## Task Groups

1. **Defining progress and output contracts**
   - Inventory the progress events needed by baseline, tuning, ensembling, refitting, and `run`.
   - Define a shared progress representation and formatting contract, including elapsed time, fold lifecycle, trial scores, objective direction, and failures.
   - Define how CLI progress and native estimator output are routed to stderr while stdout remains JSON.
   - Define the opt-in Python API mechanism and keep library calls silent by default.

2. **Implementing shared progress behavior and CLI options**
   - Add the shared progress mechanism and immediate-flush CLI output.
   - Add mutually exclusive `--verbose` and `--quiet` handling consistently to in-scope commands.
   - Ensure quiet mode suppresses progress without suppressing actionable errors.
   - Route supported estimator-native output away from stdout in verbose mode.

3. **Integrating baseline, tuning, and ensembling**
   - Report applicable stage/model starts and fold starts/completions with elapsed time.
   - Report completed tuning-trial validation scores and best-so-far updates in the correct metric direction.
   - Represent failed trials clearly and preserve meaningful output when resuming studies.
   - Exercise optional estimator adapters when their dependencies are installed.

4. **Integrating refit and end-to-end workflow**
   - Propagate progress mode and events through refitting and the `run` orchestration path.
   - Avoid duplicate or contradictory events when the workflow composes child stages.
   - Verify standalone and end-to-end commands honor the same verbosity semantics.

5. **Documenting supported behavior and limits**
   - Document CLI options, stderr/stdout behavior, and the opt-in Python API.
   - Document supported native estimator verbosity and estimator-specific limits.
   - Clarify that native training loss may differ from the selected validation metric and that iteration output is estimator-dependent.
   - Update directly related CLI/API and agent guidance documentation.

6. **Validating progress and regression safety**
   - Add synthetic tests proving progress is observable before a long-running operation completes.
   - Test stdout JSON and stderr progress separation in default, verbose, and quiet modes, including native estimator output.
   - Test fold/trial messages, elapsed time, score direction, best-score updates, failed trials, and resumed studies.
   - Verify progress does not change model parameters, seeds, splits, scores, persisted artifact contracts, or resume compatibility.
   - Run focused tests for all in-scope stages and the relevant existing regression suite.
