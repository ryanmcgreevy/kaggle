# Training and Tuning Progress Output Requirements

## Goal

Provide consistent, timely progress feedback for long-running AutoML CLI stages without compromising machine-readable stdout, reproducibility, or library usability.

## Context

The CLI supports baseline evaluation, tuning, ensembling, refitting, and an end-to-end workflow. These operations can run folds, trials, and estimator training for a substantial time. Users need to understand what is running and how validation results are progressing, while scripts and agents need stdout to remain valid JSON. The feature must work locally, without adding a core dependency or requiring terminal output from Python API callers.

## Scope and Decisions

- Apply progress behavior consistently to baseline, tuning (including resumed studies), ensembling, refitting, and the end-to-end workflow.
- Emit default CLI progress to stderr and flush each progress message immediately. Keep stdout reserved for machine-readable JSON.
- Report stage and model starts, fold starts and completions, elapsed time, and completed-trial validation scores with the best score so far.
- Add mutually exclusive `--verbose` and `--quiet` options to relevant standalone commands and `run`.
- `--quiet` suppresses progress but does not suppress actionable errors.
- `--verbose` retains regular progress and additionally enables native estimator training output where supported. Route native output away from stdout.
- Expose the same progress mechanism through an opt-in Python API. Library callers receive no terminal output unless they explicitly opt in.
- Report validation scores according to the configured objective direction; failed trials must not be presented as completed scored trials. Resumed studies must distinguish prior trials from newly completed trials as appropriate.
- Progress is observational only: it must not change model parameters unrelated to logging, random seeds, splits, validation scores, persisted artifact contracts, or study resume compatibility.
- Document which estimators support native verbosity and its limits. Training loss is not necessarily the selected validation metric, and iteration-level output depends on the estimator.
- Do not add a core dependency, require competition data, or introduce network/cloud requirements.

## Functional Requirements

1. Every in-scope CLI command emits progress to stderr by default and continues to emit parseable JSON on stdout.
2. Progress messages are flushed as they are produced and include the stage/model, fold lifecycle, elapsed time, and completed-trial score/best-so-far details applicable to that stage.
3. Best-score updates respect whether the configured metric is minimized or maximized. Failed trials are reported as failures, not as successful scored completions.
4. `--verbose` and `--quiet` are mutually exclusive and behave consistently across standalone stages and `run`.
5. Verbose estimator-native output is directed away from stdout, including for optional estimator adapters when installed.
6. Quiet mode suppresses progress while preserving actionable error reporting and non-zero failure status.
7. Python callers can opt in to progress through the API and receive it through a documented mechanism without unsolicited library terminal output.
8. The implementation and tests cover resumed tuning studies and failed trials without breaking resume compatibility.
9. Progress behavior does not alter model results, randomness, split behavior, saved artifact schemas, or existing workflow behavior when callers do not request verbosity.

## Non-Goals

- Changing model training, validation, tuning budgets, or artifact formats.
- Guaranteeing estimator-specific iteration-level progress where the underlying library does not support it.
- Treating native training loss as equivalent to the selected validation metric.
- Requiring a new core dependency or enabling progress by default in library calls.
- Adding remote tracking, hosted execution, or competition-specific output behavior.
