# Roadmap

This roadmap builds the workflow in small increments. Each phase should leave the project runnable, add a focused acceptance check, and avoid requiring Kaggle data, credentials, network access, cloud services, or a GPU unless that phase explicitly tests an optional integration.

## Cross-Cutting Acceptance

Every phase that introduces or changes a reusable capability must document its invocation and its inputs, outputs, assumptions, limits, and validation checks in the same increment. Keep the Python API usable independently; when a CLI command exists for the capability, document that invocation too. This guidance is for both people and agents and must use the same API/CLI, not an agent-only execution path.

Stages should return structured results and persist machine-readable artifacts when they produce outputs needed by later stages. A later stage must be able to consume the documented result or artifact without relying on hidden in-memory state or rerunning unrelated work. Add focused tests for stage boundaries and artifact contracts as those boundaries are introduced.

## Phase 0: Package and Test Skeleton

**Status:** Complete (2026-10-08).

**Deliverable:** Minimal Python package layout, project metadata, test configuration, and a CLI entry point that prints help.

**Acceptance:** Install the core package in a clean environment, import it, run the CLI help command, and pass the empty/skeleton test suite without optional model or tracking packages.

**Verification:** Installed into the project-local Python 3.12 `.venv`; package import, `automl --help`, `python -m automl --help`, `pip check`, and all 4 tests passed. The initial fresh-environment install downloaded build/test dependencies; subsequent tests and CLI checks run locally without network access.

## Phase 1: Input Data Contract

**Status:** Complete (2026-10-08).

**Deliverable:** A documented run configuration and loader for train CSV, test CSV, optional sample submission, target, and identifier columns.

**Acceptance:** Synthetic fixtures cover valid files, missing files, duplicate columns, train/test column mismatches, and invalid target/ID configuration with actionable errors.

**Verification:** Installed into the project-local `.venv`; `pip check` and all 18 tests passed, covering valid inputs, missing/empty files, duplicate columns, train/test mismatches, target/ID/sample-submission errors, config errors, and input non-mutation. The only new core dependency is `pandas`; no CLI behavior was added.

## Phase 2: Task and Metric Selection

**Status:** Complete (2026-10-08).

**Deliverable:** Explicit classification/regression task and metric configuration, with safe suggestions where task type or metric is ambiguous.

**Acceptance:** Tests verify supported binary, multiclass, and regression metric/task combinations, and verify that ambiguous or unsupported choices are surfaced instead of silently guessed.

**Verification:** Installed into the project-local `.venv`; `pip check`, CLI help, and all 40 tests passed, covering every built-in task/metric pair, unset/unknown/mismatched choices, target inconsistencies, ambiguous integer targets, custom metrics, and `[task]` TOML parsing. The only new core dependency is `scikit-learn`; no CLI behavior was added.

## Phase 3: Validation Strategy

**Status:** Complete (2026-10-08).

**Deliverable:** Reproducible holdout and cross-validation split construction appropriate to the selected task, with configurable seed and fold count.

**Acceptance:** Tests prove repeatable splits for a fixed seed, no train/validation overlap, and valid stratification when class counts permit it; invalid fold/class combinations return a useful error.

**Verification:** Installed into the project-local `.venv`; `pip check` and all 66 tests passed, covering seeded repeatability, no overlap, k-fold partitioning, stratification, holdout size, config/`[validation]` errors, rare-class and null-target errors, and the JSON split artifact round trip. No new dependency; no CLI behavior was added.

## Phase 4: EDA Summary

**Deliverable:** A reusable, non-mutating EDA stage that summarizes dimensions, dtypes, missingness, duplicates, numeric distributions/outliers, categorical cardinality, target distribution, and train/test differences.

**Acceptance:** Synthetic mixed-type data produces complete structured results, including edge cases such as constant columns, all-null columns, and absent optional targets in test data.

## Phase 5: EDA PDF

**Deliverable:** PDF report generation from the structured EDA results, with charts and concise data-quality findings.

**Acceptance:** A test run produces a non-empty, readable PDF in the requested output directory; report generation does not mutate input data or require a notebook session.

## Phase 6: Leakage-Safe Preprocessing

**Deliverable:** Reusable numeric and categorical preprocessing pipelines with configurable imputation and encoding, compatible with scikit-learn CV.

**Acceptance:** Tests cover mixed types and missing values, estimator cloning, and fold-local fitting; validation data cannot influence learned preprocessing statistics.

## Phase 7: Quick Baseline

**Deliverable:** One fast scikit-learn baseline per supported task type, evaluated with the configured validation strategy and metric.

**Acceptance:** Synthetic classification and regression runs return finite metrics, predictions, effective configuration, and a machine-readable result without optional dependencies.

## Phase 8: Boosting Adapters

**Deliverable:** Add supported boosting estimators as optional, individually selectable adapters, beginning with the libraries chosen for the first release.

**Acceptance:** Each installed adapter passes a small fit/predict/metric smoke test; when an adapter dependency is absent, the baseline path still works and the CLI explains how to enable it.

## Phase 9: Optuna Tuning Core

**Deliverable:** A reusable tuner for one baseline/boosting estimator, with seeded sampler, explicit metric direction, trial/time budget, CV objective, and persisted trial results.

**Acceptance:** A small synthetic study completes within its configured budget, produces repeatable results where estimator behavior permits, and can be inspected or resumed from local study artifacts.

## Phase 10: Model-Family Search Spaces

**Deliverable:** Separate, documented Optuna search spaces for each supported estimator; users can choose families and budgets explicitly.

**Acceptance:** Tests verify sampled parameters are valid for the installed estimator and task; failed trials are recorded with a useful reason and do not corrupt successful results.

## Phase 11: Optional MLflow Tracking

**Deliverable:** Tracking adapter that mirrors run configuration, metrics, trial history, and complete fitted pipelines to MLflow when explicitly enabled.

**Acceptance:** Local runs still pass with MLflow absent. An integration test against a local MLflow tracking URI verifies a run can be logged and queried; no hosted service is required.

## Phase 12: Best-Model Results PDF

**Deliverable:** Comparison report for selected best models, including validation metrics, variability, configuration, and relevant diagnostics.

**Acceptance:** A report is generated from saved structured run results, remains accurate after process restart, and clearly distinguishes validation scores from leaderboard claims.

## Phase 13: Refit and Prediction

**Deliverable:** Refit the selected pipeline on the declared training data and generate predictions for the test rows.

**Acceptance:** Tests confirm the selected estimator and preprocessing are refit together, prediction count equals test row count, and outputs contain no invalid or non-finite values where the task disallows them.

## Phase 14: Submission Validation and CSV

**Deliverable:** Build a submission CSV using the sample submission when supplied, preserving required identifiers, column names, and order.

**Acceptance:** Tests cover schema/order/row-count mismatches, identifier alignment, and valid file writing. The workflow never calls Kaggle submission APIs or overwrites an existing output without explicit instruction.

When identifiers are available, predictions must remain aligned to the corresponding test rows. If sample-submission identifiers differ in order, either explicitly align predictions by identifier or reject the mismatch; matching identifier sets alone is not sufficient.

## Phase 15: End-to-End CLI

**Deliverable:** Compose the implemented stages into documented CLI commands for inspection, EDA, baseline, tuning, reporting, and submission generation.

**Acceptance:** A no-network synthetic-data smoke test runs the full local path from CSV inputs to both PDFs and a schema-validated submission; each stage can also be invoked independently.

## Phase 16: Agent Guidance Consolidation

**Deliverable:** Consolidate the per-capability invocation guidance already added with each capability into `automl/AGENTS.md` workflow and safety rules, plus focused `SKILL.md` files for stable capability groups (initially data/EDA, model/tuning, and reporting/submission).

**Acceptance:** Each skill states invocation conditions, inputs, outputs, assumptions, limits, and validation steps; an agent can follow the guidance to compose implemented stages through the documented Python API or CLI without undocumented state or credentials. This phase consolidates and checks guidance; it is not the first point at which capabilities become agent-usable.

## Phase 17: Optional PyTorch Estimator

**Deliverable:** Add a PyTorch-based tabular estimator only after a concrete use case, with scikit-learn-compatible cloning and explicit resource controls.

**Acceptance:** CPU-only synthetic tests verify fit/predict, reproducibility settings, parameter handling, and integration with the existing CV/tuning interfaces. GPU use remains optional.

## Phase 18: Hardening and Documentation

**Deliverable:** User documentation, representative synthetic examples, dependency compatibility checks, and robustness improvements based on real usage.

**Acceptance:** A fresh local setup follows documented commands; core tests pass without optional integrations; optional dependency groups have focused smoke tests; all generated artifacts and limitations are documented.

## Later, Explicitly Optional Work

Remote SageMaker or other cloud execution, richer model registries, ensembling, feature stores, and competition-specific skills should be proposed as separate increments after the local workflow is reliable. They must not become hidden requirements of the core pipeline.