# Roadmap

This roadmap builds the workflow in small increments. Each phase should leave the project runnable, add a focused acceptance check, and avoid requiring Kaggle data, credentials, network access, cloud services, or a GPU unless that phase explicitly tests an optional integration.

## Phase 0: Package and Test Skeleton

**Status:** Complete (2026-10-08).

**Deliverable:** Minimal Python package layout, project metadata, test configuration, and a CLI entry point that prints help.

**Acceptance:** Install the core package in a clean environment, import it, run the CLI help command, and pass the empty/skeleton test suite without optional model or tracking packages.

**Verification:** Installed into the project-local Python 3.12 `.venv`; package import, `automl --help`, `python -m automl --help`, `pip check`, and all 4 tests passed. The initial fresh-environment install downloaded build/test dependencies; subsequent tests and CLI checks run locally without network access.

## Phase 1: Input Data Contract

**Deliverable:** A documented run configuration and loader for train CSV, test CSV, optional sample submission, target, and identifier columns.

**Acceptance:** Synthetic fixtures cover valid files, missing files, duplicate columns, train/test column mismatches, and invalid target/ID configuration with actionable errors.

## Phase 2: Task and Metric Selection

**Deliverable:** Explicit classification/regression task and metric configuration, with safe suggestions where task type or metric is ambiguous.

**Acceptance:** Tests verify supported binary, multiclass, and regression metric/task combinations, and verify that ambiguous or unsupported choices are surfaced instead of silently guessed.

## Phase 3: Validation Strategy

**Deliverable:** Reproducible holdout and cross-validation split construction appropriate to the selected task, with configurable seed and fold count.

**Acceptance:** Tests prove repeatable splits for a fixed seed, no train/validation overlap, and valid stratification when class counts permit it; invalid fold/class combinations return a useful error.

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

## Phase 15: End-to-End CLI

**Deliverable:** Compose the implemented stages into documented CLI commands for inspection, EDA, baseline, tuning, reporting, and submission generation.

**Acceptance:** A no-network synthetic-data smoke test runs the full local path from CSV inputs to both PDFs and a schema-validated submission; each stage can also be invoked independently.

## Phase 16: Agent Guidance

**Deliverable:** Add `automl/AGENTS.md` with workflow and safety rules, plus one focused `SKILL.md` per stable capability (initially data/EDA, model/tuning, and reporting/submission).

**Acceptance:** Each skill states invocation conditions, inputs, outputs, assumptions, and validation steps; an agent can follow the guidance to run the CLI without undocumented state or credentials.

## Phase 17: Optional PyTorch Estimator

**Deliverable:** Add a PyTorch-based tabular estimator only after a concrete use case, with scikit-learn-compatible cloning and explicit resource controls.

**Acceptance:** CPU-only synthetic tests verify fit/predict, reproducibility settings, parameter handling, and integration with the existing CV/tuning interfaces. GPU use remains optional.

## Phase 18: Hardening and Documentation

**Deliverable:** User documentation, representative synthetic examples, dependency compatibility checks, and robustness improvements based on real usage.

**Acceptance:** A fresh local setup follows documented commands; core tests pass without optional integrations; optional dependency groups have focused smoke tests; all generated artifacts and limitations are documented.

## Later, Explicitly Optional Work

Remote SageMaker or other cloud execution, richer model registries, ensembling, feature stores, and competition-specific skills should be proposed as separate increments after the local workflow is reliable. They must not become hidden requirements of the core pipeline.