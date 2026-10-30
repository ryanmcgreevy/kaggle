# Ideas for Later Investigation

## Independent Step Skills Without a Shared AutoML Core

**Recorded:** 2026-10-30.

**Status:** Exploratory alternative, not an approved implementation plan. This
does not replace [roadmap Phase 22](./roadmap.md#phase-22-agents-only-skills-and-python-script-interface),
which currently retains a shared Python core.

### Goal

Simplify AutoML into discrete, understandable, independently replaceable skills
using the standard Agent Skills format. Each skill owns its `SKILL.md`, Python
scripts under `scripts/`, dependency declarations, and tests. Scripts use
third-party libraries directly, with no imports from the existing `automl`
package or another skill's private modules.

Share artifact contracts, not implementation code. The agent connects steps by
passing explicit file paths rather than relying on a central workflow engine,
shared runtime, or hidden in-memory state. A workflow skill can describe stage
order and decision points without implementing orchestration code.

### Candidate Skill Boundaries

| Skill | Responsibility |
|---|---|
| Data inspection | Validate input files and schema; produce an inspection summary. |
| Validation splits | Generate and persist explicit row assignments. |
| EDA | Produce descriptive statistics and optional reports. |
| Model experiment | Train and evaluate one model family; optionally tune it. |
| Model comparison | Read experiment results and rank compatible candidates. |
| Ensemble | Combine compatible saved predictions or models. |
| Prediction | Apply a fitted model to test data. |
| Submission | Validate predictions against the expected submission format. |

Keep boundaries cohesive rather than creating a skill for every function.
Combine baseline and tuning within model experiments because they share
preprocessing, fitting, and evaluation. Investigate separate family-specific
experiment skills where native categorical handling or tuning differs enough
to justify them. Define ownership of full-data fitting/refitting explicitly
before implementing the experiment-to-prediction hand-off.

### Deliberate Simplifications

- Use scikit-learn and estimator libraries directly instead of maintaining a
  general-purpose estimator abstraction.
- Start with local artifacts; defer MLflow integration.
- Prefer explicit stage inputs over a large universal configuration.
- Avoid universal candidate registries and interchangeable storage backends.
- Accept small amounts of duplicated argument parsing and boundary validation
  rather than introducing another shared framework.
- Select capabilities based on actual needs instead of promising complete
  parity with the existing package.

Removing the shared core alone does not simplify the system. Reduce scope and
abstraction deliberately rather than scattering the same complexity among
scripts.

### Contracts and Safeguards to Preserve

Document versioned artifact formats and validate them at each consumer:

- Task, metric, score direction, effective parameters, seeds, and dependency
  versions.
- Data and split fingerprints, with a precisely specified hashing method.
- Row identifiers and prediction ordering.
- Probability columns and class ordering.
- Out-of-fold prediction coverage and training/validation provenance.
- Model compatibility and actionable failures for invalid or incompatible
  artifacts.

Use contract examples and synthetic cross-skill integration tests to detect
drift without introducing a shared production runtime. Preserve leakage-safe
preprocessing, explicit budgets, overwrite protection, progress reporting,
local-data handling, and prediction/submission alignment. Keep consequential
confirmation gates in skill procedures and enforce deterministic safeguards in
scripts. Do not upload data or submit to Kaggle automatically.

For persistence, prefer standard library/estimator objects over pickled custom
classes that require another skill's implementation module to be importable.
Document dependency/version constraints and load only trusted model artifacts.
Investigate how genuinely necessary custom transforms can remain reloadable
without reintroducing cross-skill code dependencies.

### Tradeoffs and Open Questions

- Individual steps become easier to understand, test, distribute, and replace,
  but cross-step guarantees require more explicit contract maintenance.
- Duplicated validation and preprocessing can drift; identify acceptable
  duplication and test behavioral consistency where it matters.
- Ensembling is the hardest boundary: combining predictions is straightforward,
  but leakage-safe stacking evaluation requires stronger provenance and
  training protocols.
- Consider averaging/voting first and deferring general stacking, or support
  stacking with explicitly limited validation semantics. Never label an
  in-sample fit score as validation.
- Decide whether skills must be independently installable, how dependencies
  are resolved, and whether isolated environments are needed.
- Define supported workspace discovery and working-directory-independent
  invocation.
- Decide how much existing artifact compatibility and tuning-resume behavior
  is worth retaining versus documenting a clean migration boundary.

### Investigation Sequence

1. Build one independent model-experiment skill as a proof of concept.
2. Connect inspection, splits, experiment, prediction, and submission through
   explicit artifacts.
3. Verify contracts and failure paths with synthetic integration tests,
   including row alignment, split reuse, model reload, and leakage safeguards.
4. Add comparison, reports, and ensembles selectively based on demonstrated
   needs.
5. Compare complexity, usability, duplication, and supported capabilities
   against the shared-core Phase 22 approach before selecting a direction.
6. Retire the existing implementation only after the selected replacement
   covers the agreed needs and the migration is verified.
