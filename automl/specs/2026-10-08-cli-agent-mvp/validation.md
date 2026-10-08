# Validation: Current-Capability CLI and Agent MVP (Phase 6)

Run from `automl/` on Python 3.12 or newer using synthetic fixtures, without Kaggle data, network access, or credentials.

## Required Checks

1. **Focused and full tests**

   ```bash
   python -m pytest tests/test_cli.py
   python -m pytest
   ```

   Pass when CLI tests cover both installed and module entry points, all four commands, JSON output, task/metric confirmation, artifact contracts, and actionable errors.

2. **CLI help and dependency health**

   ```bash
   automl --help
   python -m automl --help
   python -m pip check
   ```

3. **Synthetic workflow**

   Use a temporary TOML and train/test CSV. Run `validate`, `split`, `eda`, and `report` in sequence. Verify the split artifact loads with `load_splits`, EDA artifact loads with `load_eda`, and report outputs include a non-empty PDF and valid findings JSON when Matplotlib is installed.

4. **Choice and output safety**

   Pass when unset task/metric choices fail with suggestions and no selection, invalid input fails with a useful message, split/EDA reruns refuse overwrite, report reruns refuse overwrite by default, and report replacement succeeds only with `--overwrite`.

5. **Optional dependency**

   Pass when core import, validation, splits, and EDA do not need Matplotlib. Without Matplotlib, report failure names `pip install -e '.[reports]'`.

6. **Agent skill structure**

   Validate each SKILL.md frontmatter, folder/name match, and keyword-rich description; verify activation for the intended workflow. Specifically test discovery from the repository-root workspace. The nested `automl/.agents/skills/` location is not guaranteed by the VS Code project-skill location reference; do not report automatic discovery as verified unless observed.

## Merge Readiness

- The full test suite and `pip check` pass.
- README and all skills document only supported behavior and match tested CLI commands.
- The roadmap Phase 6 verification note records actual results and any skill-discovery limitation.
- No competition data, credentials, or unrelated worktree content is added.