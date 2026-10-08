# Validation: Package and Test Skeleton

The feature is ready to merge when all required checks below pass on Python 3.12 or newer from the `automl/` project root, in an environment without competition data, Kaggle credentials, cloud access, or optional ML dependencies.

## Required Checks

1. **Install the project and test extra**

   ```bash
   python -m pip install -e '.[test]'
   ```

   Pass when pip installs the local project successfully without installing runtime ML dependencies.

2. **Import the package**

   ```bash
   python -c 'import automl; print(automl.__file__)'
   ```

   Pass when the import succeeds and resolves to the installed project source.

3. **Run the test suite**

   ```bash
   python -m pytest
   ```

   Pass when all skeleton tests succeed, including import and both CLI help invocations.

4. **Check the installed CLI entry point**

   ```bash
   automl --help
   ```

   Pass when help is printed and the command exits with status 0.

5. **Check module invocation**

   ```bash
   python -m automl --help
   ```

   Pass when equivalent help is printed and the command exits with status 0.

## Merge Readiness

- `pyproject.toml` declares the `automl` distribution, Python `>=3.12`, setuptools build backend, package discovery for `src/automl/`, the CLI entry point, and a pytest test extra.
- No runtime ML dependencies were introduced.
- The package can be imported without optional dependencies.
- Both help paths work from an editable install and tests cover them.
- The diff contains no data, credentials, generated model artifacts, or unrelated changes.
- Installation, test, and invocation commands are documented for the next contributor.