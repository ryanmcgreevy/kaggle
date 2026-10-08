# AutoML

Local-first tools for tabular Kaggle competitions. The package and CLI are a Phase 0 skeleton; no data inspection, model training, tuning, reporting, or submission commands are implemented yet.

## Install and Test

From this directory, install the package and its test extra in editable mode:

```bash
python -m pip install -e '.[test]'
```

Run the test suite:

```bash
python -m pytest
```

## CLI

Show the current command-line help:

```bash
automl --help
python -m automl --help
```