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

## Input Data Contract

A run is described by a TOML file. Relative paths resolve against the config file's directory. Unknown keys are rejected.

```toml
version = 1
seed = 7

[data]
train = "train.csv"
test = "test.csv"
sample_submission = "sample_submission.csv"
target = "y"
id_columns = ["id"]
```

- `target` is required in train and must be absent from test.
- `id_columns` is optional and explicit; each must exist in train and test and differ from the target.
- Train and test must otherwise have identical columns; mismatches list missing and extra columns.
- `sample_submission` is optional; its row count (and ID values, when `id_columns` is set) must match test.

```python
from automl import load_config, load_data

bundle = load_data(load_config("run.toml"))   # keyword overrides, e.g. target="y"
bundle.train, bundle.test, bundle.sample_submission
```

Invalid inputs raise `automl.DataContractError`, which lists every problem found.
