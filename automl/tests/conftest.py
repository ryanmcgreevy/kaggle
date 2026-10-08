from pathlib import Path

import pytest

CONFIG_TEXT = """\
version = 1
seed = 7

[data]
train = "train.csv"
test = "test.csv"
sample_submission = "sample_submission.csv"
target = "y"
id_columns = ["id"]
"""


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    (tmp_path / "train.csv").write_text("id,a,b,y\n1,1.0,x,0\n2,2.0,y,1\n3,,x,0\n")
    (tmp_path / "test.csv").write_text("id,a,b\n4,1.5,x\n5,2.5,y\n")
    (tmp_path / "sample_submission.csv").write_text("id,y\n4,0\n5,0\n")
    (tmp_path / "run.toml").write_text(CONFIG_TEXT)
    return tmp_path


@pytest.fixture
def eda_frames():
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 100
    num = rng.normal(10, 2, n)
    num[:3] = [100.0, -80.0, 90.0]
    num[10:14] = np.nan
    base = pd.DataFrame(
        {
            "row_id": np.arange(n),
            "serial": np.arange(n) + 1000,
            "num": num,
            "cat": rng.choice(["a", "b", "c"], n),
            "flag": rng.random(n) > 0.5,
            "const": 1.0,
            "allnull": np.nan,
            "y": np.array([0] * 70 + [1] * 30),
        }
    )
    train = pd.concat([base, base.iloc[:2]], ignore_index=True)
    m = 60
    test = pd.DataFrame(
        {
            "row_id": np.arange(n, n + m),
            "serial": np.arange(m) + 5000,
            "num": rng.normal(20, 2, m),
            "cat": rng.choice(["a", "b", "d"], m),
            "flag": rng.random(m) > 0.5,
            "const": 1.0,
            "allnull": np.nan,
        }
    )
    return train, test
