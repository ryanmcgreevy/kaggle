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
