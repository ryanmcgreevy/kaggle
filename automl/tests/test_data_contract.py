import pytest

from automl import DataConfig, DataContractError, load_config, load_data
from tests.conftest import CONFIG_TEXT


def cfg(d, **kw):
    base = dict(train_path=d / "train.csv", test_path=d / "test.csv", target="y")
    base.update(kw)
    return DataConfig(**base)


def fails(config, *fragments):
    with pytest.raises(DataContractError) as exc:
        load_data(config)
    for f in fragments:
        assert f in str(exc.value)


def test_valid_with_and_without_sample(data_dir):
    bundle = load_data(load_config(data_dir / "run.toml"))
    assert bundle.config.seed == 7
    assert bundle.config.id_columns == ("id",)
    assert list(bundle.train.columns) == ["id", "a", "b", "y"]
    assert bundle.sample_submission is not None
    assert load_data(cfg(data_dir)).sample_submission is None


def test_readme_example_config_loads(data_dir):
    from pathlib import Path

    readme = (Path(__file__).parents[1] / "README.md").read_text()
    block = readme.split("```toml\n")[1].split("```")[0]
    assert block == CONFIG_TEXT
    (data_dir / "readme.toml").write_text(block)
    assert load_data(load_config(data_dir / "readme.toml")).config.target == "y"


def test_relative_paths_and_overrides(data_dir):
    config = load_config(data_dir / "run.toml", target="b", id_columns=())
    assert config.train_path == data_dir / "train.csv"
    assert config.target == "b"
    assert config.id_columns == ()


def test_missing_and_empty_files(data_dir):
    (data_dir / "test.csv").unlink()
    fails(cfg(data_dir), "test file not found", "test.csv")
    (data_dir / "train.csv").write_text("")
    fails(cfg(data_dir), "train file", "is empty")
    (data_dir / "train.csv").write_text("id,a,b,y\n")
    fails(cfg(data_dir), "no rows")


@pytest.mark.parametrize("name", ["train", "test"])
def test_duplicate_columns(data_dir, name):
    (data_dir / f"{name}.csv").write_text("id,a,a\n1,2,3\n")
    fails(cfg(data_dir), f"{name}.csv", "duplicate columns: ['a']")


def test_column_mismatch_reports_both(data_dir):
    (data_dir / "test.csv").write_text("id,a,z\n4,1,2\n5,2,3\n")
    fails(cfg(data_dir), "missing from test ['b']", "only in test ['z']")


def test_target_rules(data_dir):
    fails(cfg(data_dir, target="nope"), "'nope' not found in train")
    (data_dir / "test.csv").write_text("id,a,b,y\n4,1,x,0\n")
    fails(cfg(data_dir), "must not be present in test")


def test_id_rules(data_dir):
    fails(cfg(data_dir, id_columns=("zzz",)), "id column 'zzz' not found in train", "not found in test")
    for ids in [("y",), ("id", "id")]:
        with pytest.raises(DataContractError):
            cfg(data_dir, id_columns=ids)


def test_sample_submission_mismatch(data_dir):
    (data_dir / "sample_submission.csv").write_text("id,y\n4,0\n")
    sample = data_dir / "sample_submission.csv"
    fails(cfg(data_dir, sample_submission_path=sample), "1 rows but test file has 2")
    sample.write_text("id,y\n4,0\n9,0\n")
    fails(cfg(data_dir, id_columns=("id",), sample_submission_path=sample), "values differ")
    sample.write_text("k,y\n4,0\n5,0\n")
    fails(cfg(data_dir, id_columns=("id",), sample_submission_path=sample), "not found in sample submission")


def test_bad_config_files(data_dir):
    path = data_dir / "bad.toml"
    path.write_text("version = [")
    with pytest.raises(DataContractError, match="not valid TOML"):
        load_config(path)
    path.write_text('version = 2\nfoo = 1\n[data]\ntrain="a"\ntest="b"\ntarget="y"\nbar=1\n')
    with pytest.raises(DataContractError) as exc:
        load_config(path)
    msg = str(exc.value)
    assert "unknown top-level keys ['foo']" in msg and "'version' 2" in msg and "['bar']" in msg
    with pytest.raises(DataContractError, match="not found"):
        load_config(data_dir / "missing.toml")


def test_sample_submission_file_problems(data_dir):
    sample = data_dir / "sample_submission.csv"
    sample.write_text("id,y,y\n4,0,0\n5,0,0\n")
    fails(cfg(data_dir, sample_submission_path=sample), "sample submission", "duplicate columns: ['y']")
    sample.write_text("")
    fails(cfg(data_dir, sample_submission_path=sample), "sample submission file", "is empty")


def test_config_missing_sections_and_bad_types(data_dir):
    path = data_dir / "bad.toml"
    path.write_text("version = 1\n")
    with pytest.raises(DataContractError, match=r"missing \[data\] table"):
        load_config(path)
    path.write_text('version = 1\n[data]\ntrain = "a"\n')
    with pytest.raises(DataContractError) as exc:
        load_config(path)
    assert "missing required key 'test'" in str(exc.value)
    assert "missing required key 'target'" in str(exc.value)
    path.write_text('version = 1\n[data]\ntrain = 1\ntest = "b"\ntarget = "y"\nid_columns = "id"\n')
    with pytest.raises(DataContractError) as exc:
        load_config(path)
    assert "'train' must be a path string" in str(exc.value)
    assert "'id_columns' must be a list of strings" in str(exc.value)


def test_inputs_not_mutated(data_dir):
    before = {p.name: p.read_bytes() for p in data_dir.glob("*.csv")}
    load_data(load_config(data_dir / "run.toml"))
    assert before == {p.name: p.read_bytes() for p in data_dir.glob("*.csv")}
