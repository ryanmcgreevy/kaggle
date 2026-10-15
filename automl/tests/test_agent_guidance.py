import json
import re
import socket
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from automl.cli import build_parser, main

ROOT = Path(__file__).resolve().parents[1]
SKILLS = ROOT / ".agents" / "skills"
EXPECTED = {
    "automl-workflow", "automl-data-intake", "automl-validation-splits", "automl-eda-reporting",
    "automl-baseline", "automl-tuning", "automl-model-report", "automl-refit", "automl-submission",
    "automl-ensemble",
}
STAGE_SKILLS = EXPECTED - {"automl-workflow"}
HEADINGS = ("## Use when", "## Procedure", "## Inputs", "## Outputs", "## Assumptions and limits", "## Check")
PRODUCES = {
    "automl-validation-splits": ("splits.json",),
    "automl-baseline": ("baseline.json", "oof_predictions.csv"),
    "automl-tuning": ("tuning_families.json",),
    "automl-model-report": ("model_report.pdf", "model_report_summary.json"),
    "automl-ensemble": ("ensemble.json", "ensemble.joblib"),
    "automl-refit": ("predictions.csv", "refit.json", "pipeline.joblib"),
    "automl-submission": ("submission.csv", "submission.json"),
}


def _skill(name: str) -> str:
    return (SKILLS / name / "SKILL.md").read_text()


def _frontmatter(text: str) -> dict[str, str]:
    match = re.match(r"---\n(.*?)\n---\n", text, re.S)
    assert match, "missing frontmatter"
    fields = {}
    for line in match.group(1).splitlines():
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip().strip('"')
    return fields


def _commands(text: str) -> list[list[str]]:
    found = []
    for block in re.findall(r"```(?:bash|sh)?\n(.*?)```", text, re.S):
        for line in block.splitlines():
            if line.startswith("automl "):
                found.append(line.split()[1:])
    for line in re.findall(r"`(automl [^`]+)`", text):
        found.append(line.split()[1:])
    return found


def test_all_skills_exist_with_valid_frontmatter():
    assert {p.name for p in SKILLS.iterdir() if p.is_dir()} == EXPECTED
    for name in EXPECTED:
        fields = _frontmatter(_skill(name))
        assert fields["name"] == name
        assert len(fields["description"]) > 40


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_skills_have_required_headings(name):
    text = _skill(name)
    for heading in HEADINGS:
        assert heading in text, f"{name} lacks {heading}"


def test_agents_md_routes_to_every_skill_and_states_the_rules():
    text = (ROOT / "AGENTS.md").read_text()
    for name in EXPECTED:
        assert f"`{name}`" in text
    lowered = text.lower()
    assert "use only the existing `automl` cli and python api" in lowered
    assert "user approves" in lowered
    assert "never submit or upload to kaggle" in lowered
    assert "do not assume automatic discovery" in lowered
    assert len(text.splitlines()) < 60


def test_workflow_and_stage_skills_state_confirmations():
    text = _skill("automl-workflow")
    for phrase in ("new or empty", "tuning budget", "no `--overwrite`", "manual upload"):
        assert phrase in text
    assert "never choose one" in _skill("automl-tuning").lower()
    assert "never upload" in _skill("automl-submission").lower()
    for extra, name in (("reports", "automl-model-report"), ("tuning", "automl-tuning"), ("boosting", "automl-baseline")):
        assert f"'.[{extra}]'" in _skill(name)
    assert "out-of-fold" in _skill("automl-ensemble").lower()


def _parser_options():
    parser = build_parser()
    subparsers = next(a for a in parser._actions if a.choices)
    return {
        name: {s for action in sub._actions for s in action.option_strings}
        for name, sub in subparsers.choices.items()
    }


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_documented_commands_and_options_exist(name):
    options = _parser_options()
    text = _skill(name)
    for command in _commands(text):
        assert command[0] in options, f"{name}: unknown command {command[0]}"
        for token in command[1:]:
            if token.startswith("--"):
                assert token in options[command[0]], f"{name}: {command[0]} has no {token}"
    for flag in re.findall(r"`(--[a-z-]+)", text):
        assert any(flag in o for o in options.values()), f"{name}: unknown option {flag}"


def test_skill_api_names_are_exported():
    import automl

    for name in EXPECTED:
        for line in _skill(name).splitlines():
            if "API" not in line:
                continue
            for api in re.findall(r"`([a-z_]+)`", line.split("API", 1)[1]):
                assert hasattr(automl, api), f"{name}: {api} is not exported"


def test_skills_have_no_local_paths_or_secrets():
    for name in EXPECTED:
        text = _skill(name) + (ROOT / "AGENTS.md").read_text()
        assert "/home/" not in text
        assert not re.search(r"(?i)(password|secret_key|api_key)\s*=", text)


# ---- executable walkthrough ----


def _project(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    n, m = 90, 30
    a = rng.normal(size=n + m)
    b = rng.choice(["x", "y", "z"], n + m)
    y = ((a + 0.8 * (b == "x") + rng.normal(scale=0.5, size=n + m)) > 0.4).astype(int)
    frame = pd.DataFrame({"id": np.arange(n + m), "a": a, "b": b, "y": y})
    frame.iloc[:n].to_csv(tmp_path / "train.csv", index=False)
    frame.iloc[n:].drop(columns="y").to_csv(tmp_path / "test.csv", index=False)
    pd.DataFrame({"id": np.arange(n, n + m), "y": 0.0}).to_csv(tmp_path / "sample_submission.csv", index=False)
    (tmp_path / "run.toml").write_text(
        'version = 1\nseed = 7\n\n[data]\ntrain = "train.csv"\ntest = "test.csv"\n'
        'sample_submission = "sample_submission.csv"\ntarget = "y"\nid_columns = ["id"]\n\n'
        '[task]\ntask = "binary"\nmetric = "roc_auc"\n\n[validation]\nn_folds = 3\n'
    )
    return tmp_path


@pytest.fixture
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


def _walkthrough_commands(project: Path) -> list[list[str]]:
    block = re.search(r"## Stage order\n\n```bash\n(.*?)```", _skill("automl-workflow"), re.S).group(1)
    steps = []
    for line in block.splitlines():
        tokens = line.split()[1:]
        steps.append([
            "2" if t == "30" else str(project / t) if t.startswith(("run.toml", "runs/")) else t for t in tokens
        ])
    return steps


def test_workflow_skill_commands_run_end_to_end(tmp_path, capsys, no_network):
    pytest.importorskip("optuna")
    pytest.importorskip("matplotlib")
    project = _project(tmp_path)
    steps = _walkthrough_commands(project)
    assert [s[0] for s in steps] == [
        "validate", "split", "eda", "report", "baseline", "tune", "model-report", "refit", "submit",
    ]

    for step in steps:
        code = main(step)
        out, err = capsys.readouterr()
        assert code == 0, f"{step[0]} failed: {err}"
        json.loads(out)

    runs = project / "runs" / "1"
    expected = {
        "automl-validation-splits": runs / "splits",
        "automl-baseline": runs / "baseline",
        "automl-tuning": runs / "tuning",
        "automl-model-report": runs / "model-reports",
        "automl-refit": runs / "refit",
        "automl-submission": runs / "submission",
    }
    for skill, directory in expected.items():
        for artifact in PRODUCES[skill]:
            assert (directory / artifact).is_file(), f"{skill}: {artifact} not produced"
    assert (runs / "eda" / "eda.json").is_file()
    assert (runs / "reports" / "eda_report.pdf").is_file()

    submission = pd.read_csv(runs / "submission" / "submission.csv")
    sample = pd.read_csv(project / "sample_submission.csv")
    test = pd.read_csv(project / "test.csv")
    assert list(submission.columns) == list(sample.columns)
    assert submission["id"].tolist() == test["id"].tolist()


def test_run_command_documented_in_workflow_skill_works(tmp_path, capsys, no_network):
    pytest.importorskip("optuna")
    pytest.importorskip("matplotlib")
    project = _project(tmp_path)
    out = project / "runs" / "2"

    code = main(["run", str(project / "run.toml"), "--output-dir", str(out),
                 "--families", "hist_gradient_boosting", "--n-trials", "2"])
    capsys.readouterr()

    assert code == 0
    for name in ("splits", "eda", "baseline", "tuning", "reports", "refit", "submission"):
        assert (out / name).is_dir()
    assert _skill("automl-workflow").count("splits/ eda/ baseline/ tuning/ reports/ refit/ submission/") == 1
