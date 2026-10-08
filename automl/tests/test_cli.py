import json
import subprocess
import sys
import sysconfig
from pathlib import Path

from automl import load_eda, load_splits


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_cli(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=10,
    )


def assert_help_output(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0
    assert "usage: automl" in result.stdout
    assert "validate" in result.stdout
    assert "split" in result.stdout
    assert "eda" in result.stdout
    assert "report" in result.stdout


def test_module_help() -> None:
    assert_help_output(run_cli([sys.executable, "-m", "automl", "--help"]))


def test_installed_command_help() -> None:
    script_name = "automl.exe" if sys.platform == "win32" else "automl"
    executable = Path(sysconfig.get_path("scripts")) / script_name
    assert executable.is_file(), "Install the project with `python -m pip install -e '.[test]'`."
    assert_help_output(run_cli([str(executable), "--help"]))


def test_invoking_without_a_command_prints_help() -> None:
    assert_help_output(run_cli([sys.executable, "-m", "automl"]))


def _explicit_config(data_dir: Path) -> Path:
    config_path = data_dir / "run.toml"
    config_path.write_text(
        config_path.read_text()
        + '\n[task]\ntask = "binary"\nmetric = "roc_auc"\n'
    )
    return config_path


def test_validate_command_reports_machine_readable_result(data_dir: Path) -> None:
    config_path = _explicit_config(data_dir)
    result = run_cli([sys.executable, "-m", "automl", "validate", str(config_path)])

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "valid"
    assert payload["task"] == "binary"
    assert payload["metric"] == "roc_auc"
    assert payload["train_rows"] == 3
    assert payload["test_rows"] == 2


def test_validate_requires_explicit_task_and_metric(data_dir: Path) -> None:
    result = run_cli([sys.executable, "-m", "automl", "validate", str(data_dir / "run.toml")])

    assert result.returncode == 1
    assert "setting 'task' is not set" in result.stderr


def test_split_command_writes_non_overwriting_artifact(data_dir: Path, tmp_path: Path) -> None:
    (data_dir / "train.csv").write_text(
        "id,a,b,y\n" + "".join(f"{row},{row},x,{row % 2}\n" for row in range(10))
    )
    config_path = _explicit_config(data_dir)
    output_dir = tmp_path / "splits"
    result = run_cli(
        [sys.executable, "-m", "automl", "split", str(config_path), "--output-dir", str(output_dir)]
    )

    assert result.returncode == 0
    payload = json.loads((output_dir / "splits.json").read_text())
    assert payload["schema_version"] == 1
    assert payload["task"] == "binary"
    assert json.loads(result.stdout)["n_splits"] == 5
    assert len(load_splits(output_dir / "splits.json").splits) == 5

    duplicate = run_cli(
        [sys.executable, "-m", "automl", "split", str(config_path), "--output-dir", str(output_dir)]
    )
    assert duplicate.returncode == 1
    assert "already exists" in duplicate.stderr


def test_eda_command_writes_json_and_report_is_optional(data_dir: Path, tmp_path: Path) -> None:
    config_path = _explicit_config(data_dir)
    output_dir = tmp_path / "eda"
    eda = run_cli(
        [sys.executable, "-m", "automl", "eda", str(config_path), "--output-dir", str(output_dir)]
    )

    assert eda.returncode == 0
    artifact = output_dir / "eda.json"
    payload = json.loads(artifact.read_text())
    assert payload["schema_version"] == 1
    assert payload["dataset"]["n_rows"] == 3
    assert load_eda(artifact)["dataset"]["n_rows"] == 3

    duplicate_eda = run_cli(
        [sys.executable, "-m", "automl", "eda", str(config_path), "--output-dir", str(output_dir)]
    )
    assert duplicate_eda.returncode == 1
    assert "already exists" in duplicate_eda.stderr

    report = run_cli(
        [
            sys.executable,
            "-m",
            "automl",
            "report",
            str(config_path),
            str(artifact),
            "--output-dir",
            str(output_dir / "reports"),
        ]
    )
    if report.returncode == 0:
        report_result = json.loads(report.stdout)
        report_path = Path(report_result["path"])
        findings_path = Path(report_result["findings_path"])
        assert report_path.is_file() and report_path.stat().st_size > 0
        assert findings_path.is_file()
        assert json.loads(findings_path.read_text())["schema_version"] == 1

        duplicate_report = run_cli(
            [
                sys.executable,
                "-m",
                "automl",
                "report",
                str(config_path),
                str(artifact),
                "--output-dir",
                str(output_dir / "reports"),
            ]
        )
        assert duplicate_report.returncode == 1
        assert "already exists" in duplicate_report.stderr

        overwrite_report = run_cli(
            [
                sys.executable,
                "-m",
                "automl",
                "report",
                str(config_path),
                str(artifact),
                "--output-dir",
                str(output_dir / "reports"),
                "--overwrite",
            ]
        )
        assert overwrite_report.returncode == 0
    else:
        assert "pip install -e '.[reports]'" in report.stderr


def test_installed_command_runs_validation(data_dir: Path) -> None:
    config_path = _explicit_config(data_dir)
    script_name = "automl.exe" if sys.platform == "win32" else "automl"
    executable = Path(sysconfig.get_path("scripts")) / script_name
    result = run_cli([str(executable), "validate", str(config_path)])

    assert result.returncode == 0
    assert json.loads(result.stdout)["status"] == "valid"