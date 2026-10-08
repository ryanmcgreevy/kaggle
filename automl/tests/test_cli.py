import subprocess
import sys
import sysconfig
from pathlib import Path


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
    assert "No workflow commands are available yet." in result.stdout


def test_module_help() -> None:
    assert_help_output(run_cli([sys.executable, "-m", "automl", "--help"]))


def test_installed_command_help() -> None:
    script_name = "automl.exe" if sys.platform == "win32" else "automl"
    executable = Path(sysconfig.get_path("scripts")) / script_name
    assert executable.is_file(), "Install the project with `python -m pip install -e '.[test]'`."
    assert_help_output(run_cli([str(executable), "--help"]))


def test_invoking_without_a_command_prints_help() -> None:
    assert_help_output(run_cli([sys.executable, "-m", "automl"]))