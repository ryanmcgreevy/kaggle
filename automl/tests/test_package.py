from pathlib import Path

import automl


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_package_import_resolves_to_source_tree() -> None:
    assert Path(automl.__file__).resolve() == PROJECT_ROOT / "src" / "automl" / "__init__.py"