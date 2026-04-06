from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
DATA_ROOT = PROJECT_ROOT / "data"
RAW_INFO_ROOT = DATA_ROOT / "Info"
PROCESSED_DATA_ROOT = PROJECT_ROOT / "processed_data"
RUNS_ROOT = PROJECT_ROOT / "runs"


def resolve_project_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value.resolve()
    return (PROJECT_ROOT / value).resolve()

