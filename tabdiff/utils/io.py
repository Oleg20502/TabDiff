from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import yaml


def ensure_parent(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_json(data: Any, path: str | Path) -> None:
    target = ensure_parent(path)
    target.write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")


def load_yaml(path: str | Path) -> dict[str, Any]:
    loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping in {path}, got {type(loaded).__name__}.")
    return loaded


def dump_yaml(data: Any, path: str | Path) -> None:
    target = ensure_parent(path)
    text = yaml.safe_dump(data, sort_keys=False, allow_unicode=False)
    target.write_text(text, encoding="utf-8")


def load_pickle(path: str | Path) -> Any:
    return pickle.loads(Path(path).read_bytes())


def dump_pickle(data: Any, path: str | Path) -> None:
    target = ensure_parent(path)
    target.write_bytes(pickle.dumps(data))

