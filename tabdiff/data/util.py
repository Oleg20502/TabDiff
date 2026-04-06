from __future__ import annotations

import enum
from pathlib import Path
from typing import Any, Union

import numpy as np

from tabdiff.utils.io import dump_pickle, load_json, load_pickle


class TaskType(enum.Enum):
    BINCLASS = "binclass"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"

    def __str__(self) -> str:
        return self.value


def raise_unknown(unknown_what: str, unknown_value: Any):
    raise ValueError(f"Unknown {unknown_what}: {unknown_value}")


def get_categories(X_train_cat: np.ndarray | None):
    return (
        None
        if X_train_cat is None
        else [len(set(X_train_cat[:, i])) for i in range(X_train_cat.shape[1])]
    )


__all__ = [
    "Path",
    "TaskType",
    "dump_pickle",
    "get_categories",
    "load_json",
    "load_pickle",
    "raise_unknown",
    "Union",
]
