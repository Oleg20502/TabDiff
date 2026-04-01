#!/usr/bin/env python3
"""
Debug TabDiff MLE / XGBoost fit failures.

The production path in eval/mle/mle.py wraps model.fit() in try/except ValueError: pass,
which leads to NotFittedError on predict. This script runs the same data preparation and
fit loop but prints full tracebacks on the first failure.

Run from the repository root (paths synthetic/ and data/ are relative):

  python scripts/debug_mle_fit.py --samples path/to/samples.csv
  python scripts/debug_mle_fit.py --samples tabdiff/result/adult/try_1_variational/350/samples.csv --first-only --cpu
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from eval.mle.mle import prepare_ml_problem  # noqa: E402


def _diag_xy(name: str, x: np.ndarray, y: np.ndarray) -> None:
    print(f"\n[{name}] x shape={x.shape} dtype={x.dtype} y shape={y.shape} dtype={y.dtype}")
    try:
        xf = np.asarray(x, dtype=np.float64)
        print(f"  x nan={int(np.isnan(xf).sum())} inf={int(np.isinf(xf).sum())} min={np.nanmin(xf)} max={np.nanmax(xf)}")
    except (ValueError, TypeError) as e:
        print(f"  x could not cast to float64 for stats: {e}")
    uy, cnt = np.unique(y, return_counts=True)
    print(f"  y unique={uy.tolist()} counts={cnt.tolist()}")


def _override_tree_method_cpu(kwargs: dict) -> dict:
    out = {k: (list(v) if isinstance(v, (list, tuple)) else [v]) for k, v in kwargs.items()}
    out["tree_method"] = ["hist"]
    return out


def run_classification(
    x_trains: np.ndarray,
    y_trains: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    classifiers: list,
    *,
    first_only: bool,
    use_cpu: bool,
) -> None:
    unique_labels = np.unique(y_trains)
    if len(unique_labels) < 2:
        print(
            "\nWarning: training labels have a single class; production code avoids predict() in that case.\n"
            "If your synthetic data lost one class, fix sampling — XGBoost needs both classes for binary MLE."
        )

    for spec in classifiers:
        model_class = spec["class"]
        raw_kw = spec.get("kwargs", {})
        model_kwargs = _override_tree_method_cpu(raw_kw) if use_cpu else raw_kw
        grid = list(ParameterGrid(model_kwargs))
        if first_only:
            grid = grid[:1]

        print(f"\n=== {model_class.__name__}: {len(grid)} hyperparameter setting(s) ===")

        for i, param in enumerate(grid):
            print(f"\n--- [{i + 1}/{len(grid)}] {param} ---")
            model = model_class(**param)
            try:
                model.fit(x_trains, y_trains)
            except Exception:
                print("FIT FAILED (full traceback):")
                traceback.print_exc()
                sys.exit(1)

            if len(unique_labels) == 1:
                print("Single-class y_trains: skipping predict (matches mle.py branch).")
            else:
                pred = model.predict(x_valid)
                print(f"fit ok; predict(valid)[:5]={pred[:5].tolist()}")


def run_regression(
    x_trains: np.ndarray,
    y_trains: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    regressors: list,
    *,
    first_only: bool,
    use_cpu: bool,
) -> None:
    y_trains_f = np.log(np.clip(y_trains, 1, 20000))
    for spec in regressors:
        model_class = spec["class"]
        raw_kw = spec.get("kwargs", {})
        model_kwargs = _override_tree_method_cpu(raw_kw) if use_cpu else raw_kw
        grid = list(ParameterGrid(model_kwargs))
        if first_only:
            grid = grid[:1]
        print(f"\n=== {model_class.__name__}: {len(grid)} hyperparameter setting(s) ===")
        for i, param in enumerate(grid):
            print(f"\n--- [{i + 1}/{len(grid)}] {param} ---")
            model = model_class(**param)
            try:
                model.fit(x_trains, y_trains_f)
            except Exception:
                print("FIT FAILED (full traceback):")
                traceback.print_exc()
                sys.exit(1)
            pred = model.predict(x_valid)
            print(f"fit ok; predict(valid)[:5]={pred[:5].tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MLE-style XGBoost fit without swallowing ValueError (debug NotFittedError)."
    )
    parser.add_argument(
        "--samples",
        required=True,
        help="Path to synthetic samples.csv (same table passed to TabMetrics.evaluate_mle).",
    )
    parser.add_argument("--dataname", default="adult", help="Dataset name under data/ and synthetic/.")
    parser.add_argument(
        "--first-only",
        action="store_true",
        help="Only the first ParameterGrid point (fastest GPU/debug).",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force tree_method=hist for XGBoost models to isolate GPU / gpu_hist issues.",
    )
    args = parser.parse_args()

    os.chdir(REPO_ROOT)

    samples_path = os.path.abspath(args.samples) if os.path.isabs(args.samples) else os.path.abspath(os.path.join(os.getcwd(), args.samples))
    if not os.path.isfile(samples_path):
        print(f"error: samples file not found: {samples_path}", file=sys.stderr)
        sys.exit(2)

    info_path = os.path.join("data", args.dataname, "info.json")
    if not os.path.isfile(info_path):
        print(f"error: info.json not found: {info_path}", file=sys.stderr)
        sys.exit(2)

    with open(info_path, encoding="utf-8") as f:
        info = deepcopy(json.load(f))

    task_type = info["task_type"]
    train = pd.read_csv(samples_path).to_numpy()
    test_path = os.path.join("synthetic", args.dataname, "test.csv")
    if not os.path.isfile(test_path):
        print(f"error: test split not found: {test_path}", file=sys.stderr)
        sys.exit(2)
    test = pd.read_csv(test_path).to_numpy()

    val_path = os.path.join("synthetic", args.dataname, "val.csv")
    val = pd.read_csv(val_path).to_numpy() if os.path.isfile(val_path) else None
    if val is None:
        print("Note: no val.csv — using same 1/9 random split as prepare_ml_problem when val is None.")

    print(f"task_type={task_type}")
    print(f"samples={samples_path}")
    print(f"train(syn) {train.shape} test {test.shape} val {None if val is None else val.shape}")

    x_trains, y_trains, x_valid, y_valid, x_test, y_test, models = prepare_ml_problem(
        train, test, info, val=val
    )
    _diag_xy("train fold", x_trains, y_trains)
    _diag_xy("valid", x_valid, y_valid)
    _diag_xy("test", x_test, y_test)

    if task_type == "regression":
        run_regression(
            x_trains, y_trains, x_valid, y_valid, models, first_only=args.first_only, use_cpu=args.cpu
        )
    else:
        run_classification(
            x_trains, y_trains, x_valid, y_valid, models, first_only=args.first_only, use_cpu=args.cpu
        )

    print("\nAll attempted fits completed without exception.")


if __name__ == "__main__":
    main()
