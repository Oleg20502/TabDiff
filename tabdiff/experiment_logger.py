"""Unified experiment logging: Weights & Biases or TensorBoard."""

from __future__ import annotations

import os
from typing import Any, Mapping, MutableMapping, Optional

import numpy as np


def _flatten_scalars(
    d: Mapping[str, Any], prefix: str = ""
) -> dict[str, float]:
    """Flatten nested dicts to scalar floats for TensorBoard."""
    out: dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else str(k)
        if isinstance(v, Mapping):
            out.update(_flatten_scalars(v, key))
        elif isinstance(v, (bool, np.bool_)):
            out[key] = float(int(v))
        elif isinstance(v, (int, float, np.floating, np.integer)):
            out[key] = float(v)
    return out


class ExperimentLogger:
    """Common interface for wandb / TensorBoard / no-op."""

    def define_metric(self, metric_name: str, **kwargs: Any) -> None:
        raise NotImplementedError

    def log(self, data: MutableMapping[str, Any]) -> None:
        raise NotImplementedError

    def finish(self) -> None:
        raise NotImplementedError


class NullExperimentLogger(ExperimentLogger):
    def define_metric(self, metric_name: str, **kwargs: Any) -> None:
        pass

    def log(self, data: MutableMapping[str, Any]) -> None:
        pass

    def finish(self) -> None:
        pass


class WandbExperimentLogger(ExperimentLogger):
    def __init__(self, run: Any) -> None:
        self._run = run

    def define_metric(self, metric_name: str, **kwargs: Any) -> None:
        self._run.define_metric(metric_name, **kwargs)

    def log(self, data: MutableMapping[str, Any]) -> None:
        self._run.log(data)

    def finish(self) -> None:
        self._run.finish()


class TensorBoardExperimentLogger(ExperimentLogger):
    def __init__(self, log_dir: str) -> None:
        from torch.utils.tensorboard import SummaryWriter

        os.makedirs(log_dir, exist_ok=True)
        self._writer = SummaryWriter(log_dir=log_dir)
        self._fallback_step = 0

    def define_metric(self, metric_name: str, **kwargs: Any) -> None:
        pass

    def log(self, data: MutableMapping[str, Any]) -> None:
        step = data.get("epoch")
        if step is None:
            self._fallback_step += 1
            step = self._fallback_step
        else:
            step = int(step)
        flat = _flatten_scalars(data)
        for name, value in flat.items():
            if name == "epoch":
                continue
            self._writer.add_scalar(name, value, step)

    def finish(self) -> None:
        self._writer.flush()
        self._writer.close()


def build_experiment_logger(
    backend: str,
    *,
    project_name: str,
    run_name: str,
    config: Optional[dict] = None,
    tensorboard_log_dir: Optional[str] = None,
) -> ExperimentLogger:
    """
    backend: 'wandb' | 'tensorboard' | 'none'
    tensorboard_log_dir: required when backend == 'tensorboard'
    """
    if backend == "none":
        return NullExperimentLogger()
    if backend == "tensorboard":
        if not tensorboard_log_dir:
            raise ValueError("tensorboard_log_dir is required for tensorboard backend")
        return TensorBoardExperimentLogger(tensorboard_log_dir)
    if backend == "wandb":
        import wandb

        run = wandb.init(
            project=project_name,
            name=run_name,
            config=config,
            mode="online",
        )
        return WandbExperimentLogger(run)
    raise ValueError(f"Unknown logger backend: {backend}")
