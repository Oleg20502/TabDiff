from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tabdiff.utils.paths import RUNS_ROOT, resolve_project_path


@dataclass(frozen=True)
class RunContext:
    run_dir: Path
    config_dir: Path
    checkpoints_dir: Path
    logs_dir: Path
    train_eval_dir: Path
    test_root: Path
    report_root: Path
    imputation_root: Path

    @classmethod
    def create(cls, output_root: str | Path, dataset: str, run_name: str) -> "RunContext":
        root = resolve_project_path(output_root)
        run_dir = root / dataset / run_name
        config_dir = run_dir / "config"
        checkpoints_dir = run_dir / "checkpoints"
        logs_dir = run_dir / "logs"
        train_eval_dir = run_dir / "train_eval"
        test_root = run_dir / "test"
        report_root = run_dir / "report"
        imputation_root = run_dir / "imputation"
        for path in (
            config_dir,
            checkpoints_dir,
            logs_dir,
            train_eval_dir,
            test_root,
            report_root,
            imputation_root,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return cls(
            run_dir=run_dir,
            config_dir=config_dir,
            checkpoints_dir=checkpoints_dir,
            logs_dir=logs_dir,
            train_eval_dir=train_eval_dir,
            test_root=test_root,
            report_root=report_root,
            imputation_root=imputation_root,
        )

    @classmethod
    def from_run_dir(cls, run_dir: str | Path) -> "RunContext":
        resolved = resolve_project_path(run_dir)
        return cls.create(resolved.parent.parent, resolved.parent.name, resolved.name)


def default_runs_root() -> Path:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    return RUNS_ROOT
