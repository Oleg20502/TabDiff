from __future__ import annotations

import argparse

from tabdiff.config import load_train_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train TabDiff with a YAML config.")
    parser.add_argument("--config", required=True, help="Path to the training YAML config.")
    parser.add_argument("--device", default=None, help="Execution device, e.g. cpu or cuda:0.")
    parser.add_argument("--run-name", default=None, help="Optional run-name override.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from tabdiff.engine.runtime import resolve_device, run_training

    config = load_train_config(args.config)
    run_dir = run_training(config, resolve_device(args.device), run_name_override=args.run_name)
    print(f"Training finished. Run directory: {run_dir}")


if __name__ == "__main__":
    main()
