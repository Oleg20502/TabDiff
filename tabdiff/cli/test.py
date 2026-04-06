from __future__ import annotations

import argparse

from tabdiff.config import load_test_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test TabDiff from an existing run directory.")
    parser.add_argument("--run-dir", required=True, help="Run directory produced by training.")
    parser.add_argument("--config", required=True, help="Path to the test YAML config.")
    parser.add_argument("--device", default=None, help="Execution device, e.g. cpu or cuda:0.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from tabdiff.engine.runtime import resolve_device, run_test

    config = load_test_config(args.config)
    output_dir = run_test(args.run_dir, config, resolve_device(args.device))
    print(f"Test finished. Outputs: {output_dir}")


if __name__ == "__main__":
    main()
