from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from tabdiff.utils.io import load_yaml
from tabdiff.utils.paths import PROJECT_ROOT, resolve_project_path


TRAIN_BACKBONES = {"unimod_mlp", "transformer_encoder", "mlp"}
RECOGNITION_BACKBONES = {"mlp", "unimod_mlp", "unimod_transformer_heads"}
CHECKPOINT_SELECTORS = {"best_ema", "best", "latest"}
TEST_KINDS = {"sample", "report", "impute"}


def _require_mapping(container: dict[str, Any], key: str) -> dict[str, Any]:
    value = container.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Expected '{key}' to be a mapping.")
    return value


def _normalize_path(path: str | Path) -> Path:
    value = Path(path)
    if not value.is_absolute():
        value = (PROJECT_ROOT / value).resolve()
    else:
        value = value.resolve()
    return value


def _with_default(data: dict[str, Any], key: str, default: Any) -> None:
    if key not in data:
        data[key] = default


def _validate_train_config(config: dict[str, Any], source_path: Path) -> dict[str, Any]:
    run = _require_mapping(config, "run")
    data = _require_mapping(config, "data")
    model = _require_mapping(config, "model")
    diffusion = _require_mapping(config, "diffusion")
    train = _require_mapping(config, "train")
    logging = _require_mapping(config, "logging")
    sampling = _require_mapping(config, "sampling")

    _with_default(run, "output_root", "runs")
    _with_default(run, "seed", 0)
    _with_default(run, "deterministic", False)
    _with_default(run, "debug", False)

    if not run.get("dataset"):
        raise ValueError(f"{source_path}: run.dataset is required.")
    if not run.get("name"):
        raise ValueError(f"{source_path}: run.name is required.")

    _with_default(data, "dequant_dist", "none")
    _with_default(data, "int_dequant_factor", 0.0)

    denoiser = _require_mapping(model, "denoiser")
    backbone = denoiser.get("backbone")
    if backbone not in TRAIN_BACKBONES:
        raise ValueError(
            f"{source_path}: model.denoiser.backbone must be one of {sorted(TRAIN_BACKBONES)}, got {backbone!r}."
        )
    params = denoiser.get("params")
    if not isinstance(params, dict):
        raise ValueError(f"{source_path}: model.denoiser.params must be a mapping.")

    _with_default(model, "y_only", False)
    variational = model.get("variational")
    if variational is None:
        variational = {"enabled": False}
        model["variational"] = variational
    if not isinstance(variational, dict):
        raise ValueError(f"{source_path}: model.variational must be a mapping.")
    _with_default(variational, "enabled", False)
    if variational["enabled"]:
        recognition = variational.get("recognition")
        if not isinstance(recognition, dict):
            raise ValueError(f"{source_path}: model.variational.recognition must be a mapping when enabled.")
        recognition_backbone = recognition.get("backbone")
        if recognition_backbone not in RECOGNITION_BACKBONES:
            raise ValueError(
                f"{source_path}: model.variational.recognition.backbone must be one of {sorted(RECOGNITION_BACKBONES)}, got {recognition_backbone!r}."
            )
        if not isinstance(recognition.get("params"), dict):
            raise ValueError(f"{source_path}: model.variational.recognition.params must be a mapping.")
        _with_default(variational, "latent_policy", "consistency")
        _with_default(variational, "posterior_inputs", "x0_xt_t")
        _with_default(variational, "latent_dim", 64)
        _with_default(variational, "kl_schedule", "warmup")
        _with_default(variational, "kl_weight", 0.01)
        _with_default(variational, "kl_warmup_steps", 1000)

    _with_default(diffusion, "num_timesteps", 50)
    _with_default(diffusion, "noise_dist", "uniform_t")
    _require_mapping(diffusion, "edm")
    _require_mapping(diffusion, "noise_schedule")
    _require_mapping(diffusion, "sampler")

    for key in (
        "steps",
        "lr",
        "weight_decay",
        "ema_decay",
        "batch_size",
        "check_val_every",
        "lr_scheduler",
        "factor",
        "reduce_lr_patience",
        "closs_weight_schedule",
        "c_lambda",
        "d_lambda",
    ):
        if key not in train:
            raise ValueError(f"{source_path}: train.{key} is required.")

    _with_default(logging, "backend", "tensorboard")
    _with_default(logging, "plot_density", False)
    _with_default(sampling, "batch_size", 4096)

    config["config_path"] = str(source_path)
    return config


def _validate_test_config(config: dict[str, Any], source_path: Path) -> dict[str, Any]:
    test = _require_mapping(config, "test")
    _with_default(test, "kind", "sample")
    kind = test["kind"]
    if kind not in TEST_KINDS:
        raise ValueError(f"{source_path}: test.kind must be one of {sorted(TEST_KINDS)}, got {kind!r}.")
    _with_default(test, "name", "default")
    _with_default(test, "checkpoint", "best_ema")
    checkpoint = test["checkpoint"]
    if isinstance(checkpoint, str) and checkpoint in CHECKPOINT_SELECTORS:
        pass
    elif isinstance(checkpoint, str):
        test["checkpoint"] = checkpoint
    else:
        raise ValueError(f"{source_path}: test.checkpoint must be a string.")

    _with_default(test, "num_samples", None)
    _with_default(test, "sample_batch_size", None)
    _with_default(test, "num_timesteps", None)
    _with_default(test, "stochastic_sampler", None)
    _with_default(test, "second_order_correction", None)

    report = test.get("report")
    if report is None:
        report = {}
        test["report"] = report
    if not isinstance(report, dict):
        raise ValueError(f"{source_path}: test.report must be a mapping.")
    _with_default(report, "num_runs", 20)

    imputation = test.get("imputation")
    if imputation is None:
        imputation = {}
        test["imputation"] = imputation
    if not isinstance(imputation, dict):
        raise ValueError(f"{source_path}: test.imputation must be a mapping.")
    _with_default(imputation, "trial_start", 0)
    _with_default(imputation, "trial_size", 50)
    _with_default(imputation, "resample_rounds", 1)
    _with_default(imputation, "condition", "x_t")
    _with_default(imputation, "w_num", 0.6)
    _with_default(imputation, "w_cat", 0.6)
    _with_default(imputation, "guidance_run_dir", None)

    config["config_path"] = str(source_path)
    return config


def load_train_config(path: str | Path) -> dict[str, Any]:
    source_path = _normalize_path(path)
    return _validate_train_config(load_yaml(source_path), source_path)


def load_test_config(path: str | Path) -> dict[str, Any]:
    source_path = _normalize_path(path)
    return _validate_test_config(load_yaml(source_path), source_path)


def load_training_manifest(run_dir: str | Path) -> dict[str, Any]:
    run_path = resolve_project_path(run_dir)
    manifest_path = run_path / "config" / "resolved_train.yaml"
    manifest = load_yaml(manifest_path)
    manifest["run_dir"] = str(run_path)
    return manifest


def resolve_train_config(config: dict[str, Any], *, run_name_override: str | None = None) -> dict[str, Any]:
    resolved = deepcopy(config)
    if run_name_override:
        resolved["run"]["name"] = run_name_override
    return resolved

