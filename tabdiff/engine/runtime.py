from __future__ import annotations

import copy
import os
import random
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from tabdiff.config.loader import load_training_manifest, resolve_train_config
from tabdiff.data import TabDiffDataset
from tabdiff.engine.run_context import RunContext
from tabdiff.experiment_logger import build_experiment_logger
from tabdiff.kl_schedulers import build_kl_weight_schedule
from tabdiff.metrics import TabMetrics
from tabdiff.models.builders import build_denoiser_backbone, build_recognition_model
from tabdiff.models.denoiser import Model
from tabdiff.models.noise_schedule import LogLinearNoise_PerColumn, PowerMeanNoise_PerColumn
from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
from tabdiff.trainer import Trainer
from tabdiff.utils.io import dump_json, dump_yaml, load_json
from tabdiff.utils.paths import DATA_ROOT, PROCESSED_DATA_ROOT, resolve_project_path

warnings.filterwarnings("ignore")


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__!r} is not JSON serializable")


def resolve_device(device: str | None) -> str:
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _set_deterministic(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _load_info(dataset_name: str) -> dict[str, Any]:
    return load_json(DATA_ROOT / dataset_name / "info.json")


def _processed_data_paths(dataset_name: str) -> tuple[str, str, str | None]:
    dataset_dir = PROCESSED_DATA_ROOT / dataset_name
    real_path = dataset_dir / "real.csv"
    test_path = dataset_dir / "test.csv"
    val_path = dataset_dir / "val.csv"
    return str(real_path), str(test_path), str(val_path) if val_path.exists() else None


def _build_metrics(dataset_name: str, info: dict[str, Any], device: str, mode: str) -> TabMetrics:
    is_dcr = "dcr" in dataset_name
    real_path, test_path, val_path = _processed_data_paths(dataset_name)
    if val_path is None:
        print(
            f"{dataset_name} does not have a validation split in processed_data/. "
            "MLE evaluation will derive validation data from training data."
        )
    if mode == "train":
        metric_list = ["density"]
    elif is_dcr:
        metric_list = ["dcr"]
    else:
        metric_list = ["density", "mle", "c2st"]
    return TabMetrics(real_path, test_path, val_path, info, device, metric_list=metric_list)


def _apply_debug_overrides(train_config: dict[str, Any]) -> None:
    train_config["train"]["check_val_every"] = 2
    train_config["diffusion"]["num_timesteps"] = 4
    train_config["train"]["batch_size"] = 4096
    train_config["sampling"]["batch_size"] = 10000


def _select_checkpoint(checkpoints_dir: Path, selector: str) -> Path:
    if selector == "best_ema":
        matches = sorted(checkpoints_dir.glob("best_ema_model_*.pt"))
    elif selector == "best":
        matches = sorted(checkpoints_dir.glob("best_model_*.pt"))
    elif selector == "latest":
        matches = sorted(checkpoints_dir.glob("model_*.pt"))
        if not matches:
            matches = sorted(checkpoints_dir.glob("ema_model_*.pt"))
    else:
        candidate = Path(selector)
        if candidate.is_absolute():
            return candidate
        direct = checkpoints_dir / selector
        if direct.exists():
            return direct
        return resolve_project_path(selector)
    if not matches:
        raise FileNotFoundError(f"Could not resolve checkpoint selector {selector!r} in {checkpoints_dir}.")
    return matches[-1]


def _maybe_adjust_y_only_schedule(
    train_config: dict[str, Any],
    dataset_info: dict[str, Any],
    d_numerical: int,
    categories: np.ndarray,
) -> None:
    model_cfg = train_config["model"]
    if not model_cfg.get("y_only", False):
        return
    source_run_dir = model_cfg.get("y_only_source_run_dir")
    if not source_run_dir:
        return
    source_manifest = load_training_manifest(source_run_dir)
    source_diffusion = source_manifest["diffusion"]
    if source_diffusion.get("scheduler") != "power_mean_per_column":
        return
    schedule_cfg = train_config["diffusion"]["noise_schedule"]
    if dataset_info["task_type"] == "regression":
        noise_schedule = PowerMeanNoise_PerColumn(
            num_numerical=d_numerical,
            **source_diffusion["noise_schedule"],
        )
        schedule_cfg["rho"] = noise_schedule.rho()[0].item()
    else:
        noise_schedule = LogLinearNoise_PerColumn(
            num_categories=len(categories),
            **source_diffusion["noise_schedule"],
        )
        schedule_cfg["k"] = noise_schedule.k()[0].item()


def _build_logger(train_config: dict[str, Any], run_context: RunContext):
    logging_cfg = train_config["logging"]
    backend = logging_cfg.get("backend", "tensorboard")
    if train_config["run"].get("debug", False):
        backend = "none"
    tensorboard_dir = str(run_context.logs_dir / "tensorboard") if backend == "tensorboard" else None
    return build_experiment_logger(
        backend,
        project_name=f"tabdiff_{train_config['run']['dataset']}",
        run_name=train_config["run"]["name"],
        config=train_config if backend == "wandb" else None,
        tensorboard_log_dir=tensorboard_dir,
    )


def _build_training_components(
    train_config: dict[str, Any],
    device: str,
    *,
    run_context: RunContext,
    checkpoint_path: Path | None,
    evaluation_dir: Path,
    num_samples_to_generate: int | None = None,
    guidance_run_dir: str | None = None,
) -> tuple[Trainer, Any]:
    dataset_name = train_config["run"]["dataset"]
    dataset_info = _load_info(dataset_name)
    data_dir = DATA_ROOT / dataset_name
    data_cfg = train_config["data"]

    train_data = TabDiffDataset(
        dataset_name,
        str(data_dir),
        dataset_info,
        y_only=train_config["model"].get("y_only", False),
        isTrain=True,
        dequant_dist=data_cfg["dequant_dist"],
        int_dequant_factor=data_cfg["int_dequant_factor"],
    )
    val_data = TabDiffDataset(
        dataset_name,
        str(data_dir),
        dataset_info,
        y_only=train_config["model"].get("y_only", False),
        isTrain=False,
        dequant_dist=data_cfg["dequant_dist"],
        int_dequant_factor=data_cfg["int_dequant_factor"],
    )
    train_loader = DataLoader(
        train_data,
        batch_size=train_config["train"]["batch_size"],
        shuffle=True,
        num_workers=4,
    )
    metrics = _build_metrics(dataset_name, dataset_info, device, mode="test" if checkpoint_path else "train")

    d_numerical = train_data.d_numerical
    categories = train_data.categories

    model_cfg = copy.deepcopy(train_config["model"])
    _maybe_adjust_y_only_schedule(train_config, dataset_info, d_numerical, categories)
    denoiser_cfg = copy.deepcopy(model_cfg["denoiser"])
    variational_cfg = copy.deepcopy(model_cfg["variational"])
    if model_cfg.get("y_only", False):
        denoiser_params = denoiser_cfg["params"]
        denoiser_params["use_mlp"] = False
        denoiser_params["dim_t"] = 128

    latent_dim = int(variational_cfg.get("latent_dim", 0)) if variational_cfg.get("enabled", False) else 0
    denoiser_cfg["params"]["d_numerical"] = d_numerical
    denoiser_cfg["params"]["categories"] = (categories + 1).tolist()
    backbone = build_denoiser_backbone(
        denoiser_cfg,
        d_numerical=d_numerical,
        categories=(categories + 1).tolist(),
        latent_dim=latent_dim,
    )
    model = Model(backbone, **train_config["diffusion"]["edm"]).to(device)

    recognition_model = None
    if variational_cfg.get("enabled", False):
        recognition_model = build_recognition_model(
            variational_cfg,
            num_numerical_features=d_numerical,
            num_classes_per_column=categories.tolist(),
        ).to(device)

    y_only_model = None
    if guidance_run_dir:
        guidance_manifest = load_training_manifest(guidance_run_dir)
        guidance_checkpoint = _select_checkpoint(resolve_project_path(guidance_run_dir) / "checkpoints", "best_ema")
        guidance_latent_dim = 0
        guidance_variational = guidance_manifest["model"]["variational"]
        if guidance_variational.get("enabled", False):
            guidance_latent_dim = int(guidance_variational["latent_dim"])
        guidance_backbone = build_denoiser_backbone(
            guidance_manifest["model"]["denoiser"],
            d_numerical=d_numerical,
            categories=(categories + 1).tolist(),
            latent_dim=guidance_latent_dim,
        )
        y_only_model = Model(guidance_backbone, **guidance_manifest["diffusion"]["edm"]).to(device)
        state_dicts = torch.load(guidance_checkpoint, map_location=device)
        y_only_model.load_state_dict(state_dicts["denoise_fn"])

    diffusion_cfg = copy.deepcopy(train_config["diffusion"])
    diffusion = UnifiedCtimeDiffusion(
        num_classes=categories,
        num_numerical_features=d_numerical,
        denoise_fn=model,
        y_only_model=y_only_model,
        num_timesteps=diffusion_cfg["num_timesteps"],
        scheduler=diffusion_cfg["scheduler"],
        cat_scheduler=diffusion_cfg["cat_scheduler"],
        noise_dist=diffusion_cfg["noise_dist"],
        edm_params=diffusion_cfg["edm"],
        noise_dist_params=diffusion_cfg.get("noise_dist_params", {}),
        noise_schedule_params=diffusion_cfg["noise_schedule"],
        sampler_params=diffusion_cfg["sampler"],
        device=device,
        recognition_model=recognition_model,
        latent_dim=latent_dim,
        latent_policy=variational_cfg.get("latent_policy", "consistency"),
        kl_weight=variational_cfg.get("kl_weight", 1.0),
    ).to(device)
    diffusion.train()

    logger = _build_logger(train_config, run_context)
    trainer = Trainer(
        diffusion,
        train_loader,
        train_data,
        val_data,
        metrics,
        logger,
        lr=train_config["train"]["lr"],
        weight_decay=train_config["train"]["weight_decay"],
        steps=train_config["train"]["steps"],
        batch_size=train_config["train"]["batch_size"],
        check_val_every=train_config["train"]["check_val_every"],
        sample_batch_size=train_config["sampling"]["batch_size"],
        checkpoint_dir=str(run_context.checkpoints_dir),
        evaluation_dir=str(evaluation_dir),
        num_samples_to_generate=num_samples_to_generate,
        lr_scheduler=train_config["train"]["lr_scheduler"],
        reduce_lr_patience=train_config["train"]["reduce_lr_patience"],
        factor=train_config["train"]["factor"],
        ema_decay=train_config["train"]["ema_decay"],
        closs_weight_schedule=train_config["train"]["closs_weight_schedule"],
        c_lambda=train_config["train"]["c_lambda"],
        d_lambda=train_config["train"]["d_lambda"],
        device=device,
        ckpt_path=str(checkpoint_path) if checkpoint_path else None,
        y_only=model_cfg.get("y_only", False),
        kl_schedule=build_kl_weight_schedule(variational_cfg, variational_cfg.get("enabled", False)),
        plot_density=bool(train_config["logging"].get("plot_density", False)),
    )
    return trainer, logger


def _save_train_metadata(raw_config: dict[str, Any], resolved_config: dict[str, Any], run_context: RunContext) -> None:
    dump_yaml(raw_config, run_context.config_dir / "train.yaml")
    dump_yaml(resolved_config, run_context.config_dir / "resolved_train.yaml")
    dump_json(resolved_config, run_context.config_dir / "resolved_train.json")


def run_training(train_config: dict[str, Any], device: str, run_name_override: str | None = None) -> Path:
    resolved_config = resolve_train_config(train_config, run_name_override=run_name_override)
    run_cfg = resolved_config["run"]
    if run_cfg.get("debug", False):
        _apply_debug_overrides(resolved_config)
    if run_cfg.get("deterministic", False):
        _set_deterministic(int(run_cfg.get("seed", 0)))

    run_context = RunContext.create(run_cfg["output_root"], run_cfg["dataset"], run_cfg["name"])
    _save_train_metadata(train_config, resolved_config, run_context)

    trainer, logger = _build_training_components(
        resolved_config,
        device,
        run_context=run_context,
        checkpoint_path=None,
        evaluation_dir=run_context.train_eval_dir,
    )
    try:
        trainer.run_loop()
    finally:
        logger.finish()
    return run_context.run_dir


def _override_test_sampling(train_config: dict[str, Any], test_config: dict[str, Any]) -> dict[str, Any]:
    resolved = copy.deepcopy(train_config)
    test_cfg = test_config["test"]
    if test_cfg.get("sample_batch_size") is not None:
        resolved["sampling"]["batch_size"] = int(test_cfg["sample_batch_size"])
    if test_cfg.get("num_timesteps") is not None:
        resolved["diffusion"]["num_timesteps"] = int(test_cfg["num_timesteps"])
    if test_cfg.get("stochastic_sampler") is not None:
        resolved["diffusion"]["sampler"]["stochastic_sampler"] = bool(test_cfg["stochastic_sampler"])
    if test_cfg.get("second_order_correction") is not None:
        resolved["diffusion"]["sampler"]["second_order_correction"] = bool(test_cfg["second_order_correction"])
    return resolved


def run_test(run_dir: str | Path, test_config: dict[str, Any], device: str) -> Path:
    run_context = RunContext.from_run_dir(run_dir)
    train_config = _override_test_sampling(load_training_manifest(run_context.run_dir), test_config)
    test_cfg = test_config["test"]
    checkpoint_path = _select_checkpoint(run_context.checkpoints_dir, test_cfg["checkpoint"])
    test_name = test_cfg["name"]

    if test_cfg["kind"] == "report":
        evaluation_dir = run_context.report_root / test_name
    elif test_cfg["kind"] == "impute":
        evaluation_dir = run_context.imputation_root / test_name
    else:
        evaluation_dir = run_context.test_root / test_name
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    dump_yaml(test_config, evaluation_dir / "test_config.yaml")

    trainer, logger = _build_training_components(
        train_config,
        device,
        run_context=run_context,
        checkpoint_path=checkpoint_path,
        evaluation_dir=evaluation_dir,
        num_samples_to_generate=test_cfg.get("num_samples"),
        guidance_run_dir=test_cfg["imputation"].get("guidance_run_dir"),
    )

    try:
        if test_cfg["kind"] == "report":
            if "dcr" in train_config["run"]["dataset"]:
                trainer.report_test_dcr(int(test_cfg["report"]["num_runs"]))
            else:
                trainer.report_test(int(test_cfg["report"]["num_runs"]))
        elif test_cfg["kind"] == "impute":
            imputation_cfg = test_cfg["imputation"]
            trainer.test_impute(
                int(imputation_cfg["trial_start"]),
                int(imputation_cfg["trial_size"]),
                int(imputation_cfg["resample_rounds"]),
                str(imputation_cfg["condition"]),
                str(evaluation_dir),
                float(imputation_cfg["w_num"]),
                float(imputation_cfg["w_cat"]),
            )
        else:
            trainer.test()
    finally:
        logger.finish()
    return evaluation_dir
