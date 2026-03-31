"""Construct denoiser and recognition modules from merged TOML / pickle config."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from tabdiff.modules.denoiser_backbones import MLPBackbone, TransformerEncoderBackbone
from tabdiff.modules.main_modules import UniModMLP
from tabdiff.modules.recognition import RecognitionModel

DENOISER_REGISTRY = {
    "unimod_mlp": UniModMLP,
    "transformer_encoder": TransformerEncoderBackbone,
    "mlp": MLPBackbone,
}


def merged_denoiser_params(raw_config: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    spec = raw_config.get("denoiser") or {}
    backbone = spec.get("backbone", "unimod_mlp")
    if backbone not in DENOISER_REGISTRY:
        raise ValueError(
            f"Unknown denoiser.backbone {backbone!r}; "
            f"expected one of {list(DENOISER_REGISTRY)}."
        )
    base = dict(raw_config.get("unimodmlp_params") or {})
    extra = spec.get(backbone)
    if not isinstance(extra, dict):
        extra = {}
    return backbone, {**base, **extra}


def build_denoiser_backbone(
    raw_config: Dict[str, Any],
    *,
    d_numerical: int,
    categories: List[int],
    latent_dim: int,
):
    backbone, params = merged_denoiser_params(raw_config)
    params = dict(params)
    params["d_numerical"] = d_numerical
    params["categories"] = categories
    params["latent_dim"] = latent_dim
    cls = DENOISER_REGISTRY[backbone]
    return cls(**params)


def _variational_recognition_backbone(var_cfg: Dict[str, Any]) -> str:
    return var_cfg.get("recognition_backbone", "mlp")


def _merged_recognition_params(var_cfg: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    backbone = _variational_recognition_backbone(var_cfg)
    base = dict(var_cfg.get("recognition_params") or {})

    if backbone == "transformer_encoder":
        backbone = "unimod_mlp"
        te = var_cfg.get("recognition_transformer_encoder")
        um = var_cfg.get("recognition_unimod_mlp")
        extra_te = dict(te) if isinstance(te, dict) else {}
        extra_um = dict(um) if isinstance(um, dict) else {}
        extra = {**extra_te, **extra_um}
        if "unimod_num_layers" not in extra and "transformer_layers" in extra_te:
            extra["unimod_num_layers"] = extra_te["transformer_layers"]
    else:
        key = f"recognition_{backbone}"
        extra = var_cfg.get(key)
        if not isinstance(extra, dict):
            extra = {}

    if backbone not in RecognitionModel.BACKBONES:
        raise ValueError(
            f"Unknown variational.recognition_backbone {backbone!r}; "
            f"expected one of {sorted(RecognitionModel.BACKBONES)}."
        )
    return backbone, {**base, **extra}


def build_recognition_model(
    var_cfg: Dict[str, Any],
    *,
    num_numerical_features: int,
    num_classes_per_column: List[int],
    latent_dim: int,
    posterior_inputs: str,
) -> RecognitionModel:
    backbone, params = _merged_recognition_params(var_cfg)
    params = dict(params)
    return RecognitionModel(
        num_numerical_features=num_numerical_features,
        num_classes_per_column=num_classes_per_column,
        latent_dim=latent_dim,
        posterior_inputs=posterior_inputs,
        backbone=backbone,
        backbone_params=params,
    )


def ensure_denoiser_config_inplace(raw_config: Dict[str, Any]) -> None:
    """If [denoiser] is missing (legacy TOML), default to unimod_mlp for saved runs."""
    if "denoiser" not in raw_config or not isinstance(raw_config["denoiser"], dict):
        raw_config["denoiser"] = {"backbone": "unimod_mlp"}
    elif "backbone" not in raw_config["denoiser"]:
        raw_config["denoiser"]["backbone"] = "unimod_mlp"
