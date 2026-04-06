from __future__ import annotations

from typing import Any

from tabdiff.models.backbones import MLPBackbone, TransformerEncoderBackbone, UniModMLP
from tabdiff.models.recognition import RecognitionModel


DENOISER_REGISTRY = {
    "unimod_mlp": UniModMLP,
    "transformer_encoder": TransformerEncoderBackbone,
    "mlp": MLPBackbone,
}


def build_denoiser_backbone(
    denoiser_config: dict[str, Any],
    *,
    d_numerical: int,
    categories: list[int],
    latent_dim: int,
):
    backbone = denoiser_config["backbone"]
    if backbone not in DENOISER_REGISTRY:
        raise ValueError(f"Unknown denoiser backbone: {backbone!r}")
    params = dict(denoiser_config.get("params", {}))
    params["d_numerical"] = d_numerical
    params["categories"] = categories
    params["latent_dim"] = latent_dim
    return DENOISER_REGISTRY[backbone](**params)


def build_recognition_model(
    variational_config: dict[str, Any],
    *,
    num_numerical_features: int,
    num_classes_per_column: list[int],
):
    recognition = variational_config["recognition"]
    return RecognitionModel(
        num_numerical_features=num_numerical_features,
        num_classes_per_column=num_classes_per_column,
        latent_dim=int(variational_config["latent_dim"]),
        backbone=recognition["backbone"],
        backbone_params=dict(recognition.get("params", {})),
        posterior_inputs=variational_config.get("posterior_inputs", "x0"),
    )
