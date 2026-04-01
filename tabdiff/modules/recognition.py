"""Recognition network for the variational latent variable v."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tabdiff.modules.transformer import Tokenizer, Transformer


class RecognitionMLPEncoder(nn.Module):
    """Stack of Linear / LayerNorm / SiLU (original recognition encoder)."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        layers: list[nn.Module] = []
        d = input_dim
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(d, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                ]
            )
            d = hidden_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RecognitionUniModBody(nn.Module):
    """Tokenizer → token sequence → optional time broadcast → Transformer → pool → hidden."""

    def __init__(
        self,
        num_numerical_features: int,
        num_classes_per_column: list,
        posterior_inputs: str,
        d_token: int,
        num_layers: int,
        n_head: int,
        factor: int,
        bias: bool,
        time_embed_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.posterior_inputs = posterior_inputs
        self.vocab_sizes: List[int] = [k + 1 for k in num_classes_per_column]
        tok_cats = self.vocab_sizes if self.vocab_sizes else None
        self.tokenizer = Tokenizer(num_numerical_features, tok_cats, d_token, bias)
        self.encoder = Transformer(
            num_layers, d_token, n_head, d_token, factor
        )
        self.time_embed = (
            _SinusoidalEmbedding(time_embed_dim)
            if posterior_inputs != "x0"
            else None
        )
        self.time_to_token = (
            nn.Linear(time_embed_dim, d_token)
            if self.time_embed is not None
            else None
        )
        self.out = nn.Linear(d_token, hidden_dim)

    def _seq(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        # Tokenizer expects concatenated one-hot (same as denoiser), not int indices.
        if not self.vocab_sizes:
            oh = None
        else:
            oh = torch.cat(
                [
                    F.one_hot(x_cat[:, i].long(), num_classes=self.vocab_sizes[i]).to(
                        x_num.dtype
                    )
                    for i in range(len(self.vocab_sizes))
                ],
                dim=-1,
            )
        e = self.tokenizer(x_num, oh)
        return e[:, 1:, :]

    def forward(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
        x_num_t: torch.Tensor,
        x_cat_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        if self.posterior_inputs == "x0_xt_t":
            h = torch.cat(
                [self._seq(x_num, x_cat), self._seq(x_num_t, x_cat_t)], dim=1
            )
        elif self.posterior_inputs == "xt_t":
            h = self._seq(x_num_t, x_cat_t)
        elif self.posterior_inputs == "x0_t":
            h = self._seq(x_num, x_cat)
        else:
            h = self._seq(x_num, x_cat)

        if self.time_embed is not None and self.time_to_token is not None:
            te = self.time_to_token(self.time_embed(t))
            h = h + te.unsqueeze(1)

        h = self.encoder(h)
        return self.out(h.mean(dim=1))


class RecognitionUniModTransformerHeadsBody(nn.Module):
    """Tokenizer → shared Transformer body → separate Transformer heads → mean-pool features.

    Generalizes :class:`RecognitionUniModBody` by replacing the single linear map to
    ``hidden_dim`` with two Transformer stacks whose outputs are mean-pooled to
    ``d_token``-dim vectors (then :class:`RecognitionModel` applies ``mu_head`` /
    ``logvar_head`` linear projections to ``latent_dim``).
    """

    def __init__(
        self,
        num_numerical_features: int,
        num_classes_per_column: list,
        posterior_inputs: str,
        d_token: int,
        body_num_layers: int,
        mu_head_num_layers: int,
        log_var_head_num_layers: int,
        n_head: int,
        factor: int,
        bias: bool,
        time_embed_dim: int,
    ):
        super().__init__()
        self.posterior_inputs = posterior_inputs
        self.vocab_sizes: List[int] = [k + 1 for k in num_classes_per_column]
        tok_cats = self.vocab_sizes if self.vocab_sizes else None
        self.tokenizer = Tokenizer(num_numerical_features, tok_cats, d_token, bias)
        self.body = Transformer(
            body_num_layers, d_token, n_head, d_token, factor
        )
        self.mu_head_transformer = Transformer(
            mu_head_num_layers, d_token, n_head, d_token, factor
        )
        self.log_var_head_transformer = Transformer(
            log_var_head_num_layers, d_token, n_head, d_token, factor
        )
        self.time_embed = (
            _SinusoidalEmbedding(time_embed_dim)
            if posterior_inputs != "x0"
            else None
        )
        self.time_to_token = (
            nn.Linear(time_embed_dim, d_token)
            if self.time_embed is not None
            else None
        )

    def _seq(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        if not self.vocab_sizes:
            oh = None
        else:
            oh = torch.cat(
                [
                    F.one_hot(x_cat[:, i].long(), num_classes=self.vocab_sizes[i]).to(
                        x_num.dtype
                    )
                    for i in range(len(self.vocab_sizes))
                ],
                dim=-1,
            )
        e = self.tokenizer(x_num, oh)
        return e[:, 1:, :]

    def forward(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
        x_num_t: torch.Tensor,
        x_cat_t: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.posterior_inputs == "x0_xt_t":
            h = torch.cat(
                [self._seq(x_num, x_cat), self._seq(x_num_t, x_cat_t)], dim=1
            )
        elif self.posterior_inputs == "xt_t":
            h = self._seq(x_num_t, x_cat_t)
        elif self.posterior_inputs == "x0_t":
            h = self._seq(x_num, x_cat)
        else:
            h = self._seq(x_num, x_cat)

        if self.time_embed is not None and self.time_to_token is not None:
            te = self.time_to_token(self.time_embed(t))
            h = h + te.unsqueeze(1)

        h = self.body(h)
        h_mu = self.mu_head_transformer(h)
        h_lv = self.log_var_head_transformer(h)
        return h_mu.mean(dim=1), h_lv.mean(dim=1)


class RecognitionModel(nn.Module):
    """Recognition network r_phi(v | x_num, x_cat) with pluggable encoder backbone."""

    BACKBONES = frozenset({"mlp", "unimod_mlp", "unimod_transformer_heads"})

    def __init__(
        self,
        num_numerical_features: int,
        num_classes_per_column: list,
        latent_dim: int,
        backbone: str = "mlp",
        backbone_params: Optional[Dict] = None,
        posterior_inputs: Literal["x0_xt_t", "xt_t", "x0_t", "x0"] = "x0",
        min_logvar: float = -10.0,
        max_logvar: float = 2.0,
        **kwargs,
    ):
        """
        Args:
            backbone: ``mlp`` (flat MLP), ``unimod_mlp`` (Tokenizer + Transformer +
                linear to hidden), or ``unimod_transformer_heads`` (Tokenizer + body
                Transformer + separate Transformer heads for mu / log-var features).
            backbone_params: Hyperparameters for the chosen backbone (merged with
                any extra ``**kwargs`` for backward compatibility).
        """
        super().__init__()
        params = {**(backbone_params or {}), **kwargs}
        if backbone not in self.BACKBONES:
            raise ValueError(
                f"Unknown recognition backbone {backbone!r}; expected one of {sorted(self.BACKBONES)}"
            )
        self.backbone_name = backbone
        self.num_numerical_features = num_numerical_features
        self.num_cat_cols = len(num_classes_per_column)
        self.latent_dim = latent_dim
        self.posterior_inputs = posterior_inputs
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar

        hidden_dim = int(params.get("hidden_dim", 128))
        num_layers = int(params.get("num_layers", 2))
        time_embed_dim = int(params.get("time_embed_dim", 64))
        cat_embed_dim = int(params.get("cat_embed_dim", 32))

        state_dim = num_numerical_features + self.num_cat_cols * cat_embed_dim
        if posterior_inputs == "x0_xt_t":
            flat_dim = state_dim * 2 + time_embed_dim
        elif posterior_inputs in ("xt_t", "x0_t"):
            flat_dim = state_dim + time_embed_dim
        elif posterior_inputs == "x0":
            flat_dim = state_dim
        else:
            raise ValueError(f"Unknown posterior_inputs: {posterior_inputs}")

        if posterior_inputs != "x0":
            self.time_embed = _SinusoidalEmbedding(time_embed_dim)
        else:
            self.time_embed = None

        if num_classes_per_column:
            self.cat_embeds = nn.ModuleList(
                [
                    nn.Embedding(k + 1, cat_embed_dim)
                    for k in num_classes_per_column
                ]
            )
        else:
            self.cat_embeds = None

        if backbone == "mlp":
            self.body = RecognitionMLPEncoder(flat_dim, hidden_dim, num_layers)
            feat_dim = hidden_dim
        elif backbone == "unimod_transformer_heads":
            d_token = int(params.get("d_token", 4))
            n_head = int(params.get("n_head", 1))
            factor = int(params.get("factor", 32))
            bias = bool(params.get("bias", True))
            body_nl = int(params.get("body_num_layers", params.get("unimod_num_layers", num_layers)))
            mu_nl = int(params.get("mu_head_num_layers", 2))
            lv_nl = int(
                params.get(
                    "log_var_head_num_layers",
                    params.get("logvar_head_num_layers", 2),
                )
            )
            self.body = RecognitionUniModTransformerHeadsBody(
                num_numerical_features,
                num_classes_per_column,
                posterior_inputs,
                d_token,
                body_nl,
                mu_nl,
                lv_nl,
                n_head,
                factor,
                bias,
                time_embed_dim,
            )
            feat_dim = d_token
        else:
            d_token = int(params.get("d_token", 4))
            n_head = int(params.get("n_head", 1))
            factor = int(params.get("factor", 32))
            bias = bool(params.get("bias", True))
            um_layers = int(params.get("unimod_num_layers", num_layers))
            self.body = RecognitionUniModBody(
                num_numerical_features,
                num_classes_per_column,
                posterior_inputs,
                d_token,
                um_layers,
                n_head,
                factor,
                bias,
                time_embed_dim,
                hidden_dim,
            )
            feat_dim = hidden_dim

        self.mu_head = nn.Linear(feat_dim, latent_dim)
        self.logvar_head = nn.Linear(feat_dim, latent_dim)

    def _embed_state(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        parts = [x_num] if self.num_numerical_features > 0 else []
        if self.cat_embeds is not None and x_cat.shape[1] > 0:
            for i, embed in enumerate(self.cat_embeds):
                parts.append(embed(x_cat[:, i]))
        return torch.cat(parts, dim=-1) if parts else x_num

    def _flat_inputs(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
        x_num_t: torch.Tensor,
        x_cat_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        if self.posterior_inputs == "x0_xt_t":
            assert self.time_embed is not None
            t_emb = self.time_embed(t)
            state_0 = self._embed_state(x_num, x_cat)
            state_t = self._embed_state(x_num_t, x_cat_t)
            return torch.cat([state_0, state_t, t_emb], dim=-1)
        if self.posterior_inputs == "xt_t":
            assert self.time_embed is not None
            t_emb = self.time_embed(t)
            state_t = self._embed_state(x_num_t, x_cat_t)
            return torch.cat([state_t, t_emb], dim=-1)
        if self.posterior_inputs == "x0_t":
            assert self.time_embed is not None
            t_emb = self.time_embed(t)
            state_0 = self._embed_state(x_num, x_cat)
            return torch.cat([state_0, t_emb], dim=-1)
        return self._embed_state(x_num, x_cat)

    def forward(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
        x_num_t: torch.Tensor,
        x_cat_t: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.backbone_name == "unimod_transformer_heads":
            h_mu, h_lv = self.body(x_num, x_cat, x_num_t, x_cat_t, t)
            mu = self.mu_head(h_mu)
            logvar = self.logvar_head(h_lv)
        elif self.backbone_name == "unimod_mlp":
            h = self.body(x_num, x_cat, x_num_t, x_cat_t, t)
            mu = self.mu_head(h)
            logvar = self.logvar_head(h)
        else:
            inp = self._flat_inputs(x_num, x_cat, x_num_t, x_cat_t, t)
            h = self.body(inp)
            mu = self.mu_head(h)
            logvar = self.logvar_head(h)
        logvar = torch.clamp(logvar, self.min_logvar, self.max_logvar)
        return mu, logvar

    def sample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
        return kl.sum(dim=-1)


class LatentPolicy(nn.Module):
    """How the latent v is drawn for training and generation."""

    def __init__(
        self,
        policy_type: Literal["fresh", "prior_only", "consistency"],
        latent_dim: int,
        recognition_model: Optional[RecognitionModel] = None,
    ):
        super().__init__()
        self.policy_type = policy_type
        self.latent_dim = latent_dim
        self.recognition = recognition_model
        self._cached_v: Optional[torch.Tensor] = None

    def sample_for_training(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
        x_num_t: torch.Tensor,
        x_cat_t: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, device = x_num.shape[0], x_num.device
        if self.policy_type == "prior_only":
            v = torch.randn(B, self.latent_dim, device=device)
            kl = torch.zeros(B, device=device)
        else:
            if self.recognition is None:
                raise ValueError("Recognition model required for non-prior policy")
            mu, logvar = self.recognition(x_num, x_cat, x_num_t, x_cat_t, t)
            v = self.recognition.sample(mu, logvar)
            kl = self.recognition.kl_divergence(mu, logvar)
        return v, kl

    def sample_for_generation(
        self,
        batch_size: int,
        device: torch.device,
        step_idx: int = 0,
    ) -> torch.Tensor:
        if self.policy_type == "consistency":
            if step_idx == 0 or self._cached_v is None:
                self._cached_v = torch.randn(
                    batch_size, self.latent_dim, device=device
                )
            return self._cached_v
        return torch.randn(batch_size, self.latent_dim, device=device)

    def reset_cache(self) -> None:
        self._cached_v = None


class _SinusoidalEmbedding(nn.Module):
    def __init__(self, embed_dim: int, scale: float = 1000.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = scale
        half = embed_dim // 2
        if half < 1:
            half = 1
        freqs = torch.exp(
            -torch.arange(half, dtype=torch.float32)
            * (torch.log(torch.tensor(10000.0)) / half)
        )
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float() * self.scale
        args = t.unsqueeze(-1) * self.freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.embed_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb
