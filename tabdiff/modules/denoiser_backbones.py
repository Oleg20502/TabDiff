"""Denoiser backbone variants: UniMod (encoder–MLP–decoder), encoder-only transformer, flat MLP."""

from __future__ import annotations

import torch
import torch.nn as nn

from tabdiff.modules.main_modules import PositionalEmbedding
from tabdiff.modules.transformer import Reconstructor, Tokenizer, Transformer


class TransformerEncoderBackbone(nn.Module):
    """Tokenizer → N×Transformer (no MLPDiffusion bottleneck, no decoder stack) → Reconstructor.

    Time / latent conditioning is added as a broadcast embedding to all tokens before the stack.
    """

    def __init__(
        self,
        d_numerical: int,
        categories: list,
        num_layers: int,
        d_token: int,
        n_head: int = 1,
        factor: int = 4,
        bias: bool = True,
        latent_dim: int = 0,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            pass  # absorb legacy keys (e.g. dim_t, use_mlp from merged unimod config)
        self.d_numerical = d_numerical
        self.categories = categories
        self.latent_dim = latent_dim

        self.tokenizer = Tokenizer(d_numerical, categories, d_token, bias=bias)
        self.encoder = Transformer(
            num_layers, d_token, n_head, d_token, factor
        )
        self.detokenizer = Reconstructor(d_numerical, categories, d_token)

        self.map_noise = PositionalEmbedding(num_channels=d_token)
        self.time_embed = nn.Sequential(
            nn.Linear(d_token, d_token),
            nn.SiLU(),
            nn.Linear(d_token, d_token),
        )
        self.latent_proj = (
            nn.Sequential(nn.Linear(latent_dim, d_token), nn.SiLU())
            if latent_dim > 0
            else None
        )

    def _cond_vec(self, timesteps: torch.Tensor, v: torch.Tensor | None) -> torch.Tensor:
        t = timesteps.float().flatten()
        emb = self.map_noise(t)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
        emb = self.time_embed(emb)
        if v is not None and self.latent_proj is not None:
            emb = emb + self.latent_proj(v)
        return emb

    def forward(self, x_num, x_cat, timesteps, v=None):
        e = self.tokenizer(x_num, x_cat)
        h = e[:, 1:, :]
        cond = self._cond_vec(timesteps, v)
        h = h + cond.unsqueeze(1)
        h = self.encoder(h)
        x_num_pred, x_cat_pred = self.detokenizer(h)
        x_cat_pred = (
            torch.cat(x_cat_pred, dim=-1)
            if len(x_cat_pred) > 0
            else torch.zeros_like(x_cat).to(x_num_pred.dtype)
        )
        return x_num_pred, x_cat_pred


class MLPBackbone(nn.Module):
    """Flat numerical + per-column categorical embeddings, MLP, separate num / cat heads."""

    def __init__(
        self,
        d_numerical: int,
        categories: list,
        hidden_dim: int = 256,
        num_layers: int = 4,
        time_embed_dim: int = 64,
        cat_embed_dim: int = 32,
        latent_dim: int = 0,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            pass
        self.d_numerical = d_numerical
        self.categories = list(categories)
        self.latent_dim = latent_dim
        self.num_cat_cols = len(self.categories)

        # x_cat from diffusion is concatenated one-hot / soft probs per column
        # (width = sum(categories[j]), same layout as Tokenizer), not int indices.
        state_dim = d_numerical + self.num_cat_cols * cat_embed_dim
        in_dim = state_dim + time_embed_dim + latent_dim

        if self.num_cat_cols > 0:
            self.cat_projs = nn.ModuleList(
                [nn.Linear(k, cat_embed_dim) for k in self.categories]
            )
        else:
            self.cat_projs = None

        self.time_embed = _SinusoidalEmbedding(time_embed_dim)

        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(d, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                ]
            )
            d = hidden_dim
        self.mlp = nn.Sequential(*layers)

        self.num_head = (
            nn.Linear(hidden_dim, d_numerical) if d_numerical > 0 else None
        )
        self.cat_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, k) for k in self.categories]
        )

    def forward(self, x_num, x_cat, timesteps, v=None):
        parts = []
        if self.d_numerical > 0:
            parts.append(x_num)
        if self.cat_projs is not None and x_cat.shape[1] > 0:
            cum = 0
            for j, proj in enumerate(self.cat_projs):
                k = self.categories[j]
                seg = x_cat[:, cum : cum + k].to(proj.weight.dtype)
                parts.append(proj(seg))
                cum += k
        h = torch.cat(parts, dim=-1) if parts else x_num

        t_emb = self.time_embed(timesteps)
        chunks = [h, t_emb]
        if self.latent_dim > 0:
            if v is None:
                raise ValueError("latent_dim > 0 but v is None in MLPBackbone.forward")
            chunks.append(v)
        h = torch.cat(chunks, dim=-1)
        h = self.mlp(h)

        if self.num_head is not None:
            x_num_pred = self.num_head(h)
        else:
            x_num_pred = torch.zeros(h.shape[0], 0, device=h.device, dtype=h.dtype)

        if len(self.cat_heads) > 0:
            x_cat_pred = torch.cat([head(h) for head in self.cat_heads], dim=-1)
        else:
            x_cat_pred = torch.zeros_like(x_cat).to(h.dtype)

        return x_num_pred, x_cat_pred


class _SinusoidalEmbedding(nn.Module):
    def __init__(self, embed_dim: int, scale: float = 1000.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = scale
        half = max(embed_dim // 2, 1)
        freqs = torch.exp(
            -torch.arange(half, dtype=torch.float32)
            * (torch.log(torch.tensor(10000.0)) / half)
        )
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float().flatten() * self.scale
        args = t.unsqueeze(-1) * self.freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.embed_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


__all__ = [
    "MLPBackbone",
    "TransformerEncoderBackbone",
]
