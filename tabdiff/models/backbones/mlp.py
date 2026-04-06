from __future__ import annotations

import torch
import torch.nn as nn

from tabdiff.models.backbones.common import SinusoidalEmbedding


class MLPBackbone(nn.Module):
    def __init__(
        self,
        d_numerical: int,
        categories: list[int],
        hidden_dim: int = 256,
        num_layers: int = 4,
        time_embed_dim: int = 64,
        cat_embed_dim: int = 32,
        latent_dim: int = 0,
        **_: object,
    ):
        super().__init__()
        self.d_numerical = d_numerical
        self.categories = list(categories)
        self.latent_dim = latent_dim
        self.num_cat_cols = len(self.categories)
        state_dim = d_numerical + self.num_cat_cols * cat_embed_dim
        in_dim = state_dim + time_embed_dim + latent_dim
        if self.num_cat_cols > 0:
            self.cat_projs = nn.ModuleList([nn.Linear(k, cat_embed_dim) for k in self.categories])
        else:
            self.cat_projs = None
        self.time_embed = SinusoidalEmbedding(time_embed_dim)

        layers: list[nn.Module] = []
        current_dim = in_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(current_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()])
            current_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)
        self.num_head = nn.Linear(hidden_dim, d_numerical) if d_numerical > 0 else None
        self.cat_heads = nn.ModuleList([nn.Linear(hidden_dim, k) for k in self.categories])

    def forward(self, x_num, x_cat, timesteps, v=None):
        parts = []
        if self.d_numerical > 0:
            parts.append(x_num)
        if self.cat_projs is not None and x_cat.shape[1] > 0:
            offset = 0
            for index, proj in enumerate(self.cat_projs):
                width = self.categories[index]
                parts.append(proj(x_cat[:, offset : offset + width].to(proj.weight.dtype)))
                offset += width
        h = torch.cat(parts, dim=-1) if parts else x_num
        chunks = [h, self.time_embed(timesteps)]
        if self.latent_dim > 0:
            if v is None:
                raise ValueError("latent_dim > 0 but no latent sample was provided.")
            chunks.append(v)
        h = self.mlp(torch.cat(chunks, dim=-1))
        x_num_pred = (
            self.num_head(h)
            if self.num_head is not None
            else torch.zeros(h.shape[0], 0, device=h.device, dtype=h.dtype)
        )
        x_cat_pred = (
            torch.cat([head(h) for head in self.cat_heads], dim=-1)
            if len(self.cat_heads) > 0
            else torch.zeros_like(x_cat).to(h.dtype)
        )
        return x_num_pred, x_cat_pred

