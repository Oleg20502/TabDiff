from __future__ import annotations

import torch
import torch.nn as nn

from tabdiff.models.backbones.common import PositionalEmbedding
from tabdiff.modules.transformer import Reconstructor, Tokenizer, Transformer


class TransformerEncoderBackbone(nn.Module):
    def __init__(
        self,
        d_numerical: int,
        categories: list[int],
        num_layers: int,
        d_token: int,
        n_head: int = 1,
        factor: int = 4,
        bias: bool = True,
        latent_dim: int = 0,
        **_: object,
    ):
        super().__init__()
        self.d_numerical = d_numerical
        self.categories = categories
        self.latent_dim = latent_dim
        self.tokenizer = Tokenizer(d_numerical, categories, d_token, bias=bias)
        self.encoder = Transformer(num_layers, d_token, n_head, d_token, factor)
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
        emb = self.map_noise(timesteps.float().flatten())
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
        emb = self.time_embed(emb)
        if v is not None and self.latent_proj is not None:
            emb = emb + self.latent_proj(v)
        return emb

    def forward(self, x_num, x_cat, timesteps, v=None):
        e = self.tokenizer(x_num, x_cat)
        h = e[:, 1:, :]
        h = h + self._cond_vec(timesteps, v).unsqueeze(1)
        h = self.encoder(h)
        x_num_pred, x_cat_pred = self.detokenizer(h)
        x_cat_pred = (
            torch.cat(x_cat_pred, dim=-1)
            if len(x_cat_pred) > 0
            else torch.zeros_like(x_cat).to(x_num_pred.dtype)
        )
        return x_num_pred, x_cat_pred

