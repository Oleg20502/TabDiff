from __future__ import annotations

import torch
import torch.nn as nn

from tabdiff.models.backbones.common import PositionalEmbedding
from tabdiff.modules.transformer import Reconstructor, Tokenizer, Transformer


class MLPDiffusion(nn.Module):
    def __init__(self, d_in: int, dim_t: int = 512, use_mlp: bool = True, latent_dim: int = 0):
        super().__init__()
        self.proj = nn.Linear(d_in, dim_t)
        self.mlp = (
            nn.Sequential(
                nn.Linear(dim_t, dim_t * 2),
                nn.SiLU(),
                nn.Linear(dim_t * 2, dim_t * 2),
                nn.SiLU(),
                nn.Linear(dim_t * 2, dim_t),
                nn.SiLU(),
                nn.Linear(dim_t, d_in),
            )
            if use_mlp
            else nn.Linear(dim_t, d_in)
        )
        self.map_noise = PositionalEmbedding(num_channels=dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t),
        )
        self.latent_proj = (
            nn.Sequential(nn.Linear(latent_dim, dim_t), nn.SiLU())
            if latent_dim > 0
            else None
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, v: torch.Tensor | None = None) -> torch.Tensor:
        emb = self.map_noise(timesteps)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
        emb = self.time_embed(emb)
        if v is not None and self.latent_proj is not None:
            emb = emb + self.latent_proj(v)
        x = self.proj(x) + emb
        return self.mlp(x)


class UniModMLP(nn.Module):
    def __init__(
        self,
        d_numerical: int,
        categories: list[int],
        num_layers: int,
        d_token: int,
        n_head: int = 1,
        factor: int = 4,
        bias: bool = True,
        dim_t: int = 512,
        use_mlp: bool = True,
        latent_dim: int = 0,
        **_: object,
    ):
        super().__init__()
        self.d_numerical = d_numerical
        self.categories = categories
        self.tokenizer = Tokenizer(d_numerical, categories, d_token, bias=bias)
        self.encoder = Transformer(num_layers, d_token, n_head, d_token, factor)
        d_in = d_token * (d_numerical + len(categories))
        self.mlp = MLPDiffusion(d_in, dim_t=dim_t, use_mlp=use_mlp, latent_dim=latent_dim)
        self.decoder = Transformer(num_layers, d_token, n_head, d_token, factor)
        self.detokenizer = Reconstructor(d_numerical, categories, d_token)

    def forward(self, x_num, x_cat, timesteps, v=None):
        e = self.tokenizer(x_num, x_cat)
        decoder_input = e[:, 1:, :]
        y = self.encoder(decoder_input)
        pred_y = self.mlp(y.reshape(y.shape[0], -1), timesteps, v=v)
        pred_e = self.decoder(pred_y.reshape(*y.shape))
        x_num_pred, x_cat_pred = self.detokenizer(pred_e)
        x_cat_pred = (
            torch.cat(x_cat_pred, dim=-1)
            if len(x_cat_pred) > 0
            else torch.zeros_like(x_cat).to(x_num_pred.dtype)
        )
        return x_num_pred, x_cat_pred

