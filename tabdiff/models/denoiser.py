from __future__ import annotations

import torch
import torch.nn as nn


class PreconditionedDenoiser(nn.Module):
    def __init__(self, denoise_fn, sigma_data: float = 0.5, net_conditioning: str = "sigma"):
        super().__init__()
        self.sigma_data = sigma_data
        self.net_conditioning = net_conditioning
        self.denoise_fn = denoise_fn

    def forward(self, x_num, x_cat, t, sigma, v=None):
        x_num = x_num.to(torch.float32)
        sigma = sigma.to(torch.float32)
        if sigma.dim() > 1:
            sigma_cond = (0.002 ** (1 / 7) + t * (80 ** (1 / 7) - 0.002 ** (1 / 7))).pow(7)
        else:
            sigma_cond = sigma
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma_cond.log() / 4
        x_in = c_in * x_num
        if self.net_conditioning == "sigma":
            f_x, x_cat_pred = self.denoise_fn(x_in, x_cat, c_noise.flatten(), v=v)
        elif self.net_conditioning == "t":
            f_x, x_cat_pred = self.denoise_fn(x_in, x_cat, t, v=v)
        else:
            raise ValueError(f"Unknown net_conditioning: {self.net_conditioning}")
        d_x = c_skip * x_num + c_out * f_x.to(torch.float32)
        return d_x, x_cat_pred


class Model(nn.Module):
    def __init__(
        self,
        denoise_fn,
        sigma_data: float = 0.5,
        precond: bool = False,
        net_conditioning: str = "sigma",
        **_: object,
    ):
        super().__init__()
        self.precond = precond
        self.denoiser = (
            PreconditionedDenoiser(denoise_fn, sigma_data=sigma_data, net_conditioning=net_conditioning)
            if precond
            else denoise_fn
        )

    def forward(self, x_num, x_cat, t, sigma=None, v=None):
        if self.precond:
            return self.denoiser(x_num, x_cat, t, sigma, v=v)
        return self.denoiser(x_num, x_cat, t, v=v)

