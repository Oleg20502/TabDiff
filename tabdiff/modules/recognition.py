"""Recognition network for the variational latent variable v.

Ports the VA-DDPM RecognitionModel and LatentPolicy into TabDiff's interface.
Key difference from VA-DDPM: TabDiff categorical columns have K_j original
classes + 1 mask token at index K_j, so embeddings use vocabulary size K_j+1.
"""

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn


class RecognitionModel(nn.Module):
    """Recognition network r_phi(v | x_num, x_cat).

    Produces parameters (mu, logvar) of a diagonal Gaussian posterior over
    the latent variable v that will condition the denoiser.

    Inputs can be configured via `posterior_inputs`:
      - "x0"      : clean data only (x_num, x_cat)
      - "x0_t"    : clean data + time embedding
      - "xt_t"    : noisy data + time embedding
      - "x0_xt_t" : clean data + noisy data + time embedding
    """

    def __init__(
        self,
        num_numerical_features: int,
        num_classes_per_column: list,
        latent_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        time_embed_dim: int = 64,
        cat_embed_dim: int = 32,
        posterior_inputs: Literal["x0_xt_t", "xt_t", "x0_t", "x0"] = "x0",
        min_logvar: float = -10.0,
        max_logvar: float = 2.0,
    ):
        """
        Args:
            num_numerical_features: Number of continuous columns.
            num_classes_per_column: Original class counts K_j per categorical
                column (without the mask token).  Embeddings use K_j+1 to
                accommodate TabDiff's mask token at index K_j.
            latent_dim: Dimension of latent variable v.
            hidden_dim: Width of the MLP encoder.
            num_layers: Depth of the MLP encoder.
            time_embed_dim: Dimension of the sinusoidal time embedding.
            cat_embed_dim: Embedding dimension per categorical column.
            posterior_inputs: Which inputs are fed into the encoder.
            min_logvar / max_logvar: Clamp range for log-variance.
        """
        super().__init__()
        self.num_numerical_features = num_numerical_features
        self.num_cat_cols = len(num_classes_per_column)
        self.latent_dim = latent_dim
        self.posterior_inputs = posterior_inputs
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar

        # Each state (x_num or x_num_t) contributes:
        #   num_numerical_features  (continuous, passed directly)
        # + num_cat_cols * cat_embed_dim  (categorical embeddings)
        state_dim = num_numerical_features + self.num_cat_cols * cat_embed_dim

        if posterior_inputs == "x0_xt_t":
            input_dim = state_dim * 2 + time_embed_dim
        elif posterior_inputs in ("xt_t", "x0_t"):
            input_dim = state_dim + time_embed_dim
        elif posterior_inputs == "x0":
            input_dim = state_dim
        else:
            raise ValueError(f"Unknown posterior_inputs: {posterior_inputs}")

        # Sinusoidal time embedding (only built when needed)
        if posterior_inputs != "x0":
            self.time_embed = _SinusoidalEmbedding(time_embed_dim)
        else:
            self.time_embed = None

        # Per-column categorical embeddings.
        # Vocabulary size = K_j + 1 to cover the mask token at index K_j.
        if num_classes_per_column:
            self.cat_embeds = nn.ModuleList([
                nn.Embedding(k + 1, cat_embed_dim)
                for k in num_classes_per_column
            ])
        else:
            self.cat_embeds = None

        # MLP encoder
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            ])
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)

        # Output heads
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_state(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """Embed a single tabular state into a flat vector.

        Args:
            x_num: Continuous features [B, num_numerical_features]
            x_cat: Categorical integer indices [B, num_cat_cols]

        Returns:
            Embedded state [B, state_dim]
        """
        parts = [x_num] if self.num_numerical_features > 0 else []
        if self.cat_embeds is not None and x_cat.shape[1] > 0:
            for i, embed in enumerate(self.cat_embeds):
                parts.append(embed(x_cat[:, i]))  # [B, cat_embed_dim]
        return torch.cat(parts, dim=-1) if parts else x_num

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
        x_num_t: torch.Tensor,
        x_cat_t: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute posterior parameters (mu, logvar).

        Args:
            x_num:   Clean continuous features  [B, num_numerical_features]
            x_cat:   Clean categorical indices  [B, num_cat_cols]
            x_num_t: Noisy continuous features  [B, num_numerical_features]
            x_cat_t: Noisy categorical indices  [B, num_cat_cols]  (may contain mask tokens)
            t:       Time values                [B]  in [0, 1]

        Returns:
            mu     [B, latent_dim]
            logvar [B, latent_dim]
        """
        if self.posterior_inputs == "x0_xt_t":
            t_emb = self.time_embed(t)
            state_0 = self._embed_state(x_num, x_cat)
            state_t = self._embed_state(x_num_t, x_cat_t)
            inputs = torch.cat([state_0, state_t, t_emb], dim=-1)
        elif self.posterior_inputs == "xt_t":
            t_emb = self.time_embed(t)
            state_t = self._embed_state(x_num_t, x_cat_t)
            inputs = torch.cat([state_t, t_emb], dim=-1)
        elif self.posterior_inputs == "x0_t":
            t_emb = self.time_embed(t)
            state_0 = self._embed_state(x_num, x_cat)
            inputs = torch.cat([state_0, t_emb], dim=-1)
        else:  # "x0"
            inputs = self._embed_state(x_num, x_cat)

        h = self.encoder(inputs)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        logvar = torch.clamp(logvar, self.min_logvar, self.max_logvar)
        return mu, logvar

    def sample(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Reparameterisation trick: v = mu + exp(0.5*logvar) * eps.

        Args:
            mu:     [B, latent_dim]
            logvar: [B, latent_dim]

        Returns:
            Sampled latent [B, latent_dim]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def kl_divergence(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """KL( N(mu, sigma^2) || N(0,1) ) summed over latent dim.

        Args:
            mu:     [B, latent_dim]
            logvar: [B, latent_dim]

        Returns:
            Per-sample KL [B]
        """
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
        return kl.sum(dim=-1)


class LatentPolicy(nn.Module):
    """Abstracts how the latent variable v is used during training and sampling.

    Modes:
      - "shared"     : Recognition model is used during training; prior at sampling.
      - "prior_only" : Always sample from N(0,I); no KL loss.
      - "consistency": Like "shared" at training; v sampled once and reused
                       across all reverse steps at generation time.
    """

    def __init__(
        self,
        policy_type: Literal["shared", "prior_only", "consistency"],
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
        """Sample latent for a training step.

        Returns:
            v   [B, latent_dim]
            kl  [B]  (zero tensor for prior_only)
        """
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
        """Sample latent for a generation step.

        In "consistency" mode the same v is reused across all reverse steps
        (call reset_cache() before each new generation batch).
        """
        if self.policy_type == "consistency":
            if step_idx == 0 or self._cached_v is None:
                self._cached_v = torch.randn(batch_size, self.latent_dim, device=device)
            return self._cached_v
        return torch.randn(batch_size, self.latent_dim, device=device)

    def reset_cache(self) -> None:
        """Reset the cached v (call before starting a new generation batch)."""
        self._cached_v = None


# ---------------------------------------------------------------------------
# Internal: lightweight sinusoidal embedding (no extra dependencies)
# ---------------------------------------------------------------------------

class _SinusoidalEmbedding(nn.Module):
    """Fixed sinusoidal positional embedding for the time input t in [0, 1]."""

    def __init__(self, embed_dim: int, scale: float = 1000.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = scale
        half = embed_dim // 2
        freqs = torch.exp(
            -torch.arange(half, dtype=torch.float32) * (torch.log(torch.tensor(10000.0)) / half)
        )
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float() * self.scale
        args = t.unsqueeze(-1) * self.freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.embed_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb
