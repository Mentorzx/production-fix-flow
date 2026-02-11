"""Variational Autoencoder with Indian Buffet Process prior for DSLFM-KGC.

This module implements the VAE encoder that maps entity embeddings to:
1. Feature latent space (continuous)
2. Community membership space (sparse, via IBP prior)

Design Patterns:
    - Strategy: Different priors can be swapped
    - Template Method: Base VAE with customizable prior
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from pff.shared.core.logging import logger


class IndianBuffetProcessPrior(nn.Module):
    """Indian Buffet Process prior for sparse community discovery.

    The IBP allows learning the number of latent communities from data,
    encouraging sparse community assignments where each entity belongs
    to only a few communities.

    Args:
        alpha: IBP concentration parameter. Higher = more communities.
        max_communities: Maximum number of communities (truncation).
        temperature: Temperature for Gumbel-Softmax sampling.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        max_communities: int = 128,
        temperature: float = 0.5,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.max_communities = max_communities
        self.temperature = temperature

        self.log_pi = nn.Parameter(torch.zeros(max_communities))
        self._init_stick_breaking()

    def _init_stick_breaking(self) -> None:
        """Initialize using stick-breaking construction of IBP."""
        k = torch.arange(self.max_communities, device=self.log_pi.data.device, dtype=torch.float32)
        expected_pi = self.alpha / (self.alpha + k + 1.0)
        self.log_pi.data.copy_(torch.log(expected_pi + 1e-8))

    def get_prior_probs(self) -> torch.Tensor:
        """Get community activation probabilities.

        Returns:
            Probabilities for each community [max_communities].
        """
        return torch.sigmoid(self.log_pi)

    def sample_communities(
        self,
        batch_size: int,
        device: torch.device,
        hard: bool = False,
    ) -> torch.Tensor:
        """Sample community memberships using Gumbel-Softmax.

        Args:
            batch_size: Number of samples.
            device: Target device.
            hard: If True, use straight-through estimator.

        Returns:
            Soft community assignments [batch_size, max_communities].
        """
        pi = self.get_prior_probs().to(device)

        logits = torch.stack(
            [
                torch.log(1 - pi + 1e-8),
                torch.log(pi + 1e-8),
            ],
            dim=-1,
        )

        logits = logits.unsqueeze(0).expand(batch_size, -1, -1)

        samples = F.gumbel_softmax(logits, tau=self.temperature, hard=hard)

        return samples[:, :, 1]

    def kl_divergence(self, q_z: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence between posterior and IBP prior.

        Numerical stability: Always compute in float32 to avoid underflow
        in float16/bfloat16, especially under autocast (AMP).

        Args:
            q_z: Posterior community probabilities [batch, max_communities].

        Returns:
            KL divergence scalar.
        """

        device_type = q_z.device.type
        with torch.autocast(enabled=False, device_type=device_type):
            q_z_f32 = q_z.to(torch.float32)
            pi = self.get_prior_probs().to(q_z.device).to(torch.float32)

            eps = max(1e-6, torch.finfo(torch.float32).eps * 10)

            q_z_clamped = q_z_f32.clamp(min=eps, max=1.0 - eps)
            pi_clamped = pi.clamp(min=eps, max=1.0 - eps)

            log_q = torch.log(q_z_clamped)
            log_pi = torch.log(pi_clamped)

            log1m_q = torch.log1p(-q_z_clamped)
            log1m_pi = torch.log1p(-pi_clamped)

            kl = q_z_f32 * (log_q - log_pi) + (1.0 - q_z_f32) * (log1m_q - log1m_pi)

            kl_sum = kl.sum(dim=-1).mean()

            if not torch.isfinite(kl_sum):
                q_min = q_z_f32.min().item()
                q_max = q_z_f32.max().item()
                pi_min = pi_clamped.min().item()
                pi_max = pi_clamped.max().item()
                logger.error(
                    f"Non-finite KL divergence detected: "
                    f"kl_sum={kl_sum.item()} "
                    f"q_z_range=[{q_min:.6f}, {q_max:.6f}] "
                    f"pi_range=[{pi_min:.6f}, {pi_max:.6f}]"
                )

                return torch.tensor(0.0, device=q_z.device, dtype=q_z.dtype)

            return kl_sum.to(q_z.dtype)

    def sparsity_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Additional sparsity regularization.

        Encourages each entity to belong to few communities.

        Args:
            z: Community assignments [batch, max_communities].

        Returns:
            Sparsity loss scalar.
        """
        return z.abs().mean()


class DSLFMVAEEncoder(nn.Module):
    """VAE encoder for DSLFM-KGC.

    Maps entity embeddings to:
    1. Feature latent variables (Gaussian prior)
    2. Community membership variables (IBP prior)

    Args:
        input_dim: Input embedding dimension.
        feature_dim: Feature latent dimension.
        max_communities: Maximum number of communities.
        hidden_dim: Hidden layer dimension.
        ibp_alpha: IBP concentration parameter.
    """

    def __init__(
        self,
        input_dim: int,
        feature_dim: int = 256,
        max_communities: int = 128,
        hidden_dim: int = 512,
        ibp_alpha: float = 1.0,
        use_checkpointing: bool = False,
        dropout_p: float = 0.0,
        logvar_clip_min: float = -20.0,
        logvar_clip_max: float = 10.0,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.max_communities = max_communities
        self.use_checkpointing = use_checkpointing
        self.dropout_p = dropout_p
        self.logvar_clip_min = float(logvar_clip_min)
        self.logvar_clip_max = float(logvar_clip_max)
        if self.logvar_clip_max < self.logvar_clip_min:
            raise ValueError(
                f"logvar_clip_max must be >= logvar_clip_min, got "
                f"min={self.logvar_clip_min} max={self.logvar_clip_max}"
            )

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.fc_mu = nn.Linear(hidden_dim, feature_dim)
        self.fc_logvar = nn.Linear(hidden_dim, feature_dim)

        self.fc_community_logits = nn.Linear(hidden_dim, max_communities)

        self.ibp_prior = IndianBuffetProcessPrior(
            alpha=ibp_alpha,
            max_communities=max_communities,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _encode_impl(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Core encoding logic (checkpointable).

        Args:
            x: Input embeddings [batch, input_dim].

        Returns:
            Tuple of (feature_mu, feature_logvar, community_logits).
        """
        h = self.encoder(x)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(min=self.logvar_clip_min, max=self.logvar_clip_max)
        community_logits = self.fc_community_logits(h)

        return mu, logvar, community_logits

    def encode(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode input to latent parameters.

        Uses gradient checkpointing when enabled to reduce VRAM usage
        at the cost of recomputing activations during backward pass.

        Args:
            x: Input embeddings [batch, input_dim].

        Returns:
            Tuple of (feature_mu, feature_logvar, community_logits).
        """
        if self.use_checkpointing and self.training:
            return grad_checkpoint(
                self._encode_impl,
                x,
                use_reentrant=False,
            )
        return self._encode_impl(x)

    def reparameterize_gaussian(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Reparameterization trick for Gaussian.

        Args:
            mu: Mean [batch, dim].
            logvar: Log variance [batch, dim].

        Returns:
            Sampled latent [batch, dim].
        """
        logvar = logvar.clamp(min=self.logvar_clip_min, max=self.logvar_clip_max)
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def reparameterize_community(
        self,
        logits: torch.Tensor,
        temperature: float = 0.5,
        hard: bool = False,
    ) -> torch.Tensor:
        """Sample community memberships with Gumbel-Softmax.

        During training, uses a binary Gumbel-Softmax relaxation per community.
        During evaluation, returns sigmoid probabilities.

        Args:
            logits: Community logits [batch, max_communities].
            temperature: Gumbel-Softmax temperature.
            hard: Use straight-through estimator.

        Returns:
            Soft community assignments [batch, max_communities].
        """
        temperature = max(float(temperature), 1e-4)
        if self.training:
            logits_binary = torch.stack(
                [
                    torch.zeros_like(logits),
                    logits,
                ],
                dim=-1,
            )

            samples = F.gumbel_softmax(logits_binary, tau=temperature, hard=hard)
            return samples[:, :, 1]

        return torch.sigmoid(logits)

    def forward(
        self,
        x: torch.Tensor,
        temperature: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through VAE encoder.

        Args:
            x: Input embeddings [batch, input_dim].
            temperature: Gumbel-Softmax temperature.

        Returns:
            Dictionary with:
                - features: Feature latents [batch, feature_dim]
                - communities: Community assignments [batch, max_communities]
                - mu: Feature mean
                - logvar: Feature log variance
                - community_logits: Raw community logits
        """
        mu, logvar, community_logits = self.encode(x)

        features = self.reparameterize_gaussian(mu, logvar)
        communities = self.reparameterize_community(community_logits, temperature)

        return {
            "features": features,
            "communities": communities,
            "mu": mu,
            "logvar": logvar,
            "community_logits": community_logits,
        }

    def kl_loss(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        community_probs: torch.Tensor,
        free_bits: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """Compute KL divergence losses with optional free-bits.

        Free-bits prevents posterior collapse by enforcing a minimum KL
        per dimension (Bowman et al., 2016; Kingma et al., 2016).

        Args:
            mu: Feature mean.
            logvar: Feature log variance.
            community_probs: Community probabilities.
            free_bits: Minimum KL per dimension (0 = disabled).

        Returns:
            Dictionary with kl_gaussian, kl_ibp, and total kl_loss.
        """
        logvar = logvar.clamp(min=self.logvar_clip_min, max=self.logvar_clip_max)
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

        if free_bits > 0:
            kl_per_dim = torch.maximum(
                kl_per_dim, torch.tensor(free_bits, device=mu.device, dtype=mu.dtype)
            )

        kl_gaussian = kl_per_dim.sum(dim=-1).mean()

        kl_ibp = self.ibp_prior.kl_divergence(community_probs)

        return {
            "kl_gaussian": kl_gaussian,
            "kl_ibp": kl_ibp,
            "kl_loss": kl_gaussian + kl_ibp,
        }
