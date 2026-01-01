"""
Finite Scalar Quantization (FSQ) Module.

FSQ is an alternative to VQ-VAE's learned codebook that eliminates codebook collapse
by design. Instead of learning codebook vectors, it quantizes each latent dimension
to a fixed number of levels.

Key advantages:
- No codebook collapse (impossible by design)
- No commitment loss needed
- No EMA updates needed
- Implicit codebook size = product of all levels

Reference: https://arxiv.org/abs/2309.15505 (Google's FSQ paper)

Example:
    levels = [5, 5, 5, 5, 5, 5, 5, 5]  # 8 dimensions, 5 levels each
    implicit_codebook_size = 5^8 = 390,625 codes
"""

from typing import List, Tuple, Optional
import math

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class FSQ(nn.Module):
    """
    Finite Scalar Quantization.

    Quantizes each dimension of the latent vector to a fixed number of levels.
    Uses straight-through estimator for gradients.

    Args:
        levels: List of quantization levels per dimension.
                E.g., [5, 5, 5, 5, 5, 5, 5, 5] gives 5^8 = 390,625 implicit codes.
        eps: Small epsilon for numerical stability.

    Example:
        >>> fsq = FSQ(levels=[5, 5, 5, 5, 5, 5, 5, 5])
        >>> z = torch.randn(2, 8, 8, 8, 8)  # [batch, dim, x, y, z]
        >>> z_q, indices = fsq(z)
        >>> z_q.shape  # Same as input
        torch.Size([2, 8, 8, 8, 8])
    """

    def __init__(self, levels: List[int], eps: float = 1e-3):
        super().__init__()

        self.levels = levels
        self.dim = len(levels)
        self.eps = eps

        # Implicit codebook size
        self.codebook_size = int(np.prod(levels))

        # Register levels as buffer for device movement
        self.register_buffer('_levels', torch.tensor(levels, dtype=torch.float32))

        # Precompute basis for index calculation
        basis = []
        acc = 1
        for L in reversed(levels):
            basis.append(acc)
            acc *= L
        self.register_buffer('_basis', torch.tensor(list(reversed(basis)), dtype=torch.long))

        # Precompute half-levels for quantization
        # For level L, values are in {0, 1, ..., L-1}, centered at (L-1)/2
        half_levels = [(L - 1) / 2 for L in levels]
        self.register_buffer('_half_levels', torch.tensor(half_levels, dtype=torch.float32))

    @property
    def num_codes(self) -> int:
        """Return implicit codebook size."""
        return self.codebook_size

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Quantize latent vectors.

        Args:
            z: Latent vectors of shape [..., dim] where dim = len(levels).
               The last dimension must match the number of levels.

        Returns:
            z_q: Quantized vectors, same shape as input.
            indices: Integer indices of shape [...], each in [0, codebook_size).
        """
        # Bound input to (-1, 1) using tanh
        z_bounded = torch.tanh(z)

        # Quantize
        z_q = self._quantize(z_bounded)

        # Straight-through estimator: forward uses quantized, backward uses continuous
        z_q = z_bounded + (z_q - z_bounded).detach()

        # Compute indices for transformer training
        indices = self._to_indices(z_q)

        return z_q, indices

    def _quantize(self, z: Tensor) -> Tensor:
        """
        Quantize each dimension to its levels.

        Maps [-1, 1] -> {-(L-1)/2, ..., 0, ..., (L-1)/2} / ((L-1)/2)
        which gives values in [-1, 1] at quantized positions.
        """
        # z is in (-1, 1) from tanh
        # Scale to [0, L-1] for each dimension, round, scale back

        z_q_list = []
        for i in range(self.dim):
            L = self._levels[i]
            half_L = self._half_levels[i]

            z_i = z[..., i]

            # Map (-1, 1) to (-half_L, half_L)
            z_i = z_i * half_L

            # Round to nearest integer
            z_i = torch.round(z_i)

            # Clamp to valid range (in case tanh outputs exactly +-1)
            z_i = torch.clamp(z_i, -half_L, half_L)

            # Map back to (-1, 1)
            z_i = z_i / half_L

            z_q_list.append(z_i)

        return torch.stack(z_q_list, dim=-1)

    def _to_indices(self, z_q: Tensor) -> Tensor:
        """
        Convert quantized vectors to integer indices.

        Each unique quantized vector maps to a unique integer in [0, codebook_size).
        """
        indices = torch.zeros(z_q.shape[:-1], dtype=torch.long, device=z_q.device)

        for i in range(self.dim):
            L = self._levels[i].long()
            half_L = self._half_levels[i]

            z_i = z_q[..., i]

            # Map from [-1, 1] to [0, L-1]
            level_idx = ((z_i * half_L) + half_L).round().long()
            level_idx = torch.clamp(level_idx, 0, L - 1)

            # Accumulate into index
            indices = indices + level_idx * self._basis[i]

        return indices

    def indices_to_codes(self, indices: Tensor) -> Tensor:
        """
        Convert integer indices back to quantized vectors.

        Useful for decoding during generation.

        Args:
            indices: Integer indices of shape [...], each in [0, codebook_size).

        Returns:
            z_q: Quantized vectors of shape [..., dim].
        """
        z_q_list = []
        remaining = indices.clone()

        for i in range(self.dim):
            L = self._levels[i].long()
            half_L = self._half_levels[i]

            # Extract this dimension's level
            level_idx = remaining // self._basis[i]
            remaining = remaining % self._basis[i]

            # Map from [0, L-1] to [-1, 1]
            z_i = (level_idx.float() - half_L) / half_L
            z_q_list.append(z_i)

        return torch.stack(z_q_list, dim=-1)

    def get_codebook_usage(self, indices: Tensor) -> Tuple[float, float]:
        """
        Compute codebook usage statistics.

        Args:
            indices: Batch of indices from forward pass.

        Returns:
            usage: Fraction of codes used at least once.
            perplexity: Effective number of codes (exp of entropy).
        """
        # Flatten indices
        flat_indices = indices.flatten()

        # Count occurrences
        counts = torch.bincount(flat_indices, minlength=self.codebook_size).float()

        # Usage: fraction of codes used at least once
        usage = (counts > 0).float().mean().item()

        # Perplexity: exp(entropy)
        probs = counts / counts.sum()
        probs = probs[probs > 0]  # Avoid log(0)
        entropy = -(probs * torch.log(probs)).sum()
        perplexity = torch.exp(entropy).item()

        return usage, perplexity


class FSQWithProjection(nn.Module):
    """
    FSQ with learned input/output projections.

    Useful when the encoder output dimension doesn't match the desired FSQ dimension.

    Args:
        input_dim: Dimension of encoder output.
        fsq_dim: Number of FSQ dimensions (len(levels)).
        levels: Quantization levels per dimension.
    """

    def __init__(self, input_dim: int, levels: List[int]):
        super().__init__()

        fsq_dim = len(levels)

        # Project from encoder dim to FSQ dim
        self.pre_proj = nn.Linear(input_dim, fsq_dim)

        # FSQ quantization
        self.fsq = FSQ(levels)

        # Project back to encoder dim for decoder
        self.post_proj = nn.Linear(fsq_dim, input_dim)

    @property
    def codebook_size(self) -> int:
        return self.fsq.codebook_size

    @property
    def num_codes(self) -> int:
        return self.fsq.num_codes

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            z: Encoder output of shape [..., input_dim].

        Returns:
            z_q: Quantized and projected back, shape [..., input_dim].
            indices: Integer indices, shape [...].
        """
        # Project to FSQ dimension
        z_fsq = self.pre_proj(z)

        # Quantize
        z_q_fsq, indices = self.fsq(z_fsq)

        # Project back to encoder dimension
        z_q = self.post_proj(z_q_fsq)

        return z_q, indices

    def indices_to_codes(self, indices: Tensor) -> Tensor:
        """Decode indices back to vectors in encoder dimension."""
        z_q_fsq = self.fsq.indices_to_codes(indices)
        return self.post_proj(z_q_fsq)


# Recommended configurations for different use cases
FSQ_CONFIGS = {
    # Small: good for initial experiments
    'small': [5, 5, 5, 5, 5, 5, 5, 5],  # 8 dims, 390,625 codes

    # Medium: more capacity
    'medium': [8, 8, 8, 8, 8, 5, 5, 5],  # 8 dims, 6,553,600 codes

    # Large: maximum capacity
    'large': [8, 8, 8, 8, 8, 8, 8, 8],  # 8 dims, 16,777,216 codes

    # Tiny: for debugging
    'tiny': [3, 3, 3, 3],  # 4 dims, 81 codes
}


def get_fsq_config(name: str) -> List[int]:
    """Get a predefined FSQ configuration by name."""
    if name not in FSQ_CONFIGS:
        raise ValueError(f"Unknown FSQ config: {name}. Available: {list(FSQ_CONFIGS.keys())}")
    return FSQ_CONFIGS[name]
