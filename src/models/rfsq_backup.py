"""
Robust Residual FSQ (RFSQ) Module.

RFSQ improves upon standard FSQ by applying multi-stage residual quantization
with LayerNorm conditioning. This prevents residual magnitude decay across
stages and captures finer details that single-stage FSQ misses.

Key innovations:
- Multi-stage residual quantization: Stage 1 captures coarse structure, Stage 2+ refines details
- LayerNorm conditioning: Prevents residual magnitude decay across stages
- Finer granularity: More effective code usage without increasing codebook size

Reference: https://github.com/zhuxiaoxuhit/robust_rfsq
Paper: "Improving Finite Scalar Quantization via Progressive Training"

Example:
    levels = [5, 5, 5, 5]  # 4 dimensions, 5 levels each
    num_stages = 2
    rfsq = RFSQ(levels, num_stages)
    # Total implicit codes: 625 × 625 = 390,625
"""

from typing import List, Tuple, Dict, Optional
import math

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .fsq import FSQ


class InvertibleLayerNorm(nn.Module):
    """
    LayerNorm that stores statistics for exact inverse transformation.

    This is critical for RFSQ: we normalize before quantization, then
    inverse-transform after to preserve the original scale of residuals.
    Without this, residual magnitudes decay exponentially across stages.

    Args:
        num_features: Number of channels/features (C dimension).
        eps: Small epsilon for numerical stability.

    Note:
        Unlike standard LayerNorm which normalizes over the last dimension,
        this normalizes over spatial dimensions (X, Y, Z) for 3D data.
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

        # Learnable affine parameters (per-channel)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

        # Stored during forward for inverse (not persistent - don't save to checkpoint)
        self.register_buffer('stored_mean', None, persistent=False)
        self.register_buffer('stored_std', None, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Normalize input and store statistics for inverse.

        Args:
            x: Input tensor of shape [B, C, X, Y, Z] or [B, X, Y, Z, C].
               Assumes C = num_features.

        Returns:
            Normalized tensor of same shape.
        """
        # Determine if channels are last or first
        channels_last = x.shape[-1] == self.num_features

        if channels_last:
            # [B, X, Y, Z, C] -> normalize over X, Y, Z (dims 1, 2, 3)
            self.stored_mean = x.mean(dim=(1, 2, 3), keepdim=True)
            self.stored_std = x.std(dim=(1, 2, 3), keepdim=True) + self.eps
            x_norm = (x - self.stored_mean) / self.stored_std
            # Apply affine: weight and bias are [C], broadcast to [..., C]
            return x_norm * self.weight + self.bias
        else:
            # [B, C, X, Y, Z] -> normalize over X, Y, Z (dims 2, 3, 4)
            self.stored_mean = x.mean(dim=(2, 3, 4), keepdim=True)
            self.stored_std = x.std(dim=(2, 3, 4), keepdim=True) + self.eps
            x_norm = (x - self.stored_mean) / self.stored_std
            # Apply affine: weight and bias are [C], need to reshape to [1, C, 1, 1, 1]
            return x_norm * self.weight.view(1, -1, 1, 1, 1) + self.bias.view(1, -1, 1, 1, 1)

    def inverse(self, x_norm: Tensor) -> Tensor:
        """
        Inverse transform: undo normalization using stored statistics.

        Args:
            x_norm: Normalized tensor from forward pass.

        Returns:
            Tensor in original scale.

        Raises:
            RuntimeError: If called before forward() (no stored statistics).
        """
        if self.stored_mean is None or self.stored_std is None:
            raise RuntimeError("Must call forward() before inverse()")

        # Determine if channels are last or first
        channels_last = x_norm.shape[-1] == self.num_features

        if channels_last:
            # Undo affine
            x = (x_norm - self.bias) / self.weight
            # Undo normalization
            return x * self.stored_std + self.stored_mean
        else:
            # Undo affine
            x = (x_norm - self.bias.view(1, -1, 1, 1, 1)) / self.weight.view(1, -1, 1, 1, 1)
            # Undo normalization
            return x * self.stored_std + self.stored_mean


class RFSQStage(nn.Module):
    """
    Single stage of Residual FSQ with LayerNorm conditioning.

    Each stage:
    1. Normalizes the residual (stores stats for inverse)
    2. Quantizes in normalized space using FSQ
    3. Inverse-transforms back to original scale
    4. Computes new residual for next stage

    Args:
        levels: List of quantization levels per dimension.
    """

    def __init__(self, levels: List[int]):
        super().__init__()
        self.levels = levels
        self.fsq = FSQ(levels)
        self.layernorm = InvertibleLayerNorm(len(levels))

    @property
    def codebook_size(self) -> int:
        """Number of implicit codes in this stage."""
        return self.fsq.codebook_size

    def forward(self, residual: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Quantize residual with LayerNorm conditioning.

        Args:
            residual: Input residual tensor [B, X, Y, Z, C] where C = len(levels).

        Returns:
            z_q: Quantized representation (in original scale).
            new_residual: residual - z_q (for next stage).
            indices: Integer indices from FSQ.
        """
        # 1. Normalize residual (prevents magnitude decay)
        z_norm = self.layernorm(residual)

        # 2. Quantize in normalized space
        z_q_norm, indices = self.fsq(z_norm)

        # 3. Inverse transform back to original scale
        z_q = self.layernorm.inverse(z_q_norm)

        # 4. Compute new residual for next stage
        new_residual = residual - z_q

        return z_q, new_residual, indices


class RFSQ(nn.Module):
    """
    Robust Residual FSQ with multiple stages.

    Applies residual quantization across multiple stages. Each stage quantizes
    the residual left by previous stages, progressively capturing finer details.

    Key insight: LayerNorm conditioning normalizes residuals before quantization,
    preventing the exponential decay of residual magnitudes that would otherwise
    cause later stages to become ineffective.

    Args:
        levels_per_stage: List of quantization levels per dimension.
                         E.g., [5, 5, 5, 5] gives 625 codes per stage.
        num_stages: Number of residual stages (default: 2).

    Example:
        >>> rfsq = RFSQ([5, 5, 5, 5], num_stages=2)
        >>> z = torch.randn(2, 8, 8, 8, 4)  # [B, X, Y, Z, C]
        >>> z_q, all_indices = rfsq(z)
        >>> len(all_indices)  # 2 stages
        2
    """

    def __init__(self, levels_per_stage: List[int], num_stages: int = 2):
        super().__init__()

        self.levels_per_stage = levels_per_stage
        self.num_stages = num_stages
        self.dim = len(levels_per_stage)

        # Create stages
        self.stages = nn.ModuleList([
            RFSQStage(levels_per_stage) for _ in range(num_stages)
        ])

        # Total implicit codebook size (product of all stages)
        codes_per_stage = int(np.prod(levels_per_stage))
        self.codebook_size = codes_per_stage ** num_stages
        self.codes_per_stage = codes_per_stage

    @property
    def num_codes(self) -> int:
        """Return total implicit codebook size."""
        return self.codebook_size

    def forward(self, z: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """
        Multi-stage residual quantization.

        Args:
            z: Encoder output [B, X, Y, Z, C] where C = len(levels_per_stage).

        Returns:
            z_q: Quantized sum of all stages (same shape as input).
            all_indices: List of indices from each stage.
        """
        residual = z
        z_q_sum = torch.zeros_like(z)
        all_indices = []

        for stage in self.stages:
            z_q, residual, indices = stage(residual)
            z_q_sum = z_q_sum + z_q
            all_indices.append(indices)

        return z_q_sum, all_indices

    def forward_with_norms(self, z: Tensor) -> Tuple[Tensor, List[Tensor], List[float]]:
        """
        Forward pass that also returns residual norms for monitoring.

        Useful for verifying that LayerNorm prevents residual decay.

        Args:
            z: Encoder output [B, X, Y, Z, C].

        Returns:
            z_q: Quantized sum of all stages.
            all_indices: List of indices from each stage.
            residual_norms: List of residual norms before each stage.
        """
        residual = z
        z_q_sum = torch.zeros_like(z)
        all_indices = []
        residual_norms = []

        for stage in self.stages:
            # Record residual norm before this stage
            residual_norms.append(residual.norm().item())

            z_q, residual, indices = stage(residual)
            z_q_sum = z_q_sum + z_q
            all_indices.append(indices)

        # Also record final residual norm
        residual_norms.append(residual.norm().item())

        return z_q_sum, all_indices, residual_norms

    def get_codebook_usage(self, all_indices: List[Tensor]) -> Dict[str, float]:
        """
        Compute codebook usage statistics per stage.

        Args:
            all_indices: List of indices from forward pass.

        Returns:
            Dictionary with per-stage usage and perplexity.
        """
        metrics = {}

        for i, indices in enumerate(all_indices):
            usage, perplexity = self.stages[i].fsq.get_codebook_usage(indices)
            metrics[f'stage{i}_usage'] = usage
            metrics[f'stage{i}_perplexity'] = perplexity

        # Overall metrics
        metrics['avg_usage'] = sum(metrics[f'stage{i}_usage'] for i in range(self.num_stages)) / self.num_stages
        metrics['avg_perplexity'] = sum(metrics[f'stage{i}_perplexity'] for i in range(self.num_stages)) / self.num_stages

        return metrics

    def indices_to_codes(self, all_indices: List[Tensor]) -> Tensor:
        """
        Convert indices from all stages back to quantized vectors.

        Note: This is approximate because we don't have the stored LayerNorm
        statistics from the original forward pass. For exact reconstruction,
        you need the original z_e input.

        Args:
            all_indices: List of indices from each stage.

        Returns:
            z_q: Sum of quantized vectors from all stages.
        """
        z_q_sum = None

        for i, indices in enumerate(all_indices):
            z_q_stage = self.stages[i].fsq.indices_to_codes(indices)
            if z_q_sum is None:
                z_q_sum = z_q_stage
            else:
                z_q_sum = z_q_sum + z_q_stage

        return z_q_sum


# Recommended RFSQ configurations
RFSQ_CONFIGS = {
    # v6 default: 2 stages, 625 codes each, ~390K total
    'v6_default': {
        'levels': [5, 5, 5, 5],
        'num_stages': 2,
    },

    # High capacity: 2 stages, 2401 codes each, ~5.8M total
    'high_capacity': {
        'levels': [7, 7, 7, 7],
        'num_stages': 2,
    },

    # 3-stage: more residual refinement
    'three_stage': {
        'levels': [5, 5, 5, 5],
        'num_stages': 3,
    },

    # Tiny: for testing
    'tiny': {
        'levels': [3, 3, 3, 3],
        'num_stages': 2,
    },
}


def get_rfsq_config(name: str) -> Dict:
    """Get a predefined RFSQ configuration by name."""
    if name not in RFSQ_CONFIGS:
        raise ValueError(f"Unknown RFSQ config: {name}. Available: {list(RFSQ_CONFIGS.keys())}")
    return RFSQ_CONFIGS[name]
