"""Loss functions for VQ-VAE training.

This module contains custom loss functions for Minecraft structure generation
with VQ-VAE models, including frequency-weighted losses, volume penalties,
and perceptual losses.
"""

from typing import Dict, Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyWeightedLoss(nn.Module):
    """Frequency-weighted cross-entropy loss with volume penalty and perceptual loss.

    Combines three loss components:
    1. Frequency-weighted cross-entropy: Upweights rare blocks to prevent mode collapse
    2. Volume penalty: Penalizes deviation from ground truth volume (fixes over-prediction)
    3. Perceptual loss: Spatial smoothness regularization on latent codes

    This loss function addresses key issues from v6-freq:
    - Mode collapse on common blocks (frequency weighting)
    - 1.68x volume over-prediction (volume penalty)
    - Lack of spatial coherence in latent space (perceptual loss)

    Args:
        frequency_weights: Tensor of shape [vocab_size] with pre-computed frequency weights.
            Typically computed as: weight[i] = (total_count / count[i]) ** 0.5
        frequency_cap: Maximum weight value (prevents extreme weights on very rare blocks).
            Default: 5.0 (reduced from v6-freq's 10.0 for better balance)
        terrain_weight: Weight for terrain block cross-entropy. Default: 0.2
        building_weight: Weight for building block cross-entropy. Default: 1.0
        air_weight: Weight for air block cross-entropy. Default: 0.1
        volume_penalty_weight: Weight for volume penalty loss. Default: 10.0
            Penalizes deviation from ground truth volume ratio.
        false_air_penalty_weight: Weight for false air penalty. Default: 5.0
            CRITICAL for recall: Penalizes predicting air where structure exists.
            This prevents the model from "cheating" by erasing structures.
        perceptual_weight: Weight for perceptual (spatial smoothness) loss. Default: 0.1
        air_tokens: Set of token IDs representing air blocks. Default: {102, 576, 3352}
            (minecraft:air, minecraft:cave_air, minecraft:void_air)

    Example:
        >>> # Compute frequency weights from training data
        >>> block_counts = torch.bincount(train_data.flatten())
        >>> total = block_counts.sum()
        >>> freq_weights = (total / block_counts.clamp(min=1)).sqrt()
        >>>
        >>> # Create loss function
        >>> criterion = FrequencyWeightedLoss(
        ...     frequency_weights=freq_weights,
        ...     frequency_cap=5.0,
        ...     volume_penalty_weight=1.0
        ... )
        >>>
        >>> # Compute loss
        >>> logits, z_q, indices = model(batch)
        >>> loss_dict = criterion(logits, batch, z_q)
        >>> loss_dict['loss'].backward()
    """

    def __init__(
        self,
        frequency_weights: torch.Tensor,
        frequency_cap: float = 5.0,
        terrain_weight: float = 0.2,
        building_weight: float = 1.0,
        air_weight: float = 0.1,
        volume_penalty_weight: float = 10.0,  # Balanced: strong but not overwhelming
        false_air_penalty_weight: float = 5.0,  # NEW: Protect recall by penalizing erasure
        perceptual_weight: float = 0.1,
        air_tokens: Optional[Set[int]] = None,
    ):
        super().__init__()

        # Clamp frequency weights to prevent extreme values
        clamped_weights = frequency_weights.clamp(max=frequency_cap)
        self.register_buffer('freq_weights', clamped_weights)

        self.terrain_weight = terrain_weight
        self.building_weight = building_weight
        self.air_weight = air_weight
        self.volume_penalty_weight = volume_penalty_weight
        self.false_air_penalty_weight = false_air_penalty_weight
        self.perceptual_weight = perceptual_weight

        # Air tokens (minecraft:air, minecraft:cave_air, minecraft:void_air)
        if air_tokens is None:
            air_tokens = {102, 576, 3352}
        self.air_tokens = air_tokens

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        z_q: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined loss.

        Args:
            logits: Predicted logits of shape [B, vocab_size, H, W, D]
            target: Ground truth block IDs of shape [B, H, W, D]
            z_q: Quantized latent codes of shape [B, C, H_latent, W_latent, D_latent]
                Used for perceptual loss computation

        Returns:
            Dictionary with keys:
                - 'loss': Combined total loss (for backward pass)
                - 'ce_loss': Cross-entropy component (detached, for logging)
                - 'volume_loss': Volume penalty component (detached, for logging)
                - 'perceptual_loss': Perceptual component (detached, for logging)
                - 'volume_ratio': Predicted volume / GT volume (detached, for logging)
        """
        # 1. Frequency-weighted cross-entropy
        # Reshape: [B, vocab_size, H, W, D] -> [B*H*W*D, vocab_size]
        batch_size = logits.shape[0]
        vocab_size = logits.shape[1]
        logits_flat = logits.permute(0, 2, 3, 4, 1).reshape(-1, vocab_size)
        target_flat = target.reshape(-1)

        ce_loss = F.cross_entropy(
            logits_flat,
            target_flat,
            weight=self.freq_weights,
            reduction='mean'
        )

        # 2. DIRECT LOGIT-BASED VOLUME CONTROL
        # Previous softmax approach failed because gradients vanish with confident predictions.
        # NEW APPROACH: Directly manipulate logits, bypassing softmax entirely.

        # Ground truth masks
        air_tokens_tensor = torch.tensor(
            list(self.air_tokens),
            device=target.device,
            dtype=target.dtype
        )
        gt_is_air = torch.isin(target, air_tokens_tensor)
        gt_is_structure = ~gt_is_air
        gt_volume = gt_is_structure.float().sum()

        # Compute hard volume ratio for logging
        with torch.no_grad():
            pred_hard = torch.argmax(logits, dim=1)
            pred_is_air_hard = torch.isin(pred_hard, air_tokens_tensor)
            pred_volume_hard = (~pred_is_air_hard).float().sum()
            volume_ratio_hard = pred_volume_hard / (gt_volume + 1e-6)

            # Recall = (structure voxels not erased) / (total structure voxels)
            structure_preserved = gt_is_structure & (~pred_is_air_hard)
            recall = structure_preserved.float().sum() / (gt_is_structure.float().sum() + 1e-6)

        # DIRECT LOGIT PENALTY: At locations where GT is AIR, boost air logits
        # This DIRECTLY encourages the model to predict air where it should.
        # Key insight: Instead of penalizing volume ratio (which goes through softmax),
        # we directly maximize air logits at air locations.

        # Stack air logits: [num_air_tokens, B, H, W, D]
        air_logits_list = [logits[:, t, :, :, :] for t in self.air_tokens if t < logits.shape[1]]
        if air_logits_list:
            air_logits = torch.stack(air_logits_list, dim=0)
            max_air_logit = air_logits.max(dim=0)[0]  # [B, H, W, D] - best air logit per voxel
        else:
            max_air_logit = torch.zeros_like(logits[:, 0, :, :, :])

        # Get max non-air logit per voxel
        # Create mask for air tokens in vocab dimension
        air_mask = torch.zeros(logits.shape[1], device=logits.device, dtype=torch.bool)
        for t in self.air_tokens:
            if t < logits.shape[1]:
                air_mask[t] = True

        # Set air logits to -inf to find max non-air
        logits_non_air = logits.clone()
        logits_non_air[:, air_mask, :, :, :] = float('-inf')
        max_non_air_logit = logits_non_air.max(dim=1)[0]  # [B, H, W, D]

        # VOLUME LOSS: At GT air locations, penalize if max_non_air > max_air
        # This directly encourages air predictions where GT is air
        if gt_is_air.any():
            # margin: how much higher should air logit be than non-air logit
            margin = 1.0
            # Penalize when non-air logit is higher than air logit at air locations
            violation = F.relu(max_non_air_logit[gt_is_air] - max_air_logit[gt_is_air] + margin)
            volume_loss = violation.mean()
        else:
            volume_loss = torch.tensor(0.0, device=logits.device)

        # 3. FALSE AIR PENALTY: At GT structure locations, penalize if max_air > max_non_air
        # This protects recall by discouraging air predictions where structure exists
        if gt_is_structure.any():
            margin = 1.0
            # Penalize when air logit is higher than non-air logit at structure locations
            violation = F.relu(max_air_logit[gt_is_structure] - max_non_air_logit[gt_is_structure] + margin)
            false_air_loss = violation.mean()
        else:
            false_air_loss = torch.tensor(0.0, device=logits.device)

        # 4. Perceptual loss (spatial smoothness)
        perceptual_loss = self._compute_perceptual_loss(z_q)

        # Combined loss with BALANCED weights:
        # - CE loss: primary reconstruction objective
        # - Volume loss: prevents over-prediction (ratio > 1)
        # - False air loss: prevents under-prediction / erasure (protects recall)
        # - Perceptual loss: spatial coherence
        total_loss = (
            ce_loss +
            self.volume_penalty_weight * volume_loss +
            self.false_air_penalty_weight * false_air_loss +
            self.perceptual_weight * perceptual_loss
        )

        return {
            'loss': total_loss,
            'ce_loss': ce_loss.detach(),
            'volume_loss': volume_loss.detach(),
            'false_air_loss': false_air_loss.detach() if torch.is_tensor(false_air_loss) else false_air_loss,
            'perceptual_loss': perceptual_loss.detach(),
            'volume_ratio': volume_ratio_hard,  # Hard ratio for accurate logging
            'recall': recall,  # Structure preservation (critical metric!)
        }

    def _compute_perceptual_loss(self, z_q: torch.Tensor) -> torch.Tensor:
        """Compute spatial smoothness regularization on latent codes.

        Penalizes large gradients in the latent space to encourage spatially
        coherent latent representations. This helps prevent neighboring voxels
        from having drastically different latent codes.

        Args:
            z_q: Quantized latent of shape [B, C, H, W, D]

        Returns:
            Scalar perceptual loss (mean absolute difference across spatial dimensions)
        """
        # Compute absolute differences along each spatial dimension
        # This measures the "gradient" or "smoothness" of the latent codes

        # Differences along H dimension: |z[:,:,i+1,:,:] - z[:,:,i,:,:]|
        diff_h = (z_q[:, :, 1:, :, :] - z_q[:, :, :-1, :, :]).abs().mean()

        # Differences along W dimension: |z[:,:,:,j+1,:] - z[:,:,:,j,:]|
        diff_w = (z_q[:, :, :, 1:, :] - z_q[:, :, :, :-1, :]).abs().mean()

        # Differences along D dimension: |z[:,:,:,:,k+1] - z[:,:,:,:,k]|
        diff_d = (z_q[:, :, :, :, 1:] - z_q[:, :, :, :, :-1]).abs().mean()

        # Average across all three spatial dimensions
        smooth_loss = (diff_h + diff_w + diff_d) / 3.0

        return smooth_loss


class TerrainWeightedLoss(nn.Module):
    """Terrain-aware cross-entropy loss (used in v6-freq).

    DEPRECATED: Use FrequencyWeightedLoss instead, which includes all v6-freq
    features plus volume penalty and perceptual loss.

    This class is kept for backwards compatibility with v6-freq checkpoints.
    """

    def __init__(
        self,
        terrain_weight: float = 0.2,
        building_weight: float = 1.0,
        air_weight: float = 0.1,
    ):
        super().__init__()
        self.terrain_weight = terrain_weight
        self.building_weight = building_weight
        self.air_weight = air_weight

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        terrain_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute terrain-weighted cross-entropy.

        Args:
            logits: Predicted logits [B, vocab_size, H, W, D]
            target: Ground truth [B, H, W, D]
            terrain_mask: Optional terrain mask [B, H, W, D]

        Returns:
            Dictionary with 'loss' key
        """
        batch_size = logits.shape[0]
        vocab_size = logits.shape[1]

        logits_flat = logits.permute(0, 2, 3, 4, 1).reshape(-1, vocab_size)
        target_flat = target.reshape(-1)

        # Simple cross-entropy (terrain weighting would require per-sample weights)
        ce_loss = F.cross_entropy(logits_flat, target_flat, reduction='mean')

        return {'loss': ce_loss}


def compute_frequency_weights(
    block_ids: torch.Tensor,
    vocab_size: int,
    smoothing: float = 0.5,
    min_count: int = 1,
) -> torch.Tensor:
    """Compute frequency-based weights for cross-entropy loss.

    Weights are computed as: weight[i] = (total_count / count[i]) ** smoothing

    This upweights rare blocks to prevent the model from ignoring them.
    Common choices for smoothing:
    - 0.5 (sqrt): Moderate upweighting (recommended)
    - 1.0 (linear): Aggressive upweighting (may destabilize training)
    - 0.25: Conservative upweighting

    Args:
        block_ids: Tensor of block IDs from training data, any shape
        vocab_size: Total vocabulary size (e.g., 3717)
        smoothing: Exponent for frequency weighting. Default: 0.5 (sqrt)
        min_count: Minimum count to avoid division by zero. Default: 1

    Returns:
        Tensor of shape [vocab_size] with frequency weights

    Example:
        >>> # Load all training data
        >>> train_ids = torch.cat([load_h5(f)['build'] for f in train_files])
        >>>
        >>> # Compute weights
        >>> freq_weights = compute_frequency_weights(
        ...     train_ids,
        ...     vocab_size=3717,
        ...     smoothing=0.5
        ... )
        >>>
        >>> # Use in loss
        >>> criterion = FrequencyWeightedLoss(freq_weights, frequency_cap=5.0)
    """
    # Count occurrences of each block ID
    counts = torch.bincount(block_ids.flatten(), minlength=vocab_size)

    # Ensure minimum count (avoid division by zero)
    counts = counts.clamp(min=min_count)

    # Compute weights: (total / count) ** smoothing
    total = counts.sum()
    weights = (total.float() / counts.float()) ** smoothing

    return weights
