"""Loss functions for VQ-VAE training.

This module contains custom loss functions for Minecraft structure generation
with VQ-VAE models, including frequency-weighted losses, volume penalties,
perceptual losses, and generative losses (adversarial + structural).
"""

from typing import Dict, Optional, Set, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from .discriminator import StructureDiscriminator3D
    from .structural_losses import StructuralCoherenceLoss


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


class V11Loss(nn.Module):
    """Asymmetric loss for v11 that heavily penalizes structure erasure.

    Designed to fix v10's volume ratio (0.633x) and recall (62.82%) issues.
    Both problems stem from the model predicting too much air and erasing
    37% of actual structures.

    Key innovations:
    1. Asymmetric per-voxel weighting: 20x penalty for structure→air errors
    2. Soft volume penalty: Differentiable through softmax (unlike VolumeHead)
    3. Explicit false air penalty: Margin-based logit control

    Args:
        air_tokens: Set of token IDs representing air blocks.
            Default: {102, 576, 3352} (air, cave_air, void_air)
        gamma: Focal loss gamma parameter. Default: 2.0
        structure_to_air_weight: Penalty multiplier for erasing structures. Default: 20.0
        air_to_structure_weight: Penalty for hallucinating blocks. Default: 1.0
        false_air_penalty_weight: Weight for logit-level false air penalty. Default: 10.0
        soft_volume_weight: Weight for soft volume L1 penalty. Default: 5.0
        frequency_weights: Optional frequency weights for CE loss.
        frequency_cap: Cap for frequency weights. Default: 5.0

    Example:
        >>> criterion = V11Loss(
        ...     air_tokens={102, 576, 3352},
        ...     structure_to_air_weight=20.0,
        ...     false_air_penalty_weight=10.0,
        ...     soft_volume_weight=5.0,
        ... )
        >>> loss_dict = criterion(logits, target)
        >>> loss_dict['loss'].backward()
    """

    def __init__(
        self,
        air_tokens: Optional[Set[int]] = None,
        gamma: float = 2.0,
        structure_to_air_weight: float = 20.0,
        air_to_structure_weight: float = 1.0,
        false_air_penalty_weight: float = 10.0,
        soft_volume_weight: float = 5.0,
        frequency_weights: Optional[torch.Tensor] = None,
        frequency_cap: float = 5.0,
    ):
        super().__init__()

        # Air tokens (minecraft:air, minecraft:cave_air, minecraft:void_air)
        if air_tokens is None:
            air_tokens = {102, 576, 3352}
        self.air_tokens = list(air_tokens)

        self.gamma = gamma
        self.structure_to_air_weight = structure_to_air_weight
        self.air_to_structure_weight = air_to_structure_weight
        self.false_air_penalty_weight = false_air_penalty_weight
        self.soft_volume_weight = soft_volume_weight

        # Optional frequency weights
        if frequency_weights is not None:
            clamped = frequency_weights.clamp(max=frequency_cap)
            self.register_buffer('freq_weights', clamped)
        else:
            self.freq_weights = None

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute asymmetric loss with structure protection.

        Args:
            logits: Predicted logits [B, vocab_size, H, W, D]
            target: Ground truth block IDs [B, H, W, D]

        Returns:
            Dictionary with loss components and metrics
        """
        device = logits.device
        B, C, X, Y, Z = logits.shape
        total_voxels = X * Y * Z

        # Create air token tensor
        air_tensor = torch.tensor(self.air_tokens, device=device, dtype=target.dtype)

        # Ground truth masks
        gt_is_air = torch.isin(target, air_tensor)
        gt_is_struct = ~gt_is_air

        # Flatten for processing
        logits_flat = logits.permute(0, 2, 3, 4, 1).reshape(-1, C)  # [N, C]
        target_flat = target.reshape(-1)  # [N]
        gt_is_struct_flat = gt_is_struct.reshape(-1)  # [N]
        gt_is_air_flat = gt_is_air.reshape(-1)  # [N]

        # Get predictions for weight computation
        pred = logits_flat.argmax(dim=1)  # [N]
        pred_is_air = torch.isin(pred, air_tensor)
        pred_is_struct = ~pred_is_air

        # === 1. Asymmetric Weighted Focal Cross-Entropy ===
        # Create per-voxel weights based on error type
        weights = torch.ones(logits_flat.shape[0], device=device, dtype=torch.float)

        # HEAVY penalty: structure in GT, predicted as air (ERASURE)
        # This is the critical error type that v10 made too often
        struct_to_air_mask = gt_is_struct_flat & pred_is_air
        weights[struct_to_air_mask] = self.structure_to_air_weight

        # Normal penalty: air in GT, predicted as structure (HALLUCINATION)
        air_to_struct_mask = gt_is_air_flat & pred_is_struct
        weights[air_to_struct_mask] = self.air_to_structure_weight

        # Compute focal cross-entropy
        log_probs = F.log_softmax(logits_flat, dim=1)
        probs = torch.exp(log_probs)
        p_t = probs.gather(1, target_flat.unsqueeze(1)).squeeze(1)  # [N]
        focal_weight = (1 - p_t) ** self.gamma  # [N]

        # Per-sample CE loss
        if self.freq_weights is not None:
            ce_per_sample = F.cross_entropy(
                logits_flat, target_flat,
                weight=self.freq_weights, reduction='none'
            )
        else:
            ce_per_sample = F.cross_entropy(
                logits_flat, target_flat, reduction='none'
            )

        # Combined weighted focal loss
        weighted_focal_loss = (focal_weight * weights * ce_per_sample).mean()

        # === 2. False Air Penalty (Logit-Level) ===
        # At GT structure locations, penalize if air logits > non-air logits
        if gt_is_struct_flat.any():
            struct_logits = logits_flat[gt_is_struct_flat]  # [N_struct, C]

            # Create air mask for vocab
            air_mask = torch.zeros(C, device=device, dtype=torch.bool)
            for t in self.air_tokens:
                if t < C:
                    air_mask[t] = True

            # Get max air logit at each structure location
            if air_mask.any():
                air_logits_at_struct = struct_logits[:, air_mask].max(dim=1)[0]  # [N_struct]
            else:
                air_logits_at_struct = torch.zeros(struct_logits.shape[0], device=device)

            # Get max non-air logit at each structure location
            non_air_logits = struct_logits.clone()
            non_air_logits[:, air_mask] = float('-inf')
            max_non_air = non_air_logits.max(dim=1)[0]  # [N_struct]

            # Penalize when air logit > non-air logit (margin = 1.0)
            margin = 1.0
            violation = F.relu(air_logits_at_struct - max_non_air + margin)
            false_air_penalty = violation.mean()
        else:
            false_air_penalty = torch.tensor(0.0, device=device)

        # === 3. Soft Volume Penalty (Differentiable through softmax) ===
        # Unlike v10's VolumeHead, gradients flow through the decoder
        probs_full = F.softmax(logits_flat, dim=1)  # [N, C]

        # Sum probabilities of non-air tokens = soft predicted volume
        air_mask_full = torch.zeros(C, device=device, dtype=torch.bool)
        for t in self.air_tokens:
            if t < C:
                air_mask_full[t] = True

        # Set air probs to 0, sum the rest = soft structure probability
        non_air_probs = probs_full.clone()
        non_air_probs[:, air_mask_full] = 0
        soft_pred_volume = non_air_probs.sum(dim=1).mean()  # Per-voxel average

        # GT volume (normalized)
        gt_volume = gt_is_struct_flat.float().mean()

        # L1 penalty on volume deviation
        soft_volume_loss = torch.abs(soft_pred_volume - gt_volume)

        # === Total Loss ===
        total_loss = (
            weighted_focal_loss +
            self.false_air_penalty_weight * false_air_penalty +
            self.soft_volume_weight * soft_volume_loss
        )

        # === Metrics (detached for logging) ===
        with torch.no_grad():
            correct = (pred == target_flat)
            accuracy = correct.float().mean()

            # Building accuracy (only on GT structure voxels)
            if gt_is_struct_flat.any():
                building_acc = correct[gt_is_struct_flat].float().mean()
            else:
                building_acc = torch.tensor(0.0, device=device)

            # Air accuracy
            if gt_is_air_flat.any():
                air_acc = correct[gt_is_air_flat].float().mean()
            else:
                air_acc = torch.tensor(0.0, device=device)

            # Recall = (structure preserved) / (total GT structure)
            if gt_is_struct_flat.any():
                recall = pred_is_struct[gt_is_struct_flat].float().mean()
            else:
                recall = torch.tensor(0.0, device=device)

            # False air rate = (GT struct predicted as air) / (total GT struct)
            if gt_is_struct_flat.any():
                false_air_rate = pred_is_air[gt_is_struct_flat].float().mean()
            else:
                false_air_rate = torch.tensor(0.0, device=device)

            # Hard volume ratio
            pred_volume = pred_is_struct.float().sum()
            gt_volume_hard = gt_is_struct_flat.float().sum()
            volume_ratio = pred_volume / (gt_volume_hard + 1e-6)

            # Precision = (correct struct predictions) / (total struct predictions)
            if pred_is_struct.any():
                precision = correct[pred_is_struct & gt_is_struct_flat].sum().float() / pred_is_struct.sum()
            else:
                precision = torch.tensor(0.0, device=device)

        return {
            'loss': total_loss,
            'focal_loss': weighted_focal_loss.detach(),
            'false_air_penalty': false_air_penalty.detach(),
            'soft_volume_loss': soft_volume_loss.detach(),
            'soft_pred_volume': soft_pred_volume.detach(),
            'gt_volume': gt_volume.detach(),
            'volume_ratio': volume_ratio,
            'accuracy': accuracy,
            'building_acc': building_acc,
            'air_acc': air_acc,
            'recall': recall,
            'false_air_rate': false_air_rate,
            'precision': precision,
        }


class GenerativeLoss(nn.Module):
    """Combined loss for VQ-VAE training with generative quality.

    Combines reconstruction losses with:
    1. Structural coherence losses (connectivity, patch consistency, smoothness, support)
    2. Adversarial loss from 3D PatchGAN discriminator

    This loss function teaches the model not just to reconstruct, but to generate
    structures that "look like real Minecraft builds" without requiring labels.

    Args:
        frequency_weights: Tensor of shape [vocab_size] with frequency weights.
        discriminator: Optional 3D PatchGAN discriminator for adversarial training.
        block_embeddings: [vocab_size, embed_dim] Block2Vec embeddings for smoothness loss.
        lambda_adv: Weight for adversarial loss. Default: 0.1 (increase over training)
        lambda_connectivity: Weight for connectivity loss. Default: 0.5
        lambda_patch: Weight for patch consistency loss. Default: 0.3
        lambda_smooth: Weight for surface smoothness loss. Default: 0.1
        lambda_support: Weight for support/gravity loss. Default: 0.2
        frequency_cap: Maximum frequency weight. Default: 5.0
        volume_penalty_weight: Weight for volume penalty. Default: 10.0
        false_air_penalty_weight: Weight for false air penalty. Default: 5.0
        perceptual_weight: Weight for latent perceptual loss. Default: 0.1
        air_tokens: Set of token IDs for air blocks. Default: {102, 576, 3352}
    """

    def __init__(
        self,
        frequency_weights: torch.Tensor,
        discriminator: Optional[nn.Module] = None,
        block_embeddings: Optional[torch.Tensor] = None,
        lambda_adv: float = 0.1,
        lambda_connectivity: float = 0.5,
        lambda_patch: float = 0.3,
        lambda_smooth: float = 0.1,
        lambda_support: float = 0.2,
        frequency_cap: float = 5.0,
        volume_penalty_weight: float = 10.0,
        false_air_penalty_weight: float = 5.0,
        perceptual_weight: float = 0.1,
        air_tokens: Optional[Set[int]] = None,
    ):
        super().__init__()

        # Store discriminator reference (not as submodule to avoid double optimization)
        self.discriminator = discriminator

        # Loss weights
        self.lambda_adv = lambda_adv
        self.lambda_connectivity = lambda_connectivity
        self.lambda_patch = lambda_patch
        self.lambda_smooth = lambda_smooth
        self.lambda_support = lambda_support
        self.volume_penalty_weight = volume_penalty_weight
        self.false_air_penalty_weight = false_air_penalty_weight
        self.perceptual_weight = perceptual_weight

        # Clamp and store frequency weights
        clamped_weights = frequency_weights.clamp(max=frequency_cap)
        self.register_buffer('freq_weights', clamped_weights)

        # Store block embeddings for smoothness loss
        if block_embeddings is not None:
            self.register_buffer('block_embeddings', block_embeddings)
        else:
            self.block_embeddings = None

        # Air tokens
        if air_tokens is None:
            air_tokens = {102, 576, 3352}
        self.air_tokens = air_tokens

        # Import structural losses (lazy to avoid circular imports)
        from .structural_losses import (
            ConnectivityLoss,
            PatchConsistencyLoss,
            SurfaceSmoothnessLoss,
            SupportLoss,
        )

        self.connectivity_loss = ConnectivityLoss(air_tokens=air_tokens)
        self.patch_loss = PatchConsistencyLoss(air_tokens=air_tokens)
        self.smooth_loss = SurfaceSmoothnessLoss(block_embeddings=block_embeddings)
        self.support_loss = SupportLoss(air_tokens=air_tokens)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        z_q: torch.Tensor,
        compute_adversarial: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined generative loss.

        Args:
            logits: Predicted logits [B, vocab_size, H, W, D]
            target: Ground truth block IDs [B, H, W, D]
            z_q: Quantized latent codes [B, C, H_latent, W_latent, D_latent]
            compute_adversarial: Whether to compute adversarial loss.
                Set to False during discriminator update. Default: True

        Returns:
            Dictionary with all loss components and metrics.
        """
        device = logits.device
        B = logits.shape[0]

        # Get predictions
        reconstructed = logits.argmax(dim=1)  # [B, H, W, D]

        # === 1. Reconstruction losses (from FrequencyWeightedLoss) ===
        vocab_size = logits.shape[1]
        logits_flat = logits.permute(0, 2, 3, 4, 1).reshape(-1, vocab_size)
        target_flat = target.reshape(-1)

        ce_loss = F.cross_entropy(
            logits_flat,
            target_flat,
            weight=self.freq_weights,
            reduction='mean'
        )

        # Volume and false air penalties
        air_tokens_tensor = torch.tensor(
            list(self.air_tokens), device=device, dtype=target.dtype
        )
        gt_is_air = torch.isin(target, air_tokens_tensor)
        gt_is_structure = ~gt_is_air

        # Create air mask for logits
        air_mask = torch.zeros(vocab_size, device=device, dtype=torch.bool)
        for t in self.air_tokens:
            if t < vocab_size:
                air_mask[t] = True

        # Max air and non-air logits per voxel
        air_logits_list = [logits[:, t, :, :, :] for t in self.air_tokens if t < vocab_size]
        if air_logits_list:
            air_logits = torch.stack(air_logits_list, dim=0)
            max_air_logit = air_logits.max(dim=0)[0]
        else:
            max_air_logit = torch.zeros_like(logits[:, 0, :, :, :])

        logits_non_air = logits.clone()
        logits_non_air[:, air_mask, :, :, :] = float('-inf')
        max_non_air_logit = logits_non_air.max(dim=1)[0]

        # Volume loss: penalize non-air predictions at air locations
        if gt_is_air.any():
            margin = 1.0
            violation = F.relu(max_non_air_logit[gt_is_air] - max_air_logit[gt_is_air] + margin)
            volume_loss = violation.mean()
        else:
            volume_loss = torch.tensor(0.0, device=device)

        # False air loss: penalize air predictions at structure locations
        if gt_is_structure.any():
            margin = 1.0
            violation = F.relu(max_air_logit[gt_is_structure] - max_non_air_logit[gt_is_structure] + margin)
            false_air_loss = violation.mean()
        else:
            false_air_loss = torch.tensor(0.0, device=device)

        # Perceptual loss on latent
        perceptual_loss = self._compute_perceptual_loss(z_q)

        # === 2. Structural coherence losses ===
        L_connectivity = self.connectivity_loss(reconstructed)
        L_patch = self.patch_loss(reconstructed)
        L_support = self.support_loss(reconstructed)

        # Smoothness requires embeddings
        if self.block_embeddings is not None:
            recon_emb = self.block_embeddings[reconstructed]  # [B, H, W, D, C]
            recon_emb = recon_emb.permute(0, 4, 1, 2, 3)  # [B, C, H, W, D]
            L_smooth = self.smooth_loss(reconstructed, recon_emb)
        else:
            L_smooth = torch.tensor(0.0, device=device)

        # === 3. Adversarial loss ===
        L_adv = torch.tensor(0.0, device=device)
        if compute_adversarial and self.discriminator is not None and self.block_embeddings is not None:
            # Get embeddings for reconstructed structure
            recon_emb = self.block_embeddings[reconstructed].permute(0, 4, 1, 2, 3)
            # Discriminator forward
            d_recon = self.discriminator(recon_emb)
            # Generator wants discriminator to think reconstructed is real
            L_adv = -d_recon.mean()  # Hinge loss generator

        # === Combine all losses ===
        total_loss = (
            ce_loss +
            self.volume_penalty_weight * volume_loss +
            self.false_air_penalty_weight * false_air_loss +
            self.perceptual_weight * perceptual_loss +
            self.lambda_connectivity * L_connectivity +
            self.lambda_patch * L_patch +
            self.lambda_smooth * L_smooth +
            self.lambda_support * L_support +
            self.lambda_adv * L_adv
        )

        # === Compute metrics ===
        with torch.no_grad():
            pred_is_air = torch.isin(reconstructed, air_tokens_tensor)
            pred_volume = (~pred_is_air).float().sum()
            gt_volume = gt_is_structure.float().sum()
            volume_ratio = pred_volume / (gt_volume + 1e-6)

            # Recall
            structure_preserved = gt_is_structure & (~pred_is_air)
            recall = structure_preserved.float().sum() / (gt_is_structure.float().sum() + 1e-6)

            # Accuracy
            correct = (reconstructed == target)
            building_acc = correct[gt_is_structure].float().mean() if gt_is_structure.any() else torch.tensor(0.0)

        return {
            'loss': total_loss,
            'ce_loss': ce_loss.detach(),
            'volume_loss': volume_loss.detach(),
            'false_air_loss': false_air_loss.detach() if torch.is_tensor(false_air_loss) else false_air_loss,
            'perceptual_loss': perceptual_loss.detach(),
            'connectivity_loss': L_connectivity.detach(),
            'patch_loss': L_patch.detach(),
            'smooth_loss': L_smooth.detach() if torch.is_tensor(L_smooth) else L_smooth,
            'support_loss': L_support.detach(),
            'adv_loss': L_adv.detach() if torch.is_tensor(L_adv) else L_adv,
            'volume_ratio': volume_ratio,
            'recall': recall,
            'building_acc': building_acc,
        }

    def _compute_perceptual_loss(self, z_q: torch.Tensor) -> torch.Tensor:
        """Compute spatial smoothness on latent codes."""
        diff_h = (z_q[:, :, 1:, :, :] - z_q[:, :, :-1, :, :]).abs().mean()
        diff_w = (z_q[:, :, :, 1:, :] - z_q[:, :, :, :-1, :]).abs().mean()
        diff_d = (z_q[:, :, :, :, 1:] - z_q[:, :, :, :, :-1]).abs().mean()
        return (diff_h + diff_w + diff_d) / 3.0

    def set_lambda_adv(self, value: float):
        """Update adversarial loss weight (for scheduling)."""
        self.lambda_adv = value
