"""Structural coherence losses for Minecraft structure generation.

These losses encode domain knowledge about what real Minecraft builds look like:
- Connectivity: Blocks should be connected, not floating
- Patch Consistency: Local regions should use consistent materials
- Surface Smoothness: Adjacent blocks should have similar types
- Support: Blocks should have support below (gravity awareness)

These losses work without labels by encoding architectural priors.
"""

from typing import Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConnectivityLoss(nn.Module):
    """Penalize isolated floating blocks.

    Real Minecraft structures have connected components. This loss penalizes
    blocks that have no neighbors (isolated in 3D space).

    Args:
        air_tokens: Set of token IDs representing air blocks.
            Default: {102, 576, 3352} (air, cave_air, void_air)
        neighbor_type: Type of neighborhood to consider.
            '6': Face-adjacent only (6 neighbors)
            '26': Full 3x3x3 neighborhood (26 neighbors)
    """

    def __init__(
        self,
        air_tokens: Optional[Set[int]] = None,
        neighbor_type: str = '6',
    ):
        super().__init__()

        if air_tokens is None:
            air_tokens = {102, 576, 3352}
        self.air_tokens = list(air_tokens)

        # Create neighbor counting kernel
        if neighbor_type == '6':
            # Face-adjacent only: 6 neighbors
            kernel = torch.zeros(1, 1, 3, 3, 3)
            kernel[0, 0, 1, 0, 1] = 1  # -y
            kernel[0, 0, 1, 2, 1] = 1  # +y
            kernel[0, 0, 0, 1, 1] = 1  # -x
            kernel[0, 0, 2, 1, 1] = 1  # +x
            kernel[0, 0, 1, 1, 0] = 1  # -z
            kernel[0, 0, 1, 1, 2] = 1  # +z
        else:
            # Full 3x3x3 neighborhood: 26 neighbors
            kernel = torch.ones(1, 1, 3, 3, 3)
            kernel[0, 0, 1, 1, 1] = 0  # Exclude center

        self.register_buffer('kernel', kernel)

    def forward(self, structure: torch.Tensor) -> torch.Tensor:
        """Compute connectivity loss.

        Args:
            structure: [B, 32, 32, 32] tensor of block token IDs

        Returns:
            Scalar loss (lower = better connected)
        """
        device = structure.device

        # Create air mask
        air_tensor = torch.tensor(self.air_tokens, device=device, dtype=structure.dtype)
        non_air = (~torch.isin(structure, air_tensor)).float()  # [B, 32, 32, 32]

        # Count neighbors for each position
        non_air_5d = non_air.unsqueeze(1)  # [B, 1, 32, 32, 32]
        neighbors = F.conv3d(non_air_5d, self.kernel.to(device), padding=1)
        neighbors = neighbors.squeeze(1)  # [B, 32, 32, 32]

        # Isolated = non-air with zero neighbors
        isolated = (neighbors == 0) & (non_air > 0)

        # Normalize by total non-air blocks
        total_non_air = non_air.sum() + 1e-6
        loss = isolated.float().sum() / total_non_air

        return loss


class PatchConsistencyLoss(nn.Module):
    """Penalize high entropy within local patches.

    Real builds use consistent materials in local regions (walls use same
    block type, floors use same material). High entropy within a patch
    means random/noisy block placement.

    Args:
        patch_size: Size of cubic patches. Default: 4
        air_tokens: Set of token IDs representing air blocks.
    """

    def __init__(
        self,
        patch_size: int = 4,
        air_tokens: Optional[Set[int]] = None,
    ):
        super().__init__()
        self.patch_size = patch_size

        if air_tokens is None:
            air_tokens = {102, 576, 3352}
        self.air_tokens = list(air_tokens)

    def forward(self, structure: torch.Tensor) -> torch.Tensor:
        """Compute patch consistency loss.

        Args:
            structure: [B, 32, 32, 32] tensor of block token IDs

        Returns:
            Scalar loss (lower = more consistent materials)
        """
        B = structure.shape[0]
        device = structure.device
        ps = self.patch_size

        # Unfold into patches [B, num_patches_per_dim, num_patches_per_dim, num_patches_per_dim, ps, ps, ps]
        patches = structure.unfold(1, ps, ps).unfold(2, ps, ps).unfold(3, ps, ps)
        # Reshape to [B, num_patches, ps^3]
        num_patches = patches.shape[1] * patches.shape[2] * patches.shape[3]
        patches = patches.reshape(B, num_patches, -1)  # [B, N, 64]

        # Compute entropy per patch using differentiable approximation
        # Use one-hot encoding and average probabilities
        vocab_size = 3717  # Full Minecraft vocabulary

        # For efficiency, compute entropy using unique counts per patch
        total_entropy = torch.tensor(0.0, device=device)

        for b in range(B):
            for p in range(num_patches):
                patch = patches[b, p]  # [64]

                # Count unique blocks
                unique_vals, counts = torch.unique(patch, return_counts=True)

                # Skip if patch is mostly air (non-informative)
                air_tensor = torch.tensor(self.air_tokens, device=device, dtype=patch.dtype)
                non_air_mask = ~torch.isin(unique_vals, air_tensor)
                if non_air_mask.sum() == 0:
                    continue

                # Compute entropy on non-air blocks
                non_air_counts = counts[non_air_mask].float()
                if non_air_counts.sum() == 0:
                    continue

                probs = non_air_counts / non_air_counts.sum()
                entropy = -(probs * torch.log(probs + 1e-10)).sum()
                total_entropy = total_entropy + entropy

        # Normalize by batch size and number of patches
        return total_entropy / (B * num_patches + 1e-6)


class SurfaceSmoothnessLoss(nn.Module):
    """Penalize abrupt transitions between adjacent blocks.

    Uses Block2Vec embeddings to measure semantic similarity between
    adjacent blocks. High gradients mean abrupt/noisy transitions.

    Args:
        block_embeddings: [vocab_size, embed_dim] Block2Vec embedding table
    """

    def __init__(self, block_embeddings: Optional[torch.Tensor] = None):
        super().__init__()
        if block_embeddings is not None:
            self.register_buffer('embeddings', block_embeddings)
        else:
            self.embeddings = None

    def forward(
        self,
        structure: torch.Tensor,
        block_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute surface smoothness loss.

        Args:
            structure: [B, 32, 32, 32] tensor of block token IDs
            block_embeddings: Optional [B, embed_dim, 32, 32, 32] pre-computed embeddings.
                If not provided, uses self.embeddings to look up.

        Returns:
            Scalar loss (lower = smoother transitions)
        """
        if block_embeddings is not None:
            # Use provided embeddings [B, C, X, Y, Z]
            emb = block_embeddings
        elif self.embeddings is not None:
            # Look up embeddings from stored table
            # structure: [B, X, Y, Z] -> [B, X, Y, Z, C]
            emb = self.embeddings[structure]
            # Permute to [B, C, X, Y, Z]
            emb = emb.permute(0, 4, 1, 2, 3)
        else:
            raise ValueError("Either provide block_embeddings or initialize with embedding table")

        # Compute gradient magnitude in all directions
        # |emb[..., i+1, ...] - emb[..., i, ...]|
        dx = (emb[:, :, 1:, :, :] - emb[:, :, :-1, :, :]).abs()
        dy = (emb[:, :, :, 1:, :] - emb[:, :, :, :-1, :]).abs()
        dz = (emb[:, :, :, :, 1:] - emb[:, :, :, :, :-1]).abs()

        # Average gradient magnitude across all dimensions
        smooth_loss = (dx.mean() + dy.mean() + dz.mean()) / 3.0

        return smooth_loss


class SupportLoss(nn.Module):
    """Penalize blocks without support below (Minecraft gravity awareness).

    In Minecraft, most blocks need support below them. This loss penalizes
    floating blocks that have air directly below.

    Args:
        air_tokens: Set of token IDs representing air blocks.
        exclude_bottom: Whether to exclude bottom layer from penalty.
            Default: True (bottom layer is ground-supported)
    """

    def __init__(
        self,
        air_tokens: Optional[Set[int]] = None,
        exclude_bottom: bool = True,
    ):
        super().__init__()

        if air_tokens is None:
            air_tokens = {102, 576, 3352}
        self.air_tokens = list(air_tokens)
        self.exclude_bottom = exclude_bottom

    def forward(self, structure: torch.Tensor) -> torch.Tensor:
        """Compute support loss.

        Args:
            structure: [B, X, Y, Z] tensor of block token IDs
                Assumes Y is the vertical axis (Y=0 is bottom)

        Returns:
            Scalar loss (lower = more blocks have support)
        """
        device = structure.device
        B, X, Y, Z = structure.shape

        # Create air mask
        air_tensor = torch.tensor(self.air_tokens, device=device, dtype=structure.dtype)
        is_air = torch.isin(structure, air_tensor)
        non_air = ~is_air  # [B, X, Y, Z]

        # support_below[x, y, z] = non_air[x, y-1, z]
        # Shift non_air up by 1 in Y dimension
        # Pad bottom with True (ground provides support)
        support_below = F.pad(non_air[:, :, :-1, :], (0, 0, 1, 0), value=True)
        # support_below[:, :, 0, :] = True (ground layer)

        # Unsupported = non-air block with no support below
        unsupported = non_air & ~support_below

        if self.exclude_bottom:
            # Bottom layer (Y=0) is always considered supported
            unsupported[:, :, 0, :] = False

        # Normalize by total non-air blocks
        total_non_air = non_air.float().sum() + 1e-6
        loss = unsupported.float().sum() / total_non_air

        return loss


class StructuralCoherenceLoss(nn.Module):
    """Combined structural coherence loss.

    Combines all structural losses with configurable weights.

    Args:
        lambda_connectivity: Weight for connectivity loss. Default: 0.5
        lambda_patch: Weight for patch consistency loss. Default: 0.3
        lambda_smooth: Weight for surface smoothness loss. Default: 0.1
        lambda_support: Weight for support/gravity loss. Default: 0.2
        air_tokens: Set of token IDs representing air blocks.
        block_embeddings: Optional Block2Vec embedding table for smoothness loss.
    """

    def __init__(
        self,
        lambda_connectivity: float = 0.5,
        lambda_patch: float = 0.3,
        lambda_smooth: float = 0.1,
        lambda_support: float = 0.2,
        air_tokens: Optional[Set[int]] = None,
        block_embeddings: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        self.lambda_connectivity = lambda_connectivity
        self.lambda_patch = lambda_patch
        self.lambda_smooth = lambda_smooth
        self.lambda_support = lambda_support

        # Initialize component losses
        self.connectivity_loss = ConnectivityLoss(air_tokens=air_tokens)
        self.patch_loss = PatchConsistencyLoss(air_tokens=air_tokens)
        self.smooth_loss = SurfaceSmoothnessLoss(block_embeddings=block_embeddings)
        self.support_loss = SupportLoss(air_tokens=air_tokens)

    def forward(
        self,
        structure: torch.Tensor,
        block_embeddings: Optional[torch.Tensor] = None,
    ) -> dict:
        """Compute combined structural coherence loss.

        Args:
            structure: [B, 32, 32, 32] tensor of block token IDs
            block_embeddings: Optional [B, C, 32, 32, 32] Block2Vec embeddings

        Returns:
            Dictionary with:
                - 'total': Combined weighted loss
                - 'connectivity': Connectivity component (detached)
                - 'patch': Patch consistency component (detached)
                - 'smooth': Smoothness component (detached)
                - 'support': Support component (detached)
        """
        L_conn = self.connectivity_loss(structure)
        L_patch = self.patch_loss(structure)
        L_support = self.support_loss(structure)

        # Smoothness requires embeddings
        if block_embeddings is not None or self.smooth_loss.embeddings is not None:
            L_smooth = self.smooth_loss(structure, block_embeddings)
        else:
            L_smooth = torch.tensor(0.0, device=structure.device)

        total = (
            self.lambda_connectivity * L_conn +
            self.lambda_patch * L_patch +
            self.lambda_smooth * L_smooth +
            self.lambda_support * L_support
        )

        return {
            'total': total,
            'connectivity': L_conn.detach(),
            'patch': L_patch.detach(),
            'smooth': L_smooth.detach() if torch.is_tensor(L_smooth) else L_smooth,
            'support': L_support.detach(),
        }
