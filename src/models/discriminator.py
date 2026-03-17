"""3D PatchGAN Discriminator for Minecraft structure generation.

The discriminator learns to distinguish real Minecraft structures from
generated ones. It provides adversarial supervision that implicitly teaches
"what looks like a real build" without requiring labeled data.

Architecture: 3D PatchGAN
- Input: Block embeddings [B, 40, 32, 32, 32] from Block2Vec
- Output: Patch scores [B, 1, 4, 4, 4] - real/fake per 8x8x8 region

Why PatchGAN?
- Per-patch discrimination captures local structure (walls, corners, materials)
- More efficient than full-image discriminator
- Translational equivariance: same local patterns scored consistently
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class StructureDiscriminator3D(nn.Module):
    """3D PatchGAN discriminator for Minecraft structures.

    Takes Block2Vec embeddings as input (not raw token IDs) for semantic
    understanding of block relationships.

    Args:
        in_channels: Input embedding dimension. Default: 40 (Block2Vec)
        base_channels: Base channel count, doubled at each layer. Default: 64
        use_spectral_norm: Apply spectral normalization for training stability.
            Default: True
    """

    def __init__(
        self,
        in_channels: int = 40,
        base_channels: int = 64,
        use_spectral_norm: bool = True,
    ):
        super().__init__()

        def maybe_sn(module):
            """Apply spectral norm if enabled."""
            return spectral_norm(module) if use_spectral_norm else module

        # Build discriminator layers
        # Each layer: Conv3d -> InstanceNorm (optional) -> LeakyReLU
        # Downsampling: stride=2 halves spatial dimensions

        self.layers = nn.Sequential(
            # [B, 40, 32, 32, 32] -> [B, 64, 16, 16, 16]
            maybe_sn(nn.Conv3d(in_channels, base_channels, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            # [B, 64, 16, 16, 16] -> [B, 128, 8, 8, 8]
            maybe_sn(nn.Conv3d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm3d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # [B, 128, 8, 8, 8] -> [B, 256, 4, 4, 4]
            maybe_sn(nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm3d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # [B, 256, 4, 4, 4] -> [B, 1, 4, 4, 4]
            # Final layer outputs patch-wise real/fake scores
            maybe_sn(nn.Conv3d(base_channels * 4, 1, kernel_size=3, stride=1, padding=1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Block embeddings [B, C, X, Y, Z] where C is embedding dim (40)

        Returns:
            Patch scores [B, 1, 4, 4, 4] - higher = more real
        """
        return self.layers(x)


class MultiScaleDiscriminator3D(nn.Module):
    """Multi-scale 3D discriminator for improved stability.

    Uses multiple discriminators at different scales to capture both
    local and global structure.

    Args:
        in_channels: Input embedding dimension. Default: 40
        base_channels: Base channel count. Default: 64
        num_scales: Number of discriminator scales. Default: 2
        use_spectral_norm: Apply spectral normalization. Default: True
    """

    def __init__(
        self,
        in_channels: int = 40,
        base_channels: int = 64,
        num_scales: int = 2,
        use_spectral_norm: bool = True,
    ):
        super().__init__()

        self.num_scales = num_scales

        # Create discriminator at each scale
        self.discriminators = nn.ModuleList([
            StructureDiscriminator3D(
                in_channels=in_channels,
                base_channels=base_channels,
                use_spectral_norm=use_spectral_norm,
            )
            for _ in range(num_scales)
        ])

        # Downsampling for multi-scale
        self.downsample = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> list:
        """Forward pass at multiple scales.

        Args:
            x: Block embeddings [B, C, X, Y, Z]

        Returns:
            List of patch scores at each scale
        """
        outputs = []
        current_input = x

        for i, disc in enumerate(self.discriminators):
            outputs.append(disc(current_input))
            if i < self.num_scales - 1:
                current_input = self.downsample(current_input)

        return outputs


def hinge_loss_discriminator(
    real_scores: torch.Tensor,
    fake_scores: torch.Tensor,
) -> torch.Tensor:
    """Hinge loss for discriminator.

    More stable than BCE loss for GAN training.

    Args:
        real_scores: Discriminator output for real samples
        fake_scores: Discriminator output for fake samples

    Returns:
        Discriminator loss (to minimize)
    """
    real_loss = F.relu(1.0 - real_scores).mean()
    fake_loss = F.relu(1.0 + fake_scores).mean()
    return real_loss + fake_loss


def hinge_loss_generator(fake_scores: torch.Tensor) -> torch.Tensor:
    """Hinge loss for generator.

    Generator wants discriminator to output high scores for fakes.

    Args:
        fake_scores: Discriminator output for generated samples

    Returns:
        Generator loss (to minimize)
    """
    return -fake_scores.mean()


def least_squares_discriminator(
    real_scores: torch.Tensor,
    fake_scores: torch.Tensor,
) -> torch.Tensor:
    """Least squares loss for discriminator (LSGAN).

    Alternative to hinge loss, sometimes more stable.

    Args:
        real_scores: Discriminator output for real samples
        fake_scores: Discriminator output for fake samples

    Returns:
        Discriminator loss (to minimize)
    """
    real_loss = ((real_scores - 1) ** 2).mean()
    fake_loss = (fake_scores ** 2).mean()
    return 0.5 * (real_loss + fake_loss)


def least_squares_generator(fake_scores: torch.Tensor) -> torch.Tensor:
    """Least squares loss for generator (LSGAN).

    Args:
        fake_scores: Discriminator output for generated samples

    Returns:
        Generator loss (to minimize)
    """
    return 0.5 * ((fake_scores - 1) ** 2).mean()


def compute_gradient_penalty(
    discriminator: nn.Module,
    real_data: torch.Tensor,
    fake_data: torch.Tensor,
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    """Compute gradient penalty for WGAN-GP style training.

    Encourages discriminator to have gradients with norm 1, improving stability.

    Args:
        discriminator: The discriminator network
        real_data: Real samples [B, C, X, Y, Z]
        fake_data: Generated samples [B, C, X, Y, Z]
        lambda_gp: Gradient penalty weight. Default: 10.0

    Returns:
        Gradient penalty loss
    """
    batch_size = real_data.size(0)
    device = real_data.device

    # Random interpolation between real and fake
    alpha = torch.rand(batch_size, 1, 1, 1, 1, device=device)
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates.requires_grad_(True)

    # Discriminator output for interpolated samples
    disc_interpolates = discriminator(interpolates)

    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Compute gradient penalty
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()

    return gradient_penalty


class EmbeddingWrapper(nn.Module):
    """Wrapper to convert token IDs to Block2Vec embeddings for discriminator.

    Args:
        embedding_table: [vocab_size, embed_dim] Block2Vec embeddings
    """

    def __init__(self, embedding_table: torch.Tensor):
        super().__init__()
        self.register_buffer('embeddings', embedding_table)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Convert token IDs to embeddings.

        Args:
            token_ids: [B, X, Y, Z] tensor of block token IDs

        Returns:
            [B, C, X, Y, Z] block embeddings (channels-first for conv)
        """
        # Look up embeddings: [B, X, Y, Z] -> [B, X, Y, Z, C]
        emb = self.embeddings[token_ids]
        # Permute to channels-first: [B, C, X, Y, Z]
        emb = emb.permute(0, 4, 1, 2, 3)
        return emb


class DiscriminatorWithEmbedding(nn.Module):
    """Discriminator that takes token IDs directly and handles embedding.

    Combines EmbeddingWrapper with StructureDiscriminator3D for convenience.

    Args:
        embedding_table: [vocab_size, embed_dim] Block2Vec embeddings
        base_channels: Base channel count. Default: 64
        use_spectral_norm: Apply spectral normalization. Default: True
    """

    def __init__(
        self,
        embedding_table: torch.Tensor,
        base_channels: int = 64,
        use_spectral_norm: bool = True,
    ):
        super().__init__()

        embed_dim = embedding_table.shape[1]
        self.embed_wrapper = EmbeddingWrapper(embedding_table)
        self.discriminator = StructureDiscriminator3D(
            in_channels=embed_dim,
            base_channels=base_channels,
            use_spectral_norm=use_spectral_norm,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass from token IDs.

        Args:
            token_ids: [B, X, Y, Z] tensor of block token IDs

        Returns:
            Patch scores [B, 1, 4, 4, 4]
        """
        emb = self.embed_wrapper(token_ids)
        return self.discriminator(emb)

    def forward_from_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass from pre-computed embeddings.

        Args:
            embeddings: [B, C, X, Y, Z] block embeddings

        Returns:
            Patch scores [B, 1, 4, 4, 4]
        """
        return self.discriminator(embeddings)
