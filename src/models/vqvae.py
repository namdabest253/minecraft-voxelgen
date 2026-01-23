"""
VQ-VAE: Vector Quantized Variational AutoEncoder for Minecraft structures.

Compresses 32x32x32 block structures into a discrete latent representation
using a learned codebook of spatial patterns. The discrete codes can later
be predicted by a text-conditioned model for generation.

Architecture:
    Input (32x32x32 blocks)
        → Embed each block using Block2Vec (32x32x32x32)
        → Encoder CNN (compress to 8x8x8x256)
        → Vector Quantization (map each position to nearest codebook entry)
        → Decoder CNN (expand back to 32x32x32)
        → Predict block type at each position

Key improvements in this version:
    - EMA codebook updates (no gradient descent on codebook)
    - Dead code reset (reinitialize unused codes)
    - Higher commitment cost for better codebook utilization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any


class VectorQuantizerEMA(nn.Module):
    """Vector Quantization with Exponential Moving Average codebook updates.

    This version uses EMA to update the codebook instead of gradient descent,
    which helps prevent codebook collapse. Also includes dead code reset.

    The codebook is updated as:
        N_i = decay * N_i + (1 - decay) * n_i  (count of assignments to code i)
        m_i = decay * m_i + (1 - decay) * sum(z_e assigned to i)
        e_i = m_i / N_i  (codebook vector)

    Args:
        num_embeddings: Number of codebook entries (K)
        embedding_dim: Dimension of each codebook vector (D)
        commitment_cost: Weight for commitment loss (β), typically 0.25-1.0
        decay: EMA decay rate (0.99 is typical)
        epsilon: Small constant to avoid division by zero
        dead_code_threshold: Reset codes used less than this fraction
    """

    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 256,
        commitment_cost: float = 0.5,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        dead_code_threshold: float = 0.01,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.dead_code_threshold = dead_code_threshold

        # Codebook embeddings (not learned via gradients)
        # Shape: [num_embeddings, embedding_dim]
        self.register_buffer("codebook", torch.randn(num_embeddings, embedding_dim))

        # EMA cluster size (count of assignments to each code)
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))

        # EMA sum of encoder outputs assigned to each code
        self.register_buffer("ema_embed_sum", torch.randn(num_embeddings, embedding_dim))

        # Track if codebook has been initialized from data
        self.register_buffer("initialized", torch.tensor(False))

        # Track usage for dead code detection
        self.register_buffer("usage_count", torch.zeros(num_embeddings))

    def _init_codebook(self, flat_z_e: torch.Tensor):
        """Initialize codebook from first batch of encoder outputs."""
        # Sample random encoder outputs as initial codebook
        n_samples = flat_z_e.shape[0]
        if n_samples >= self.num_embeddings:
            # Randomly select encoder outputs
            indices = torch.randperm(n_samples)[:self.num_embeddings]
            self.codebook.data.copy_(flat_z_e[indices])
        else:
            # Not enough samples, use what we have + random
            self.codebook.data[:n_samples].copy_(flat_z_e)

        # Initialize EMA buffers
        self.ema_cluster_size.fill_(1.0)
        self.ema_embed_sum.data.copy_(self.codebook.data)
        self.initialized.fill_(True)

    def _reset_dead_codes(self, flat_z_e: torch.Tensor, encoding_indices: torch.Tensor):
        """Reset codebook entries that are rarely used."""
        # Count usage in this batch
        batch_usage = torch.bincount(
            encoding_indices.view(-1),
            minlength=self.num_embeddings
        ).float()

        # Update cumulative usage
        self.usage_count = self.decay * self.usage_count + (1 - self.decay) * batch_usage

        # Find dead codes (used less than threshold)
        avg_usage = self.usage_count.sum() / self.num_embeddings
        dead_mask = self.usage_count < (avg_usage * self.dead_code_threshold)
        n_dead = dead_mask.sum().item()

        if n_dead > 0 and flat_z_e.shape[0] > 0:
            # Sample random encoder outputs to replace dead codes
            n_samples = min(int(n_dead), flat_z_e.shape[0])
            indices = torch.randperm(flat_z_e.shape[0], device=flat_z_e.device)[:n_samples]
            samples = flat_z_e[indices]

            # Find which codes to reset
            dead_indices = torch.where(dead_mask)[0][:n_samples]

            # Reset the dead codes
            self.codebook.data[dead_indices] = samples
            self.ema_cluster_size[dead_indices] = 1.0
            self.ema_embed_sum.data[dead_indices] = samples
            self.usage_count[dead_indices] = avg_usage

        return n_dead

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize encoder outputs to nearest codebook entries.

        Args:
            z_e: Encoder output [batch, channels, depth, height, width]
                 channels should equal embedding_dim

        Returns:
            z_q: Quantized output (same shape as z_e)
            vq_loss: Commitment loss only (codebook updated via EMA)
            encoding_indices: Which codebook entry was selected [batch, D, H, W]
        """
        # z_e shape: [B, C, D, H, W] where C = embedding_dim

        # Reshape to [B*D*H*W, C] for distance calculation
        z_e_permuted = z_e.permute(0, 2, 3, 4, 1).contiguous()  # [B, D, H, W, C]
        flat_z_e = z_e_permuted.view(-1, self.embedding_dim)  # [B*D*H*W, C]

        # Initialize codebook from first batch
        if not self.initialized and self.training:
            self._init_codebook(flat_z_e)

        # Compute distances to all codebook entries
        # ||z_e - codebook||² = ||z_e||² + ||codebook||² - 2 * z_e · codebook
        z_e_sq = (flat_z_e ** 2).sum(dim=1, keepdim=True)  # [N, 1]
        codebook_sq = (self.codebook ** 2).sum(dim=1, keepdim=True).t()  # [1, K]
        dot_product = torch.mm(flat_z_e, self.codebook.t())  # [N, K]
        distances = z_e_sq + codebook_sq - 2 * dot_product  # [N, K]

        # Find nearest codebook entry for each position
        encoding_indices = distances.argmin(dim=1)  # [N]

        # Look up the codebook vectors
        z_q_flat = F.embedding(encoding_indices, self.codebook)  # [N, C]

        # === EMA Codebook Update (only during training) ===
        if self.training:
            # One-hot encode the assignments
            encodings = F.one_hot(encoding_indices, self.num_embeddings).float()  # [N, K]

            # Update EMA cluster size
            batch_cluster_size = encodings.sum(0)  # [K]
            self.ema_cluster_size.data.mul_(self.decay).add_(
                batch_cluster_size, alpha=1 - self.decay
            )

            # Update EMA embedding sum
            batch_embed_sum = encodings.t() @ flat_z_e  # [K, C]
            self.ema_embed_sum.data.mul_(self.decay).add_(
                batch_embed_sum, alpha=1 - self.decay
            )

            # Laplace smoothing to avoid division by zero
            n = self.ema_cluster_size.sum()
            smoothed_cluster_size = (
                (self.ema_cluster_size + self.epsilon) /
                (n + self.num_embeddings * self.epsilon) * n
            )

            # Update codebook
            self.codebook.data.copy_(
                self.ema_embed_sum / smoothed_cluster_size.unsqueeze(1)
            )

            # Reset dead codes periodically
            self._reset_dead_codes(flat_z_e, encoding_indices)

        # Reshape back to [B, D, H, W, C]
        z_q_permuted = z_q_flat.view(z_e_permuted.shape)

        # === Commitment Loss ===
        # Only commitment loss - codebook is updated via EMA, not gradients
        commitment_loss = F.mse_loss(z_e_permuted, z_q_permuted.detach())
        vq_loss = self.commitment_cost * commitment_loss

        # === Straight-Through Estimator ===
        z_q_st = z_e_permuted + (z_q_permuted - z_e_permuted).detach()

        # Permute back to [B, C, D, H, W]
        z_q = z_q_st.permute(0, 4, 1, 2, 3).contiguous()

        # Reshape encoding indices to spatial grid
        spatial_shape = z_e_permuted.shape[:-1]  # [B, D, H, W]
        encoding_indices = encoding_indices.view(spatial_shape)

        return z_q, vq_loss, encoding_indices

    def get_codebook_usage(self) -> Tuple[int, float]:
        """Get codebook utilization statistics.

        Returns:
            Tuple of (num_used_codes, usage_fraction)
        """
        avg_usage = self.usage_count.sum() / self.num_embeddings
        used_mask = self.usage_count > (avg_usage * self.dead_code_threshold)
        num_used = used_mask.sum().item()
        return int(num_used), num_used / self.num_embeddings


# Keep original VectorQuantizer for backwards compatibility
class VectorQuantizer(nn.Module):
    """Original Vector Quantization layer (gradient-based, no EMA).

    Kept for backwards compatibility. For new training, use VectorQuantizerEMA.
    """

    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 256,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_e_permuted = z_e.permute(0, 2, 3, 4, 1).contiguous()
        flat_z_e = z_e_permuted.view(-1, self.embedding_dim)

        z_e_sq = (flat_z_e ** 2).sum(dim=1, keepdim=True)
        codebook_sq = (self.codebook.weight ** 2).sum(dim=1, keepdim=True).t()
        dot_product = torch.mm(flat_z_e, self.codebook.weight.t())
        distances = z_e_sq + codebook_sq - 2 * dot_product

        encoding_indices = distances.argmin(dim=1)
        z_q_flat = self.codebook(encoding_indices)
        z_q_permuted = z_q_flat.view(z_e_permuted.shape)

        codebook_loss = F.mse_loss(z_q_permuted, z_e_permuted.detach())
        commitment_loss = F.mse_loss(z_e_permuted, z_q_permuted.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        z_q_st = z_e_permuted + (z_q_permuted - z_e_permuted).detach()
        z_q = z_q_st.permute(0, 4, 1, 2, 3).contiguous()

        spatial_shape = z_e_permuted.shape[:-1]
        encoding_indices = encoding_indices.view(spatial_shape)

        return z_q, vq_loss, encoding_indices


class ResidualBlock3D(nn.Module):
    """3D Residual block for encoder/decoder.

    Uses pre-activation (BN-ReLU-Conv) residual connections.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)

        # Skip connection with 1x1 conv if dimensions change
        self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        identity = self.skip(x)

        out = self.bn1(x)
        out = F.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)

        return out + identity


class Encoder(nn.Module):
    """3D CNN Encoder that compresses 32x32x32 to 8x8x8.

    Architecture:
        32x32x32 → Conv+Pool → 16x16x16 → Conv+Pool → 8x8x8

    Args:
        in_channels: Input channels (block embedding dim, e.g., 32)
        hidden_dims: Hidden layer dimensions [64, 128, 256]
        latent_dim: Output channels (should match codebook embedding_dim)
    """

    def __init__(
        self,
        in_channels: int = 32,
        hidden_dims: list = None,
        latent_dim: int = 256,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 128, 256]

        layers = []
        current_channels = in_channels

        for hidden_dim in hidden_dims:
            layers.extend([
                # Downsample by 2x in each dimension
                nn.Conv3d(current_channels, hidden_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU(inplace=True),
                # Residual block for more capacity
                ResidualBlock3D(hidden_dim, hidden_dim),
            ])
            current_channels = hidden_dim

        # Final projection to latent dimension
        layers.append(nn.Conv3d(current_channels, latent_dim, kernel_size=3, padding=1))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation.

        Args:
            x: Input tensor [batch, channels, 32, 32, 32]

        Returns:
            Latent tensor [batch, latent_dim, 4, 4, 4]
        """
        return self.encoder(x)


class EncoderV4(nn.Module):
    """3D CNN Encoder for VQ-VAE v4 that compresses 32x32x32 to 8x8x8.

    Key difference from original Encoder:
        - Only 2 downsampling stages (32→16→8) instead of 3 (32→16→8→4)
        - Results in 64:1 compression instead of 512:1
        - Extra residual blocks at 8x8x8 for capacity

    Architecture:
        32x32x32 → Conv4/2 → 16x16x16 → Conv4/2 → 8x8x8 → ResBlocks → 8x8x8

    Args:
        in_channels: Input channels (block embedding dim, e.g., 32)
        hidden_dims: Hidden layer dimensions [96, 192] (2 stages)
        latent_dim: Output channels (should match codebook embedding_dim)
        dropout: Dropout rate for regularization
    """

    def __init__(
        self,
        in_channels: int = 32,
        hidden_dims: list = None,
        latent_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [96, 192]  # Only 2 stages for 32→8

        layers = []
        current_channels = in_channels

        for hidden_dim in hidden_dims:
            layers.extend([
                # Downsample by 2x in each dimension
                nn.Conv3d(current_channels, hidden_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout3d(dropout),
                # Residual block for more capacity
                ResidualBlock3D(hidden_dim, hidden_dim),
            ])
            current_channels = hidden_dim

        # Extra residual blocks at final resolution (8x8x8) for capacity
        # This compensates for having fewer downsampling stages
        layers.extend([
            ResidualBlock3D(current_channels, current_channels),
            ResidualBlock3D(current_channels, current_channels),
        ])

        # Final projection to latent dimension
        layers.append(nn.Conv3d(current_channels, latent_dim, kernel_size=3, padding=1))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation.

        Args:
            x: Input tensor [batch, channels, 32, 32, 32]

        Returns:
            Latent tensor [batch, latent_dim, 8, 8, 8]
        """
        return self.encoder(x)


class DecoderV4(nn.Module):
    """3D CNN Decoder for VQ-VAE v4 that expands 8x8x8 back to 32x32x32.

    Key difference from original Decoder:
        - Only 2 upsampling stages (8→16→32) instead of 3 (4→8→16→32)
        - Takes 8x8x8 input instead of 4x4x4
        - Extra residual blocks at 8x8x8 for capacity

    Architecture:
        8x8x8 → ResBlocks → ConvT4/2 → 16x16x16 → ConvT4/2 → 32x32x32 → Predict blocks

    Args:
        latent_dim: Input channels (codebook embedding_dim)
        hidden_dims: Hidden layer dimensions [192, 96] (reversed from encoder)
        num_blocks: Number of block types to predict
        dropout: Dropout rate for regularization
    """

    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dims: list = None,
        num_blocks: int = 3717,
        dropout: float = 0.1,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [192, 96]  # Reverse of encoder (2 stages)

        layers = []
        current_channels = latent_dim

        # Extra residual blocks at input resolution (8x8x8) for capacity
        layers.extend([
            ResidualBlock3D(current_channels, current_channels),
            ResidualBlock3D(current_channels, current_channels),
        ])

        for hidden_dim in hidden_dims:
            layers.extend([
                # Residual block before upsampling
                ResidualBlock3D(current_channels, current_channels),
                # Upsample by 2x in each dimension
                nn.ConvTranspose3d(
                    current_channels, hidden_dim,
                    kernel_size=4, stride=2, padding=1
                ),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout3d(dropout),
            ])
            current_channels = hidden_dim

        # Final prediction layer
        # Outputs logits for each block type at each position
        layers.append(nn.Conv3d(current_channels, num_blocks, kernel_size=3, padding=1))

        self.decoder = nn.Sequential(*layers)

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to block predictions.

        Args:
            z_q: Quantized latent [batch, latent_dim, 8, 8, 8]

        Returns:
            Block logits [batch, num_blocks, 32, 32, 32]
        """
        return self.decoder(z_q)


class Decoder(nn.Module):
    """3D CNN Decoder that expands 8x8x8 back to 32x32x32.

    Architecture:
        8x8x8 → ConvT → 16x16x16 → ConvT → 32x32x32 → Predict blocks

    Args:
        latent_dim: Input channels (codebook embedding_dim)
        hidden_dims: Hidden layer dimensions [256, 128, 64] (reversed from encoder)
        out_channels: Number of block types to predict
    """

    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dims: list = None,
        num_blocks: int = 3717,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        layers = []
        current_channels = latent_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                # Residual block
                ResidualBlock3D(current_channels, current_channels),
                # Upsample by 2x in each dimension
                nn.ConvTranspose3d(
                    current_channels, hidden_dim,
                    kernel_size=4, stride=2, padding=1
                ),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU(inplace=True),
            ])
            current_channels = hidden_dim

        # Final prediction layer
        # Outputs logits for each block type at each position
        layers.append(nn.Conv3d(current_channels, num_blocks, kernel_size=3, padding=1))

        self.decoder = nn.Sequential(*layers)

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to block predictions.

        Args:
            z_q: Quantized latent [batch, latent_dim, 4, 4, 4]

        Returns:
            Block logits [batch, num_blocks, 32, 32, 32]
        """
        return self.decoder(z_q)


class VQVAE(nn.Module):
    """Vector Quantized Variational AutoEncoder for Minecraft structures.

    Full pipeline:
        1. Embed input blocks using pre-trained Block2Vec
        2. Encode to compressed latent grid
        3. Quantize each position to nearest codebook entry
        4. Decode to predict block at each position

    Args:
        vocab_size: Number of block types (for output prediction)
        block_embedding_dim: Dimension of block2vec embeddings (typically 32)
        hidden_dims: Encoder hidden dimensions [64, 128, 256]
        latent_dim: Codebook embedding dimension (256)
        num_codebook_entries: Number of codebook vectors (512)
        commitment_cost: VQ commitment loss weight (0.5 recommended for EMA)
        pretrained_embeddings: Optional pre-trained block embeddings [vocab, embed_dim]
        use_ema: Use EMA codebook updates (recommended) vs gradient descent
        ema_decay: EMA decay rate (only used if use_ema=True)
    """

    def __init__(
        self,
        vocab_size: int = 3717,
        block_embedding_dim: int = 32,
        hidden_dims: list = None,
        latent_dim: int = 256,
        num_codebook_entries: int = 512,
        commitment_cost: float = 0.5,
        pretrained_embeddings: torch.Tensor = None,
        use_ema: bool = True,
        ema_decay: float = 0.99,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.block_embedding_dim = block_embedding_dim
        self.latent_dim = latent_dim
        self.num_codebook_entries = num_codebook_entries
        self.use_ema = use_ema

        # Block embedding layer
        self.block_embeddings = nn.Embedding(vocab_size, block_embedding_dim)

        # Load pretrained embeddings if provided
        if pretrained_embeddings is not None:
            assert pretrained_embeddings.shape == (vocab_size, block_embedding_dim), \
                f"Expected shape ({vocab_size}, {block_embedding_dim}), got {pretrained_embeddings.shape}"
            self.block_embeddings.weight.data.copy_(pretrained_embeddings)
            # Freeze embeddings - they're already trained
            self.block_embeddings.weight.requires_grad = False

        if hidden_dims is None:
            hidden_dims = [64, 128, 256]

        # Encoder: 32³ → 4³
        self.encoder = Encoder(
            in_channels=block_embedding_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
        )

        # Vector Quantizer with codebook
        # Use EMA version by default (prevents codebook collapse)
        if use_ema:
            self.quantizer = VectorQuantizerEMA(
                num_embeddings=num_codebook_entries,
                embedding_dim=latent_dim,
                commitment_cost=commitment_cost,
                decay=ema_decay,
            )
        else:
            self.quantizer = VectorQuantizer(
                num_embeddings=num_codebook_entries,
                embedding_dim=latent_dim,
                commitment_cost=commitment_cost,
            )

        # Decoder: 4³ → 32³
        self.decoder = Decoder(
            latent_dim=latent_dim,
            hidden_dims=list(reversed(hidden_dims)),
            num_blocks=vocab_size,
        )

    def encode(self, block_ids: torch.Tensor) -> torch.Tensor:
        """Encode block structure to continuous latent representation.

        Args:
            block_ids: Block token IDs [batch, 32, 32, 32]

        Returns:
            Encoder output [batch, latent_dim, 4, 4, 4]
        """
        # Embed blocks: [B, 32, 32, 32] → [B, 32, 32, 32, embed_dim]
        embedded = self.block_embeddings(block_ids)

        # Permute to channel-first: [B, embed_dim, 32, 32, 32]
        embedded = embedded.permute(0, 4, 1, 2, 3).contiguous()

        # Encode
        return self.encoder(embedded)

    def quantize(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize encoder output using codebook.

        Args:
            z_e: Encoder output [batch, latent_dim, D, H, W]

        Returns:
            z_q: Quantized latent (same shape)
            vq_loss: VQ loss for training
            indices: Codebook indices [batch, D, H, W]
        """
        return self.quantizer(z_e)

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode quantized latent to block predictions.

        Args:
            z_q: Quantized latent [batch, latent_dim, D, H, W]

        Returns:
            Block logits [batch, vocab_size, 32, 32, 32]
        """
        return self.decoder(z_q)

    def forward(self, block_ids: torch.Tensor) -> Dict[str, Any]:
        """Full forward pass.

        Args:
            block_ids: Block token IDs [batch, 32, 32, 32]

        Returns:
            Dictionary with:
                - logits: Block predictions [batch, vocab_size, 32, 32, 32]
                - vq_loss: VQ loss for training
                - indices: Codebook indices [batch, D, H, W]
                - z_e: Encoder output (for analysis)
                - z_q: Quantized latent (for analysis)
        """
        # Encode
        z_e = self.encode(block_ids)

        # Quantize
        z_q, vq_loss, indices = self.quantize(z_e)

        # Decode
        logits = self.decode(z_q)

        return {
            "logits": logits,
            "vq_loss": vq_loss,
            "indices": indices,
            "z_e": z_e,
            "z_q": z_q,
        }

    def compute_loss(
        self,
        block_ids: torch.Tensor,
        air_index: int = 0,
        structure_weight: float = 10.0,
        use_focal: bool = False,
        focal_gamma: float = 2.0,
    ) -> Dict[str, torch.Tensor]:
        """Compute training loss with class imbalance handling.

        To combat the air block dominance (~90% of voxels), we:
        1. Weight non-air blocks higher in the loss
        2. Optionally use focal loss to focus on hard examples

        Args:
            block_ids: Ground truth block IDs [batch, 32, 32, 32]
            air_index: Block ID for air (typically 0)
            structure_weight: Weight multiplier for non-air blocks (default 10x)
            use_focal: Use focal loss for hard example mining
            focal_gamma: Focal loss focusing parameter (higher = more focus on hard)

        Returns:
            Dictionary with:
                - loss: Total loss
                - reconstruction_loss: Cross-entropy for block prediction
                - vq_loss: Vector quantization loss
                - accuracy: Block prediction accuracy (on non-air blocks)
                - air_accuracy: Accuracy on air blocks
                - structure_accuracy: Accuracy on non-air blocks
        """
        outputs = self(block_ids)

        # Reconstruction loss
        # logits: [B, vocab_size, 32, 32, 32] → [B, 32, 32, 32, vocab_size]
        logits = outputs["logits"].permute(0, 2, 3, 4, 1).contiguous()

        # Flatten for cross-entropy
        logits_flat = logits.view(-1, self.vocab_size)  # [B*32*32*32, vocab_size]
        targets_flat = block_ids.view(-1)  # [B*32*32*32]

        # Create weight mask: higher weight for non-air blocks
        is_structure = (targets_flat != air_index).float()
        weights = torch.ones_like(is_structure)
        weights = weights + is_structure * (structure_weight - 1)  # air=1, structure=structure_weight

        if use_focal:
            # Focal loss: -alpha * (1-p)^gamma * log(p)
            # Reduces loss for well-classified examples, focuses on hard ones
            ce_loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
            probs = F.softmax(logits_flat, dim=1)
            pt = probs.gather(1, targets_flat.unsqueeze(1)).squeeze()
            focal_weight = (1 - pt) ** focal_gamma
            reconstruction_loss = (focal_weight * weights * ce_loss).mean()
        else:
            # Weighted cross-entropy
            ce_loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
            reconstruction_loss = (weights * ce_loss).mean()

        # Total loss
        total_loss = reconstruction_loss + outputs["vq_loss"]

        # Compute detailed accuracy metrics
        with torch.no_grad():
            predictions = logits_flat.argmax(dim=1)

            # Overall accuracy
            correct = (predictions == targets_flat).float()
            accuracy = correct.mean()

            # Air block accuracy
            air_mask = targets_flat == air_index
            if air_mask.sum() > 0:
                air_accuracy = correct[air_mask].mean()
            else:
                air_accuracy = torch.tensor(0.0, device=block_ids.device)

            # Structure (non-air) accuracy
            structure_mask = ~air_mask
            if structure_mask.sum() > 0:
                structure_accuracy = correct[structure_mask].mean()
            else:
                structure_accuracy = torch.tensor(0.0, device=block_ids.device)

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "vq_loss": outputs["vq_loss"],
            "accuracy": accuracy,
            "air_accuracy": air_accuracy,
            "structure_accuracy": structure_accuracy,
        }

    @torch.no_grad()
    def encode_structure(self, block_ids: torch.Tensor) -> torch.Tensor:
        """Encode a structure to codebook indices (for generation).

        Args:
            block_ids: Block token IDs [batch, 32, 32, 32]

        Returns:
            Codebook indices [batch, D, H, W]
        """
        z_e = self.encode(block_ids)
        _, _, indices = self.quantize(z_e)
        return indices

    @torch.no_grad()
    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode codebook indices to block predictions (for generation).

        Args:
            indices: Codebook indices [batch, D, H, W]

        Returns:
            Predicted block IDs [batch, 32, 32, 32]
        """
        # Look up codebook vectors - handle both EMA and non-EMA quantizers
        if self.use_ema:
            # EMA quantizer stores codebook as buffer
            z_q = F.embedding(indices, self.quantizer.codebook)  # [B, D, H, W, latent_dim]
        else:
            # Original quantizer uses nn.Embedding
            z_q = self.quantizer.codebook(indices)  # [B, D, H, W, latent_dim]

        z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()  # [B, latent_dim, D, H, W]

        # Decode to block logits
        logits = self.decode(z_q)  # [B, vocab_size, 32, 32, 32]

        # Get predicted block IDs
        block_ids = logits.argmax(dim=1)  # [B, 32, 32, 32]

        return block_ids

    def get_codebook_usage(self, indices: torch.Tensor) -> torch.Tensor:
        """Count how many times each codebook entry is used.

        Useful for monitoring codebook utilization during training.
        Dead codebook entries (never used) indicate training issues.

        Args:
            indices: Codebook indices from a batch

        Returns:
            Usage count for each codebook entry [num_codebook_entries]
        """
        indices_flat = indices.view(-1)
        usage = torch.bincount(indices_flat, minlength=self.num_codebook_entries)
        return usage


# =============================================================================
# VQ-VAE v4: New components for improved reconstruction
# =============================================================================


class ShapePreservationLoss(nn.Module):
    """Loss functions to prevent the model from erasing buildings.

    The core problem: model predicts AIR where there should be structure blocks.
    This causes buildings to "disappear" in reconstructions.

    This module provides several loss terms to combat this:
        1. false_air_penalty: Heavily penalize predicting air where structure exists
        2. volume_preservation: Penalize if reconstruction has fewer non-air blocks
        3. structure_recall: Reward correctly predicting ANY non-air block

    The philosophy is: SHAPE FIRST, DETAILS SECOND.
    It's better to predict the wrong block type than to predict air.
    """

    def __init__(
        self,
        false_air_weight: float = 5.0,
        volume_weight: float = 2.0,
        air_tokens: list = None,
    ):
        """
        Args:
            false_air_weight: Weight for penalizing false air predictions
            volume_weight: Weight for volume preservation loss
            air_tokens: List of token IDs that represent air blocks
        """
        super().__init__()
        self.false_air_weight = false_air_weight
        self.volume_weight = volume_weight
        # Default air tokens if not provided
        self.air_tokens = air_tokens if air_tokens is not None else [19, 164, 932]

    def compute_false_air_penalty(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        air_tokens_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Compute penalty for predicting air where there was structure.

        This is the KEY loss for preventing building erasure.
        When original has a block but we predict air = HEAVILY PENALIZED.

        Args:
            predictions: Predicted block IDs [N]
            targets: Target block IDs [N]
            air_tokens_tensor: Tensor of air token IDs

        Returns:
            Scalar loss value
        """
        # Where original had actual structure (non-air)
        is_structure_original = ~torch.isin(targets, air_tokens_tensor)

        # Where we predicted air
        is_air_predicted = torch.isin(predictions, air_tokens_tensor)

        # FALSE AIR: predicted air where there was structure
        # This is the primary failure mode we want to eliminate
        false_air_mask = is_structure_original & is_air_predicted

        # Compute penalty as fraction of structure blocks we erased
        if is_structure_original.sum() > 0:
            false_air_rate = false_air_mask.float().sum() / is_structure_original.float().sum()
        else:
            false_air_rate = torch.tensor(0.0, device=predictions.device)

        return false_air_rate

    def compute_volume_preservation_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        air_tokens_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss for not preserving structure volume.

        Penalizes reconstructions that have fewer non-air blocks than original.
        This encourages the model to maintain the "mass" of the structure.

        Args:
            predictions: Predicted block IDs [N]
            targets: Target block IDs [N]
            air_tokens_tensor: Tensor of air token IDs

        Returns:
            Scalar loss value (0 if volume preserved or increased, >0 if volume lost)
        """
        # Count non-air blocks in original
        original_volume = (~torch.isin(targets, air_tokens_tensor)).float().sum()

        # Count non-air blocks in reconstruction
        predicted_volume = (~torch.isin(predictions, air_tokens_tensor)).float().sum()

        # Only penalize if we LOST volume (predicted fewer structure blocks)
        # Don't penalize if we predicted more structure than original
        if original_volume > 0:
            volume_loss = F.relu(original_volume - predicted_volume) / original_volume
        else:
            volume_loss = torch.tensor(0.0, device=predictions.device)

        return volume_loss

    def compute_structure_recall(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        air_tokens_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Compute structure recall: what fraction of original structure was preserved.

        This measures: "Of all the blocks that SHOULD be non-air, how many did we
        correctly predict as non-air (regardless of exact block type)?"

        High structure recall = good shape preservation.

        Args:
            predictions: Predicted block IDs [N]
            targets: Target block IDs [N]
            air_tokens_tensor: Tensor of air token IDs

        Returns:
            Structure recall value in [0, 1]
        """
        is_structure_original = ~torch.isin(targets, air_tokens_tensor)
        is_structure_predicted = ~torch.isin(predictions, air_tokens_tensor)

        # True positives: structure in both original and reconstruction
        true_structure = is_structure_original & is_structure_predicted

        if is_structure_original.sum() > 0:
            recall = true_structure.float().sum() / is_structure_original.float().sum()
        else:
            recall = torch.tensor(1.0, device=predictions.device)

        return recall

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        air_tokens_tensor: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute all shape preservation metrics and losses.

        Args:
            logits: Model output logits [N, vocab_size]
            targets: Target block IDs [N]
            air_tokens_tensor: Tensor of air token IDs

        Returns:
            Dictionary with:
                - false_air_penalty: Loss for predicting air where structure exists
                - volume_loss: Loss for losing structure volume
                - structure_recall: Metric for shape preservation
                - total: Weighted combination of losses
        """
        predictions = logits.argmax(dim=1)

        false_air = self.compute_false_air_penalty(predictions, targets, air_tokens_tensor)
        volume_loss = self.compute_volume_preservation_loss(predictions, targets, air_tokens_tensor)
        structure_recall = self.compute_structure_recall(predictions, targets, air_tokens_tensor)

        total = self.false_air_weight * false_air + self.volume_weight * volume_loss

        return {
            'false_air_penalty': false_air,
            'volume_loss': volume_loss,
            'structure_recall': structure_recall,
            'total': total,
        }


class AsymmetricStructureLoss(nn.Module):
    """Asymmetric cross-entropy that penalizes structure→air errors more than air→structure.

    Standard cross-entropy treats all errors equally. But for shape preservation:
        - Predicting air where there was structure = BAD (erases building)
        - Predicting structure where there was air = Less bad (adds extra blocks)

    This loss applies different weights based on error type.
    """

    def __init__(
        self,
        structure_to_air_weight: float = 10.0,  # Heavy penalty for erasing
        air_to_structure_weight: float = 1.0,    # Light penalty for adding
        structure_to_structure_weight: float = 1.0,  # Normal penalty for wrong block
    ):
        super().__init__()
        self.structure_to_air_weight = structure_to_air_weight
        self.air_to_structure_weight = air_to_structure_weight
        self.structure_to_structure_weight = structure_to_structure_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        air_tokens_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Compute asymmetric cross-entropy loss.

        Args:
            logits: Model output [N, vocab_size]
            targets: Target block IDs [N]
            air_tokens_tensor: Tensor of air token IDs

        Returns:
            Weighted cross-entropy loss
        """
        # Get predictions for weight computation
        predictions = logits.argmax(dim=1)

        # Classify each position
        is_air_target = torch.isin(targets, air_tokens_tensor)
        is_air_pred = torch.isin(predictions, air_tokens_tensor)

        is_struct_target = ~is_air_target
        is_struct_pred = ~is_air_pred

        # Create weight tensor based on error type
        weights = torch.ones_like(targets, dtype=torch.float)

        # Structure in original, predicted as air = HEAVY penalty
        struct_to_air_mask = is_struct_target & is_air_pred
        weights[struct_to_air_mask] = self.structure_to_air_weight

        # Air in original, predicted as structure = light penalty
        air_to_struct_mask = is_air_target & is_struct_pred
        weights[air_to_struct_mask] = self.air_to_structure_weight

        # Structure to structure (wrong block type) = normal penalty
        struct_to_struct_wrong = is_struct_target & is_struct_pred & (predictions != targets)
        weights[struct_to_struct_wrong] = self.structure_to_structure_weight

        # Compute weighted cross-entropy
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        weighted_loss = (ce_loss * weights).sum() / weights.sum()

        return weighted_loss


def compute_similarity_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity matrix between all block embeddings.

    Args:
        embeddings: Block embeddings [vocab_size, embed_dim]

    Returns:
        Similarity matrix [vocab_size, vocab_size] with values in [0, 1]
    """
    with torch.no_grad():
        normed = F.normalize(embeddings, dim=1)
        sim = normed @ normed.t()  # Cosine similarity in [-1, 1]
        # Scale to [0, 1] for use as partial credit score
        sim = (sim + 1) / 2
    return sim


def similarity_weighted_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    similarity_matrix: torch.Tensor,
    is_structure: torch.Tensor = None,
) -> Dict[str, torch.Tensor]:
    """Compute both exact-match and similarity-weighted accuracy.

    Instead of 0/1 for wrong/right, gives partial credit based on
    embedding similarity between predicted and target blocks.

    Args:
        predictions: Predicted block IDs [N]
        targets: Target block IDs [N]
        similarity_matrix: Pre-computed similarity [vocab, vocab]
        is_structure: Optional mask for non-air blocks [N]

    Returns:
        Dictionary with exact and similarity-weighted metrics
    """
    # Exact match
    exact_match = (predictions == targets).float()

    # Similarity-weighted (partial credit)
    # Look up similarity between each prediction and its target
    sim_scores = similarity_matrix[predictions, targets]

    if is_structure is not None:
        struct_mask = is_structure.bool()
        air_mask = ~struct_mask

        struct_exact = exact_match[struct_mask].mean() if struct_mask.any() else torch.tensor(0.0)
        struct_sim = sim_scores[struct_mask].mean() if struct_mask.any() else torch.tensor(0.0)
        air_exact = exact_match[air_mask].mean() if air_mask.any() else torch.tensor(0.0)
        air_sim = sim_scores[air_mask].mean() if air_mask.any() else torch.tensor(0.0)
    else:
        struct_exact = exact_match.mean()
        struct_sim = sim_scores.mean()
        air_exact = torch.tensor(0.0, device=predictions.device)
        air_sim = torch.tensor(0.0, device=predictions.device)

    return {
        'exact_match': exact_match.mean(),
        'similarity_weighted': sim_scores.mean(),
        'struct_exact': struct_exact,
        'struct_similarity': struct_sim,
        'air_exact': air_exact,
        'air_similarity': air_sim,
    }


class EmbeddingAwareLoss(nn.Module):
    """Loss function that combines cross-entropy with embedding similarity.

    Standard cross-entropy treats all wrong predictions equally. This loss
    gives partial credit when the predicted block is semantically similar
    to the target (e.g., oak_planks vs spruce_planks).

    Total loss = CE_loss + alpha * embedding_similarity_loss

    Args:
        alpha: Weight for embedding similarity loss (default 0.5)
        temperature: Softmax temperature for computing expected embedding
    """

    def __init__(self, alpha: float = 0.5, temperature: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        embeddings: torch.Tensor,
        weights: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute embedding-aware loss.

        Args:
            logits: Model output [B, vocab, D, H, W]
            targets: Target block IDs [B, D, H, W]
            embeddings: Block embeddings [vocab, embed_dim]
            weights: Optional per-voxel weights [B, D, H, W]

        Returns:
            Dictionary with loss components
        """
        B, V, D, H, W = logits.shape

        # Reshape for computation
        logits_flat = logits.permute(0, 2, 3, 4, 1).reshape(-1, V)  # [N, V]
        targets_flat = targets.reshape(-1)  # [N]

        # Standard cross-entropy
        if weights is not None:
            weights_flat = weights.reshape(-1)
            ce_loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
            ce_loss = (ce_loss * weights_flat).sum() / weights_flat.sum()
        else:
            ce_loss = F.cross_entropy(logits_flat, targets_flat)

        # Embedding similarity loss
        # Get predicted probabilities (with temperature scaling)
        probs = F.softmax(logits_flat / self.temperature, dim=1)  # [N, V]

        # Normalize embeddings for cosine similarity
        embeddings_normed = F.normalize(embeddings, dim=1)  # [V, E]

        # Expected embedding under prediction distribution
        predicted_emb = probs @ embeddings_normed  # [N, E]

        # Target embeddings (normalized)
        target_emb = embeddings_normed[targets_flat]  # [N, E]

        # Cosine similarity (1 = same, -1 = opposite)
        similarity = (predicted_emb * target_emb).sum(dim=1)  # [N]

        # Loss is 1 - similarity (0 when perfect match, 2 when opposite)
        emb_loss = 1 - similarity

        if weights is not None:
            emb_loss = (emb_loss * weights_flat).sum() / weights_flat.sum()
        else:
            emb_loss = emb_loss.mean()

        total_loss = ce_loss + self.alpha * emb_loss

        return {
            'total': total_loss,
            'ce_loss': ce_loss,
            'embedding_loss': emb_loss,
        }


class VQVAEv4(nn.Module):
    """VQ-VAE v4 with improved reconstruction and SHAPE PRESERVATION.

    Key improvements over VQVAE (v3):
        1. 8x8x8 latent grid (64:1 compression) vs 4x4x4 (512:1)
        2. Trainable embeddings with regularization
        3. Embedding-aware loss for semantic similarity credit
        4. **NEW: Shape preservation losses to prevent building erasure**
           - False air penalty: heavily penalize predicting air where structure exists
           - Volume preservation: penalize losing structure volume
           - Asymmetric loss: structure→air errors penalized more than air→structure

    Philosophy: SHAPE FIRST, DETAILS SECOND.
    It's better to predict the wrong block type than to erase the building.

    Args:
        vocab_size: Number of block types
        block_embedding_dim: Dimension of block embeddings
        hidden_dims: Encoder hidden dimensions [96, 192]
        latent_dim: Codebook embedding dimension
        num_codebook_entries: Number of codebook vectors
        commitment_cost: VQ commitment loss weight
        pretrained_embeddings: Pre-trained block embeddings
        train_embeddings: Whether to train embeddings (default True)
        embedding_loss_alpha: Weight for embedding similarity loss
        stability_weight: Weight for embedding stability regularization
        diversity_weight: Weight for embedding diversity regularization
        false_air_weight: Weight for penalizing false air predictions (NEW)
        volume_weight: Weight for volume preservation loss (NEW)
        structure_to_air_weight: Asymmetric weight for structure→air errors (NEW)
        dropout: Dropout rate
        ema_decay: EMA decay rate for codebook
    """

    def __init__(
        self,
        vocab_size: int = 3717,
        block_embedding_dim: int = 32,
        hidden_dims: list = None,
        latent_dim: int = 256,
        num_codebook_entries: int = 512,
        commitment_cost: float = 0.5,
        pretrained_embeddings: torch.Tensor = None,
        train_embeddings: bool = True,
        embedding_loss_alpha: float = 0.5,
        stability_weight: float = 0.01,
        diversity_weight: float = 0.001,
        false_air_weight: float = 5.0,
        volume_weight: float = 2.0,
        structure_to_air_weight: float = 10.0,
        dropout: float = 0.1,
        ema_decay: float = 0.99,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.block_embedding_dim = block_embedding_dim
        self.latent_dim = latent_dim
        self.num_codebook_entries = num_codebook_entries
        self.train_embeddings = train_embeddings
        self.embedding_loss_alpha = embedding_loss_alpha
        self.stability_weight = stability_weight
        self.diversity_weight = diversity_weight
        self.false_air_weight = false_air_weight
        self.volume_weight = volume_weight
        self.structure_to_air_weight = structure_to_air_weight

        if hidden_dims is None:
            hidden_dims = [96, 192]  # V4 defaults for 32→8

        # Block embedding layer
        self.block_embeddings = nn.Embedding(vocab_size, block_embedding_dim)

        # Load and optionally freeze pretrained embeddings
        if pretrained_embeddings is not None:
            assert pretrained_embeddings.shape == (vocab_size, block_embedding_dim), \
                f"Expected shape ({vocab_size}, {block_embedding_dim}), got {pretrained_embeddings.shape}"
            self.block_embeddings.weight.data.copy_(pretrained_embeddings)
            # Store original for regularization
            self.register_buffer('original_embeddings', pretrained_embeddings.clone())
        else:
            # If no pretrained, use current weights as reference
            self.register_buffer('original_embeddings', self.block_embeddings.weight.data.clone())

        # Control trainability
        self.block_embeddings.weight.requires_grad = train_embeddings

        # Encoder: 32³ → 8³ (V4 architecture)
        self.encoder = EncoderV4(
            in_channels=block_embedding_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            dropout=dropout,
        )

        # Vector Quantizer with EMA
        self.quantizer = VectorQuantizerEMA(
            num_embeddings=num_codebook_entries,
            embedding_dim=latent_dim,
            commitment_cost=commitment_cost,
            decay=ema_decay,
        )

        # Decoder: 8³ → 32³ (V4 architecture)
        self.decoder = DecoderV4(
            latent_dim=latent_dim,
            hidden_dims=list(reversed(hidden_dims)),
            num_blocks=vocab_size,
            dropout=dropout,
        )

        # Embedding-aware loss
        self.embedding_loss_fn = EmbeddingAwareLoss(
            alpha=embedding_loss_alpha,
            temperature=0.1,
        )

        # Shape preservation loss (NEW - prevents building erasure)
        self.shape_loss_fn = ShapePreservationLoss(
            false_air_weight=false_air_weight,
            volume_weight=volume_weight,
        )

        # Asymmetric loss (NEW - penalizes structure→air more than air→structure)
        self.asymmetric_loss_fn = AsymmetricStructureLoss(
            structure_to_air_weight=structure_to_air_weight,
            air_to_structure_weight=1.0,
            structure_to_structure_weight=1.0,
        )

        # Cache similarity matrix (refresh when embeddings change)
        self._similarity_matrix = None
        self._similarity_matrix_valid = False

    def get_similarity_matrix(self) -> torch.Tensor:
        """Get cached similarity matrix, recomputing if embeddings changed."""
        if not self._similarity_matrix_valid or self._similarity_matrix is None:
            self._similarity_matrix = compute_similarity_matrix(
                self.block_embeddings.weight.detach()
            )
            self._similarity_matrix_valid = True
        return self._similarity_matrix

    def invalidate_similarity_cache(self):
        """Mark similarity matrix as needing recomputation."""
        self._similarity_matrix_valid = False

    def set_train_embeddings(self, train: bool):
        """Enable or disable embedding training (for phased training)."""
        self.train_embeddings = train
        self.block_embeddings.weight.requires_grad = train
        if train:
            self.invalidate_similarity_cache()

    def encode(self, block_ids: torch.Tensor) -> torch.Tensor:
        """Encode block structure to continuous latent representation."""
        embedded = self.block_embeddings(block_ids)
        embedded = embedded.permute(0, 4, 1, 2, 3).contiguous()
        return self.encoder(embedded)

    def quantize(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize encoder output using codebook."""
        return self.quantizer(z_e)

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode quantized latent to block predictions."""
        return self.decoder(z_q)

    def forward(self, block_ids: torch.Tensor) -> Dict[str, Any]:
        """Full forward pass."""
        z_e = self.encode(block_ids)
        z_q, vq_loss, indices = self.quantize(z_e)
        logits = self.decode(z_q)

        return {
            "logits": logits,
            "vq_loss": vq_loss,
            "indices": indices,
            "z_e": z_e,
            "z_q": z_q,
        }

    def compute_embedding_regularization(self) -> Dict[str, torch.Tensor]:
        """Compute regularization terms for embeddings."""
        current = self.block_embeddings.weight
        original = self.original_embeddings

        # Stability: penalize drift from initialization
        stability_loss = F.mse_loss(current, original)

        # Diversity: penalize if embeddings become too similar
        normed = F.normalize(current, dim=1)
        similarity_matrix = normed @ normed.t()
        # Mask out diagonal (self-similarity)
        off_diag_mask = 1 - torch.eye(current.size(0), device=current.device)
        avg_similarity = (similarity_matrix * off_diag_mask).sum() / off_diag_mask.sum()
        # Penalize if average similarity > 0.3
        diversity_loss = F.relu(avg_similarity - 0.3)

        return {
            'stability': stability_loss,
            'diversity': diversity_loss,
        }

    def compute_loss(
        self,
        block_ids: torch.Tensor,
        air_tokens_tensor: torch.Tensor = None,
        structure_weight: float = 50.0,  # INCREASED from 10.0
        use_embedding_loss: bool = True,
        use_shape_loss: bool = True,
        use_asymmetric_loss: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Compute training loss with shape preservation and embedding-aware reconstruction.

        The loss now emphasizes SHAPE FIRST, DETAILS SECOND:
        1. Asymmetric CE: penalizes structure→air errors heavily
        2. Shape preservation: false air penalty + volume loss
        3. Standard weighted CE + embedding similarity loss

        Args:
            block_ids: Ground truth block IDs [batch, 32, 32, 32]
            air_tokens_tensor: Tensor of air token IDs for masking
            structure_weight: Weight multiplier for non-air blocks (default 50.0)
            use_embedding_loss: Whether to use embedding similarity loss
            use_shape_loss: Whether to use shape preservation losses (NEW)
            use_asymmetric_loss: Whether to use asymmetric CE (NEW)

        Returns:
            Dictionary with all loss components and metrics
        """
        outputs = self(block_ids)
        logits = outputs["logits"]

        # Flatten for loss computation
        logits_flat = logits.permute(0, 2, 3, 4, 1).reshape(-1, self.vocab_size)
        targets_flat = block_ids.view(-1)

        # Setup air tokens
        if air_tokens_tensor is not None:
            air_tokens_device = air_tokens_tensor.to(targets_flat.device)
        else:
            # Default air tokens
            air_tokens_device = torch.tensor([19, 164, 932], device=targets_flat.device)

        is_air = torch.isin(targets_flat, air_tokens_device)
        is_structure = ~is_air

        # === PRIMARY LOSS: Asymmetric Cross-Entropy ===
        # This is the main driver for shape preservation
        if use_asymmetric_loss:
            # Asymmetric loss: structure→air errors penalized 10x more
            asymmetric_ce = self.asymmetric_loss_fn(logits_flat, targets_flat, air_tokens_device)
        else:
            # Standard weighted cross-entropy
            weights = torch.ones_like(targets_flat, dtype=torch.float)
            weights[is_structure] = structure_weight
            ce_per_voxel = F.cross_entropy(logits_flat, targets_flat, reduction='none')
            asymmetric_ce = (ce_per_voxel * weights).sum() / weights.sum()

        # === SHAPE PRESERVATION LOSS ===
        # Explicitly penalize false air predictions and volume loss
        if use_shape_loss:
            shape_losses = self.shape_loss_fn(logits_flat, targets_flat, air_tokens_device)
            false_air_loss = self.false_air_weight * shape_losses['false_air_penalty']
            volume_loss = self.volume_weight * shape_losses['volume_loss']
            structure_recall = shape_losses['structure_recall']
        else:
            false_air_loss = torch.tensor(0.0, device=block_ids.device)
            volume_loss = torch.tensor(0.0, device=block_ids.device)
            structure_recall = torch.tensor(0.0, device=block_ids.device)

        # === EMBEDDING SIMILARITY LOSS ===
        # Partial credit for semantically similar predictions
        if use_embedding_loss and self.embedding_loss_alpha > 0:
            weights = torch.ones_like(targets_flat, dtype=torch.float)
            weights[is_structure] = structure_weight
            weights = weights.view(block_ids.shape)

            loss_dict = self.embedding_loss_fn(
                logits, block_ids, self.block_embeddings.weight, weights
            )
            emb_loss = loss_dict['embedding_loss']
        else:
            emb_loss = torch.tensor(0.0, device=block_ids.device)

        # === VQ LOSS ===
        vq_loss = outputs["vq_loss"]

        # === EMBEDDING REGULARIZATION ===
        if self.train_embeddings:
            emb_reg = self.compute_embedding_regularization()
            stability_loss = self.stability_weight * emb_reg['stability']
            diversity_loss = self.diversity_weight * emb_reg['diversity']
        else:
            stability_loss = torch.tensor(0.0, device=block_ids.device)
            diversity_loss = torch.tensor(0.0, device=block_ids.device)

        # === TOTAL LOSS ===
        # Shape preservation is now a primary component
        total_loss = (
            asymmetric_ce +
            false_air_loss +
            volume_loss +
            self.embedding_loss_alpha * emb_loss +
            vq_loss +
            stability_loss +
            diversity_loss
        )

        # === COMPUTE METRICS ===
        with torch.no_grad():
            predictions = logits_flat.argmax(dim=1)

            # Shape preservation metrics (THE NEW KEY METRICS)
            is_air_pred = torch.isin(predictions, air_tokens_device)
            is_struct_pred = ~is_air_pred

            # False air rate: what fraction of original structure did we erase?
            if is_structure.sum() > 0:
                false_air_rate = (is_structure & is_air_pred).float().sum() / is_structure.float().sum()
            else:
                false_air_rate = torch.tensor(0.0, device=block_ids.device)

            # Volume preservation: ratio of predicted volume to original volume
            original_volume = is_structure.float().sum()
            predicted_volume = is_struct_pred.float().sum()
            if original_volume > 0:
                volume_ratio = predicted_volume / original_volume
            else:
                volume_ratio = torch.tensor(1.0, device=block_ids.device)

            # Get similarity matrix for weighted accuracy
            sim_matrix = self.get_similarity_matrix().to(predictions.device)

            # Compute both exact and similarity-weighted metrics
            metrics = similarity_weighted_accuracy(
                predictions, targets_flat, sim_matrix, is_structure.float()
            )

        return {
            # Total loss
            "loss": total_loss,
            # Loss components
            "asymmetric_ce": asymmetric_ce,
            "false_air_loss": false_air_loss,
            "volume_loss": volume_loss,
            "embedding_loss": emb_loss,
            "vq_loss": vq_loss,
            "stability_loss": stability_loss,
            "diversity_loss": diversity_loss,
            # Shape preservation metrics (NEW - THE KEY METRICS)
            "false_air_rate": false_air_rate,  # Want this LOW (< 10%)
            "structure_recall": structure_recall,  # Want this HIGH (> 90%)
            "volume_ratio": volume_ratio,  # Want this CLOSE TO 1.0
            # Standard accuracy metrics
            "accuracy": metrics['exact_match'],
            "structure_accuracy": metrics['struct_exact'],
            "air_accuracy": metrics['air_exact'],
            "similarity_accuracy": metrics['similarity_weighted'],
            "struct_similarity": metrics['struct_similarity'],
        }

    @torch.no_grad()
    def encode_structure(self, block_ids: torch.Tensor) -> torch.Tensor:
        """Encode a structure to codebook indices (for generation)."""
        z_e = self.encode(block_ids)
        _, _, indices = self.quantize(z_e)
        return indices

    @torch.no_grad()
    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode codebook indices to block predictions (for generation)."""
        z_q = F.embedding(indices, self.quantizer.codebook)
        z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()
        logits = self.decode(z_q)
        return logits.argmax(dim=1)

    def get_codebook_usage(self, indices: torch.Tensor) -> torch.Tensor:
        """Count how many times each codebook entry is used."""
        indices_flat = indices.view(-1)
        usage = torch.bincount(indices_flat, minlength=self.num_codebook_entries)
        return usage


# =============================================================================
# VQ-VAE v8-B: 16x16x16 Latent Resolution Upgrade
# =============================================================================


class EncoderV8B(nn.Module):
    """3D CNN Encoder for VQ-VAE v8-B: 32x32x32 → 16x16x16 latent.

    Key changes from v6:
        - Single downsampling stage (32→16) instead of two (32→16→8)
        - 8x more spatial positions (4,096 vs 512)
        - Reduces compression from 64:1 to 8:1
        - 6 ResBlocks at latent resolution for capacity

    Architecture:
        32x32x32 → Conv4/2 → 16x16x16 → ResBlocks → 16x16x16

    Args:
        in_channels: Input channels (block embedding dim, typically 40)
        hidden_dim: Single hidden dimension throughout (192)
        rfsq_dim: Output dimension for RFSQ (4)
        num_resblocks_per_stage: ResBlocks after downsampling (2)
        num_resblocks_latent: ResBlocks at latent resolution (6)
        dropout: Dropout rate for regularization (0.1)
    """

    def __init__(
        self,
        in_channels: int = 40,
        hidden_dim: int = 192,
        rfsq_dim: int = 4,
        num_resblocks_per_stage: int = 2,
        num_resblocks_latent: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Initial projection
        self.initial = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        # Downsampling stage: 32 → 16
        self.downsample = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout)
        )

        # ResBlocks after downsampling
        self.stage_blocks = nn.Sequential(*[
            ResidualBlock3D(hidden_dim, hidden_dim)
            for _ in range(num_resblocks_per_stage)
        ])

        # ResBlocks at latent resolution (16×16×16)
        self.latent_blocks = nn.Sequential(*[
            ResidualBlock3D(hidden_dim, hidden_dim)
            for _ in range(num_resblocks_latent)
        ])

        # Final projection to RFSQ dimension
        self.latent_proj = nn.Conv3d(hidden_dim, rfsq_dim, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation.

        Args:
            x: Input tensor [batch, channels, 32, 32, 32]

        Returns:
            Latent tensor [batch, rfsq_dim, 16, 16, 16]
        """
        x = self.initial(x)           # [B, 192, 32, 32, 32]
        x = self.downsample(x)         # [B, 192, 16, 16, 16]
        x = self.stage_blocks(x)       # [B, 192, 16, 16, 16]
        x = self.latent_blocks(x)      # [B, 192, 16, 16, 16]
        z_e = self.latent_proj(x)      # [B, 4, 16, 16, 16]
        return z_e


class DecoderV8B(nn.Module):
    """3D CNN Decoder for VQ-VAE v8-B: 16x16x16 latent → 32x32x32 output.

    Key changes from v6:
        - Single upsampling stage (16→32) instead of two (8→16→32)
        - Mirror of encoder architecture
        - 6 ResBlocks at latent resolution

    Architecture:
        16x16x16 → ResBlocks → ConvT4/2 → 32x32x32 → Predict blocks

    Args:
        rfsq_dim: Input dimension from RFSQ (4)
        hidden_dim: Single hidden dimension throughout (192)
        num_blocks: Number of block types to predict (3717)
        num_resblocks_per_stage: ResBlocks before upsampling (2)
        num_resblocks_latent: ResBlocks at latent resolution (6)
        dropout: Dropout rate for regularization (0.1)
    """

    def __init__(
        self,
        rfsq_dim: int = 4,
        hidden_dim: int = 192,
        num_blocks: int = 3717,
        num_resblocks_per_stage: int = 2,
        num_resblocks_latent: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Initial projection from RFSQ
        self.initial = nn.Sequential(
            nn.Conv3d(rfsq_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        # ResBlocks at latent resolution
        self.latent_blocks = nn.Sequential(*[
            ResidualBlock3D(hidden_dim, hidden_dim)
            for _ in range(num_resblocks_latent)
        ])

        # ResBlocks before upsampling
        self.stage_blocks = nn.Sequential(*[
            ResidualBlock3D(hidden_dim, hidden_dim)
            for _ in range(num_resblocks_per_stage)
        ])

        # Upsampling stage: 16 → 32
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout)
        )

        # Final prediction head
        self.final = nn.Conv3d(hidden_dim, num_blocks, 3, padding=1)

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to block predictions.

        Args:
            z_q: Quantized latent [batch, rfsq_dim, 16, 16, 16]

        Returns:
            Block logits [batch, num_blocks, 32, 32, 32]
        """
        x = self.initial(z_q)          # [B, 192, 16, 16, 16]
        x = self.latent_blocks(x)      # [B, 192, 16, 16, 16]
        x = self.stage_blocks(x)       # [B, 192, 16, 16, 16]
        x = self.upsample(x)           # [B, 192, 32, 32, 32]
        logits = self.final(x)         # [B, 3717, 32, 32, 32]
        return logits


class VQVAEv8B(nn.Module):
    """VQ-VAE v8-B with 16×16×16 latent resolution for better generation.

    This version increases latent resolution from 8³ to 16³, providing 8x more
    spatial positions to preserve structural detail. Maintains pure latent-only
    architecture (NO skip connections) to ensure generation capability.

    Key improvements over v6-freq:
        1. 16×16×16 latent grid (4,096 positions vs 512)
        2. 8:1 compression (32³→16³) vs 64:1 (32³→8³)
        3. More ResBlocks at latent resolution (6 vs 2)
        4. Volume penalty loss (fixes 1.68x over-prediction)
        5. Perceptual loss (spatial smoothness in latent)

    Target: 60-65% building accuracy (vs v6-freq's 49.2%)

    Args:
        vocab_size: Number of block types (3717)
        emb_dim: Block embedding dimension (40)
        hidden_dim: Encoder/decoder hidden dimension (192)
        rfsq_levels: RFSQ quantization levels per dimension ([5,5,5,5])
        num_stages: Number of RFSQ stages (2)
        dropout: Dropout rate (0.1)
        pretrained_embeddings: Optional pre-trained block embeddings
    """

    def __init__(
        self,
        vocab_size: int = 3717,
        emb_dim: int = 40,
        hidden_dim: int = 192,
        rfsq_levels: list = None,
        num_stages: int = 2,
        dropout: float = 0.1,
        pretrained_embeddings: torch.Tensor = None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim

        if rfsq_levels is None:
            rfsq_levels = [5, 5, 5, 5]  # 4 dims, 5 levels each

        self.rfsq_dim = len(rfsq_levels)

        # Block embedding layer
        self.block_emb = nn.Embedding(vocab_size, emb_dim)

        # Load pretrained embeddings if provided
        if pretrained_embeddings is not None:
            assert pretrained_embeddings.shape == (vocab_size, emb_dim), \
                f"Expected shape ({vocab_size}, {emb_dim}), got {pretrained_embeddings.shape}"
            self.block_emb.weight.data.copy_(pretrained_embeddings)
            # Freeze embeddings - they're already trained
            self.block_emb.weight.requires_grad = False

        # Encoder: 32³ → 16³
        self.encoder = EncoderV8B(
            in_channels=emb_dim,
            hidden_dim=hidden_dim,
            rfsq_dim=self.rfsq_dim,
            dropout=dropout,
        )

        # RFSQ quantizer (import needed at top of file)
        from .rfsq import RFSQ
        self.quantizer = RFSQ(
            levels_per_stage=rfsq_levels,
            num_stages=num_stages,
        )

        # Decoder: 16³ → 32³
        self.decoder = DecoderV8B(
            rfsq_dim=self.rfsq_dim,
            hidden_dim=hidden_dim,
            num_blocks=vocab_size,
            dropout=dropout,
        )

    def encode(self, block_ids: torch.Tensor) -> torch.Tensor:
        """Encode block structure to continuous latent representation.

        Args:
            block_ids: Block token IDs [batch, 32, 32, 32]

        Returns:
            Encoder output [batch, rfsq_dim, 16, 16, 16]
        """
        # Embed blocks: [B, 32, 32, 32] → [B, 32, 32, 32, emb_dim]
        embedded = self.block_emb(block_ids)

        # Permute to channel-first: [B, emb_dim, 32, 32, 32]
        embedded = embedded.permute(0, 4, 1, 2, 3).contiguous()

        # Encode
        return self.encoder(embedded)

    def quantize(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize encoder output using RFSQ.

        Args:
            z_e: Encoder output [batch, rfsq_dim, 16, 16, 16]

        Returns:
            z_q: Quantized latent (same shape)
            indices: Codebook indices [batch, 16, 16, 16, rfsq_dim]
        """
        return self.quantizer(z_e)

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode quantized latent to block predictions.

        Args:
            z_q: Quantized latent [batch, rfsq_dim, 16, 16, 16]

        Returns:
            Block logits [batch, vocab_size, 32, 32, 32]
        """
        return self.decoder(z_q)

    def forward(self, block_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass.

        Args:
            block_ids: Block token IDs [batch, 32, 32, 32]

        Returns:
            Tuple of (logits, z_q, indices):
                - logits: Block predictions [batch, vocab_size, 32, 32, 32]
                - z_q: Quantized latent [batch, rfsq_dim, 16, 16, 16]
                - indices: Codebook indices [batch, 16, 16, 16, rfsq_dim]
        """
        # Encode
        z_e = self.encode(block_ids)

        # Quantize (RFSQ doesn't return a loss, just z_q and indices)
        z_q, indices = self.quantize(z_e)

        # Decode
        logits = self.decode(z_q)

        return logits, z_q, indices

    @torch.no_grad()
    def encode_structure(self, block_ids: torch.Tensor) -> torch.Tensor:
        """Encode a structure to codebook indices (for generation).

        Args:
            block_ids: Block token IDs [batch, 32, 32, 32]

        Returns:
            Codebook indices [batch, 16, 16, 16, rfsq_dim]
        """
        z_e = self.encode(block_ids)
        _, indices = self.quantize(z_e)
        return indices

    @torch.no_grad()
    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode codebook indices to block predictions (for generation).

        Args:
            indices: Codebook indices [batch, 16, 16, 16, rfsq_dim]

        Returns:
            Predicted block IDs [batch, 32, 32, 32]
        """
        # Dequantize indices to get z_q
        # RFSQ's decode method handles this
        z_q = self.quantizer.decode(indices)

        # Decode to block logits
        logits = self.decode(z_q)

        # Get predicted block IDs
        block_ids = logits.argmax(dim=1)

        return block_ids
