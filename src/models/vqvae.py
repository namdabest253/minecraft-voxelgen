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
