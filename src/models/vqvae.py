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
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any


class VectorQuantizer(nn.Module):
    """Vector Quantization layer with straight-through gradient estimator.

    Maintains a codebook of K learnable vectors. Each input vector is
    replaced with its nearest codebook entry. Gradients flow through
    using the straight-through estimator.

    Args:
        num_embeddings: Number of codebook entries (K)
        embedding_dim: Dimension of each codebook vector (D)
        commitment_cost: Weight for commitment loss (β), typically 0.25
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

        # The codebook: K vectors of dimension D
        # Shape: [num_embeddings, embedding_dim]
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)

        # Initialize codebook with uniform distribution
        # Range chosen to match typical encoder outputs
        self.codebook.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize encoder outputs to nearest codebook entries.

        Args:
            z_e: Encoder output [batch, channels, depth, height, width]
                 channels should equal embedding_dim

        Returns:
            z_q: Quantized output (same shape as z_e)
            vq_loss: Codebook + commitment loss
            encoding_indices: Which codebook entry was selected [batch, D, H, W]
        """
        # z_e shape: [B, C, D, H, W] where C = embedding_dim
        batch_size = z_e.shape[0]

        # Reshape to [B*D*H*W, C] for distance calculation
        # First permute to [B, D, H, W, C], then flatten
        z_e_permuted = z_e.permute(0, 2, 3, 4, 1).contiguous()  # [B, D, H, W, C]
        flat_z_e = z_e_permuted.view(-1, self.embedding_dim)  # [B*D*H*W, C]

        # Compute distances to all codebook entries
        # ||z_e - codebook||² = ||z_e||² + ||codebook||² - 2 * z_e · codebook
        # This is more numerically stable than direct distance computation

        # ||z_e||² for each input vector: [B*D*H*W, 1]
        z_e_sq = (flat_z_e ** 2).sum(dim=1, keepdim=True)

        # ||codebook||² for each codebook entry: [1, K]
        codebook_sq = (self.codebook.weight ** 2).sum(dim=1, keepdim=True).t()

        # z_e · codebook: [B*D*H*W, K]
        dot_product = torch.mm(flat_z_e, self.codebook.weight.t())

        # Squared distances: [B*D*H*W, K]
        distances = z_e_sq + codebook_sq - 2 * dot_product

        # Find nearest codebook entry for each position
        encoding_indices = distances.argmin(dim=1)  # [B*D*H*W]

        # Look up the codebook vectors
        z_q_flat = self.codebook(encoding_indices)  # [B*D*H*W, C]

        # Reshape back to [B, D, H, W, C]
        z_q_permuted = z_q_flat.view(z_e_permuted.shape)

        # === Compute VQ Loss ===
        # Codebook loss: move codebook toward encoder outputs
        # sg[z_e] means stop gradient - treat z_e as constant
        codebook_loss = F.mse_loss(z_q_permuted, z_e_permuted.detach())

        # Commitment loss: move encoder outputs toward codebook
        # sg[z_q] means stop gradient - treat z_q as constant
        commitment_loss = F.mse_loss(z_e_permuted, z_q_permuted.detach())

        # Total VQ loss
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # === Straight-Through Estimator ===
        # Forward: use z_q (quantized)
        # Backward: gradients flow to z_e (as if no quantization happened)
        #
        # Math trick: z_q = z_e + (z_q - z_e).detach()
        # - Forward: z_e + (z_q - z_e) = z_q ✓
        # - Backward: gradient only flows through z_e ✓
        z_q_st = z_e_permuted + (z_q_permuted - z_e_permuted).detach()

        # Permute back to [B, C, D, H, W]
        z_q = z_q_st.permute(0, 4, 1, 2, 3).contiguous()

        # Reshape encoding indices to spatial grid
        spatial_shape = z_e_permuted.shape[:-1]  # [B, D, H, W]
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
        commitment_cost: VQ commitment loss weight (0.25)
        pretrained_embeddings: Optional pre-trained block embeddings [vocab, embed_dim]
    """

    def __init__(
        self,
        vocab_size: int = 3717,
        block_embedding_dim: int = 32,
        hidden_dims: list = None,
        latent_dim: int = 256,
        num_codebook_entries: int = 512,
        commitment_cost: float = 0.25,
        pretrained_embeddings: torch.Tensor = None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.block_embedding_dim = block_embedding_dim
        self.latent_dim = latent_dim
        self.num_codebook_entries = num_codebook_entries

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
        ignore_index: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """Compute training loss.

        Args:
            block_ids: Ground truth block IDs [batch, 32, 32, 32]
            ignore_index: Block ID to ignore in loss (typically air=0)

        Returns:
            Dictionary with:
                - loss: Total loss
                - reconstruction_loss: Cross-entropy for block prediction
                - vq_loss: Vector quantization loss
                - accuracy: Block prediction accuracy
        """
        outputs = self(block_ids)

        # Reconstruction loss
        # logits: [B, vocab_size, 32, 32, 32] → [B, 32, 32, 32, vocab_size]
        logits = outputs["logits"].permute(0, 2, 3, 4, 1).contiguous()

        # Flatten for cross-entropy
        logits_flat = logits.view(-1, self.vocab_size)  # [B*32*32*32, vocab_size]
        targets_flat = block_ids.view(-1)  # [B*32*32*32]

        # Cross-entropy loss (ignoring air blocks if specified)
        reconstruction_loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=ignore_index,
        )

        # Total loss
        total_loss = reconstruction_loss + outputs["vq_loss"]

        # Compute accuracy (excluding ignored blocks)
        with torch.no_grad():
            predictions = logits_flat.argmax(dim=1)
            mask = targets_flat != ignore_index
            correct = (predictions[mask] == targets_flat[mask]).float().sum()
            total = mask.sum()
            accuracy = correct / total if total > 0 else torch.tensor(0.0)

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "vq_loss": outputs["vq_loss"],
            "accuracy": accuracy,
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
        # Look up codebook vectors
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
