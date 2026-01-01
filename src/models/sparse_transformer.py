"""
Sparse Structure Transformer for Minecraft structure reconstruction.

A fundamentally different approach from dense VQ-VAE:
- Treats structures as sparse sets of (position, block) pairs
- No air blocks in representation (air is implicit as absence)
- Uses Transformer architecture for set-to-set learning
- Predicts block embeddings instead of class logits
- Equal weight per non-air block (no class imbalance)

Architecture:
    Input: Set of (position, block_embedding) pairs
        → Fourier positional encoding for 3D coordinates
        → Transformer encoder (self-attention over blocks)
        → Optional VQ compression to fixed latent
        → Transformer decoder (reconstruct block embeddings)
        → Nearest neighbor lookup to get block IDs

Key innovations:
1. Sparse representation eliminates 80% air dominance
2. Embedding prediction (MSE loss) instead of 3717-way softmax
3. Block2Vec embeddings used both as input AND target
4. Fourier features for continuous 3D position encoding
"""

import math
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierPositionalEncoding(nn.Module):
    """Fourier feature encoding for 3D coordinates.

    Maps 3D positions (x, y, z) to high-dimensional features using
    sinusoidal functions at multiple frequencies. This allows the
    network to learn high-frequency spatial patterns.

    Based on NeRF (Neural Radiance Fields) positional encoding.

    Args:
        num_frequencies: Number of frequency bands (L in NeRF paper)
        max_coord: Maximum coordinate value (for normalization)
        include_input: Whether to concatenate original coordinates
    """

    def __init__(
        self,
        num_frequencies: int = 10,
        max_coord: int = 32,
        include_input: bool = True,
    ):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.max_coord = max_coord
        self.include_input = include_input

        # Frequency bands: 2^0, 2^1, ..., 2^(L-1)
        freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        self.register_buffer("freq_bands", freq_bands)

        # Output dimension: 3 coords * num_freq * 2 (sin + cos) + optional 3 (input)
        self.output_dim = 3 * num_frequencies * 2
        if include_input:
            self.output_dim += 3

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Encode 3D positions to Fourier features.

        Args:
            positions: [B, N, 3] tensor of (x, y, z) coordinates

        Returns:
            [B, N, output_dim] tensor of Fourier features
        """
        # Normalize coordinates to [-1, 1]
        normalized = positions / self.max_coord * 2 - 1  # [B, N, 3]

        # Compute frequencies: [B, N, 3, num_freq]
        scaled = normalized.unsqueeze(-1) * self.freq_bands * math.pi

        # Apply sin and cos: [B, N, 3, num_freq * 2]
        encoded = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)

        # Flatten: [B, N, 3 * num_freq * 2]
        encoded = encoded.view(*positions.shape[:-1], -1)

        if self.include_input:
            # Concatenate normalized input coordinates
            encoded = torch.cat([normalized, encoded], dim=-1)

        return encoded


class SetPooling(nn.Module):
    """Pooling layer to aggregate variable-length sets into fixed representation.

    Uses attention-based pooling (similar to Set Transformer's PMA)
    to create a fixed number of output vectors from variable input.

    Args:
        input_dim: Input feature dimension
        output_dim: Output feature dimension
        num_outputs: Number of output vectors (seeds)
        num_heads: Number of attention heads
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_outputs: int = 16,
        num_heads: int = 8,
    ):
        super().__init__()
        self.num_outputs = num_outputs

        # Learnable seed vectors that attend to input
        self.seeds = nn.Parameter(torch.randn(num_outputs, input_dim))

        # Multi-head attention: seeds attend to input
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Project to output dimension
        self.proj = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pool variable-length set to fixed representation.

        Args:
            x: [B, N, D] input features
            mask: [B, N] attention mask (True = valid, False = padding)

        Returns:
            [B, num_outputs, output_dim] pooled features
        """
        batch_size = x.size(0)

        # Expand seeds for batch: [B, num_outputs, D]
        seeds = self.seeds.unsqueeze(0).expand(batch_size, -1, -1)

        # Create key padding mask for attention (True = ignore)
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask  # Invert: True means ignore

        # Seeds attend to input
        pooled, _ = self.attention(
            query=seeds,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
        )

        # Project and normalize
        output = self.proj(pooled)
        output = self.norm(output)

        return output


class VectorQuantizerEMA(nn.Module):
    """Vector Quantization with EMA updates for sparse transformer.

    Simplified version focused on encoding variable-length sets.

    Args:
        num_embeddings: Number of codebook entries
        embedding_dim: Dimension of each code
        commitment_cost: Weight for commitment loss
        decay: EMA decay rate
    """

    def __init__(
        self,
        num_embeddings: int = 1024,
        embedding_dim: int = 256,
        commitment_cost: float = 0.5,
        decay: float = 0.99,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay

        # Codebook
        self.register_buffer("codebook", torch.randn(num_embeddings, embedding_dim))
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_embed_sum", torch.zeros(num_embeddings, embedding_dim))

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize encoder output.

        Args:
            z_e: [B, K, D] encoder output (K = num pooled outputs)

        Returns:
            z_q: Quantized output (same shape)
            vq_loss: Commitment loss
            indices: Codebook indices [B, K]
        """
        batch_size, num_codes, dim = z_e.shape
        flat_z_e = z_e.view(-1, dim)  # [B*K, D]

        # Compute distances
        z_e_sq = (flat_z_e ** 2).sum(dim=1, keepdim=True)
        codebook_sq = (self.codebook ** 2).sum(dim=1, keepdim=True).t()
        distances = z_e_sq + codebook_sq - 2 * torch.mm(flat_z_e, self.codebook.t())

        # Find nearest
        indices = distances.argmin(dim=1)  # [B*K]
        z_q_flat = F.embedding(indices, self.codebook)

        # EMA update during training
        if self.training:
            encodings = F.one_hot(indices, self.num_embeddings).float()
            batch_cluster_size = encodings.sum(0)
            self.ema_cluster_size.mul_(self.decay).add_(batch_cluster_size, alpha=1 - self.decay)

            batch_embed_sum = encodings.t() @ flat_z_e
            self.ema_embed_sum.mul_(self.decay).add_(batch_embed_sum, alpha=1 - self.decay)

            # Update codebook
            n = self.ema_cluster_size.sum()
            smoothed = (self.ema_cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n
            self.codebook.data.copy_(self.ema_embed_sum / smoothed.unsqueeze(1))

        # Reshape
        z_q = z_q_flat.view(batch_size, num_codes, dim)
        indices = indices.view(batch_size, num_codes)

        # Commitment loss
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        vq_loss = self.commitment_cost * commitment_loss

        # Straight-through estimator
        z_q = z_e + (z_q - z_e).detach()

        return z_q, vq_loss, indices


class SparseStructureTransformer(nn.Module):
    """Transformer-based autoencoder for sparse Minecraft structures.

    Encodes structures as sets of (position, block_embedding) pairs,
    compresses to a latent representation, and decodes back to
    block embeddings which are matched to the vocabulary.

    Args:
        embed_dim: Block embedding dimension (from Block2Vec)
        hidden_dim: Transformer hidden dimension
        n_encoder_layers: Number of encoder transformer layers
        n_decoder_layers: Number of decoder transformer layers
        n_heads: Number of attention heads
        pos_frequencies: Number of Fourier frequency bands
        max_coord: Maximum coordinate value
        num_latent_codes: Number of latent codes after pooling
        vq_num_embeddings: VQ codebook size (0 = no VQ)
        dropout: Dropout rate
    """

    def __init__(
        self,
        embed_dim: int = 40,
        hidden_dim: int = 256,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        n_heads: int = 8,
        pos_frequencies: int = 10,
        max_coord: int = 32,
        num_latent_codes: int = 16,
        vq_num_embeddings: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_latent_codes = num_latent_codes
        self.use_vq = vq_num_embeddings > 0

        # Positional encoding for 3D coordinates
        self.pos_encoder = FourierPositionalEncoding(
            num_frequencies=pos_frequencies,
            max_coord=max_coord,
            include_input=True,
        )
        pos_dim = self.pos_encoder.output_dim

        # Input projection: position features + block embedding -> hidden_dim
        self.input_proj = nn.Linear(pos_dim + embed_dim, hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        # Latent compression via pooling
        self.pool = SetPooling(
            input_dim=hidden_dim,
            output_dim=hidden_dim,
            num_outputs=num_latent_codes,
            num_heads=n_heads,
        )

        # Optional vector quantization
        self.vq: Optional[VectorQuantizerEMA] = None
        if self.use_vq:
            self.vq = VectorQuantizerEMA(
                num_embeddings=vq_num_embeddings,
                embedding_dim=hidden_dim,
                commitment_cost=0.5,
            )

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)

        # Position query projection (for decoder)
        self.pos_query_proj = nn.Linear(pos_dim, hidden_dim)

        # Output projection: hidden_dim -> block embedding
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Glorot."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(
        self,
        positions: torch.Tensor,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode sparse structure to latent representation.

        Args:
            positions: [B, N, 3] block positions
            embeddings: [B, N, embed_dim] block embeddings
            attention_mask: [B, N] True = valid, False = padding

        Returns:
            latent: [B, num_latent_codes, hidden_dim] latent representation
            vq_loss: VQ loss if using VQ, else None
        """
        # Encode positions with Fourier features
        pos_features = self.pos_encoder(positions)  # [B, N, pos_dim]

        # Combine position and block embeddings
        combined = torch.cat([pos_features, embeddings], dim=-1)  # [B, N, pos_dim + embed_dim]
        x = self.input_proj(combined)  # [B, N, hidden_dim]

        # Create src_key_padding_mask for transformer (True = ignore)
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask

        # Transformer encoder
        encoded = self.encoder(x, src_key_padding_mask=src_key_padding_mask)  # [B, N, hidden_dim]

        # Pool to fixed-size latent
        latent = self.pool(encoded, mask=attention_mask)  # [B, num_latent_codes, hidden_dim]

        # Optional vector quantization
        vq_loss = None
        if self.use_vq and self.vq is not None:
            latent, vq_loss, _ = self.vq(latent)

        return latent, vq_loss

    def decode(
        self,
        latent: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode latent representation to block embeddings.

        Args:
            latent: [B, num_latent_codes, hidden_dim] latent representation
            positions: [B, N, 3] target block positions
            attention_mask: [B, N] True = valid, False = padding

        Returns:
            pred_embeddings: [B, N, embed_dim] predicted block embeddings
        """
        # Encode target positions
        pos_features = self.pos_encoder(positions)  # [B, N, pos_dim]
        queries = self.pos_query_proj(pos_features)  # [B, N, hidden_dim]

        # Create tgt_key_padding_mask for transformer (True = ignore)
        tgt_key_padding_mask = None
        if attention_mask is not None:
            tgt_key_padding_mask = ~attention_mask

        # Transformer decoder: queries attend to latent
        decoded = self.decoder(
            tgt=queries,
            memory=latent,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )  # [B, N, hidden_dim]

        # Project to embedding space
        pred_embeddings = self.output_proj(decoded)  # [B, N, embed_dim]

        return pred_embeddings

    def forward(
        self,
        positions: torch.Tensor,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass: encode and decode.

        Args:
            positions: [B, N, 3] block positions
            embeddings: [B, N, embed_dim] block embeddings
            attention_mask: [B, N] True = valid, False = padding

        Returns:
            Dictionary with:
                - pred_embeddings: [B, N, embed_dim] predicted embeddings
                - vq_loss: VQ loss if using VQ, else 0
        """
        latent, vq_loss = self.encode(positions, embeddings, attention_mask)
        pred_embeddings = self.decode(latent, positions, attention_mask)

        return {
            "pred_embeddings": pred_embeddings,
            "vq_loss": vq_loss if vq_loss is not None else torch.tensor(0.0, device=positions.device),
        }

    def compute_loss(
        self,
        positions: torch.Tensor,
        embeddings: torch.Tensor,
        block_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        all_embeddings: Optional[torch.Tensor] = None,
        aux_weight: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """Compute training loss.

        Primary loss: MSE between predicted and target embeddings
        Auxiliary loss: Cross-entropy for block classification (optional)

        Args:
            positions: [B, N, 3] block positions
            embeddings: [B, N, embed_dim] target block embeddings
            block_ids: [B, N] target block IDs
            attention_mask: [B, N] True = valid, False = padding
            all_embeddings: [vocab_size, embed_dim] full embedding matrix (for aux loss)
            aux_weight: Weight for auxiliary classification loss

        Returns:
            Dictionary with loss components
        """
        outputs = self(positions, embeddings, attention_mask)
        pred_embeddings = outputs["pred_embeddings"]
        vq_loss = outputs["vq_loss"]

        # Create mask for valid positions
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)  # [B, N, 1]
        else:
            mask = torch.ones_like(pred_embeddings[..., :1])

        # Embedding reconstruction loss (MSE)
        embed_diff = (pred_embeddings - embeddings) ** 2  # [B, N, embed_dim]
        embed_loss = (embed_diff * mask).sum() / mask.sum() / pred_embeddings.size(-1)

        # Total loss
        total_loss = embed_loss + vq_loss

        # Auxiliary classification loss (optional)
        aux_loss = torch.tensor(0.0, device=positions.device)
        accuracy = torch.tensor(0.0, device=positions.device)

        if all_embeddings is not None and aux_weight > 0:
            # Compute distances to all block embeddings
            # pred_embeddings: [B, N, D], all_embeddings: [V, D]
            pred_flat = pred_embeddings.view(-1, pred_embeddings.size(-1))  # [B*N, D]
            distances = torch.cdist(pred_flat, all_embeddings)  # [B*N, V]
            logits = -distances  # Higher = closer = better

            # Cross-entropy loss
            targets_flat = block_ids.view(-1)
            mask_flat = attention_mask.view(-1) if attention_mask is not None else torch.ones_like(targets_flat, dtype=torch.bool)

            ce_loss = F.cross_entropy(logits, targets_flat, reduction='none')
            aux_loss = (ce_loss * mask_flat.float()).sum() / mask_flat.sum()

            total_loss = total_loss + aux_weight * aux_loss

            # Compute accuracy
            with torch.no_grad():
                preds = logits.argmin(dim=1)  # Nearest neighbor
                correct = (preds == targets_flat).float()
                accuracy = (correct * mask_flat.float()).sum() / mask_flat.sum()

        return {
            "loss": total_loss,
            "embed_loss": embed_loss,
            "vq_loss": vq_loss,
            "aux_loss": aux_loss,
            "accuracy": accuracy,
        }

    @torch.no_grad()
    def predict_blocks(
        self,
        positions: torch.Tensor,
        embeddings: torch.Tensor,
        all_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict block IDs via nearest neighbor in embedding space.

        Args:
            positions: [B, N, 3] block positions
            embeddings: [B, N, embed_dim] input block embeddings
            all_embeddings: [vocab_size, embed_dim] full embedding matrix
            attention_mask: [B, N] True = valid

        Returns:
            pred_block_ids: [B, N] predicted block IDs
        """
        outputs = self(positions, embeddings, attention_mask)
        pred_embeddings = outputs["pred_embeddings"]  # [B, N, embed_dim]

        # Nearest neighbor lookup
        B, N, D = pred_embeddings.shape
        pred_flat = pred_embeddings.view(-1, D)  # [B*N, D]
        distances = torch.cdist(pred_flat, all_embeddings)  # [B*N, V]
        pred_ids = distances.argmin(dim=1)  # [B*N]

        return pred_ids.view(B, N)

    def get_num_params(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SparseStructureTransformerLite(nn.Module):
    """Lightweight version without VQ for faster iteration.

    Simpler architecture:
    - Direct autoencoder without quantization
    - Can be used for ablation studies
    """

    def __init__(
        self,
        embed_dim: int = 40,
        hidden_dim: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        pos_frequencies: int = 6,
        max_coord: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Positional encoding
        self.pos_encoder = FourierPositionalEncoding(
            num_frequencies=pos_frequencies,
            max_coord=max_coord,
        )
        pos_dim = self.pos_encoder.output_dim

        # Input projection
        self.input_proj = nn.Linear(pos_dim + embed_dim, hidden_dim)

        # Transformer encoder (acts as autoencoder with self-attention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, embed_dim)

    def forward(
        self,
        positions: torch.Tensor,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        pos_features = self.pos_encoder(positions)
        x = torch.cat([pos_features, embeddings], dim=-1)
        x = self.input_proj(x)

        src_key_padding_mask = ~attention_mask if attention_mask is not None else None
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        pred_embeddings = self.output_proj(x)

        return {
            "pred_embeddings": pred_embeddings,
            "vq_loss": torch.tensor(0.0, device=positions.device),
        }

