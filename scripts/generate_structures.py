"""Generate Minecraft structures using trained prior and VQ-VAE models."""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Set

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.export.schematic_exporter import export_batch_to_schematic


# ============================================================================
# RFSQ Modules (copied from training notebook)
# ============================================================================


class FSQ(nn.Module):
    def __init__(self, levels: List[int], eps: float = 1e-3):
        super().__init__()
        self.levels = levels
        self.dim = len(levels)
        self.eps = eps
        self.codebook_size = int(np.prod(levels))
        self.register_buffer("_levels", torch.tensor(levels, dtype=torch.float32))
        basis = []
        acc = 1
        for L in reversed(levels):
            basis.append(acc)
            acc *= L
        self.register_buffer("_basis", torch.tensor(list(reversed(basis)), dtype=torch.long))
        half_levels = [(L - 1) / 2 for L in levels]
        self.register_buffer("_half_levels", torch.tensor(half_levels, dtype=torch.float32))

    def forward(self, z: torch.Tensor):
        z_bounded = torch.tanh(z)
        z_q = self._quantize(z_bounded)
        z_q = z_bounded + (z_q - z_bounded).detach()
        indices = self._to_indices(z_q)
        return z_q, indices

    def _quantize(self, z: torch.Tensor) -> torch.Tensor:
        z_q_list = []
        for i in range(self.dim):
            half_L = self._half_levels[i]
            z_i = z[..., i] * half_L
            z_i = torch.round(z_i).clamp(-half_L, half_L) / half_L
            z_q_list.append(z_i)
        return torch.stack(z_q_list, dim=-1)

    def _to_indices(self, z_q: torch.Tensor) -> torch.Tensor:
        indices = torch.zeros(z_q.shape[:-1], dtype=torch.long, device=z_q.device)
        for i in range(self.dim):
            L = self._levels[i].long()
            half_L = self._half_levels[i]
            level_idx = ((z_q[..., i] * half_L) + half_L).round().long().clamp(0, L - 1)
            indices = indices + level_idx * self._basis[i]
        return indices

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        codes = []
        remaining = indices.clone()
        for i in range(self.dim):
            L = self._levels[i].long().item()
            half_L = self._half_levels[i].item()
            level_idx = remaining % L
            remaining = remaining // L
            code_val = (level_idx.float() - half_L) / half_L
            codes.append(code_val)
        return torch.stack(codes[::-1], dim=-1)


class InvertibleLayerNorm(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("stored_mean", None, persistent=False)
        self.register_buffer("stored_std", None, persistent=False)
        # Fixed statistics for generation (loaded from training data)
        self.register_buffer("fixed_mean", torch.zeros(1, 1, 1, 1, num_features))
        self.register_buffer("fixed_std", torch.ones(1, 1, 1, 1, num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.stored_mean = x.mean(dim=(1, 2, 3), keepdim=True)
        self.stored_std = x.std(dim=(1, 2, 3), keepdim=True) + self.eps
        x_norm = (x - self.stored_mean) / self.stored_std
        return x_norm * self.weight + self.bias

    def inverse(self, x_norm: torch.Tensor) -> torch.Tensor:
        x = (x_norm - self.bias) / self.weight
        return x * self.stored_std + self.stored_mean

    def inverse_fixed(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Inverse using fixed statistics (for generation without encoder pass)."""
        x = (x_norm - self.bias) / self.weight
        return x * self.fixed_std + self.fixed_mean


class RFSQStage(nn.Module):
    def __init__(self, levels: List[int]):
        super().__init__()
        self.fsq = FSQ(levels)
        self.layernorm = InvertibleLayerNorm(len(levels))

    def forward(self, residual: torch.Tensor):
        z_norm = self.layernorm(residual)
        z_q_norm, indices = self.fsq(z_norm)
        z_q = self.layernorm.inverse(z_q_norm)
        new_residual = residual - z_q
        return z_q, new_residual, indices

    def indices_to_quantized(self, indices: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Convert indices back to quantized values (for generation)."""
        z_q_norm = self.fsq.indices_to_codes(indices)
        return z_q_norm

    def indices_to_quantized_fixed(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert indices to quantized values using FIXED LayerNorm stats."""
        z_q_norm = self.fsq.indices_to_codes(indices)
        return self.layernorm.inverse_fixed(z_q_norm)


class RFSQ(nn.Module):
    def __init__(self, levels_per_stage: List[int], num_stages: int = 2):
        super().__init__()
        self.num_stages = num_stages
        self.stages = nn.ModuleList([RFSQStage(levels_per_stage) for _ in range(num_stages)])

    def forward(self, z: torch.Tensor):
        residual = z
        z_q_sum = torch.zeros_like(z)
        all_indices = []
        for stage in self.stages:
            z_q, residual, indices = stage(residual)
            z_q_sum = z_q_sum + z_q
            all_indices.append(indices)
        return z_q_sum, all_indices

    def indices_to_quantized(self, indices_list: List[torch.Tensor]) -> torch.Tensor:
        """Convert list of indices back to quantized latent (for generation).

        This is a simplified version - actual inverse requires statistics from encoding.
        For generation from prior, we pass indices through decoder which handles this.
        """
        z_q_sum = None
        for stage_idx, indices in enumerate(indices_list):
            z_q_norm = self.stages[stage_idx].fsq.indices_to_codes(indices)
            # Apply inverse LayerNorm with dummy stats (decoder will handle properly)
            if z_q_sum is None:
                z_q_sum = z_q_norm
            else:
                z_q_sum = z_q_sum + z_q_norm
        return z_q_sum

    def indices_to_quantized_fixed(self, indices_list: List[torch.Tensor]) -> torch.Tensor:
        """Convert indices to quantized values using FIXED LayerNorm statistics.

        This is the CORRECT method to use for generation from prior!
        """
        z_q_sum = None
        for stage_idx, indices in enumerate(indices_list):
            z_q = self.stages[stage_idx].indices_to_quantized_fixed(indices)
            if z_q_sum is None:
                z_q_sum = z_q
            else:
                z_q_sum = z_q_sum + z_q
        return z_q_sum

    def set_fixed_stats(self, stats: dict) -> None:
        """Load fixed LayerNorm statistics from saved file."""
        for i, stage in enumerate(self.stages):
            stage.layernorm.fixed_mean.copy_(stats[f'stage{i}']['mean'])
            stage.layernorm.fixed_std.copy_(stats[f'stage{i}']['std'])


# ============================================================================
# VQ-VAE v16b Decoder
# ============================================================================


class ResidualBlock3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class DualHeadDecoderV16b(nn.Module):
    def __init__(self, rfsq_dim: int = 4, hidden_dim: int = 128, num_blocks: int = 3717):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv3d(rfsq_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.res1 = nn.Sequential(*[ResidualBlock3D(hidden_dim) for _ in range(4)])
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.res2 = ResidualBlock3D(hidden_dim)
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.trunk_final = ResidualBlock3D(hidden_dim)
        self.binary_head = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.BatchNorm3d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dim // 2, 2, 3, padding=1),
        )
        self.block_head = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dim, num_blocks, 3, padding=1),
        )

    def forward(self, z_q):
        x = self.initial(z_q)
        x = self.res1(x)
        x = self.res2(self.up1(x))
        x = self.up2(x)
        features = self.trunk_final(x)
        return self.binary_head(features), self.block_head(features)


class VQVAEDecoder(nn.Module):
    """Simplified VQ-VAE for generation (decoder only)."""

    def __init__(
        self,
        vocab_size: int = 3717,
        hidden_dim: int = 128,
        rfsq_levels: List[int] = None,
        num_stages: int = 2,
        air_tokens: Set[int] = None,
    ):
        super().__init__()
        if rfsq_levels is None:
            rfsq_levels = [5, 5, 5, 5]
        self.rfsq_dim = len(rfsq_levels)
        self.vocab_size = vocab_size
        self.air_tokens = air_tokens or {102, 576, 3352}

        self.quantizer = RFSQ(rfsq_levels, num_stages)
        self.decoder = DualHeadDecoderV16b(self.rfsq_dim, hidden_dim, vocab_size)

    def load_fixed_stats(self, stats_path: str, device: torch.device) -> None:
        """Load fixed LayerNorm statistics for generation."""
        stats = torch.load(stats_path, map_location=device)
        self.quantizer.set_fixed_stats(stats)
        print(f"Loaded LayerNorm statistics from {stats_path}")

    def decode_from_indices(self, indices: List[torch.Tensor]) -> torch.Tensor:
        """Decode RFSQ indices to block IDs.

        Args:
            indices: List of [B, 8, 8, 8] tensors (one per stage)

        Returns:
            Block IDs tensor of shape [B, 32, 32, 32]
        """
        # Convert indices to quantized latent codes using FIXED statistics
        z_q = self.quantizer.indices_to_quantized_fixed(indices)

        # Permute from [B, 8, 8, 8, 4] to [B, 4, 8, 8, 8]
        z_q = z_q.permute(0, 4, 1, 2, 3)

        # Decode
        binary_logits, block_logits = self.decoder(z_q)

        # Dual-head prediction
        binary_pred = binary_logits.argmax(dim=1)  # [B, 32, 32, 32]
        block_pred = block_logits.argmax(dim=1)  # [B, 32, 32, 32]

        # Combine: if binary predicts air (0), use air token; else use block type
        air_token = 102  # minecraft:air
        final_pred = torch.where(binary_pred == 0, air_token, block_pred)

        return final_pred


# ============================================================================
# Transformer Prior
# ============================================================================


class PositionalEncoding3D(nn.Module):
    def __init__(self, d_model: int, grid_size: int = 8):
        super().__init__()
        self.d_model = d_model
        self.grid_size = grid_size
        self.pos_embedding = nn.Parameter(torch.randn(1, grid_size * grid_size * grid_size, d_model) * 0.02)
        self.stage_embedding = nn.Embedding(2, d_model)

    def forward(self, seq_len: int, stage: int, device: torch.device) -> torch.Tensor:
        pos_enc = self.pos_embedding[:, :seq_len, :].to(device)
        stage_enc = self.stage_embedding(torch.tensor([stage], device=device)).unsqueeze(0)
        return pos_enc + stage_enc


class TransformerPrior(nn.Module):
    def __init__(
        self,
        vocab_size: int = 625,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        num_stages: int = 2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_stages = num_stages

        self.token_embedding = nn.Embedding(vocab_size + 1, d_model)
        self.bos_token = vocab_size

        self.pos_encoding = PositionalEncoding3D(d_model, grid_size=8)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.output_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)

        self.register_buffer("causal_mask", torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool())

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, indices: torch.Tensor, stage: int = 0) -> torch.Tensor:
        B, seq_len = indices.shape
        x = self.token_embedding(indices)
        x = x + self.pos_encoding(seq_len, stage, indices.device)
        mask = self.causal_mask[:seq_len, :seq_len]
        x = self.transformer(x, mask=mask)
        x = self.output_norm(x)
        logits = self.output_head(x)
        return logits

    @torch.no_grad()
    def generate(
        self,
        batch_size: int = 1,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
        device: torch.device = None,
    ) -> List[torch.Tensor]:
        """Generate latent codes autoregressively."""
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        all_indices = []

        for stage in range(self.num_stages):
            generated = torch.full((batch_size, 1), self.bos_token, dtype=torch.long, device=device)

            for pos in range(self.max_seq_len):
                logits = self.forward(generated, stage=stage)
                next_logits = logits[:, -1, :] / temperature

                if top_k is not None:
                    v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < v[:, [-1]]] = float("-inf")

                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_logits[indices_to_remove] = float("-inf")

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)

            stage_indices = generated[:, 1:].reshape(batch_size, 8, 8, 8)
            all_indices.append(stage_indices)

        return all_indices


# ============================================================================
# Main Generation Script
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Generate Minecraft structures using trained models")
    parser.add_argument("--prior-checkpoint", type=str, required=True, help="Path to prior checkpoint")
    parser.add_argument("--vqvae-checkpoint", type=str, required=True, help="Path to VQ-VAE checkpoint")
    parser.add_argument("--layernorm-stats", type=str, required=True, help="Path to layernorm_stats.pt")
    parser.add_argument("--vocab", type=str, required=True, help="Path to tok2block.json")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for schematic files")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of structures to generate")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling (None for disabled)")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p nucleus sampling")
    parser.add_argument("--prefix", type=str, default="generated", help="Output filename prefix")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load vocabulary
    print(f"\nLoading vocabulary from {args.vocab}")
    with open(args.vocab, "r") as f:
        tok2block = {int(k): v for k, v in json.load(f).items()}
    vocab_size = len(tok2block)
    print(f"Vocabulary size: {vocab_size}")

    # Load models
    print(f"\nLoading VQ-VAE from {args.vqvae_checkpoint}")
    vqvae = VQVAEDecoder(vocab_size=vocab_size).to(device)
    vqvae.load_state_dict(torch.load(args.vqvae_checkpoint, map_location=device), strict=False)
    vqvae.eval()

    # Load LayerNorm statistics (CRITICAL for generation!)
    print(f"Loading LayerNorm stats from {args.layernorm_stats}")
    vqvae.load_fixed_stats(args.layernorm_stats, device)

    print(f"\nLoading prior from {args.prior_checkpoint}")
    prior = TransformerPrior().to(device)
    prior.load_state_dict(torch.load(args.prior_checkpoint, map_location=device))
    prior.eval()

    # Generate
    print(f"\nGenerating {args.num_samples} structures...")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-k: {args.top_k}")
    print(f"  Top-p: {args.top_p}")

    with torch.no_grad():
        # Generate latent codes from prior
        latent_indices = prior.generate(
            batch_size=args.num_samples,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
        )

        # Decode to structures
        structures = vqvae.decode_from_indices(latent_indices)

    print(f"Generated structures shape: {structures.shape}")

    # Export to schematic
    print(f"\nExporting to {args.output_dir}...")
    saved_paths = export_batch_to_schematic(
        structures=structures,
        tok2block=tok2block,
        output_dir=args.output_dir,
        prefix=args.prefix,
    )

    print(f"\n{'='*70}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"Generated {len(saved_paths)} structures")
    print(f"\nSaved to:")
    for path in saved_paths:
        print(f"  {path}")

    print(f"\nTo paste in Minecraft:")
    print(f"  1. Copy .schematic files to: server/plugins/FastAsyncWorldEdit/schematics/")
    print(f"  2. In-game: //schematic load {args.prefix}_0")
    print(f"  3. Position yourself and: //paste")


if __name__ == "__main__":
    main()
