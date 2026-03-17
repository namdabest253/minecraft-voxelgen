"""
Collect LayerNorm statistics from training data for VQ-VAE v16b.

This script runs a pass over the training data to compute average
LayerNorm statistics (mean and std) for each RFSQ stage. These
statistics are then saved and can be loaded for generation.

Usage:
    python scripts/collect_layernorm_stats.py \
        --checkpoint data/output/vqvae/v16b/vqvae_v16b_best.pt \
        --embeddings data/output/block2vec/v3/block_embeddings_v3.npy \
        --vocab data/vocabulary/tok2block.json \
        --data-dir data/splits/train \
        --output data/output/vqvae/v16b/layernorm_stats.pt
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# VQ-VAE v16b Model Definition (same as training)
# ============================================================================

class FSQ(nn.Module):
    def __init__(self, levels: List[int], eps: float = 1e-3):
        super().__init__()
        self.levels = levels
        self.dim = len(levels)
        self.eps = eps
        self.codebook_size = int(np.prod(levels))
        self.register_buffer('_levels', torch.tensor(levels, dtype=torch.float32))
        basis = []
        acc = 1
        for L in reversed(levels):
            basis.append(acc)
            acc *= L
        self.register_buffer('_basis', torch.tensor(list(reversed(basis)), dtype=torch.long))
        half_levels = [(L - 1) / 2 for L in levels]
        self.register_buffer('_half_levels', torch.tensor(half_levels, dtype=torch.float32))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        self.register_buffer('stored_mean', None, persistent=False)
        self.register_buffer('stored_std', None, persistent=False)
        self.register_buffer('fixed_mean', torch.zeros(1, 1, 1, 1, num_features))
        self.register_buffer('fixed_std', torch.ones(1, 1, 1, 1, num_features))
        self.use_fixed_stats = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels_last = x.shape[-1] == self.num_features
        if channels_last:
            self.stored_mean = x.mean(dim=(1, 2, 3), keepdim=True)
            self.stored_std = x.std(dim=(1, 2, 3), keepdim=True) + self.eps
            x_norm = (x - self.stored_mean) / self.stored_std
            return x_norm * self.weight + self.bias
        else:
            self.stored_mean = x.mean(dim=(2, 3, 4), keepdim=True)
            self.stored_std = x.std(dim=(2, 3, 4), keepdim=True) + self.eps
            x_norm = (x - self.stored_mean) / self.stored_std
            return x_norm * self.weight.view(1, -1, 1, 1, 1) + self.bias.view(1, -1, 1, 1, 1)

    def inverse(self, x_norm: torch.Tensor) -> torch.Tensor:
        channels_last = x_norm.shape[-1] == self.num_features
        if channels_last:
            x = (x_norm - self.bias) / self.weight
            if self.use_fixed_stats:
                return x * self.fixed_std + self.fixed_mean
            return x * self.stored_std + self.stored_mean
        else:
            x = (x_norm - self.bias.view(1, -1, 1, 1, 1)) / self.weight.view(1, -1, 1, 1, 1)
            if self.use_fixed_stats:
                fixed_mean_cf = self.fixed_mean.permute(0, 4, 1, 2, 3)
                fixed_std_cf = self.fixed_std.permute(0, 4, 1, 2, 3)
                return x * fixed_std_cf + fixed_mean_cf
            return x * self.stored_std + self.stored_mean


class RFSQStage(nn.Module):
    def __init__(self, levels: List[int]):
        super().__init__()
        self.levels = levels
        self.fsq = FSQ(levels)
        self.layernorm = InvertibleLayerNorm(len(levels))

    def forward(self, residual: torch.Tensor):
        z_norm = self.layernorm(residual)
        z_q_norm, indices = self.fsq(z_norm)
        z_q = self.layernorm.inverse(z_q_norm)
        new_residual = residual - z_q
        return z_q, new_residual, indices


class RFSQ(nn.Module):
    def __init__(self, levels_per_stage: List[int], num_stages: int = 2):
        super().__init__()
        self.num_stages = num_stages
        self.levels_per_stage = levels_per_stage
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


class EncoderV16b(nn.Module):
    def __init__(self, in_channels: int = 40, hidden_dim: int = 128, rfsq_dim: int = 4):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.down1 = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.res1 = ResidualBlock3D(hidden_dim)
        self.down2 = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.res2 = nn.Sequential(*[ResidualBlock3D(hidden_dim) for _ in range(4)])
        self.proj = nn.Conv3d(hidden_dim, rfsq_dim, 3, padding=1)

    def forward(self, x):
        x = self.initial(x)
        x = self.res1(self.down1(x))
        x = self.res2(self.down2(x))
        return self.proj(x)


class DualHeadDecoderV16b(nn.Module):
    def __init__(self, rfsq_dim: int = 4, hidden_dim: int = 128, num_blocks: int = 3717):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv3d(rfsq_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.res1 = nn.Sequential(*[ResidualBlock3D(hidden_dim) for _ in range(4)])
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.res2 = ResidualBlock3D(hidden_dim)
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.trunk_final = ResidualBlock3D(hidden_dim)
        self.binary_head = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.BatchNorm3d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dim // 2, 2, 3, padding=1)
        )
        self.block_head = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dim, num_blocks, 3, padding=1)
        )

    def forward(self, z_q):
        x = self.initial(z_q)
        x = self.res1(x)
        x = self.res2(self.up1(x))
        x = self.up2(x)
        features = self.trunk_final(x)
        return self.binary_head(features), self.block_head(features)


class VQVAEv16b(nn.Module):
    def __init__(self, vocab_size: int = 3717, emb_dim: int = 40, hidden_dim: int = 128,
                 rfsq_levels: List[int] = None, num_stages: int = 2,
                 pretrained_embeddings: torch.Tensor = None,
                 air_tokens: Set[int] = None):
        super().__init__()
        if rfsq_levels is None:
            rfsq_levels = [5, 5, 5, 5]
        self.rfsq_dim = len(rfsq_levels)
        self.vocab_size = vocab_size
        self.air_tokens = air_tokens or {102, 576, 3352}

        self.block_emb = nn.Embedding(vocab_size, emb_dim)
        if pretrained_embeddings is not None:
            self.block_emb.weight.data.copy_(pretrained_embeddings)
            self.block_emb.weight.requires_grad = False

        self.encoder = EncoderV16b(emb_dim, hidden_dim, self.rfsq_dim)
        self.quantizer = RFSQ(rfsq_levels, num_stages)
        self.decoder = DualHeadDecoderV16b(self.rfsq_dim, hidden_dim, vocab_size)

    def encode(self, block_ids: torch.Tensor) -> torch.Tensor:
        """Encode structures to latent space."""
        x = self.block_emb(block_ids).permute(0, 4, 1, 2, 3)
        z_e = self.encoder(x).permute(0, 2, 3, 4, 1)
        return z_e


def collect_statistics(
    vqvae: VQVAEv16b,
    data_dir: str,
    device: torch.device,
    max_samples: int = None,
    batch_size: int = 8,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Collect LayerNorm statistics from training data.

    Args:
        vqvae: VQ-VAE model
        data_dir: Directory with H5 training files
        device: Device to use
        max_samples: Maximum samples to use (None for all)
        batch_size: Processing batch size

    Returns:
        Dictionary with average statistics for each stage
    """
    vqvae.eval()

    # Find all H5 files
    data_path = Path(data_dir)
    h5_files = sorted(data_path.glob("*.h5"))
    if max_samples:
        h5_files = h5_files[:max_samples]

    print(f"Collecting statistics from {len(h5_files)} files...")

    # Accumulate statistics
    all_stats = {
        'stage0': {'means': [], 'stds': []},
        'stage1': {'means': [], 'stds': []},
    }

    with torch.no_grad():
        for i in tqdm(range(0, len(h5_files), batch_size), desc="Processing"):
            batch_files = h5_files[i:i+batch_size]

            # Load batch
            structures = []
            for h5_path in batch_files:
                try:
                    with h5py.File(h5_path, 'r') as f:
                        key = list(f.keys())[0]
                        structure = f[key][:].astype(np.int64)
                        structures.append(torch.from_numpy(structure).long())
                except Exception as e:
                    print(f"Warning: Failed to load {h5_path}: {e}")
                    continue

            if not structures:
                continue

            # Stack and process
            batch = torch.stack(structures).to(device)
            z_e = vqvae.encode(batch)

            # Run through RFSQ stages and collect statistics
            residual = z_e
            for stage_idx, stage in enumerate(vqvae.quantizer.stages):
                # Forward through layernorm
                z_norm = stage.layernorm(residual)
                z_q_norm, indices = stage.fsq(z_norm)
                z_q = stage.layernorm.inverse(z_q_norm)
                residual = residual - z_q

                # Store statistics (averaged across batch)
                mean_avg = stage.layernorm.stored_mean.mean(dim=0, keepdim=True)
                std_avg = stage.layernorm.stored_std.mean(dim=0, keepdim=True)

                all_stats[f'stage{stage_idx}']['means'].append(mean_avg.cpu())
                all_stats[f'stage{stage_idx}']['stds'].append(std_avg.cpu())

    # Compute overall averages
    final_stats = {}
    for stage_key in ['stage0', 'stage1']:
        all_means = torch.cat(all_stats[stage_key]['means'], dim=0)
        all_stds = torch.cat(all_stats[stage_key]['stds'], dim=0)

        final_stats[stage_key] = {
            'mean': all_means.mean(dim=0, keepdim=True),
            'std': all_stds.mean(dim=0, keepdim=True),
        }

        print(f"\n{stage_key}:")
        print(f"  Mean shape: {final_stats[stage_key]['mean'].shape}")
        print(f"  Mean values: {final_stats[stage_key]['mean'].squeeze()}")
        print(f"  Std shape: {final_stats[stage_key]['std'].shape}")
        print(f"  Std values: {final_stats[stage_key]['std'].squeeze()}")

    return final_stats


def main():
    parser = argparse.ArgumentParser(description="Collect LayerNorm statistics")
    parser.add_argument("--checkpoint", type=str, required=True, help="VQ-VAE checkpoint path")
    parser.add_argument("--embeddings", type=str, required=True, help="Block embeddings path")
    parser.add_argument("--vocab", type=str, required=True, help="tok2block.json path")
    parser.add_argument("--data-dir", type=str, required=True, help="Training data directory")
    parser.add_argument("--output", type=str, required=True, help="Output stats file path")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to process")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load vocabulary
    print(f"\nLoading vocabulary from {args.vocab}")
    with open(args.vocab, "r") as f:
        tok2block = {int(k): v for k, v in json.load(f).items()}
    vocab_size = len(tok2block)

    # Find air tokens
    air_tokens: Set[int] = set()
    for tok, block in tok2block.items():
        if 'air' in block.lower() and 'stair' not in block.lower():
            air_tokens.add(tok)

    # Load embeddings
    print(f"Loading embeddings from {args.embeddings}")
    embeddings = np.load(args.embeddings).astype(np.float32)
    emb_dim = embeddings.shape[1]

    # Create model
    print(f"Creating VQ-VAE v16b...")
    vqvae = VQVAEv16b(
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        hidden_dim=128,
        rfsq_levels=[5, 5, 5, 5],
        num_stages=2,
        pretrained_embeddings=torch.from_numpy(embeddings),
        air_tokens=air_tokens,
    ).to(device)

    # Load checkpoint (strict=False for new fixed_mean/fixed_std buffers)
    print(f"Loading checkpoint from {args.checkpoint}")
    vqvae.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=False)
    vqvae.eval()

    # Collect statistics
    stats = collect_statistics(
        vqvae,
        args.data_dir,
        device,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
    )

    # Save statistics
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, output_path)

    print(f"\n{'='*70}")
    print(f"Statistics saved to: {output_path}")
    print(f"{'='*70}")
    print("\nTo use in generation:")
    print(f"  stats = torch.load('{output_path}')")
    print(f"  vqvae.quantizer.set_fixed_stats(stats)")
    print(f"  z_q = vqvae.quantizer.indices_to_codes_fixed(indices)")


if __name__ == "__main__":
    main()
