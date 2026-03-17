"""
Test VQ-VAE v16b reconstruction quality and validate generation pipeline.

This script tests two paths:
1. Normal path: encode → quantize → decode (should work well)
2. Index path: encode → quantize → indices_to_codes → decode (tests generation pipeline)

If path 1 works but path 2 fails, it confirms the generation pipeline issue.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.export.schematic_exporter import export_to_schematic


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
        """Convert indices back to quantized values (normalized space)."""
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.stored_mean = x.mean(dim=(1, 2, 3), keepdim=True)
        self.stored_std = x.std(dim=(1, 2, 3), keepdim=True) + self.eps
        x_norm = (x - self.stored_mean) / self.stored_std
        return x_norm * self.weight + self.bias

    def inverse(self, x_norm: torch.Tensor) -> torch.Tensor:
        x = (x_norm - self.bias) / self.weight
        return x * self.stored_std + self.stored_mean


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

    def indices_to_codes_naive(self, indices_list: List[torch.Tensor]) -> torch.Tensor:
        """Naive conversion: just sum the FSQ codes without LayerNorm inverse.

        This is what the generation script does - and likely broken!
        """
        z_q_sum = None
        for stage_idx, indices in enumerate(indices_list):
            z_q_codes = self.stages[stage_idx].fsq.indices_to_codes(indices)
            if z_q_sum is None:
                z_q_sum = z_q_codes
            else:
                z_q_sum = z_q_sum + z_q_codes
        return z_q_sum


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

    def forward(self, block_ids: torch.Tensor):
        """Full forward pass - returns z_q and indices."""
        x = self.block_emb(block_ids).permute(0, 4, 1, 2, 3)
        z_e = self.encoder(x).permute(0, 2, 3, 4, 1)  # [B, 8, 8, 8, 4]
        z_q, indices = self.quantizer(z_e)
        return z_e, z_q, indices

    def decode(self, z_q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode z_q to predictions."""
        z_q_permuted = z_q.permute(0, 4, 1, 2, 3)  # [B, 4, 8, 8, 8]
        binary_logits, block_logits = self.decoder(z_q_permuted)
        return binary_logits, block_logits

    def predict_blocks(self, binary_logits: torch.Tensor, block_logits: torch.Tensor) -> torch.Tensor:
        """Combine binary and block predictions."""
        binary_pred = binary_logits.argmax(dim=1)  # [B, 32, 32, 32]
        block_pred = block_logits.argmax(dim=1)    # [B, 32, 32, 32]
        air_token = 102
        return torch.where(binary_pred == 0, air_token, block_pred)


# ============================================================================
# Testing Functions
# ============================================================================

def compute_metrics(pred: torch.Tensor, target: torch.Tensor, air_tokens: Set[int]) -> Dict:
    """Compute reconstruction metrics."""
    air_tokens_tensor = torch.tensor(list(air_tokens), device=pred.device)

    gt_air = torch.isin(target, air_tokens_tensor)
    pred_air = torch.isin(pred, air_tokens_tensor)
    gt_building = ~gt_air

    # Building accuracy (correct block type where there is structure)
    if gt_building.sum() > 0:
        building_acc = (pred[gt_building] == target[gt_building]).float().mean().item()
    else:
        building_acc = 0.0

    # Recall (predicted structure where there is structure)
    if gt_building.sum() > 0:
        recall = (~pred_air[gt_building]).float().mean().item()
    else:
        recall = 1.0

    # False air rate
    if gt_building.sum() > 0:
        far = pred_air[gt_building].float().mean().item()
    else:
        far = 0.0

    # Volume ratio
    pred_volume = (~pred_air).sum().item()
    gt_volume = gt_building.sum().item()
    vol_ratio = pred_volume / max(1, gt_volume)

    # Overall accuracy
    overall_acc = (pred == target).float().mean().item()

    return {
        'building_acc': building_acc,
        'recall': recall,
        'far': far,
        'vol_ratio': vol_ratio,
        'overall_acc': overall_acc,
    }


def test_reconstruction(
    vqvae: VQVAEv16b,
    structures: List[torch.Tensor],
    device: torch.device,
    air_tokens: Set[int],
) -> Dict:
    """Test reconstruction via both paths.

    Path 1 (Normal): encode → quantize → decode
    Path 2 (Index): encode → quantize → indices → indices_to_codes → decode
    """
    vqvae.eval()

    results = {
        'normal': {'building_acc': [], 'recall': [], 'far': [], 'vol_ratio': []},
        'index': {'building_acc': [], 'recall': [], 'far': [], 'vol_ratio': []},
    }

    with torch.no_grad():
        for structure in structures:
            block_ids = structure.unsqueeze(0).to(device)

            # Forward pass
            z_e, z_q, indices = vqvae(block_ids)

            # Path 1: Normal decode using z_q
            binary_logits, block_logits = vqvae.decode(z_q)
            pred_normal = vqvae.predict_blocks(binary_logits, block_logits).squeeze(0)

            # Path 2: Decode using indices_to_codes (naive)
            z_q_from_indices = vqvae.quantizer.indices_to_codes_naive(indices)
            binary_logits2, block_logits2 = vqvae.decode(z_q_from_indices)
            pred_index = vqvae.predict_blocks(binary_logits2, block_logits2).squeeze(0)

            # Compute metrics
            metrics_normal = compute_metrics(pred_normal, structure.to(device), air_tokens)
            metrics_index = compute_metrics(pred_index, structure.to(device), air_tokens)

            for key in ['building_acc', 'recall', 'far', 'vol_ratio']:
                results['normal'][key].append(metrics_normal[key])
                results['index'][key].append(metrics_index[key])

    # Average results
    summary = {}
    for path in ['normal', 'index']:
        summary[path] = {
            key: np.mean(values) for key, values in results[path].items()
        }

    return summary, results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test VQ-VAE v16b reconstruction")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to VQ-VAE checkpoint")
    parser.add_argument("--embeddings", type=str, required=True, help="Path to block embeddings")
    parser.add_argument("--vocab", type=str, required=True, help="Path to tok2block.json")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to validation H5 files")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for schematics")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of samples to test")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load vocabulary
    print(f"\nLoading vocabulary from {args.vocab}")
    with open(args.vocab, "r") as f:
        tok2block = {int(k): v for k, v in json.load(f).items()}
    vocab_size = len(tok2block)
    print(f"Vocabulary size: {vocab_size}")

    # Find air tokens
    air_tokens: Set[int] = set()
    for tok, block in tok2block.items():
        if 'air' in block.lower() and 'stair' not in block.lower():
            air_tokens.add(tok)
    print(f"Air tokens: {air_tokens}")

    # Load embeddings
    print(f"\nLoading embeddings from {args.embeddings}")
    embeddings = np.load(args.embeddings).astype(np.float32)
    emb_dim = embeddings.shape[1]
    print(f"Embeddings shape: {embeddings.shape}")

    # Create model
    print(f"\nCreating VQ-VAE v16b...")
    vqvae = VQVAEv16b(
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        hidden_dim=128,
        rfsq_levels=[5, 5, 5, 5],
        num_stages=2,
        pretrained_embeddings=torch.from_numpy(embeddings),
        air_tokens=air_tokens,
    ).to(device)

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    vqvae.load_state_dict(torch.load(args.checkpoint, map_location=device))
    vqvae.eval()

    # Load validation data
    print(f"\nLoading data from {args.data_dir}")
    data_dir = Path(args.data_dir)
    h5_files = sorted(data_dir.glob("*.h5"))[:args.num_samples]
    print(f"Found {len(h5_files)} files")

    structures = []
    for h5_path in h5_files:
        with h5py.File(h5_path, 'r') as f:
            key = list(f.keys())[0]
            structure = f[key][:].astype(np.int64)
            structures.append(torch.from_numpy(structure).long())

    # Test reconstruction
    print(f"\n{'='*70}")
    print("TESTING RECONSTRUCTION PATHS")
    print(f"{'='*70}")

    summary, detailed = test_reconstruction(vqvae, structures, device, air_tokens)

    print(f"\nPath 1 (Normal: encode -> z_q -> decode):")
    print(f"  Building Accuracy: {summary['normal']['building_acc']*100:.1f}%")
    print(f"  Recall: {summary['normal']['recall']*100:.1f}%")
    print(f"  False Air Rate: {summary['normal']['far']*100:.1f}%")
    print(f"  Volume Ratio: {summary['normal']['vol_ratio']:.3f}x")

    print(f"\nPath 2 (Index: encode -> indices -> indices_to_codes -> decode):")
    print(f"  Building Accuracy: {summary['index']['building_acc']*100:.1f}%")
    print(f"  Recall: {summary['index']['recall']*100:.1f}%")
    print(f"  False Air Rate: {summary['index']['far']*100:.1f}%")
    print(f"  Volume Ratio: {summary['index']['vol_ratio']:.3f}x")

    # Diagnosis
    print(f"\n{'='*70}")
    print("DIAGNOSIS")
    print(f"{'='*70}")

    normal_acc = summary['normal']['building_acc']
    index_acc = summary['index']['building_acc']

    if normal_acc > 0.6 and index_acc < 0.3:
        print("CONFIRMED: Generation pipeline is broken!")
        print("  - Normal reconstruction works (~70% accuracy)")
        print("  - Index-based reconstruction fails (<30% accuracy)")
        print("  - The indices_to_codes function loses LayerNorm statistics")
        print("\nRECOMMENDATION: Fix the RFSQ inverse path before increasing codebook size")
    elif normal_acc > 0.6 and index_acc > 0.5:
        print("Index path works reasonably well.")
        print("The generation issue may be in the prior, not the VQ-VAE.")
    else:
        print("Both paths have issues. Further investigation needed.")

    # Export schematics
    print(f"\n{'='*70}")
    print("EXPORTING SCHEMATICS")
    print(f"{'='*70}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export first 5 samples
    num_export = min(5, len(structures))
    with torch.no_grad():
        for i in range(num_export):
            structure = structures[i]
            block_ids = structure.unsqueeze(0).to(device)

            # Get reconstructions
            z_e, z_q, indices = vqvae(block_ids)
            binary_logits, block_logits = vqvae.decode(z_q)
            pred_normal = vqvae.predict_blocks(binary_logits, block_logits).squeeze(0)

            z_q_from_indices = vqvae.quantizer.indices_to_codes_naive(indices)
            binary_logits2, block_logits2 = vqvae.decode(z_q_from_indices)
            pred_index = vqvae.predict_blocks(binary_logits2, block_logits2).squeeze(0)

            # Export
            export_to_schematic(
                structure.numpy(), tok2block,
                output_dir / f"sample_{i+1}_original.schem"
            )
            export_to_schematic(
                pred_normal.cpu().numpy(), tok2block,
                output_dir / f"sample_{i+1}_v16b_normal.schem"
            )
            export_to_schematic(
                pred_index.cpu().numpy(), tok2block,
                output_dir / f"sample_{i+1}_v16b_index.schem"
            )

    print(f"\nExported {num_export} samples to {output_dir}")
    print("\nTo compare in Minecraft:")
    print("  1. Copy .schem files to server/plugins/FastAsyncWorldEdit/schematics/")
    print("  2. //schem load sample_1_original")
    print("  3. //paste")
    print("  4. Compare with sample_1_v16b_normal and sample_1_v16b_index")


if __name__ == "__main__":
    main()
