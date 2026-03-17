"""
Quick verification that fixed LayerNorm statistics fix the generation pipeline.

Tests three paths:
1. Normal: encode -> z_q -> decode (baseline, ~66%)
2. Naive index: encode -> indices -> naive codes -> decode (broken, ~2%)
3. Fixed index: encode -> indices -> fixed codes -> decode (should be ~66%)
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

sys.path.insert(0, str(Path(__file__).parent.parent))


# Minimal model definitions
class FSQ(nn.Module):
    def __init__(self, levels: List[int]):
        super().__init__()
        self.levels = levels
        self.dim = len(levels)
        self.codebook_size = int(np.prod(levels))
        self.register_buffer('_levels', torch.tensor(levels, dtype=torch.float32))
        basis = [1]
        for L in reversed(levels[1:]):
            basis.append(basis[-1] * L)
        self.register_buffer('_basis', torch.tensor(list(reversed(basis)), dtype=torch.long))
        self.register_buffer('_half_levels', torch.tensor([(L-1)/2 for L in levels], dtype=torch.float32))

    def forward(self, z):
        z_bounded = torch.tanh(z)
        z_q = self._quantize(z_bounded)
        z_q = z_bounded + (z_q - z_bounded).detach()
        indices = self._to_indices(z_q)
        return z_q, indices

    def _quantize(self, z):
        z_q_list = []
        for i in range(self.dim):
            half_L = self._half_levels[i]
            z_i = z[..., i] * half_L
            z_i = torch.round(z_i).clamp(-half_L, half_L) / half_L
            z_q_list.append(z_i)
        return torch.stack(z_q_list, dim=-1)

    def _to_indices(self, z_q):
        indices = torch.zeros(z_q.shape[:-1], dtype=torch.long, device=z_q.device)
        for i in range(self.dim):
            L = self._levels[i].long()
            half_L = self._half_levels[i]
            level_idx = ((z_q[..., i] * half_L) + half_L).round().long().clamp(0, L-1)
            indices = indices + level_idx * self._basis[i]
        return indices

    def indices_to_codes(self, indices):
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

    def forward(self, x):
        self.stored_mean = x.mean(dim=(1, 2, 3), keepdim=True)
        self.stored_std = x.std(dim=(1, 2, 3), keepdim=True) + self.eps
        x_norm = (x - self.stored_mean) / self.stored_std
        return x_norm * self.weight + self.bias

    def inverse(self, x_norm):
        x = (x_norm - self.bias) / self.weight
        return x * self.stored_std + self.stored_mean

    def inverse_fixed(self, x_norm):
        x = (x_norm - self.bias) / self.weight
        return x * self.fixed_std + self.fixed_mean


class RFSQStage(nn.Module):
    def __init__(self, levels):
        super().__init__()
        self.fsq = FSQ(levels)
        self.layernorm = InvertibleLayerNorm(len(levels))

    def forward(self, residual):
        z_norm = self.layernorm(residual)
        z_q_norm, indices = self.fsq(z_norm)
        z_q = self.layernorm.inverse(z_q_norm)
        return z_q, residual - z_q, indices


class RFSQ(nn.Module):
    def __init__(self, levels, num_stages=2):
        super().__init__()
        self.stages = nn.ModuleList([RFSQStage(levels) for _ in range(num_stages)])

    def forward(self, z):
        residual = z
        z_q_sum = torch.zeros_like(z)
        all_indices = []
        for stage in self.stages:
            z_q, residual, indices = stage(residual)
            z_q_sum = z_q_sum + z_q
            all_indices.append(indices)
        return z_q_sum, all_indices

    def indices_to_codes_naive(self, all_indices):
        z_q_sum = None
        for i, indices in enumerate(all_indices):
            codes = self.stages[i].fsq.indices_to_codes(indices)
            z_q_sum = codes if z_q_sum is None else z_q_sum + codes
        return z_q_sum

    def indices_to_codes_fixed(self, all_indices):
        z_q_sum = None
        for i, indices in enumerate(all_indices):
            codes = self.stages[i].fsq.indices_to_codes(indices)
            z_q = self.stages[i].layernorm.inverse_fixed(codes)
            z_q_sum = z_q if z_q_sum is None else z_q_sum + z_q
        return z_q_sum

    def set_fixed_stats(self, stats):
        for i, stage in enumerate(self.stages):
            stage.layernorm.fixed_mean.copy_(stats[f'stage{i}']['mean'])
            stage.layernorm.fixed_std.copy_(stats[f'stage{i}']['std'])


class ResidualBlock3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x):
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + x)


class EncoderV16b(nn.Module):
    def __init__(self, in_channels=40, hidden_dim=128, rfsq_dim=4):
        super().__init__()
        self.initial = nn.Sequential(nn.Conv3d(in_channels, hidden_dim, 3, padding=1), nn.BatchNorm3d(hidden_dim), nn.ReLU())
        self.down1 = nn.Sequential(nn.Conv3d(hidden_dim, hidden_dim, 4, stride=2, padding=1), nn.BatchNorm3d(hidden_dim), nn.ReLU())
        self.res1 = ResidualBlock3D(hidden_dim)
        self.down2 = nn.Sequential(nn.Conv3d(hidden_dim, hidden_dim, 4, stride=2, padding=1), nn.BatchNorm3d(hidden_dim), nn.ReLU())
        self.res2 = nn.Sequential(*[ResidualBlock3D(hidden_dim) for _ in range(4)])
        self.proj = nn.Conv3d(hidden_dim, rfsq_dim, 3, padding=1)

    def forward(self, x):
        return self.proj(self.res2(self.down2(self.res1(self.down1(self.initial(x))))))


class DualHeadDecoderV16b(nn.Module):
    def __init__(self, rfsq_dim=4, hidden_dim=128, num_blocks=3717):
        super().__init__()
        self.initial = nn.Sequential(nn.Conv3d(rfsq_dim, hidden_dim, 3, padding=1), nn.BatchNorm3d(hidden_dim), nn.ReLU())
        self.res1 = nn.Sequential(*[ResidualBlock3D(hidden_dim) for _ in range(4)])
        self.up1 = nn.Sequential(nn.ConvTranspose3d(hidden_dim, hidden_dim, 4, stride=2, padding=1), nn.BatchNorm3d(hidden_dim), nn.ReLU())
        self.res2 = ResidualBlock3D(hidden_dim)
        self.up2 = nn.Sequential(nn.ConvTranspose3d(hidden_dim, hidden_dim, 4, stride=2, padding=1), nn.BatchNorm3d(hidden_dim), nn.ReLU())
        self.trunk_final = ResidualBlock3D(hidden_dim)
        self.binary_head = nn.Sequential(nn.Conv3d(hidden_dim, hidden_dim//2, 3, padding=1), nn.BatchNorm3d(hidden_dim//2), nn.ReLU(), nn.Conv3d(hidden_dim//2, 2, 3, padding=1))
        self.block_head = nn.Sequential(nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1), nn.BatchNorm3d(hidden_dim), nn.ReLU(), nn.Conv3d(hidden_dim, num_blocks, 3, padding=1))

    def forward(self, z_q):
        x = self.trunk_final(self.up2(self.res2(self.up1(self.res1(self.initial(z_q))))))
        return self.binary_head(x), self.block_head(x)


class VQVAEv16b(nn.Module):
    def __init__(self, vocab_size=3717, emb_dim=40, hidden_dim=128, rfsq_levels=None, num_stages=2, pretrained_embeddings=None, air_tokens=None):
        super().__init__()
        rfsq_levels = rfsq_levels or [5, 5, 5, 5]
        self.block_emb = nn.Embedding(vocab_size, emb_dim)
        if pretrained_embeddings is not None:
            self.block_emb.weight.data.copy_(pretrained_embeddings)
        self.encoder = EncoderV16b(emb_dim, hidden_dim, len(rfsq_levels))
        self.quantizer = RFSQ(rfsq_levels, num_stages)
        self.decoder = DualHeadDecoderV16b(len(rfsq_levels), hidden_dim, vocab_size)
        self.air_tokens = air_tokens or {102, 576, 3352}

    def forward(self, block_ids):
        x = self.block_emb(block_ids).permute(0, 4, 1, 2, 3)
        z_e = self.encoder(x).permute(0, 2, 3, 4, 1)
        z_q, indices = self.quantizer(z_e)
        return z_e, z_q, indices

    def decode(self, z_q):
        z_q_perm = z_q.permute(0, 4, 1, 2, 3)
        return self.decoder(z_q_perm)

    def predict(self, binary_logits, block_logits):
        binary_pred = binary_logits.argmax(dim=1)
        block_pred = block_logits.argmax(dim=1)
        return torch.where(binary_pred == 0, 102, block_pred)


def compute_building_acc(pred, target, air_tokens):
    air_tensor = torch.tensor(list(air_tokens), device=pred.device)
    gt_air = torch.isin(target, air_tensor)
    gt_building = ~gt_air
    if gt_building.sum() == 0:
        return 0.0
    return (pred[gt_building] == target[gt_building]).float().mean().item()


def main():
    device = torch.device("cpu")
    print("Loading model...")

    # Load vocab and embeddings
    with open("data/vocabulary/tok2block.json") as f:
        tok2block = {int(k): v for k, v in json.load(f).items()}
    air_tokens = {tok for tok, block in tok2block.items() if 'air' in block.lower() and 'stair' not in block.lower()}
    embeddings = torch.from_numpy(np.load("data/output/block2vec/v3/block_embeddings_v3.npy").astype(np.float32))

    # Create and load model
    vqvae = VQVAEv16b(vocab_size=len(tok2block), emb_dim=40, pretrained_embeddings=embeddings, air_tokens=air_tokens).to(device)
    vqvae.load_state_dict(torch.load("data/output/vqvae/v16b/vqvae_v16b_best.pt", map_location=device), strict=False)
    vqvae.eval()

    # Load fixed statistics
    print("Loading fixed statistics...")
    stats = torch.load("data/output/vqvae/v16b/layernorm_stats.pt", map_location=device)
    vqvae.quantizer.set_fixed_stats(stats)

    # Load test samples
    print("Loading test samples...")
    h5_files = sorted(Path("data/splits/val").glob("*.h5"))[:20]
    structures = []
    for h5_path in h5_files:
        with h5py.File(h5_path, 'r') as f:
            structures.append(torch.from_numpy(f[list(f.keys())[0]][:].astype(np.int64)).long())

    # Test all three paths
    results = {'normal': [], 'naive': [], 'fixed': []}

    print("\nTesting reconstruction paths...")
    with torch.no_grad():
        for i, structure in enumerate(structures):
            block_ids = structure.unsqueeze(0).to(device)
            z_e, z_q, indices = vqvae(block_ids)

            # Path 1: Normal
            b_logits, bl_logits = vqvae.decode(z_q)
            pred_normal = vqvae.predict(b_logits, bl_logits).squeeze(0)

            # Path 2: Naive index
            z_q_naive = vqvae.quantizer.indices_to_codes_naive(indices)
            b_logits2, bl_logits2 = vqvae.decode(z_q_naive)
            pred_naive = vqvae.predict(b_logits2, bl_logits2).squeeze(0)

            # Path 3: Fixed index
            z_q_fixed = vqvae.quantizer.indices_to_codes_fixed(indices)
            b_logits3, bl_logits3 = vqvae.decode(z_q_fixed)
            pred_fixed = vqvae.predict(b_logits3, bl_logits3).squeeze(0)

            results['normal'].append(compute_building_acc(pred_normal, structure.to(device), air_tokens))
            results['naive'].append(compute_building_acc(pred_naive, structure.to(device), air_tokens))
            results['fixed'].append(compute_building_acc(pred_fixed, structure.to(device), air_tokens))

    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Normal path (z_q):       {np.mean(results['normal'])*100:.1f}% building accuracy")
    print(f"Naive index path:        {np.mean(results['naive'])*100:.1f}% building accuracy")
    print(f"Fixed index path:        {np.mean(results['fixed'])*100:.1f}% building accuracy")
    print("="*70)

    improvement = np.mean(results['fixed']) - np.mean(results['naive'])
    if improvement > 0.3:
        print(f"\nSUCCESS! Fixed path improved by {improvement*100:.1f} percentage points!")
        print("The generation pipeline fix is working.")
    else:
        print(f"\nFix had limited effect (+{improvement*100:.1f}pp). May need different approach.")


if __name__ == "__main__":
    main()
