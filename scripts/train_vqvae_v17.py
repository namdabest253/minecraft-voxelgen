"""
VQ-VAE v17 Training Script with Adversarial and Structural Losses.

This script trains a VQ-VAE with:
1. Reconstruction losses (frequency-weighted CE, volume penalty, false air penalty)
2. Structural coherence losses (connectivity, patch consistency, smoothness, support)
3. Adversarial loss from 3D PatchGAN discriminator

The goal is to train a model that not only reconstructs well, but generates
structures that "look like real Minecraft builds" without requiring labels.

Usage:
    python scripts/train_vqvae_v17.py --config configs/vqvae_v17_config.yaml
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.discriminator import (
    StructureDiscriminator3D,
    hinge_loss_discriminator,
    hinge_loss_generator,
)
from src.models.structural_losses import (
    ConnectivityLoss,
    PatchConsistencyLoss,
    SurfaceSmoothnessLoss,
    SupportLoss,
)


# ============================================================================
# Model Architecture (same as v16b but with added generation losses)
# ============================================================================

class FSQ(nn.Module):
    """Finite Scalar Quantization."""
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


class InvertibleLayerNorm(nn.Module):
    """LayerNorm that can store/restore statistics for inverse transform."""
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
    """Single stage of Residual FSQ."""
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
    """Residual FSQ with multiple stages."""
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
    """3D Residual block."""
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


class EncoderV17(nn.Module):
    """Encoder: 32x32x32 -> 8x8x8 latent."""
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


class DualHeadDecoderV17(nn.Module):
    """Dual-head decoder: binary (air/structure) + block type."""
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

        # Binary head: air vs structure
        self.binary_head = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.BatchNorm3d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dim // 2, 2, 3, padding=1)
        )

        # Block head: predicts block type
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


class VQVAEv17(nn.Module):
    """VQ-VAE v17 with support for adversarial training."""

    def __init__(
        self,
        vocab_size: int = 3717,
        emb_dim: int = 40,
        hidden_dim: int = 128,
        rfsq_levels: List[int] = None,
        num_stages: int = 2,
        pretrained_embeddings: torch.Tensor = None,
        air_tokens: Set[int] = None,
    ):
        super().__init__()
        if rfsq_levels is None:
            rfsq_levels = [5, 5, 5, 5]
        self.rfsq_dim = len(rfsq_levels)
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.air_tokens = air_tokens or {102, 576, 3352}

        # Block embeddings
        self.block_emb = nn.Embedding(vocab_size, emb_dim)
        if pretrained_embeddings is not None:
            self.block_emb.weight.data.copy_(pretrained_embeddings)
            self.block_emb.weight.requires_grad = False

        # Store embeddings as buffer for discriminator
        if pretrained_embeddings is not None:
            self.register_buffer('block_embeddings', pretrained_embeddings.clone())
        else:
            self.register_buffer('block_embeddings', self.block_emb.weight.data.clone())

        # Encoder, Quantizer, Decoder
        self.encoder = EncoderV17(emb_dim, hidden_dim, self.rfsq_dim)
        self.quantizer = RFSQ(rfsq_levels, num_stages)
        self.decoder = DualHeadDecoderV17(self.rfsq_dim, hidden_dim, vocab_size)

    def forward(self, block_ids: torch.Tensor):
        """Forward pass."""
        x = self.block_emb(block_ids).permute(0, 4, 1, 2, 3)
        z_e = self.encoder(x).permute(0, 2, 3, 4, 1)  # [B, 8, 8, 8, 4]
        z_q, indices = self.quantizer(z_e)
        return z_e, z_q, indices

    def decode(self, z_q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode z_q to binary and block logits."""
        z_q_permuted = z_q.permute(0, 4, 1, 2, 3)  # [B, 4, 8, 8, 8]
        return self.decoder(z_q_permuted)

    def predict_blocks(
        self,
        binary_logits: torch.Tensor,
        block_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Combine binary and block predictions."""
        binary_pred = binary_logits.argmax(dim=1)
        block_pred = block_logits.argmax(dim=1)
        air_token = 102
        return torch.where(binary_pred == 0, air_token, block_pred)

    def get_embeddings(self, block_ids: torch.Tensor) -> torch.Tensor:
        """Get block embeddings for discriminator input."""
        emb = self.block_embeddings[block_ids]  # [B, X, Y, Z, C]
        return emb.permute(0, 4, 1, 2, 3)  # [B, C, X, Y, Z]


# ============================================================================
# Dataset
# ============================================================================

class StructureDataset(Dataset):
    """Dataset for loading 32x32x32 structures."""

    def __init__(self, data_dir: str, augment: bool = False):
        self.data_dir = Path(data_dir)
        self.h5_files = sorted(self.data_dir.glob("*.h5"))
        self.augment = augment

        if not self.h5_files:
            raise ValueError(f"No H5 files found in {data_dir}")

        print(f"Found {len(self.h5_files)} files in {data_dir}")

    def __len__(self):
        return len(self.h5_files)

    def __getitem__(self, idx):
        with h5py.File(self.h5_files[idx], 'r') as f:
            key = list(f.keys())[0]
            structure = f[key][:].astype(np.int64)

        if self.augment:
            # Random rotation around Y
            k = np.random.randint(0, 4)
            if k > 0:
                structure = np.rot90(structure, k=k, axes=(0, 2))
            # Random flips
            if np.random.random() > 0.5:
                structure = np.flip(structure, axis=2)
            if np.random.random() > 0.5:
                structure = np.flip(structure, axis=0)
            structure = np.ascontiguousarray(structure)

        return torch.from_numpy(structure).long()


# ============================================================================
# Loss Functions
# ============================================================================

def compute_reconstruction_loss(
    binary_logits: torch.Tensor,
    block_logits: torch.Tensor,
    target: torch.Tensor,
    freq_weights: torch.Tensor,
    air_tokens: Set[int],
    struct_to_air_weight: float = 5.0,
    volume_weight: float = 5.0,
) -> Dict[str, torch.Tensor]:
    """Compute reconstruction losses."""
    device = target.device
    B = target.shape[0]

    # Air mask
    air_tokens_tensor = torch.tensor(list(air_tokens), device=device, dtype=target.dtype)
    gt_is_air = torch.isin(target, air_tokens_tensor)
    gt_is_structure = ~gt_is_air

    # Binary CE loss with asymmetric weighting
    binary_target = gt_is_structure.long()  # 0=air, 1=structure
    binary_pred = binary_logits.argmax(dim=1)

    # Weight: penalize struct->air more
    binary_weight = torch.ones_like(binary_target, dtype=torch.float)
    struct_to_air = gt_is_structure & (binary_pred == 0)
    binary_weight[struct_to_air] = struct_to_air_weight

    binary_logits_flat = binary_logits.permute(0, 2, 3, 4, 1).reshape(-1, 2)
    binary_target_flat = binary_target.reshape(-1)
    binary_weight_flat = binary_weight.reshape(-1)

    binary_ce = F.cross_entropy(binary_logits_flat, binary_target_flat, reduction='none')
    binary_loss = (binary_ce * binary_weight_flat).mean()

    # Block CE loss (only at structure locations)
    block_logits_flat = block_logits.permute(0, 2, 3, 4, 1).reshape(-1, block_logits.shape[1])
    target_flat = target.reshape(-1)
    gt_is_structure_flat = gt_is_structure.reshape(-1)

    if gt_is_structure_flat.any():
        block_ce = F.cross_entropy(
            block_logits_flat[gt_is_structure_flat],
            target_flat[gt_is_structure_flat],
            weight=freq_weights,
            reduction='mean'
        )
    else:
        block_ce = torch.tensor(0.0, device=device)

    # Volume penalty (STE for hard predictions)
    with torch.no_grad():
        pred_structure = binary_pred == 1
        gt_volume = gt_is_structure.float().sum()
        pred_volume = pred_structure.float().sum()
        volume_ratio = pred_volume / (gt_volume + 1e-6)

    # Soft volume loss through binary logits
    binary_probs = F.softmax(binary_logits_flat, dim=1)[:, 1]  # P(structure)
    soft_pred_volume = binary_probs.sum()
    soft_gt_volume = gt_is_structure_flat.float().sum()
    volume_loss = F.l1_loss(soft_pred_volume / (B * 32**3), soft_gt_volume / (B * 32**3))

    # Metrics
    with torch.no_grad():
        final_pred = torch.where(binary_pred == 0, 102, block_logits.argmax(dim=1))
        pred_is_air = torch.isin(final_pred, air_tokens_tensor)

        correct = (final_pred == target)
        building_acc = correct[gt_is_structure].float().mean() if gt_is_structure.any() else torch.tensor(0.0)
        recall = (~pred_is_air)[gt_is_structure].float().mean() if gt_is_structure.any() else torch.tensor(1.0)
        far = pred_is_air[gt_is_structure].float().mean() if gt_is_structure.any() else torch.tensor(0.0)

    return {
        'binary_loss': binary_loss,
        'block_loss': block_ce,
        'volume_loss': volume_loss,
        'volume_ratio': volume_ratio,
        'building_acc': building_acc,
        'recall': recall,
        'far': far,
    }


# ============================================================================
# Training Loop
# ============================================================================

def train_epoch(
    vqvae: VQVAEv17,
    discriminator: StructureDiscriminator3D,
    dataloader: DataLoader,
    g_optimizer: torch.optim.Optimizer,
    d_optimizer: torch.optim.Optimizer,
    freq_weights: torch.Tensor,
    connectivity_loss_fn: ConnectivityLoss,
    patch_loss_fn: PatchConsistencyLoss,
    support_loss_fn: SupportLoss,
    device: torch.device,
    epoch: int,
    config: Dict,
) -> Dict[str, float]:
    """Train for one epoch."""

    vqvae.train()
    discriminator.train()

    # Loss weights (with schedule)
    lambda_adv = min(config['lambda_adv_max'], config['lambda_adv_start'] + epoch * config['lambda_adv_schedule'])

    metrics = {
        'g_loss': 0, 'd_loss': 0, 'binary_loss': 0, 'block_loss': 0,
        'adv_loss': 0, 'conn_loss': 0, 'volume_ratio': 0, 'building_acc': 0,
        'recall': 0, 'far': 0,
    }
    num_batches = 0

    for batch_idx, structures in enumerate(dataloader):
        structures = structures.to(device)
        B = structures.shape[0]

        # ==================== Discriminator Step ====================
        with torch.no_grad():
            z_e, z_q, indices = vqvae(structures)
            binary_logits, block_logits = vqvae.decode(z_q)
            reconstructed = vqvae.predict_blocks(binary_logits, block_logits)

        # Get embeddings
        real_emb = vqvae.get_embeddings(structures)
        recon_emb = vqvae.get_embeddings(reconstructed)

        # Discriminator forward
        d_real = discriminator(real_emb)
        d_fake = discriminator(recon_emb.detach())

        # Discriminator loss
        d_loss = hinge_loss_discriminator(d_real, d_fake)

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # ==================== Generator Step ====================
        z_e, z_q, indices = vqvae(structures)
        binary_logits, block_logits = vqvae.decode(z_q)
        reconstructed = vqvae.predict_blocks(binary_logits, block_logits)

        # Reconstruction losses
        recon_losses = compute_reconstruction_loss(
            binary_logits, block_logits, structures, freq_weights,
            vqvae.air_tokens, config['struct_to_air_weight'], config['volume_weight']
        )

        # Structural coherence losses
        L_conn = connectivity_loss_fn(reconstructed)
        L_support = support_loss_fn(reconstructed)

        # Adversarial loss
        recon_emb = vqvae.get_embeddings(reconstructed)
        d_fake = discriminator(recon_emb)
        L_adv = hinge_loss_generator(d_fake)

        # Combined generator loss
        g_loss = (
            recon_losses['binary_loss'] +
            recon_losses['block_loss'] +
            config['volume_weight'] * recon_losses['volume_loss'] +
            config['lambda_conn'] * L_conn +
            config['lambda_support'] * L_support +
            lambda_adv * L_adv
        )

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        # Update metrics
        metrics['g_loss'] += g_loss.item()
        metrics['d_loss'] += d_loss.item()
        metrics['binary_loss'] += recon_losses['binary_loss'].item()
        metrics['block_loss'] += recon_losses['block_loss'].item()
        metrics['adv_loss'] += L_adv.item()
        metrics['conn_loss'] += L_conn.item()
        metrics['volume_ratio'] += recon_losses['volume_ratio'].item()
        metrics['building_acc'] += recon_losses['building_acc'].item()
        metrics['recall'] += recon_losses['recall'].item()
        metrics['far'] += recon_losses['far'].item()
        num_batches += 1

        # Progress
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx+1}/{len(dataloader)} | "
                  f"G: {g_loss.item():.4f} | D: {d_loss.item():.4f} | "
                  f"Acc: {recon_losses['building_acc'].item()*100:.1f}%")

    # Average metrics
    for key in metrics:
        metrics[key] /= num_batches

    return metrics


def validate(
    vqvae: VQVAEv17,
    dataloader: DataLoader,
    freq_weights: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    """Validate model."""
    vqvae.eval()

    metrics = {'building_acc': 0, 'recall': 0, 'far': 0, 'volume_ratio': 0}
    num_batches = 0

    with torch.no_grad():
        for structures in dataloader:
            structures = structures.to(device)

            z_e, z_q, indices = vqvae(structures)
            binary_logits, block_logits = vqvae.decode(z_q)

            recon_losses = compute_reconstruction_loss(
                binary_logits, block_logits, structures, freq_weights,
                vqvae.air_tokens, struct_to_air_weight=1.0, volume_weight=1.0
            )

            metrics['building_acc'] += recon_losses['building_acc'].item()
            metrics['recall'] += recon_losses['recall'].item()
            metrics['far'] += recon_losses['far'].item()
            metrics['volume_ratio'] += recon_losses['volume_ratio'].item()
            num_batches += 1

    for key in metrics:
        metrics[key] /= num_batches

    return metrics


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train VQ-VAE v17 with adversarial losses")
    parser.add_argument("--train-dir", type=str, required=True, help="Training data directory")
    parser.add_argument("--val-dir", type=str, required=True, help="Validation data directory")
    parser.add_argument("--embeddings", type=str, required=True, help="Block embeddings path")
    parser.add_argument("--vocab", type=str, required=True, help="tok2block.json path")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Config
    config = {
        'struct_to_air_weight': 5.0,
        'volume_weight': 5.0,
        'lambda_conn': 0.5,
        'lambda_support': 0.2,
        'lambda_adv_start': 0.01,
        'lambda_adv_max': 0.1,
        'lambda_adv_schedule': 0.01,  # Increase per epoch
    }

    # Load vocabulary
    print(f"\nLoading vocabulary from {args.vocab}")
    with open(args.vocab, 'r') as f:
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
    print(f"Loading embeddings from {args.embeddings}")
    embeddings = np.load(args.embeddings).astype(np.float32)
    emb_dim = embeddings.shape[1]
    print(f"Embeddings shape: {embeddings.shape}")

    # Compute frequency weights
    print(f"\nLoading training data from {args.train_dir}")
    train_dataset = StructureDataset(args.train_dir, augment=True)
    val_dataset = StructureDataset(args.val_dir, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Compute frequency weights from first N samples
    print("Computing frequency weights...")
    block_counts = torch.zeros(vocab_size)
    for i, structure in enumerate(train_dataset):
        counts = torch.bincount(structure.flatten(), minlength=vocab_size)
        block_counts += counts
        if i >= 500:  # Sample 500 structures
            break
    block_counts = block_counts.clamp(min=1)
    freq_weights = (block_counts.sum() / block_counts).sqrt().clamp(max=5.0).to(device)

    # Create models
    print("\nCreating models...")
    vqvae = VQVAEv17(
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        hidden_dim=128,
        rfsq_levels=[5, 5, 5, 5],
        num_stages=2,
        pretrained_embeddings=torch.from_numpy(embeddings),
        air_tokens=air_tokens,
    ).to(device)

    discriminator = StructureDiscriminator3D(
        in_channels=emb_dim,
        base_channels=64,
        use_spectral_norm=True,
    ).to(device)

    # Structural loss functions
    connectivity_loss_fn = ConnectivityLoss(air_tokens=air_tokens).to(device)
    patch_loss_fn = PatchConsistencyLoss(air_tokens=air_tokens).to(device)
    support_loss_fn = SupportLoss(air_tokens=air_tokens).to(device)

    # Optimizers
    g_params = list(vqvae.encoder.parameters()) + list(vqvae.decoder.parameters())
    g_optimizer = torch.optim.Adam(g_params, lr=args.lr, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print(f"\n{'='*70}")
    print("Starting training")
    print(f"{'='*70}")

    best_building_acc = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        start_time = time.time()

        train_metrics = train_epoch(
            vqvae, discriminator, train_loader,
            g_optimizer, d_optimizer, freq_weights,
            connectivity_loss_fn, patch_loss_fn, support_loss_fn,
            device, epoch, config
        )

        val_metrics = validate(vqvae, val_loader, freq_weights, device)

        epoch_time = time.time() - start_time

        print(f"\nEpoch {epoch+1} Summary ({epoch_time:.1f}s):")
        print(f"  Train | G: {train_metrics['g_loss']:.4f} | D: {train_metrics['d_loss']:.4f}")
        print(f"  Train | Acc: {train_metrics['building_acc']*100:.1f}% | "
              f"Recall: {train_metrics['recall']*100:.1f}% | "
              f"Vol: {train_metrics['volume_ratio']:.3f}x")
        print(f"  Val   | Acc: {val_metrics['building_acc']*100:.1f}% | "
              f"Recall: {val_metrics['recall']*100:.1f}% | "
              f"Vol: {val_metrics['volume_ratio']:.3f}x")

        # Save best model
        if val_metrics['building_acc'] > best_building_acc:
            best_building_acc = val_metrics['building_acc']
            torch.save(vqvae.state_dict(), output_dir / "vqvae_v17_best.pt")
            torch.save(discriminator.state_dict(), output_dir / "discriminator_v17_best.pt")
            print(f"  Saved best model (acc: {best_building_acc*100:.1f}%)")

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'vqvae': vqvae.state_dict(),
                'discriminator': discriminator.state_dict(),
                'g_optimizer': g_optimizer.state_dict(),
                'd_optimizer': d_optimizer.state_dict(),
            }, output_dir / f"checkpoint_epoch_{epoch+1}.pt")

    print(f"\n{'='*70}")
    print(f"Training complete! Best accuracy: {best_building_acc*100:.1f}%")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
