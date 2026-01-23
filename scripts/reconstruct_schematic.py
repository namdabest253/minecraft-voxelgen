"""
Reconstruct Minecraft schematics using VQ-VAE v7 (FSQ).

Takes .schem files as input, runs through the model, outputs reconstructed .schem files.

Coordinate systems:
- Minecraft/Schematic: X (east/west), Y (up/down), Z (north/south)
- Sponge schematic BlockData: stored in YZX order (Y slowest, X fastest)
- Model (H5 training data): shape (X, Y, Z) = (32, 32, 32)

Usage:
    python scripts/reconstruct_schematic.py input.schem
    python scripts/reconstruct_schematic.py --all  # Process all in schematics folder
"""

import argparse
import gzip
import json
import struct
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "data/output/vqvae/v7/vqvae_v7_best.pt"
VOCAB_PATH = PROJECT_ROOT / "data/vocabulary/tok2block.json"
EMBEDDINGS_PATH = PROJECT_ROOT / "data/output/block2vec/v3/block_embeddings_v3.npy"
# Use relative path from project root to work in both Windows and WSL
FAWE_SCHEMATICS = PROJECT_ROOT.parent / "server" / "plugins" / "FastAsyncWorldEdit" / "schematics"

# Model expects 32x32x32 structures
STRUCTURE_SIZE = 32

# Air tokens in our vocabulary
AIR_TOKEN = 102  # minecraft:air

# v7 Configuration (RFSQ)
RFSQ_LEVELS_PER_STAGE = [5, 5, 5, 5]
NUM_STAGES = 2
HIDDEN_DIMS = [96, 192]

# v5 Configuration (FSQ) - for backward compatibility
FSQ_LEVELS = [5, 5, 5, 5, 5, 5, 5, 5]


# ============================================================
# Model Definition (must match training)
# ============================================================

class ResidualBlock3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(channels)
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


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
        self.register_buffer('_usage', torch.zeros(self.codebook_size))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_bounded = torch.tanh(z)
        z_q_list = []
        for i in range(self.dim):
            L = self._levels[i]
            half_L = self._half_levels[i]
            z_i = z_bounded[..., i]
            z_i = z_i * half_L
            z_i = torch.round(z_i)
            z_i = torch.clamp(z_i, -half_L, half_L)
            z_i = z_i / half_L
            z_q_list.append(z_i)
        z_q = torch.stack(z_q_list, dim=-1)
        z_q = z_bounded + (z_q - z_bounded).detach()
        indices = torch.zeros(z_q.shape[:-1], dtype=torch.long, device=z_q.device)
        for i in range(self.dim):
            L = self._levels[i].long()
            half_L = self._half_levels[i]
            z_i = z_q[..., i]
            level_idx = ((z_i * half_L) + half_L).round().long()
            level_idx = torch.clamp(level_idx, 0, L - 1)
            indices = indices + level_idx * self._basis[i]
        return z_q, indices


class EncoderV5(nn.Module):
    def __init__(self, in_channels: int, hidden_dims: List[int], fsq_dim: int, dropout: float = 0.1):
        super().__init__()
        layers = []
        current = in_channels
        for h in hidden_dims:
            layers.extend([
                nn.Conv3d(current, h, 4, stride=2, padding=1),
                nn.BatchNorm3d(h),
                nn.ReLU(inplace=True),
                nn.Dropout3d(dropout),
                ResidualBlock3D(h),
            ])
            current = h
        layers.extend([
            ResidualBlock3D(current),
            ResidualBlock3D(current),
            nn.Conv3d(current, fsq_dim, 3, padding=1),
        ])
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class DecoderV5(nn.Module):
    def __init__(self, fsq_dim: int, hidden_dims: List[int], num_blocks: int, dropout: float = 0.1):
        super().__init__()
        layers = [
            nn.Conv3d(fsq_dim, hidden_dims[0], 3, padding=1),
            nn.BatchNorm3d(hidden_dims[0]),
            nn.ReLU(inplace=True),
            ResidualBlock3D(hidden_dims[0]),
            ResidualBlock3D(hidden_dims[0]),
        ]
        current = hidden_dims[0]
        for h in hidden_dims[1:]:
            layers.extend([
                ResidualBlock3D(current),
                nn.ConvTranspose3d(current, h, 4, stride=2, padding=1),
                nn.BatchNorm3d(h),
                nn.ReLU(inplace=True),
                nn.Dropout3d(dropout),
            ])
            current = h
        layers.extend([
            ResidualBlock3D(current),
            nn.ConvTranspose3d(current, current, 4, stride=2, padding=1),
            nn.BatchNorm3d(current),
            nn.ReLU(inplace=True),
            nn.Conv3d(current, num_blocks, 3, padding=1),
        ])
        self.decoder = nn.Sequential(*layers)

    def forward(self, z_q):
        return self.decoder(z_q)


class VQVAEv5(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dims: List[int],
                 fsq_levels: List[int], pretrained_emb: np.ndarray, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.fsq_dim = len(fsq_levels)
        self.block_emb = nn.Embedding(vocab_size, emb_dim)
        self.block_emb.weight.data.copy_(torch.from_numpy(pretrained_emb))
        self.block_emb.weight.requires_grad = False
        self.encoder = EncoderV5(emb_dim, hidden_dims, self.fsq_dim, dropout)
        self.fsq = FSQ(fsq_levels)
        self.decoder = DecoderV5(self.fsq_dim, list(reversed(hidden_dims)), vocab_size, dropout)

    def forward(self, block_ids: torch.Tensor) -> torch.Tensor:
        # block_ids: [B, X, Y, Z]
        x = self.block_emb(block_ids)  # [B, X, Y, Z, emb_dim]
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # [B, emb_dim, X, Y, Z]
        z_e = self.encoder(x)
        z_e = z_e.permute(0, 2, 3, 4, 1).contiguous()
        z_q, indices = self.fsq(z_e)
        z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()
        logits = self.decoder(z_q)  # [B, vocab, X, Y, Z]
        return logits


# ============================================================
# VQ-VAE v7 Model (RFSQ + U-Net Skip Connections)
# ============================================================

class InvertibleLayerNorm(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.stored_mean = None
        self.stored_std = None

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

    @property
    def codebook_size(self) -> int:
        return self.fsq.codebook_size

    def forward(self, residual):
        z_norm = self.layernorm(residual)
        z_q_norm, indices = self.fsq(z_norm)
        z_q = self.layernorm.inverse(z_q_norm)
        new_residual = residual - z_q
        return z_q, new_residual, indices


class RFSQ(nn.Module):
    def __init__(self, levels_per_stage: List[int], num_stages: int = 2):
        super().__init__()
        self.num_stages = num_stages
        self.dim = len(levels_per_stage)
        self.stages = nn.ModuleList([RFSQStage(levels_per_stage) for _ in range(num_stages)])
        codes_per_stage = int(np.prod(levels_per_stage))
        self.codebook_size = codes_per_stage ** num_stages
        self.codes_per_stage = codes_per_stage

    def forward(self, z):
        residual = z
        z_q_sum = torch.zeros_like(z)
        all_indices = []
        for stage in self.stages:
            z_q, residual, indices = stage(residual)
            z_q_sum = z_q_sum + z_q
            all_indices.append(indices)
        return z_q_sum, all_indices


class EncoderV7(nn.Module):
    """Encoder with dual skip connection outputs for U-Net architecture."""

    def __init__(self, in_channels: int, hidden_dims: List[int], rfsq_dim: int, dropout: float = 0.1):
        super().__init__()

        # Stage 1: 32 -> 16 (output used for skip_16)
        self.stage1 = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dims[0], 4, stride=2, padding=1),
            nn.BatchNorm3d(hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),
            ResidualBlock3D(hidden_dims[0]),
        )

        # Stage 2: 16 -> 8
        self.stage2 = nn.Sequential(
            nn.Conv3d(hidden_dims[0], hidden_dims[1], 4, stride=2, padding=1),
            nn.BatchNorm3d(hidden_dims[1]),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),
            ResidualBlock3D(hidden_dims[1]),
        )

        # Final processing at 8x8x8
        self.final = nn.Sequential(
            ResidualBlock3D(hidden_dims[1]),
            ResidualBlock3D(hidden_dims[1]),
            nn.Conv3d(hidden_dims[1], rfsq_dim, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        skip_32 = x  # [B, 32, 32, 32, 32]
        skip_16 = self.stage1(x)  # [B, 96, 16, 16, 16]
        x = self.stage2(skip_16)  # [B, 192, 8, 8, 8]
        z_e = self.final(x)  # [B, 4, 8, 8, 8]
        return z_e, skip_16, skip_32


class DecoderV7(nn.Module):
    """Decoder with dual U-Net skip connections."""

    def __init__(self, rfsq_dim: int, hidden_dims: List[int], num_blocks: int,
                 emb_dim: int, dropout: float = 0.1):
        super().__init__()

        # Initial projection from RFSQ dim
        self.initial = nn.Sequential(
            nn.Conv3d(rfsq_dim, hidden_dims[0], 3, padding=1),
            nn.BatchNorm3d(hidden_dims[0]),
            nn.ReLU(inplace=True),
            ResidualBlock3D(hidden_dims[0]),
            ResidualBlock3D(hidden_dims[0]),
        )

        # Upsample 8 -> 16
        self.up1 = nn.Sequential(
            ResidualBlock3D(hidden_dims[0]),
            nn.ConvTranspose3d(hidden_dims[0], hidden_dims[1], 4, stride=2, padding=1),
            nn.BatchNorm3d(hidden_dims[1]),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),
        )

        # Skip connection 1 projection: 96 (up1) + 96 (skip_16) = 192 -> 96
        self.skip1_proj = nn.Sequential(
            nn.Conv3d(hidden_dims[1] * 2, hidden_dims[1], 1),
            nn.BatchNorm3d(hidden_dims[1]),
            nn.ReLU(inplace=True),
        )

        self.post_skip1 = nn.Sequential(
            ResidualBlock3D(hidden_dims[1]),
            ResidualBlock3D(hidden_dims[1]),
        )

        # Upsample 16 -> 32
        self.up2 = nn.Sequential(
            ResidualBlock3D(hidden_dims[1]),
            nn.ConvTranspose3d(hidden_dims[1], hidden_dims[1], 4, stride=2, padding=1),
            nn.BatchNorm3d(hidden_dims[1]),
            nn.ReLU(inplace=True),
        )

        # Skip connection 2 projection: 96 (up2) + 32 (skip_32/emb) = 128 -> 96
        self.skip2_proj = nn.Sequential(
            nn.Conv3d(hidden_dims[1] + emb_dim, hidden_dims[1], 1),
            nn.BatchNorm3d(hidden_dims[1]),
            nn.ReLU(inplace=True),
        )

        self.post_skip2 = ResidualBlock3D(hidden_dims[1])
        self.final = nn.Conv3d(hidden_dims[1], num_blocks, 3, padding=1)

    def forward(self, z_q: torch.Tensor, skip_16: torch.Tensor, skip_32: torch.Tensor) -> torch.Tensor:
        x = self.initial(z_q)  # [B, 192, 8, 8, 8]
        x = self.up1(x)  # [B, 96, 16, 16, 16]
        x = torch.cat([x, skip_16], dim=1)  # [B, 192, 16, 16, 16]
        x = self.skip1_proj(x)  # [B, 96, 16, 16, 16]
        x = self.post_skip1(x)  # [B, 96, 16, 16, 16]
        x = self.up2(x)  # [B, 96, 32, 32, 32]
        x = torch.cat([x, skip_32], dim=1)  # [B, 128, 32, 32, 32]
        x = self.skip2_proj(x)  # [B, 96, 32, 32, 32]
        x = self.post_skip2(x)  # [B, 96, 32, 32, 32]
        logits = self.final(x)  # [B, vocab_size, 32, 32, 32]
        return logits


class VQVAEv7(nn.Module):
    """VQ-VAE v7 with U-Net skip connections + RFSQ."""

    def __init__(self, vocab_size: int, emb_dim: int, hidden_dims: List[int],
                 rfsq_levels: List[int], num_stages: int, pretrained_emb: np.ndarray,
                 dropout: float = 0.1, **kwargs):
        super().__init__()

        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.rfsq_dim = len(rfsq_levels)
        self.num_stages = num_stages

        # Block embeddings (frozen)
        self.block_emb = nn.Embedding(vocab_size, emb_dim)
        self.block_emb.weight.data.copy_(torch.from_numpy(pretrained_emb))
        self.block_emb.weight.requires_grad = False

        # Encoder with dual skip outputs
        self.encoder = EncoderV7(emb_dim, hidden_dims, self.rfsq_dim, dropout)

        # RFSQ quantization
        self.rfsq = RFSQ(rfsq_levels, num_stages)

        # Decoder with dual skip inputs
        self.decoder = DecoderV7(
            self.rfsq_dim,
            list(reversed(hidden_dims)),
            vocab_size,
            emb_dim,
            dropout
        )

    def forward(self, block_ids: torch.Tensor) -> torch.Tensor:
        # Embed blocks
        x = self.block_emb(block_ids)  # [B, 32, 32, 32, emb_dim]
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # [B, emb_dim, 32, 32, 32]

        # Encode with dual skip connections
        z_e, skip_16, skip_32 = self.encoder(x)

        # Permute for RFSQ (expects channels last)
        z_e = z_e.permute(0, 2, 3, 4, 1).contiguous()  # [B, 8, 8, 8, 4]

        # Quantize
        z_q, all_indices = self.rfsq(z_e)

        # Permute back for decoder
        z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()  # [B, 4, 8, 8, 8]

        # Decode with dual skip connections
        logits = self.decoder(z_q, skip_16, skip_32)  # [B, vocab, 32, 32, 32]

        return logits


# ============================================================
# Schematic I/O - Using raw NBT handling for compatibility
# ============================================================

def read_nbt(path: Path) -> dict:
    """Read NBT file and return as dict (handles gzipped files)."""
    import nbtlib
    nbt = nbtlib.load(path)
    return nbt


def read_schematic(path: Path) -> Tuple[np.ndarray, Dict[int, str], Dict]:
    """
    Read a .schem file and return block array + metadata.

    Returns:
        blocks: numpy array of shape (X, Y, Z) matching model format
        palette: dict mapping index -> block name
        metadata: dict with Width, Height, Length, DataVersion
    """
    nbt = read_nbt(path)

    # Dimensions: Width=X, Height=Y, Length=Z
    width = int(nbt['Width'])    # X
    height = int(nbt['Height'])  # Y
    length = int(nbt['Length'])  # Z
    data_version = int(nbt.get('DataVersion', 3578))

    # Build reverse palette (index -> block name)
    palette = {}
    for block_name, idx in nbt['Palette'].items():
        palette[int(idx)] = block_name

    # Decode block data (may be varint encoded for large palettes)
    block_data = nbt['BlockData']
    palette_size = len(palette)

    if palette_size < 128:
        # Simple case: 1 byte per block
        blocks_flat = np.array([int(b) for b in block_data], dtype=np.int32)
    else:
        # Varint decoding needed
        blocks_flat = []
        i = 0
        data = [int(b) & 0xFF for b in block_data]  # Convert signed bytes to unsigned
        while i < len(data):
            value = 0
            shift = 0
            while True:
                b = data[i]
                i += 1
                value |= (b & 0x7F) << shift
                if (b & 0x80) == 0:
                    break
                shift += 7
            blocks_flat.append(value)
        blocks_flat = np.array(blocks_flat, dtype=np.int32)

    # Sponge schematic stores data in YZX order:
    # index = (y * length + z) * width + x
    # So reshape to (Y, Z, X) first, then transpose to (X, Y, Z) for model
    blocks_yzx = blocks_flat.reshape((height, length, width))  # (Y, Z, X)
    blocks_xyz = blocks_yzx.transpose(2, 0, 1)  # (X, Y, Z)

    metadata = {
        'Width': width,    # X
        'Height': height,  # Y
        'Length': length,  # Z
        'DataVersion': data_version,
    }

    return blocks_xyz, palette, metadata


def write_schematic(path: Path, blocks_xyz: np.ndarray, palette: Dict[int, str], data_version: int = 3578):
    """
    Write a .schem file from block array.

    Args:
        blocks_xyz: numpy array of shape (X, Y, Z) with palette indices
        palette: dict mapping index -> block name
        data_version: Minecraft data version (3578 = 1.20.4)
    """
    import nbtlib
    from nbtlib import Byte, Short, Int, Compound, List as NBTList, ByteArray

    width, height, length = blocks_xyz.shape  # X, Y, Z

    # Transpose from (X, Y, Z) to (Y, Z, X) for Sponge format
    blocks_yzx = blocks_xyz.transpose(1, 2, 0)  # (Y, Z, X)
    flat = blocks_yzx.flatten()

    palette_size = len(palette)

    if palette_size < 128:
        # Simple case: 1 byte per block
        block_data = ByteArray([int(b) for b in flat])
    else:
        # Varint encoding
        encoded = []
        for val in flat:
            val = int(val)
            while val >= 0x80:
                encoded.append((val & 0x7F) | 0x80)
                val >>= 7
            encoded.append(val)
        block_data = ByteArray(encoded)

    # Build NBT palette (block name -> index)
    nbt_palette = Compound({name: Int(idx) for idx, name in palette.items()})

    # Create schematic compound
    schematic = Compound({
        'Version': Int(2),
        'DataVersion': Int(data_version),
        'Width': Short(width),
        'Height': Short(height),
        'Length': Short(length),
        'PaletteMax': Int(len(palette)),
        'Palette': nbt_palette,
        'BlockData': block_data,
        'BlockEntities': NBTList[Compound]([]),
        'Metadata': Compound({
            'WEOffsetX': Int(0),
            'WEOffsetY': Int(0),
            'WEOffsetZ': Int(0),
        }),
    })

    # Create file with the schematic as root
    # Sponge schematic format requires root_name='Schematic' for WorldEdit compatibility
    nbt_file = nbtlib.File(schematic, root_name='Schematic', gzipped=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    nbt_file.save(path)


def schematic_to_tokens(blocks: np.ndarray, palette: Dict[int, str],
                        block2tok: Dict[str, int], air_token: int = AIR_TOKEN) -> np.ndarray:
    """Convert schematic palette indices to our vocabulary tokens."""
    result = np.full(blocks.shape, air_token, dtype=np.int64)

    for pal_idx, block_name in palette.items():
        # Try exact match first
        if block_name in block2tok:
            token = block2tok[block_name]
        else:
            # Strip block state
            base_name = block_name.split('[')[0] if '[' in block_name else block_name
            if base_name in block2tok:
                token = block2tok[base_name]
            else:
                token = air_token  # Unknown block -> air

        result[blocks == pal_idx] = token

    return result


def tokens_to_schematic(tokens: np.ndarray, tok2block: Dict[int, str]) -> Tuple[np.ndarray, Dict[int, str]]:
    """Convert vocabulary tokens to schematic palette format."""
    unique_tokens = np.unique(tokens)

    # Build palette
    palette = {}
    token_to_pal = {}
    for i, tok in enumerate(unique_tokens):
        block_name = tok2block.get(str(int(tok)), "minecraft:stone")
        palette[i] = block_name
        token_to_pal[int(tok)] = i

    # Convert tokens to palette indices
    result = np.zeros(tokens.shape, dtype=np.int32)
    for tok, pal_idx in token_to_pal.items():
        result[tokens == tok] = pal_idx

    return result, palette


def resize_to_32(blocks: np.ndarray, air_value: int = 0) -> np.ndarray:
    """Resize/pad structure to 32x32x32. Input is (X, Y, Z)."""
    x, y, z = blocks.shape

    if x == y == z == 32:
        return blocks

    # Create 32x32x32 array filled with air
    result = np.full((32, 32, 32), air_value, dtype=blocks.dtype)

    # Copy blocks (from origin)
    cx, cy, cz = min(x, 32), min(y, 32), min(z, 32)
    result[:cx, :cy, :cz] = blocks[:cx, :cy, :cz]

    return result


# ============================================================
# Main
# ============================================================

def load_model(checkpoint_path: Path, embeddings: np.ndarray, device: str = "cpu"):
    """Load VQ-VAE model from checkpoint (auto-detects v5 or v7)."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Detect version by checking for v7-specific keys
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Check if this is v7 (has encoder.stage1) or v5 (has encoder.encoder)
    is_v7 = any(k.startswith('encoder.stage1') for k in state_dict.keys())

    config = checkpoint.get('config', {})
    vocab_size = config.get('vocab_size', 3717)
    emb_dim = config.get('emb_dim', embeddings.shape[1])
    hidden_dims = config.get('hidden_dims', HIDDEN_DIMS)
    dropout = config.get('dropout', 0.1)

    if is_v7:
        print("  Detected v7 checkpoint (RFSQ + U-Net)")
        rfsq_levels = config.get('rfsq_levels', RFSQ_LEVELS_PER_STAGE)
        num_stages = config.get('num_stages', NUM_STAGES)

        model = VQVAEv7(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            hidden_dims=hidden_dims,
            rfsq_levels=rfsq_levels,
            num_stages=num_stages,
            pretrained_emb=embeddings,
            dropout=dropout,
        )
    else:
        print("  Detected v5 checkpoint (FSQ)")
        fsq_levels = config.get('fsq_levels', FSQ_LEVELS)

        model = VQVAEv5(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            hidden_dims=hidden_dims,
            fsq_levels=fsq_levels,
            pretrained_emb=embeddings,
            dropout=dropout,
        )

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


def reconstruct(model, tokens: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Reconstruct a structure. Input/output shape: (X, Y, Z)."""
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(tokens).long().unsqueeze(0).to(device)
        logits = model(x)  # [B, vocab, X, Y, Z]
        preds = logits.argmax(dim=1)[0].cpu().numpy()  # [X, Y, Z]
    return preds


def process_schematic(input_path: Path, output_path: Path, model, tok2block, block2tok, device):
    """Process a single schematic file."""
    print(f"  Reading: {input_path.name}")

    # Read schematic (returns X, Y, Z format)
    blocks_xyz, palette, metadata = read_schematic(input_path)
    x, y, z = blocks_xyz.shape
    print(f"    Original size: X={x}, Y={y}, Z={z}")

    # Convert to tokens
    tokens = schematic_to_tokens(blocks_xyz, palette, block2tok)

    # Resize to 32x32x32
    tokens_32 = resize_to_32(tokens, air_value=AIR_TOKEN)

    # Reconstruct
    print(f"    Reconstructing...")
    reconstructed_tokens = reconstruct(model, tokens_32, device=device)

    # Convert back to schematic format
    recon_blocks, recon_palette = tokens_to_schematic(reconstructed_tokens, tok2block)

    # Write output
    write_schematic(output_path, recon_blocks, recon_palette, metadata['DataVersion'])
    print(f"    Saved: {output_path.name}")

    # Compute accuracy on non-air blocks
    air_tokens = {102, 576, 3352}  # air, cave_air, void_air
    orig_air_mask = np.isin(tokens_32, list(air_tokens))
    struct_mask = ~orig_air_mask
    if struct_mask.sum() > 0:
        accuracy = (tokens_32[struct_mask] == reconstructed_tokens[struct_mask]).mean()
        print(f"    Structure accuracy: {accuracy:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Reconstruct schematics using R-RFSQ-VAE v7")
    parser.add_argument("input", nargs="?", help="Input .schem file (or --all)")
    parser.add_argument("--all", action="store_true", help="Process all schematics in FAWE folder")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--suffix", type=str, default="_v7", help="Suffix for output files")

    args = parser.parse_args()

    print("=" * 60)
    print("R-RFSQ-VAE v7 Schematic Reconstructor")
    print("=" * 60)

    # Load vocabulary
    print(f"\nLoading vocabulary...")
    with open(VOCAB_PATH, 'r') as f:
        tok2block = json.load(f)
    block2tok = {v: int(k) for k, v in tok2block.items()}
    print(f"  {len(tok2block)} blocks")

    # Load embeddings
    print(f"Loading embeddings...")
    embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model... (device: {device})")
    model = load_model(CHECKPOINT_PATH, embeddings, device=device)

    # Determine input files
    if args.all:
        # Debug: show the path being used
        print(f"\nLooking for schematics in: {FAWE_SCHEMATICS}")
        print(f"  Absolute path: {FAWE_SCHEMATICS.resolve()}")
        print(f"  Path exists: {FAWE_SCHEMATICS.exists()}")
        print(f"  Is directory: {FAWE_SCHEMATICS.is_dir()}")
        
        # Check if directory exists
        if not FAWE_SCHEMATICS.exists():
            print(f"ERROR: Schematics directory does not exist: {FAWE_SCHEMATICS}")
            sys.exit(1)
        
        if not FAWE_SCHEMATICS.is_dir():
            print(f"ERROR: Path is not a directory: {FAWE_SCHEMATICS}")
            sys.exit(1)
        
        # Try multiple methods to list files
        import os
        input_files = []  # Initialize to avoid UnboundLocalError
        
        try:
            # Method 1: os.listdir (more reliable on Windows/OneDrive)
            os_files = os.listdir(str(FAWE_SCHEMATICS))
            print(f"  os.listdir found {len(os_files)} items")
            if len(os_files) > 0:
                print(f"  Sample files from os.listdir: {os_files[:5]}")
                # Convert to Path objects and filter
                path_files = [FAWE_SCHEMATICS / f for f in os_files]
                input_files = [f for f in path_files if f.is_file() and f.suffix.lower() == '.schem']
                print(f"  Found {len(input_files)} .schem files via os.listdir")
            else:
                print(f"  os.listdir returned empty list")
        except Exception as e:
            print(f"  os.listdir failed: {e}")
            input_files = []
        
        # Method 2: Path.glob (if os.listdir didn't work)
        if len(input_files) == 0:
            input_files = list(FAWE_SCHEMATICS.glob("*.schem"))
            if len(input_files) == 0:
                input_files = list(FAWE_SCHEMATICS.glob("*.SCHEM"))
            if len(input_files) > 0:
                print(f"  Found {len(input_files)} .schem files via glob")
        
        # Method 3: Path.iterdir (fallback)
        if len(input_files) == 0:
            try:
                all_files = list(FAWE_SCHEMATICS.iterdir())
                input_files = [f for f in all_files if f.is_file() and f.suffix.lower() == '.schem']
                if len(input_files) > 0:
                    print(f"  Found {len(input_files)} .schem files via iterdir")
            except Exception as e:
                print(f"  iterdir failed: {e}")
        
        print(f"\nTotal found: {len(input_files)} .schem files to process")
        
        if len(input_files) == 0:
            print(f"\nWARNING: No .schem files found!")
            print(f"  Tried path: {FAWE_SCHEMATICS.resolve()}")
            print(f"  Please verify the path is correct and files are accessible")
        
        # Filter out files that are already reconstructions
        before_filter = len(input_files)
        input_files = [f for f in input_files if args.suffix not in f.stem]
        after_filter = len(input_files)
        
        if before_filter > after_filter:
            print(f"  Filtered out {before_filter - after_filter} files with suffix '{args.suffix}'")
        
        print(f"  Processing {len(input_files)} schematics")
    elif args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            # Try in FAWE schematics folder
            input_path = FAWE_SCHEMATICS / args.input
            if not input_path.exists():
                input_path = FAWE_SCHEMATICS / (args.input + ".schem")
        if not input_path.exists():
            print(f"ERROR: File not found: {args.input}")
            sys.exit(1)
        input_files = [input_path]
    else:
        parser.print_help()
        sys.exit(1)

    # Determine output directory
    output_dir = Path(args.output_dir) if args.output_dir else FAWE_SCHEMATICS
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process files
    print(f"\nProcessing {len(input_files)} file(s)...")
    for input_path in input_files:
        output_name = input_path.stem + args.suffix + ".schem"
        output_path = output_dir / output_name

        try:
            process_schematic(input_path, output_path, model, tok2block, block2tok, device)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("\nDone!")
    print(f"Output files in: {output_dir}")


if __name__ == "__main__":
    main()
