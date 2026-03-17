"""
Latent Space Explorer v2: Using specific good structures for testing.
"""

import gzip
import json
import random
import sys
from pathlib import Path
from typing import List, Set, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.export.schematic_exporter import export_to_schematic


# =============================================================================
# Model Components (same as before)
# =============================================================================

class FSQ(nn.Module):
    def __init__(self, levels: List[int], eps: float = 1e-3):
        super().__init__()
        self.levels = levels
        self.dim = len(levels)
        self.eps = eps
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
                 pretrained_embeddings: torch.Tensor = None, air_tokens: Set[int] = None):
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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_emb = self.block_emb(x)
        x_emb = x_emb.permute(0, 4, 1, 2, 3)
        z = self.encoder(x_emb)
        z = z.permute(0, 2, 3, 4, 1)
        return z

    def decode(self, z: torch.Tensor, quantize: bool = True) -> torch.Tensor:
        if quantize:
            z_q, _ = self.quantizer(z)
        else:
            z_q = z
        z_q_permuted = z_q.permute(0, 4, 1, 2, 3)
        binary_logits, block_logits = self.decoder(z_q_permuted)
        binary_pred = binary_logits.argmax(dim=1)
        block_pred = block_logits.argmax(dim=1)
        return torch.where(binary_pred == 0, 102, block_pred)


class LatentExplorer:
    def __init__(self, vqvae: VQVAEv16b, device: torch.device):
        self.vqvae = vqvae
        self.device = device
        self.vqvae.eval()

    @torch.no_grad()
    def encode(self, structure: torch.Tensor) -> torch.Tensor:
        return self.vqvae.encode(structure.to(self.device))

    @torch.no_grad()
    def decode(self, z: torch.Tensor, quantize: bool = True) -> torch.Tensor:
        return self.vqvae.decode(z.to(self.device), quantize=quantize)

    @torch.no_grad()
    def vary(self, structure: torch.Tensor, epsilon: float = 0.1, num_variations: int = 5) -> List[torch.Tensor]:
        z = self.encode(structure)
        variations = []
        for _ in range(num_variations):
            noise = torch.randn_like(z) * epsilon
            z_var = z + noise
            variations.append(self.decode(z_var))
        return variations

    @torch.no_grad()
    def interpolate(self, struct_a: torch.Tensor, struct_b: torch.Tensor, steps: int = 10) -> List[torch.Tensor]:
        z_a = self.encode(struct_a)
        z_b = self.encode(struct_b)
        interpolations = []
        for t in np.linspace(0, 1, steps):
            z_interp = (1 - t) * z_a + t * z_b
            interpolations.append(self.decode(z_interp))
        return interpolations

    @torch.no_grad()
    def sample_neighborhood(self, structures: List[torch.Tensor], sigma: float = 0.2, num_samples: int = 10) -> List[torch.Tensor]:
        samples = []
        for _ in range(num_samples):
            idx = random.randint(0, len(structures) - 1)
            z = self.encode(structures[idx])
            z_sample = z + torch.randn_like(z) * sigma
            samples.append(self.decode(z_sample))
        return samples


def load_structure_from_h5(filepath: Path) -> np.ndarray:
    with h5py.File(filepath, 'r') as f:
        for key in ['build', 'structure', 'data']:
            if key in f:
                data = f[key][:]
                if data.dtype == np.uint8 and len(data.shape) == 1:
                    data = np.frombuffer(gzip.decompress(data.tobytes()), dtype=np.int16)
                    data = data.reshape(32, 32, 32)
                return data.astype(np.int64)
        raise KeyError(f"No known data key in {filepath}")


def compute_density(structure: np.ndarray, air_tokens: Set[int]) -> float:
    total = structure.size
    air_count = sum(np.sum(structure == t) for t in air_tokens)
    return (total - air_count) / total


def count_unique_blocks(structure: np.ndarray, air_tokens: Set[int]) -> int:
    unique = set(np.unique(structure))
    return len(unique - air_tokens)


def main():
    print("=" * 60)
    print("LATENT SPACE EXPLORATION v2 - Better Base Structures")
    print("=" * 60)

    project_root = Path(__file__).parent.parent
    vqvae_checkpoint = project_root / "data" / "output" / "vqvae" / "v16b" / "vqvae_v16b_best.pt"
    embeddings_path = project_root / "data" / "output" / "block2vec" / "v3" / "block_embeddings_v3.npy"
    train_data_dir = project_root / "data" / "splits" / "train"
    vocab_path = project_root / "data" / "vocabulary" / "tok2block.json"
    output_dir = Path(r"C:\Users\namda\OneDrive\Desktop\Claude_Server\server\plugins\FastAsyncWorldEdit\schematics")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load vocabulary
    print("\nLoading vocabulary...")
    with open(vocab_path, 'r') as f:
        tok2block = json.load(f)
    vocab_size = len(tok2block)

    # Load embeddings
    print("Loading embeddings...")
    embeddings = np.load(embeddings_path)
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)

    air_tokens = {102, 576, 3352}

    # Build and load VQ-VAE
    print("Building VQ-VAE...")
    vqvae = VQVAEv16b(
        vocab_size=vocab_size,
        emb_dim=embeddings_tensor.shape[1],
        hidden_dim=128,
        rfsq_levels=[5, 5, 5, 5],
        num_stages=2,
        pretrained_embeddings=embeddings_tensor,
        air_tokens=air_tokens,
    ).to(device)

    print(f"Loading checkpoint...")
    checkpoint = torch.load(vqvae_checkpoint, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        vqvae.load_state_dict(checkpoint['model_state_dict'])
    else:
        vqvae.load_state_dict(checkpoint)
    vqvae.eval()

    explorer = LatentExplorer(vqvae, device)

    # Load SPECIFIC good structures
    print("\nLoading specific structures...")

    # Good structures identified earlier:
    # - batch_291_7560.h5: 15.7% density, 118 unique blocks (most diverse)
    # - batch_219_5681.h5: 18.4% density, 102 unique blocks
    # - batch_529_13731.h5: 26.4% density, 100 unique blocks
    # - batch_774_20122.h5: 14.8% density, 36 unique blocks

    specific_files = [
        "batch_291_7560.h5",      # Structure A: 118 unique blocks
        "batch_529_13731.h5",     # Structure B: 100 unique blocks
        "batch_219_5681.h5",      # Structure C: 102 unique blocks
        "batch_774_20122.h5",     # Structure D: 36 unique blocks
    ]

    structures_np = []
    for fname in specific_files:
        fpath = train_data_dir / fname
        if fpath.exists():
            try:
                data = load_structure_from_h5(fpath)
                structures_np.append(data)
                density = compute_density(data, air_tokens)
                unique = count_unique_blocks(data, air_tokens)
                print(f"  Loaded {fname}: density={density:.1%}, unique={unique}")
            except Exception as e:
                print(f"  Failed to load {fname}: {e}")

    if len(structures_np) < 2:
        print("ERROR: Need at least 2 structures. Falling back to random selection.")
        return

    structures = [torch.tensor(s, dtype=torch.long).unsqueeze(0) for s in structures_np]

    # =================================================================
    # EXPERIMENT 1: VARIATIONS OF STRUCTURE A (most diverse)
    # =================================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: VARIATIONS (Structure A - 118 unique blocks)")
    print("=" * 60)

    base_struct = structures[0]
    base_np = structures_np[0]

    # Export original
    export_to_schematic(base_np.astype(np.int32), tok2block, output_dir / "explore2_base_A.schem")
    print(f"Exported: explore2_base_A.schem (density={compute_density(base_np, air_tokens):.1%})")

    # Also export reconstruction to show VAE quality
    recon = explorer.decode(explorer.encode(base_struct))
    recon_np = recon.cpu().numpy()[0]
    export_to_schematic(recon_np.astype(np.int32), tok2block, output_dir / "explore2_recon_A.schem")
    print(f"Exported: explore2_recon_A.schem (reconstruction)")

    # Variations at different epsilon
    epsilons = [0.1, 0.2, 0.3, 0.5]
    for eps in epsilons:
        variations = explorer.vary(base_struct, epsilon=eps, num_variations=2)
        for i, var in enumerate(variations):
            var_np = var.cpu().numpy()[0]
            density = compute_density(var_np, air_tokens)
            unique = count_unique_blocks(var_np, air_tokens)
            filename = f"explore2_var_eps{eps}_v{i+1}.schem"
            export_to_schematic(var_np.astype(np.int32), tok2block, output_dir / filename)
            print(f"  {filename}: density={density:.1%}, unique={unique}")

    # =================================================================
    # EXPERIMENT 2: INTERPOLATION A -> B
    # =================================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: INTERPOLATION (A -> B)")
    print("=" * 60)

    struct_a = structures[0]  # 118 unique blocks
    struct_b = structures[1]  # 100 unique blocks

    # Export endpoints
    export_to_schematic(structures_np[0].astype(np.int32), tok2block, output_dir / "explore2_interp_A.schem")
    export_to_schematic(structures_np[1].astype(np.int32), tok2block, output_dir / "explore2_interp_B.schem")
    print(f"Structure A: density={compute_density(structures_np[0], air_tokens):.1%}, unique={count_unique_blocks(structures_np[0], air_tokens)}")
    print(f"Structure B: density={compute_density(structures_np[1], air_tokens):.1%}, unique={count_unique_blocks(structures_np[1], air_tokens)}")

    # Interpolation
    interpolations = explorer.interpolate(struct_a, struct_b, steps=11)
    for i, interp in enumerate(interpolations):
        t = i / 10
        interp_np = interp.cpu().numpy()[0]
        density = compute_density(interp_np, air_tokens)
        unique = count_unique_blocks(interp_np, air_tokens)
        filename = f"explore2_interp_t{i:02d}.schem"
        export_to_schematic(interp_np.astype(np.int32), tok2block, output_dir / filename)
        print(f"  {filename} (t={t:.1f}): density={density:.1%}, unique={unique}")

    # =================================================================
    # EXPERIMENT 3: INTERPOLATION C -> D (different pair)
    # =================================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: INTERPOLATION (C -> D)")
    print("=" * 60)

    struct_c = structures[2]  # 102 unique blocks
    struct_d = structures[3]  # 36 unique blocks

    export_to_schematic(structures_np[2].astype(np.int32), tok2block, output_dir / "explore2_interp_C.schem")
    export_to_schematic(structures_np[3].astype(np.int32), tok2block, output_dir / "explore2_interp_D.schem")
    print(f"Structure C: density={compute_density(structures_np[2], air_tokens):.1%}, unique={count_unique_blocks(structures_np[2], air_tokens)}")
    print(f"Structure D: density={compute_density(structures_np[3], air_tokens):.1%}, unique={count_unique_blocks(structures_np[3], air_tokens)}")

    interpolations_cd = explorer.interpolate(struct_c, struct_d, steps=11)
    for i, interp in enumerate(interpolations_cd):
        t = i / 10
        interp_np = interp.cpu().numpy()[0]
        density = compute_density(interp_np, air_tokens)
        unique = count_unique_blocks(interp_np, air_tokens)
        filename = f"explore2_interp_CD_t{i:02d}.schem"
        export_to_schematic(interp_np.astype(np.int32), tok2block, output_dir / filename)
        print(f"  {filename} (t={t:.1f}): density={density:.1%}, unique={unique}")

    # =================================================================
    # SUMMARY
    # =================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"""
Files exported to: {output_dir}

Experiment 1 - Variations of Structure A:
  - explore2_base_A.schem (original - 118 unique blocks)
  - explore2_recon_A.schem (VQ-VAE reconstruction)
  - explore2_var_eps*_v*.schem (variations)

Experiment 2 - Interpolation A -> B:
  - explore2_interp_A.schem, explore2_interp_B.schem (endpoints)
  - explore2_interp_t00-t10.schem (11 interpolation steps)

Experiment 3 - Interpolation C -> D:
  - explore2_interp_C.schem, explore2_interp_D.schem (endpoints)
  - explore2_interp_CD_t00-t10.schem (11 interpolation steps)

Key tests:
  1. Compare explore2_base_A vs explore2_recon_A (VAE quality)
  2. Compare explore2_base_A vs explore2_var_eps0.2_v1 (variation quality)
  3. View explore2_interp_t00 through t10 in sequence (smooth blend?)
""")
    print("Done!")


if __name__ == "__main__":
    main()
