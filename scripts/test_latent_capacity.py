"""
Test whether the VQ-VAE latent space can represent coherent structures.

This is the KEY diagnostic: if reconstructions lose architectural coherence,
the latent space itself is the bottleneck, not diffusion training.

Compares: Real → Reconstructed → Generated
"""

import json
import sys
from pathlib import Path
from typing import List, Set

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.export.schematic_exporter import export_to_schematic


# =============================================================================
# Metrics (same as analyze_structure_quality.py)
# =============================================================================

def get_binary_mask(structure: np.ndarray, air_tokens: set) -> np.ndarray:
    mask = np.ones_like(structure, dtype=np.uint8)
    for air_token in air_tokens:
        mask[structure == air_token] = 0
    return mask


def count_connected_components(binary_mask: np.ndarray) -> tuple:
    labeled, num_features = ndimage.label(binary_mask)
    if num_features == 0:
        return 0, 0.0
    component_sizes = ndimage.sum(binary_mask, labeled, range(1, num_features + 1))
    largest_size = max(component_sizes) if len(component_sizes) > 0 else 0
    total_blocks = binary_mask.sum()
    largest_ratio = largest_size / total_blocks if total_blocks > 0 else 0
    return num_features, largest_ratio


def compute_support_ratio(binary_mask: np.ndarray) -> float:
    if binary_mask.sum() == 0:
        return 0.0
    supported = np.zeros_like(binary_mask)
    supported[:, 0, :] = binary_mask[:, 0, :]
    for y in range(1, binary_mask.shape[1]):
        supported[:, y, :] = binary_mask[:, y, :] & binary_mask[:, y-1, :]
    total_blocks = binary_mask.sum()
    supported_blocks = (binary_mask & supported).sum()
    return supported_blocks / total_blocks if total_blocks > 0 else 0.0


def compute_ground_contact(binary_mask: np.ndarray) -> float:
    bottom_layer = binary_mask[:, 0, :]
    return bottom_layer.sum() / bottom_layer.size


def compute_density(binary_mask: np.ndarray) -> float:
    return binary_mask.sum() / binary_mask.size


def compute_local_coherence(structure: np.ndarray, binary_mask: np.ndarray) -> float:
    if binary_mask.sum() < 2:
        return 0.0
    same_neighbor_count = 0
    total_neighbor_count = 0
    for dx, dz in [(1, 0), (0, 1)]:
        for x in range(structure.shape[0] - dx):
            for y in range(structure.shape[1]):
                for z in range(structure.shape[2] - dz):
                    if binary_mask[x, y, z] and binary_mask[x+dx, y, z+dz]:
                        total_neighbor_count += 1
                        if structure[x, y, z] == structure[x+dx, y, z+dz]:
                            same_neighbor_count += 1
    return same_neighbor_count / total_neighbor_count if total_neighbor_count > 0 else 0.0


def compute_block_entropy(structure: np.ndarray, binary_mask: np.ndarray) -> float:
    if binary_mask.sum() == 0:
        return 0.0
    block_types = structure[binary_mask == 1]
    unique, counts = np.unique(block_types, return_counts=True)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    return entropy


def analyze_structure(structure: np.ndarray, air_tokens: set) -> dict:
    binary_mask = get_binary_mask(structure, air_tokens)
    num_components, largest_ratio = count_connected_components(binary_mask)
    return {
        'density': compute_density(binary_mask),
        'num_components': num_components,
        'largest_component_ratio': largest_ratio,
        'support_ratio': compute_support_ratio(binary_mask),
        'ground_contact': compute_ground_contact(binary_mask),
        'local_coherence': compute_local_coherence(structure, binary_mask),
        'block_entropy': compute_block_entropy(structure, binary_mask),
        'total_blocks': int(binary_mask.sum()),
    }


# =============================================================================
# VQ-VAE Model (minimal for loading)
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

    def forward(self, x):
        self.stored_mean = x.mean(dim=(1, 2, 3), keepdim=True)
        self.stored_std = x.std(dim=(1, 2, 3), keepdim=True) + self.eps
        x_norm = (x - self.stored_mean) / self.stored_std
        return x_norm * self.weight + self.bias

    def inverse(self, x_norm):
        x = (x_norm - self.bias) / self.weight
        return x * self.stored_std + self.stored_mean


class RFSQStage(nn.Module):
    def __init__(self, levels: List[int]):
        super().__init__()
        self.fsq = FSQ(levels)
        self.layernorm = InvertibleLayerNorm(len(levels))

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
        self.stages = nn.ModuleList([RFSQStage(levels_per_stage) for _ in range(num_stages)])

    def forward(self, z):
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
    def __init__(self, in_channels=40, hidden_dim=128, rfsq_dim=4):
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
    def __init__(self, rfsq_dim=4, hidden_dim=128, num_blocks=3717):
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
    def __init__(self, vocab_size=3717, emb_dim=40, hidden_dim=128,
                 rfsq_levels=None, num_stages=2, pretrained_embeddings=None, air_tokens=None):
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

    def forward(self, block_ids):
        """Full encode-decode pass."""
        # Encode
        x = self.block_emb(block_ids).permute(0, 4, 1, 2, 3)
        z = self.encoder(x)
        z = z.permute(0, 2, 3, 4, 1)  # [B, 8, 8, 8, 4]

        # Quantize
        z_q, indices = self.quantizer(z)

        # Decode
        z_q_permuted = z_q.permute(0, 4, 1, 2, 3)
        binary_logits, block_logits = self.decoder(z_q_permuted)

        # Get predictions
        binary_pred = binary_logits.argmax(dim=1)
        block_pred = block_logits.argmax(dim=1)

        # Combine: air where binary=0, block type where binary=1
        air_token = 102
        output = torch.where(binary_pred == 0, air_token, block_pred)

        return output, z


# =============================================================================
# Main Test
# =============================================================================

def main():
    BASE_PATH = Path(__file__).parent.parent
    DATA_PATH = BASE_PATH / "data"

    TRAIN_DIR = DATA_PATH / "splits" / "train"
    VQVAE_PATH = DATA_PATH / "output" / "vqvae" / "v16b" / "vqvae_v16b_best.pt"
    EMBEDDINGS_PATH = DATA_PATH / "output" / "block2vec" / "v3" / "block_embeddings_v3.npy"
    VOCAB_PATH = DATA_PATH / "vocabulary" / "tok2block.json"
    GEN_SAMPLES_PATH = DATA_PATH / "output" / "prior" / "diffusion_v4" / "generated_samples.pt"

    OUTPUT_DIR = Path(r"C:\Users\namda\OneDrive\Desktop\Claude_Server\server\plugins\FastAsyncWorldEdit\schematics\latent_test")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    air_tokens = {102, 576, 3352}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load vocab
    with open(VOCAB_PATH, 'r') as f:
        tok2block = {int(k): v for k, v in json.load(f).items()}

    # Load embeddings
    embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)
    embeddings_tensor = torch.from_numpy(embeddings)

    # Load VQ-VAE
    print(f"\nLoading VQ-VAE v16b...")
    vqvae = VQVAEv16b(
        vocab_size=len(tok2block),
        emb_dim=40,
        hidden_dim=128,
        rfsq_levels=[5, 5, 5, 5],
        num_stages=2,
        pretrained_embeddings=embeddings_tensor,
        air_tokens=air_tokens,
    ).to(device)

    vqvae.load_state_dict(torch.load(VQVAE_PATH, map_location=device, weights_only=False))
    vqvae.eval()
    print("VQ-VAE loaded!")

    # Load 10 real structures
    print(f"\nLoading real structures...")
    h5_files = sorted(TRAIN_DIR.glob("*.h5"))[:10]

    real_structures = []
    reconstructed_structures = []

    for h5_file in h5_files:
        with h5py.File(h5_file, 'r') as f:
            key = list(f.keys())[0]
            structure = f[key][:].astype(np.int64)
        real_structures.append(structure)

        # Reconstruct through VQ-VAE
        with torch.no_grad():
            block_ids = torch.from_numpy(structure).long().unsqueeze(0).to(device)
            reconstructed, latent_z = vqvae(block_ids)
            reconstructed_structures.append(reconstructed.squeeze(0).cpu().numpy())

    print(f"Loaded and reconstructed {len(real_structures)} structures")

    # Load generated structures
    print(f"\nLoading generated structures...")
    generated = torch.load(GEN_SAMPLES_PATH).numpy()
    generated_structures = [generated[i] for i in range(min(10, generated.shape[0]))]
    print(f"Loaded {len(generated_structures)} generated structures")

    # Compute metrics for all three
    print("\n" + "="*80)
    print("LATENT SPACE CAPACITY TEST")
    print("="*80)
    print("\nThis test answers: Can the latent space REPRESENT coherent structures?")
    print("If reconstructions are incoherent, the latent space is the bottleneck.\n")

    def compute_avg_metrics(structures, name):
        metrics_list = []
        for s in structures:
            metrics_list.append(analyze_structure(s, air_tokens))

        avg = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list]
            avg[key] = {'mean': np.mean(values), 'std': np.std(values)}
        return avg

    real_metrics = compute_avg_metrics(real_structures, "Real")
    recon_metrics = compute_avg_metrics(reconstructed_structures, "Reconstructed")
    gen_metrics = compute_avg_metrics(generated_structures, "Generated")

    # Print comparison table
    print(f"{'Metric':<25} {'REAL':<20} {'RECONSTRUCTED':<20} {'GENERATED':<20}")
    print("-"*85)

    key_metrics = ['ground_contact', 'support_ratio', 'local_coherence',
                   'block_entropy', 'num_components', 'density']

    for key in key_metrics:
        real_val = f"{real_metrics[key]['mean']:.3f}±{real_metrics[key]['std']:.3f}"
        recon_val = f"{recon_metrics[key]['mean']:.3f}±{recon_metrics[key]['std']:.3f}"
        gen_val = f"{gen_metrics[key]['mean']:.3f}±{gen_metrics[key]['std']:.3f}"
        print(f"{key:<25} {real_val:<20} {recon_val:<20} {gen_val:<20}")

    # Compute degradation
    print("\n" + "="*80)
    print("DEGRADATION ANALYSIS")
    print("="*80)

    print(f"\n{'Metric':<25} {'Real→Recon':<20} {'Real→Gen':<20} {'Recon→Gen':<20}")
    print("-"*85)

    for key in key_metrics:
        real_val = real_metrics[key]['mean']
        recon_val = recon_metrics[key]['mean']
        gen_val = gen_metrics[key]['mean']

        if real_val != 0:
            real_to_recon = (recon_val - real_val) / real_val * 100
            real_to_gen = (gen_val - real_val) / real_val * 100
        else:
            real_to_recon = 0
            real_to_gen = 0

        if recon_val != 0:
            recon_to_gen = (gen_val - recon_val) / recon_val * 100
        else:
            recon_to_gen = 0

        print(f"{key:<25} {real_to_recon:>+.1f}%{'':<13} {real_to_gen:>+.1f}%{'':<13} {recon_to_gen:>+.1f}%")

    # Key insight
    print("\n" + "="*80)
    print("KEY INSIGHT")
    print("="*80)

    recon_ground = recon_metrics['ground_contact']['mean']
    real_ground = real_metrics['ground_contact']['mean']
    gen_ground = gen_metrics['ground_contact']['mean']

    recon_support = recon_metrics['support_ratio']['mean']
    real_support = real_metrics['support_ratio']['mean']
    gen_support = gen_metrics['support_ratio']['mean']

    print(f"\nGround Contact: Real={real_ground:.1%} → Recon={recon_ground:.1%} → Gen={gen_ground:.1%}")
    print(f"Support Ratio:  Real={real_support:.1%} → Recon={recon_support:.1%} → Gen={gen_support:.1%}")

    if recon_ground > 0.10 and recon_support > 0.60:
        print("\n✅ RECONSTRUCTIONS MAINTAIN STRUCTURE!")
        print("   The latent space CAN represent coherent buildings.")
        print("   Problem is in DIFFUSION TRAINING, not VQ-VAE.")
        print("   → Focus on better diffusion supervision.")
    else:
        print("\n❌ RECONSTRUCTIONS LOSE STRUCTURE!")
        print("   The latent space CANNOT represent coherent buildings.")
        print("   Problem is in VQ-VAE COMPRESSION, not diffusion.")
        print("   → Need larger latent space or better VQ-VAE.")

    # Export schematics for visual comparison
    print("\n" + "="*80)
    print("EXPORTING SCHEMATICS FOR VISUAL COMPARISON")
    print("="*80)

    for i in range(min(5, len(real_structures))):
        # Real
        export_to_schematic(
            real_structures[i], tok2block,
            OUTPUT_DIR / f"real_{i+1}.schem",
            air_tokens=air_tokens
        )

        # Reconstructed
        export_to_schematic(
            reconstructed_structures[i], tok2block,
            OUTPUT_DIR / f"reconstructed_{i+1}.schem",
            air_tokens=air_tokens
        )

    for i in range(min(5, len(generated_structures))):
        # Generated
        export_to_schematic(
            generated_structures[i], tok2block,
            OUTPUT_DIR / f"generated_{i+1}.schem",
            air_tokens=air_tokens
        )

    print(f"\nSchematics saved to: {OUTPUT_DIR}")
    print("\nTo compare in Minecraft:")
    print("  //schem load real_1")
    print("  //schem load reconstructed_1")
    print("  //schem load generated_1")


if __name__ == "__main__":
    main()
