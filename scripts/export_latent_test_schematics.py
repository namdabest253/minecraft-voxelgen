"""Export real, reconstructed, and generated schematics for visual comparison."""

import json
import sys
from pathlib import Path
from typing import List, Set

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.export.schematic_exporter import export_to_schematic


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
        x = self.block_emb(block_ids).permute(0, 4, 1, 2, 3)
        z = self.encoder(x)
        z = z.permute(0, 2, 3, 4, 1)
        z_q, indices = self.quantizer(z)
        z_q_permuted = z_q.permute(0, 4, 1, 2, 3)
        binary_logits, block_logits = self.decoder(z_q_permuted)
        binary_pred = binary_logits.argmax(dim=1)
        block_pred = block_logits.argmax(dim=1)
        air_token = 102
        output = torch.where(binary_pred == 0, air_token, block_pred)
        return output, z


def main():
    BASE_PATH = Path(__file__).parent.parent
    DATA_PATH = BASE_PATH / "data"

    TRAIN_DIR = DATA_PATH / "splits" / "train"
    VQVAE_PATH = DATA_PATH / "output" / "vqvae" / "v16b" / "vqvae_v16b_best.pt"
    EMBEDDINGS_PATH = DATA_PATH / "output" / "block2vec" / "v3" / "block_embeddings_v3.npy"
    VOCAB_PATH = DATA_PATH / "vocabulary" / "tok2block.json"
    GEN_SAMPLES_PATH = DATA_PATH / "output" / "prior" / "diffusion_v4" / "generated_samples.pt"

    OUTPUT_DIR = Path(r"C:\Users\namda\OneDrive\Desktop\Claude_Server\server\plugins\FastAsyncWorldEdit\schematics")

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
    print("Loading VQ-VAE v16b...")
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

    # Load and process real structures
    print("\nProcessing real structures...")
    h5_files = sorted(TRAIN_DIR.glob("*.h5"))[:5]

    for i, h5_file in enumerate(h5_files):
        print(f"\nStructure {i+1}: {h5_file.name}")

        with h5py.File(h5_file, 'r') as f:
            key = list(f.keys())[0]
            real_structure = f[key][:].astype(np.int64)

        # Reconstruct through VQ-VAE
        with torch.no_grad():
            block_ids = torch.from_numpy(real_structure).long().unsqueeze(0).to(device)
            reconstructed, _ = vqvae(block_ids)
            recon_structure = reconstructed.squeeze(0).cpu().numpy()

        # Export real
        export_to_schematic(
            real_structure, tok2block,
            OUTPUT_DIR / f"latent_test_real_{i+1}.schem",
            air_tokens=air_tokens
        )

        # Export reconstructed
        export_to_schematic(
            recon_structure, tok2block,
            OUTPUT_DIR / f"latent_test_recon_{i+1}.schem",
            air_tokens=air_tokens
        )

    # Load and export generated structures
    print("\nProcessing generated structures...")
    generated = torch.load(GEN_SAMPLES_PATH).numpy()

    for i in range(min(5, generated.shape[0])):
        export_to_schematic(
            generated[i], tok2block,
            OUTPUT_DIR / f"latent_test_gen_{i+1}.schem",
            air_tokens=air_tokens
        )

    print("\n" + "="*70)
    print("EXPORT COMPLETE!")
    print("="*70)
    print(f"\nSchematics saved to: {OUTPUT_DIR}")
    print("\nFiles created:")
    print("  - latent_test_real_1.schem through latent_test_real_5.schem")
    print("  - latent_test_recon_1.schem through latent_test_recon_5.schem")
    print("  - latent_test_gen_1.schem through latent_test_gen_5.schem")
    print("\nTo compare in Minecraft:")
    print("  //schem load latent_test_real_1")
    print("  //paste")
    print("  (move away)")
    print("  //schem load latent_test_recon_1")
    print("  //paste")
    print("  (compare side by side)")


if __name__ == "__main__":
    main()
