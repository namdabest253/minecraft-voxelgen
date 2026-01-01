"""
Place VQ-VAE Reconstruction in Minecraft.

Simple script to load a trained model, reconstruct a build, and output
the commands needed to place both original and reconstruction in Minecraft.

Usage:
    python scripts/place_reconstruction.py --checkpoint model.pt --h5-file build.h5

The script outputs:
1. Metrics comparing original vs reconstruction
2. A list of Minecraft commands to run
3. Files with commands for batch execution
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
VOCAB_PATH = PROJECT_ROOT / "data/kaggle/output/block2vec/v2/tok2block_collapsed.json"
EMBEDDINGS_PATH = PROJECT_ROOT / "data/kaggle/output/block2vec/v3/block_embeddings_v3.npy"
AIR_TOKENS = {19, 164, 932}


# ============================================================
# Model Definition (must match training notebook)
# ============================================================

class ResidualBlock3D(nn.Module):
    def __init__(self, channels):
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


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_codes, latent_dim, commitment_cost=0.5, ema_decay=0.99):
        super().__init__()
        self.num_codes = num_codes
        self.latent_dim = latent_dim
        self.commitment_cost = commitment_cost

        self.register_buffer('codebook', torch.randn(num_codes, latent_dim))
        self.register_buffer('ema_cluster_size', torch.zeros(num_codes))
        self.register_buffer('ema_embed_sum', torch.zeros(num_codes, latent_dim))
        self.register_buffer('code_usage', torch.zeros(num_codes))

    def forward(self, z_e):
        z_e_perm = z_e.permute(0, 2, 3, 4, 1).contiguous()
        flat = z_e_perm.view(-1, self.latent_dim)

        d = (flat.pow(2).sum(1, keepdim=True)
             + self.codebook.pow(2).sum(1)
             - 2 * flat @ self.codebook.t())
        indices = d.argmin(dim=1)

        z_q_flat = self.codebook[indices]
        z_q_perm = z_q_flat.view(z_e_perm.shape)

        commitment_loss = F.mse_loss(z_e_perm, z_q_perm.detach())
        vq_loss = self.commitment_cost * commitment_loss

        z_q_st = z_e_perm + (z_q_perm - z_e_perm).detach()
        z_q = z_q_st.permute(0, 4, 1, 2, 3).contiguous()

        return z_q, vq_loss, indices.view(z_e_perm.shape[:-1])


class EncoderV4(nn.Module):
    def __init__(self, in_channels, hidden_dims, latent_dim, dropout=0.1):
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
            nn.Conv3d(current, latent_dim, 3, padding=1),
        ])
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class DecoderV4(nn.Module):
    def __init__(self, latent_dim, hidden_dims, num_blocks, dropout=0.1):
        super().__init__()
        layers = [ResidualBlock3D(latent_dim), ResidualBlock3D(latent_dim)]

        current = latent_dim
        for h in hidden_dims:
            layers.extend([
                ResidualBlock3D(current),
                nn.ConvTranspose3d(current, h, 4, stride=2, padding=1),
                nn.BatchNorm3d(h),
                nn.ReLU(inplace=True),
                nn.Dropout3d(dropout),
            ])
            current = h

        layers.append(nn.Conv3d(current, num_blocks, 3, padding=1))
        self.decoder = nn.Sequential(*layers)

    def forward(self, z_q):
        return self.decoder(z_q)


class VQVAEv4(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dims, latent_dim, num_codes,
                 pretrained_emb=None, dropout=0.1, commitment_cost=0.5, ema_decay=0.99, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim

        self.block_emb = nn.Embedding(vocab_size, emb_dim)
        if pretrained_emb is not None:
            self.block_emb.weight.data.copy_(pretrained_emb)

        self.register_buffer('original_embeddings', self.block_emb.weight.data.clone())
        self.encoder = EncoderV4(emb_dim, hidden_dims, latent_dim, dropout)
        self.quantizer = VectorQuantizerEMA(num_codes, latent_dim, commitment_cost, ema_decay)
        self.decoder = DecoderV4(latent_dim, list(reversed(hidden_dims)), vocab_size, dropout)

    def forward(self, block_ids):
        x = self.block_emb(block_ids).permute(0, 4, 1, 2, 3).contiguous()
        z_e = self.encoder(x)
        z_q, vq_loss, indices = self.quantizer(z_e)
        logits = self.decoder(z_q)
        return logits


# ============================================================
# Helper Functions
# ============================================================

def load_model(checkpoint_path, embeddings, device="cpu"):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get config from checkpoint or use defaults
    if 'config' in checkpoint:
        cfg = checkpoint['config']
    else:
        cfg = {
            'vocab_size': 3717,
            'emb_dim': 32,
            'hidden_dims': [96, 192],
            'latent_dim': 256,
            'num_codes': 512,
            'dropout': 0.1,
            'commitment_cost': 0.5,
            'ema_decay': 0.99,
        }

    model = VQVAEv4(
        vocab_size=cfg.get('vocab_size', 3717),
        emb_dim=cfg.get('emb_dim', 32),
        hidden_dims=cfg.get('hidden_dims', [96, 192]),
        latent_dim=cfg.get('latent_dim', 256),
        num_codes=cfg.get('num_codes', 512),
        pretrained_emb=embeddings,
        dropout=cfg.get('dropout', 0.1),
        commitment_cost=cfg.get('commitment_cost', 0.5),
        ema_decay=cfg.get('ema_decay', 0.99),
    )

    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model


def load_h5(path):
    """Load structure from H5 file."""
    with h5py.File(path, 'r') as f:
        key = list(f.keys())[0]
        return f[key][:].astype(np.int64)


def reconstruct(model, structure, device="cpu"):
    """Reconstruct a structure."""
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(structure).long().unsqueeze(0).to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)[0].cpu().numpy()
    return preds


def compute_metrics(original, reconstructed):
    """Compute reconstruction metrics."""
    is_air = np.isin(original, list(AIR_TOKENS))
    is_struct = ~is_air

    overall = (original == reconstructed).mean()
    air_acc = (original[is_air] == reconstructed[is_air]).mean() if is_air.sum() > 0 else 0
    struct_acc = (original[is_struct] == reconstructed[is_struct]).mean() if is_struct.sum() > 0 else 0

    return {
        'overall': overall,
        'air_accuracy': air_acc,
        'structure_accuracy': struct_acc,
        'air_voxels': int(is_air.sum()),
        'structure_voxels': int(is_struct.sum()),
        'wrong_blocks': int((original[is_struct] != reconstructed[is_struct]).sum()),
    }


def structure_to_commands(structure, tok2block, origin, skip_air=True):
    """Convert structure to Minecraft setblock commands."""
    ox, oy, oz = origin
    commands = []

    for dz in range(structure.shape[0]):
        for dy in range(structure.shape[1]):
            for dx in range(structure.shape[2]):
                token = int(structure[dz, dy, dx])

                if skip_air and token in AIR_TOKENS:
                    continue

                block = tok2block.get(str(token), "minecraft:stone")
                if "[" in block:
                    block = block.split("[")[0]

                wx, wy, wz = ox + dx, oy + dy, oz + dz
                commands.append(f"setblock {wx} {wy} {wz} {block}")

    return commands


def main():
    parser = argparse.ArgumentParser(description="Place VQ-VAE reconstruction in Minecraft")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--h5-file", type=str, required=True, help="H5 file with build")
    parser.add_argument("--origin", type=int, nargs=3, default=[100, 64, 100], help="Origin X Y Z")
    parser.add_argument("--spacing", type=int, default=40, help="Spacing between builds")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory for command files")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    print("=" * 60)
    print("VQ-VAE Reconstruction Placer")
    print("=" * 60)

    # Load vocabulary
    print(f"\nLoading vocabulary from {VOCAB_PATH}...")
    with open(VOCAB_PATH, 'r') as f:
        tok2block = json.load(f)
    print(f"  {len(tok2block)} blocks")

    # Load embeddings
    print(f"Loading embeddings from {EMBEDDINGS_PATH}...")
    embeddings = torch.from_numpy(np.load(EMBEDDINGS_PATH).astype(np.float32))
    print(f"  Shape: {embeddings.shape}")

    # Load structure
    print(f"Loading structure from {args.h5_file}...")
    original = load_h5(args.h5_file)
    print(f"  Shape: {original.shape}")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, embeddings, device=args.device)
    print(f"  Device: {args.device}")

    # Reconstruct
    print("\nReconstructing...")
    reconstructed = reconstruct(model, original, device=args.device)

    # Compute metrics
    metrics = compute_metrics(original, reconstructed)

    print("\n" + "=" * 60)
    print("RECONSTRUCTION METRICS")
    print("=" * 60)
    print(f"  Overall accuracy:     {metrics['overall']:.2%}")
    print(f"  Air accuracy:         {metrics['air_accuracy']:.2%}")
    print(f"  Structure accuracy:   {metrics['structure_accuracy']:.2%}")
    print(f"  Air voxels:           {metrics['air_voxels']:,}")
    print(f"  Structure voxels:     {metrics['structure_voxels']:,}")
    print(f"  Wrong blocks:         {metrics['wrong_blocks']:,}")

    # Generate commands
    ox, oy, oz = args.origin
    rx = ox + args.spacing

    print("\nGenerating Minecraft commands...")

    orig_commands = structure_to_commands(original, tok2block, (ox, oy, oz))
    recon_commands = structure_to_commands(reconstructed, tok2block, (rx, oy, oz))

    print(f"  Original: {len(orig_commands):,} blocks at ({ox}, {oy}, {oz})")
    print(f"  Reconstructed: {len(recon_commands):,} blocks at ({rx}, {oy}, {oz})")

    # Write command files
    output_dir = Path(args.output_dir)

    # Clear commands
    clear_file = output_dir / "01_clear.txt"
    with open(clear_file, 'w') as f:
        f.write(f"fill {ox} {oy} {oz} {ox+31} {oy+31} {oz+31} minecraft:air\n")
        f.write(f"fill {rx} {oy} {oz} {rx+31} {oy+31} {oz+31} minecraft:air\n")

    # Original commands
    orig_file = output_dir / "02_original.txt"
    with open(orig_file, 'w') as f:
        f.write("\n".join(orig_commands))

    # Reconstruction commands
    recon_file = output_dir / "03_reconstructed.txt"
    with open(recon_file, 'w') as f:
        f.write("\n".join(recon_commands))

    # Teleport command
    tp_file = output_dir / "04_teleport.txt"
    with open(tp_file, 'w') as f:
        f.write(f"tp @p {ox + 16} {oy + 40} {oz - 30}\n")

    print(f"\nCommand files saved to {output_dir}/")
    print(f"  - 01_clear.txt")
    print(f"  - 02_original.txt ({len(orig_commands)} commands)")
    print(f"  - 03_reconstructed.txt ({len(recon_commands)} commands)")
    print(f"  - 04_teleport.txt")

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("Run these commands in order via MCP or command blocks:")
    print(f"  1. Clear area:  fill {ox} {oy} {oz} {ox+31} {oy+31} {oz+31} minecraft:air")
    print(f"  2. Clear area:  fill {rx} {oy} {oz} {rx+31} {oy+31} {oz+31} minecraft:air")
    print(f"  3. Run 02_original.txt commands")
    print(f"  4. Run 03_reconstructed.txt commands")
    print(f"  5. Teleport:    tp @p {ox + 16} {oy + 40} {oz - 30}")
    print()
    print("Original will be on the LEFT, Reconstruction on the RIGHT")


if __name__ == "__main__":
    main()
