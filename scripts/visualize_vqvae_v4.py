"""
Visualize VQ-VAE v4 Reconstructions in Minecraft.

This script loads a v4 checkpoint, reconstructs a build, and generates
commands to place both original and reconstruction side-by-side.

Usage:
    python scripts/visualize_vqvae_v4.py --sample 0
    python scripts/visualize_vqvae_v4.py --h5-file path/to/build.h5
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Model Architecture (matches training notebook exactly)
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
        layers = [
            ResidualBlock3D(latent_dim),
            ResidualBlock3D(latent_dim),
        ]

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
    """VQ-VAE v4 model for inference."""

    def __init__(self, vocab_size, emb_dim, hidden_dims, latent_dim, num_codes,
                 pretrained_emb, dropout=0.1, commitment_cost=0.5, ema_decay=0.99,
                 **kwargs):  # Accept extra kwargs for compatibility
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim

        # Embeddings
        self.block_emb = nn.Embedding(vocab_size, emb_dim)
        if pretrained_emb is not None:
            self.block_emb.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.register_buffer('original_embeddings',
                           torch.from_numpy(pretrained_emb.copy()) if pretrained_emb is not None
                           else torch.zeros(vocab_size, emb_dim))

        # Encoder
        self.encoder = EncoderV4(emb_dim, hidden_dims, latent_dim, dropout)

        # Quantizer
        self.quantizer = VectorQuantizerEMA(num_codes, latent_dim, commitment_cost, ema_decay)

        # Decoder
        self.decoder = DecoderV4(latent_dim, list(reversed(hidden_dims)), vocab_size, dropout)

    def forward(self, block_ids):
        x = self.block_emb(block_ids).permute(0, 4, 1, 2, 3).contiguous()
        z_e = self.encoder(x)
        z_q, vq_loss, indices = self.quantizer(z_e)
        logits = self.decoder(z_q)
        return {'logits': logits, 'vq_loss': vq_loss, 'indices': indices}


# ============================================================
# Utility Functions
# ============================================================

# CORRECT air tokens for this vocabulary (v1)
# Token 102: minecraft:air
# Token 576: minecraft:cave_air
# Token 3352: minecraft:void_air
AIR_TOKENS = {102, 576, 3352}


def load_checkpoint(checkpoint_path: str, device: str = "cuda"):
    """Load a v4 checkpoint and create model."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    print(f"Loading model with config:")
    print(f"  vocab_size: {config['vocab_size']}")
    print(f"  emb_dim: {config['emb_dim']}")
    print(f"  hidden_dims: {config['hidden_dims']}")
    print(f"  latent_dim: {config['latent_dim']}")
    print(f"  num_codes: {config['num_codes']}")

    # Create dummy embeddings (will be overwritten by checkpoint)
    dummy_emb = np.zeros((config['vocab_size'], config['emb_dim']), dtype=np.float32)

    model = VQVAEv4(
        vocab_size=config['vocab_size'],
        emb_dim=config['emb_dim'],
        hidden_dims=config['hidden_dims'],
        latent_dim=config['latent_dim'],
        num_codes=config['num_codes'],
        pretrained_emb=dummy_emb,
        dropout=config.get('dropout', 0.1),
        commitment_cost=config.get('commitment_cost', 0.5),
        ema_decay=config.get('ema_decay', 0.99),
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, checkpoint


def load_vocabulary(vocab_path: str) -> Dict[str, str]:
    """Load token to block name mapping."""
    with open(vocab_path, "r") as f:
        tok2block = json.load(f)
    # Convert keys to int for lookup
    return {int(k): v for k, v in tok2block.items()}


def load_h5_structure(h5_path: str) -> np.ndarray:
    """Load a structure from an H5 file."""
    with h5py.File(h5_path, "r") as f:
        key = list(f.keys())[0]
        structure = f[key][:]
    return structure.astype(np.int64)


def reconstruct(model, structure: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Reconstruct a structure using the VQ-VAE."""
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(structure).long().unsqueeze(0).to(device)
        out = model(x)
        predictions = out['logits'].argmax(dim=1)
        reconstructed = predictions[0].cpu().numpy()
    return reconstructed


def compute_metrics(original: np.ndarray, reconstructed: np.ndarray) -> Dict:
    """Compute reconstruction metrics."""
    is_air_orig = np.isin(original, list(AIR_TOKENS))
    is_struct_orig = ~is_air_orig

    is_air_recon = np.isin(reconstructed, list(AIR_TOKENS))
    is_struct_recon = ~is_air_recon

    # Structure recall
    struct_preserved = is_struct_orig & is_struct_recon
    struct_recall = struct_preserved.sum() / max(is_struct_orig.sum(), 1)

    # False air rate
    false_air = is_struct_orig & is_air_recon
    false_air_rate = false_air.sum() / max(is_struct_orig.sum(), 1)

    # Volume ratio
    vol_ratio = is_struct_recon.sum() / max(is_struct_orig.sum(), 1)

    # Exact accuracy
    exact_acc = (original[is_struct_orig] == reconstructed[is_struct_orig]).mean() if is_struct_orig.any() else 0

    return {
        'struct_recall': float(struct_recall),
        'false_air_rate': float(false_air_rate),
        'vol_ratio': float(vol_ratio),
        'exact_acc': float(exact_acc),
        'original_blocks': int(is_struct_orig.sum()),
        'reconstructed_blocks': int(is_struct_recon.sum()),
        'erased_blocks': int(false_air.sum()),
    }


def structure_to_commands(
    structure: np.ndarray,
    tok2block: Dict[int, str],
    origin: Tuple[int, int, int],
    skip_air: bool = True,
) -> List[str]:
    """Convert a structure to Minecraft setblock commands."""
    ox, oy, oz = origin
    commands = []

    # Structure is [Y, Z, X] or [Z, Y, X] - need to check
    # Assuming [Y, Z, X] based on typical H5 format
    for y in range(structure.shape[0]):
        for z in range(structure.shape[1]):
            for x in range(structure.shape[2]):
                token_id = int(structure[y, z, x])

                if skip_air and token_id in AIR_TOKENS:
                    continue

                block_name = tok2block.get(token_id, "minecraft:stone")

                # Remove block states for basic placement
                if "[" in block_name:
                    block_name = block_name.split("[")[0]

                wx = ox + x
                wy = oy + y
                wz = oz + z

                commands.append(f"setblock {wx} {wy} {wz} {block_name}")

    return commands


def print_analysis(original: np.ndarray, reconstructed: np.ndarray,
                   tok2block: Dict[int, str], metrics: Dict):
    """Print detailed analysis."""
    print("\n" + "=" * 70)
    print("VQ-VAE v4 RECONSTRUCTION ANALYSIS")
    print("=" * 70)

    print(f"\n--- Shape Preservation Metrics ---")
    print(f"  Structure Recall:    {metrics['struct_recall']:.1%} (target: >90%)")
    print(f"  False Air Rate:      {metrics['false_air_rate']:.1%} (target: <10%)")
    print(f"  Volume Ratio:        {metrics['vol_ratio']:.2f} (target: ~1.0)")
    print(f"  Exact Accuracy:      {metrics['exact_acc']:.1%}")

    print(f"\n--- Block Counts ---")
    print(f"  Original structure:    {metrics['original_blocks']:,} blocks")
    print(f"  Reconstructed:         {metrics['reconstructed_blocks']:,} blocks")
    print(f"  Erased (struct->air):  {metrics['erased_blocks']:,} blocks")

    # Error analysis
    is_struct = ~np.isin(original, list(AIR_TOKENS))
    struct_orig = original[is_struct]
    struct_recon = reconstructed[is_struct]

    wrong_mask = struct_orig != struct_recon
    if wrong_mask.any():
        wrong_orig = struct_orig[wrong_mask]
        wrong_recon = struct_recon[wrong_mask]

        # Count errors
        error_pairs = Counter(zip(wrong_orig, wrong_recon))

        print(f"\n--- Top 10 Error Pairs (original -> reconstructed) ---")
        for (orig_tok, recon_tok), count in error_pairs.most_common(10):
            orig_name = tok2block.get(orig_tok, f"unknown_{orig_tok}").replace("minecraft:", "")
            recon_name = tok2block.get(recon_tok, f"unknown_{recon_tok}").replace("minecraft:", "")
            print(f"  {orig_name:30} -> {recon_name:30} : {count:,}")


def main():
    parser = argparse.ArgumentParser(description="Visualize VQ-VAE v4 reconstructions")
    parser.add_argument("--checkpoint", type=str,
                       default="data/kaggle/output/vqvae/v4/vqvae_v4_best_checkpoint.pt")
    parser.add_argument("--h5-file", type=str, default=None, help="Path to H5 build file")
    parser.add_argument("--sample", type=int, default=0, help="Validation sample index")
    parser.add_argument("--origin", type=int, nargs=3, default=[100, 64, 100], help="Origin X Y Z")
    parser.add_argument("--spacing", type=int, default=40, help="Spacing between builds")
    parser.add_argument("--output", type=str, default="reconstruction_commands.txt")
    parser.add_argument("--vocab-path", type=str,
                       default="data/kaggle/output/block2vec/v2/tok2block_collapsed.json")
    parser.add_argument("--val-dir", type=str, default="data/splits/val")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--analysis-only", action="store_true")

    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = PROJECT_ROOT
    checkpoint_path = project_root / args.checkpoint
    vocab_path = project_root / args.vocab_path

    # Load vocabulary
    print("Loading vocabulary...")
    tok2block = load_vocabulary(str(vocab_path))
    print(f"  {len(tok2block)} blocks")

    # Load model
    print(f"\nLoading model from {checkpoint_path}...")
    model, checkpoint = load_checkpoint(str(checkpoint_path), args.device)
    print(f"  Best epoch: {checkpoint.get('best_epoch', 'N/A')}")
    print(f"  Best struct_recall: {checkpoint.get('best_struct_recall', 'N/A'):.1%}")

    # Load structure
    if args.h5_file:
        h5_path = Path(args.h5_file)
    else:
        val_dir = project_root / args.val_dir
        h5_files = sorted(val_dir.glob("*.h5"))
        if not h5_files:
            print(f"No H5 files found in {val_dir}")
            return
        h5_path = h5_files[args.sample % len(h5_files)]

    print(f"\nLoading structure from {h5_path.name}...")
    original = load_h5_structure(str(h5_path))
    print(f"  Shape: {original.shape}")

    # Reconstruct
    print("\nReconstructing...")
    reconstructed = reconstruct(model, original, args.device)

    # Compute metrics
    metrics = compute_metrics(original, reconstructed)
    print_analysis(original, reconstructed, tok2block, metrics)

    if args.analysis_only:
        return

    # Generate commands
    ox, oy, oz = args.origin

    print(f"\nGenerating commands...")

    # Clear area commands
    clear_commands = [
        f"fill {ox} {oy} {oz} {ox+31} {oy+31} {oz+31} minecraft:air",
        f"fill {ox+args.spacing} {oy} {oz} {ox+args.spacing+31} {oy+31} {oz+31} minecraft:air",
    ]

    # Structure commands
    orig_commands = structure_to_commands(original, tok2block, (ox, oy, oz))
    recon_commands = structure_to_commands(reconstructed, tok2block, (ox + args.spacing, oy, oz))

    # Labels
    label_commands = [
        f"setblock {ox+16} {oy-1} {oz-2} minecraft:oak_sign[rotation=0]{{Text1:'\"ORIGINAL\"'}}",
        f"setblock {ox+args.spacing+16} {oy-1} {oz-2} minecraft:oak_sign[rotation=0]{{Text1:'\"RECONSTRUCTED\"'}}",
    ]

    # Write to file
    output_path = project_root / args.output
    with open(output_path, "w") as f:
        f.write(f"# VQ-VAE v4 Reconstruction Visualization\n")
        f.write(f"# Source: {h5_path.name}\n")
        f.write(f"# Metrics: recall={metrics['struct_recall']:.1%}, ")
        f.write(f"false_air={metrics['false_air_rate']:.1%}, ")
        f.write(f"vol_ratio={metrics['vol_ratio']:.2f}\n\n")

        f.write("# Step 1: Clear areas\n")
        for cmd in clear_commands:
            f.write(cmd + "\n")

        f.write(f"\n# Step 2: Place original at ({ox}, {oy}, {oz})\n")
        for cmd in orig_commands:
            f.write(cmd + "\n")

        f.write(f"\n# Step 3: Place reconstruction at ({ox+args.spacing}, {oy}, {oz})\n")
        for cmd in recon_commands:
            f.write(cmd + "\n")

        f.write(f"\n# Step 4: Teleport to view\n")
        f.write(f"tp @p {ox + args.spacing//2} {oy + 40} {oz - 50}\n")

    print(f"\nCommands written to {output_path}")
    print(f"  Original: {len(orig_commands):,} blocks")
    print(f"  Reconstructed: {len(recon_commands):,} blocks")

    print(f"\n--- To place in Minecraft ---")
    print(f"The commands have been saved to: {output_path}")
    print(f"You can run them using the MCP tools or use the companion script.")


if __name__ == "__main__":
    main()
