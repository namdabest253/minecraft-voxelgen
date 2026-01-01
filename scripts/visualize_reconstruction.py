"""
Visualize VQ-VAE Reconstructions in Minecraft.

This script loads a trained VQ-VAE model, reconstructs a build from an H5 file,
and places both the original and reconstruction in a Minecraft server for
side-by-side visual comparison.

Usage:
    python scripts/visualize_reconstruction.py \
        --checkpoint path/to/model.pt \
        --h5-file path/to/build.h5 \
        --origin 100 64 100 \
        --spacing 40

Requirements:
    - Trained VQ-VAE v4 checkpoint
    - H5 file containing a 32x32x32 build
    - Minecraft server running with RCON enabled
    - tok2block vocabulary mapping
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.vqvae import VQVAEv4


# Default paths (relative to project root)
DEFAULT_VOCAB_PATH = "data/kaggle/output/block2vec/v2/tok2block_collapsed.json"
DEFAULT_EMBEDDINGS_PATH = "data/kaggle/output/block2vec/v3/block_embeddings_v3.npy"

# Air tokens to skip when placing blocks
AIR_TOKENS = {19, 164, 932}  # air, cave_air, void_air

# RCON configuration
RCON_HOST = "localhost"
RCON_PORT = 25575
RCON_PASSWORD = "minecraft"  # Default, change if different


class MinecraftPlacer:
    """Places blocks in Minecraft via RCON commands."""

    def __init__(self, host: str = RCON_HOST, port: int = RCON_PORT, password: str = RCON_PASSWORD):
        self.host = host
        self.port = port
        self.password = password
        self._rcon = None

    def connect(self):
        """Connect to RCON server."""
        try:
            from mcrcon import MCRcon
            self._rcon = MCRcon(self.host, self.password, port=self.port)
            self._rcon.connect()
            print(f"Connected to Minecraft RCON at {self.host}:{self.port}")
            return True
        except ImportError:
            print("mcrcon not installed. Install with: pip install mcrcon")
            return False
        except Exception as e:
            print(f"Failed to connect to RCON: {e}")
            return False

    def disconnect(self):
        """Disconnect from RCON server."""
        if self._rcon:
            self._rcon.disconnect()
            self._rcon = None

    def run_command(self, command: str) -> str:
        """Run a single Minecraft command."""
        if not self._rcon:
            raise RuntimeError("Not connected to RCON")
        return self._rcon.command(command)

    def clear_area(self, x: int, y: int, z: int, size: int = 32):
        """Clear an area by filling with air."""
        x2, y2, z2 = x + size - 1, y + size - 1, z + size - 1
        cmd = f"fill {x} {y} {z} {x2} {y2} {z2} minecraft:air"
        return self.run_command(cmd)

    def place_structure(
        self,
        structure: np.ndarray,
        tok2block: Dict[int, str],
        origin: Tuple[int, int, int],
        skip_air: bool = True,
        batch_size: int = 100,
    ) -> Tuple[int, int]:
        """
        Place a 3D structure in Minecraft.

        Args:
            structure: 3D numpy array of token IDs [D, H, W]
            tok2block: Mapping from token ID to block name
            origin: (x, y, z) coordinates for placement origin
            skip_air: If True, don't place air blocks (faster)
            batch_size: Number of blocks to place before small delay

        Returns:
            Tuple of (blocks_placed, blocks_skipped)
        """
        ox, oy, oz = origin
        placed = 0
        skipped = 0
        commands = []

        # Iterate through structure
        # Structure is [depth, height, width] -> map to [z, y, x] in Minecraft
        for dz in range(structure.shape[0]):
            for dy in range(structure.shape[1]):
                for dx in range(structure.shape[2]):
                    token_id = int(structure[dz, dy, dx])

                    # Skip air blocks if requested
                    if skip_air and token_id in AIR_TOKENS:
                        skipped += 1
                        continue

                    # Get block name
                    block_name = tok2block.get(str(token_id), "minecraft:stone")

                    # Clean block name (remove state info for placement)
                    # e.g., "minecraft:oak_stairs[facing=east]" -> "minecraft:oak_stairs"
                    if "[" in block_name:
                        block_name = block_name.split("[")[0]

                    # Calculate world position
                    wx = ox + dx
                    wy = oy + dy
                    wz = oz + dz

                    cmd = f"setblock {wx} {wy} {wz} {block_name}"
                    commands.append(cmd)
                    placed += 1

                    # Execute in batches
                    if len(commands) >= batch_size:
                        for c in commands:
                            try:
                                self.run_command(c)
                            except Exception as e:
                                pass  # Ignore individual block errors
                        commands = []
                        time.sleep(0.05)  # Small delay to avoid overwhelming server

        # Execute remaining commands
        for c in commands:
            try:
                self.run_command(c)
            except Exception:
                pass

        return placed, skipped


def load_h5_structure(h5_path: str) -> np.ndarray:
    """Load a structure from an H5 file."""
    with h5py.File(h5_path, "r") as f:
        key = list(f.keys())[0]
        structure = f[key][:]
    return structure.astype(np.int64)


def load_vocabulary(vocab_path: str) -> Dict[int, str]:
    """Load token to block name mapping."""
    with open(vocab_path, "r") as f:
        tok2block = json.load(f)
    return tok2block


def load_embeddings(embeddings_path: str) -> np.ndarray:
    """Load pretrained block embeddings."""
    return np.load(embeddings_path)


def load_model(
    checkpoint_path: str,
    vocab_size: int = 3717,
    block_embedding_dim: int = 32,
    pretrained_embeddings: Optional[np.ndarray] = None,
    device: str = "cuda",
) -> VQVAEv4:
    """Load a trained VQVAEv4 model from checkpoint."""
    # Convert embeddings to tensor if provided
    emb_tensor = None
    if pretrained_embeddings is not None:
        emb_tensor = torch.from_numpy(pretrained_embeddings).float()

    # Create model
    model = VQVAEv4(
        vocab_size=vocab_size,
        block_embedding_dim=block_embedding_dim,
        pretrained_embeddings=emb_tensor,
        train_embeddings=False,  # Not training, just inference
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model


def reconstruct_structure(
    model: VQVAEv4,
    structure: np.ndarray,
    device: str = "cuda",
) -> np.ndarray:
    """
    Reconstruct a structure using the VQ-VAE.

    Args:
        model: Trained VQVAEv4 model
        structure: Original structure [D, H, W]
        device: Device to run inference on

    Returns:
        Reconstructed structure [D, H, W]
    """
    model.eval()

    with torch.no_grad():
        # Add batch dimension and move to device
        x = torch.from_numpy(structure).long().unsqueeze(0).to(device)

        # Forward pass
        logits, _, _, _ = model(x)

        # Get predictions
        predictions = logits.argmax(dim=1)  # [1, D, H, W]

        # Remove batch dimension
        reconstructed = predictions[0].cpu().numpy()

    return reconstructed


def compute_metrics(
    original: np.ndarray,
    reconstructed: np.ndarray,
    air_tokens: set = AIR_TOKENS,
) -> Dict[str, float]:
    """Compute reconstruction metrics."""
    # Overall accuracy
    total_correct = (original == reconstructed).sum()
    total_voxels = original.size
    overall_acc = total_correct / total_voxels

    # Air accuracy
    is_air_orig = np.isin(original, list(air_tokens))
    air_correct = (original[is_air_orig] == reconstructed[is_air_orig]).sum()
    air_total = is_air_orig.sum()
    air_acc = air_correct / air_total if air_total > 0 else 0.0

    # Structure (non-air) accuracy
    is_struct = ~is_air_orig
    struct_correct = (original[is_struct] == reconstructed[is_struct]).sum()
    struct_total = is_struct.sum()
    struct_acc = struct_correct / struct_total if struct_total > 0 else 0.0

    # Block type statistics
    unique_orig = np.unique(original)
    unique_recon = np.unique(reconstructed)

    return {
        "overall_accuracy": float(overall_acc),
        "air_accuracy": float(air_acc),
        "structure_accuracy": float(struct_acc),
        "air_voxels": int(air_total),
        "structure_voxels": int(struct_total),
        "unique_blocks_original": int(len(unique_orig)),
        "unique_blocks_reconstructed": int(len(unique_recon)),
    }


def print_structure_summary(
    structure: np.ndarray,
    tok2block: Dict[int, str],
    name: str = "Structure",
    top_n: int = 10,
):
    """Print a summary of block types in a structure."""
    unique, counts = np.unique(structure, return_counts=True)
    sorted_indices = np.argsort(-counts)

    print(f"\n{name} Block Distribution (top {top_n}):")
    print("-" * 50)
    for i in sorted_indices[:top_n]:
        token_id = unique[i]
        count = counts[i]
        block_name = tok2block.get(str(token_id), f"unknown_{token_id}")
        pct = 100 * count / structure.size
        print(f"  {block_name}: {count:,} ({pct:.1f}%)")


def generate_commands_file(
    structure: np.ndarray,
    tok2block: Dict[int, str],
    origin: Tuple[int, int, int],
    output_path: str,
    skip_air: bool = True,
):
    """Generate a file of setblock commands for manual execution."""
    ox, oy, oz = origin
    commands = []

    for dz in range(structure.shape[0]):
        for dy in range(structure.shape[1]):
            for dx in range(structure.shape[2]):
                token_id = int(structure[dz, dy, dx])

                if skip_air and token_id in AIR_TOKENS:
                    continue

                block_name = tok2block.get(str(token_id), "minecraft:stone")
                if "[" in block_name:
                    block_name = block_name.split("[")[0]

                wx = ox + dx
                wy = oy + dy
                wz = oz + dz

                commands.append(f"setblock {wx} {wy} {wz} {block_name}")

    with open(output_path, "w") as f:
        f.write("\n".join(commands))

    print(f"Generated {len(commands)} commands to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize VQ-VAE reconstructions in Minecraft"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--h5-file",
        type=str,
        required=True,
        help="Path to H5 file containing the build to reconstruct",
    )
    parser.add_argument(
        "--origin",
        type=int,
        nargs=3,
        default=[100, 64, 100],
        metavar=("X", "Y", "Z"),
        help="Origin coordinates for placing the original structure",
    )
    parser.add_argument(
        "--spacing",
        type=int,
        default=40,
        help="Spacing between original and reconstruction (default: 40)",
    )
    parser.add_argument(
        "--vocab-path",
        type=str,
        default=None,
        help="Path to tok2block vocabulary JSON",
    )
    parser.add_argument(
        "--embeddings-path",
        type=str,
        default=None,
        help="Path to pretrained embeddings .npy file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on",
    )
    parser.add_argument(
        "--no-place",
        action="store_true",
        help="Don't place in Minecraft, just compute metrics",
    )
    parser.add_argument(
        "--commands-file",
        type=str,
        default=None,
        help="Output file for setblock commands (alternative to RCON)",
    )
    parser.add_argument(
        "--clear-area",
        action="store_true",
        help="Clear the area before placing structures",
    )

    args = parser.parse_args()

    # Resolve paths
    project_root = Path(__file__).parent.parent
    vocab_path = args.vocab_path or str(project_root / DEFAULT_VOCAB_PATH)
    embeddings_path = args.embeddings_path or str(project_root / DEFAULT_EMBEDDINGS_PATH)

    print("=" * 60)
    print("VQ-VAE Reconstruction Visualizer")
    print("=" * 60)

    # Load vocabulary
    print(f"\nLoading vocabulary from {vocab_path}...")
    tok2block = load_vocabulary(vocab_path)
    print(f"  Loaded {len(tok2block)} block mappings")

    # Load embeddings
    print(f"\nLoading embeddings from {embeddings_path}...")
    embeddings = load_embeddings(embeddings_path)
    print(f"  Loaded embeddings with shape {embeddings.shape}")

    # Load structure
    print(f"\nLoading structure from {args.h5_file}...")
    original = load_h5_structure(args.h5_file)
    print(f"  Structure shape: {original.shape}")

    # Print original structure summary
    print_structure_summary(original, tok2block, "Original", top_n=10)

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = load_model(
        args.checkpoint,
        vocab_size=len(tok2block),
        pretrained_embeddings=embeddings,
        device=args.device,
    )
    print(f"  Model loaded on {args.device}")

    # Reconstruct
    print("\nReconstructing structure...")
    reconstructed = reconstruct_structure(model, original, device=args.device)

    # Print reconstructed structure summary
    print_structure_summary(reconstructed, tok2block, "Reconstructed", top_n=10)

    # Compute metrics
    print("\n" + "=" * 60)
    print("Reconstruction Metrics")
    print("=" * 60)
    metrics = compute_metrics(original, reconstructed)
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"  {name}: {value:.4f} ({value*100:.2f}%)")
        else:
            print(f"  {name}: {value:,}")

    # Generate commands file if requested
    if args.commands_file:
        ox, oy, oz = args.origin
        # Original
        orig_commands = args.commands_file.replace(".txt", "_original.txt")
        generate_commands_file(original, tok2block, (ox, oy, oz), orig_commands)
        # Reconstructed
        recon_commands = args.commands_file.replace(".txt", "_reconstructed.txt")
        generate_commands_file(
            reconstructed, tok2block, (ox + args.spacing, oy, oz), recon_commands
        )

    # Place in Minecraft if requested
    if not args.no_place:
        print("\n" + "=" * 60)
        print("Placing in Minecraft")
        print("=" * 60)

        placer = MinecraftPlacer()
        if placer.connect():
            try:
                ox, oy, oz = args.origin

                # Clear areas if requested
                if args.clear_area:
                    print("\nClearing areas...")
                    placer.clear_area(ox, oy, oz, 32)
                    placer.clear_area(ox + args.spacing, oy, oz, 32)
                    time.sleep(0.5)

                # Place original
                print(f"\nPlacing original at ({ox}, {oy}, {oz})...")
                placed, skipped = placer.place_structure(
                    original, tok2block, (ox, oy, oz)
                )
                print(f"  Placed {placed:,} blocks, skipped {skipped:,} air blocks")

                # Place reconstruction
                rx = ox + args.spacing
                print(f"\nPlacing reconstruction at ({rx}, {oy}, {oz})...")
                placed, skipped = placer.place_structure(
                    reconstructed, tok2block, (rx, oy, oz)
                )
                print(f"  Placed {placed:,} blocks, skipped {skipped:,} air blocks")

                # Add labels
                print("\nAdding labels...")
                # Place signs or use /title command
                placer.run_command(f"summon armor_stand {ox} {oy + 35} {oz} {{CustomName:'\"ORIGINAL\"',CustomNameVisible:1b,Invisible:1b,NoGravity:1b}}")
                placer.run_command(f"summon armor_stand {rx} {oy + 35} {oz} {{CustomName:'\"RECONSTRUCTED\"',CustomNameVisible:1b,Invisible:1b,NoGravity:1b}}")

                print("\nDone! Teleport to the location to view:")
                print(f"  /tp @p {ox + 16} {oy + 40} {oz - 20}")

            finally:
                placer.disconnect()
        else:
            print("Could not connect to Minecraft. Use --commands-file to generate commands instead.")

    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
