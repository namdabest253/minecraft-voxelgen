"""
Visualize VQ-VAE Reconstructions in Minecraft (MCP Version).

This script generates Minecraft commands that can be executed via the MCP
minecraft tools or pasted into a command block.

Usage:
    python scripts/visualize_reconstruction_mcp.py \
        --checkpoint path/to/model.pt \
        --h5-file path/to/build.h5 \
        --origin 100 64 100 \
        --output commands.txt

Then use the MCP tools or command blocks to execute the commands.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.vqvae import VQVAEv4


# Default paths
DEFAULT_VOCAB_PATH = "data/kaggle/output/block2vec/v2/tok2block_collapsed.json"
DEFAULT_EMBEDDINGS_PATH = "data/kaggle/output/block2vec/v3/block_embeddings_v3.npy"

# Air tokens
AIR_TOKENS = {19, 164, 932}


def load_h5_structure(h5_path: str) -> np.ndarray:
    """Load a structure from an H5 file."""
    with h5py.File(h5_path, "r") as f:
        key = list(f.keys())[0]
        structure = f[key][:]
    return structure.astype(np.int64)


def load_vocabulary(vocab_path: str) -> Dict[str, str]:
    """Load token to block name mapping."""
    with open(vocab_path, "r") as f:
        return json.load(f)


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
    emb_tensor = None
    if pretrained_embeddings is not None:
        emb_tensor = torch.from_numpy(pretrained_embeddings).float()

    model = VQVAEv4(
        vocab_size=vocab_size,
        block_embedding_dim=block_embedding_dim,
        pretrained_embeddings=emb_tensor,
        train_embeddings=False,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

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
    """Reconstruct a structure using the VQ-VAE."""
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(structure).long().unsqueeze(0).to(device)
        logits, _, _, _ = model(x)
        predictions = logits.argmax(dim=1)
        reconstructed = predictions[0].cpu().numpy()
    return reconstructed


def compute_metrics(
    original: np.ndarray,
    reconstructed: np.ndarray,
) -> Dict[str, float]:
    """Compute reconstruction metrics."""
    is_air_orig = np.isin(original, list(AIR_TOKENS))
    is_struct = ~is_air_orig

    # Overall accuracy
    overall_acc = (original == reconstructed).mean()

    # Air accuracy
    if is_air_orig.sum() > 0:
        air_acc = (original[is_air_orig] == reconstructed[is_air_orig]).mean()
    else:
        air_acc = 0.0

    # Structure accuracy
    if is_struct.sum() > 0:
        struct_acc = (original[is_struct] == reconstructed[is_struct]).mean()
    else:
        struct_acc = 0.0

    # Count mismatches by type
    struct_orig = original[is_struct]
    struct_recon = reconstructed[is_struct]
    wrong_mask = struct_orig != struct_recon

    return {
        "overall_accuracy": float(overall_acc),
        "air_accuracy": float(air_acc),
        "structure_accuracy": float(struct_acc),
        "air_voxels": int(is_air_orig.sum()),
        "structure_voxels": int(is_struct.sum()),
        "wrong_structure_blocks": int(wrong_mask.sum()),
        "unique_original": int(len(np.unique(original))),
        "unique_reconstructed": int(len(np.unique(reconstructed))),
    }


def structure_to_commands(
    structure: np.ndarray,
    tok2block: Dict[str, str],
    origin: Tuple[int, int, int],
    skip_air: bool = True,
) -> List[str]:
    """Convert a structure to Minecraft setblock commands."""
    ox, oy, oz = origin
    commands = []

    for dz in range(structure.shape[0]):
        for dy in range(structure.shape[1]):
            for dx in range(structure.shape[2]):
                token_id = int(structure[dz, dy, dx])

                if skip_air and token_id in AIR_TOKENS:
                    continue

                block_name = tok2block.get(str(token_id), "minecraft:stone")

                # Remove block states for basic placement
                if "[" in block_name:
                    block_name = block_name.split("[")[0]

                wx = ox + dx
                wy = oy + dy
                wz = oz + dz

                commands.append(f"setblock {wx} {wy} {wz} {block_name}")

    return commands


def analyze_errors(
    original: np.ndarray,
    reconstructed: np.ndarray,
    tok2block: Dict[str, str],
    top_n: int = 20,
) -> Dict:
    """Analyze what types of errors the model makes."""
    is_air = np.isin(original, list(AIR_TOKENS))
    is_struct = ~is_air

    struct_orig = original[is_struct]
    struct_recon = reconstructed[is_struct]

    # Find mismatches
    wrong_mask = struct_orig != struct_recon
    wrong_orig = struct_orig[wrong_mask]
    wrong_recon = struct_recon[wrong_mask]

    # Count error pairs
    error_pairs = {}
    for orig_tok, recon_tok in zip(wrong_orig, wrong_recon):
        orig_name = tok2block.get(str(orig_tok), f"unknown_{orig_tok}")
        recon_name = tok2block.get(str(recon_tok), f"unknown_{recon_tok}")
        key = (orig_name, recon_name)
        error_pairs[key] = error_pairs.get(key, 0) + 1

    # Sort by frequency
    sorted_errors = sorted(error_pairs.items(), key=lambda x: -x[1])[:top_n]

    # Categorize errors
    same_material = 0
    different_material = 0

    for (orig, recon), count in error_pairs.items():
        # Extract material from block name (e.g., "minecraft:oak_planks" -> "oak")
        orig_parts = orig.replace("minecraft:", "").split("_")
        recon_parts = recon.replace("minecraft:", "").split("_")

        # Check if first word (usually material) matches
        if orig_parts[0] == recon_parts[0]:
            same_material += count
        else:
            different_material += count

    return {
        "top_errors": sorted_errors,
        "same_material_errors": same_material,
        "different_material_errors": different_material,
        "total_errors": wrong_mask.sum(),
    }


def print_analysis(
    original: np.ndarray,
    reconstructed: np.ndarray,
    tok2block: Dict[str, str],
    metrics: Dict,
):
    """Print detailed analysis of reconstruction."""
    print("\n" + "=" * 70)
    print("RECONSTRUCTION ANALYSIS")
    print("=" * 70)

    print("\n--- Metrics ---")
    print(f"  Overall accuracy:    {metrics['overall_accuracy']:.4f} ({metrics['overall_accuracy']*100:.2f}%)")
    print(f"  Air accuracy:        {metrics['air_accuracy']:.4f} ({metrics['air_accuracy']*100:.2f}%)")
    print(f"  Structure accuracy:  {metrics['structure_accuracy']:.4f} ({metrics['structure_accuracy']*100:.2f}%)")
    print(f"  Air voxels:          {metrics['air_voxels']:,}")
    print(f"  Structure voxels:    {metrics['structure_voxels']:,}")
    print(f"  Wrong blocks:        {metrics['wrong_structure_blocks']:,}")

    # Error analysis
    errors = analyze_errors(original, reconstructed, tok2block)

    print("\n--- Error Breakdown ---")
    print(f"  Same material errors:      {errors['same_material_errors']:,} ({100*errors['same_material_errors']/max(1,errors['total_errors']):.1f}%)")
    print(f"  Different material errors: {errors['different_material_errors']:,} ({100*errors['different_material_errors']/max(1,errors['total_errors']):.1f}%)")

    print("\n--- Top 15 Error Pairs (original -> reconstructed) ---")
    for (orig, recon), count in errors["top_errors"][:15]:
        orig_short = orig.replace("minecraft:", "")
        recon_short = recon.replace("minecraft:", "")
        print(f"  {orig_short:30} -> {recon_short:30} : {count:,}")

    # Block distribution
    print("\n--- Original Block Distribution (top 10) ---")
    unique_orig, counts_orig = np.unique(original, return_counts=True)
    sorted_idx = np.argsort(-counts_orig)
    for i in sorted_idx[:10]:
        token = unique_orig[i]
        count = counts_orig[i]
        name = tok2block.get(str(token), f"unknown_{token}").replace("minecraft:", "")
        pct = 100 * count / original.size
        print(f"  {name:40} : {count:>8,} ({pct:>5.1f}%)")

    print("\n--- Reconstructed Block Distribution (top 10) ---")
    unique_recon, counts_recon = np.unique(reconstructed, return_counts=True)
    sorted_idx = np.argsort(-counts_recon)
    for i in sorted_idx[:10]:
        token = unique_recon[i]
        count = counts_recon[i]
        name = tok2block.get(str(token), f"unknown_{token}").replace("minecraft:", "")
        pct = 100 * count / reconstructed.size
        print(f"  {name:40} : {count:>8,} ({pct:>5.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Visualize VQ-VAE reconstructions")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--h5-file", type=str, required=True, help="Path to H5 build file")
    parser.add_argument("--origin", type=int, nargs=3, default=[100, 64, 100], help="Origin X Y Z")
    parser.add_argument("--spacing", type=int, default=40, help="Spacing between builds")
    parser.add_argument("--output", type=str, default="reconstruction_commands.txt", help="Output commands file")
    parser.add_argument("--vocab-path", type=str, default=None)
    parser.add_argument("--embeddings-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--analysis-only", action="store_true", help="Only print analysis, don't generate commands")

    args = parser.parse_args()

    # Resolve paths
    project_root = Path(__file__).parent.parent
    vocab_path = args.vocab_path or str(project_root / DEFAULT_VOCAB_PATH)
    embeddings_path = args.embeddings_path or str(project_root / DEFAULT_EMBEDDINGS_PATH)

    print("Loading vocabulary...")
    tok2block = load_vocabulary(vocab_path)

    print("Loading embeddings...")
    embeddings = load_embeddings(embeddings_path)

    print(f"Loading structure from {args.h5_file}...")
    original = load_h5_structure(args.h5_file)
    print(f"  Shape: {original.shape}")

    print(f"Loading model from {args.checkpoint}...")
    model = load_model(
        args.checkpoint,
        vocab_size=len(tok2block),
        pretrained_embeddings=embeddings,
        device=args.device,
    )

    print("Reconstructing...")
    reconstructed = reconstruct_structure(model, original, device=args.device)

    # Compute and print metrics
    metrics = compute_metrics(original, reconstructed)
    print_analysis(original, reconstructed, tok2block, metrics)

    if not args.analysis_only:
        # Generate commands
        ox, oy, oz = args.origin

        print(f"\nGenerating commands...")

        # Original commands
        orig_commands = structure_to_commands(original, tok2block, (ox, oy, oz))

        # Reconstructed commands
        recon_commands = structure_to_commands(
            reconstructed, tok2block, (ox + args.spacing, oy, oz)
        )

        # Clear commands
        clear_commands = [
            f"fill {ox} {oy} {oz} {ox+31} {oy+31} {oz+31} minecraft:air",
            f"fill {ox+args.spacing} {oy} {oz} {ox+args.spacing+31} {oy+31} {oz+31} minecraft:air",
        ]

        # Write to file
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            f.write("# Clear areas first\n")
            for cmd in clear_commands:
                f.write(cmd + "\n")

            f.write(f"\n# Original structure at ({ox}, {oy}, {oz})\n")
            for cmd in orig_commands:
                f.write(cmd + "\n")

            f.write(f"\n# Reconstructed structure at ({ox+args.spacing}, {oy}, {oz})\n")
            for cmd in recon_commands:
                f.write(cmd + "\n")

            f.write(f"\n# Teleport to view\n")
            f.write(f"tp @p {ox + 16} {oy + 40} {oz - 30}\n")

        print(f"\nCommands written to {output_path}")
        print(f"  Original: {len(orig_commands):,} blocks")
        print(f"  Reconstructed: {len(recon_commands):,} blocks")

        # Also write a JSON summary
        summary_path = output_path.with_suffix(".json")
        summary = {
            "h5_file": str(args.h5_file),
            "checkpoint": str(args.checkpoint),
            "origin": list(args.origin),
            "spacing": args.spacing,
            "metrics": metrics,
            "original_blocks": len(orig_commands),
            "reconstructed_blocks": len(recon_commands),
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary written to {summary_path}")

        print(f"\n--- Next Steps ---")
        print(f"1. Use the Minecraft MCP tools to run these commands:")
        print(f"   - Run the 'fill' commands first to clear the area")
        print(f"   - Then run the setblock commands in batches")
        print(f"2. Or copy commands into a .mcfunction file")
        print(f"3. Teleport to view: /tp @p {ox + 16} {oy + 40} {oz - 30}")


if __name__ == "__main__":
    main()
