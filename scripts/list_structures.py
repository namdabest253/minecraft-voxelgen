"""
List and preview h5 structures.

Usage:
    python list_structures.py                    # List all structures
    python list_structures.py --preview 5        # Show details of first 5
    python list_structures.py --search oak       # Find structures with oak blocks
    python list_structures.py --random 3         # Pick 3 random structures
"""

import argparse
import json
import random
from pathlib import Path

import h5py
import numpy as np


def load_vocabulary(vocab_path: Path) -> dict[int, str]:
    """Load token to block name mapping."""
    with open(vocab_path, "r") as f:
        return {int(k): v for k, v in json.load(f).items()}


def analyze_structure(h5_path: Path, tok2block: dict[int, str]) -> dict:
    """Analyze a structure and return summary."""
    with h5py.File(h5_path, "r") as f:
        key = list(f.keys())[0]
        build = f[key][:]

    # Find air tokens
    air_tokens = [tok for tok, name in tok2block.items() if "air" in name.lower() and "airs" not in name.lower()]

    # Count blocks
    unique, counts = np.unique(build, return_counts=True)
    block_counts = dict(zip(unique, counts))

    # Non-air count
    non_air = sum(c for tok, c in block_counts.items() if tok not in air_tokens)

    # Top blocks (non-air)
    top_blocks = []
    for tok, count in sorted(block_counts.items(), key=lambda x: -x[1]):
        if tok not in air_tokens:
            name = tok2block.get(tok, "unknown").replace("minecraft:", "")
            # Remove block states for cleaner display
            if "[" in name:
                name = name.split("[")[0] + "[...]"
            top_blocks.append((name, count))
            if len(top_blocks) >= 5:
                break

    return {
        "path": h5_path,
        "name": h5_path.name,
        "shape": build.shape,
        "non_air": non_air,
        "unique_blocks": len(unique),
        "top_blocks": top_blocks,
    }


def print_structure_info(info: dict, detailed: bool = False):
    """Print structure information."""
    print(f"\n{info['name']}:")
    print(f"  Shape: {info['shape']}")
    print(f"  Non-air blocks: {info['non_air']}")
    print(f"  Unique block types: {info['unique_blocks']}")

    if detailed:
        print(f"  Top blocks:")
        for name, count in info['top_blocks']:
            print(f"    {name}: {count}")
        print(f"  Path: {info['path']}")


def main():
    parser = argparse.ArgumentParser(description="List and preview h5 structures")
    parser.add_argument("--data-dir", type=Path, default=None, help="Data directory")
    parser.add_argument("--preview", type=int, default=0, help="Show detailed preview of N structures")
    parser.add_argument("--search", type=str, default=None, help="Search for structures containing block type")
    parser.add_argument("--random", type=int, default=0, help="Pick N random structures")
    parser.add_argument("--min-blocks", type=int, default=0, help="Minimum non-air blocks")
    parser.add_argument("--max-blocks", type=int, default=999999, help="Maximum non-air blocks")
    parser.add_argument("--vocab", type=Path, default=None, help="Path to vocabulary file")

    args = parser.parse_args()

    # Find data directory
    script_dir = Path(__file__).parent.parent
    if args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = script_dir / "data" / "splits" / "train"

    # Find vocabulary
    if args.vocab:
        vocab_path = args.vocab
    else:
        vocab_path = script_dir / "data" / "vocabulary" / "tok2block.json"
        if not vocab_path.exists():
            vocab_path = script_dir / "data" / "kaggle" / "input" / "v1" / "tok2block.json"

    if not vocab_path.exists():
        print(f"ERROR: Vocabulary not found at {vocab_path}")
        return

    if not data_dir.exists():
        print(f"ERROR: Data directory not found at {data_dir}")
        return

    print(f"Loading vocabulary from {vocab_path}")
    tok2block = load_vocabulary(vocab_path)

    # Find h5 files
    h5_files = list(data_dir.glob("*.h5"))
    print(f"Found {len(h5_files)} structures in {data_dir}")

    # Random selection
    if args.random > 0:
        h5_files = random.sample(h5_files, min(args.random, len(h5_files)))
        print(f"Randomly selected {len(h5_files)} structures")

    # Analyze structures
    structures = []
    for h5_path in h5_files[:args.preview if args.preview > 0 else len(h5_files)]:
        try:
            info = analyze_structure(h5_path, tok2block)

            # Filter by block count
            if info["non_air"] < args.min_blocks or info["non_air"] > args.max_blocks:
                continue

            # Search filter
            if args.search:
                block_names = [name for name, _ in info["top_blocks"]]
                if not any(args.search.lower() in name.lower() for name in block_names):
                    continue

            structures.append(info)
        except Exception as e:
            print(f"Error reading {h5_path}: {e}")

    # Print results
    if args.preview > 0 or args.search or args.random > 0:
        print(f"\n{'='*50}")
        print(f"Showing {len(structures)} structures:")
        for info in structures:
            print_structure_info(info, detailed=True)
    else:
        # Just list names and basic info
        print(f"\nStructure summary (first 20):")
        for info in structures[:20]:
            print(f"  {info['name']}: {info['non_air']} blocks, {info['unique_blocks']} types")

        if len(structures) > 20:
            print(f"  ... and {len(structures) - 20} more")

    # Print example commands
    if structures:
        example = structures[0]
        print(f"\n{'='*50}")
        print("To place a structure in Minecraft:")
        print(f"  python scripts/place_structure.py {example['path']} --pos 0 64 0 --clear")


if __name__ == "__main__":
    main()
