"""
Data Preparation Script for Minecraft AI

This script:
1. Loads all H5 files from data/processed/
2. Filters out invalid builds (too small, too large, empty)
3. Normalizes builds to target size (default 32x32x32)
4. Creates train/val/test splits
5. Saves splits to data/processed/train/, val/, test/

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --target-size 32 --min-size 5 --max-size 128
"""

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class PrepareDataConfig:
    """Configuration for data preparation."""

    # Paths
    input_dir: str = "data/processed"
    output_dir: str = "data/splits"
    vocab_path: str = "data/vocabulary/tok2block.json"

    # Filtering
    min_size: int = 5  # Minimum dimension (filter out tiny builds)
    max_size: int = 128  # Maximum dimension (filter out huge builds)
    min_unique_blocks: int = 3  # Minimum unique block types (filter spam)

    # Normalization
    target_size: tuple = (32, 32, 32)  # Target dimensions (H, W, D)
    air_token: int = 102  # Token ID for air (padding)

    # Splitting
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # Misc
    seed: int = 42


def load_h5_build(filepath: str) -> Optional[np.ndarray]:
    """Load a build from an H5 file.

    Args:
        filepath: Path to H5 file

    Returns:
        3D numpy array or None if loading fails
    """
    try:
        with h5py.File(filepath, "r") as f:
            # Get the first (usually only) dataset
            key = list(f.keys())[0]
            return f[key][:]
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def is_valid_build(
    build: np.ndarray,
    min_size: int,
    max_size: int,
    min_unique_blocks: int,
) -> bool:
    """Check if a build passes filtering criteria.

    Args:
        build: 3D numpy array
        min_size: Minimum dimension size
        max_size: Maximum dimension size
        min_unique_blocks: Minimum unique block types

    Returns:
        True if build is valid
    """
    # Check dimensions
    if any(d < min_size for d in build.shape):
        return False
    if any(d > max_size for d in build.shape):
        return False

    # Check unique blocks (filter out single-block spam)
    unique_blocks = len(np.unique(build))
    if unique_blocks < min_unique_blocks:
        return False

    return True


def normalize_build(
    build: np.ndarray,
    target_size: tuple,
    air_token: int,
) -> np.ndarray:
    """Normalize build to target size.

    - Smaller builds are padded with air
    - Larger builds are center-cropped

    Args:
        build: 3D numpy array
        target_size: Target (H, W, D)
        air_token: Token ID for air padding

    Returns:
        Normalized 3D array of shape target_size
    """
    result = np.full(target_size, air_token, dtype=build.dtype)

    # Calculate source and destination ranges
    src_h, src_w, src_d = build.shape
    tgt_h, tgt_w, tgt_d = target_size

    # For each dimension: center crop if larger, center place if smaller
    def get_ranges(src_size: int, tgt_size: int) -> tuple:
        if src_size <= tgt_size:
            # Source fits in target, center it
            offset = (tgt_size - src_size) // 2
            return (0, src_size), (offset, offset + src_size)
        else:
            # Source larger than target, center crop
            offset = (src_size - tgt_size) // 2
            return (offset, offset + tgt_size), (0, tgt_size)

    (sh_start, sh_end), (th_start, th_end) = get_ranges(src_h, tgt_h)
    (sw_start, sw_end), (tw_start, tw_end) = get_ranges(src_w, tgt_w)
    (sd_start, sd_end), (td_start, td_end) = get_ranges(src_d, tgt_d)

    result[th_start:th_end, tw_start:tw_end, td_start:td_end] = build[
        sh_start:sh_end, sw_start:sw_end, sd_start:sd_end
    ]

    return result


def create_splits(
    filepaths: list,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict:
    """Split filepaths into train/val/test sets.

    Args:
        filepaths: List of file paths
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed

    Returns:
        Dict with 'train', 'val', 'test' keys
    """
    random.seed(seed)
    shuffled = filepaths.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def prepare_data(config: PrepareDataConfig) -> dict:
    """Run the full data preparation pipeline.

    Args:
        config: Configuration object

    Returns:
        Statistics dict
    """
    input_dir = PROJECT_ROOT / config.input_dir
    output_dir = PROJECT_ROOT / config.output_dir

    # Create output directories
    for split in ["train", "val", "test"]:
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    # Find all H5 files
    h5_files = list(input_dir.glob("*.h5"))
    print(f"Found {len(h5_files)} H5 files")

    # Stats tracking
    stats = {
        "total": len(h5_files),
        "valid": 0,
        "filtered_too_small": 0,
        "filtered_too_large": 0,
        "filtered_low_diversity": 0,
        "filtered_load_error": 0,
        "shapes_before": [],
        "shapes_after": [],
    }

    # Filter and collect valid builds
    valid_files = []

    print("\nFiltering builds...")
    for filepath in tqdm(h5_files, desc="Filtering"):
        build = load_h5_build(str(filepath))

        if build is None:
            stats["filtered_load_error"] += 1
            continue

        # Check size constraints
        if any(d < config.min_size for d in build.shape):
            stats["filtered_too_small"] += 1
            continue

        if any(d > config.max_size for d in build.shape):
            stats["filtered_too_large"] += 1
            continue

        # Check diversity
        unique_blocks = len(np.unique(build))
        if unique_blocks < config.min_unique_blocks:
            stats["filtered_low_diversity"] += 1
            continue

        stats["valid"] += 1
        stats["shapes_before"].append(build.shape)
        valid_files.append(filepath)

    print(f"\nValid builds: {stats['valid']} / {stats['total']}")
    print(f"  Filtered (too small): {stats['filtered_too_small']}")
    print(f"  Filtered (too large): {stats['filtered_too_large']}")
    print(f"  Filtered (low diversity): {stats['filtered_low_diversity']}")
    print(f"  Filtered (load error): {stats['filtered_load_error']}")

    # Create splits
    splits = create_splits(
        valid_files,
        config.train_ratio,
        config.val_ratio,
        config.test_ratio,
        config.seed,
    )

    print(f"\nSplit sizes:")
    print(f"  Train: {len(splits['train'])}")
    print(f"  Val: {len(splits['val'])}")
    print(f"  Test: {len(splits['test'])}")

    # Process and save each split
    for split_name, filepaths in splits.items():
        print(f"\nProcessing {split_name} split...")
        split_dir = output_dir / split_name

        for filepath in tqdm(filepaths, desc=f"Normalizing {split_name}"):
            build = load_h5_build(str(filepath))
            if build is None:
                continue

            # Normalize
            normalized = normalize_build(
                build, config.target_size, config.air_token
            )
            stats["shapes_after"].append(normalized.shape)

            # Save
            output_path = split_dir / filepath.name
            with h5py.File(output_path, "w") as f:
                f.create_dataset(
                    "build",
                    data=normalized,
                    dtype=np.uint16,
                    compression="gzip",
                )

    # Save split metadata
    metadata = {
        "config": {
            "target_size": config.target_size,
            "min_size": config.min_size,
            "max_size": config.max_size,
            "min_unique_blocks": config.min_unique_blocks,
            "air_token": config.air_token,
            "train_ratio": config.train_ratio,
            "val_ratio": config.val_ratio,
            "test_ratio": config.test_ratio,
            "seed": config.seed,
        },
        "stats": {
            "total_files": stats["total"],
            "valid_files": stats["valid"],
            "train_count": len(splits["train"]),
            "val_count": len(splits["val"]),
            "test_count": len(splits["test"]),
        },
        "splits": {
            "train": [f.name for f in splits["train"]],
            "val": [f.name for f in splits["val"]],
            "test": [f.name for f in splits["test"]],
        },
    }

    with open(output_dir / "splits_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved splits to {output_dir}")
    print(f"Metadata saved to {output_dir / 'splits_metadata.json'}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Prepare Minecraft build data")
    parser.add_argument(
        "--target-size",
        type=int,
        default=32,
        help="Target dimension size (default: 32)",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=5,
        help="Minimum dimension size (default: 5)",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=128,
        help="Maximum dimension size (default: 128)",
    )
    parser.add_argument(
        "--min-blocks",
        type=int,
        default=3,
        help="Minimum unique block types (default: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    config = PrepareDataConfig(
        target_size=(args.target_size, args.target_size, args.target_size),
        min_size=args.min_size,
        max_size=args.max_size,
        min_unique_blocks=args.min_blocks,
        seed=args.seed,
    )

    print("=" * 60)
    print("Minecraft AI - Data Preparation")
    print("=" * 60)
    print(f"Target size: {config.target_size}")
    print(f"Min size: {config.min_size}")
    print(f"Max size: {config.max_size}")
    print(f"Min unique blocks: {config.min_unique_blocks}")
    print(f"Seed: {config.seed}")
    print("=" * 60)

    prepare_data(config)


if __name__ == "__main__":
    main()
