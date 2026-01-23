"""
Convert .schem files to .h5 format for VQ-VAE training.

This script:
1. Loads all .schem files from kaggle/processed_builds
2. Converts them to .h5 format (32x32x32, uint16, gzip)
3. Applies same filtering as prepare_data.py (min/max size, diversity)
4. Outputs to kaggle/processed_h5/
5. Updates minecraft_ai/data/splits/splits_metadata.json with combined dataset

Expected: ~24,000 new .h5 files from 28,235 input .schem files (85% success rate)

Usage:
    python minecraft_ai/scripts/convert_schematics_to_h5.py
"""

import json
import random
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import existing utilities
from scripts.reconstruct_schematic import (
    AIR_TOKEN,
    read_schematic,
    resize_to_32,
    schematic_to_tokens,
)


@dataclass
class ConversionConfig:
    """Configuration for .schem to .h5 conversion."""

    # Paths (all relative to project root)
    input_dir: Path = PROJECT_ROOT.parent / "kaggle" / "processed_builds"
    output_dir: Path = PROJECT_ROOT.parent / "kaggle" / "processed_h5"
    vocab_path: Path = PROJECT_ROOT / "data" / "vocabulary" / "tok2block.json"
    splits_dir: Path = PROJECT_ROOT / "data" / "splits"

    # Filtering (match prepare_data.py exactly)
    min_size: int = 5  # Minimum dimension
    max_size: int = 128  # Maximum dimension
    min_unique_blocks: int = 3  # Minimum diversity

    # Normalization (match existing .h5 format)
    target_size: tuple = (32, 32, 32)
    air_token: int = AIR_TOKEN  # minecraft:air (102)

    # Processing
    validate_samples: int = 100
    error_log_path: Path = PROJECT_ROOT.parent / "conversion_errors.log"


def convert_single_schematic(
    schem_path: Path,
    output_dir: Path,
    block2tok: Dict[str, int],
    config: ConversionConfig,
) -> Tuple[bool, Optional[str], Optional[Dict]]:
    """
    Convert a single .schem file to .h5 format.

    Args:
        schem_path: Path to input .schem file
        output_dir: Directory for output .h5 file
        block2tok: Vocabulary mapping (block name -> token ID)
        config: Conversion configuration

    Returns:
        (success, error_message, stats_dict)
        - success: True if conversion succeeded
        - error_message: Error description if failed, None if succeeded
        - stats_dict: Statistics about the conversion (if succeeded)
    """
    try:
        # 1. Read schematic (reuse from reconstruct_schematic.py)
        blocks_xyz, palette, metadata = read_schematic(schem_path)
        x, y, z = blocks_xyz.shape

        # 2. Apply filtering (match prepare_data.py)
        if any(d < config.min_size for d in (x, y, z)):
            return False, f"too_small: {x}x{y}x{z}", None

        if any(d > config.max_size for d in (x, y, z)):
            return False, f"too_large: {x}x{y}x{z}", None

        # 3. Convert to tokens (reuse from reconstruct_schematic.py)
        tokens = schematic_to_tokens(
            blocks_xyz, palette, block2tok, air_token=config.air_token
        )

        # 4. Check diversity
        unique_blocks = len(np.unique(tokens))
        if unique_blocks < config.min_unique_blocks:
            return False, f"low_diversity: {unique_blocks}", None

        # 5. Normalize to 32x32x32 (reuse from reconstruct_schematic.py)
        normalized = resize_to_32(tokens, air_value=config.air_token)

        # 6. Write H5 (match prepare_data.py format EXACTLY)
        output_path = output_dir / schem_path.name.replace(".schem", ".h5")
        with h5py.File(output_path, "w") as f:
            f.create_dataset(
                "build",  # Same dataset name
                data=normalized,  # 32x32x32 array
                dtype=np.uint16,  # Same dtype
                compression="gzip",  # Same compression
            )

        # 7. Collect stats
        stats = {
            "original_size": (x, y, z),
            "unique_blocks": unique_blocks,
            "palette_size": len(palette),
        }

        return True, None, stats

    except Exception as e:
        return False, f"exception: {type(e).__name__}: {str(e)}", None


def process_all_schematics(config: ConversionConfig) -> Dict:
    """
    Process all .schem files with progress tracking.

    Args:
        config: Conversion configuration

    Returns:
        Statistics dict with conversion results
    """
    print("\n" + "=" * 60)
    print("Phase 1: Converting .schem files to .h5 format")
    print("=" * 60)

    # 1. Load vocabulary
    print(f"\nLoading vocabulary from {config.vocab_path}...")
    with open(config.vocab_path, "r") as f:
        tok2block = json.load(f)
    block2tok = {v: int(k) for k, v in tok2block.items()}
    print(f"  Loaded {len(tok2block)} block types")

    # 2. Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {config.output_dir}")

    # 3. Find all .schem files
    schem_files = list(config.input_dir.glob("*.schem"))
    print(f"\nFound {len(schem_files)} .schem files in {config.input_dir}")

    if len(schem_files) == 0:
        print("ERROR: No .schem files found!")
        return {}

    # 4. Track statistics
    stats = {
        "total": len(schem_files),
        "successful": 0,
        "filtered_too_small": 0,
        "filtered_too_large": 0,
        "filtered_low_diversity": 0,
        "failed": 0,
    }

    # 5. Open error log
    error_log = open(config.error_log_path, "w")
    print(f"Error log: {config.error_log_path}")

    # 6. Process each file with progress bar
    print("\nProcessing files...")
    for schem_path in tqdm(schem_files, desc="Converting"):
        success, error_msg, file_stats = convert_single_schematic(
            schem_path, config.output_dir, block2tok, config
        )

        if success:
            stats["successful"] += 1
        else:
            # Log error and continue (per user requirement)
            error_log.write(f"{schem_path.name}: {error_msg}\n")

            # Update filter counts
            if "too_small" in error_msg:
                stats["filtered_too_small"] += 1
            elif "too_large" in error_msg:
                stats["filtered_too_large"] += 1
            elif "low_diversity" in error_msg:
                stats["filtered_low_diversity"] += 1
            else:
                stats["failed"] += 1

    error_log.close()

    # 7. Print summary
    print("\n" + "=" * 60)
    print("Conversion Complete!")
    print("=" * 60)
    print(f"  Total input files:      {stats['total']}")
    print(f"  Successful conversions: {stats['successful']} ({stats['successful']/stats['total']*100:.1f}%)")
    print(f"\nFiltered:")
    print(f"  Too small:              {stats['filtered_too_small']}")
    print(f"  Too large:              {stats['filtered_too_large']}")
    print(f"  Low diversity:          {stats['filtered_low_diversity']}")
    print(f"\nErrors:")
    print(f"  Failed (exceptions):    {stats['failed']}")

    return stats


def validate_conversions(output_dir: Path, num_samples: int = 100):
    """
    Validate a random sample of converted files.

    Args:
        output_dir: Directory containing converted .h5 files
        num_samples: Number of files to validate
    """
    print("\n" + "=" * 60)
    print("Phase 2: Validating conversions")
    print("=" * 60)

    h5_files = list(output_dir.glob("*.h5"))
    if len(h5_files) == 0:
        print("\nWARNING: No .h5 files found to validate!")
        return

    sample_size = min(num_samples, len(h5_files))
    samples = random.sample(h5_files, sample_size)

    print(f"\nValidating {sample_size} random files...")

    validation_errors = []

    for h5_path in tqdm(samples, desc="Validating"):
        try:
            with h5py.File(h5_path, "r") as f:
                # Check dataset exists
                if "build" not in f.keys():
                    raise AssertionError("Missing 'build' dataset")

                # Check shape
                data = f["build"][:]
                if data.shape != (32, 32, 32):
                    raise AssertionError(f"Wrong shape: {data.shape}")

                # Check dtype
                if data.dtype != np.uint16:
                    raise AssertionError(f"Wrong dtype: {data.dtype}")

                # Check diversity
                unique_blocks = len(np.unique(data))
                if unique_blocks < 3:
                    raise AssertionError(f"Low diversity: {unique_blocks}")

        except Exception as e:
            validation_errors.append((h5_path.name, str(e)))

    # Report results
    print("\n" + "-" * 60)
    print("Validation Results")
    print("-" * 60)
    print(f"  Samples checked: {sample_size}")
    print(f"  Passed:          {sample_size - len(validation_errors)}")
    print(f"  Failed:          {len(validation_errors)}")

    if validation_errors:
        print("\nValidation errors (first 10):")
        for filename, error in validation_errors[:10]:
            print(f"  {filename}: {error}")


def update_splits(
    new_h5_dir: Path,
    existing_splits_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    """
    Combine new and existing .h5 files, re-split, update metadata.

    This combines:
    - Existing files from minecraft_ai/data/splits/{train,val,test}
    - New files from kaggle/processed_h5/

    Then re-splits everything 80/10/10 and updates splits_metadata.json

    Args:
        new_h5_dir: Directory with new .h5 files
        existing_splits_dir: Directory with existing splits
        train_ratio: Fraction for training (default 0.8)
        val_ratio: Fraction for validation (default 0.1)
        test_ratio: Fraction for testing (default 0.1)
        seed: Random seed for reproducibility (default 42)
    """
    print("\n" + "=" * 60)
    print("Phase 3: Updating splits metadata")
    print("=" * 60)

    # 1. Collect all .h5 files
    print("\nCollecting existing files...")
    existing_files = []
    for split in ["train", "val", "test"]:
        split_dir = existing_splits_dir / split
        if split_dir.exists():
            existing_files.extend(list(split_dir.glob("*.h5")))

    print(f"Collecting new files from {new_h5_dir}...")
    new_files = list(new_h5_dir.glob("*.h5"))

    all_files = existing_files + new_files

    print("\n" + "-" * 60)
    print("Dataset Summary")
    print("-" * 60)
    print(f"  Existing files: {len(existing_files)}")
    print(f"  New files:      {len(new_files)}")
    print(f"  Total files:    {len(all_files)}")

    # 2. Create new splits
    print("\nCreating new train/val/test splits...")
    random.seed(seed)
    shuffled = all_files.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    splits = {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }

    print("\n" + "-" * 60)
    print("New Split Sizes")
    print("-" * 60)
    print(f"  Train: {len(splits['train'])}")
    print(f"  Val:   {len(splits['val'])}")
    print(f"  Test:  {len(splits['test'])}")

    # 3. Copy files to split directories
    print("\nCopying files to split directories...")
    for split_name, filepaths in splits.items():
        split_dir = existing_splits_dir / split_name

        # Clear existing files first
        print(f"\n  Clearing {split_name}/...")
        if split_dir.exists():
            for old_file in split_dir.glob("*.h5"):
                old_file.unlink()
        else:
            split_dir.mkdir(parents=True, exist_ok=True)

        # Copy new split files
        for filepath in tqdm(filepaths, desc=f"  Copying to {split_name}"):
            dest = split_dir / filepath.name
            shutil.copy2(filepath, dest)

    # 4. Update splits_metadata.json
    print("\nUpdating splits_metadata.json...")
    metadata = {
        "config": {
            "target_size": [32, 32, 32],
            "air_token": 102,
            "min_size": 5,
            "max_size": 128,
            "min_unique_blocks": 3,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "seed": seed,
        },
        "stats": {
            "total_files": len(all_files),
            "existing_files": len(existing_files),
            "new_kaggle_files": len(new_files),
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

    metadata_path = existing_splits_dir / "splits_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nUpdated {metadata_path}")


def main():
    """Main execution function."""
    print("=" * 60)
    print("Schematic to H5 Conversion Script")
    print("=" * 60)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    config = ConversionConfig()

    # Verify input directory exists
    if not config.input_dir.exists():
        print(f"\nERROR: Input directory not found: {config.input_dir}")
        print("Please check the path and try again.")
        sys.exit(1)

    # Verify vocabulary file exists
    if not config.vocab_path.exists():
        print(f"\nERROR: Vocabulary file not found: {config.vocab_path}")
        print("Please check the path and try again.")
        sys.exit(1)

    # Phase 1: Convert all .schem files
    stats = process_all_schematics(config)

    if stats.get("successful", 0) == 0:
        print("\nERROR: No files were successfully converted!")
        print("Check the error log for details.")
        sys.exit(1)

    # Phase 2: Validate sample
    validate_conversions(config.output_dir, config.validate_samples)

    # Phase 3: Update splits metadata
    update_splits(
        new_h5_dir=config.output_dir,
        existing_splits_dir=config.splits_dir,
    )

    # Phase 4: Save conversion metadata
    print("\n" + "=" * 60)
    print("Phase 4: Saving conversion metadata")
    print("=" * 60)

    metadata = {
        "conversion_date": datetime.now().isoformat(),
        "config": {
            "input_dir": str(config.input_dir),
            "output_dir": str(config.output_dir),
            "vocab_path": str(config.vocab_path),
            "min_size": config.min_size,
            "max_size": config.max_size,
            "min_unique_blocks": config.min_unique_blocks,
            "target_size": config.target_size,
            "air_token": config.air_token,
        },
        "statistics": stats,
    }

    metadata_path = config.output_dir / "conversion_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nConversion metadata saved to: {metadata_path}")
    print(f"Error log saved to: {config.error_log_path}")

    print("\n" + "=" * 60)
    print("All phases complete!")
    print("=" * 60)
    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nNext steps:")
    print("  1. Review conversion_errors.log for any patterns")
    print("  2. Inspect conversion_metadata.json statistics")
    print("  3. Load random .h5 files and verify they look correct")
    print("  4. Update CLAUDE.md to reflect new dataset size")
    print("  5. Ready to train VQ-VAE v7 with 6.4x more data!")


if __name__ == "__main__":
    main()
