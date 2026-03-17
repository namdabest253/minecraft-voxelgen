"""
Batch render all H5 structures as 2D multi-view images.

Processes H5 files from both existing splits and new kaggle conversions,
saving composite renders for CLIP scoring.

Output: minecraft_ai/data/output/renders/{filename}.png

Usage:
    python minecraft_ai/scripts/render_all_structures.py
    python minecraft_ai/scripts/render_all_structures.py --input-dir kaggle/processed_h5
    python minecraft_ai/scripts/render_all_structures.py --max-files 100
"""

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.block_colors import load_block_colors
from src.data.voxel_renderer import render_structure_multiview


def save_image_ppm(image: np.ndarray, path: Path) -> None:
    """Save an image as PPM (no PIL dependency).

    Args:
        image: [H, W, 3] uint8 array.
        path: Output path (will use .ppm extension).
    """
    h, w = image.shape[:2]
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode())
        f.write(image.tobytes())


def save_image(image: np.ndarray, path: Path) -> None:
    """Save image as PNG (with PIL) or PPM fallback.

    Args:
        image: [H, W, 3] uint8 array.
        path: Output path.
    """
    try:
        from PIL import Image

        img = Image.fromarray(image)
        img.save(path)
    except ImportError:
        # Fallback to PPM format (no dependencies)
        ppm_path = path.with_suffix(".ppm")
        save_image_ppm(image, ppm_path)


def render_all(
    input_dirs: list,
    output_dir: Path,
    tok2block_path: Path,
    max_files: int = 0,
) -> dict:
    """Render all H5 files from input directories.

    Args:
        input_dirs: List of directories containing H5 files.
        output_dir: Where to save rendered images.
        tok2block_path: Path to tok2block.json vocabulary.
        max_files: If >0, limit total files processed.

    Returns:
        Stats dict with counts and timing.
    """
    # Load block colors
    print("Loading block color map...")
    block_colors = load_block_colors(str(tok2block_path))
    print(f"  {len(block_colors)} block colors loaded")

    # Collect all H5 files
    h5_files = []
    for input_dir in input_dirs:
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"  WARNING: {input_path} does not exist, skipping")
            continue
        files = sorted(input_path.glob("*.h5"))
        print(f"  Found {len(files)} H5 files in {input_path}")
        h5_files.extend(files)

    if max_files > 0:
        h5_files = h5_files[:max_files]

    print(f"\nTotal files to render: {len(h5_files)}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {"total": len(h5_files), "rendered": 0, "skipped": 0, "errors": 0}
    start_time = time.time()

    for h5_path in tqdm(h5_files, desc="Rendering"):
        output_path = output_dir / h5_path.stem

        # Skip if already rendered
        composite_path = output_dir / f"{h5_path.stem}.png"
        if composite_path.exists():
            stats["skipped"] += 1
            continue

        try:
            # Load structure
            with h5py.File(h5_path, "r") as f:
                key = list(f.keys())[0]
                block_ids = f[key][:].astype(np.int64)

            if block_ids.shape != (32, 32, 32):
                stats["errors"] += 1
                continue

            # Render composite multi-view
            composite = render_structure_multiview(block_ids, block_colors)
            save_image(composite, composite_path)

            stats["rendered"] += 1

        except Exception as e:
            stats["errors"] += 1
            if stats["errors"] <= 10:
                print(f"\n  Error rendering {h5_path.name}: {e}")

    elapsed = time.time() - start_time
    stats["elapsed_seconds"] = elapsed

    print(f"\nRendering complete in {elapsed:.1f}s")
    print(f"  Rendered: {stats['rendered']}")
    print(f"  Skipped (existing): {stats['skipped']}")
    print(f"  Errors: {stats['errors']}")
    if stats["rendered"] > 0:
        print(f"  Avg time per render: {elapsed / stats['rendered'] * 1000:.1f}ms")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Batch render H5 structures")
    parser.add_argument(
        "--input-dir",
        type=str,
        nargs="+",
        default=None,
        help="Input directories (default: splits + kaggle/processed_h5)",
    )
    parser.add_argument("--max-files", type=int, default=0, help="Limit files to process")
    args = parser.parse_args()

    tok2block_path = PROJECT_ROOT / "data" / "vocabulary" / "tok2block.json"
    output_dir = PROJECT_ROOT / "data" / "output" / "renders"

    if args.input_dir:
        input_dirs = args.input_dir
    else:
        # Default: existing splits + new kaggle data
        input_dirs = [
            str(PROJECT_ROOT / "data" / "splits" / "train"),
            str(PROJECT_ROOT / "data" / "splits" / "val"),
            str(PROJECT_ROOT / "data" / "splits" / "test"),
            str(PROJECT_ROOT.parent / "kaggle" / "processed_h5"),
        ]

    render_all(
        input_dirs=input_dirs,
        output_dir=output_dir,
        tok2block_path=tok2block_path,
        max_files=args.max_files,
    )


if __name__ == "__main__":
    main()
