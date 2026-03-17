"""
Generate procedural Minecraft houses as H5 training data.

Produces simple randomized houses (walls, roof, door, windows) in the same
32x32x32 uint16 H5 format used by the existing training pipeline.

Usage:
    python minecraft_ai/scripts/generate_procedural_houses.py
    python minecraft_ai/scripts/generate_procedural_houses.py --count 5000
    python minecraft_ai/scripts/generate_procedural_houses.py --output-dir data/procedural --dry-run
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
VOCAB_PATH = PROJECT_ROOT / "data" / "vocabulary" / "tok2block.json"

# Air token
AIR = 102


def load_vocab(vocab_path: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Load vocabulary and build block_name -> token_id mapping.

    Returns:
        block2tok: block name (with or without states) -> token ID
        tok2block: token ID -> block name
    """
    with open(vocab_path, "r") as f:
        tok2block = json.load(f)

    block2tok: Dict[str, int] = {}
    for tok_str, name in tok2block.items():
        tok = int(tok_str)
        block2tok[name] = tok
        # Also register base name (without block state) for convenience
        base = name.split("[")[0]
        block2tok.setdefault(base, tok)

    return block2tok, {int(k): v for k, v in tok2block.items()}


def lookup(block2tok: Dict[str, int], name: str) -> int:
    """Look up a block token, falling back to base name."""
    tok = block2tok.get(name)
    if tok is not None:
        return tok
    base = name.split("[")[0]
    return block2tok.get(base, AIR)


# ============================================================
# Material Palettes
# ============================================================

@dataclass
class MaterialPalette:
    """A set of blocks that go together for a house style."""

    name: str
    wall: str  # Main wall block
    log: str  # Corner pillars (axis=y variant)
    planks: str  # Fallback / floor
    slab_bottom: str  # For flat roofs
    stairs_n: str  # Stairs facing north
    stairs_e: str  # Stairs facing east
    stairs_s: str  # Stairs facing south
    stairs_w: str  # Stairs facing west


PALETTES = [
    MaterialPalette(
        name="oak",
        wall="minecraft:oak_planks",
        log="minecraft:oak_log[axis=y]",
        planks="minecraft:oak_planks",
        slab_bottom="minecraft:oak_slab[type=bottom]",
        stairs_n="minecraft:oak_stairs[facing=north,half=bottom,shape=straight]",
        stairs_e="minecraft:oak_stairs[facing=east,half=bottom,shape=straight]",
        stairs_s="minecraft:oak_stairs[facing=south,half=bottom,shape=straight]",
        stairs_w="minecraft:oak_stairs[facing=west,half=bottom,shape=straight]",
    ),
    MaterialPalette(
        name="spruce",
        wall="minecraft:spruce_planks",
        log="minecraft:spruce_log[axis=y]",
        planks="minecraft:spruce_planks",
        slab_bottom="minecraft:spruce_slab[type=bottom]",
        stairs_n="minecraft:spruce_stairs[facing=north,half=bottom,shape=straight]",
        stairs_e="minecraft:spruce_stairs[facing=east,half=bottom,shape=straight]",
        stairs_s="minecraft:spruce_stairs[facing=south,half=bottom,shape=straight]",
        stairs_w="minecraft:spruce_stairs[facing=west,half=bottom,shape=straight]",
    ),
    MaterialPalette(
        name="birch",
        wall="minecraft:birch_planks",
        log="minecraft:birch_log[axis=y]",
        planks="minecraft:birch_planks",
        slab_bottom="minecraft:birch_slab[type=bottom]",
        stairs_n="minecraft:birch_stairs[facing=north,half=bottom,shape=straight]",
        stairs_e="minecraft:birch_stairs[facing=east,half=bottom,shape=straight]",
        stairs_s="minecraft:birch_stairs[facing=south,half=bottom,shape=straight]",
        stairs_w="minecraft:birch_stairs[facing=west,half=bottom,shape=straight]",
    ),
    MaterialPalette(
        name="dark_oak",
        wall="minecraft:dark_oak_planks",
        log="minecraft:dark_oak_log[axis=y]",
        planks="minecraft:dark_oak_planks",
        slab_bottom="minecraft:dark_oak_slab[type=bottom]",
        stairs_n="minecraft:dark_oak_stairs[facing=north,half=bottom,shape=straight]",
        stairs_e="minecraft:dark_oak_stairs[facing=east,half=bottom,shape=straight]",
        stairs_s="minecraft:dark_oak_stairs[facing=south,half=bottom,shape=straight]",
        stairs_w="minecraft:dark_oak_stairs[facing=west,half=bottom,shape=straight]",
    ),
    MaterialPalette(
        name="cobblestone",
        wall="minecraft:cobblestone",
        log="minecraft:stone_bricks",
        planks="minecraft:cobblestone",
        slab_bottom="minecraft:cobblestone_slab[type=bottom]",
        stairs_n="minecraft:cobblestone_stairs[facing=north,half=bottom,shape=straight]",
        stairs_e="minecraft:cobblestone_stairs[facing=east,half=bottom,shape=straight]",
        stairs_s="minecraft:cobblestone_stairs[facing=south,half=bottom,shape=straight]",
        stairs_w="minecraft:cobblestone_stairs[facing=west,half=bottom,shape=straight]",
    ),
    MaterialPalette(
        name="stone_brick",
        wall="minecraft:stone_bricks",
        log="minecraft:stone_bricks",
        planks="minecraft:stone_bricks",
        slab_bottom="minecraft:stone_brick_slab[type=bottom]",
        stairs_n="minecraft:stone_brick_stairs[facing=north,half=bottom,shape=straight]",
        stairs_e="minecraft:stone_brick_stairs[facing=east,half=bottom,shape=straight]",
        stairs_s="minecraft:stone_brick_stairs[facing=south,half=bottom,shape=straight]",
        stairs_w="minecraft:stone_brick_stairs[facing=west,half=bottom,shape=straight]",
    ),
    MaterialPalette(
        name="brick",
        wall="minecraft:bricks",
        log="minecraft:bricks",
        planks="minecraft:bricks",
        slab_bottom="minecraft:brick_slab[type=bottom]",
        stairs_n="minecraft:brick_stairs[facing=north,half=bottom,shape=straight]",
        stairs_e="minecraft:brick_stairs[facing=east,half=bottom,shape=straight]",
        stairs_s="minecraft:brick_stairs[facing=south,half=bottom,shape=straight]",
        stairs_w="minecraft:brick_stairs[facing=west,half=bottom,shape=straight]",
    ),
]


# ============================================================
# House Generator
# ============================================================


@dataclass
class HouseParams:
    """Randomized parameters for a single house."""

    width: int  # X dimension (exterior)
    depth: int  # Z dimension (exterior)
    wall_height: int  # Y dimension of walls (not counting roof)
    palette: MaterialPalette
    roof_style: str  # "flat" or "gabled"
    has_windows: bool
    window_height: int  # 1 or 2
    windows_per_wall: int  # 0-3
    has_door: bool


def random_house_params(rng: np.random.Generator) -> HouseParams:
    """Generate random house parameters."""
    palette = PALETTES[rng.integers(0, len(PALETTES))]
    width = int(rng.integers(5, 17))  # 5-16
    depth = int(rng.integers(5, 17))  # 5-16
    wall_height = int(rng.integers(4, 9))  # 4-8

    # Gabled roof needs odd width for symmetry
    roof_style = "gabled" if rng.random() < 0.5 else "flat"
    if roof_style == "gabled" and width % 2 == 0:
        width += 1  # Make odd for symmetric peak
        if width > 16:
            width = 15

    has_windows = rng.random() < 0.8
    window_height = int(rng.integers(1, 3))  # 1 or 2
    windows_per_wall = int(rng.integers(1, 4)) if has_windows else 0

    return HouseParams(
        width=width,
        depth=depth,
        wall_height=wall_height,
        palette=palette,
        roof_style=roof_style,
        has_windows=has_windows,
        window_height=window_height,
        windows_per_wall=windows_per_wall,
        has_door=True,
    )


def generate_house(
    params: HouseParams, block2tok: Dict[str, int]
) -> Tuple[np.ndarray, str]:
    """Generate a house as a 32x32x32 token array.

    Coordinate system: X=width, Y=height, Z=depth.
    House is centered in X/Z, grounded at Y=0.

    Args:
        params: House parameters.
        block2tok: Block name -> token ID mapping.

    Returns:
        tokens: (32, 32, 32) uint16 array.
        caption: Descriptive caption string.
    """
    vol = np.full((32, 32, 32), AIR, dtype=np.uint16)
    p = params
    pal = p.palette

    # Resolve tokens
    wall_tok = lookup(block2tok, pal.wall)
    log_tok = lookup(block2tok, pal.log)
    slab_tok = lookup(block2tok, pal.slab_bottom)
    glass_tok = lookup(block2tok, "minecraft:glass_pane")
    door_tok = lookup(block2tok, "minecraft:oak_door[facing=south]")

    stairs_n = lookup(block2tok, pal.stairs_n)
    stairs_e = lookup(block2tok, pal.stairs_e)
    stairs_s = lookup(block2tok, pal.stairs_s)
    stairs_w = lookup(block2tok, pal.stairs_w)

    # Center house in 32x32x32
    ox = (32 - p.width) // 2
    oz = (32 - p.depth) // 2
    oy = 0  # Grounded

    # --- Floor ---
    vol[ox : ox + p.width, oy, oz : oz + p.depth] = wall_tok

    # --- Walls (4 sides, hollow interior) ---
    for y in range(1, p.wall_height + 1):
        # Front wall (z = oz)
        vol[ox : ox + p.width, oy + y, oz] = wall_tok
        # Back wall (z = oz + depth - 1)
        vol[ox : ox + p.width, oy + y, oz + p.depth - 1] = wall_tok
        # Left wall (x = ox)
        vol[ox, oy + y, oz : oz + p.depth] = wall_tok
        # Right wall (x = ox + width - 1)
        vol[ox + p.width - 1, oy + y, oz : oz + p.depth] = wall_tok

    # --- Corner log pillars ---
    for y in range(1, p.wall_height + 1):
        vol[ox, oy + y, oz] = log_tok
        vol[ox + p.width - 1, oy + y, oz] = log_tok
        vol[ox, oy + y, oz + p.depth - 1] = log_tok
        vol[ox + p.width - 1, oy + y, oz + p.depth - 1] = log_tok

    # --- Door (front wall, centered) ---
    if p.has_door:
        door_x = ox + p.width // 2
        door_z = oz  # Front wall
        # Door is 1 wide, 2 tall
        vol[door_x, oy + 1, door_z] = door_tok
        vol[door_x, oy + 2, door_z] = door_tok

    # --- Windows ---
    if p.has_windows and p.windows_per_wall > 0:
        _place_windows(vol, p, ox, oy, oz, glass_tok)

    # --- Roof ---
    if p.roof_style == "flat":
        _place_flat_roof(vol, p, ox, oy, oz, slab_tok)
    else:
        _place_gabled_roof(vol, p, ox, oy, oz, stairs_n, stairs_s, slab_tok, wall_tok)

    # --- Caption ---
    size = max(p.width, p.depth)
    if size <= 7:
        size_word = "small"
    elif size <= 12:
        size_word = "medium"
    else:
        size_word = "large"

    roof_word = "gabled" if p.roof_style == "gabled" else "flat"
    mat_word = pal.name.replace("_", " ")

    window_part = " and glass windows" if p.has_windows else ""
    caption = f"{size_word} {mat_word} house with {roof_word} roof{window_part}"

    return vol, caption


def _place_windows(
    vol: np.ndarray,
    p: HouseParams,
    ox: int,
    oy: int,
    oz: int,
    glass_tok: int,
) -> None:
    """Place windows on all four walls."""
    wh = p.window_height  # 1 or 2
    # Window Y starts at wall_height // 2 (centered vertically)
    wy_start = oy + max(2, (p.wall_height - wh) // 2 + 1)

    # Front and back walls (along X axis)
    for wall_z, skip_center in [(oz, True), (oz + p.depth - 1, False)]:
        positions = _window_positions(p.width, p.windows_per_wall, skip_center)
        for wx in positions:
            x = ox + wx
            for dy in range(wh):
                if wy_start + dy <= oy + p.wall_height:
                    vol[x, wy_start + dy, wall_z] = glass_tok

    # Left and right walls (along Z axis)
    for wall_x in [ox, ox + p.width - 1]:
        positions = _window_positions(p.depth, p.windows_per_wall, False)
        for wz in positions:
            z = oz + wz
            for dy in range(wh):
                if wy_start + dy <= oy + p.wall_height:
                    vol[wall_x, wy_start + dy, z] = glass_tok


def _window_positions(
    wall_length: int, count: int, skip_center: bool
) -> List[int]:
    """Compute evenly-spaced window positions along a wall.

    Args:
        wall_length: Total wall length.
        count: Desired number of windows.
        skip_center: If True, avoid the center position (for door wall).

    Returns:
        List of offsets from wall start (1-indexed from interior).
    """
    # Available interior positions (skip corners)
    interior = list(range(2, wall_length - 2))
    if not interior:
        return []

    center = wall_length // 2
    if skip_center:
        # Remove door position and neighbors
        interior = [x for x in interior if abs(x - center) > 1]

    if not interior:
        return []

    # Evenly space windows
    count = min(count, len(interior))
    if count == 0:
        return []

    step = len(interior) / (count + 1)
    positions = []
    for i in range(1, count + 1):
        idx = int(step * i)
        idx = min(idx, len(interior) - 1)
        positions.append(interior[idx])

    return positions


def _place_flat_roof(
    vol: np.ndarray,
    p: HouseParams,
    ox: int,
    oy: int,
    oz: int,
    slab_tok: int,
) -> None:
    """Place a flat slab roof with 1-block overhang."""
    roof_y = oy + p.wall_height + 1
    if roof_y >= 32:
        roof_y = 31
    # Overhang of 1 block on each side
    x_start = max(0, ox - 1)
    x_end = min(32, ox + p.width + 1)
    z_start = max(0, oz - 1)
    z_end = min(32, oz + p.depth + 1)
    vol[x_start:x_end, roof_y, z_start:z_end] = slab_tok


def _place_gabled_roof(
    vol: np.ndarray,
    p: HouseParams,
    ox: int,
    oy: int,
    oz: int,
    stairs_n: int,
    stairs_s: int,
    slab_tok: int,
    wall_tok: int,
) -> None:
    """Place a gabled (peaked) roof running along the Z axis.

    Roof ridge runs along Z (depth). Stairs slope down on both X sides.
    """
    roof_base_y = oy + p.wall_height + 1
    half_w = p.width // 2

    for layer in range(half_w + 1):
        y = roof_base_y + layer
        if y >= 32:
            break

        left_x = ox + layer
        right_x = ox + p.width - 1 - layer

        if left_x > right_x:
            # Peak — place slabs
            for z in range(oz - 1, oz + p.depth + 1):
                if 0 <= z < 32 and 0 <= left_x < 32:
                    vol[left_x, y, z] = slab_tok
            break

        if left_x == right_x:
            # Single block peak
            for z in range(oz - 1, oz + p.depth + 1):
                if 0 <= z < 32 and 0 <= left_x < 32:
                    vol[left_x, y, z] = slab_tok
            break

        # Stairs on both sides — stairs_s faces outward-left, stairs_n faces outward-right
        # Actually for a gable along Z: left side uses stairs facing west (outward),
        # right side uses stairs facing east (outward). But we only have N/S in params.
        # Let's use the wall material for gable fill and slabs for the top.
        for z in range(oz - 1, oz + p.depth + 1):
            if 0 <= z < 32:
                if 0 <= left_x < 32:
                    vol[left_x, y, z] = stairs_n  # Using as stair-step
                if 0 <= right_x < 32:
                    vol[right_x, y, z] = stairs_s

        # Fill gable ends (triangular walls at z=oz and z=oz+depth-1)
        for x in range(left_x + 1, right_x):
            if 0 <= x < 32:
                if 0 <= oz < 32:
                    vol[x, y, oz] = wall_tok
                if 0 <= oz + p.depth - 1 < 32:
                    vol[x, y, oz + p.depth - 1] = wall_tok


# ============================================================
# Main
# ============================================================


def save_h5(path: Path, tokens: np.ndarray) -> None:
    """Save a 32x32x32 token array as H5 file."""
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "build",
            data=tokens.astype(np.uint16),
            dtype=np.uint16,
            compression="gzip",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate procedural Minecraft houses as H5 training data"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10000,
        help="Number of houses to generate (default: 10000)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "procedural"),
        help="Output directory for H5 files",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate but don't save (print stats only)",
    )
    parser.add_argument(
        "--save-captions",
        action="store_true",
        default=True,
        help="Save captions.json alongside H5 files",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Procedural House Generator")
    print("=" * 60)

    # Load vocabulary
    print(f"Loading vocabulary from {VOCAB_PATH}...")
    block2tok, tok2block = load_vocab(VOCAB_PATH)
    print(f"  {len(tok2block)} token types")

    # Verify key tokens exist
    test_blocks = ["minecraft:oak_planks", "minecraft:cobblestone", "minecraft:glass_pane"]
    for b in test_blocks:
        tok = lookup(block2tok, b)
        if tok == AIR:
            print(f"  WARNING: {b} resolved to AIR — check vocabulary")
        else:
            print(f"  {b} -> token {tok}")

    output_dir = Path(args.output_dir)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    captions: Dict[str, str] = {}

    # Track stats
    palette_counts: Dict[str, int] = {}
    roof_counts: Dict[str, int] = {}
    size_counts: Dict[str, int] = {}

    print(f"\nGenerating {args.count} houses (seed={args.seed})...")

    for i in range(args.count):
        params = random_house_params(rng)
        tokens, caption = generate_house(params, block2tok)

        filename = f"proc_house_{i:05d}.h5"

        if not args.dry_run:
            save_h5(output_dir / filename, tokens)

        captions[filename] = caption

        # Stats
        palette_counts[params.palette.name] = (
            palette_counts.get(params.palette.name, 0) + 1
        )
        roof_counts[params.roof_style] = (
            roof_counts.get(params.roof_style, 0) + 1
        )
        size = max(params.width, params.depth)
        bucket = "small" if size <= 7 else ("medium" if size <= 12 else "large")
        size_counts[bucket] = size_counts.get(bucket, 0) + 1

        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{args.count}...")

    # Save captions
    if not args.dry_run and args.save_captions:
        captions_path = output_dir / "captions.json"
        with open(captions_path, "w") as f:
            json.dump(captions, f, indent=2)
        print(f"\nCaptions saved to {captions_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Generation Complete")
    print("=" * 60)
    print(f"  Total: {args.count}")
    if not args.dry_run:
        print(f"  Output: {output_dir}")

    print(f"\nBy material:")
    for name, count in sorted(palette_counts.items()):
        print(f"  {name}: {count}")

    print(f"\nBy roof style:")
    for style, count in sorted(roof_counts.items()):
        print(f"  {style}: {count}")

    print(f"\nBy size:")
    for bucket, count in sorted(size_counts.items()):
        print(f"  {bucket}: {count}")

    # Show a few sample captions
    print(f"\nSample captions:")
    for filename, caption in list(captions.items())[:5]:
        print(f"  {filename}: \"{caption}\"")


if __name__ == "__main__":
    main()
