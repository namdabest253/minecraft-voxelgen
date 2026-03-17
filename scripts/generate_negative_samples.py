"""
Generate procedural negative training samples for discrete diffusion.

Creates synthetic H5 files (32x32x32, uint16, key="build") with known
negative characteristics so the model learns what to avoid via negative
prompting. Each sample gets a pre-assigned caption in captions.json.

Four negative types:
  1. random  — random block IDs at varying densities
  2. empty   — sparse scattered blocks (5-50)
  3. terrain — layered dirt/stone/grass with noise
  4. blob    — 3D gaussian blob of a single material

Usage:
    cd minecraft_ai
    python scripts/generate_negative_samples.py                    # 1000 samples
    python scripts/generate_negative_samples.py --count 500        # custom count
    python scripts/generate_negative_samples.py --dry-run          # preview only
"""

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

AIR_TOKEN = 102
GRID_SIZE = 32

# Terrain block token IDs (from tok2block.json, no block state variants)
TERRAIN_TOKENS = {
    "dirt": 1276,
    "grass_block": 1454,
    "stone": 3162,
    "cobblestone": 740,
    "gravel": 1455,
    "sand": 2851,
    "coarse_dirt": 695,
    "bedrock": 307,
    "deepslate": 1127,
}

# Caption templates per type (randomly selected per sample)
CAPTIONS = {
    "random": [
        "random scattered blocks with no structure",
        "chaotic noise pattern of assorted blocks",
        "random jumble of mismatched blocks",
        "disorganized mess of randomly placed blocks",
        "noisy scattering of blocks with no pattern",
    ],
    "empty": [
        "mostly empty space with a few scattered blocks",
        "sparse random blocks floating in empty space",
        "nearly empty volume with barely any blocks",
        "a handful of isolated blocks in open air",
        "almost nothing, just a few stray blocks",
    ],
    "terrain": [
        "natural terrain with dirt and stone layers",
        "flat ground made of grass and dirt",
        "raw landscape of stone and gravel",
        "natural hillside with grass_block and dirt",
        "unmodified terrain with cobblestone and stone",
    ],
    "blob": [
        "shapeless amorphous blob of blocks",
        "formless rounded mass with no detail",
        "featureless blob with no architectural features",
        "smooth amorphous lump of a single material",
        "rounded shapeless cluster of blocks",
    ],
}


def _load_all_block_tokens() -> np.ndarray:
    """Load all valid non-air token IDs from vocabulary."""
    tok2block_path = PROJECT_ROOT / "data" / "vocabulary" / "tok2block.json"
    with open(tok2block_path) as f:
        tok2block = json.load(f)
    tokens = [int(k) for k in tok2block.keys() if int(k) not in {102, 576, 3352}]
    return np.array(tokens, dtype=np.uint16)


def generate_random(rng: np.random.Generator, all_tokens: np.ndarray) -> np.ndarray:
    """Random block IDs at varying density (10-80% fill)."""
    volume = np.full((GRID_SIZE, GRID_SIZE, GRID_SIZE), AIR_TOKEN, dtype=np.uint16)
    density = rng.uniform(0.10, 0.80)
    n_blocks = int(GRID_SIZE**3 * density)
    flat_idx = rng.choice(GRID_SIZE**3, size=n_blocks, replace=False)
    block_ids = rng.choice(all_tokens, size=n_blocks)
    volume.flat[flat_idx] = block_ids
    return volume


def generate_empty(rng: np.random.Generator, all_tokens: np.ndarray) -> np.ndarray:
    """Very sparse: 5-50 random blocks."""
    volume = np.full((GRID_SIZE, GRID_SIZE, GRID_SIZE), AIR_TOKEN, dtype=np.uint16)
    n_blocks = rng.integers(5, 51)
    flat_idx = rng.choice(GRID_SIZE**3, size=n_blocks, replace=False)
    block_ids = rng.choice(all_tokens, size=n_blocks)
    volume.flat[flat_idx] = block_ids
    return volume


def generate_terrain(rng: np.random.Generator) -> np.ndarray:
    """Layered terrain: bedrock → stone → dirt → grass with height noise."""
    volume = np.full((GRID_SIZE, GRID_SIZE, GRID_SIZE), AIR_TOKEN, dtype=np.uint16)
    terrain_toks = TERRAIN_TOKENS

    # Generate a 2D heightmap with smooth noise
    base_height = rng.integers(8, 20)
    heightmap = np.full((GRID_SIZE, GRID_SIZE), base_height, dtype=np.float64)

    # Add low-frequency noise (smooth hills)
    for _ in range(3):
        cx, cz = rng.uniform(0, GRID_SIZE, size=2)
        radius = rng.uniform(6, 16)
        amplitude = rng.uniform(-4, 4)
        for x in range(GRID_SIZE):
            for z in range(GRID_SIZE):
                dist = np.sqrt((x - cx) ** 2 + (z - cz) ** 2)
                heightmap[x, z] += amplitude * max(0, 1 - dist / radius)

    # Add per-column noise
    heightmap += rng.integers(-1, 2, size=(GRID_SIZE, GRID_SIZE)).astype(np.float64)
    heightmap = np.clip(heightmap, 2, GRID_SIZE - 1).astype(int)

    for x in range(GRID_SIZE):
        for z in range(GRID_SIZE):
            h = heightmap[x, z]
            # Layers: bedrock (y=0), stone, dirt, grass on top
            volume[x, 0, z] = terrain_toks["bedrock"]
            for y in range(1, max(1, h - 4)):
                volume[x, y, z] = terrain_toks["stone"]
            for y in range(max(1, h - 4), h):
                volume[x, y, z] = terrain_toks["dirt"]
            if h < GRID_SIZE:
                volume[x, h, z] = terrain_toks["grass_block"]

    # Scatter some gravel/cobblestone patches
    n_patches = rng.integers(1, 4)
    for _ in range(n_patches):
        px, pz = rng.integers(0, GRID_SIZE, size=2)
        patch_tok = rng.choice(
            [terrain_toks["gravel"], terrain_toks["cobblestone"], terrain_toks["sand"]]
        )
        radius = rng.integers(2, 5)
        for dx in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                nx, nz = px + dx, pz + dz
                if 0 <= nx < GRID_SIZE and 0 <= nz < GRID_SIZE:
                    if dx * dx + dz * dz <= radius * radius:
                        h = heightmap[nx, nz]
                        if h < GRID_SIZE:
                            volume[nx, h, nz] = patch_tok

    return volume


def generate_blob(rng: np.random.Generator, all_tokens: np.ndarray) -> np.ndarray:
    """3D gaussian blob of a single material."""
    volume = np.full((GRID_SIZE, GRID_SIZE, GRID_SIZE), AIR_TOKEN, dtype=np.uint16)

    # Pick a single block type
    block_id = rng.choice(all_tokens)

    # Random center and radius
    center = rng.uniform(8, 24, size=3)
    sigma = rng.uniform(3, 8, size=3)

    coords = np.mgrid[0:GRID_SIZE, 0:GRID_SIZE, 0:GRID_SIZE].astype(np.float64)
    dist_sq = sum(
        ((coords[i] - center[i]) / sigma[i]) ** 2 for i in range(3)
    )

    # Fill where distance < 1 (inside the ellipsoid) with some surface noise
    threshold = 1.0 + rng.normal(0, 0.1, size=dist_sq.shape)
    mask = dist_sq < threshold
    volume[mask] = block_id

    return volume


# Map type names to generators
GENERATORS = {
    "random": lambda rng, tokens: generate_random(rng, tokens),
    "empty": lambda rng, tokens: generate_empty(rng, tokens),
    "terrain": lambda rng, tokens: generate_terrain(rng),
    "blob": lambda rng, tokens: generate_blob(rng, tokens),
}


def generate_negative_samples(
    output_dir: Path,
    captions_path: Path,
    count: int = 1000,
    seed: int = 12345,
    dry_run: bool = False,
) -> dict:
    """Generate negative H5 samples and update captions.json.

    Args:
        output_dir: Directory to write H5 files.
        captions_path: Path to captions.json to update.
        count: Total number of samples to generate.
        seed: Random seed.
        dry_run: If True, print plan but don't write files.

    Returns:
        Stats dict.
    """
    rng = np.random.default_rng(seed)
    all_tokens = _load_all_block_tokens()

    types = list(GENERATORS.keys())
    per_type = count // len(types)
    remainder = count - per_type * len(types)

    plan = {t: per_type for t in types}
    # Distribute remainder across types
    for i in range(remainder):
        plan[types[i]] += 1

    print(f"Generating {count} negative samples:")
    for t, n in plan.items():
        print(f"  {t}: {n}")
    print(f"Output: {output_dir}")

    if dry_run:
        print("\n[DRY RUN] No files written.")
        return plan

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing captions
    existing_captions = {}
    if captions_path.exists():
        with open(captions_path) as f:
            existing_captions = json.load(f)

    stats = {t: 0 for t in types}
    new_captions = {}

    for neg_type, n_samples in plan.items():
        gen_fn = GENERATORS[neg_type]
        caption_pool = CAPTIONS[neg_type]

        for i in range(n_samples):
            filename = f"neg_{neg_type}_{i:04d}.h5"
            h5_path = output_dir / filename

            volume = gen_fn(rng, all_tokens)

            with h5py.File(h5_path, "w") as f:
                f.create_dataset("build", data=volume, dtype="uint16")

            # Count non-air for metadata
            non_air = int(np.sum(volume != AIR_TOKEN))
            if non_air < 500:
                scale = "small"
            elif non_air < 2000:
                scale = "medium"
            else:
                scale = "large"

            caption = caption_pool[rng.integers(0, len(caption_pool))]

            new_captions[filename] = {
                "caption": caption,
                "top_blocks": [],
                "scale": scale,
                "non_air_voxels": non_air,
                "negative_type": neg_type,
            }
            stats[neg_type] += 1

    # Merge into captions.json
    existing_captions.update(new_captions)
    captions_path.parent.mkdir(parents=True, exist_ok=True)
    with open(captions_path, "w") as f:
        json.dump(existing_captions, f, indent=2)

    total = sum(stats.values())
    print(f"\nGenerated {total} negative samples")
    for t, n in stats.items():
        print(f"  {t}: {n}")
    print(f"Updated captions: {captions_path} ({len(existing_captions)} total entries)")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate procedural negative training samples"
    )
    parser.add_argument(
        "--count", type=int, default=1000,
        help="Total number of negative samples (default: 1000)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for H5 files (default: data/negative_samples/)",
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview plan without writing files")
    args = parser.parse_args()

    output_dir = (
        Path(args.output_dir) if args.output_dir
        else PROJECT_ROOT / "data" / "negative_samples"
    )
    captions_path = PROJECT_ROOT / "data" / "output" / "captions" / "captions.json"

    generate_negative_samples(
        output_dir=output_dir,
        captions_path=captions_path,
        count=args.count,
        seed=args.seed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
