"""
Place an h5 structure in Minecraft via RCON.

Usage:
    python place_structure.py <h5_file> --pos <x> <y> <z>
    python place_structure.py data/splits/train/batch_103_2672.h5 --pos 0 64 0
    python place_structure.py data/splits/train/batch_103_2672.h5 --pos 0 64 0 --clear

Options:
    --pos X Y Z     Position to place the structure (required)
    --clear         Clear the area first (fill with air)
    --dry-run       Print commands without executing
    --batch N       Commands per batch (default: 100)
"""

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np

# RCON connection settings
RCON_HOST = "localhost"
RCON_PORT = 25575
RCON_PASSWORD = "minecraft"


def load_vocabulary(vocab_path: Path) -> dict[int, str]:
    """Load token to block name mapping."""
    with open(vocab_path, "r") as f:
        return {int(k): v for k, v in json.load(f).items()}


def load_structure(h5_path: Path) -> np.ndarray:
    """Load structure from h5 file."""
    with h5py.File(h5_path, "r") as f:
        key = list(f.keys())[0]
        return f[key][:]


def token_to_block_name(token: int, tok2block: dict[int, str]) -> str:
    """Convert token ID to Minecraft block name."""
    name = tok2block.get(token, "minecraft:air")
    # Remove minecraft: prefix if present for setblock command
    # Actually setblock needs it, so keep it
    return name


def is_air(block_name: str) -> bool:
    """Check if block is air."""
    return "air" in block_name.lower() and "airs" not in block_name.lower()


def generate_setblock_commands(
    build: np.ndarray,
    tok2block: dict[int, str],
    base_x: int,
    base_y: int,
    base_z: int,
) -> list[str]:
    """Generate setblock commands for all non-air blocks."""
    commands = []
    h, w, d = build.shape  # y, x, z in our format

    for y in range(h):
        for x in range(w):
            for z in range(d):
                token = int(build[y, x, z])
                block_name = token_to_block_name(token, tok2block)

                if is_air(block_name):
                    continue

                # Calculate world position
                world_x = base_x + x
                world_y = base_y + y
                world_z = base_z + z

                cmd = f"setblock {world_x} {world_y} {world_z} {block_name}"
                commands.append(cmd)

    return commands


def connect_rcon():
    """Connect to RCON server."""
    try:
        from mcrcon import MCRcon
        return MCRcon(RCON_HOST, RCON_PASSWORD, port=RCON_PORT)
    except ImportError:
        print("ERROR: mcrcon not installed. Run: pip install mcrcon")
        return None


def execute_commands(commands: list[str], batch_size: int = 100, dry_run: bool = False):
    """Execute commands via RCON."""
    if dry_run:
        print(f"Dry run - would execute {len(commands)} commands:")
        for cmd in commands[:10]:
            print(f"  /{cmd}")
        if len(commands) > 10:
            print(f"  ... and {len(commands) - 10} more")
        return

    rcon = connect_rcon()
    if rcon is None:
        return

    try:
        rcon.connect()
        print(f"Connected to RCON at {RCON_HOST}:{RCON_PORT}")

        total = len(commands)
        errors = 0

        for i, cmd in enumerate(commands):
            try:
                response = rcon.command(cmd)
                if "error" in response.lower() or "unknown" in response.lower():
                    errors += 1
                    if errors <= 5:
                        print(f"  Warning: {cmd[:50]}... -> {response}")
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  Error: {cmd[:50]}... -> {e}")

            # Progress update
            if (i + 1) % batch_size == 0 or i == total - 1:
                pct = 100 * (i + 1) / total
                print(f"  Progress: {i + 1}/{total} ({pct:.1f}%) - {errors} errors")

            # Small delay to avoid overwhelming the server
            if (i + 1) % 50 == 0:
                time.sleep(0.1)

        print(f"Done! Placed {total - errors}/{total} blocks")

    finally:
        rcon.disconnect()


def clear_area(base_x: int, base_y: int, base_z: int, size: int = 32, dry_run: bool = False):
    """Clear the area before placing structure."""
    cmd = f"fill {base_x} {base_y} {base_z} {base_x + size - 1} {base_y + size - 1} {base_z + size - 1} minecraft:air"

    if dry_run:
        print(f"Would clear area: /{cmd}")
        return

    rcon = connect_rcon()
    if rcon is None:
        return

    try:
        rcon.connect()
        response = rcon.command(cmd)
        print(f"Cleared area: {response}")
    finally:
        rcon.disconnect()


def main():
    parser = argparse.ArgumentParser(description="Place h5 structure in Minecraft")
    parser.add_argument("h5_file", type=Path, help="Path to h5 file")
    parser.add_argument("--pos", type=int, nargs=3, required=True, metavar=("X", "Y", "Z"),
                        help="Position to place structure")
    parser.add_argument("--clear", action="store_true", help="Clear area first")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--batch", type=int, default=100, help="Commands per progress update")
    parser.add_argument("--vocab", type=Path, default=None, help="Path to vocabulary file")

    args = parser.parse_args()

    # Find vocabulary file
    script_dir = Path(__file__).parent.parent
    if args.vocab:
        vocab_path = args.vocab
    else:
        vocab_path = script_dir / "data" / "vocabulary" / "tok2block.json"
        if not vocab_path.exists():
            vocab_path = script_dir / "data" / "kaggle" / "input" / "v1" / "tok2block.json"

    if not vocab_path.exists():
        print(f"ERROR: Vocabulary file not found at {vocab_path}")
        print("Specify with --vocab <path>")
        return

    if not args.h5_file.exists():
        print(f"ERROR: H5 file not found at {args.h5_file}")
        return

    print(f"Loading vocabulary from {vocab_path}")
    tok2block = load_vocabulary(vocab_path)
    print(f"  {len(tok2block)} blocks in vocabulary")

    print(f"Loading structure from {args.h5_file}")
    build = load_structure(args.h5_file)
    print(f"  Shape: {build.shape}")

    base_x, base_y, base_z = args.pos
    print(f"Placing at position: ({base_x}, {base_y}, {base_z})")

    # Clear area if requested
    if args.clear:
        print("Clearing area...")
        clear_area(base_x, base_y, base_z, size=max(build.shape), dry_run=args.dry_run)

    # Generate commands
    print("Generating setblock commands...")
    commands = generate_setblock_commands(build, tok2block, base_x, base_y, base_z)
    print(f"  {len(commands)} non-air blocks to place")

    # Execute
    print("Executing commands...")
    execute_commands(commands, batch_size=args.batch, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
