"""
Block State Collapsing Utility for Block2Vec V2.

Collapses block state variants into base block types while preserving
semantically meaningful states (water level, bed parts, etc.).

Example:
    minecraft:oak_stairs[facing=north,half=bottom] -> minecraft:oak_stairs
    minecraft:water[level=0] -> minecraft:water[level=0]  # preserved
    minecraft:bed[part=head] -> minecraft:bed[part=head]  # preserved
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional


# States that should be PRESERVED (not collapsed)
# These represent semantically different blocks, not just orientation
MEANINGFUL_STATES = {
    # Water/lava levels - source vs flowing
    "water": ["level"],
    "lava": ["level"],

    # Multi-block structures
    "bed": ["part"],  # head vs foot
    "tall_grass": ["half"],  # upper vs lower
    "tall_seagrass": ["half"],
    "large_fern": ["half"],
    "sunflower": ["half"],
    "lilac": ["half"],
    "rose_bush": ["half"],
    "peony": ["half"],

    # Doors (half matters for multi-block)
    "oak_door": ["half"],
    "spruce_door": ["half"],
    "birch_door": ["half"],
    "jungle_door": ["half"],
    "acacia_door": ["half"],
    "dark_oak_door": ["half"],
    "mangrove_door": ["half"],
    "cherry_door": ["half"],
    "bamboo_door": ["half"],
    "crimson_door": ["half"],
    "warped_door": ["half"],
    "iron_door": ["half"],

    # Pistons - extended state matters
    "piston": ["extended"],
    "sticky_piston": ["extended"],

    # Redstone - power level might matter
    # Actually, for building reconstruction, power level doesn't matter
    # "redstone_wire": ["power"],

    # Chorus plant connections
    "chorus_plant": ["up", "down", "north", "south", "east", "west"],

    # Snow layers - height matters
    "snow": ["layers"],

    # Candles - count matters
    "candle": ["candles"],
    "white_candle": ["candles"],
    "orange_candle": ["candles"],
    "magenta_candle": ["candles"],
    "light_blue_candle": ["candles"],
    "yellow_candle": ["candles"],
    "lime_candle": ["candles"],
    "pink_candle": ["candles"],
    "gray_candle": ["candles"],
    "light_gray_candle": ["candles"],
    "cyan_candle": ["candles"],
    "purple_candle": ["candles"],
    "blue_candle": ["candles"],
    "brown_candle": ["candles"],
    "green_candle": ["candles"],
    "red_candle": ["candles"],
    "black_candle": ["candles"],

    # Turtle eggs - count matters
    "turtle_egg": ["eggs"],

    # Sea pickles - count matters
    "sea_pickle": ["pickles"],
}


def parse_block_string(block_string: str) -> tuple[str, dict[str, str]]:
    """Parse a block string into base name and state dict.

    Args:
        block_string: e.g., "minecraft:oak_stairs[facing=north,half=bottom]"

    Returns:
        Tuple of (base_name, state_dict)
        e.g., ("minecraft:oak_stairs", {"facing": "north", "half": "bottom"})
    """
    # Check for state
    if "[" in block_string:
        base_name = block_string[:block_string.index("[")]
        state_str = block_string[block_string.index("[")+1:block_string.rindex("]")]

        # Parse states
        states = {}
        for pair in state_str.split(","):
            if "=" in pair:
                key, value = pair.split("=", 1)
                states[key] = value

        return base_name, states
    else:
        return block_string, {}


def get_short_name(base_name: str) -> str:
    """Get the short block name without namespace.

    Args:
        base_name: e.g., "minecraft:oak_stairs"

    Returns:
        Short name, e.g., "oak_stairs"
    """
    if ":" in base_name:
        return base_name.split(":", 1)[1]
    return base_name


def collapse_block(block_string: str) -> str:
    """Collapse a block string to its base form, preserving meaningful states.

    Args:
        block_string: Full block string with states

    Returns:
        Collapsed block string
    """
    base_name, states = parse_block_string(block_string)
    short_name = get_short_name(base_name)

    # Check if this block has meaningful states to preserve
    preserved_states = {}
    if short_name in MEANINGFUL_STATES:
        for state_key in MEANINGFUL_STATES[short_name]:
            if state_key in states:
                preserved_states[state_key] = states[state_key]

    # Reconstruct block string
    if preserved_states:
        state_str = ",".join(f"{k}={v}" for k, v in sorted(preserved_states.items()))
        return f"{base_name}[{state_str}]"
    else:
        return base_name


def create_collapsed_vocabulary(
    tok2block: dict[str, str],
    verbose: bool = False
) -> tuple[dict[str, str], dict[int, int], dict[str, list[int]]]:
    """Create a collapsed vocabulary from the original.

    Args:
        tok2block: Original token -> block mapping
        verbose: Print detailed info

    Returns:
        Tuple of:
            - collapsed_tok2block: New token -> collapsed block mapping
            - original_to_collapsed: Original token ID -> collapsed token ID
            - collapsed_to_originals: Collapsed block -> list of original token IDs
    """
    # Step 1: Collapse all blocks
    block_to_collapsed = {}
    for token_id, block_string in tok2block.items():
        collapsed = collapse_block(block_string)
        block_to_collapsed[block_string] = collapsed

    # Step 2: Create new vocabulary (unique collapsed blocks)
    unique_collapsed = sorted(set(block_to_collapsed.values()))
    collapsed_to_new_id = {block: i for i, block in enumerate(unique_collapsed)}

    # Step 3: Create mappings
    collapsed_tok2block = {str(i): block for i, block in enumerate(unique_collapsed)}

    original_to_collapsed = {}
    collapsed_to_originals = defaultdict(list)

    for token_id, block_string in tok2block.items():
        original_id = int(token_id)
        collapsed_block = block_to_collapsed[block_string]
        collapsed_id = collapsed_to_new_id[collapsed_block]

        original_to_collapsed[original_id] = collapsed_id
        collapsed_to_originals[collapsed_block].append(original_id)

    if verbose:
        print(f"Original vocabulary: {len(tok2block)} tokens")
        print(f"Collapsed vocabulary: {len(unique_collapsed)} tokens")
        print(f"Compression ratio: {len(tok2block) / len(unique_collapsed):.1f}x")

        # Show some examples
        print("\nExample collapses:")
        examples_shown = 0
        for orig_id, block in list(tok2block.items())[:500]:
            collapsed = block_to_collapsed[block]
            if collapsed != block and examples_shown < 10:
                print(f"  {block}")
                print(f"    -> {collapsed}")
                examples_shown += 1

        # Show preserved states
        print("\nPreserved states examples:")
        for block in unique_collapsed:
            if "[" in block and any(k in block for k in ["level=", "part=", "half=", "layers="]):
                print(f"  {block}")
                if len([b for b in unique_collapsed if "[" in b]) > 10:
                    break

    return collapsed_tok2block, original_to_collapsed, dict(collapsed_to_originals)


def main():
    parser = argparse.ArgumentParser(
        description="Collapse block states for Block2Vec V2"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/kaggle/input/tok2block.json",
        help="Path to original tok2block.json"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/kaggle/input/v2",
        help="Directory for output files"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed info"
    )

    args = parser.parse_args()

    # Load original vocabulary
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, "r") as f:
        tok2block = json.load(f)

    print(f"Loaded vocabulary: {len(tok2block)} tokens")

    # Create collapsed vocabulary
    collapsed_tok2block, original_to_collapsed, collapsed_to_originals = \
        create_collapsed_vocabulary(tok2block, verbose=args.verbose)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save collapsed vocabulary
    vocab_path = output_dir / "tok2block_collapsed.json"
    with open(vocab_path, "w") as f:
        json.dump(collapsed_tok2block, f, indent=2)
    print(f"Saved collapsed vocabulary: {vocab_path}")

    # Save original -> collapsed mapping
    mapping_path = output_dir / "original_to_collapsed.json"
    with open(mapping_path, "w") as f:
        json.dump({str(k): v for k, v in original_to_collapsed.items()}, f)
    print(f"Saved original->collapsed mapping: {mapping_path}")

    # Save collapsed -> originals mapping
    reverse_path = output_dir / "collapsed_to_originals.json"
    with open(reverse_path, "w") as f:
        json.dump(collapsed_to_originals, f, indent=2)
    print(f"Saved collapsed->originals mapping: {reverse_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Original vocabulary size: {len(tok2block)}")
    print(f"Collapsed vocabulary size: {len(collapsed_tok2block)}")
    print(f"Reduction: {len(tok2block) - len(collapsed_tok2block)} tokens ({100 * (1 - len(collapsed_tok2block) / len(tok2block)):.1f}%)")

    # Find air token in collapsed vocab
    for token_id, block in collapsed_tok2block.items():
        if block == "minecraft:air":
            print(f"Air token ID in collapsed vocab: {token_id}")
            break


if __name__ == "__main__":
    main()
