"""Legacy MCEdit/WorldEdit schematic exporter (.schematic format)."""

from pathlib import Path
from typing import Dict, List, Union, Set

import nbtlib
import numpy as np
import torch


# Block name to numeric ID mapping (Minecraft 1.12 and earlier)
# For FAWE, we'll use modern block names but in the old format
def get_block_id_and_data(block_name: str) -> tuple:
    """Convert modern block name to legacy ID format.

    For simplicity, we'll use a limited palette and let WorldEdit handle conversion.
    """
    # Strip minecraft: prefix
    if ":" in block_name:
        block_name = block_name.split(":")[1]

    # Strip block states
    if "[" in block_name:
        block_name = block_name.split("[")[0]

    # Common blocks mapping (simplified)
    common_blocks = {
        "air": (0, 0),
        "stone": (1, 0),
        "grass_block": (2, 0),
        "dirt": (3, 0),
        "cobblestone": (4, 0),
        "oak_planks": (5, 0),
        "oak_wood": (17, 0),
        "birch_wood": (17, 2),
        "spruce_wood": (17, 1),
        "jungle_wood": (17, 3),
        "acacia_wood": (162, 0),
        "dark_oak_wood": (162, 1),
        "glass": (20, 0),
        "sandstone": (24, 0),
        "wool": (35, 0),
        "gold_block": (41, 0),
        "iron_block": (42, 0),
        "brick": (45, 0),
        "tnt": (46, 0),
        "bookshelf": (47, 0),
        "mossy_cobblestone": (48, 0),
        "obsidian": (49, 0),
        "diamond_block": (57, 0),
        "crafting_table": (58, 0),
        "snow": (80, 0),
        "clay": (82, 0),
        "pumpkin": (86, 0),
        "netherrack": (87, 0),
        "glowstone": (89, 0),
        "stone_bricks": (98, 0),
        "quartz_block": (155, 0),
        "quartz_slab": (44, 7),
        "quartz_pillar": (155, 2),
        "coal_block": (173, 0),
        "packed_ice": (174, 0),
        "acacia_planks": (5, 4),
        "birch_planks": (5, 2),
        "spruce_planks": (5, 1),
        "jungle_planks": (5, 3),
        "dark_oak_planks": (5, 5),
        "black_stained_glass": (95, 15),
        "black_stained_glass_pane": (160, 15),
        "bamboo_planks": (5, 0),  # Fallback to oak
        "blackstone": (1, 0),  # Fallback to stone
        "blackstone_slab": (44, 0),  # Stone slab
        "blackstone_stairs": (109, 0),  # Stone brick stairs
    }

    return common_blocks.get(block_name, (1, 0))  # Default to stone


def export_to_legacy_schematic(
    structure: Union[np.ndarray, torch.Tensor],
    tok2block: Dict[int, str],
    output_path: Union[str, Path],
    width: int = 32,
    height: int = 32,
    length: int = 32,
    air_tokens: Set[int] = None,
) -> None:
    """Export structure to legacy MCEdit/WorldEdit schematic format.

    Args:
        structure: 3D array of token IDs, shape (X, Y, Z)
        tok2block: Token ID to block name mapping
        output_path: Path to save .schematic file
        width: Structure width (X dimension)
        height: Structure height (Y dimension)
        length: Structure length (Z dimension)
        air_tokens: Set of air token IDs
    """
    # Convert to numpy if tensor
    if isinstance(structure, torch.Tensor):
        structure = structure.cpu().numpy()

    # Reshape if flattened
    if structure.ndim == 1:
        structure = structure.reshape(width, height, length)

    # Ensure correct shape
    assert structure.shape == (width, height, length), (
        f"Structure shape {structure.shape} doesn't match "
        f"dimensions ({width}, {height}, {length})"
    )

    # Default air tokens
    if air_tokens is None:
        air_tokens = {102, 576, 3352}

    # Build block ID and data arrays (Y, Z, X order for Minecraft)
    total_blocks = width * height * length
    blocks_array = bytearray(total_blocks)
    data_array = bytearray(total_blocks)
    non_air_count = 0

    idx = 0
    for y in range(height):
        for z in range(length):
            for x in range(width):
                token_id = int(structure[x, y, z])

                # Get block info
                if token_id in air_tokens:
                    block_id, block_data = 0, 0
                else:
                    block_name = tok2block.get(token_id, "minecraft:stone")
                    if ":" in block_name:
                        block_name = block_name.split(":")[1]
                    if "[" in block_name:
                        block_name = block_name.split("[")[0]

                    block_id, block_data = get_block_id_and_data(block_name)
                    non_air_count += 1

                blocks_array[idx] = block_id & 0xFF
                data_array[idx] = block_data & 0xF
                idx += 1

    # Create legacy schematic NBT structure
    schematic_data = nbtlib.Compound(
        {
            "Width": nbtlib.Short(width),
            "Height": nbtlib.Short(height),
            "Length": nbtlib.Short(length),
            "Materials": nbtlib.String("Alpha"),
            "Blocks": nbtlib.ByteArray(blocks_array),
            "Data": nbtlib.ByteArray(data_array),
            "Entities": nbtlib.List[nbtlib.Compound]([]),
            "TileEntities": nbtlib.List[nbtlib.Compound]([]),
        }
    )

    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Legacy schematics use root tag "Schematic"
    schematic_file = nbtlib.File({"Schematic": schematic_data}, gzipped=True)
    schematic_file.save(output_path)

    density = 100 * non_air_count / total_blocks

    print(f"Saved legacy schematic: {output_path}")
    print(f"  Dimensions: {width}×{height}×{length}")
    print(f"  Non-air blocks: {non_air_count:,} / {total_blocks:,} ({density:.1f}%)")
    print(f"\nTo paste in Minecraft:")
    print(f"  1. //schem load {output_path.stem}")
    print(f"  2. Position yourself where you want it")
    print(f"  3. //paste")


def export_batch_to_legacy_schematic(
    structures: Union[np.ndarray, torch.Tensor],
    tok2block: Dict[int, str],
    output_dir: Union[str, Path],
    prefix: str = "generated",
    width: int = 32,
    height: int = 32,
    length: int = 32,
    air_tokens: Set[int] = None,
) -> List[Path]:
    """Export batch of structures to legacy schematic files.

    Args:
        structures: 4D array of token IDs, shape (batch, X, Y, Z)
        tok2block: Token ID to block name mapping
        output_dir: Directory to save .schematic files
        prefix: Filename prefix
        width: Structure width
        height: Structure height
        length: Structure length
        air_tokens: Set of air token IDs

    Returns:
        List of paths to saved schematic files
    """
    if isinstance(structures, torch.Tensor):
        structures = structures.cpu().numpy()

    batch_size = structures.shape[0]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for i in range(batch_size):
        output_path = output_dir / f"{prefix}_{i}.schematic"
        export_to_legacy_schematic(
            structures[i], tok2block, output_path, width, height, length, air_tokens
        )
        saved_paths.append(output_path)

    return saved_paths
