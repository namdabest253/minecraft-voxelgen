"""Schematic exporter for WorldEdit/FastAsyncWorldEdit (.schem format)."""

from pathlib import Path
from typing import Dict, List, Union, Set
import struct

import nbtlib
from nbtlib import Int, Short, Compound, List as NBTList, ByteArray
import numpy as np
import torch


def encode_varint(value: int) -> bytes:
    """Encode integer as variable-length integer (varint) for schematic format."""
    result = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value != 0:
            byte |= 0x80
        result.append(byte)
        if value == 0:
            break
    return bytes(result)


def export_to_schematic(
    structure: Union[np.ndarray, torch.Tensor],
    tok2block: Dict[int, str],
    output_path: Union[str, Path],
    width: int = 32,
    height: int = 32,
    length: int = 32,
    air_tokens: Set[int] = None,
    offset: tuple = (0, 0, 0),
) -> None:
    """Export structure to Sponge Schematic v2 format (.schem) for WorldEdit.

    Args:
        structure: 3D array of token IDs, shape (X, Y, Z)
        tok2block: Token ID to block name mapping
        output_path: Path to save .schem file
        width: Structure width (X dimension)
        height: Structure height (Y dimension)
        length: Structure length (Z dimension)
        air_tokens: Set of air token IDs
        offset: Schematic offset (X, Y, Z)
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

    # Build palette: map block names to palette indices
    palette = {}
    palette_max = 0

    # Always include air as index 0
    palette["minecraft:air"] = 0
    palette_max = 1

    # Scan structure to build palette
    unique_tokens = np.unique(structure)
    for token_id in unique_tokens:
        if int(token_id) in air_tokens:
            continue  # Air already added

        # tok2block has string keys, so convert token_id to string
        block_name = tok2block.get(str(int(token_id)), "minecraft:stone")
        if ":" not in block_name:
            block_name = f"minecraft:{block_name}"

        # Remove block states for schematic format (WorldEdit handles this)
        if "[" in block_name:
            block_name = block_name.split("[")[0]

        if block_name not in palette:
            palette[block_name] = palette_max
            palette_max += 1

    # Build block data array (Y, Z, X order for Minecraft)
    total_blocks = width * height * length
    non_air_count = 0

    # Build flat array of palette indices first
    flat_indices = []
    for y in range(height):
        for z in range(length):
            for x in range(width):
                token_id = int(structure[x, y, z])

                # Determine palette index
                if token_id in air_tokens:
                    palette_idx = 0  # Air
                else:
                    # tok2block has string keys
                    block_name = tok2block.get(str(token_id), "minecraft:stone")
                    if ":" not in block_name:
                        block_name = f"minecraft:{block_name}"
                    # Remove block states
                    if "[" in block_name:
                        block_name = block_name.split("[")[0]

                    palette_idx = palette.get(block_name, 0)
                    if palette_idx != 0:
                        non_air_count += 1

                flat_indices.append(palette_idx)

    # Encode based on palette size
    if palette_max < 128:
        # Simple case: 1 byte per block
        block_data = ByteArray([int(b) for b in flat_indices])
    else:
        # Varint encoding for large palettes
        encoded = []
        for val in flat_indices:
            val = int(val)
            while val >= 0x80:
                encoded.append((val & 0x7F) | 0x80)
                val >>= 7
            encoded.append(val)
        block_data = ByteArray(encoded)

    # Convert palette dict to NBT format
    palette_nbt = Compound({block_name: Int(idx) for block_name, idx in palette.items()})

    # Create Sponge Schematic v2 structure
    schematic_data = Compound({
        "Version": Int(2),
        "DataVersion": Int(3465),
        "Width": Short(width),
        "Height": Short(height),
        "Length": Short(length),
        "PaletteMax": Int(palette_max),
        "Palette": palette_nbt,
        "BlockData": ByteArray(block_data),
        "BlockEntities": NBTList[Compound]([]),
        "Metadata": Compound({
            "WEOffsetX": Int(0),
            "WEOffsetY": Int(0),
            "WEOffsetZ": Int(0),
        }),
    })

    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # CRITICAL: Use root_name='Schematic' for WorldEdit compatibility
    schematic_file = nbtlib.File(schematic_data, root_name='Schematic', gzipped=True)
    schematic_file.save(output_path)

    density = 100 * non_air_count / total_blocks

    print(f"Saved schematic: {output_path}")
    print(f"  Dimensions: {width}×{height}×{length}")
    print(f"  Palette: {palette_max} unique blocks")
    print(f"  Non-air blocks: {non_air_count:,} / {total_blocks:,} ({density:.1f}%)")
    print(f"\nTo paste in Minecraft:")
    print(f"  1. //schem load {output_path.stem}")
    print(f"  2. Position yourself where you want it")
    print(f"  3. //paste")


def export_batch_to_schematic(
    structures: Union[np.ndarray, torch.Tensor],
    tok2block: Dict[int, str],
    output_dir: Union[str, Path],
    prefix: str = "generated",
    width: int = 32,
    height: int = 32,
    length: int = 32,
    air_tokens: Set[int] = None,
) -> List[Path]:
    """Export batch of structures to schematic files.

    Args:
        structures: 4D array of token IDs, shape (batch, X, Y, Z)
        tok2block: Token ID to block name mapping
        output_dir: Directory to save .schem files
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
        output_path = output_dir / f"{prefix}_{i}.schem"
        export_to_schematic(
            structures[i], tok2block, output_path, width, height, length, air_tokens
        )
        saved_paths.append(output_path)

    return saved_paths
