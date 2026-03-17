"""NBT structure exporter for Minecraft."""

from pathlib import Path
from typing import List, Optional, Union

import nbtlib
import numpy as np
import torch


def export_to_nbt(
    structure: Union[np.ndarray, torch.Tensor],
    tok2block: dict,
    output_path: Union[str, Path],
    width: int = 32,
    height: int = 32,
    depth: int = 32,
    air_tokens: set = None,
) -> None:
    """Export a 3D structure to Minecraft NBT format.

    Args:
        structure: 3D array of token IDs, shape (W, H, D) or flattened (W*H*D,)
        tok2block: Dictionary mapping token IDs to block names
        output_path: Path to save .nbt file
        width: Structure width (X dimension)
        height: Structure height (Y dimension)
        depth: Structure depth (Z dimension)
        air_tokens: Set of token IDs representing air blocks (default: {102, 576, 3352})
    """
    # Convert to numpy if tensor
    if isinstance(structure, torch.Tensor):
        structure = structure.cpu().numpy()

    # Reshape if flattened
    if structure.ndim == 1:
        structure = structure.reshape(width, height, depth)

    # Ensure correct shape
    assert structure.shape == (width, height, depth), (
        f"Structure shape {structure.shape} doesn't match "
        f"dimensions ({width}, {height}, {depth})"
    )

    # Default air tokens
    if air_tokens is None:
        air_tokens = {102, 576, 3352}  # minecraft:air, cave_air, void_air

    # Build palette: unique NON-AIR blocks in structure
    unique_tokens = np.unique(structure)
    palette = []
    token_to_palette_idx = {}

    for token_id in unique_tokens:
        # Skip air blocks in palette
        if int(token_id) in air_tokens:
            continue

        block_name = tok2block.get(int(token_id), "minecraft:stone")
        # Ensure proper namespace
        if ":" not in block_name:
            block_name = f"minecraft:{block_name}"

        palette_idx = len(palette)
        token_to_palette_idx[int(token_id)] = palette_idx
        palette.append(nbtlib.Compound({"Name": nbtlib.String(block_name)}))

    # Build block list - ONLY non-air blocks (critical optimization)
    blocks = []
    for y in range(height):
        for z in range(depth):
            for x in range(width):
                token_id = int(structure[x, y, z])

                # Skip air blocks - this is the key fix!
                if token_id in air_tokens:
                    continue

                palette_idx = token_to_palette_idx.get(token_id)
                if palette_idx is None:
                    continue  # Skip unknown blocks

                blocks.append(
                    nbtlib.Compound(
                        {
                            "pos": nbtlib.List[nbtlib.Int](
                                [nbtlib.Int(x), nbtlib.Int(y), nbtlib.Int(z)]
                            ),
                            "state": nbtlib.Int(palette_idx),
                        }
                    )
                )

    # Create NBT structure
    nbt_data = nbtlib.Compound(
        {
            "size": nbtlib.List[nbtlib.Int](
                [nbtlib.Int(width), nbtlib.Int(height), nbtlib.Int(depth)]
            ),
            "palette": nbtlib.List[nbtlib.Compound](palette),
            "blocks": nbtlib.List[nbtlib.Compound](blocks),
            "entities": nbtlib.List[nbtlib.Compound]([]),
            "DataVersion": nbtlib.Int(3953),  # Minecraft 1.21.1
        }
    )

    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nbt_file = nbtlib.File({"": nbt_data}, gzipped=True)
    nbt_file.save(output_path)

    print(f"Saved NBT structure: {output_path}")
    print(f"  Size: {width}×{height}×{depth}")
    print(f"  Palette: {len(palette)} unique blocks (air excluded)")
    print(f"  Non-air blocks: {len(blocks)} / {width*height*depth} ({100*len(blocks)/(width*height*depth):.1f}%)")


def export_batch_to_nbt(
    structures: Union[np.ndarray, torch.Tensor],
    tok2block: dict,
    output_dir: Union[str, Path],
    prefix: str = "generated",
    width: int = 32,
    height: int = 32,
    depth: int = 32,
    air_tokens: set = None,
) -> List[Path]:
    """Export a batch of structures to NBT files.

    Args:
        structures: 4D array of token IDs, shape (batch, W, H, D)
        tok2block: Dictionary mapping token IDs to block names
        output_dir: Directory to save .nbt files
        prefix: Filename prefix
        width: Structure width
        height: Structure height
        depth: Structure depth
        air_tokens: Set of token IDs representing air blocks

    Returns:
        List of paths to saved files
    """
    if isinstance(structures, torch.Tensor):
        structures = structures.cpu().numpy()

    batch_size = structures.shape[0]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for i in range(batch_size):
        output_path = output_dir / f"{prefix}_{i}.nbt"
        export_to_nbt(structures[i], tok2block, output_path, width, height, depth, air_tokens)
        saved_paths.append(output_path)

    return saved_paths
