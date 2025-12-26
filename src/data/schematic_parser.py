"""Parse Minecraft schematic files to numpy arrays."""

from pathlib import Path

import numpy as np


def parse_schematic(path: str | Path) -> np.ndarray:
    """Parse a .schematic file to a 3D numpy array of block IDs.

    Args:
        path: Path to the .schematic file

    Returns:
        3D numpy array of shape (height, depth, width) containing block IDs.
        Block ID 0 represents air.

    Raises:
        FileNotFoundError: If the schematic file doesn't exist
        ValueError: If the file format is invalid

    Example:
        >>> blocks = parse_schematic("builds/castle.schematic")
        >>> print(blocks.shape)
        (50, 30, 40)
        >>> print(blocks.dtype)
        int32
    """
    # TODO: Implement schematic parsing
    # This will use nbtlib to read the NBT format
    raise NotImplementedError("Schematic parser not yet implemented")


def parse_nbt_structure(path: str | Path) -> np.ndarray:
    """Parse a vanilla Minecraft .nbt structure file.

    Args:
        path: Path to the .nbt structure file

    Returns:
        3D numpy array of block IDs

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is invalid
    """
    # TODO: Implement NBT structure parsing
    raise NotImplementedError("NBT structure parser not yet implemented")
