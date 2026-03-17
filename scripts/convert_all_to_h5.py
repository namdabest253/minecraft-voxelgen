"""
Convert .schem, .schematic, and .litematic files to .h5 format for training.

Handles three Minecraft structure file formats:
- .schem (Sponge Schematic v2): Modern WorldEdit format with block name palette
- .schematic (MCEdit/old WorldEdit): Legacy format with numeric block IDs (pre-1.13)
- .litematic (Litematica): Region-based format with bit-packed block states

Pipeline:
1. Scan input directories for all supported files
2. Read each file into a 3D block array
3. Filter by size, diversity, and air fraction
4. Normalize to 32x32x32 (center + crop/pad)
5. Save as .h5 with gzip compression

Usage:
    python minecraft_ai/scripts/convert_all_to_h5.py
    python minecraft_ai/scripts/convert_all_to_h5.py --dry-run
    python minecraft_ai/scripts/convert_all_to_h5.py --max-files 100
"""

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import h5py
import numpy as np
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse existing utilities
from scripts.reconstruct_schematic import (
    AIR_TOKEN,
    read_schematic as read_schem_file,
    schematic_to_tokens,
)

# Air tokens in our vocabulary
AIR_TOKENS = {102, 576, 3352}  # air, cave_air, void_air


# ============================================================
# Legacy numeric block ID -> modern block name mapping
# ============================================================

# Pre-1.13 Minecraft block IDs. Data values are ignored for simplicity
# (we map to the base block state). This covers IDs 0-255.
LEGACY_BLOCK_MAP: Dict[int, str] = {
    0: "minecraft:air",
    1: "minecraft:stone",
    2: "minecraft:grass_block",
    3: "minecraft:dirt",
    4: "minecraft:cobblestone",
    5: "minecraft:oak_planks",
    6: "minecraft:oak_sapling",
    7: "minecraft:bedrock",
    8: "minecraft:water",
    9: "minecraft:water",
    10: "minecraft:lava",
    11: "minecraft:lava",
    12: "minecraft:sand",
    13: "minecraft:gravel",
    14: "minecraft:gold_ore",
    15: "minecraft:iron_ore",
    16: "minecraft:coal_ore",
    17: "minecraft:oak_log",
    18: "minecraft:oak_leaves",
    19: "minecraft:sponge",
    20: "minecraft:glass",
    21: "minecraft:lapis_ore",
    22: "minecraft:lapis_block",
    23: "minecraft:dispenser",
    24: "minecraft:sandstone",
    25: "minecraft:note_block",
    26: "minecraft:red_bed",
    27: "minecraft:powered_rail",
    28: "minecraft:detector_rail",
    29: "minecraft:sticky_piston",
    30: "minecraft:cobweb",
    31: "minecraft:short_grass",
    32: "minecraft:dead_bush",
    33: "minecraft:piston",
    34: "minecraft:piston_head",
    35: "minecraft:white_wool",
    36: "minecraft:moving_piston",
    37: "minecraft:dandelion",
    38: "minecraft:poppy",
    39: "minecraft:brown_mushroom",
    40: "minecraft:red_mushroom",
    41: "minecraft:gold_block",
    42: "minecraft:iron_block",
    43: "minecraft:smooth_stone_slab[type=double]",
    44: "minecraft:smooth_stone_slab",
    45: "minecraft:bricks",
    46: "minecraft:tnt",
    47: "minecraft:bookshelf",
    48: "minecraft:mossy_cobblestone",
    49: "minecraft:obsidian",
    50: "minecraft:wall_torch",
    51: "minecraft:fire",
    52: "minecraft:spawner",
    53: "minecraft:oak_stairs",
    54: "minecraft:chest",
    55: "minecraft:redstone_wire",
    56: "minecraft:diamond_ore",
    57: "minecraft:diamond_block",
    58: "minecraft:crafting_table",
    59: "minecraft:wheat",
    60: "minecraft:farmland",
    61: "minecraft:furnace",
    62: "minecraft:furnace",
    63: "minecraft:oak_sign",
    64: "minecraft:oak_door",
    65: "minecraft:ladder",
    66: "minecraft:rail",
    67: "minecraft:cobblestone_stairs",
    68: "minecraft:oak_wall_sign",
    69: "minecraft:lever",
    70: "minecraft:stone_pressure_plate",
    71: "minecraft:iron_door",
    72: "minecraft:oak_pressure_plate",
    73: "minecraft:redstone_ore",
    74: "minecraft:redstone_ore",
    75: "minecraft:redstone_wall_torch",
    76: "minecraft:redstone_wall_torch",
    77: "minecraft:stone_button",
    78: "minecraft:snow",
    79: "minecraft:ice",
    80: "minecraft:snow_block",
    81: "minecraft:cactus",
    82: "minecraft:clay",
    83: "minecraft:sugar_cane",
    84: "minecraft:jukebox",
    85: "minecraft:oak_fence",
    86: "minecraft:carved_pumpkin",
    87: "minecraft:netherrack",
    88: "minecraft:soul_sand",
    89: "minecraft:glowstone",
    90: "minecraft:nether_portal",
    91: "minecraft:jack_o_lantern",
    92: "minecraft:cake",
    93: "minecraft:repeater",
    94: "minecraft:repeater",
    95: "minecraft:white_stained_glass",
    96: "minecraft:oak_trapdoor",
    97: "minecraft:infested_stone",
    98: "minecraft:stone_bricks",
    99: "minecraft:brown_mushroom_block",
    100: "minecraft:red_mushroom_block",
    101: "minecraft:iron_bars",
    102: "minecraft:glass_pane",
    103: "minecraft:melon",
    104: "minecraft:pumpkin_stem",
    105: "minecraft:melon_stem",
    106: "minecraft:vine",
    107: "minecraft:oak_fence_gate",
    108: "minecraft:brick_stairs",
    109: "minecraft:stone_brick_stairs",
    110: "minecraft:mycelium",
    111: "minecraft:lily_pad",
    112: "minecraft:nether_bricks",
    113: "minecraft:nether_brick_fence",
    114: "minecraft:nether_brick_stairs",
    115: "minecraft:nether_wart",
    116: "minecraft:enchanting_table",
    117: "minecraft:brewing_stand",
    118: "minecraft:cauldron",
    119: "minecraft:end_portal",
    120: "minecraft:end_portal_frame",
    121: "minecraft:end_stone",
    122: "minecraft:dragon_egg",
    123: "minecraft:redstone_lamp",
    124: "minecraft:redstone_lamp",
    125: "minecraft:oak_slab[type=double]",
    126: "minecraft:oak_slab",
    127: "minecraft:cocoa",
    128: "minecraft:sandstone_stairs",
    129: "minecraft:emerald_ore",
    130: "minecraft:ender_chest",
    131: "minecraft:tripwire_hook",
    132: "minecraft:tripwire",
    133: "minecraft:emerald_block",
    134: "minecraft:spruce_stairs",
    135: "minecraft:birch_stairs",
    136: "minecraft:jungle_stairs",
    137: "minecraft:command_block",
    138: "minecraft:beacon",
    139: "minecraft:cobblestone_wall",
    140: "minecraft:flower_pot",
    141: "minecraft:carrots",
    142: "minecraft:potatoes",
    143: "minecraft:oak_button",
    144: "minecraft:skeleton_skull",
    145: "minecraft:anvil",
    146: "minecraft:trapped_chest",
    147: "minecraft:light_weighted_pressure_plate",
    148: "minecraft:heavy_weighted_pressure_plate",
    149: "minecraft:comparator",
    150: "minecraft:comparator",
    151: "minecraft:daylight_detector",
    152: "minecraft:redstone_block",
    153: "minecraft:nether_quartz_ore",
    154: "minecraft:hopper",
    155: "minecraft:quartz_block",
    156: "minecraft:quartz_stairs",
    157: "minecraft:activator_rail",
    158: "minecraft:dropper",
    159: "minecraft:white_terracotta",
    160: "minecraft:white_stained_glass_pane",
    161: "minecraft:acacia_leaves",
    162: "minecraft:acacia_log",
    163: "minecraft:acacia_stairs",
    164: "minecraft:dark_oak_stairs",
    165: "minecraft:slime_block",
    166: "minecraft:barrier",
    167: "minecraft:iron_trapdoor",
    168: "minecraft:prismarine",
    169: "minecraft:sea_lantern",
    170: "minecraft:hay_block",
    171: "minecraft:white_carpet",
    172: "minecraft:terracotta",
    173: "minecraft:coal_block",
    174: "minecraft:packed_ice",
    175: "minecraft:sunflower",
    176: "minecraft:white_banner",
    177: "minecraft:white_wall_banner",
    178: "minecraft:daylight_detector",
    179: "minecraft:red_sandstone",
    180: "minecraft:red_sandstone_stairs",
    181: "minecraft:red_sandstone_slab[type=double]",
    182: "minecraft:red_sandstone_slab",
    183: "minecraft:spruce_fence_gate",
    184: "minecraft:birch_fence_gate",
    185: "minecraft:jungle_fence_gate",
    186: "minecraft:dark_oak_fence_gate",
    187: "minecraft:acacia_fence_gate",
    188: "minecraft:spruce_fence",
    189: "minecraft:birch_fence",
    190: "minecraft:jungle_fence",
    191: "minecraft:dark_oak_fence",
    192: "minecraft:acacia_fence",
    193: "minecraft:spruce_door",
    194: "minecraft:birch_door",
    195: "minecraft:jungle_door",
    196: "minecraft:acacia_door",
    197: "minecraft:dark_oak_door",
    198: "minecraft:end_rod",
    199: "minecraft:chorus_plant",
    200: "minecraft:chorus_flower",
    201: "minecraft:purpur_block",
    202: "minecraft:purpur_pillar",
    203: "minecraft:purpur_stairs",
    204: "minecraft:purpur_slab[type=double]",
    205: "minecraft:purpur_slab",
    206: "minecraft:end_stone_bricks",
    207: "minecraft:beetroots",
    208: "minecraft:dirt_path",
    209: "minecraft:end_gateway",
    210: "minecraft:repeating_command_block",
    211: "minecraft:chain_command_block",
    212: "minecraft:frosted_ice",
    213: "minecraft:magma_block",
    214: "minecraft:nether_wart_block",
    215: "minecraft:red_nether_bricks",
    216: "minecraft:bone_block",
    217: "minecraft:structure_void",
    218: "minecraft:observer",
    219: "minecraft:white_shulker_box",
    220: "minecraft:orange_shulker_box",
    221: "minecraft:magenta_shulker_box",
    222: "minecraft:light_blue_shulker_box",
    223: "minecraft:yellow_shulker_box",
    224: "minecraft:lime_shulker_box",
    225: "minecraft:pink_shulker_box",
    226: "minecraft:gray_shulker_box",
    227: "minecraft:light_gray_shulker_box",
    228: "minecraft:cyan_shulker_box",
    229: "minecraft:purple_shulker_box",
    230: "minecraft:blue_shulker_box",
    231: "minecraft:brown_shulker_box",
    232: "minecraft:green_shulker_box",
    233: "minecraft:red_shulker_box",
    234: "minecraft:black_shulker_box",
    235: "minecraft:white_glazed_terracotta",
    236: "minecraft:orange_glazed_terracotta",
    237: "minecraft:magenta_glazed_terracotta",
    238: "minecraft:light_blue_glazed_terracotta",
    239: "minecraft:yellow_glazed_terracotta",
    240: "minecraft:lime_glazed_terracotta",
    241: "minecraft:pink_glazed_terracotta",
    242: "minecraft:gray_glazed_terracotta",
    243: "minecraft:light_gray_glazed_terracotta",
    244: "minecraft:cyan_glazed_terracotta",
    245: "minecraft:magenta_glazed_terracotta",
    246: "minecraft:blue_glazed_terracotta",
    247: "minecraft:brown_glazed_terracotta",
    248: "minecraft:green_glazed_terracotta",
    249: "minecraft:red_glazed_terracotta",
    250: "minecraft:black_glazed_terracotta",
    251: "minecraft:white_concrete",
    252: "minecraft:white_concrete_powder",
    253: "minecraft:structure_block",
    255: "minecraft:structure_block",
}

# Data value sub-variants for blocks that use data values for color/type
# Format: (base_id, data_value) -> block_name
LEGACY_DATA_MAP: Dict[Tuple[int, int], str] = {
    # Wool colors (35)
    (35, 0): "minecraft:white_wool",
    (35, 1): "minecraft:orange_wool",
    (35, 2): "minecraft:magenta_wool",
    (35, 3): "minecraft:light_blue_wool",
    (35, 4): "minecraft:yellow_wool",
    (35, 5): "minecraft:lime_wool",
    (35, 6): "minecraft:pink_wool",
    (35, 7): "minecraft:gray_wool",
    (35, 8): "minecraft:light_gray_wool",
    (35, 9): "minecraft:cyan_wool",
    (35, 10): "minecraft:purple_wool",
    (35, 11): "minecraft:blue_wool",
    (35, 12): "minecraft:brown_wool",
    (35, 13): "minecraft:green_wool",
    (35, 14): "minecraft:red_wool",
    (35, 15): "minecraft:black_wool",
    # Wood types (5)
    (5, 0): "minecraft:oak_planks",
    (5, 1): "minecraft:spruce_planks",
    (5, 2): "minecraft:birch_planks",
    (5, 3): "minecraft:jungle_planks",
    (5, 4): "minecraft:acacia_planks",
    (5, 5): "minecraft:dark_oak_planks",
    # Log types (17)
    (17, 0): "minecraft:oak_log",
    (17, 1): "minecraft:spruce_log",
    (17, 2): "minecraft:birch_log",
    (17, 3): "minecraft:jungle_log",
    # Log2 types (162)
    (162, 0): "minecraft:acacia_log",
    (162, 1): "minecraft:dark_oak_log",
    # Leaves (18)
    (18, 0): "minecraft:oak_leaves",
    (18, 1): "minecraft:spruce_leaves",
    (18, 2): "minecraft:birch_leaves",
    (18, 3): "minecraft:jungle_leaves",
    # Leaves2 (161)
    (161, 0): "minecraft:acacia_leaves",
    (161, 1): "minecraft:dark_oak_leaves",
    # Sapling (6)
    (6, 0): "minecraft:oak_sapling",
    (6, 1): "minecraft:spruce_sapling",
    (6, 2): "minecraft:birch_sapling",
    (6, 3): "minecraft:jungle_sapling",
    (6, 4): "minecraft:acacia_sapling",
    (6, 5): "minecraft:dark_oak_sapling",
    # Stone types (1)
    (1, 0): "minecraft:stone",
    (1, 1): "minecraft:granite",
    (1, 2): "minecraft:polished_granite",
    (1, 3): "minecraft:diorite",
    (1, 4): "minecraft:polished_diorite",
    (1, 5): "minecraft:andesite",
    (1, 6): "minecraft:polished_andesite",
    # Sand (12)
    (12, 0): "minecraft:sand",
    (12, 1): "minecraft:red_sand",
    # Dirt (3)
    (3, 0): "minecraft:dirt",
    (3, 1): "minecraft:coarse_dirt",
    (3, 2): "minecraft:podzol",
    # Sandstone (24)
    (24, 0): "minecraft:sandstone",
    (24, 1): "minecraft:chiseled_sandstone",
    (24, 2): "minecraft:cut_sandstone",
    # Stone bricks (98)
    (98, 0): "minecraft:stone_bricks",
    (98, 1): "minecraft:mossy_stone_bricks",
    (98, 2): "minecraft:cracked_stone_bricks",
    (98, 3): "minecraft:chiseled_stone_bricks",
    # Quartz block (155)
    (155, 0): "minecraft:quartz_block",
    (155, 1): "minecraft:chiseled_quartz_block",
    (155, 2): "minecraft:quartz_pillar",
    # Prismarine (168)
    (168, 0): "minecraft:prismarine",
    (168, 1): "minecraft:prismarine_bricks",
    (168, 2): "minecraft:dark_prismarine",
    # Terracotta colors (159)
    (159, 0): "minecraft:white_terracotta",
    (159, 1): "minecraft:orange_terracotta",
    (159, 2): "minecraft:magenta_terracotta",
    (159, 3): "minecraft:light_blue_terracotta",
    (159, 4): "minecraft:yellow_terracotta",
    (159, 5): "minecraft:lime_terracotta",
    (159, 6): "minecraft:pink_terracotta",
    (159, 7): "minecraft:gray_terracotta",
    (159, 8): "minecraft:light_gray_terracotta",
    (159, 9): "minecraft:cyan_terracotta",
    (159, 10): "minecraft:purple_terracotta",
    (159, 11): "minecraft:blue_terracotta",
    (159, 12): "minecraft:brown_terracotta",
    (159, 13): "minecraft:green_terracotta",
    (159, 14): "minecraft:red_terracotta",
    (159, 15): "minecraft:black_terracotta",
    # Stained glass (95)
    (95, 0): "minecraft:white_stained_glass",
    (95, 1): "minecraft:orange_stained_glass",
    (95, 2): "minecraft:magenta_stained_glass",
    (95, 3): "minecraft:light_blue_stained_glass",
    (95, 4): "minecraft:yellow_stained_glass",
    (95, 5): "minecraft:lime_stained_glass",
    (95, 6): "minecraft:pink_stained_glass",
    (95, 7): "minecraft:gray_stained_glass",
    (95, 8): "minecraft:light_gray_stained_glass",
    (95, 9): "minecraft:cyan_stained_glass",
    (95, 10): "minecraft:purple_stained_glass",
    (95, 11): "minecraft:blue_stained_glass",
    (95, 12): "minecraft:brown_stained_glass",
    (95, 13): "minecraft:green_stained_glass",
    (95, 14): "minecraft:red_stained_glass",
    (95, 15): "minecraft:black_stained_glass",
    # Stained glass pane (160)
    (160, 0): "minecraft:white_stained_glass_pane",
    (160, 1): "minecraft:orange_stained_glass_pane",
    (160, 2): "minecraft:magenta_stained_glass_pane",
    (160, 3): "minecraft:light_blue_stained_glass_pane",
    (160, 4): "minecraft:yellow_stained_glass_pane",
    (160, 5): "minecraft:lime_stained_glass_pane",
    (160, 6): "minecraft:pink_stained_glass_pane",
    (160, 7): "minecraft:gray_stained_glass_pane",
    (160, 8): "minecraft:light_gray_stained_glass_pane",
    (160, 9): "minecraft:cyan_stained_glass_pane",
    (160, 10): "minecraft:purple_stained_glass_pane",
    (160, 11): "minecraft:blue_stained_glass_pane",
    (160, 12): "minecraft:brown_stained_glass_pane",
    (160, 13): "minecraft:green_stained_glass_pane",
    (160, 14): "minecraft:red_stained_glass_pane",
    (160, 15): "minecraft:black_stained_glass_pane",
    # Carpet (171)
    (171, 0): "minecraft:white_carpet",
    (171, 1): "minecraft:orange_carpet",
    (171, 2): "minecraft:magenta_carpet",
    (171, 3): "minecraft:light_blue_carpet",
    (171, 4): "minecraft:yellow_carpet",
    (171, 5): "minecraft:lime_carpet",
    (171, 6): "minecraft:pink_carpet",
    (171, 7): "minecraft:gray_carpet",
    (171, 8): "minecraft:light_gray_carpet",
    (171, 9): "minecraft:cyan_carpet",
    (171, 10): "minecraft:purple_carpet",
    (171, 11): "minecraft:blue_carpet",
    (171, 12): "minecraft:brown_carpet",
    (171, 13): "minecraft:green_carpet",
    (171, 14): "minecraft:red_carpet",
    (171, 15): "minecraft:black_carpet",
    # Concrete (251)
    (251, 0): "minecraft:white_concrete",
    (251, 1): "minecraft:orange_concrete",
    (251, 2): "minecraft:magenta_concrete",
    (251, 3): "minecraft:light_blue_concrete",
    (251, 4): "minecraft:yellow_concrete",
    (251, 5): "minecraft:lime_concrete",
    (251, 6): "minecraft:pink_concrete",
    (251, 7): "minecraft:gray_concrete",
    (251, 8): "minecraft:light_gray_concrete",
    (251, 9): "minecraft:cyan_concrete",
    (251, 10): "minecraft:purple_concrete",
    (251, 11): "minecraft:blue_concrete",
    (251, 12): "minecraft:brown_concrete",
    (251, 13): "minecraft:green_concrete",
    (251, 14): "minecraft:red_concrete",
    (251, 15): "minecraft:black_concrete",
    # Concrete powder (252)
    (252, 0): "minecraft:white_concrete_powder",
    (252, 1): "minecraft:orange_concrete_powder",
    (252, 2): "minecraft:magenta_concrete_powder",
    (252, 3): "minecraft:light_blue_concrete_powder",
    (252, 4): "minecraft:yellow_concrete_powder",
    (252, 5): "minecraft:lime_concrete_powder",
    (252, 6): "minecraft:pink_concrete_powder",
    (252, 7): "minecraft:gray_concrete_powder",
    (252, 8): "minecraft:light_gray_concrete_powder",
    (252, 9): "minecraft:cyan_concrete_powder",
    (252, 10): "minecraft:purple_concrete_powder",
    (252, 11): "minecraft:blue_concrete_powder",
    (252, 12): "minecraft:brown_concrete_powder",
    (252, 13): "minecraft:green_concrete_powder",
    (252, 14): "minecraft:red_concrete_powder",
    (252, 15): "minecraft:black_concrete_powder",
    # Red sandstone (179)
    (179, 0): "minecraft:red_sandstone",
    (179, 1): "minecraft:chiseled_red_sandstone",
    (179, 2): "minecraft:cut_red_sandstone",
    # Sponge (19)
    (19, 0): "minecraft:sponge",
    (19, 1): "minecraft:wet_sponge",
    # Double flowers (175)
    (175, 0): "minecraft:sunflower",
    (175, 1): "minecraft:lilac",
    (175, 2): "minecraft:tall_grass",
    (175, 3): "minecraft:large_fern",
    (175, 4): "minecraft:rose_bush",
    (175, 5): "minecraft:peony",
    # Flowers (38)
    (38, 0): "minecraft:poppy",
    (38, 1): "minecraft:blue_orchid",
    (38, 2): "minecraft:allium",
    (38, 3): "minecraft:azure_bluet",
    (38, 4): "minecraft:red_tulip",
    (38, 5): "minecraft:orange_tulip",
    (38, 6): "minecraft:white_tulip",
    (38, 7): "minecraft:pink_tulip",
    (38, 8): "minecraft:oxeye_daisy",
    # Cobblestone wall (139)
    (139, 0): "minecraft:cobblestone_wall",
    (139, 1): "minecraft:mossy_cobblestone_wall",
}


@dataclass
class ConversionConfig:
    """Configuration for structure file to H5 conversion."""

    # Input directories
    input_dirs: List[Path] = field(default_factory=lambda: [
        Path("/mnt/c/Users/namda/Onedrive/Desktop/Claude_Server/schematics"),
        Path("/mnt/c/Users/namda/Onedrive/Desktop/Claude_Server/kaggle/processed_builds"),
    ])

    # Output directory
    output_dir: Path = PROJECT_ROOT / "data" / "processed_new"

    # Vocabulary
    vocab_path: Path = PROJECT_ROOT / "data" / "vocabulary" / "tok2block.json"

    # Filtering
    max_dimension: int = 128  # Skip if any dimension exceeds this
    min_dimension: int = 3  # Skip if all dimensions are below this
    min_unique_blocks: int = 3  # Minimum block type diversity
    max_air_fraction: float = 0.98  # Skip if >98% air after normalization
    min_non_air_voxels: int = 20  # Minimum non-air voxels

    # Normalization
    target_size: Tuple[int, int, int] = (32, 32, 32)
    air_token: int = AIR_TOKEN  # 102

    # Processing
    max_files: int = 0  # 0 = no limit
    skip_existing: bool = True
    dry_run: bool = False


# ============================================================
# Vectorized Block Token Mapping (Optimization #1)
# ============================================================

def _build_legacy_token_lut(block2tok: Dict[str, int]) -> np.ndarray:
    """Build a lookup table for legacy block ID -> token mapping.

    Precomputes all 256 legacy block ID mappings into an array
    for O(1) vectorized lookup instead of dict.get() millions of times.

    Returns:
        Array of shape [256] with token IDs (defaults to air_token).
    """
    lut = np.full(256, AIR_TOKEN, dtype=np.int32)
    for block_id, block_name in LEGACY_BLOCK_MAP.items():
        token = block2tok.get(block_name)
        if token is None:
            base = block_name.split("[")[0]
            token = block2tok.get(base, AIR_TOKEN)
        lut[block_id] = token
    return lut


# ============================================================
# File Readers
# ============================================================


def read_schematic_legacy(
    path: Path, block2tok: Dict[str, int], air_token: int,
    legacy_lut: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """Read a .schematic file (pre-1.13 MCEdit/WorldEdit format).

    Uses vectorized lookup table for block ID → token mapping (Optimization #2).

    Args:
        path: Path to .schematic file.
        block2tok: Modern block name -> token ID mapping.
        air_token: Token ID for air.
        legacy_lut: Optional precomputed legacy block ID lookup table [256].

    Returns:
        tokens: Array of shape (X, Y, Z) with vocabulary token IDs.
        original_size: (X, Y, Z) dimensions.
    """
    import nbtlib

    nbt = nbtlib.load(path)

    width = int(nbt["Width"])  # X
    height = int(nbt["Height"])  # Y
    length = int(nbt["Length"])  # Z

    blocks_raw = np.array(nbt["Blocks"], dtype=np.uint8)
    data_raw = np.array(nbt["Data"], dtype=np.uint8)

    # Stored in YZX order: index = (y * length + z) * width + x
    blocks_yzx = blocks_raw.reshape((height, length, width))
    data_yzx = data_raw.reshape((height, length, width))

    # Transpose to XYZ
    blocks_xyz = blocks_yzx.transpose(2, 0, 1)  # (X, Y, Z)
    data_xyz = data_yzx.transpose(2, 0, 1)

    # Build legacy LUT if not provided
    if legacy_lut is None:
        legacy_lut = _build_legacy_token_lut(block2tok)

    # Vectorized: first pass with base block ID lookup (O(1) via LUT)
    result = legacy_lut[blocks_xyz].astype(np.int64)

    # Second pass: override with data-value-specific mappings where needed
    for (block_id, data_val), block_name in LEGACY_DATA_MAP.items():
        sub_mask = (blocks_xyz == block_id) & (data_xyz == data_val)
        if not np.any(sub_mask):
            continue

        token = block2tok.get(block_name)
        if token is None:
            base = block_name.split("[")[0]
            token = block2tok.get(base, air_token)
        result[sub_mask] = token

    return result, (width, height, length)


def read_litematic(
    path: Path, block2tok: Dict[str, int], air_token: int
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """Read a .litematic file (Litematica format).

    Args:
        path: Path to .litematic file.
        block2tok: Modern block name -> token ID mapping.
        air_token: Token ID for air.

    Returns:
        tokens: Array of shape (X, Y, Z) with vocabulary token IDs.
        original_size: (X, Y, Z) dimensions.
    """
    import nbtlib

    nbt = nbtlib.load(path)

    regions = nbt["Regions"]
    if len(regions) == 0:
        raise ValueError("No regions found in litematic file")

    # Use the first/largest region
    region_name = list(regions.keys())[0]
    region = regions[region_name]

    # Get dimensions (can be negative in litematica)
    sx = abs(int(region["Size"]["x"]))
    sy = abs(int(region["Size"]["y"]))
    sz = abs(int(region["Size"]["z"]))

    if sx == 0 or sy == 0 or sz == 0:
        raise ValueError(f"Zero-size region: {sx}x{sy}x{sz}")

    # Parse palette
    palette = region["BlockStatePalette"]
    palette_names: List[str] = []
    for entry in palette:
        name = str(entry["Name"])
        palette_names.append(name)

    palette_size = len(palette_names)

    # Decode bit-packed BlockStates
    block_states = np.array(region["BlockStates"], dtype=np.int64)

    # Bits per entry
    bits_per_entry = max(2, math.ceil(math.log2(palette_size))) if palette_size > 1 else 2
    total_blocks = sx * sy * sz

    # Unpack from 64-bit longs
    block_indices = _unpack_litematic_blocks(block_states, bits_per_entry, total_blocks)

    # Litematica stores in XZY order: index = y * sx * sz + z * sx + x
    blocks_flat = block_indices[:total_blocks]
    blocks_xzy = blocks_flat.reshape((sy, sz, sx))  # (Y, Z, X)
    blocks_xyz = blocks_xzy.transpose(2, 0, 1)  # (X, Y, Z)

    # Vectorized: precompute palette index → token mapping (Optimization #2)
    palette_lut = np.zeros(palette_size, dtype=np.int32)
    for pal_idx, block_name in enumerate(palette_names):
        # Try exact match
        token = block2tok.get(block_name)
        if token is None:
            # Strip block state
            base = block_name.split("[")[0]
            token = block2tok.get(base, air_token)
        palette_lut[pal_idx] = token

    # Single vectorized lookup: palette index → token
    result = palette_lut[blocks_xyz].astype(np.int64)

    return result, (sx, sy, sz)


def _unpack_litematic_blocks(
    packed: np.ndarray, bits_per_entry: int, total_blocks: int
) -> np.ndarray:
    """Unpack bit-packed block indices from litematic BlockStates.

    Litematica packs indices into 64-bit longs. Each long holds
    floor(64 / bits_per_entry) entries, and entries do NOT span
    across long boundaries.

    Args:
        packed: Array of int64 values (the BlockStates LongArray).
        bits_per_entry: Number of bits per block index.
        total_blocks: Total number of blocks to unpack.

    Returns:
        Array of block palette indices.
    """
    mask = (1 << bits_per_entry) - 1
    entries_per_long = 64 // bits_per_entry
    result = np.zeros(total_blocks, dtype=np.int32)

    idx = 0
    for long_val in packed:
        # Convert signed int64 to unsigned bits
        val = int(long_val) & 0xFFFFFFFFFFFFFFFF
        for j in range(entries_per_long):
            if idx >= total_blocks:
                break
            result[idx] = (val >> (j * bits_per_entry)) & mask
            idx += 1
        if idx >= total_blocks:
            break

    return result


def read_schem(
    path: Path, block2tok: Dict[str, int], air_token: int
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """Read a .schem file (Sponge Schematic v2).

    Wraps existing read_schematic + schematic_to_tokens from
    reconstruct_schematic.py.

    Args:
        path: Path to .schem file.
        block2tok: Modern block name -> token ID mapping.
        air_token: Token ID for air.

    Returns:
        tokens: Array of shape (X, Y, Z) with vocabulary token IDs.
        original_size: (X, Y, Z) dimensions.
    """
    blocks_xyz, palette, metadata = read_schem_file(path)
    x, y, z = blocks_xyz.shape
    tokens = schematic_to_tokens(blocks_xyz, palette, block2tok, air_token=air_token)
    return tokens, (x, y, z)


# ============================================================
# Normalization & Filtering
# ============================================================


def normalize_to_32(
    tokens: np.ndarray, air_token: int, center: bool = True
) -> np.ndarray:
    """Normalize structure to 32x32x32 with centering.

    Centers the non-air bounding box within the 32x32x32 volume.
    Crops if any dimension exceeds 32 (takes center portion).

    Args:
        tokens: Array of shape (X, Y, Z) with token IDs.
        air_token: Token ID for air.
        center: If True, center the structure. If False, align to origin.

    Returns:
        Normalized 32x32x32 array.
    """
    # Find bounding box of non-air blocks
    non_air = ~np.isin(tokens, list(AIR_TOKENS))
    if not np.any(non_air):
        return np.full((32, 32, 32), air_token, dtype=tokens.dtype)

    coords = np.where(non_air)
    mins = [int(c.min()) for c in coords]
    maxs = [int(c.max()) for c in coords]

    # Extract bounding box
    crop = tokens[
        mins[0] : maxs[0] + 1,
        mins[1] : maxs[1] + 1,
        mins[2] : maxs[2] + 1,
    ]

    cx, cy, cz = crop.shape

    # If any cropped dimension exceeds 32, take center 32
    if cx > 32:
        start = (cx - 32) // 2
        crop = crop[start : start + 32, :, :]
        cx = 32
    if cy > 32:
        start = (cy - 32) // 2
        crop = crop[:, start : start + 32, :]
        cy = 32
    if cz > 32:
        start = (cz - 32) // 2
        crop = crop[:, :, start : start + 32]
        cz = 32

    # Create output and center the structure
    result = np.full((32, 32, 32), air_token, dtype=tokens.dtype)

    if center:
        ox = (32 - cx) // 2
        oy = 0  # Keep Y grounded (builds sit on ground)
        oz = (32 - cz) // 2
    else:
        ox, oy, oz = 0, 0, 0

    result[ox : ox + cx, oy : oy + cy, oz : oz + cz] = crop

    return result


def passes_filter(
    tokens: np.ndarray, config: ConversionConfig
) -> Tuple[bool, str]:
    """Check if a normalized structure passes quality filters.

    Args:
        tokens: 32x32x32 array of token IDs.
        config: Conversion configuration.

    Returns:
        (passes, reason): Whether it passes and why it was rejected.
    """
    # Count non-air voxels
    non_air_mask = ~np.isin(tokens, list(AIR_TOKENS))
    non_air_count = int(non_air_mask.sum())
    total = 32 * 32 * 32

    if non_air_count < config.min_non_air_voxels:
        return False, f"too_few_blocks:{non_air_count}"

    air_fraction = 1.0 - (non_air_count / total)
    if air_fraction > config.max_air_fraction:
        return False, f"too_much_air:{air_fraction:.3f}"

    # Block diversity
    unique_blocks = len(np.unique(tokens[non_air_mask]))
    if unique_blocks < config.min_unique_blocks:
        return False, f"low_diversity:{unique_blocks}"

    return True, "ok"


# ============================================================
# Main Processing
# ============================================================


def find_all_files(input_dirs: List[Path]) -> Dict[str, List[Path]]:
    """Find all supported structure files in input directories.

    Args:
        input_dirs: Directories to scan.

    Returns:
        Dict mapping extension to list of file paths.
    """
    extensions = {".schem", ".schematic", ".litematic"}
    files: Dict[str, List[Path]] = {ext: [] for ext in extensions}

    for input_dir in input_dirs:
        if not input_dir.exists():
            print(f"  WARNING: Directory not found: {input_dir}")
            continue

        for entry in os.scandir(str(input_dir)):
            if not entry.is_file():
                continue
            ext = Path(entry.name).suffix.lower()
            if ext in extensions:
                files[ext].append(Path(entry.path))

    return files


def make_output_name(path: Path) -> str:
    """Generate a unique output filename for a structure file.

    Prefixes with format type to avoid collisions between directories.
    """
    stem = path.stem
    # Prefix with parent directory name to avoid collisions
    parent = path.parent.name
    return f"{parent}_{stem}.h5"


def convert_single_file(
    path: Path,
    block2tok: Dict[str, int],
    config: ConversionConfig,
    legacy_lut: Optional[np.ndarray] = None,
) -> Tuple[bool, str, Optional[np.ndarray]]:
    """Convert a single structure file to normalized token array.

    Args:
        path: Path to input file.
        block2tok: Block name -> token ID mapping.
        config: Conversion configuration.
        legacy_lut: Optional precomputed legacy block ID lookup table.

    Returns:
        (success, message, tokens_32): Result with optional 32x32x32 array.
    """
    ext = path.suffix.lower()

    try:
        # Read file based on extension
        if ext == ".schem":
            tokens, (sx, sy, sz) = read_schem(path, block2tok, config.air_token)
        elif ext == ".schematic":
            tokens, (sx, sy, sz) = read_schematic_legacy(
                path, block2tok, config.air_token, legacy_lut=legacy_lut
            )
        elif ext == ".litematic":
            tokens, (sx, sy, sz) = read_litematic(path, block2tok, config.air_token)
        else:
            return False, f"unsupported_format:{ext}", None

        # Size filter (pre-normalization)
        if max(sx, sy, sz) > config.max_dimension:
            return False, f"too_large:{sx}x{sy}x{sz}", None

        if max(sx, sy, sz) < config.min_dimension:
            return False, f"too_small:{sx}x{sy}x{sz}", None

        # Normalize to 32x32x32
        tokens_32 = normalize_to_32(tokens, config.air_token, center=True)

        # Quality filter
        passes, reason = passes_filter(tokens_32, config)
        if not passes:
            return False, reason, None

        return True, "ok", tokens_32

    except Exception as e:
        return False, f"error:{type(e).__name__}:{e}", None


# ============================================================
# Multiprocessing Worker (Optimization #1)
# ============================================================

# Global state for worker process (initialized once per worker)
_worker_block2tok: Dict[str, int] = {}
_worker_config: ConversionConfig = None
_worker_legacy_lut: Optional[np.ndarray] = None


def _worker_init(block2tok: Dict[str, int], config: ConversionConfig) -> None:
    """Initialize worker process with shared data (called once per worker)."""
    global _worker_block2tok, _worker_config, _worker_legacy_lut
    _worker_block2tok = block2tok
    _worker_config = config
    _worker_legacy_lut = _build_legacy_token_lut(block2tok)


def _worker_convert(path: Path) -> Tuple[Path, bool, str, Optional[np.ndarray]]:
    """Worker function for multiprocessing pool.

    Args:
        path: Path to input file.

    Returns:
        (path, success, message, tokens_32)
    """
    success, message, tokens = convert_single_file(
        path, _worker_block2tok, _worker_config, legacy_lut=_worker_legacy_lut
    )
    return (path, success, message, tokens)


def save_h5(path: Path, tokens: np.ndarray) -> None:
    """Save a 32x32x32 token array as H5 file.

    Matches the format used by existing training data.
    """
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "build",
            data=tokens.astype(np.uint16),
            dtype=np.uint16,
            compression="gzip",
        )


def main():
    parser = argparse.ArgumentParser(
        description="Convert .schem/.schematic/.litematic to .h5"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Scan and filter without writing files",
    )
    parser.add_argument(
        "--max-files", type=int, default=0,
        help="Max files to process (0 = all)",
    )
    parser.add_argument(
        "--no-skip-existing", action="store_true",
        help="Re-process files that already have H5 output",
    )
    parser.add_argument(
        "--input-dir", type=str, action="append", default=None,
        help="Additional input directory (can specify multiple)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory",
    )
    args = parser.parse_args()

    config = ConversionConfig()
    if args.dry_run:
        config.dry_run = True
    if args.max_files > 0:
        config.max_files = args.max_files
    if args.no_skip_existing:
        config.skip_existing = False
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
    if args.input_dir:
        config.input_dirs.extend(Path(d) for d in args.input_dir)

    print("=" * 60)
    print("Universal Structure → H5 Converter")
    print("(Optimized: Multiprocessing + Vectorized Lookups)")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output:  {config.output_dir}")
    print(f"Dry run: {config.dry_run}")

    # Load vocabulary
    print(f"\nLoading vocabulary from {config.vocab_path}...")
    with open(config.vocab_path, "r") as f:
        tok2block = json.load(f)
    block2tok = {v: int(k) for k, v in tok2block.items()}
    print(f"  {len(tok2block)} block types")

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Find all files
    print("\nScanning input directories...")
    all_files = find_all_files(config.input_dirs)

    total_found = sum(len(v) for v in all_files.values())
    print(f"\nFound {total_found} files:")
    for ext, files in sorted(all_files.items()):
        print(f"  {ext}: {len(files)}")

    # Build processing list
    process_list: List[Path] = []
    for ext in [".schem", ".schematic", ".litematic"]:
        process_list.extend(all_files[ext])

    # Check for existing outputs
    if config.skip_existing:
        existing = set(os.listdir(str(config.output_dir)))
        before = len(process_list)
        process_list = [
            p for p in process_list
            if make_output_name(p) not in existing
        ]
        skipped = before - len(process_list)
        if skipped > 0:
            print(f"\nSkipping {skipped} already-processed files")

    if config.max_files > 0:
        process_list = process_list[: config.max_files]

    print(f"\nProcessing {len(process_list)} files...")

    # Statistics
    stats = {
        "total": len(process_list),
        "success": 0,
        "too_large": 0,
        "too_small": 0,
        "too_few_blocks": 0,
        "too_much_air": 0,
        "low_diversity": 0,
        "errors": 0,
        "by_format": {".schem": 0, ".schematic": 0, ".litematic": 0},
    }

    error_log_path = config.output_dir / "conversion_errors.log"
    error_log = open(error_log_path, "w") if not config.dry_run else None

    # Detect number of CPU cores for multiprocessing
    num_workers = min(8, os.cpu_count() or 4)
    print(f"\nUsing {num_workers} worker processes for parallel conversion...")

    # Process files with multiprocessing pool
    with Pool(processes=num_workers, initializer=_worker_init, initargs=(block2tok, config)) as pool:
        # Use imap_unordered for better performance (results come back in any order)
        results = pool.imap_unordered(_worker_convert, process_list, chunksize=4)

        # Process results with progress bar
        for path, success, message, tokens in tqdm(results, total=len(process_list), desc="Converting"):
            if success and tokens is not None:
                stats["success"] += 1
                ext = path.suffix.lower()
                stats["by_format"][ext] += 1

                if not config.dry_run:
                    out_name = make_output_name(path)
                    save_h5(config.output_dir / out_name, tokens)
            else:
                # Categorize failure
                if message.startswith("too_large"):
                    stats["too_large"] += 1
                elif message.startswith("too_small"):
                    stats["too_small"] += 1
                elif message.startswith("too_few"):
                    stats["too_few_blocks"] += 1
                elif message.startswith("too_much_air"):
                    stats["too_much_air"] += 1
                elif message.startswith("low_diversity"):
                    stats["low_diversity"] += 1
                elif message.startswith("error"):
                    stats["errors"] += 1

                if error_log:
                    error_log.write(f"{path.name}: {message}\n")

    if error_log:
        error_log.close()

    # Summary
    print("\n" + "=" * 60)
    print("Conversion Complete")
    print("=" * 60)
    pct = (stats["success"] / max(stats["total"], 1)) * 100
    print(f"  Processed:  {stats['total']}")
    print(f"  Successful: {stats['success']} ({pct:.1f}%)")
    print(f"\nBy format:")
    for ext, count in stats["by_format"].items():
        print(f"  {ext}: {count}")
    print(f"\nFiltered out:")
    print(f"  Too large (>{config.max_dimension}):  {stats['too_large']}")
    print(f"  Too small (<{config.min_dimension}):   {stats['too_small']}")
    print(f"  Too few blocks (<{config.min_non_air_voxels}): {stats['too_few_blocks']}")
    print(f"  Too much air (>{config.max_air_fraction:.0%}):  {stats['too_much_air']}")
    print(f"  Low diversity (<{config.min_unique_blocks}):  {stats['low_diversity']}")
    print(f"  Read errors:           {stats['errors']}")

    if not config.dry_run:
        # Count total H5 files in output
        total_h5 = len([f for f in os.listdir(str(config.output_dir)) if f.endswith(".h5")])
        print(f"\nTotal H5 files in output: {total_h5}")
        print(f"Error log: {error_log_path}")

        # Save metadata
        meta_path = config.output_dir / "conversion_metadata.json"
        meta = {
            "conversion_date": datetime.now().isoformat(),
            "config": {
                "input_dirs": [str(d) for d in config.input_dirs],
                "output_dir": str(config.output_dir),
                "max_dimension": config.max_dimension,
                "min_dimension": config.min_dimension,
                "min_unique_blocks": config.min_unique_blocks,
                "max_air_fraction": config.max_air_fraction,
                "min_non_air_voxels": config.min_non_air_voxels,
            },
            "statistics": stats,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Metadata: {meta_path}")

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
