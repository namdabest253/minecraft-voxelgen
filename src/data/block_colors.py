"""
Block Color Map — Maps Minecraft block tokens to RGB colors.

Used by the voxel renderer to produce 2D images from 3D structures
for CLIP scoring.

Color sources:
1. Known Minecraft block colors (manually curated for common blocks)
2. Category-based defaults (wood, stone, etc.)
3. Hash-based deterministic fallback for unknown blocks
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

from .terrain_detection import AIR_TOKENS

# Average texture colors for common Minecraft blocks
# Format: base_block_name -> (R, G, B)
MINECRAFT_BLOCK_COLORS: Dict[str, Tuple[int, int, int]] = {
    # Stone family
    "minecraft:stone": (125, 125, 125),
    "minecraft:cobblestone": (127, 127, 127),
    "minecraft:mossy_cobblestone": (110, 127, 98),
    "minecraft:stone_bricks": (122, 122, 122),
    "minecraft:mossy_stone_bricks": (110, 122, 98),
    "minecraft:cracked_stone_bricks": (118, 118, 118),
    "minecraft:smooth_stone": (158, 158, 158),
    "minecraft:granite": (149, 103, 85),
    "minecraft:polished_granite": (154, 106, 89),
    "minecraft:diorite": (188, 188, 188),
    "minecraft:polished_diorite": (192, 192, 192),
    "minecraft:andesite": (136, 136, 136),
    "minecraft:polished_andesite": (132, 135, 132),
    "minecraft:deepslate": (80, 80, 82),
    "minecraft:cobbled_deepslate": (77, 77, 80),
    "minecraft:tuff": (108, 109, 102),
    "minecraft:calcite": (223, 224, 220),
    "minecraft:bedrock": (85, 85, 85),
    "minecraft:basalt": (73, 72, 77),
    "minecraft:smooth_basalt": (72, 72, 73),
    "minecraft:blackstone": (42, 36, 41),
    # Dirt family
    "minecraft:dirt": (134, 96, 67),
    "minecraft:grass_block": (95, 159, 53),
    "minecraft:coarse_dirt": (119, 85, 59),
    "minecraft:podzol": (91, 63, 24),
    "minecraft:mycelium": (111, 99, 107),
    "minecraft:rooted_dirt": (130, 94, 65),
    "minecraft:dirt_path": (148, 121, 65),
    "minecraft:farmland": (119, 81, 50),
    "minecraft:mud": (60, 58, 62),
    # Sand family
    "minecraft:sand": (219, 207, 163),
    "minecraft:red_sand": (190, 102, 33),
    "minecraft:gravel": (131, 127, 126),
    "minecraft:clay": (160, 166, 179),
    "minecraft:soul_sand": (81, 62, 50),
    "minecraft:soul_soil": (75, 57, 46),
    # Wood planks
    "minecraft:oak_planks": (162, 130, 78),
    "minecraft:spruce_planks": (114, 84, 48),
    "minecraft:birch_planks": (196, 179, 123),
    "minecraft:jungle_planks": (160, 115, 80),
    "minecraft:acacia_planks": (168, 90, 50),
    "minecraft:dark_oak_planks": (66, 43, 20),
    "minecraft:mangrove_planks": (117, 54, 48),
    "minecraft:cherry_planks": (226, 178, 172),
    "minecraft:bamboo_planks": (194, 175, 80),
    "minecraft:crimson_planks": (101, 48, 70),
    "minecraft:warped_planks": (43, 104, 99),
    # Wood logs
    "minecraft:oak_log": (109, 85, 50),
    "minecraft:spruce_log": (58, 37, 16),
    "minecraft:birch_log": (216, 215, 210),
    "minecraft:jungle_log": (85, 68, 25),
    "minecraft:acacia_log": (103, 96, 86),
    "minecraft:dark_oak_log": (60, 46, 26),
    "minecraft:mangrove_log": (84, 56, 40),
    "minecraft:cherry_log": (52, 28, 36),
    # Leaves
    "minecraft:oak_leaves": (59, 107, 22),
    "minecraft:spruce_leaves": (58, 92, 58),
    "minecraft:birch_leaves": (73, 115, 43),
    "minecraft:jungle_leaves": (47, 107, 14),
    "minecraft:acacia_leaves": (63, 107, 22),
    "minecraft:dark_oak_leaves": (55, 97, 18),
    "minecraft:azalea_leaves": (83, 112, 38),
    "minecraft:cherry_leaves": (233, 178, 201),
    "minecraft:mangrove_leaves": (67, 107, 24),
    # Glass
    "minecraft:glass": (200, 220, 230),
    "minecraft:white_stained_glass": (255, 255, 255),
    "minecraft:orange_stained_glass": (216, 127, 51),
    "minecraft:magenta_stained_glass": (178, 76, 216),
    "minecraft:light_blue_stained_glass": (102, 153, 216),
    "minecraft:yellow_stained_glass": (229, 229, 51),
    "minecraft:lime_stained_glass": (127, 204, 25),
    "minecraft:pink_stained_glass": (242, 127, 165),
    "minecraft:gray_stained_glass": (76, 76, 76),
    "minecraft:light_gray_stained_glass": (153, 153, 153),
    "minecraft:cyan_stained_glass": (76, 127, 153),
    "minecraft:purple_stained_glass": (127, 63, 178),
    "minecraft:blue_stained_glass": (51, 76, 178),
    "minecraft:brown_stained_glass": (102, 76, 51),
    "minecraft:green_stained_glass": (102, 127, 51),
    "minecraft:red_stained_glass": (153, 51, 51),
    "minecraft:black_stained_glass": (25, 25, 25),
    # Wool
    "minecraft:white_wool": (234, 236, 236),
    "minecraft:orange_wool": (240, 118, 19),
    "minecraft:magenta_wool": (189, 68, 179),
    "minecraft:light_blue_wool": (58, 175, 217),
    "minecraft:yellow_wool": (248, 198, 39),
    "minecraft:lime_wool": (112, 185, 25),
    "minecraft:pink_wool": (237, 141, 172),
    "minecraft:gray_wool": (63, 68, 72),
    "minecraft:light_gray_wool": (142, 142, 134),
    "minecraft:cyan_wool": (21, 137, 145),
    "minecraft:purple_wool": (121, 42, 172),
    "minecraft:blue_wool": (53, 57, 157),
    "minecraft:brown_wool": (114, 71, 40),
    "minecraft:green_wool": (84, 109, 27),
    "minecraft:red_wool": (161, 39, 34),
    "minecraft:black_wool": (20, 21, 25),
    # Concrete
    "minecraft:white_concrete": (207, 213, 214),
    "minecraft:orange_concrete": (224, 97, 1),
    "minecraft:magenta_concrete": (169, 48, 159),
    "minecraft:light_blue_concrete": (36, 137, 199),
    "minecraft:yellow_concrete": (240, 175, 21),
    "minecraft:lime_concrete": (94, 169, 24),
    "minecraft:pink_concrete": (214, 101, 143),
    "minecraft:gray_concrete": (55, 58, 62),
    "minecraft:light_gray_concrete": (125, 125, 115),
    "minecraft:cyan_concrete": (21, 119, 136),
    "minecraft:purple_concrete": (100, 31, 156),
    "minecraft:blue_concrete": (45, 47, 143),
    "minecraft:brown_concrete": (96, 59, 31),
    "minecraft:green_concrete": (73, 91, 36),
    "minecraft:red_concrete": (142, 33, 33),
    "minecraft:black_concrete": (8, 10, 15),
    # Terracotta
    "minecraft:terracotta": (152, 94, 67),
    "minecraft:white_terracotta": (210, 178, 161),
    "minecraft:orange_terracotta": (162, 84, 38),
    "minecraft:magenta_terracotta": (150, 88, 109),
    "minecraft:light_blue_terracotta": (113, 109, 138),
    "minecraft:yellow_terracotta": (186, 133, 35),
    "minecraft:lime_terracotta": (103, 117, 53),
    "minecraft:pink_terracotta": (162, 78, 79),
    "minecraft:gray_terracotta": (58, 42, 36),
    "minecraft:light_gray_terracotta": (135, 107, 98),
    "minecraft:cyan_terracotta": (87, 91, 91),
    "minecraft:purple_terracotta": (118, 70, 86),
    "minecraft:blue_terracotta": (74, 60, 91),
    "minecraft:brown_terracotta": (77, 51, 36),
    "minecraft:green_terracotta": (76, 83, 42),
    "minecraft:red_terracotta": (143, 61, 47),
    "minecraft:black_terracotta": (37, 23, 16),
    # Metals and ores
    "minecraft:iron_block": (220, 220, 220),
    "minecraft:gold_block": (246, 208, 61),
    "minecraft:diamond_block": (98, 237, 228),
    "minecraft:emerald_block": (42, 183, 66),
    "minecraft:lapis_block": (38, 67, 138),
    "minecraft:redstone_block": (171, 27, 9),
    "minecraft:copper_block": (192, 107, 79),
    "minecraft:netherite_block": (66, 61, 63),
    "minecraft:coal_block": (16, 15, 15),
    "minecraft:iron_ore": (136, 130, 127),
    "minecraft:gold_ore": (145, 134, 108),
    "minecraft:diamond_ore": (121, 141, 140),
    "minecraft:coal_ore": (105, 105, 105),
    # Brick types
    "minecraft:bricks": (150, 97, 83),
    "minecraft:nether_bricks": (44, 21, 26),
    "minecraft:red_nether_bricks": (69, 7, 9),
    "minecraft:end_stone_bricks": (218, 224, 162),
    "minecraft:prismarine_bricks": (99, 171, 158),
    "minecraft:deepslate_bricks": (70, 70, 73),
    # Nether blocks
    "minecraft:netherrack": (97, 38, 38),
    "minecraft:crimson_nylium": (130, 31, 31),
    "minecraft:warped_nylium": (22, 126, 134),
    "minecraft:glowstone": (171, 131, 73),
    "minecraft:nether_wart_block": (114, 2, 2),
    "minecraft:warped_wart_block": (22, 119, 121),
    "minecraft:shroomlight": (240, 146, 70),
    "minecraft:crying_obsidian": (32, 10, 60),
    "minecraft:obsidian": (15, 11, 25),
    # End blocks
    "minecraft:end_stone": (219, 223, 158),
    "minecraft:purpur_block": (170, 126, 170),
    # Utility blocks
    "minecraft:crafting_table": (116, 73, 41),
    "minecraft:furnace": (130, 130, 130),
    "minecraft:chest": (162, 130, 78),
    "minecraft:bookshelf": (116, 88, 52),
    "minecraft:tnt": (200, 50, 50),
    # Water and lava
    "minecraft:water": (47, 67, 244),
    "minecraft:lava": (207, 85, 16),
    # Ice and snow
    "minecraft:ice": (145, 183, 253),
    "minecraft:packed_ice": (141, 180, 250),
    "minecraft:blue_ice": (116, 167, 253),
    "minecraft:snow_block": (249, 254, 254),
    "minecraft:snow": (249, 254, 254),
    "minecraft:powder_snow": (248, 253, 253),
    # Misc
    "minecraft:moss_block": (89, 109, 45),
    "minecraft:hay_block": (166, 138, 18),
    "minecraft:sponge": (195, 192, 74),
    "minecraft:melon": (111, 145, 30),
    "minecraft:pumpkin": (198, 118, 24),
    "minecraft:jack_o_lantern": (208, 138, 34),
    "minecraft:honeycomb_block": (229, 148, 29),
    "minecraft:sea_lantern": (172, 199, 190),
    "minecraft:redstone_lamp": (160, 95, 40),
    "minecraft:bone_block": (209, 206, 185),
    "minecraft:quartz_block": (235, 229, 222),
    "minecraft:smooth_quartz": (235, 229, 222),
    "minecraft:sandstone": (216, 203, 155),
    "minecraft:smooth_sandstone": (223, 211, 163),
    "minecraft:red_sandstone": (186, 99, 29),
    "minecraft:smooth_red_sandstone": (181, 97, 31),
    "minecraft:prismarine": (99, 156, 131),
    "minecraft:dark_prismarine": (51, 91, 75),
    # Slabs, stairs, walls share base block color (handled by stripping suffix)
}

# Category-based defaults for blocks matching name patterns
CATEGORY_COLORS: list = [
    # (substring, color)
    ("oak", (162, 130, 78)),
    ("spruce", (114, 84, 48)),
    ("birch", (196, 179, 123)),
    ("jungle", (160, 115, 80)),
    ("acacia", (168, 90, 50)),
    ("dark_oak", (66, 43, 20)),
    ("mangrove", (117, 54, 48)),
    ("cherry", (226, 178, 172)),
    ("bamboo", (194, 175, 80)),
    ("crimson", (101, 48, 70)),
    ("warped", (43, 104, 99)),
    ("copper", (192, 107, 79)),
    ("iron", (220, 220, 220)),
    ("gold", (246, 208, 61)),
    ("diamond", (98, 237, 228)),
    ("emerald", (42, 183, 66)),
    ("lapis", (38, 67, 138)),
    ("redstone", (171, 27, 9)),
    ("quartz", (235, 229, 222)),
    ("sandstone", (216, 203, 155)),
    ("prismarine", (99, 156, 131)),
    ("deepslate", (80, 80, 82)),
    ("nether_brick", (44, 21, 26)),
    ("stone", (125, 125, 125)),
    ("cobblestone", (127, 127, 127)),
    ("brick", (150, 97, 83)),
    ("mud", (60, 58, 62)),
    ("coral", (200, 100, 120)),
    ("amethyst", (131, 87, 196)),
    ("dripstone", (134, 107, 92)),
    ("sculk", (12, 37, 42)),
]

# Background color for air
AIR_COLOR = (255, 255, 255)


def _hash_color(name: str) -> Tuple[int, int, int]:
    """Generate a deterministic color from a block name hash."""
    h = hashlib.md5(name.encode()).hexdigest()
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    # Avoid too-dark or too-light colors
    r = max(40, min(220, r))
    g = max(40, min(220, g))
    b = max(40, min(220, b))
    return (r, g, b)


def _strip_block_state(block_name: str) -> str:
    """Remove block state from name: 'minecraft:oak_log[axis=y]' -> 'minecraft:oak_log'."""
    return block_name.split("[")[0]


def _strip_variant_suffix(block_name: str) -> str:
    """Strip slab/stairs/wall/fence suffix to get base block.

    'minecraft:oak_slab' -> 'minecraft:oak_planks'
    'minecraft:stone_brick_stairs' -> 'minecraft:stone_bricks'
    """
    suffixes_to_base = {
        "_slab": "_planks",
        "_stairs": "",
        "_wall": "",
        "_fence": "",
        "_fence_gate": "",
        "_pressure_plate": "",
        "_button": "",
        "_door": "",
        "_trapdoor": "",
        "_sign": "",
        "_wall_sign": "",
        "_hanging_sign": "",
    }
    for suffix, replacement in suffixes_to_base.items():
        if block_name.endswith(suffix):
            base = block_name[: -len(suffix)]
            if replacement:
                return base + replacement
            return base
    return block_name


def load_block_colors(
    tok2block_path: str,
) -> Dict[int, Tuple[int, int, int]]:
    """Return {token_id: (R, G, B)} for all block tokens.

    Resolution order:
    1. Air tokens -> white
    2. Exact match in MINECRAFT_BLOCK_COLORS
    3. Base block match (strip state + variant suffix)
    4. Category substring match
    5. Hash-based fallback

    Args:
        tok2block_path: Path to tok2block.json

    Returns:
        Dict mapping token ID to RGB tuple.
    """
    with open(tok2block_path, "r") as f:
        tok2block = json.load(f)

    colors: Dict[int, Tuple[int, int, int]] = {}

    for tok_str, block_name in tok2block.items():
        tok_id = int(tok_str)

        # Air tokens
        if tok_id in AIR_TOKENS:
            colors[tok_id] = AIR_COLOR
            continue

        base_name = _strip_block_state(block_name)

        # 1. Exact match
        if base_name in MINECRAFT_BLOCK_COLORS:
            colors[tok_id] = MINECRAFT_BLOCK_COLORS[base_name]
            continue

        # 2. Strip variant suffix and try again
        stripped = _strip_variant_suffix(base_name)
        if stripped in MINECRAFT_BLOCK_COLORS:
            colors[tok_id] = MINECRAFT_BLOCK_COLORS[stripped]
            continue

        # 3. Category substring match
        matched = False
        for substring, color in CATEGORY_COLORS:
            if substring in base_name:
                colors[tok_id] = color
                matched = True
                break

        if matched:
            continue

        # 4. Hash-based fallback
        colors[tok_id] = _hash_color(base_name)

    return colors
