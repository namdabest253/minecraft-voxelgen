"""
Block Decomposition for Compositional Embeddings (V4).

Decomposes Minecraft block names into semantic components:
- Material: oak, spruce, stone, iron, diamond, etc.
- Shape: planks, stairs, slab, fence, wall, door, etc.
- Properties: solid, transparent, light_emitting, etc.

This enables compositional embeddings where:
    embedding(oak_planks) = material_emb(oak) + shape_emb(planks) + property_emb(solid)

Guarantees that blocks with same shape cluster together regardless of material.

V4 UPDATE: Added SHAPE_GROUPS to consolidate similar blocks under common shapes.
For example, all flowers (dandelion, poppy, allium) now share the "flower" shape.
This reduces unique shapes from ~320 to ~230 and improves clustering.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class BlockComponents:
    """Decomposed components of a block."""
    original_name: str
    base_name: str  # Without minecraft: prefix and block states
    material: Optional[str] = None
    shape: str = "block"  # Default shape is full block
    properties: list[str] = field(default_factory=list)

    def __repr__(self):
        props = ", ".join(self.properties) if self.properties else "none"
        return f"BlockComponents({self.base_name}: material={self.material}, shape={self.shape}, props=[{props}])"


# === MATERIAL DEFINITIONS ===
# Materials are the "what it's made of" component

WOOD_MATERIALS = {
    "oak", "spruce", "birch", "jungle", "acacia", "dark_oak",
    "mangrove", "cherry", "bamboo", "crimson", "warped"
}

STONE_MATERIALS = {
    "stone", "cobblestone", "mossy_cobblestone", "smooth_stone",
    "granite", "polished_granite", "diorite", "polished_diorite",
    "andesite", "polished_andesite", "deepslate", "cobbled_deepslate",
    "polished_deepslate", "calcite", "tuff", "dripstone", "blackstone",
    "polished_blackstone", "basalt", "smooth_basalt", "polished_basalt"
}

BRICK_MATERIALS = {
    "brick", "stone_brick", "mossy_stone_brick", "nether_brick",
    "red_nether_brick", "end_stone_brick", "prismarine_brick",
    "deepslate_brick", "polished_blackstone_brick", "mud_brick"
}

SANDSTONE_MATERIALS = {
    "sandstone", "red_sandstone", "smooth_sandstone", "smooth_red_sandstone",
    "cut_sandstone", "cut_red_sandstone", "chiseled_sandstone", "chiseled_red_sandstone"
}

METAL_MATERIALS = {
    "iron", "gold", "copper", "exposed_copper", "weathered_copper",
    "oxidized_copper", "waxed_copper", "netherite"
}

MINERAL_MATERIALS = {
    "diamond", "emerald", "lapis", "redstone", "coal", "quartz",
    "amethyst"
}

CONCRETE_MATERIALS = {
    "white", "orange", "magenta", "light_blue", "yellow", "lime",
    "pink", "gray", "light_gray", "cyan", "purple", "blue",
    "brown", "green", "red", "black"
}

NETHER_MATERIALS = {
    "nether", "soul", "warped", "crimson", "shroomlight",
    "glowstone", "netherrack", "magma"
}

END_MATERIALS = {
    "end_stone", "purpur", "end"
}

# Combined material lookup with categories
MATERIAL_CATEGORIES = {
    "wood": WOOD_MATERIALS,
    "stone": STONE_MATERIALS,
    "brick": BRICK_MATERIALS,
    "sandstone": SANDSTONE_MATERIALS,
    "metal": METAL_MATERIALS,
    "mineral": MINERAL_MATERIALS,
    "concrete": CONCRETE_MATERIALS,
    "nether": NETHER_MATERIALS,
    "end": END_MATERIALS,
}


# === SHAPE DEFINITIONS ===
# Shapes define the geometric form of the block

SHAPE_PATTERNS = {
    # Exact suffix matches (order matters - longer first)
    "_pressure_plate": "pressure_plate",
    "_wall_hanging_sign": "wall_hanging_sign",
    "_wall_sign": "wall_sign",
    "_wall_banner": "wall_banner",
    "_wall_head": "wall_head",
    "_wall_skull": "wall_skull",
    "_wall_torch": "wall_torch",
    "_wall_fan": "wall_fan",
    "_hanging_sign": "hanging_sign",
    "_fence_gate": "fence_gate",
    "_trap_door": "trapdoor",
    "_trapdoor": "trapdoor",
    "_coral_block": "coral_block",
    "_coral_fan": "coral_fan",
    "_coral": "coral",
    "_mushroom_block": "mushroom_block",
    "_mushroom": "mushroom",
    "_concrete_powder": "concrete_powder",
    "_glazed_terracotta": "glazed_terracotta",
    "_stained_glass_pane": "stained_glass_pane",
    "_stained_glass": "stained_glass",
    "_candle_cake": "candle_cake",
    "_shulker_box": "shulker_box",
    "_amethyst_bud": "amethyst_bud",
    "_froglight": "froglight",
    "_button": "button",
    "_stairs": "stairs",
    "_planks": "planks",
    "_slab": "slab",
    "_wall": "wall",
    "_fence": "fence",
    "_door": "door",
    "_sign": "sign",
    "_log": "log",
    "_wood": "wood",  # e.g., oak_wood (bark on all sides)
    "_stem": "stem",  # crimson_stem, warped_stem, melon_stem
    "_hyphae": "hyphae",  # crimson_hyphae, warped_hyphae
    "_roots": "roots",  # crimson_roots, warped_roots, hanging_roots
    "_leaves": "leaves",
    "_sapling": "sapling",
    "_carpet": "carpet",
    "_bed": "bed",
    "_banner": "banner",
    "_candle": "candle",
    "_head": "head",
    "_skull": "skull",
    "_pot": "pot",
    "_ore": "ore",
    "_block": "block",  # e.g., iron_block, diamond_block
    "_glass": "glass",
    "_pane": "pane",
    "_bars": "bars",
    "_chain": "chain",
    "_lantern": "lantern",
    "_torch": "torch",
    "_rail": "rail",
    "_chest": "chest",
    "_terracotta": "terracotta",
    "_concrete": "concrete",
    "_wool": "wool",
    "_tulip": "flower",
    "_vines": "vines",
    "_plant": "plant",
}

# Blocks that ARE shapes themselves (no material prefix)
STANDALONE_SHAPES = {
    # Air and fluids
    "air", "cave_air", "void_air", "water", "lava", "fire", "soul_fire",
    "bubble_column", "powder_snow",

    # Terrain blocks
    "grass_block", "dirt", "coarse_dirt", "rooted_dirt", "podzol", "mycelium",
    "sand", "red_sand", "gravel", "clay", "mud", "packed_mud", "farmland", "dirt_path",
    "snow", "snow_block", "snow_layer", "ice", "packed_ice", "blue_ice", "frosted_ice",
    "glass", "tinted_glass", "bedrock", "obsidian", "crying_obsidian",

    # Storage and crafting
    "chest", "trapped_chest", "ender_chest", "barrel", "shulker_box",
    "crafting_table", "furnace", "blast_furnace", "smoker",
    "anvil", "chipped_anvil", "damaged_anvil",
    "grindstone", "stonecutter", "cartography_table",
    "fletching_table", "smithing_table", "loom", "lectern",
    "composter", "brewing_stand", "cauldron", "water_cauldron", "lava_cauldron", "powder_snow_cauldron",

    # Redstone
    "hopper", "dropper", "dispenser", "observer", "piston", "piston_head",
    "sticky_piston", "moving_piston", "slime_block", "honey_block",
    "tnt", "target", "repeater", "comparator", "daylight_detector",
    "note_block", "jukebox", "lever", "tripwire", "tripwire_hook",
    "lightning_rod", "redstone_wire",

    # Special blocks
    "respawn_anchor", "lodestone", "beacon", "conduit",
    "enchanting_table", "end_portal_frame", "end_portal", "end_gateway", "dragon_egg",
    "bell", "campfire", "soul_campfire",
    "spawner", "structure_block", "structure_void", "jigsaw", "barrier", "light",
    "command_block", "chain_command_block", "repeating_command_block",

    # Torches and lighting
    "torch", "soul_torch", "redstone_torch", "wall_torch", "soul_wall_torch", "redstone_wall_torch",
    "lantern", "soul_lantern", "chain", "end_rod", "sea_lantern",

    # Flowers (single blocks)
    "dandelion", "poppy", "blue_orchid", "allium", "azure_bluet",
    "oxeye_daisy", "cornflower", "lily_of_the_valley", "wither_rose",
    "sunflower", "lilac", "rose_bush", "peony", "torchflower",
    "pitcher_plant", "pitcher_crop", "spore_blossom",

    # Tall plants and grass
    "grass", "tall_grass", "fern", "large_fern", "dead_bush",
    "seagrass", "tall_seagrass", "kelp", "kelp_plant",
    "sugar_cane", "bamboo", "cactus",
    "vine", "glow_lichen", "sculk_vein",
    "hanging_roots", "azalea", "flowering_azalea",
    "big_dripleaf", "big_dripleaf_stem", "small_dripleaf",
    "lily_pad", "moss_carpet",

    # Crops
    "wheat", "carrots", "potatoes", "beetroots", "melon", "pumpkin",
    "carved_pumpkin", "jack_o_lantern", "melon_stem", "pumpkin_stem",
    "attached_melon_stem", "attached_pumpkin_stem",
    "sweet_berry_bush", "cocoa", "nether_wart", "torchflower_crop",

    # Cave/underground
    "pointed_dripstone", "sculk", "sculk_sensor", "calibrated_sculk_sensor",
    "sculk_catalyst", "sculk_shrieker", "moss_block",
    "amethyst_cluster", "budding_amethyst",
    "cave_vines", "cave_vines_plant", "glow_item_frame", "item_frame",
    "twisting_vines", "twisting_vines_plant", "weeping_vines", "weeping_vines_plant",

    # Corals (standalone base types - prefixed ones use patterns)
    "brain_coral", "bubble_coral", "fire_coral", "horn_coral", "tube_coral",
    "dead_brain_coral", "dead_bubble_coral", "dead_fire_coral", "dead_horn_coral", "dead_tube_coral",

    # Nether
    "nether_sprouts", "crimson_fungus", "warped_fungus", "crimson_nylium", "warped_nylium",
    "shroomlight", "nether_wart_block", "warped_wart_block",

    # Misc blocks
    "sponge", "wet_sponge", "cobweb", "bookshelf", "chiseled_bookshelf",
    "hay_block", "bone_block", "honeycomb_block", "dried_kelp_block",
    "mushroom_stem", "chorus_flower", "chorus_plant",
    "decorated_pot", "flower_pot", "scaffolding", "ladder",
    "rail", "powered_rail", "detector_rail", "activator_rail",
    "frogspawn", "turtle_egg", "sniffer_egg",

    # Skulls and heads (base forms)
    "skull", "creeper_head", "dragon_head", "piglin_head", "player_head", "zombie_head",
    "skeleton_skull", "wither_skeleton_skull",

    # Infested blocks
    "infested_stone", "infested_cobblestone", "infested_stone_bricks",
    "infested_mossy_stone_bricks", "infested_cracked_stone_bricks",
    "infested_chiseled_stone_bricks", "infested_deepslate",

    # Prismarine
    "prismarine", "prismarine_bricks", "dark_prismarine",

    # Generic/base blocks (when no color prefix)
    "terracotta", "concrete_powder", "banner", "wall_banner",
    "button", "carpet", "wool", "candle", "candle_cake",
    "stained_glass", "stained_glass_pane", "stained_hardened_clay",

    # Remaining unique blocks
    "ancient_debris", "azalea_leaves_flowers",
    "bee_hive", "bee_nest", "beehive",
    "bricks", "cake",
    "exposed_cut_copper", "oxidized_cut_copper", "weathered_cut_copper",
    "gilded_blackstone", "reinforced_deepslate",
    "mossy_stone_bricks", "mud_bricks",
    "sea_pickle", "suspicious_gravel", "suspicious_sand",
}


# === V4 SHAPE GROUPS ===
# Map specific blocks to common shape categories to improve clustering
# This reduces unique shapes from ~320 to ~230

SHAPE_GROUPS = {
    # FLOWERS → "flower" or "tall_flower"
    "allium": "flower",
    "azure_bluet": "flower",
    "blue_orchid": "flower",
    "cornflower": "flower",
    "dandelion": "flower",
    "lily_of_the_valley": "flower",
    "oxeye_daisy": "flower",
    "poppy": "flower",
    "wither_rose": "flower",
    "torchflower": "flower",
    "spore_blossom": "flower",
    "lilac": "tall_flower",
    "peony": "tall_flower",
    "rose_bush": "tall_flower",
    "sunflower": "tall_flower",

    # CORALS → "coral" or "dead_coral"
    "brain_coral": "coral",
    "bubble_coral": "coral",
    "fire_coral": "coral",
    "horn_coral": "coral",
    "tube_coral": "coral",
    "dead_brain_coral": "dead_coral",
    "dead_bubble_coral": "dead_coral",
    "dead_fire_coral": "dead_coral",
    "dead_horn_coral": "dead_coral",
    "dead_tube_coral": "dead_coral",

    # HEADS → "head"
    "creeper_head": "head",
    "dragon_head": "head",
    "piglin_head": "head",
    "player_head": "head",
    "zombie_head": "head",
    "skeleton_skull": "skull",
    "wither_skeleton_skull": "skull",

    # RAILS → "rail"
    "activator_rail": "rail",
    "detector_rail": "rail",
    "powered_rail": "rail",

    # CROPS → "crop"
    "beetroots": "crop",
    "carrots": "crop",
    "potatoes": "crop",
    "wheat": "crop",
    "pitcher_crop": "crop",
    "torchflower_crop": "crop",
    "nether_wart": "crop",
    "cocoa": "crop",
    "sweet_berry_bush": "crop",

    # STEMS → "stem"
    "melon_stem": "stem",
    "pumpkin_stem": "stem",
    "attached_melon_stem": "stem",
    "attached_pumpkin_stem": "stem",

    # ANVILS → "anvil"
    "chipped_anvil": "anvil",
    "damaged_anvil": "anvil",

    # COMMAND BLOCKS → "command_block"
    "chain_command_block": "command_block",
    "repeating_command_block": "command_block",

    # INFESTED BLOCKS → "infested_block"
    "infested_stone": "infested_block",
    "infested_cobblestone": "infested_block",
    "infested_stone_bricks": "infested_block",
    "infested_mossy_stone_bricks": "infested_block",
    "infested_cracked_stone_bricks": "infested_block",
    "infested_chiseled_stone_bricks": "infested_block",
    "infested_deepslate": "infested_block",

    # AIR → "air"
    "cave_air": "air",
    "void_air": "air",

    # VINES → "vines"
    "cave_vines": "vines",
    "cave_vines_plant": "vines",
    "twisting_vines": "vines",
    "twisting_vines_plant": "vines",
    "weeping_vines": "vines",
    "weeping_vines_plant": "vines",
    "vine": "vines",
    "glow_lichen": "vines",

    # TORCHES → "torch" or "wall_torch"
    "soul_torch": "torch",
    "redstone_torch": "torch",
    "soul_wall_torch": "wall_torch",
    "redstone_wall_torch": "wall_torch",

    # LANTERNS → "lantern"
    "soul_lantern": "lantern",
    "sea_lantern": "lantern",

    # CAMPFIRES → "campfire"
    "soul_campfire": "campfire",

    # FIRE → "fire"
    "soul_fire": "fire",

    # FUNGI → "fungus"
    "crimson_fungus": "fungus",
    "warped_fungus": "fungus",

    # NYLIUM → "nylium"
    "crimson_nylium": "nylium",
    "warped_nylium": "nylium",

    # CAULDRONS → "cauldron"
    "water_cauldron": "cauldron",
    "lava_cauldron": "cauldron",
    "powder_snow_cauldron": "cauldron",

    # GRASS/FERN → "grass" or "tall_grass"
    "fern": "grass",
    "dead_bush": "grass",
    "tall_grass": "tall_grass",
    "large_fern": "tall_grass",

    # ICE → "ice"
    "packed_ice": "ice",
    "blue_ice": "ice",
    "frosted_ice": "ice",

    # KELP → "kelp"
    "kelp_plant": "kelp",

    # SEAGRASS → "seagrass"
    "tall_seagrass": "seagrass",

    # SCULK → "sculk" or "sculk_sensor"
    "sculk_catalyst": "sculk",
    "sculk_shrieker": "sculk",
    "sculk_vein": "sculk",
    "calibrated_sculk_sensor": "sculk_sensor",

    # AZALEA → "azalea"
    "flowering_azalea": "azalea",
    "azalea_leaves_flowers": "azalea",

    # DRIPLEAF → "dripleaf"
    "big_dripleaf": "dripleaf",
    "big_dripleaf_stem": "dripleaf",
    "small_dripleaf": "dripleaf",

    # FRAMES → "item_frame"
    "glow_item_frame": "item_frame",

    # WART BLOCKS → "wart_block"
    "nether_wart_block": "wart_block",
    "warped_wart_block": "wart_block",

    # SUSPICIOUS → "suspicious_block"
    "suspicious_sand": "suspicious_block",
    "suspicious_gravel": "suspicious_block",

    # FURNACES → "furnace"
    "blast_furnace": "furnace",
    "smoker": "furnace",

    # PISTONS → "piston"
    "sticky_piston": "piston",
    "moving_piston": "piston",

    # DIRT → "dirt"
    "coarse_dirt": "dirt",
    "rooted_dirt": "dirt",

    # ROOTS → "roots"
    "hanging_roots": "roots",

    # MUSHROOM BLOCKS → "mushroom_block"
    "mushroom_stem": "mushroom_block",

    # AMETHYST → "amethyst_cluster"
    "amethyst_bud": "amethyst_cluster",

    # CHEST → "chest"
    "trapped_chest": "chest",
    "ender_chest": "chest",

    # SIGN variants (wall signs handled by suffix)
    "wall_hanging_sign": "hanging_sign",
}


# === PROPERTY DEFINITIONS ===
# Properties are functional characteristics

TRANSPARENT_BLOCKS = {
    "glass", "pane", "ice", "leaves", "barrier", "light",
    "air", "water", "lava"
}

LIGHT_EMITTING = {
    "torch", "lantern", "glowstone", "shroomlight", "sea_lantern",
    "end_rod", "campfire", "fire", "lava", "beacon", "conduit",
    "respawn_anchor", "crying_obsidian", "magma", "jack_o_lantern",
    "redstone_lamp", "froglight", "sculk_catalyst", "sculk_shrieker"
}

GRAVITY_AFFECTED = {
    "sand", "gravel", "anvil", "dragon_egg", "concrete_powder",
    "suspicious_sand", "suspicious_gravel"
}

INTERACTABLE = {
    "door", "trapdoor", "fence_gate", "button", "lever",
    "chest", "barrel", "furnace", "crafting_table", "anvil",
    "enchanting_table", "brewing_stand", "beacon", "bed"
}

REDSTONE_COMPONENTS = {
    "redstone", "repeater", "comparator", "observer", "piston",
    "dropper", "dispenser", "hopper", "lever", "button",
    "pressure_plate", "tripwire", "daylight_detector", "target",
    "note_block", "jukebox", "bell", "lightning_rod", "sculk_sensor"
}


def extract_base_name(block_name: str) -> str:
    """Remove namespace and block states from block name."""
    # Remove namespace
    name = block_name.replace("minecraft:", "")
    # Remove block states
    if "[" in name:
        name = name[:name.index("[")]
    return name


def identify_material(base_name: str) -> Optional[str]:
    """Identify the material component of a block name."""
    # Check each material category
    for category, materials in MATERIAL_CATEGORIES.items():
        for material in materials:
            # Check if block starts with material
            if base_name.startswith(material + "_"):
                return material
            # Check if block starts with material (for things like "oak_planks")
            if base_name.startswith(material):
                # Make sure it's not just a substring match
                rest = base_name[len(material):]
                if rest == "" or rest.startswith("_"):
                    return material

    # Special cases for colored blocks
    for color in CONCRETE_MATERIALS:
        if base_name.startswith(color + "_"):
            return color

    return None


def identify_shape(base_name: str) -> str:
    """Identify the shape component of a block name."""
    # V4: Check shape groups first to consolidate similar blocks
    if base_name in SHAPE_GROUPS:
        return SHAPE_GROUPS[base_name]

    # Check standalone shapes
    if base_name in STANDALONE_SHAPES:
        return base_name

    # Special case: potted plants
    if base_name.startswith("potted_"):
        return "potted_plant"

    # Special case: stripped wood/logs
    if base_name.startswith("stripped_"):
        rest = base_name[9:]  # Remove "stripped_"
        if rest.endswith("_log"):
            return "stripped_log"
        elif rest.endswith("_wood"):
            return "stripped_wood"
        elif rest.endswith("_stem"):
            return "stripped_stem"
        elif rest.endswith("_hyphae"):
            return "stripped_hyphae"
        elif rest.endswith("_block"):
            return "stripped_block"

    # Special case: chiseled variants
    if base_name.startswith("chiseled_"):
        return "chiseled_block"

    # Special case: cracked variants
    if base_name.startswith("cracked_"):
        return "cracked_block"

    # Special case: cut variants (cut_copper, cut_sandstone)
    if base_name.startswith("cut_"):
        return "cut_block"

    # Special case: smooth variants
    if base_name.startswith("smooth_"):
        rest = base_name[7:]
        if any(rest.endswith(s) for s in ["_slab", "_stairs"]):
            pass  # Let it fall through to pattern matching
        else:
            return "smooth_block"

    # Special case: waxed copper variants
    if base_name.startswith("waxed_"):
        rest = base_name[6:]
        if rest.endswith("_slab"):
            return "slab"
        elif rest.endswith("_stairs"):
            return "stairs"
        else:
            return "waxed_block"

    # Special case: raw ore blocks
    if base_name.startswith("raw_") and base_name.endswith("_block"):
        return "raw_block"

    # Check shape patterns (longer patterns first for correct matching)
    sorted_patterns = sorted(SHAPE_PATTERNS.keys(), key=len, reverse=True)
    for pattern in sorted_patterns:
        if base_name.endswith(pattern):
            return SHAPE_PATTERNS[pattern]

    # Default to "block" for full blocks
    return "block"


def identify_properties(base_name: str, shape: str) -> list[str]:
    """Identify functional properties of a block."""
    properties = []

    # Check transparency
    for trans_keyword in TRANSPARENT_BLOCKS:
        if trans_keyword in base_name or trans_keyword == shape:
            properties.append("transparent")
            break

    # Check light emission
    for light_keyword in LIGHT_EMITTING:
        if light_keyword in base_name or light_keyword == shape:
            properties.append("light_emitting")
            break

    # Check gravity
    for grav_keyword in GRAVITY_AFFECTED:
        if grav_keyword in base_name:
            properties.append("gravity")
            break

    # Check interactability
    for interact_keyword in INTERACTABLE:
        if interact_keyword in base_name or interact_keyword == shape:
            properties.append("interactable")
            break

    # Check redstone
    for redstone_keyword in REDSTONE_COMPONENTS:
        if redstone_keyword in base_name or redstone_keyword == shape:
            properties.append("redstone")
            break

    # Add solid property if not transparent and not special
    if "transparent" not in properties and base_name not in {"air", "water", "lava", "fire"}:
        properties.append("solid")

    return properties


def decompose_block(block_name: str) -> BlockComponents:
    """
    Decompose a block name into its semantic components.

    Args:
        block_name: Full block name (e.g., "minecraft:oak_stairs[facing=north]")

    Returns:
        BlockComponents with material, shape, and properties
    """
    base_name = extract_base_name(block_name)

    material = identify_material(base_name)
    shape = identify_shape(base_name)
    properties = identify_properties(base_name, shape)

    return BlockComponents(
        original_name=block_name,
        base_name=base_name,
        material=material,
        shape=shape,
        properties=properties
    )


def create_component_vocabularies(tok2block: dict[int, str]) -> dict:
    """
    Create vocabularies for materials, shapes, and properties from block vocabulary.

    Args:
        tok2block: Token ID to block name mapping

    Returns:
        Dictionary containing:
        - materials: list of unique materials
        - shapes: list of unique shapes
        - properties: list of unique properties
        - block_components: dict mapping token_id to BlockComponents
        - material2idx: material to index mapping
        - shape2idx: shape to index mapping
        - property2idx: property to index mapping
    """
    materials = set()
    shapes = set()
    properties = set()
    block_components = {}

    for token_id, block_name in tok2block.items():
        components = decompose_block(block_name)
        block_components[token_id] = components

        if components.material:
            materials.add(components.material)
        shapes.add(components.shape)
        properties.update(components.properties)

    # Create sorted lists for consistent indexing
    materials_list = ["_none_"] + sorted(materials)  # _none_ for blocks without material
    shapes_list = sorted(shapes)
    properties_list = sorted(properties)

    return {
        "materials": materials_list,
        "shapes": shapes_list,
        "properties": properties_list,
        "block_components": block_components,
        "material2idx": {m: i for i, m in enumerate(materials_list)},
        "shape2idx": {s: i for i, s in enumerate(shapes_list)},
        "property2idx": {p: i for i, p in enumerate(properties_list)},
    }


def print_decomposition_stats(vocab_data: dict):
    """Print statistics about the decomposition."""
    print(f"\n{'='*60}")
    print("Block Decomposition Statistics")
    print(f"{'='*60}")
    print(f"Materials: {len(vocab_data['materials'])} unique")
    print(f"Shapes: {len(vocab_data['shapes'])} unique")
    print(f"Properties: {len(vocab_data['properties'])} unique")

    print(f"\nMaterials: {vocab_data['materials'][:20]}...")
    print(f"\nShapes: {vocab_data['shapes']}")
    print(f"\nProperties: {vocab_data['properties']}")

    # Count blocks by shape
    shape_counts = {}
    for components in vocab_data['block_components'].values():
        shape = components.shape
        shape_counts[shape] = shape_counts.get(shape, 0) + 1

    print(f"\nBlocks per shape (top 15):")
    for shape, count in sorted(shape_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {shape}: {count}")


def save_decomposition(vocab_data: dict, output_dir: Path):
    """Save decomposition data to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save component vocabularies
    with open(output_dir / "materials.json", "w") as f:
        json.dump(vocab_data["materials"], f, indent=2)

    with open(output_dir / "shapes.json", "w") as f:
        json.dump(vocab_data["shapes"], f, indent=2)

    with open(output_dir / "properties.json", "w") as f:
        json.dump(vocab_data["properties"], f, indent=2)

    # Save block -> components mapping
    block_mapping = {}
    for token_id, components in vocab_data["block_components"].items():
        block_mapping[str(token_id)] = {
            "base_name": components.base_name,
            "material": components.material,
            "shape": components.shape,
            "properties": components.properties,
        }

    with open(output_dir / "block_components.json", "w") as f:
        json.dump(block_mapping, f, indent=2)

    print(f"Saved decomposition to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Decompose block vocabulary into components")
    parser.add_argument("--vocab", type=Path, required=True, help="Path to tok2block.json")
    parser.add_argument("--output", type=Path, default=Path("data/block_components"),
                        help="Output directory")
    parser.add_argument("--examples", type=int, default=20, help="Number of examples to show")

    args = parser.parse_args()

    # Load vocabulary
    with open(args.vocab) as f:
        tok2block = {int(k): v for k, v in json.load(f).items()}

    print(f"Loaded {len(tok2block)} blocks")

    # Create decomposition
    vocab_data = create_component_vocabularies(tok2block)

    # Print stats
    print_decomposition_stats(vocab_data)

    # Show examples
    print(f"\nExample decompositions:")
    examples = list(vocab_data['block_components'].items())[:args.examples]
    for token_id, components in examples:
        print(f"  {components}")

    # Save
    save_decomposition(vocab_data, args.output)
