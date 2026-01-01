"""Script to update V4 notebook with SHAPE_GROUPS."""
import json

# Read the notebook
with open(r'C:\Users\namda\OneDrive\Desktop\Claude_Server\minecraft_ai\data\kaggle\notebooks\block2vec_v4_compositional.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The SHAPE_GROUPS dictionary to add to cell 7
SHAPE_GROUPS_CODE = '''
# === V4 SHAPE GROUPS ===
# Map specific blocks to common shape categories to improve clustering
# This reduces unique shapes from ~320 to ~230

SHAPE_GROUPS = {
    # FLOWERS -> "flower" or "tall_flower"
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

    # CORALS -> "coral" or "dead_coral"
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

    # HEADS -> "head"
    "creeper_head": "head",
    "dragon_head": "head",
    "piglin_head": "head",
    "player_head": "head",
    "zombie_head": "head",
    "skeleton_skull": "skull",
    "wither_skeleton_skull": "skull",

    # RAILS -> "rail"
    "activator_rail": "rail",
    "detector_rail": "rail",
    "powered_rail": "rail",

    # CROPS -> "crop"
    "beetroots": "crop",
    "carrots": "crop",
    "potatoes": "crop",
    "wheat": "crop",
    "pitcher_crop": "crop",
    "torchflower_crop": "crop",
    "nether_wart": "crop",
    "cocoa": "crop",
    "sweet_berry_bush": "crop",

    # STEMS -> "stem"
    "melon_stem": "stem",
    "pumpkin_stem": "stem",
    "attached_melon_stem": "stem",
    "attached_pumpkin_stem": "stem",

    # ANVILS -> "anvil"
    "chipped_anvil": "anvil",
    "damaged_anvil": "anvil",

    # COMMAND BLOCKS -> "command_block"
    "chain_command_block": "command_block",
    "repeating_command_block": "command_block",

    # INFESTED BLOCKS -> "infested_block"
    "infested_stone": "infested_block",
    "infested_cobblestone": "infested_block",
    "infested_stone_bricks": "infested_block",
    "infested_mossy_stone_bricks": "infested_block",
    "infested_cracked_stone_bricks": "infested_block",
    "infested_chiseled_stone_bricks": "infested_block",
    "infested_deepslate": "infested_block",

    # AIR -> "air"
    "cave_air": "air",
    "void_air": "air",

    # VINES -> "vines"
    "cave_vines": "vines",
    "cave_vines_plant": "vines",
    "twisting_vines": "vines",
    "twisting_vines_plant": "vines",
    "weeping_vines": "vines",
    "weeping_vines_plant": "vines",
    "vine": "vines",
    "glow_lichen": "vines",

    # TORCHES -> "torch" or "wall_torch"
    "soul_torch": "torch",
    "redstone_torch": "torch",
    "soul_wall_torch": "wall_torch",
    "redstone_wall_torch": "wall_torch",

    # LANTERNS -> "lantern"
    "soul_lantern": "lantern",
    "sea_lantern": "lantern",

    # CAMPFIRES -> "campfire"
    "soul_campfire": "campfire",

    # FIRE -> "fire"
    "soul_fire": "fire",

    # FUNGI -> "fungus"
    "crimson_fungus": "fungus",
    "warped_fungus": "fungus",

    # NYLIUM -> "nylium"
    "crimson_nylium": "nylium",
    "warped_nylium": "nylium",

    # CAULDRONS -> "cauldron"
    "water_cauldron": "cauldron",
    "lava_cauldron": "cauldron",
    "powder_snow_cauldron": "cauldron",

    # GRASS/FERN -> "grass" or "tall_grass"
    "fern": "grass",
    "dead_bush": "grass",
    "tall_grass": "tall_grass",
    "large_fern": "tall_grass",

    # ICE -> "ice"
    "packed_ice": "ice",
    "blue_ice": "ice",
    "frosted_ice": "ice",

    # KELP -> "kelp"
    "kelp_plant": "kelp",

    # SEAGRASS -> "seagrass"
    "tall_seagrass": "seagrass",

    # SCULK -> "sculk" or "sculk_sensor"
    "sculk_catalyst": "sculk",
    "sculk_shrieker": "sculk",
    "sculk_vein": "sculk",
    "calibrated_sculk_sensor": "sculk_sensor",

    # AZALEA -> "azalea"
    "flowering_azalea": "azalea",
    "azalea_leaves_flowers": "azalea",

    # DRIPLEAF -> "dripleaf"
    "big_dripleaf": "dripleaf",
    "big_dripleaf_stem": "dripleaf",
    "small_dripleaf": "dripleaf",

    # FRAMES -> "item_frame"
    "glow_item_frame": "item_frame",

    # WART BLOCKS -> "wart_block"
    "nether_wart_block": "wart_block",
    "warped_wart_block": "wart_block",

    # SUSPICIOUS -> "suspicious_block"
    "suspicious_sand": "suspicious_block",
    "suspicious_gravel": "suspicious_block",

    # FURNACES -> "furnace"
    "blast_furnace": "furnace",
    "smoker": "furnace",

    # PISTONS -> "piston"
    "sticky_piston": "piston",
    "moving_piston": "piston",

    # DIRT -> "dirt"
    "coarse_dirt": "dirt",
    "rooted_dirt": "dirt",

    # ROOTS -> "roots"
    "hanging_roots": "roots",

    # MUSHROOM BLOCKS -> "mushroom_block"
    "mushroom_stem": "mushroom_block",

    # AMETHYST -> "amethyst_cluster"
    "amethyst_bud": "amethyst_cluster",

    # CHEST -> "chest"
    "trapped_chest": "chest",
    "ender_chest": "chest",

    # SIGN variants (wall signs handled by suffix)
    "wall_hanging_sign": "hanging_sign",
}

'''

# Step 1: Replace V3 with V4 in all cells
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' or cell['cell_type'] == 'markdown':
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        source = source.replace('V3', 'V4')
        source = source.replace('v3', 'v4')
        source = source.replace('V1 and V2', 'V1, V2, and V3')
        cell['source'] = source

print("Step 1: Replaced V3 -> V4 in all cells")

# Step 2: Add SHAPE_GROUPS to cell 7 (after STANDALONE_SHAPES)
source7 = nb['cells'][7]['source']
# Find the end of STANDALONE_SHAPES set and add SHAPE_GROUPS after it
marker = '"sea_pickle", "suspicious_gravel", "suspicious_sand",\n}'
if marker in source7:
    insert_pos = source7.find(marker) + len(marker)
    source7 = source7[:insert_pos] + "\n" + SHAPE_GROUPS_CODE + source7[insert_pos:]
    nb['cells'][7]['source'] = source7
    print("Step 2: Added SHAPE_GROUPS dictionary to cell 7")
else:
    print("WARNING: Could not find STANDALONE_SHAPES end marker in cell 7")

# Also add print statement for SHAPE_GROUPS count at the end of cell 7
if 'print(f"Defined {len(STANDALONE_SHAPES)}' in nb['cells'][7]['source']:
    nb['cells'][7]['source'] = nb['cells'][7]['source'].replace(
        'print(f"Defined {len(STANDALONE_SHAPES)} standalone shapes")',
        'print(f"Defined {len(STANDALONE_SHAPES)} standalone shapes")\nprint(f"Defined {len(SHAPE_GROUPS)} shape group mappings (V4)")'
    )
    print("Step 2b: Added SHAPE_GROUPS count print")

# Step 3: Modify identify_shape in cell 8 to check SHAPE_GROUPS first
source8 = nb['cells'][8]['source']
old_identify = '''def identify_shape(base_name: str) -> str:
    """
    Identify the shape component of a block name.

    Examples:
        "oak_planks" '''

new_identify_start = '''def identify_shape(base_name: str) -> str:
    """
    Identify the shape component of a block name.

    V4: First checks SHAPE_GROUPS for block consolidation.

    Examples:
        "oak_planks" '''

# Replace the function docstring start
if old_identify in source8:
    source8 = source8.replace(old_identify, new_identify_start)
    print("Step 3a: Updated identify_shape docstring")

# Add the SHAPE_GROUPS check after the docstring
old_check = '''    # Check standalone shapes first
    if base_name in STANDALONE_SHAPES:
        return base_name'''

new_check = '''    # V4: Check shape groups first to consolidate similar blocks
    if base_name in SHAPE_GROUPS:
        return SHAPE_GROUPS[base_name]

    # Check standalone shapes
    if base_name in STANDALONE_SHAPES:
        return base_name'''

if old_check in source8:
    source8 = source8.replace(old_check, new_check)
    nb['cells'][8]['source'] = source8
    print("Step 3b: Added SHAPE_GROUPS check to identify_shape")
else:
    print("WARNING: Could not find identify_shape standalone check pattern")

# Save the notebook
with open(r'C:\Users\namda\OneDrive\Desktop\Claude_Server\minecraft_ai\data\kaggle\notebooks\block2vec_v4_compositional.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("\nV4 notebook updated successfully!")
print("Changes made:")
print("  - All V3/v3 references changed to V4/v4")
print("  - SHAPE_GROUPS dictionary added to cell 7")
print("  - identify_shape() updated to check SHAPE_GROUPS first")
