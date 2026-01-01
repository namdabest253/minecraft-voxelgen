"""Create t-SNE visualization with block textures and shape indicators for V3 embeddings."""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Rectangle
from sklearn.manifold import TSNE
from pathlib import Path
from PIL import Image
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Paths - detect WSL vs Windows
import platform
if platform.system() == "Linux" and "microsoft" in platform.release().lower():
    # WSL environment
    BASE_PATH = Path("/mnt/c/Users/namda/OneDrive/Desktop/Claude_Server/minecraft_ai")
else:
    # Windows environment
    BASE_PATH = Path("C:/Users/namda/OneDrive/Desktop/Claude_Server/minecraft_ai")

EMBEDDINGS_PATH = BASE_PATH / "data/kaggle/output/block2vec/v3/block_embeddings_v3.npy"
# CRITICAL: Must use FULL vocabulary (3717 entries) that matches embeddings, NOT collapsed (1007)
VOCAB_PATH = BASE_PATH / "data/vocabulary/tok2block.json"
VOCAB_INFO_PATH = BASE_PATH / "data/kaggle/output/block2vec/v3/vocab_info_v3.json"
TEXTURES_PATH = BASE_PATH / "data/kaggle/plot/block_textures"
OUTPUT_PATH = BASE_PATH / "data/kaggle/output/block2vec/v3/tsne_textures.png"

# Shape configuration: colors and line styles
# Line styles: '-' solid, '--' dashed, ':' dotted, '-.' dash-dot
SHAPE_CONFIG = {
    # Common building blocks - SOLID lines
    'planks':      {'color': '#FF6B6B', 'linestyle': '-', 'name': 'Planks'},
    'stairs':      {'color': '#4ECDC4', 'linestyle': '--', 'name': 'Stairs'},
    'slab':        {'color': '#45B7D1', 'linestyle': ':', 'name': 'Slabs'},
    'fence':       {'color': '#96CEB4', 'linestyle': '-.', 'name': 'Fence'},
    'fence_gate':  {'color': '#88D8B0', 'linestyle': '-.', 'name': 'Fence Gate'},
    'door':        {'color': '#FFEAA7', 'linestyle': '-', 'name': 'Doors'},
    'trapdoor':    {'color': '#FFD93D', 'linestyle': '--', 'name': 'Trapdoors'},
    'wall':        {'color': '#F4A460', 'linestyle': '-', 'name': 'Walls'},
    
    # Wood variants - DASHED lines
    'log':         {'color': '#DDA0DD', 'linestyle': '-', 'name': 'Logs'},
    'wood':        {'color': '#DA70D6', 'linestyle': '--', 'name': 'Wood'},
    'stripped_log':{'color': '#EE82EE', 'linestyle': ':', 'name': 'Stripped Log'},
    'stripped_wood':{'color': '#FF00FF', 'linestyle': ':', 'name': 'Stripped Wood'},
    'leaves':      {'color': '#90EE90', 'linestyle': '-', 'name': 'Leaves'},
    
    # Colored blocks - SOLID lines
    'wool':        {'color': '#FFB6C1', 'linestyle': '-', 'name': 'Wool'},
    'concrete':    {'color': '#87CEEB', 'linestyle': '-', 'name': 'Concrete'},
    'concrete_powder': {'color': '#ADD8E6', 'linestyle': '--', 'name': 'Concrete Powder'},
    'terracotta':  {'color': '#CD853F', 'linestyle': '-', 'name': 'Terracotta'},
    'glazed_terracotta': {'color': '#D2691E', 'linestyle': '--', 'name': 'Glazed Terracotta'},
    'carpet':      {'color': '#FFC0CB', 'linestyle': ':', 'name': 'Carpet'},
    'bed':         {'color': '#FF69B4', 'linestyle': '-', 'name': 'Beds'},
    
    # Glass - DOTTED lines
    'glass':       {'color': '#E0FFFF', 'linestyle': ':', 'name': 'Glass'},
    'pane':        {'color': '#B0E0E6', 'linestyle': ':', 'name': 'Glass Pane'},
    'stained_glass': {'color': '#AFEEEE', 'linestyle': ':', 'name': 'Stained Glass'},
    'stained_glass_pane': {'color': '#98FB98', 'linestyle': ':', 'name': 'Stained Glass Pane'},
    
    # Stone variants
    'block':       {'color': '#A9A9A9', 'linestyle': '-', 'name': 'Blocks'},
    'bricks':      {'color': '#B22222', 'linestyle': '-', 'name': 'Bricks'},
    'ore':         {'color': '#FFD700', 'linestyle': '-', 'name': 'Ores'},
    
    # Redstone & mechanisms
    'button':      {'color': '#FF4500', 'linestyle': ':', 'name': 'Buttons'},
    'pressure_plate': {'color': '#FF6347', 'linestyle': ':', 'name': 'Pressure Plates'},
    'lever':       {'color': '#DC143C', 'linestyle': '-.', 'name': 'Levers'},
    
    # Decorative
    'sign':        {'color': '#DEB887', 'linestyle': '-', 'name': 'Signs'},
    'wall_sign':   {'color': '#D2B48C', 'linestyle': '--', 'name': 'Wall Signs'},
    'hanging_sign':{'color': '#C4A484', 'linestyle': ':', 'name': 'Hanging Signs'},
    'banner':      {'color': '#9932CC', 'linestyle': '-', 'name': 'Banners'},
    'candle':      {'color': '#FFA07A', 'linestyle': ':', 'name': 'Candles'},
    'lantern':     {'color': '#FFE4B5', 'linestyle': '-', 'name': 'Lanterns'},
    'torch':       {'color': '#FF8C00', 'linestyle': '-', 'name': 'Torches'},
    
    # Plants
    'flower':      {'color': '#FF1493', 'linestyle': '-', 'name': 'Flowers'},
    'sapling':     {'color': '#228B22', 'linestyle': '-', 'name': 'Saplings'},
    'potted_plant':{'color': '#8B4513', 'linestyle': '-', 'name': 'Potted Plants'},
    
    # Misc
    'chest':       {'color': '#8B4513', 'linestyle': '-', 'name': 'Chests'},
    'shulker_box': {'color': '#9370DB', 'linestyle': '-', 'name': 'Shulker Boxes'},
    'coral':       {'color': '#FF7F50', 'linestyle': '-', 'name': 'Coral'},
    'coral_block': {'color': '#FF6347', 'linestyle': '-', 'name': 'Coral Block'},
}

DEFAULT_CONFIG = {'color': '#666666', 'linestyle': '-', 'name': 'Other'}


def identify_shape(block_name: str) -> str:
    """Identify the shape from a block name."""
    name = block_name.replace("minecraft:", "")
    if "[" in name:
        name = name.split("[")[0]
    
    # Check for stripped variants first
    if name.startswith("stripped_"):
        if "_log" in name:
            return "stripped_log"
        elif "_wood" in name:
            return "stripped_wood"
    
    # Check patterns in order (longer first to avoid partial matches)
    shape_patterns = [
        ("_stained_glass_pane", "stained_glass_pane"),
        ("_stained_glass", "stained_glass"),
        ("_glazed_terracotta", "glazed_terracotta"),
        ("_concrete_powder", "concrete_powder"),
        ("_pressure_plate", "pressure_plate"),
        ("_wall_hanging_sign", "hanging_sign"),
        ("_hanging_sign", "hanging_sign"),
        ("_wall_sign", "wall_sign"),
        ("_wall_banner", "banner"),
        ("_fence_gate", "fence_gate"),
        ("_coral_block", "coral_block"),
        ("_shulker_box", "shulker_box"),
        ("_trapdoor", "trapdoor"),
        ("_concrete", "concrete"),
        ("_terracotta", "terracotta"),
        ("_planks", "planks"),
        ("_stairs", "stairs"),
        ("_button", "button"),
        ("_carpet", "carpet"),
        ("_slab", "slab"),
        ("_fence", "fence"),
        ("_door", "door"),
        ("_wall", "wall"),
        ("_sign", "sign"),
        ("_wool", "wool"),
        ("_log", "log"),
        ("_wood", "wood"),
        ("_bed", "bed"),
        ("_ore", "ore"),
        ("_pane", "pane"),
        ("_glass", "glass"),
        ("_leaves", "leaves"),
        ("_sapling", "sapling"),
        ("_banner", "banner"),
        ("_candle", "candle"),
        ("_coral", "coral"),
        ("bricks", "bricks"),
        ("lantern", "lantern"),
        ("torch", "torch"),
        ("chest", "chest"),
    ]
    
    # Check for potted plants
    if name.startswith("potted_"):
        return "potted_plant"
    
    for pattern, shape in shape_patterns:
        if pattern in name:
            return shape
    
    return "block"


def get_texture_path(block_name: str) -> Path:
    """Find texture file for a block."""
    name = block_name.replace("minecraft:", "")
    if "[" in name:
        name = name.split("[")[0]
    
    patterns = [
        f"{name}.png",
        f"{name}_side.png",
        f"{name}_front.png",
        f"{name}_top.png",
        f"{name}s.png",
    ]
    
    if "_stairs" in name:
        base = name.replace("_stairs", "")
        patterns.extend([f"{base}.png", f"{base}_planks.png", f"{base}s.png"])
    elif "_slab" in name:
        base = name.replace("_slab", "")
        patterns.extend([f"{base}.png", f"{base}_planks.png", f"{base}s.png"])
    elif "_fence" in name:
        base = name.replace("_fence", "").replace("_gate", "")
        patterns.extend([f"{base}.png", f"{base}_planks.png"])
    elif "_door" in name:
        patterns.extend([f"{name}_bottom.png", f"{name}_top.png"])
    elif "_wall" in name:
        base = name.replace("_wall", "")
        patterns.extend([f"{base}.png", f"{base}s.png"])
    elif "_planks" in name:
        patterns.append(f"{name}.png")
    elif "_log" in name:
        patterns.append(f"{name}.png")
    elif "_wool" in name:
        base = name.replace("_wool", "")
        patterns.extend([f"{base}_wool.png"])
    elif "_concrete" in name and "_powder" not in name:
        patterns.append(f"{name}.png")
    elif "_terracotta" in name:
        patterns.append(f"{name}.png")
    elif "_glass" in name:
        patterns.append(f"{name}.png")
    elif "_bed" in name:
        base = name.replace("_bed", "")
        patterns.extend([f"{base}_wool.png"])
    elif "_button" in name:
        base = name.replace("_button", "")
        patterns.extend([f"{base}.png", f"{base}_planks.png"])
    elif "_pressure_plate" in name:
        base = name.replace("_pressure_plate", "")
        patterns.extend([f"{base}.png", f"{base}_planks.png"])
    elif "_sign" in name:
        base = name.replace("_sign", "").replace("_wall", "").replace("_hanging", "")
        patterns.extend([f"{base}.png", f"{base}_planks.png"])
    elif "_trapdoor" in name:
        patterns.append(f"{name}.png")
    elif "_leaves" in name:
        patterns.append(f"{name}.png")
    elif "_sapling" in name:
        patterns.append(f"{name}.png")
    elif "ore" in name:
        patterns.append(f"{name}.png")
    
    for pattern in patterns:
        path = TEXTURES_PATH / pattern
        if path.exists():
            return path
    
    return None


def load_texture(path: Path, size: int = 16) -> np.ndarray:
    """Load and resize a texture."""
    try:
        img = Image.open(path).convert("RGBA")
        img = img.resize((size, size), Image.Resampling.NEAREST)
        return np.array(img)
    except Exception:
        return None


def main():
    # Load data
    print("Loading embeddings...")
    embeddings = np.load(EMBEDDINGS_PATH)
    print(f"Embeddings shape: {embeddings.shape}")
    
    print("Loading vocabulary...")
    with open(VOCAB_PATH) as f:
        tok2block = {int(k): v for k, v in json.load(f).items()}
    print(f"Vocabulary size: {len(tok2block)}")
    
    # Find textures for all blocks
    # IMPORTANT: The full vocabulary has block states (e.g., oak_log[axis=x], oak_log[axis=y])
    # We deduplicate by taking the FIRST token for each base block name
    print("\nFinding textures...")
    block_textures = {}
    block_shapes = {}
    block_names = {}
    seen_blocks = set()  # Track which base block names we've already added
    found = 0
    missing = []
    
    for tok, block_name in sorted(tok2block.items()):  # Sort to get consistent first token
        if block_name == "UNKNOWN_BLOCK":
            continue
        if "air" in block_name.lower() and "stair" not in block_name.lower():
            continue
        
        # Get the base block name (without state)
        clean_name = block_name.replace("minecraft:", "").split("[")[0]
        
        # Skip if we've already added this base block
        if clean_name in seen_blocks:
            continue
        
        tex_path = get_texture_path(block_name)
        if tex_path:
            tex = load_texture(tex_path)
            if tex is not None:
                seen_blocks.add(clean_name)
                block_textures[tok] = tex
                block_shapes[tok] = identify_shape(block_name)
                block_names[tok] = clean_name
                found += 1
        else:
            if clean_name not in seen_blocks:
                missing.append(block_name)
    
    print(f"Found textures: {found} unique blocks (from {len(tok2block)} vocab entries)")
    print(f"Missing textures: {len(missing)}")
    
    # Count shapes
    shape_counts = Counter(block_shapes.values())
    print(f"\nShape distribution (top 15):")
    for shape, count in shape_counts.most_common(15):
        config = SHAPE_CONFIG.get(shape, DEFAULT_CONFIG)
        print(f"  {shape}: {count} blocks ({config['linestyle']} {config['color']})")
    
    # Filter to only blocks with textures
    valid_tokens = list(block_textures.keys())
    valid_embeddings = embeddings[valid_tokens]
    print(f"\nUsing {len(valid_tokens)} blocks with textures")
    
    # Run t-SNE
    print("\nRunning t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(valid_tokens)-1))
    coords = tsne.fit_transform(valid_embeddings)
    print("t-SNE complete!")
    
    # Calculate scale for texture sizing
    x_range = coords[:, 0].max() - coords[:, 0].min()
    y_range = coords[:, 1].max() - coords[:, 1].min()
    scale = max(x_range, y_range)
    tex_size = scale * 0.025
    
    # Create clean visualization with just textures (no outlines)
    print("\nCreating visualization...")
    fig, ax = plt.subplots(figsize=(24, 24), dpi=150)
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')
    
    for i, tok in enumerate(valid_tokens):
        x, y = coords[i]
        tex = block_textures[tok]
        
        # Draw texture only
        imagebox = OffsetImage(tex, zoom=1.0)
        imagebox.image.axes = ax
        ab = AnnotationBbox(imagebox, (x, y), frameon=False, pad=0, zorder=2)
        ax.add_artist(ab)
    
    # Set limits with padding
    x_margin = x_range * 0.05
    y_margin = y_range * 0.05
    ax.set_xlim(coords[:, 0].min() - x_margin, coords[:, 0].max() + x_margin)
    ax.set_ylim(coords[:, 1].min() - y_margin, coords[:, 1].max() + y_margin)
    
    ax.set_title("Block2Vec V3 t-SNE Embeddings",
                 fontsize=18, color='white', pad=20, weight='bold')
    ax.set_xlabel("t-SNE 1", fontsize=12, color='white')
    ax.set_ylabel("t-SNE 2", fontsize=12, color='white')
    ax.tick_params(colors='white')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, facecolor='#1a1a2e', edgecolor='none', bbox_inches='tight')
    print(f"\nSaved to: {OUTPUT_PATH}")
    
    # Also create a larger version
    print("\nCreating large version...")
    fig2, ax2 = plt.subplots(figsize=(32, 32), dpi=150)
    ax2.set_facecolor('#1a1a2e')
    fig2.patch.set_facecolor('#1a1a2e')
    
    for i, tok in enumerate(valid_tokens):
        x, y = coords[i]
        tex = block_textures[tok]
        
        # Draw texture (larger)
        imagebox = OffsetImage(tex, zoom=1.5)
        imagebox.image.axes = ax2
        ab = AnnotationBbox(imagebox, (x, y), frameon=False, pad=0, zorder=2)
        ax2.add_artist(ab)
    
    ax2.set_xlim(coords[:, 0].min() - x_margin, coords[:, 0].max() + x_margin)
    ax2.set_ylim(coords[:, 1].min() - y_margin, coords[:, 1].max() + y_margin)
    
    ax2.set_title("Block2Vec V3 t-SNE Embeddings (Large)",
                  fontsize=20, color='white', pad=20, weight='bold')
    ax2.tick_params(colors='white')
    
    plt.tight_layout()
    large_path = OUTPUT_PATH.parent / "tsne_textures_large.png"
    plt.savefig(large_path, facecolor='#1a1a2e', edgecolor='none', bbox_inches='tight')
    print(f"Saved large version to: {large_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

