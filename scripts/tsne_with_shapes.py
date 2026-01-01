"""Create t-SNE visualization with block textures and shape outlines."""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Rectangle
from sklearn.manifold import TSNE
from pathlib import Path
from PIL import Image, ImageDraw
import warnings
warnings.filterwarnings('ignore')

# Paths
EMBEDDINGS_PATH = Path("C:/Users/namda/OneDrive/Desktop/Claude_Server/minecraft_ai/data/kaggle/output/block2vec/v3/block_embeddings_v3.npy")
VOCAB_PATH = Path("C:/Users/namda/OneDrive/Desktop/Claude_Server/minecraft_ai/data/kaggle/output/block2vec/v2/tok2block_collapsed.json")
TEXTURES_PATH = Path("C:/Users/namda/OneDrive/Desktop/Claude_Server/minecraft_ai/data/kaggle/plot/block_textures")
OUTPUT_PATH = Path("C:/Users/namda/OneDrive/Desktop/Claude_Server/minecraft_ai/data/kaggle/output/block2vec/v3/tsne_with_shapes.png")

# Shape categories and their colors (RGB)
SHAPE_COLORS = {
    'stairs': (255, 0, 0),        # Red
    'slab': (0, 0, 255),          # Blue
    'wall': (0, 255, 0),          # Green
    'fence': (255, 165, 0),       # Orange
    'fence_gate': (255, 140, 0),  # Dark Orange
    'door': (255, 0, 255),        # Magenta
    'trapdoor': (200, 0, 200),    # Purple
    'button': (255, 255, 0),      # Yellow
    'pressure_plate': (200, 200, 0),  # Dark Yellow
    'sign': (0, 255, 255),        # Cyan
    'planks': (139, 69, 19),      # Brown
    'log': (101, 67, 33),         # Dark Brown
    'ore': (255, 215, 0),         # Gold
    'leaves': (34, 139, 34),      # Forest Green
    'wool': (255, 182, 193),      # Light Pink
    'concrete': (128, 128, 128),  # Gray
    'terracotta': (205, 92, 92),  # Indian Red
    'glass': (173, 216, 230),     # Light Blue
    'carpet': (238, 130, 238),    # Violet
    # Default for full blocks
    'block': None,                # No outline
}

def get_block_shape(block_name: str) -> str:
    """Determine the shape category of a block."""
    name = block_name.replace("minecraft:", "").lower()

    # Remove state info
    if "[" in name:
        name = name.split("[")[0]

    # Check for specific shapes
    if "_stairs" in name:
        return "stairs"
    elif "_slab" in name:
        return "slab"
    elif "_wall" in name and "wall_" not in name:
        return "wall"
    elif "_fence_gate" in name:
        return "fence_gate"
    elif "_fence" in name:
        return "fence"
    elif "_door" in name:
        return "door"
    elif "_trapdoor" in name:
        return "trapdoor"
    elif "_button" in name:
        return "button"
    elif "_pressure_plate" in name:
        return "pressure_plate"
    elif "_sign" in name:
        return "sign"
    elif "_planks" in name:
        return "planks"
    elif "_log" in name or "_wood" in name:
        return "log"
    elif "_ore" in name:
        return "ore"
    elif "_leaves" in name:
        return "leaves"
    elif "_wool" in name:
        return "wool"
    elif "_concrete" in name and "_powder" not in name:
        return "concrete"
    elif "_terracotta" in name:
        return "terracotta"
    elif "_glass" in name:
        return "glass"
    elif "_carpet" in name:
        return "carpet"
    else:
        return "block"

def add_outline(img: Image.Image, color: tuple, thickness: int = 2) -> Image.Image:
    """Add a colored outline around the image."""
    if color is None:
        return img

    # Create a new image with space for outline
    new_size = (img.width + thickness * 2, img.height + thickness * 2)
    new_img = Image.new("RGBA", new_size, (0, 0, 0, 0))

    # Draw outline
    draw = ImageDraw.Draw(new_img)
    draw.rectangle(
        [0, 0, new_size[0] - 1, new_size[1] - 1],
        outline=(*color, 255),
        width=thickness
    )

    # Paste original image in center
    new_img.paste(img, (thickness, thickness))

    return new_img

# Load data
print("Loading embeddings...")
embeddings = np.load(EMBEDDINGS_PATH)
print(f"Embeddings shape: {embeddings.shape}")

print("Loading vocabulary...")
with open(VOCAB_PATH) as f:
    tok2block = {int(k): v for k, v in json.load(f).items()}
print(f"Vocabulary size: {len(tok2block)}")

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
    elif "_wool" in name:
        base = name.replace("_wool", "")
        patterns.extend([f"{base}_wool.png"])
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

def load_texture(path: Path, size: int = 16) -> Image.Image:
    """Load and resize a texture."""
    try:
        img = Image.open(path).convert("RGBA")
        img = img.resize((size, size), Image.Resampling.NEAREST)
        return img
    except:
        return None

# Find textures and shapes for all blocks
print("\nFinding textures and categorizing shapes...")
block_data = {}
shape_counts = {}
found = 0

for tok, block_name in tok2block.items():
    if block_name == "UNKNOWN_BLOCK":
        continue
    if "air" in block_name.lower() and "stair" not in block_name.lower():
        continue

    tex_path = get_texture_path(block_name)
    if tex_path:
        tex = load_texture(tex_path)
        if tex is not None:
            shape = get_block_shape(block_name)
            color = SHAPE_COLORS.get(shape, None)
            tex_with_outline = add_outline(tex, color, thickness=2)

            block_data[tok] = {
                'texture': np.array(tex_with_outline),
                'shape': shape,
                'name': block_name
            }
            shape_counts[shape] = shape_counts.get(shape, 0) + 1
            found += 1

print(f"Found textures: {found}")
print(f"\nShape distribution:")
for shape, count in sorted(shape_counts.items(), key=lambda x: -x[1]):
    color = SHAPE_COLORS.get(shape)
    color_str = f"RGB{color}" if color else "No outline"
    print(f"  {shape}: {count} blocks ({color_str})")

# Filter to only blocks with textures
valid_tokens = list(block_data.keys())
valid_embeddings = embeddings[valid_tokens]
print(f"\nUsing {len(valid_tokens)} blocks with textures")

# Run t-SNE
print("\nRunning t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(valid_tokens)-1))
coords = tsne.fit_transform(valid_embeddings)
print("t-SNE complete!")

# Create visualization
print("\nCreating visualization...")
fig, ax = plt.subplots(figsize=(24, 24), dpi=150)
ax.set_facecolor('#1a1a1a')
fig.patch.set_facecolor('#1a1a1a')

for i, tok in enumerate(valid_tokens):
    x, y = coords[i]
    tex = block_data[tok]['texture']

    imagebox = OffsetImage(tex, zoom=1.2)
    imagebox.image.axes = ax
    ab = AnnotationBbox(imagebox, (x, y), frameon=False, pad=0)
    ax.add_artist(ab)

x_margin = (coords[:, 0].max() - coords[:, 0].min()) * 0.05
y_margin = (coords[:, 1].max() - coords[:, 1].min()) * 0.05
ax.set_xlim(coords[:, 0].min() - x_margin, coords[:, 0].max() + x_margin)
ax.set_ylim(coords[:, 1].min() - y_margin, coords[:, 1].max() + y_margin)

ax.set_title("Block2Vec V3 t-SNE with Shape Indicators", fontsize=18, color='white', pad=20)
ax.tick_params(colors='white')

# Add legend
legend_y = 0.98
legend_x = 0.02
for shape, color in SHAPE_COLORS.items():
    if color is not None and shape_counts.get(shape, 0) > 0:
        ax.annotate(
            f"■ {shape} ({shape_counts.get(shape, 0)})",
            xy=(legend_x, legend_y),
            xycoords='axes fraction',
            fontsize=10,
            color=[c/255 for c in color],
            fontweight='bold',
            va='top'
        )
        legend_y -= 0.025

ax.annotate(
    f"□ block/other ({shape_counts.get('block', 0)})",
    xy=(legend_x, legend_y),
    xycoords='axes fraction',
    fontsize=10,
    color='white',
    va='top'
)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, facecolor='#1a1a1a', edgecolor='none', bbox_inches='tight')
print(f"\nSaved to: {OUTPUT_PATH}")

# Also analyze clustering quality
print("\n" + "="*60)
print("CLUSTERING ANALYSIS")
print("="*60)

# For each shape, check if blocks are clustered
from scipy.spatial.distance import pdist, cdist

for shape in ['stairs', 'slab', 'planks', 'ore', 'wool', 'log']:
    shape_tokens = [tok for tok in valid_tokens if block_data[tok]['shape'] == shape]
    if len(shape_tokens) < 2:
        continue

    shape_indices = [valid_tokens.index(tok) for tok in shape_tokens]
    shape_coords = coords[shape_indices]

    # Average distance within shape
    within_dist = np.mean(pdist(shape_coords))

    # Average distance to all other points
    other_indices = [i for i in range(len(valid_tokens)) if i not in shape_indices]
    other_coords = coords[other_indices]
    across_dist = np.mean(cdist(shape_coords, other_coords))

    ratio = within_dist / across_dist if across_dist > 0 else 0
    status = "✓ CLUSTERED" if ratio < 0.7 else "⚠ SCATTERED" if ratio < 1.0 else "✗ NOT CLUSTERED"

    print(f"{shape:15} ({len(shape_tokens):3} blocks): within={within_dist:.1f}, across={across_dist:.1f}, ratio={ratio:.2f} {status}")

print("\nDone!")
