"""Create t-SNE visualization with block textures for V3 embeddings."""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.manifold import TSNE
from pathlib import Path
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Paths
EMBEDDINGS_PATH = Path("C:/Users/namda/OneDrive/Desktop/Claude_Server/minecraft_ai/data/kaggle/output/block2vec/v3/block_embeddings_v3.npy")
VOCAB_PATH = Path("C:/Users/namda/OneDrive/Desktop/Claude_Server/minecraft_ai/data/kaggle/output/block2vec/v2/tok2block_collapsed.json")
TEXTURES_PATH = Path("C:/Users/namda/OneDrive/Desktop/Claude_Server/minecraft_ai/data/kaggle/plot/block_textures")
OUTPUT_PATH = Path("C:/Users/namda/OneDrive/Desktop/Claude_Server/minecraft_ai/data/kaggle/output/block2vec/v3/tsne_textures.png")

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
    # Remove minecraft: prefix
    name = block_name.replace("minecraft:", "")

    # Remove state info in brackets like [facing=north]
    if "[" in name:
        name = name.split("[")[0]

    # Common texture name patterns to try
    patterns = [
        f"{name}.png",
        f"{name}_side.png",
        f"{name}_front.png",
        f"{name}_top.png",
        f"{name}s.png",  # e.g., brick -> bricks
    ]

    # Special cases
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
    except Exception as e:
        return None

# Find textures for all blocks
print("\nFinding textures...")
block_textures = {}
found = 0
missing = []

for tok, block_name in tok2block.items():
    if block_name == "UNKNOWN_BLOCK":
        continue
    if "air" in block_name.lower() and "stair" not in block_name.lower():
        continue  # Skip air blocks

    tex_path = get_texture_path(block_name)
    if tex_path:
        tex = load_texture(tex_path)
        if tex is not None:
            block_textures[tok] = tex
            found += 1
    else:
        missing.append(block_name)

print(f"Found textures: {found}")
print(f"Missing textures: {len(missing)}")
if missing[:10]:
    print(f"Examples missing: {missing[:10]}")

# Filter to only blocks with textures
valid_tokens = list(block_textures.keys())
valid_embeddings = embeddings[valid_tokens]
print(f"\nUsing {len(valid_tokens)} blocks with textures")

# Run t-SNE
print("\nRunning t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(valid_tokens)-1))
coords = tsne.fit_transform(valid_embeddings)
print("t-SNE complete!")

# Create visualization
print("\nCreating visualization...")
fig, ax = plt.subplots(figsize=(20, 20), dpi=150)
ax.set_facecolor('#1a1a1a')
fig.patch.set_facecolor('#1a1a1a')

# Plot each texture
for i, tok in enumerate(valid_tokens):
    x, y = coords[i]
    tex = block_textures[tok]

    # Create image with slight zoom
    imagebox = OffsetImage(tex, zoom=1.0)
    imagebox.image.axes = ax

    ab = AnnotationBbox(imagebox, (x, y), frameon=False, pad=0)
    ax.add_artist(ab)

# Set limits with padding
x_margin = (coords[:, 0].max() - coords[:, 0].min()) * 0.05
y_margin = (coords[:, 1].max() - coords[:, 1].min()) * 0.05
ax.set_xlim(coords[:, 0].min() - x_margin, coords[:, 0].max() + x_margin)
ax.set_ylim(coords[:, 1].min() - y_margin, coords[:, 1].max() + y_margin)

ax.set_title("Block2Vec V3 t-SNE (Compositional Embeddings)",
             fontsize=16, color='white', pad=20)
ax.set_xlabel("t-SNE 1", fontsize=12, color='white')
ax.set_ylabel("t-SNE 2", fontsize=12, color='white')
ax.tick_params(colors='white')

plt.tight_layout()
plt.savefig(OUTPUT_PATH, facecolor='#1a1a1a', edgecolor='none', bbox_inches='tight')
print(f"\nSaved to: {OUTPUT_PATH}")

# Also create a zoomed version showing a subset
print("\nCreating zoomed version...")
fig2, ax2 = plt.subplots(figsize=(24, 24), dpi=150)
ax2.set_facecolor('#1a1a1a')
fig2.patch.set_facecolor('#1a1a1a')

for i, tok in enumerate(valid_tokens):
    x, y = coords[i]
    tex = block_textures[tok]
    imagebox = OffsetImage(tex, zoom=1.5)  # Larger textures
    imagebox.image.axes = ax2
    ab = AnnotationBbox(imagebox, (x, y), frameon=False, pad=0)
    ax2.add_artist(ab)

ax2.set_xlim(coords[:, 0].min() - x_margin, coords[:, 0].max() + x_margin)
ax2.set_ylim(coords[:, 1].min() - y_margin, coords[:, 1].max() + y_margin)
ax2.set_title("Block2Vec V3 t-SNE (Compositional Embeddings) - Large",
              fontsize=18, color='white', pad=20)
ax2.tick_params(colors='white')

plt.tight_layout()
zoomed_path = OUTPUT_PATH.parent / "tsne_textures_large.png"
plt.savefig(zoomed_path, facecolor='#1a1a1a', edgecolor='none', bbox_inches='tight')
print(f"Saved large version to: {zoomed_path}")

print("\nDone!")
