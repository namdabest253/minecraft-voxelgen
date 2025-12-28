"""
Visualize t-SNE embeddings with actual Minecraft block textures.

Instead of colored dots, this script places small block texture images
at each t-SNE coordinate, making clusters much more interpretable.

Usage:
    python scripts/visualize_tsne_blocks.py \
        --embeddings models/block2vec/block_embeddings.npy \
        --vocab data/kaggle/output/tok2block.json \
        --textures path/to/minecraft/textures \
        --output outputs/tsne_blocks.png
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving files
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from sklearn.manifold import TSNE


def load_texture(texture_path: Path, size: int = 16) -> Optional[np.ndarray]:
    """Load and resize a texture image."""
    try:
        img = Image.open(texture_path).convert("RGBA")
        img = img.resize((size, size), Image.NEAREST)  # Keep pixelated look
        return np.array(img)
    except Exception:
        return None


def get_block_base_name(block_name: str) -> str:
    """
    Extract base block name from full Minecraft block state.

    Examples:
        'minecraft:oak_planks' -> 'oak_planks'
        'minecraft:oak_stairs[facing=north,half=bottom]' -> 'oak_stairs'
        'minecraft:stone_bricks' -> 'stone_bricks'
    """
    # Remove namespace
    name = block_name.replace("minecraft:", "")
    # Remove block state properties
    name = re.sub(r"\[.*\]", "", name)
    return name


def find_texture(block_name: str, texture_dir: Path) -> Optional[Path]:
    """
    Find the texture file for a block.

    Minecraft textures are usually named after the block, but some have
    different naming conventions (e.g., 'grass_block_top', 'oak_log_top').
    """
    base_name = get_block_base_name(block_name)

    # Try different naming patterns
    patterns = [
        f"{base_name}.png",
        f"{base_name}_top.png",  # For blocks with different top texture
        f"{base_name}_side.png",
        f"{base_name}_front.png",
        f"{base_name}s.png",  # Pluralized (e.g., bricks)
    ]

    # Special cases - map block names to texture files
    special_mappings = {
        "grass_block": "grass_block_top.png",
        "dirt_path": "dirt_path_top.png",
        "farmland": "farmland_moist.png",
        "water": "water_still.png",
        "lava": "lava_still.png",
        "fire": "fire_0.png",
        "wall_torch": "torch.png",
        "redstone_wall_torch": "redstone_torch.png",
        "soul_wall_torch": "soul_torch.png",
        "piston_head": "piston_top.png",
        "moving_piston": "piston_side.png",
    }

    if base_name in special_mappings:
        patterns.insert(0, special_mappings[base_name])

    for pattern in patterns:
        texture_path = texture_dir / pattern
        if texture_path.exists():
            return texture_path

    # Handle derived block types (stairs, slabs, walls, fences, etc.)
    # These share textures with their base material
    derived_suffixes = [
        "_stairs", "_slab", "_wall", "_fence", "_fence_gate",
        "_button", "_pressure_plate", "_sign", "_wall_sign",
        "_hanging_sign", "_wall_hanging_sign", "_door", "_trapdoor"
    ]
    for suffix in derived_suffixes:
        if base_name.endswith(suffix):
            material = base_name[:-len(suffix)]
            # Try common material texture names
            material_patterns = [
                f"{material}.png",
                f"{material}_planks.png",  # Wood types
                f"{material}_block.png",   # Some block types
                f"{material}s.png",        # Pluralized
            ]
            for mp in material_patterns:
                texture_path = texture_dir / mp
                if texture_path.exists():
                    return texture_path

    # Handle wall versions of blocks (e.g., wall_banner -> banner)
    if base_name.startswith("wall_"):
        non_wall = base_name[5:]  # Remove "wall_" prefix
        for pattern in [f"{non_wall}.png", f"{non_wall}_top.png"]:
            texture_path = texture_dir / pattern
            if texture_path.exists():
                return texture_path

    # Handle potted plants (e.g., potted_oak_sapling -> oak_sapling)
    if base_name.startswith("potted_"):
        plant = base_name[7:]  # Remove "potted_" prefix
        texture_path = texture_dir / f"{plant}.png"
        if texture_path.exists():
            return texture_path

    # Handle infested blocks (e.g., infested_stone -> stone)
    if base_name.startswith("infested_"):
        real_block = base_name[9:]  # Remove "infested_" prefix
        texture_path = texture_dir / f"{real_block}.png"
        if texture_path.exists():
            return texture_path

    # Try partial match - search for any texture containing base_name
    for texture_path in texture_dir.glob(f"*{base_name}*.png"):
        return texture_path

    # Try recursive search as last resort
    for texture_path in texture_dir.rglob(f"{base_name}*.png"):
        return texture_path

    return None


def create_placeholder_texture(color: Tuple[int, int, int], size: int = 16) -> np.ndarray:
    """Create a solid color placeholder texture."""
    img = np.zeros((size, size, 4), dtype=np.uint8)
    img[:, :, 0] = color[0]
    img[:, :, 1] = color[1]
    img[:, :, 2] = color[2]
    img[:, :, 3] = 255
    return img


def get_block_color(block_name: str) -> Tuple[int, int, int]:
    """Get a representative color for a block based on its name."""
    name = block_name.lower()

    color_map = {
        "stone": (128, 128, 128),
        "dirt": (139, 90, 43),
        "grass": (86, 125, 70),
        "wood": (160, 120, 80),
        "plank": (180, 140, 90),
        "log": (100, 80, 50),
        "sand": (219, 211, 160),
        "gravel": (136, 126, 126),
        "gold": (255, 215, 0),
        "iron": (200, 200, 200),
        "diamond": (80, 220, 230),
        "emerald": (0, 200, 100),
        "coal": (40, 40, 40),
        "redstone": (200, 0, 0),
        "lapis": (30, 80, 180),
        "water": (50, 100, 200),
        "lava": (255, 100, 0),
        "glass": (200, 220, 255),
        "wool": (240, 240, 240),
        "brick": (150, 90, 70),
        "cobble": (100, 100, 100),
        "obsidian": (20, 15, 30),
        "netherrack": (100, 40, 40),
        "soul": (70, 50, 40),
        "quartz": (235, 230, 220),
        "prismarine": (80, 150, 140),
        "purpur": (160, 120, 180),
        "concrete": (150, 150, 150),
        "terracotta": (160, 90, 70),
        "coral": (200, 100, 120),
        "ice": (180, 220, 255),
        "snow": (250, 250, 255),
        "leaf": (60, 140, 60),
        "leaves": (60, 140, 60),
        "air": (135, 206, 235),
    }

    for key, color in color_map.items():
        if key in name:
            return color

    # Default gray
    return (128, 128, 128)


def visualize_tsne_with_textures(
    embeddings: np.ndarray,
    tok2block: Dict[int, str],
    texture_dir: Optional[Path] = None,
    output_path: Path = Path("tsne_blocks.png"),
    max_blocks: int = 500,
    image_size: int = 16,
    figure_size: Tuple[int, int] = (20, 20),
    perplexity: int = 30,
    seed: int = 42,
):
    """
    Create t-SNE visualization with block textures.

    Args:
        embeddings: Block embeddings array [vocab_size, embedding_dim]
        tok2block: Token to block name mapping
        texture_dir: Directory containing Minecraft textures (optional)
        output_path: Where to save the visualization
        max_blocks: Maximum number of blocks to display (for readability)
        image_size: Size of texture images in pixels
        figure_size: Figure size in inches
        perplexity: t-SNE perplexity parameter
        seed: Random seed
    """
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Vocabulary size: {len(tok2block)}")

    # Sample blocks if too many
    vocab_size = len(tok2block)
    if vocab_size > max_blocks:
        print(f"Sampling {max_blocks} blocks from {vocab_size} total...")
        np.random.seed(seed)
        indices = np.random.choice(vocab_size, max_blocks, replace=False)
    else:
        indices = np.arange(vocab_size)

    sampled_embeddings = embeddings[indices]
    sampled_tokens = [list(tok2block.keys())[i] for i in indices]
    sampled_names = [tok2block[t] for t in sampled_tokens]

    # Run t-SNE
    print("Running t-SNE (this may take a minute)...")
    tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity)
    coords = tsne.fit_transform(sampled_embeddings)

    # Normalize coordinates to [0, 1]
    coords_min = coords.min(axis=0)
    coords_max = coords.max(axis=0)
    coords_normalized = (coords - coords_min) / (coords_max - coords_min)

    # Create figure
    fig, ax = plt.subplots(figsize=figure_size)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # Load textures and place images
    print("Placing block textures...")
    textures_found = 0
    textures_missing = 0

    for i, (x, y) in enumerate(coords_normalized):
        block_name = sampled_names[i]

        # Try to load texture
        texture = None
        if texture_dir and texture_dir.exists():
            texture_path = find_texture(block_name, texture_dir)
            if texture_path:
                texture = load_texture(texture_path, image_size)
                textures_found += 1

        # Use placeholder if no texture found
        if texture is None:
            color = get_block_color(block_name)
            texture = create_placeholder_texture(color, image_size)
            textures_missing += 1

        # Create image annotation
        imagebox = OffsetImage(texture, zoom=1.0)
        ab = AnnotationBbox(
            imagebox, (x, y),
            frameon=False,
            pad=0,
        )
        ax.add_artist(ab)

    print(f"Textures found: {textures_found}")
    print(f"Placeholders used: {textures_missing}")

    # Style the plot
    ax.set_title("Block2Vec Embeddings - t-SNE Visualization", fontsize=16, pad=20)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.set_facecolor("#f0f0f0")

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved visualization to {output_path}")
    plt.close(fig)  # Clean up memory


def main():
    parser = argparse.ArgumentParser(
        description="Visualize t-SNE embeddings with Minecraft block textures"
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("models/block2vec/block_embeddings.npy"),
        help="Path to embeddings .npy file",
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        default=Path("data/kaggle/output/tok2block.json"),
        help="Path to vocabulary JSON file",
    )
    parser.add_argument(
        "--textures",
        type=Path,
        default=None,
        help="Path to Minecraft textures directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/tsne_blocks.png"),
        help="Output path for visualization",
    )
    parser.add_argument(
        "--max-blocks",
        type=int,
        default=500,
        help="Maximum number of blocks to display",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=20,
        help="Size of block images in pixels",
    )
    parser.add_argument(
        "--perplexity",
        type=int,
        default=30,
        help="t-SNE perplexity parameter",
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading embeddings from {args.embeddings}...")
    embeddings = np.load(args.embeddings)

    print(f"Loading vocabulary from {args.vocab}...")
    with open(args.vocab) as f:
        tok2block = {int(k): v for k, v in json.load(f).items()}

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Generate visualization
    visualize_tsne_with_textures(
        embeddings=embeddings,
        tok2block=tok2block,
        texture_dir=args.textures,
        output_path=args.output,
        max_blocks=args.max_blocks,
        image_size=args.image_size,
        perplexity=args.perplexity,
    )


if __name__ == "__main__":
    main()
