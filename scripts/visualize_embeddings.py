"""
Visualization script for Block2Vec embeddings.

Creates:
1. t-SNE/UMAP 2D projection colored by block category
2. Similarity heatmap for common blocks
3. Nearest neighbors table for key blocks

Usage:
    python scripts/visualize_embeddings.py
    python scripts/visualize_embeddings.py --embeddings checkpoints/block2vec/block_embeddings.npy
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def extract_block_category(block_name: str) -> str:
    """Extract category from block name for coloring.

    Examples:
        minecraft:oak_planks -> wood
        minecraft:stone_bricks -> stone
        minecraft:red_wool -> wool
    """
    # Remove minecraft: prefix and block state
    name = block_name.replace("minecraft:", "")
    name = re.sub(r"\[.*\]", "", name)

    # Category mapping
    categories = {
        "wood": ["planks", "log", "wood", "fence", "door", "trapdoor", "stairs", "slab"],
        "stone": ["stone", "cobble", "brick", "andesite", "diorite", "granite", "deepslate"],
        "ore": ["ore", "coal", "iron", "gold", "diamond", "emerald", "lapis", "redstone", "copper"],
        "glass": ["glass"],
        "wool": ["wool", "carpet"],
        "concrete": ["concrete"],
        "terracotta": ["terracotta"],
        "leaves": ["leaves"],
        "crops": ["wheat", "carrot", "potato", "beetroot", "melon", "pumpkin"],
        "light": ["torch", "lantern", "glowstone", "sea_lantern", "shroomlight"],
        "redstone": ["redstone", "piston", "dispenser", "dropper", "observer", "repeater", "comparator"],
        "water": ["water"],
        "lava": ["lava"],
        "air": ["air"],
    }

    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword in name:
                return category

    return "other"


def load_vocabulary(vocab_path: str) -> dict[int, str]:
    """Load token to block name mapping."""
    with open(vocab_path, "r") as f:
        tok2block = json.load(f)
    return {int(k): v for k, v in tok2block.items()}


def create_tsne_plot(
    embeddings: np.ndarray,
    tok2block: dict[int, str],
    output_path: str,
    perplexity: int = 30,
    n_iter: int = 1000,
    sample_size: int = 1000,
) -> None:
    """Create t-SNE visualization of embeddings.

    Args:
        embeddings: [vocab_size, embedding_dim] array
        tok2block: Token to block name mapping
        output_path: Path to save the figure
        perplexity: t-SNE perplexity parameter
        n_iter: Number of iterations
        sample_size: Number of blocks to sample (for speed)
    """
    print("Creating t-SNE visualization...")

    vocab_size = len(embeddings)

    # Sample if too many blocks
    if vocab_size > sample_size:
        indices = np.random.choice(vocab_size, sample_size, replace=False)
    else:
        indices = np.arange(vocab_size)

    sampled_embeddings = embeddings[indices]

    # Get categories for coloring
    categories = [extract_block_category(tok2block.get(i, "unknown")) for i in indices]
    unique_categories = list(set(categories))

    # Create color map
    cmap = plt.cm.get_cmap("tab20", len(unique_categories))
    cat_to_color = {cat: cmap(i) for i, cat in enumerate(unique_categories)}
    colors = [cat_to_color[cat] for cat in categories]

    # Run t-SNE
    print(f"  Running t-SNE on {len(indices)} blocks...")
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, len(indices) - 1),
        n_iter=n_iter,
        random_state=42,
        init="pca",
        learning_rate="auto",
    )
    coords = tsne.fit_transform(sampled_embeddings)

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot each category separately for legend
    for cat in unique_categories:
        mask = [c == cat for c in categories]
        cat_coords = coords[mask]
        if len(cat_coords) > 0:
            ax.scatter(
                cat_coords[:, 0],
                cat_coords[:, 1],
                c=[cat_to_color[cat]],
                label=cat,
                alpha=0.6,
                s=20,
            )

    ax.set_title("Block2Vec Embeddings (t-SNE projection)", fontsize=14)
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved t-SNE plot: {output_path}")


def create_similarity_heatmap(
    embeddings: np.ndarray,
    tok2block: dict[int, str],
    output_path: str,
    blocks_to_show: list[str] = None,
) -> None:
    """Create similarity heatmap for selected blocks.

    Args:
        embeddings: [vocab_size, embedding_dim] array
        tok2block: Token to block name mapping
        output_path: Path to save the figure
        blocks_to_show: List of block name patterns to include
    """
    print("Creating similarity heatmap...")

    if blocks_to_show is None:
        # Default: common building blocks
        blocks_to_show = [
            "stone", "cobblestone", "stone_bricks",
            "oak_planks", "spruce_planks", "birch_planks",
            "oak_log", "spruce_log", "birch_log",
            "glass", "white_wool", "red_wool",
            "iron_block", "gold_block", "diamond_block",
            "dirt", "grass_block", "sand",
            "water", "lava", "air",
        ]

    # Find token IDs for these blocks (use base block without state)
    block2tok = {}
    for tok, block_name in tok2block.items():
        # Get base name without state
        base = re.sub(r"\[.*\]", "", block_name.replace("minecraft:", ""))
        if base not in block2tok:
            block2tok[base] = tok

    # Get indices for blocks to show
    selected_tokens = []
    selected_names = []
    for block in blocks_to_show:
        if block in block2tok:
            selected_tokens.append(block2tok[block])
            selected_names.append(block)

    if len(selected_tokens) < 2:
        print("  Not enough blocks found for heatmap")
        return

    # Get embeddings for selected blocks
    selected_embeddings = embeddings[selected_tokens]

    # Compute cosine similarity
    sim_matrix = cosine_similarity(selected_embeddings)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(sim_matrix, cmap="RdYlBu_r", vmin=-1, vmax=1)

    # Add labels
    ax.set_xticks(range(len(selected_names)))
    ax.set_yticks(range(len(selected_names)))
    ax.set_xticklabels(selected_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(selected_names, fontsize=9)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Cosine Similarity", fontsize=10)

    # Add values in cells
    for i in range(len(selected_names)):
        for j in range(len(selected_names)):
            text = ax.text(
                j, i, f"{sim_matrix[i, j]:.2f}",
                ha="center", va="center",
                color="white" if abs(sim_matrix[i, j]) > 0.5 else "black",
                fontsize=7,
            )

    ax.set_title("Block Embedding Similarity (Cosine)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved similarity heatmap: {output_path}")


def create_nearest_neighbors_table(
    embeddings: np.ndarray,
    tok2block: dict[int, str],
    output_path: str,
    query_blocks: list[str] = None,
    top_k: int = 8,
) -> None:
    """Create nearest neighbors table for key blocks.

    Args:
        embeddings: [vocab_size, embedding_dim] array
        tok2block: Token to block name mapping
        output_path: Path to save the figure
        query_blocks: List of block name patterns to query
        top_k: Number of neighbors to show
    """
    print("Creating nearest neighbors table...")

    if query_blocks is None:
        query_blocks = [
            "stone", "oak_planks", "glass", "iron_block",
            "water", "torch", "air", "diamond_ore",
        ]

    # Build reverse mapping (base name -> token)
    block2tok = {}
    for tok, block_name in tok2block.items():
        base = re.sub(r"\[.*\]", "", block_name.replace("minecraft:", ""))
        if base not in block2tok:
            block2tok[base] = tok

    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normalized = embeddings / norms

    # Find neighbors for each query block
    results = {}
    for block in query_blocks:
        if block not in block2tok:
            continue

        token = block2tok[block]
        query = normalized[token]

        # Compute similarities
        similarities = np.dot(normalized, query)

        # Get top-k (excluding self)
        top_indices = np.argsort(similarities)[::-1]
        neighbors = []
        for idx in top_indices:
            if idx != token:
                neighbor_name = tok2block.get(idx, f"token_{idx}")
                neighbor_base = re.sub(r"\[.*\]", "", neighbor_name.replace("minecraft:", ""))
                neighbors.append((neighbor_base, similarities[idx]))
            if len(neighbors) >= top_k:
                break

        results[block] = neighbors

    # Create figure with table
    fig, ax = plt.subplots(figsize=(14, len(results) * 0.6 + 2))
    ax.axis("off")

    # Build table data
    headers = ["Query Block"] + [f"Neighbor {i+1}" for i in range(top_k)]
    table_data = []

    for block, neighbors in results.items():
        row = [block]
        for name, sim in neighbors:
            row.append(f"{name}\n({sim:.3f})")
        # Pad if needed
        while len(row) < len(headers):
            row.append("")
        table_data.append(row)

    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )

    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)

    # Color header row
    for i, key in enumerate(table.get_celld().keys()):
        cell = table.get_celld()[key]
        if key[0] == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", weight="bold")
        elif key[1] == 0:
            cell.set_facecolor("#D6DCE4")
            cell.set_text_props(weight="bold")

    ax.set_title("Nearest Neighbors by Embedding Similarity", fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved nearest neighbors table: {output_path}")


def create_loss_plot(stats_path: str, output_path: str) -> None:
    """Create training loss plot.

    Args:
        stats_path: Path to training_stats.json
        output_path: Path to save the figure
    """
    print("Creating loss plot...")

    with open(stats_path, "r") as f:
        stats = json.load(f)

    epoch_losses = stats.get("epoch_losses", [])
    if not epoch_losses:
        print("  No epoch losses found")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(epoch_losses) + 1)
    ax.plot(epochs, epoch_losses, "b-", linewidth=2, marker="o", markersize=4)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Block2Vec Training Loss", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved loss plot: {output_path}")


def visualize_embeddings(
    embeddings_path: str,
    vocab_path: str,
    output_dir: str,
    stats_path: str = None,
) -> None:
    """Run all visualizations.

    Args:
        embeddings_path: Path to block_embeddings.npy
        vocab_path: Path to tok2block.json
        output_dir: Directory to save visualizations
        stats_path: Optional path to training_stats.json
    """
    print("=" * 60)
    print("Block2Vec Embedding Visualization")
    print("=" * 60)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load embeddings
    print(f"\nLoading embeddings from {embeddings_path}...")
    embeddings = np.load(embeddings_path)
    print(f"  Shape: {embeddings.shape}")

    # Load vocabulary
    tok2block = load_vocabulary(vocab_path)
    print(f"  Vocabulary size: {len(tok2block)}")

    # Create visualizations
    print("\nGenerating visualizations...")

    create_tsne_plot(
        embeddings, tok2block,
        str(output_path / "tsne_embeddings.png"),
    )

    create_similarity_heatmap(
        embeddings, tok2block,
        str(output_path / "similarity_heatmap.png"),
    )

    create_nearest_neighbors_table(
        embeddings, tok2block,
        str(output_path / "nearest_neighbors.png"),
    )

    # Training loss if available
    if stats_path and Path(stats_path).exists():
        create_loss_plot(stats_path, str(output_path / "training_loss.png"))

    print("\n" + "=" * 60)
    print(f"Visualizations saved to {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Visualize Block2Vec embeddings")

    parser.add_argument(
        "--embeddings",
        type=str,
        default="checkpoints/block2vec/block_embeddings.npy",
        help="Path to embeddings file",
    )
    parser.add_argument(
        "--vocab",
        type=str,
        default="data/vocabulary/tok2block.json",
        help="Path to vocabulary file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/block2vec_visualizations",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--stats",
        type=str,
        default="checkpoints/block2vec/training_stats.json",
        help="Path to training stats file",
    )

    args = parser.parse_args()

    # Resolve paths relative to project root
    embeddings_path = PROJECT_ROOT / args.embeddings
    vocab_path = PROJECT_ROOT / args.vocab
    output_dir = PROJECT_ROOT / args.output_dir
    stats_path = PROJECT_ROOT / args.stats

    visualize_embeddings(
        embeddings_path=str(embeddings_path),
        vocab_path=str(vocab_path),
        output_dir=str(output_dir),
        stats_path=str(stats_path),
    )


if __name__ == "__main__":
    main()
