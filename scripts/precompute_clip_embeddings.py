"""
Precompute CLIP text embeddings for all category labels.

Caches frozen CLIP text embeddings so they don't need to be recomputed
during discrete diffusion training.

Output: minecraft_ai/data/output/clip_embeddings/text_embeddings.pt

Usage:
    python minecraft_ai/scripts/precompute_clip_embeddings.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def precompute_clip_embeddings(
    output_path: Path,
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
) -> dict:
    """Compute and save CLIP text embeddings for all categories.

    Args:
        output_path: Where to save the embeddings .pt file.
        model_name: OpenCLIP model name.
        pretrained: Pretrained weights name.

    Returns:
        Dict with category names and embedding info.
    """
    from src.models.clip_scorer import (
        CLIPScorer,
        NEGATIVE_PROMPTS,
        STRUCTURE_PROMPTS,
    )

    scorer = CLIPScorer(model_name=model_name, pretrained=pretrained)

    all_categories = {**STRUCTURE_PROMPTS, **NEGATIVE_PROMPTS}
    embeddings = {}

    print(f"Computing CLIP text embeddings for {len(all_categories)} categories...")

    for category, prompts in all_categories.items():
        # Average embedding across all prompts for this category
        emb = scorer.get_text_embedding(category)
        embeddings[category] = torch.from_numpy(emb).float()
        print(f"  {category}: {len(prompts)} prompts -> [{emb.shape[0]}] embedding")

    # Also compute a null/unconditional embedding (empty string)
    null_emb = scorer.get_text_embedding("")
    embeddings["_null"] = torch.from_numpy(null_emb).float()
    print(f"  _null: unconditional embedding -> [{null_emb.shape[0]}]")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, output_path)
    print(f"\nSaved to {output_path}")

    # Also save metadata
    embed_dim = list(embeddings.values())[0].shape[0]
    print(f"\nEmbedding dim: {embed_dim}")
    print(f"Categories: {list(embeddings.keys())}")

    return {"categories": list(embeddings.keys()), "embed_dim": embed_dim}


def main():
    output_path = PROJECT_ROOT / "data" / "output" / "clip_embeddings" / "text_embeddings.pt"
    precompute_clip_embeddings(output_path)


if __name__ == "__main__":
    main()
