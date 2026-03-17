"""
Dataset for CLIP-conditioned discrete diffusion training.

Yields (token_indices, clip_embedding, has_condition) tuples where:
- token_indices: [1024] flattened RFSQ indices (512 stage1 + 512 stage2)
- clip_embedding: [512] CLIP text embedding for the structure's category
- has_condition: bool — False when CFG dropout is applied

Loads precomputed RFSQ indices and CLIP text embeddings.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


class ConditionedLatentDataset(Dataset):
    """Dataset for CLIP-conditioned discrete diffusion training.

    Args:
        latent_dir: Directory containing precomputed .pt latent files.
        clip_labels_path: Path to clip_scores.json (maps filename -> category).
        clip_embeddings_path: Path to text_embeddings.pt (category -> embedding).
        split_filenames: Optional list of filenames to include (from filtered_splits.json).
        cfg_dropout: Probability of dropping CLIP conditioning (for CFG training).
        seed: Random seed.
    """

    def __init__(
        self,
        latent_dir: str,
        clip_labels_path: str,
        clip_embeddings_path: str,
        split_filenames: Optional[List[str]] = None,
        cfg_dropout: float = 0.1,
        seed: int = 42,
    ):
        self.latent_dir = Path(latent_dir)
        self.cfg_dropout = cfg_dropout
        self.rng = random.Random(seed)

        # Load CLIP labels (filename -> {best_category, ...})
        with open(clip_labels_path, "r") as f:
            self.clip_labels = json.load(f)

        # Load precomputed CLIP text embeddings (category -> [512] tensor)
        self.clip_embeddings = torch.load(clip_embeddings_path, weights_only=False)
        self.null_embedding = self.clip_embeddings.get(
            "_null",
            torch.zeros_like(next(iter(self.clip_embeddings.values()))),
        )

        # Find available latent files
        if split_filenames is not None:
            # Only use files from the specified split
            self.filenames = []
            for fn in split_filenames:
                stem = Path(fn).stem
                pt_path = self.latent_dir / f"{stem}.pt"
                if pt_path.exists():
                    self.filenames.append(fn)
        else:
            # Use all available .pt files
            pt_files = sorted(self.latent_dir.glob("*.pt"))
            self.filenames = [f"{f.stem}.h5" for f in pt_files]

        print(f"ConditionedLatentDataset: {len(self.filenames)} samples "
              f"(CFG dropout={cfg_dropout})")

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a single sample.

        Returns:
            Dict with:
                "indices": [1024] int64 — flattened RFSQ indices
                "clip_embedding": [embed_dim] float32 — CLIP text embedding
                "has_condition": bool tensor — False for CFG dropout
        """
        filename = self.filenames[idx]
        stem = Path(filename).stem

        # Load precomputed RFSQ indices
        pt_path = self.latent_dir / f"{stem}.pt"
        latent = torch.load(pt_path, weights_only=False)

        stage1 = latent["stage1"].flatten()  # [512]
        stage2 = latent["stage2"].flatten()  # [512]
        indices = torch.cat([stage1, stage2], dim=0).long()  # [1024]

        # Get category and CLIP embedding
        label_info = self.clip_labels.get(filename, {})
        category = label_info.get("best_category", "generic_structure")

        # Use the precomputed category embedding
        if category in self.clip_embeddings:
            clip_emb = self.clip_embeddings[category].clone()
        else:
            clip_emb = self.clip_embeddings.get(
                "generic_structure", self.null_embedding
            ).clone()

        # CFG dropout: randomly drop conditioning
        has_condition = True
        if self.rng.random() < self.cfg_dropout:
            clip_emb = self.null_embedding.clone()
            has_condition = False

        return {
            "indices": indices,
            "clip_embedding": clip_emb,
            "has_condition": torch.tensor(has_condition, dtype=torch.bool),
        }


def collate_conditioned(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for ConditionedLatentDataset.

    Args:
        batch: List of sample dicts.

    Returns:
        Batched dict with stacked tensors.
    """
    return {
        "indices": torch.stack([b["indices"] for b in batch]),
        "clip_embedding": torch.stack([b["clip_embedding"] for b in batch]),
        "has_condition": torch.stack([b["has_condition"] for b in batch]),
    }
