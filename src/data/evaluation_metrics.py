"""
Evaluation metrics for Sparse Structure Transformer.

New metrics that matter for structure reconstruction:
- Per-category recall (stairs, doors, slabs, etc.)
- Embedding cosine similarity
- Edit distance (block-level changes)
- Category confusion matrix

These metrics directly address the problems with dense VQ-VAE:
- VQ-VAE: 0% accuracy on stairs/doors/slabs
- Expected: >50% accuracy on these categories
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# Air token IDs
AIR_TOKENS: Set[int] = {102, 576, 3352}


def get_block_category(block_name: str) -> str:
    """Categorize a block by its shape/type.

    Args:
        block_name: Full block name like "minecraft:oak_stairs[facing=north]"

    Returns:
        Category string like "stairs", "slabs", "wood", etc.
    """
    name = block_name.replace("minecraft:", "").split("[")[0].lower()

    # Shape categories (most specific first)
    if "stair" in name:
        return "stairs"
    if "slab" in name:
        return "slabs"
    if "_wall" in name or name.endswith("_wall"):
        return "walls"
    if "fence_gate" in name:
        return "fence_gates"
    if "fence" in name:
        return "fences"
    if "door" in name:
        return "doors"
    if "trapdoor" in name:
        return "trapdoors"
    if "button" in name:
        return "buttons"
    if "pressure_plate" in name:
        return "pressure_plates"
    if "sign" in name:
        return "signs"

    # Material categories
    wood_types = ["oak", "spruce", "birch", "jungle", "acacia", "dark_oak", "mangrove", "cherry", "bamboo"]
    if any(w in name for w in wood_types):
        if "planks" in name:
            return "planks"
        if "log" in name or "wood" in name:
            return "logs"
        return "wood"

    if any(w in name for w in ["stone", "cobble", "granite", "diorite", "andesite", "deepslate", "brick"]):
        return "stone"
    if "glass" in name:
        return "glass"
    if "wool" in name:
        return "wool"
    if "concrete" in name:
        return "concrete"
    if "terracotta" in name:
        return "terracotta"
    if any(w in name for w in ["torch", "lantern", "lamp", "light"]):
        return "lighting"
    if any(w in name for w in ["redstone", "piston", "observer", "comparator", "repeater", "lever"]):
        return "redstone"
    if any(w in name for w in ["flower", "grass", "fern", "leaves", "sapling", "vine", "moss"]):
        return "plants"
    if any(w in name for w in ["iron", "gold", "copper", "diamond", "emerald", "netherite"]):
        return "metal"
    if "ore" in name:
        return "ores"

    return "other"


class CategoryMetrics:
    """Track per-category reconstruction accuracy.

    This directly measures what VQ-VAE fails at:
    - Are stairs reconstructed correctly?
    - Are doors preserved?
    - Are slabs accurate?
    """

    def __init__(self, tok2block: Dict[int, str]):
        """
        Args:
            tok2block: Mapping from token IDs to block names
        """
        self.tok2block = tok2block
        self.reset()

    def reset(self):
        """Reset all counters."""
        self.category_correct: Dict[str, int] = defaultdict(int)
        self.category_total: Dict[str, int] = defaultdict(int)
        self.category_predictions: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def update(
        self,
        pred_ids: torch.Tensor,
        true_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """Update metrics with a batch of predictions.

        Args:
            pred_ids: [B, N] or [N] predicted block IDs
            true_ids: [B, N] or [N] ground truth block IDs
            mask: [B, N] or [N] boolean mask (True = valid)
        """
        pred_ids = pred_ids.view(-1).cpu().numpy()
        true_ids = true_ids.view(-1).cpu().numpy()

        if mask is not None:
            mask = mask.view(-1).cpu().numpy()
            valid_indices = np.where(mask)[0]
        else:
            valid_indices = np.arange(len(pred_ids))

        for idx in valid_indices:
            true_id = int(true_ids[idx])
            pred_id = int(pred_ids[idx])

            if true_id not in self.tok2block:
                continue

            true_name = self.tok2block[true_id]
            true_cat = get_block_category(true_name)

            self.category_total[true_cat] += 1

            if pred_id == true_id:
                self.category_correct[true_cat] += 1

            # Track what category we predicted
            if pred_id in self.tok2block:
                pred_name = self.tok2block[pred_id]
                pred_cat = get_block_category(pred_name)
                self.category_predictions[true_cat][pred_cat] += 1

    def compute(self) -> Dict[str, float]:
        """Compute per-category accuracy.

        Returns:
            Dictionary mapping category -> accuracy
        """
        results = {}
        for cat in self.category_total:
            if self.category_total[cat] > 0:
                results[cat] = self.category_correct[cat] / self.category_total[cat]
        return results

    def get_confusion_matrix(self) -> Dict[str, Dict[str, float]]:
        """Get category confusion matrix (normalized).

        Returns:
            Nested dict: true_category -> predicted_category -> proportion
        """
        confusion = {}
        for true_cat in self.category_predictions:
            total = sum(self.category_predictions[true_cat].values())
            if total > 0:
                confusion[true_cat] = {
                    pred_cat: count / total
                    for pred_cat, count in self.category_predictions[true_cat].items()
                }
        return confusion

    def get_critical_metrics(self) -> Dict[str, float]:
        """Get metrics for categories that VQ-VAE fails on.

        These are the key indicators of whether the sparse approach works.
        """
        critical_categories = ["stairs", "slabs", "doors", "fences", "walls"]
        metrics = {}

        for cat in critical_categories:
            if cat in self.category_total and self.category_total[cat] > 0:
                metrics[f"{cat}_accuracy"] = self.category_correct[cat] / self.category_total[cat]
                metrics[f"{cat}_total"] = self.category_total[cat]
            else:
                metrics[f"{cat}_accuracy"] = 0.0
                metrics[f"{cat}_total"] = 0

        return metrics


class EmbeddingSimilarity:
    """Track embedding-level reconstruction quality.

    Unlike block-level accuracy, this measures how "close" we are
    even when the exact block doesn't match. This leverages Block2Vec
    semantic structure.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset accumulators."""
        self.total_similarity = 0.0
        self.total_count = 0
        self.similarities_by_distance: Dict[str, List[float]] = defaultdict(list)

    def update(
        self,
        pred_embeddings: torch.Tensor,
        true_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """Update with batch of embeddings.

        Args:
            pred_embeddings: [B, N, D] predicted embeddings
            true_embeddings: [B, N, D] ground truth embeddings
            mask: [B, N] boolean mask
        """
        # Normalize for cosine similarity
        pred_norm = F.normalize(pred_embeddings, dim=-1)
        true_norm = F.normalize(true_embeddings, dim=-1)

        # Cosine similarity per position
        cos_sim = (pred_norm * true_norm).sum(dim=-1)  # [B, N]

        if mask is not None:
            valid_sims = cos_sim[mask]
        else:
            valid_sims = cos_sim.view(-1)

        self.total_similarity += valid_sims.sum().item()
        self.total_count += valid_sims.numel()

    def compute(self) -> Dict[str, float]:
        """Compute average cosine similarity.

        Returns:
            Dictionary with similarity metrics
        """
        avg_sim = self.total_similarity / self.total_count if self.total_count > 0 else 0

        return {
            "mean_cosine_similarity": avg_sim,
            "total_predictions": self.total_count,
        }


class EditDistance:
    """Compute edit distance metrics between structures.

    Measures how many blocks would need to be changed to transform
    the prediction into the ground truth.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset accumulators."""
        self.total_edits = 0
        self.total_blocks = 0
        self.perfect_matches = 0
        self.total_structures = 0

    def update(
        self,
        pred_ids: torch.Tensor,
        true_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """Update with batch of structure predictions.

        Args:
            pred_ids: [B, N] predicted block IDs
            true_ids: [B, N] ground truth block IDs
            mask: [B, N] boolean mask
        """
        B = pred_ids.size(0)

        for b in range(B):
            if mask is not None:
                valid = mask[b]
                pred = pred_ids[b][valid]
                true = true_ids[b][valid]
            else:
                pred = pred_ids[b]
                true = true_ids[b]

            n_blocks = len(pred)
            n_edits = (pred != true).sum().item()

            self.total_edits += n_edits
            self.total_blocks += n_blocks
            self.total_structures += 1

            if n_edits == 0:
                self.perfect_matches += 1

    def compute(self) -> Dict[str, float]:
        """Compute edit distance metrics.

        Returns:
            Dictionary with edit distance statistics
        """
        avg_edit_rate = self.total_edits / self.total_blocks if self.total_blocks > 0 else 0
        perfect_rate = self.perfect_matches / self.total_structures if self.total_structures > 0 else 0

        return {
            "edit_rate": avg_edit_rate,
            "perfect_match_rate": perfect_rate,
            "total_edits": self.total_edits,
            "total_blocks": self.total_blocks,
        }


class ComprehensiveMetrics:
    """Aggregates all evaluation metrics.

    Provides a single interface for tracking:
    - Per-category accuracy
    - Embedding similarity
    - Edit distance
    - Overall statistics
    """

    def __init__(self, tok2block: Dict[int, str]):
        """
        Args:
            tok2block: Mapping from token IDs to block names
        """
        self.category_metrics = CategoryMetrics(tok2block)
        self.embedding_metrics = EmbeddingSimilarity()
        self.edit_metrics = EditDistance()

        self.total_correct = 0
        self.total_blocks = 0

    def reset(self):
        """Reset all metrics."""
        self.category_metrics.reset()
        self.embedding_metrics.reset()
        self.edit_metrics.reset()
        self.total_correct = 0
        self.total_blocks = 0

    def update(
        self,
        pred_ids: torch.Tensor,
        true_ids: torch.Tensor,
        pred_embeddings: torch.Tensor,
        true_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """Update all metrics with batch.

        Args:
            pred_ids: [B, N] predicted block IDs
            true_ids: [B, N] ground truth block IDs
            pred_embeddings: [B, N, D] predicted embeddings
            true_embeddings: [B, N, D] ground truth embeddings
            mask: [B, N] boolean mask
        """
        # Overall accuracy
        if mask is not None:
            correct = ((pred_ids == true_ids) & mask).sum().item()
            total = mask.sum().item()
        else:
            correct = (pred_ids == true_ids).sum().item()
            total = pred_ids.numel()

        self.total_correct += correct
        self.total_blocks += total

        # Update component metrics
        self.category_metrics.update(pred_ids, true_ids, mask)
        self.embedding_metrics.update(pred_embeddings, true_embeddings, mask)
        self.edit_metrics.update(pred_ids, true_ids, mask)

    def compute(self) -> Dict[str, any]:
        """Compute all metrics.

        Returns:
            Comprehensive dictionary of all metrics
        """
        overall_accuracy = self.total_correct / self.total_blocks if self.total_blocks > 0 else 0

        return {
            "overall_accuracy": overall_accuracy,
            "total_blocks": self.total_blocks,
            "category_accuracy": self.category_metrics.compute(),
            "critical_metrics": self.category_metrics.get_critical_metrics(),
            "embedding_similarity": self.embedding_metrics.compute(),
            "edit_distance": self.edit_metrics.compute(),
        }

    def summary(self) -> str:
        """Get human-readable summary.

        Returns:
            Formatted string with key metrics
        """
        metrics = self.compute()

        lines = [
            "=" * 60,
            "EVALUATION SUMMARY",
            "=" * 60,
            f"Overall Accuracy: {metrics['overall_accuracy']:.2%}",
            "",
            "Critical Categories (VQ-VAE fails on these):",
        ]

        critical = metrics["critical_metrics"]
        for key in ["stairs", "slabs", "doors", "fences", "walls"]:
            acc = critical.get(f"{key}_accuracy", 0)
            total = critical.get(f"{key}_total", 0)
            lines.append(f"  {key:12s}: {acc:.2%} ({total} samples)")

        lines.extend([
            "",
            f"Embedding Similarity: {metrics['embedding_similarity']['mean_cosine_similarity']:.4f}",
            f"Edit Rate: {metrics['edit_distance']['edit_rate']:.2%}",
            f"Perfect Match Rate: {metrics['edit_distance']['perfect_match_rate']:.2%}",
            "=" * 60,
        ])

        return "\n".join(lines)

