"""
CLIP Filtering & Auto-Labeling Pipeline.

Combines three signals to filter structures and assign category labels:
1. CLIP scores: Zero-shot classification from composite renders
2. Terrain ratio: Fraction of non-air voxels that are terrain blocks
3. Structural heuristics: Air fraction, Y-distribution, block diversity

Uses 700 manually labeled samples from curation_results.json for calibration.

Output:
- minecraft_ai/data/output/clip_labels/clip_scores.json
- minecraft_ai/data/output/clip_labels/filtered_splits.json

Usage:
    python minecraft_ai/scripts/clip_filter_and_label.py
    python minecraft_ai/scripts/clip_filter_and_label.py --calibrate-only
    python minecraft_ai/scripts/clip_filter_and_label.py --threshold 0.5
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.terrain_detection import AIR_TOKENS, build_terrain_token_set


def load_tok2block(vocab_path: Path) -> Dict[int, str]:
    """Load token-to-block mapping."""
    with open(vocab_path, "r") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def compute_structural_heuristics(block_ids: np.ndarray) -> Dict[str, float]:
    """Compute structural quality heuristics from raw voxels.

    Args:
        block_ids: [32, 32, 32] voxel grid.

    Returns:
        Dict with heuristic scores.
    """
    air_set = set(AIR_TOKENS)

    # Air fraction
    total_voxels = block_ids.size
    air_mask = np.isin(block_ids, list(air_set))
    air_count = air_mask.sum()
    air_fraction = air_count / total_voxels

    # Non-air voxels
    non_air_mask = ~air_mask
    non_air_count = non_air_mask.sum()

    if non_air_count == 0:
        return {
            "air_fraction": 1.0,
            "block_diversity": 0,
            "y_spread": 0.0,
            "y_concentration": 1.0,
        }

    # Block diversity (unique non-air block types)
    non_air_blocks = block_ids[non_air_mask]
    unique_blocks = len(np.unique(non_air_blocks))

    # Y-distribution: how spread out are blocks vertically?
    # shape is [X, Y, Z] — Y is axis 1
    y_positions = np.where(non_air_mask)[1]  # Y coords of non-air blocks
    y_min, y_max = y_positions.min(), y_positions.max()
    y_spread = (y_max - y_min + 1) / block_ids.shape[1]

    # Y-concentration: fraction of Y levels that contain blocks
    y_levels_used = len(np.unique(y_positions))
    y_concentration = y_levels_used / block_ids.shape[1]

    return {
        "air_fraction": float(air_fraction),
        "block_diversity": int(unique_blocks),
        "y_spread": float(y_spread),
        "y_concentration": float(y_concentration),
    }


def compute_terrain_ratio(
    block_ids: np.ndarray,
    terrain_tokens: set,
) -> float:
    """Compute fraction of non-air voxels that are terrain blocks.

    Args:
        block_ids: [32, 32, 32] voxel grid.
        terrain_tokens: Set of terrain token IDs.

    Returns:
        Terrain ratio (0.0 to 1.0).
    """
    air_set = set(AIR_TOKENS)
    non_air_mask = ~np.isin(block_ids, list(air_set))
    non_air_count = non_air_mask.sum()

    if non_air_count == 0:
        return 0.0

    terrain_mask = np.isin(block_ids, list(terrain_tokens)) & non_air_mask
    return float(terrain_mask.sum() / non_air_count)


def compute_quality_score(
    heuristics: Dict[str, float],
    clip_scores: Optional[Dict[str, float]],
    terrain_ratio: float,
) -> Tuple[float, str, Dict]:
    """Compute combined quality score from all signals.

    Args:
        heuristics: From compute_structural_heuristics().
        clip_scores: From CLIPScorer.classify_structure() (None if no CLIP).
        terrain_ratio: From compute_terrain_ratio().

    Returns:
        (quality_score 0-1, best_category, metadata)
    """
    score = 0.0
    reasons = []

    # --- Heuristic penalties ---

    # Too much air = likely empty/sparse
    if heuristics["air_fraction"] > 0.98:
        score -= 0.5
        reasons.append("nearly_empty")
    elif heuristics["air_fraction"] > 0.95:
        score -= 0.2
        reasons.append("very_sparse")

    # Too little air = likely terrain chunk
    if heuristics["air_fraction"] < 0.5:
        score -= 0.1
        reasons.append("very_dense")

    # Low diversity = likely terrain
    if heuristics["block_diversity"] < 5:
        score -= 0.3
        reasons.append("low_diversity")
    elif heuristics["block_diversity"] > 15:
        score += 0.1

    # Y-spread: structures typically span multiple Y levels
    if heuristics["y_spread"] > 0.3:
        score += 0.1
    elif heuristics["y_spread"] < 0.1:
        score -= 0.1
        reasons.append("flat")

    # --- Terrain ratio ---
    if terrain_ratio > 0.8:
        score -= 0.4
        reasons.append("mostly_terrain")
    elif terrain_ratio > 0.5:
        score -= 0.2
        reasons.append("high_terrain")
    elif terrain_ratio < 0.2:
        score += 0.1

    # --- CLIP scores ---
    best_category = "unknown"
    best_structure_score = 0.0

    if clip_scores is not None:
        from src.models.clip_scorer import NEGATIVE_PROMPTS, STRUCTURE_PROMPTS

        structure_cats = list(STRUCTURE_PROMPTS.keys())
        negative_cats = list(NEGATIVE_PROMPTS.keys())

        # Best structure category score
        for cat in structure_cats:
            if cat in clip_scores and clip_scores[cat] > best_structure_score:
                best_structure_score = clip_scores[cat]
                best_category = cat

        # Best negative category score
        best_negative_score = max(
            clip_scores.get(cat, 0.0) for cat in negative_cats
        )

        # CLIP signal: structure vs negative margin
        clip_margin = best_structure_score - best_negative_score
        score += clip_margin * 2.0  # Scale up CLIP signal

        if clip_margin > 0.05:
            score += 0.2
        elif clip_margin < -0.05:
            score -= 0.2
            reasons.append("clip_negative")
    else:
        # Without CLIP, use heuristics-only baseline
        score += 0.3  # Start positive, penalize down

    # Normalize to 0-1
    quality_score = max(0.0, min(1.0, score + 0.5))

    metadata = {
        "quality_score": quality_score,
        "best_category": best_category,
        "terrain_ratio": terrain_ratio,
        "heuristics": heuristics,
        "reasons": reasons,
    }
    if clip_scores is not None:
        metadata["clip_scores"] = clip_scores

    return quality_score, best_category, metadata


def calibrate_thresholds(
    curation_path: Path,
    h5_dirs: List[Path],
    tok2block: Dict[int, str],
    terrain_tokens: set,
    clip_scorer=None,
    renders_dir: Optional[Path] = None,
) -> Dict[str, float]:
    """Calibrate quality thresholds using labeled samples.

    Args:
        curation_path: Path to curation_results.json.
        h5_dirs: Directories to search for H5 files.
        tok2block: Token to block name mapping.
        terrain_tokens: Set of terrain token IDs.
        clip_scorer: Optional CLIPScorer instance.
        renders_dir: Directory with pre-rendered composites.

    Returns:
        Dict with optimal thresholds and calibration metrics.
    """
    with open(curation_path, "r") as f:
        curation = json.load(f)

    # Build filename -> label mapping
    labels = {}
    for entry in curation.get("accepted", []):
        fname = entry if isinstance(entry, str) else entry.get("filename", "")
        if fname:
            labels[fname] = True
    for entry in curation.get("rejected", []):
        fname = entry if isinstance(entry, str) else entry.get("filename", "")
        if fname:
            labels[fname] = False

    print(f"Calibration set: {sum(labels.values())} accepted, "
          f"{sum(not v for v in labels.values())} rejected")

    # Build H5 file index
    h5_index = {}
    for d in h5_dirs:
        if d.exists():
            for f in d.glob("*.h5"):
                h5_index[f.name] = f

    # Score all labeled samples
    scores_positive = []
    scores_negative = []

    for fname, is_structure in tqdm(labels.items(), desc="Calibrating"):
        if fname not in h5_index:
            continue

        h5_path = h5_index[fname]
        try:
            with h5py.File(h5_path, "r") as f:
                key = list(f.keys())[0]
                block_ids = f[key][:].astype(np.int64)
        except Exception:
            continue

        heuristics = compute_structural_heuristics(block_ids)
        t_ratio = compute_terrain_ratio(block_ids, terrain_tokens)

        # CLIP scores if available
        clip_scores = None
        if clip_scorer is not None and renders_dir is not None:
            render_path = renders_dir / f"{Path(fname).stem}.png"
            if render_path.exists():
                try:
                    from PIL import Image

                    img = np.array(Image.open(render_path).convert("RGB"))
                    clip_scores = clip_scorer.classify_structure(img)
                except Exception:
                    pass

        quality, _, _ = compute_quality_score(heuristics, clip_scores, t_ratio)

        if is_structure:
            scores_positive.append(quality)
        else:
            scores_negative.append(quality)

    if not scores_positive or not scores_negative:
        print("WARNING: Not enough labeled samples found in H5 directories")
        return {"accept_threshold": 0.7, "reject_threshold": 0.4}

    scores_positive = np.array(scores_positive)
    scores_negative = np.array(scores_negative)

    # Find threshold that maximizes F1
    best_f1 = 0.0
    best_threshold = 0.5
    for threshold in np.arange(0.1, 0.9, 0.05):
        tp = (scores_positive >= threshold).sum()
        fp = (scores_negative >= threshold).sum()
        fn = (scores_positive < threshold).sum()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    # Also find a lower threshold for the review band
    review_threshold = max(0.2, best_threshold - 0.3)

    result = {
        "accept_threshold": float(best_threshold),
        "reject_threshold": float(review_threshold),
        "best_f1": float(best_f1),
        "positive_mean": float(scores_positive.mean()),
        "positive_std": float(scores_positive.std()),
        "negative_mean": float(scores_negative.mean()),
        "negative_std": float(scores_negative.std()),
        "n_positive": len(scores_positive),
        "n_negative": len(scores_negative),
    }

    print(f"\nCalibration results:")
    print(f"  Accept threshold: {result['accept_threshold']:.2f}")
    print(f"  Reject threshold: {result['reject_threshold']:.2f}")
    print(f"  Best F1: {result['best_f1']:.3f}")
    print(f"  Positive scores: {result['positive_mean']:.3f} ± {result['positive_std']:.3f}")
    print(f"  Negative scores: {result['negative_mean']:.3f} ± {result['negative_std']:.3f}")

    return result


def run_filtering(
    h5_dirs: List[Path],
    output_dir: Path,
    tok2block: Dict[int, str],
    terrain_tokens: set,
    accept_threshold: float = 0.7,
    reject_threshold: float = 0.4,
    clip_scorer=None,
    renders_dir: Optional[Path] = None,
    tag_index: Optional[Dict] = None,
) -> Dict:
    """Run the full filtering pipeline on all H5 files.

    Args:
        h5_dirs: Directories containing H5 files.
        output_dir: Where to write output JSON files.
        tok2block: Token to block mapping.
        terrain_tokens: Set of terrain token IDs.
        accept_threshold: Auto-accept threshold.
        reject_threshold: Auto-reject threshold (below this).
        clip_scorer: Optional CLIPScorer instance.
        renders_dir: Directory with pre-rendered composites.
        tag_index: Optional {h5_filename: {tags, page_url}} mapping.

    Returns:
        Summary stats dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all H5 files
    h5_files = []
    for d in h5_dirs:
        if d.exists():
            h5_files.extend(sorted(d.glob("*.h5")))

    print(f"\nFiltering {len(h5_files)} H5 files...")
    print(f"  Accept threshold: {accept_threshold}")
    print(f"  Reject threshold: {reject_threshold}")

    all_scores = {}
    accepted = []
    review = []
    rejected = []

    for h5_path in tqdm(h5_files, desc="Filtering"):
        try:
            with h5py.File(h5_path, "r") as f:
                key = list(f.keys())[0]
                block_ids = f[key][:].astype(np.int64)
        except Exception:
            rejected.append(h5_path.name)
            continue

        if block_ids.shape != (32, 32, 32):
            rejected.append(h5_path.name)
            continue

        heuristics = compute_structural_heuristics(block_ids)
        t_ratio = compute_terrain_ratio(block_ids, terrain_tokens)

        # CLIP scores if available
        clip_scores = None
        if clip_scorer is not None and renders_dir is not None:
            render_path = renders_dir / f"{h5_path.stem}.png"
            if render_path.exists():
                try:
                    from PIL import Image

                    img = np.array(Image.open(render_path).convert("RGB"))
                    clip_scores = clip_scorer.classify_structure(img)
                except Exception:
                    pass

        quality, best_cat, metadata = compute_quality_score(
            heuristics, clip_scores, t_ratio
        )

        # Add CSV tags if available
        if tag_index and h5_path.name in tag_index:
            metadata["csv_tags"] = tag_index[h5_path.name].get("tags", [])

        metadata["is_structure"] = quality >= accept_threshold

        all_scores[h5_path.name] = metadata

        if quality >= accept_threshold:
            accepted.append(h5_path.name)
        elif quality >= reject_threshold:
            review.append(h5_path.name)
        else:
            rejected.append(h5_path.name)

    # Save scores
    scores_path = output_dir / "clip_scores.json"
    with open(scores_path, "w") as f:
        json.dump(all_scores, f, indent=2)
    print(f"\nScores saved to {scores_path}")

    # Create filtered splits
    random.seed(42)
    random.shuffle(accepted)
    n = len(accepted)
    train_end = int(n * 0.8)
    val_end = train_end + int(n * 0.1)

    splits = {
        "train": accepted[:train_end],
        "val": accepted[train_end:val_end],
        "test": accepted[val_end:],
        "review": review,
        "rejected": rejected,
    }

    splits_path = output_dir / "filtered_splits.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"Splits saved to {splits_path}")

    stats = {
        "total": len(h5_files),
        "accepted": len(accepted),
        "review": len(review),
        "rejected": len(rejected),
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "test": len(splits["test"]),
    }

    print(f"\nFiltering summary:")
    print(f"  Total:    {stats['total']}")
    print(f"  Accepted: {stats['accepted']} ({stats['accepted']/max(stats['total'],1)*100:.1f}%)")
    print(f"  Review:   {stats['review']} ({stats['review']/max(stats['total'],1)*100:.1f}%)")
    print(f"  Rejected: {stats['rejected']} ({stats['rejected']/max(stats['total'],1)*100:.1f}%)")
    print(f"\n  Train: {stats['train']} | Val: {stats['val']} | Test: {stats['test']}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="CLIP filtering & labeling pipeline")
    parser.add_argument("--calibrate-only", action="store_true")
    parser.add_argument("--no-clip", action="store_true", help="Run heuristics-only mode")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--reject-threshold", type=float, default=None)
    args = parser.parse_args()

    vocab_path = PROJECT_ROOT / "data" / "vocabulary" / "tok2block.json"
    curation_path = PROJECT_ROOT / "data" / "output" / "curation" / "curation_results.json"
    renders_dir = PROJECT_ROOT / "data" / "output" / "renders"
    output_dir = PROJECT_ROOT / "data" / "output" / "clip_labels"
    tag_index_path = PROJECT_ROOT / "data" / "output" / "tag_index.json"

    tok2block = load_tok2block(vocab_path)
    terrain_tokens = build_terrain_token_set(tok2block)

    # Load tag index if available
    tag_index = None
    if tag_index_path.exists():
        with open(tag_index_path, "r") as f:
            tag_index = json.load(f)
        print(f"Loaded tag index: {len(tag_index)} entries")

    # Initialize CLIP scorer
    clip_scorer = None
    if not args.no_clip:
        try:
            from src.models.clip_scorer import CLIPScorer

            print("Loading CLIP model...")
            clip_scorer = CLIPScorer()
            print("CLIP model loaded")
        except ImportError:
            print("WARNING: open_clip_torch not installed, running heuristics-only")

    # H5 directories
    h5_dirs = [
        PROJECT_ROOT / "data" / "splits" / "train",
        PROJECT_ROOT / "data" / "splits" / "val",
        PROJECT_ROOT / "data" / "splits" / "test",
        PROJECT_ROOT.parent / "kaggle" / "processed_h5",
    ]

    # Calibrate
    if curation_path.exists():
        cal = calibrate_thresholds(
            curation_path, h5_dirs, tok2block, terrain_tokens,
            clip_scorer=clip_scorer, renders_dir=renders_dir,
        )
        accept_thresh = args.threshold or cal["accept_threshold"]
        reject_thresh = args.reject_threshold or cal["reject_threshold"]
    else:
        print("No curation data found, using default thresholds")
        accept_thresh = args.threshold or 0.7
        reject_thresh = args.reject_threshold or 0.4

    if args.calibrate_only:
        return

    # Run filtering
    run_filtering(
        h5_dirs=h5_dirs,
        output_dir=output_dir,
        tok2block=tok2block,
        terrain_tokens=terrain_tokens,
        accept_threshold=accept_thresh,
        reject_threshold=reject_thresh,
        clip_scorer=clip_scorer,
        renders_dir=renders_dir,
        tag_index=tag_index,
    )


if __name__ == "__main__":
    main()
