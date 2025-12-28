"""
Evaluate Block2Vec embedding quality.

This script runs several tests to quantify how good our block embeddings are:
1. Category coherence - Are nearest neighbors in the same category?
2. Block state consistency - Are variants of same block similar?
3. Analogy tests - Do arithmetic relationships hold?
4. Air contamination - Is air too similar to everything?

Usage:
    python scripts/evaluate_block2vec.py \
        --embeddings data/kaggle/output/block_embeddings.npy \
        --vocab data/kaggle/input/tok2block.json \
        --output outputs/block2vec_evaluation.json
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def load_data(embeddings_path: Path, vocab_path: Path) -> Tuple[np.ndarray, Dict[int, str]]:
    """Load embeddings and vocabulary."""
    embeddings = np.load(embeddings_path)
    with open(vocab_path) as f:
        tok2block = {int(k): v for k, v in json.load(f).items()}
    return embeddings, tok2block


def get_base_block_name(block_name: str) -> str:
    """Extract base block name without namespace or state."""
    name = block_name.replace("minecraft:", "")
    name = re.sub(r"\[.*\]", "", name)
    return name


def categorize_block(block_name: str) -> str:
    """Assign a block to a category based on its name."""
    name = get_base_block_name(block_name).lower()

    # Define categories
    if "air" in name and "stair" not in name:
        return "air"
    elif any(x in name for x in ["_planks"]):
        return "planks"
    elif any(x in name for x in ["_log", "_wood", "_stem"]) and "stripped" not in name:
        return "logs"
    elif "stripped" in name and any(x in name for x in ["log", "wood", "stem"]):
        return "stripped_logs"
    elif "_stairs" in name:
        return "stairs"
    elif "_slab" in name:
        return "slabs"
    elif "_fence" in name and "gate" not in name:
        return "fences"
    elif "_door" in name:
        return "doors"
    elif "_wall" in name and "sign" not in name:
        return "walls"
    elif "_ore" in name:
        return "ores"
    elif any(x in name for x in ["_block"]) and any(x in name for x in ["diamond", "gold", "iron", "emerald", "lapis", "coal", "copper", "netherite"]):
        return "metal_blocks"
    elif "glass" in name and "pane" not in name:
        return "glass"
    elif "glass_pane" in name:
        return "glass_panes"
    elif "_wool" in name:
        return "wool"
    elif "_concrete" in name and "powder" not in name:
        return "concrete"
    elif "_terracotta" in name:
        return "terracotta"
    elif "leaves" in name:
        return "leaves"
    elif "stone" in name and "brick" not in name and "sand" not in name:
        return "stone_types"
    elif "brick" in name:
        return "bricks"
    elif "sand" in name and "stone" not in name:
        return "sand"
    elif "water" in name:
        return "water"
    elif "lava" in name:
        return "lava"
    else:
        return "other"


def find_nearest_neighbors(
    embeddings: np.ndarray,
    query_idx: int,
    k: int = 10
) -> List[Tuple[int, float]]:
    """Find k nearest neighbors for a block."""
    query = embeddings[query_idx].reshape(1, -1)
    similarities = cosine_similarity(query, embeddings)[0]

    # Get top k+1 (excluding self)
    top_indices = np.argsort(similarities)[::-1]

    results = []
    for idx in top_indices:
        if idx != query_idx:
            results.append((int(idx), float(similarities[idx])))
            if len(results) >= k:
                break
    return results


def test_category_coherence(
    embeddings: np.ndarray,
    tok2block: Dict[int, str],
    k: int = 5
) -> Dict[str, float]:
    """
    Test if nearest neighbors are in the same category.

    For each block, find k nearest neighbors and check what fraction
    are in the same category. Higher = better embeddings.
    """
    print("\n=== Test 1: Category Coherence ===")

    # Categorize all blocks
    block_categories = {}
    category_members = defaultdict(list)

    for tok, name in tok2block.items():
        cat = categorize_block(name)
        block_categories[tok] = cat
        if cat != "other":
            category_members[cat].append(tok)

    # Test each category
    results = {}

    for category, members in sorted(category_members.items()):
        if len(members) < 2:
            continue

        correct = 0
        total = 0

        for tok in members:
            if tok >= len(embeddings):
                continue

            neighbors = find_nearest_neighbors(embeddings, tok, k)

            for neighbor_tok, sim in neighbors:
                if neighbor_tok in block_categories:
                    total += 1
                    if block_categories[neighbor_tok] == category:
                        correct += 1

        if total > 0:
            precision = correct / total
            results[category] = precision
            status = "GOOD" if precision > 0.5 else "POOR" if precision > 0.2 else "BAD"
            print(f"  {category:20} precision: {precision:.1%} ({correct}/{total}) [{status}]")

    avg_precision = np.mean(list(results.values())) if results else 0
    print(f"\n  Average precision: {avg_precision:.1%}")

    return results


def test_block_state_consistency(
    embeddings: np.ndarray,
    tok2block: Dict[int, str]
) -> Dict[str, float]:
    """
    Test if block state variants have similar embeddings.

    e.g., oak_stairs[facing=north] should be similar to oak_stairs[facing=south]
    """
    print("\n=== Test 2: Block State Consistency ===")

    # Group blocks by base name
    base_name_groups = defaultdict(list)

    for tok, name in tok2block.items():
        base = get_base_block_name(name)
        if tok < len(embeddings):
            base_name_groups[base].append(tok)

    # Filter to groups with multiple variants
    multi_variant_groups = {k: v for k, v in base_name_groups.items() if len(v) > 1}

    results = {}
    low_consistency = []

    for base_name, tokens in sorted(multi_variant_groups.items()):
        if len(tokens) < 2:
            continue

        # Compute pairwise similarities within group
        group_embeddings = embeddings[tokens]
        sims = cosine_similarity(group_embeddings)

        # Get average similarity (excluding diagonal)
        mask = ~np.eye(len(tokens), dtype=bool)
        avg_sim = sims[mask].mean()

        results[base_name] = avg_sim

        if avg_sim < 0.7:
            low_consistency.append((base_name, avg_sim, len(tokens)))

    # Summary
    all_sims = list(results.values())
    print(f"  Blocks with multiple variants: {len(results)}")
    print(f"  Average within-group similarity: {np.mean(all_sims):.3f}")
    print(f"  Minimum within-group similarity: {np.min(all_sims):.3f}")

    if low_consistency:
        print(f"\n  Low consistency blocks (sim < 0.7):")
        for name, sim, count in sorted(low_consistency, key=lambda x: x[1])[:10]:
            print(f"    {name}: {sim:.3f} ({count} variants)")

    return results


def test_analogy(
    embeddings: np.ndarray,
    tok2block: Dict[int, str],
    a: str, b: str, c: str, expected: str
) -> Tuple[bool, str, float]:
    """
    Test analogy: a is to b as c is to ?

    vec(?) = vec(b) - vec(a) + vec(c)
    """
    block2tok = {}
    for tok, name in tok2block.items():
        base = get_base_block_name(name)
        if base not in block2tok:
            block2tok[base] = tok

    if a not in block2tok or b not in block2tok or c not in block2tok:
        return False, "NOT_FOUND", 0.0

    tok_a, tok_b, tok_c = block2tok[a], block2tok[b], block2tok[c]

    if any(t >= len(embeddings) for t in [tok_a, tok_b, tok_c]):
        return False, "OUT_OF_RANGE", 0.0

    # Compute analogy vector
    result_vec = embeddings[tok_b] - embeddings[tok_a] + embeddings[tok_c]
    result_vec = result_vec.reshape(1, -1)

    # Find nearest
    similarities = cosine_similarity(result_vec, embeddings)[0]

    # Exclude input tokens
    for t in [tok_a, tok_b, tok_c]:
        similarities[t] = -999

    best_tok = int(np.argmax(similarities))
    best_name = get_base_block_name(tok2block.get(best_tok, "unknown"))
    best_sim = similarities[best_tok]

    success = best_name == expected
    return success, best_name, float(best_sim)


def test_analogies(
    embeddings: np.ndarray,
    tok2block: Dict[int, str]
) -> Dict[str, any]:
    """Run a suite of analogy tests."""
    print("\n=== Test 3: Analogy Tests ===")
    print("  Format: A is to B as C is to ? (expected: D)")

    analogies = [
        # Wood type analogies
        ("oak_planks", "oak_log", "spruce_planks", "spruce_log"),
        ("oak_planks", "oak_log", "birch_planks", "birch_log"),
        ("oak_planks", "oak_stairs", "spruce_planks", "spruce_stairs"),

        # Material analogies
        ("stone", "stone_bricks", "nether_bricks", "netherrack"),
        ("cobblestone", "cobblestone_stairs", "stone_bricks", "stone_brick_stairs"),

        # Color analogies
        ("white_wool", "red_wool", "white_concrete", "red_concrete"),
        ("white_wool", "blue_wool", "white_terracotta", "blue_terracotta"),

        # Block type analogies
        ("oak_planks", "oak_slab", "stone", "stone_slab"),
        ("oak_planks", "oak_fence", "nether_bricks", "nether_brick_fence"),
    ]

    results = []
    correct = 0

    for a, b, c, expected in analogies:
        success, actual, sim = test_analogy(embeddings, tok2block, a, b, c, expected)
        results.append({
            "a": a, "b": b, "c": c,
            "expected": expected,
            "actual": actual,
            "correct": success,
            "similarity": sim
        })

        status = "OK" if success else "WRONG"
        print(f"  {a} : {b} :: {c} : ? = {actual} (expected {expected}) [{status}]")

        if success:
            correct += 1

    accuracy = correct / len(analogies) if analogies else 0
    print(f"\n  Analogy accuracy: {accuracy:.1%} ({correct}/{len(analogies)})")

    return {"accuracy": accuracy, "results": results}


def test_air_contamination(
    embeddings: np.ndarray,
    tok2block: Dict[int, str]
) -> Dict[str, any]:
    """
    Test if air is too similar to unrelated blocks.

    Air should be similar to other "empty" blocks but not to solid blocks.
    """
    print("\n=== Test 4: Air Contamination ===")

    # Find air token
    air_tok = None
    for tok, name in tok2block.items():
        if name == "minecraft:air":
            air_tok = tok
            break

    if air_tok is None or air_tok >= len(embeddings):
        print("  Air token not found")
        return {}

    # Compute similarity to all blocks
    air_emb = embeddings[air_tok].reshape(1, -1)
    similarities = cosine_similarity(air_emb, embeddings)[0]

    # Analyze
    high_sim_threshold = 0.5
    high_sim_blocks = []

    for tok, sim in enumerate(similarities):
        if tok != air_tok and sim > high_sim_threshold:
            name = tok2block.get(tok, "unknown")
            cat = categorize_block(name)
            if cat != "air":  # Exclude other air variants
                high_sim_blocks.append((name, cat, sim))

    print(f"  Blocks with similarity > {high_sim_threshold} to air: {len(high_sim_blocks)}")

    if high_sim_blocks:
        print(f"  Top offenders:")
        for name, cat, sim in sorted(high_sim_blocks, key=lambda x: -x[2])[:10]:
            base = get_base_block_name(name)
            print(f"    {base:30} ({cat:15}) sim={sim:.3f}")

    # Category breakdown
    category_sims = defaultdict(list)
    for tok, name in tok2block.items():
        if tok < len(embeddings) and tok != air_tok:
            cat = categorize_block(name)
            category_sims[cat].append(similarities[tok])

    print(f"\n  Average similarity to air by category:")
    for cat in sorted(category_sims.keys()):
        avg_sim = np.mean(category_sims[cat])
        status = "HIGH" if avg_sim > 0.3 else "OK"
        print(f"    {cat:20} {avg_sim:.3f} [{status}]")

    return {
        "high_similarity_count": len(high_sim_blocks),
        "category_similarities": {k: float(np.mean(v)) for k, v in category_sims.items()}
    }


def test_nearest_neighbors_examples(
    embeddings: np.ndarray,
    tok2block: Dict[int, str]
) -> None:
    """Print nearest neighbors for some example blocks."""
    print("\n=== Test 5: Nearest Neighbor Examples ===")

    test_blocks = [
        "oak_planks", "stone", "glass", "diamond_ore",
        "white_wool", "water", "torch", "chest"
    ]

    block2tok = {}
    for tok, name in tok2block.items():
        base = get_base_block_name(name)
        if base not in block2tok:
            block2tok[base] = tok

    for block in test_blocks:
        if block not in block2tok:
            continue

        tok = block2tok[block]
        if tok >= len(embeddings):
            continue

        neighbors = find_nearest_neighbors(embeddings, tok, 5)

        print(f"\n  {block}:")
        for n_tok, sim in neighbors:
            n_name = get_base_block_name(tok2block.get(n_tok, "unknown"))
            n_cat = categorize_block(tok2block.get(n_tok, ""))
            print(f"    {n_name:25} ({n_cat:12}) sim={sim:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Block2Vec embeddings")
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/block2vec_evaluation.json"))
    args = parser.parse_args()

    print("=" * 60)
    print("Block2Vec Embedding Evaluation")
    print("=" * 60)

    # Load data
    embeddings, tok2block = load_data(args.embeddings, args.vocab)
    print(f"\nLoaded embeddings: {embeddings.shape}")
    print(f"Vocabulary size: {len(tok2block)}")

    # Run tests
    results = {}

    results["category_coherence"] = test_category_coherence(embeddings, tok2block)
    results["block_state_consistency"] = test_block_state_consistency(embeddings, tok2block)
    results["analogies"] = test_analogies(embeddings, tok2block)
    results["air_contamination"] = test_air_contamination(embeddings, tok2block)
    test_nearest_neighbors_examples(embeddings, tok2block)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    cat_precision = np.mean(list(results["category_coherence"].values())) if results["category_coherence"] else 0
    state_consistency = np.mean(list(results["block_state_consistency"].values())) if results["block_state_consistency"] else 0
    analogy_acc = results["analogies"].get("accuracy", 0)
    air_contamination = results["air_contamination"].get("high_similarity_count", 0)

    print(f"\n  Category coherence (precision): {cat_precision:.1%}")
    print(f"  Block state consistency:        {state_consistency:.3f}")
    print(f"  Analogy accuracy:               {analogy_acc:.1%}")
    print(f"  Air contamination (high sim):   {air_contamination} blocks")

    # Overall assessment
    print("\n  Overall Assessment:")
    issues = []
    if cat_precision < 0.3:
        issues.append("Category coherence is poor - similar blocks don't cluster")
    if state_consistency < 0.7:
        issues.append("Block variants are inconsistent - states not similar")
    if analogy_acc < 0.3:
        issues.append("Analogies fail - relationships not captured")
    if air_contamination > 100:
        issues.append("Air contamination high - air similar to too many blocks")

    if not issues:
        print("  Embeddings look GOOD - proceed with VQ-VAE")
    else:
        print("  Embeddings have ISSUES:")
        for issue in issues:
            print(f"    - {issue}")
        print("\n  Consider retraining Block2Vec with:")
        print("    - CBOW + Skip-gram hybrid")
        print("    - Collapsed block states")
        print("    - Larger context window")

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(args.output, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\n  Results saved to {args.output}")


if __name__ == "__main__":
    main()
