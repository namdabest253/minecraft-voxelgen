"""Tag Analysis Script — Phase 0

Joins CSV tags from processed_build_dataframe.csv to training H5 files.
Answers: How many House-tagged (and other category) files exist in our training set?

Usage:
    python minecraft_ai/scripts/tag_analysis.py
"""

import ast
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


def load_csv_tag_mapping(csv_path: str) -> dict[str, list[str]]:
    """Parse CSV and return {h5_filename: [tags]} mapping."""
    filename_to_tags: dict[str, list[str]] = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            processed_paths = row.get("PROCESSED_PATHS", "").strip()
            tags_str = row.get("TAGS", "").strip()

            if not processed_paths or not tags_str:
                continue

            # Parse tags list
            try:
                tags = ast.literal_eval(tags_str)
            except (ValueError, SyntaxError):
                continue

            # Parse processed paths list
            try:
                paths = ast.literal_eval(processed_paths)
            except (ValueError, SyntaxError):
                continue

            for p in paths:
                # Normalize: .schem files won't match training H5 files
                # Only H5 files are in our training splits
                if p.endswith(".h5"):
                    filename_to_tags[p] = tags
                elif p.endswith(".schem"):
                    # Try converting schem name to h5 pattern
                    # build_batch_X_Y_Z.schem -> batch_X_Y_Z.h5 (strip 'build_' prefix)
                    h5_name = p.replace(".schem", ".h5")
                    if h5_name.startswith("build_"):
                        h5_name = h5_name[len("build_"):]
                    filename_to_tags[h5_name] = tags

    return filename_to_tags


def get_split_files(splits_dir: str) -> dict[str, set[str]]:
    """Return {split_name: set(filenames)} for train/val/test."""
    result = {}
    for split in ["train", "val", "test"]:
        split_path = Path(splits_dir) / split
        if split_path.exists():
            result[split] = set(os.listdir(split_path))
        else:
            result[split] = set()
    return result


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent.parent
    csv_path = project_root / "kaggle" / "processed_build_dataframe.csv"
    splits_dir = project_root / "minecraft_ai" / "data" / "splits"

    print(f"CSV: {csv_path}")
    print(f"Splits: {splits_dir}")
    print()

    # Step 1: Load CSV tag mapping
    print("Loading CSV tag mapping...")
    tag_map = load_csv_tag_mapping(str(csv_path))
    print(f"  CSV entries with tags: {len(tag_map)}")

    # Step 2: Load training split files
    print("Loading split files...")
    splits = get_split_files(str(splits_dir))
    for split, files in splits.items():
        print(f"  {split}: {len(files)} files")

    all_training_files = splits.get("train", set())
    all_files = set()
    for s in splits.values():
        all_files.update(s)

    # Step 3: Join — find matches
    print("\n--- JOIN RESULTS ---")
    matched_train = {}
    matched_all = {}
    for filename, tags in tag_map.items():
        if filename in all_training_files:
            matched_train[filename] = tags
        if filename in all_files:
            matched_all[filename] = tags

    print(f"Training files with tags: {len(matched_train)} / {len(all_training_files)}")
    print(f"All split files with tags: {len(matched_all)} / {len(all_files)}")
    unmatched = len(all_training_files) - len(matched_train)
    print(f"Training files WITHOUT tags: {unmatched}")

    # Step 4: Tag frequency analysis (training set only)
    print("\n--- TAG FREQUENCY (training set) ---")
    tag_counter: Counter[str] = Counter()
    for tags in matched_train.values():
        for tag in tags:
            tag_counter[tag] += 1

    print(f"{'Tag':<30} {'Count':>6}  {'% of tagged':>10}")
    print("-" * 50)
    for tag, count in tag_counter.most_common(40):
        pct = 100.0 * count / len(matched_train) if matched_train else 0
        print(f"{tag:<30} {count:>6}  {pct:>9.1f}%")

    # Step 5: House-specific analysis
    house_tags = {"House", "Houses", "Home", "Mansion", "Cabin", "Cottage"}
    building_tags = {"Building", "Build"}
    castle_tags = {"Castle", "Fortress", "Palace"}
    tower_tags = {"Tower"}
    medieval_tags = {"Medieval"}

    print("\n--- CATEGORY ANALYSIS (training set) ---")
    categories = {
        "House (House/Houses/Home/Mansion/Cabin/Cottage)": house_tags,
        "Building (Building/Build)": building_tags,
        "Castle (Castle/Fortress/Palace)": castle_tags,
        "Tower": tower_tags,
        "Medieval": medieval_tags,
    }

    category_files: dict[str, list[str]] = {}
    for cat_name, cat_tags in categories.items():
        files = [
            f for f, tags in matched_train.items()
            if any(t in cat_tags for t in tags)
        ]
        category_files[cat_name] = files
        print(f"{cat_name}: {len(files)} training files")

    # House + Building union
    house_or_building = [
        f for f, tags in matched_train.items()
        if any(t in house_tags | building_tags for t in tags)
    ]
    print(f"\nHouse OR Building (union): {len(house_or_building)} training files")

    # House + Medieval union
    house_or_medieval = [
        f for f, tags in matched_train.items()
        if any(t in house_tags | medieval_tags for t in tags)
    ]
    print(f"House OR Medieval (union): {len(house_or_medieval)} training files")

    # All residential (house + building + castle + tower + medieval)
    all_structural = house_tags | building_tags | castle_tags | tower_tags | medieval_tags
    all_structural_files = [
        f for f, tags in matched_train.items()
        if any(t in all_structural for t in tags)
    ]
    print(f"All structural categories (union): {len(all_structural_files)} training files")

    # Step 6: Save results
    output = {
        "csv_entries_with_tags": len(tag_map),
        "training_files_total": len(all_training_files),
        "training_files_with_tags": len(matched_train),
        "training_files_without_tags": unmatched,
        "tag_frequencies_top40": dict(tag_counter.most_common(40)),
        "category_counts": {
            "house": len(category_files["House (House/Houses/Home/Mansion/Cabin/Cottage)"]),
            "building": len(category_files["Building (Building/Build)"]),
            "castle": len(category_files["Castle (Castle/Fortress/Palace)"]),
            "tower": len(category_files["Tower"]),
            "medieval": len(category_files["Medieval"]),
            "house_or_building": len(house_or_building),
            "house_or_medieval": len(house_or_medieval),
            "all_structural": len(all_structural_files),
        },
        "house_files": sorted(category_files["House (House/Houses/Home/Mansion/Cabin/Cottage)"]),
        "house_or_building_files": sorted(house_or_building),
    }

    output_path = project_root / "minecraft_ai" / "data" / "output" / "tag_analysis.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
