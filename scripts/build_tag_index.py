"""
Build a tag index mapping H5 filenames to CSV tags.

Parses kaggle/processed_build_dataframe.csv and maps .schem filenames
(listed in PROCESSED_PATHS) to their tags and page URLs, then converts
to H5 filenames for use in CLIP filtering.

Output: minecraft_ai/data/output/tag_index.json

Usage:
    python minecraft_ai/scripts/build_tag_index.py
"""

import ast
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
REPO_ROOT = PROJECT_ROOT.parent


def build_tag_index(
    csv_path: Path,
    output_path: Path,
) -> dict:
    """Parse CSV and build {h5_filename: {tags, page_url}} mapping.

    Args:
        csv_path: Path to processed_build_dataframe.csv
        output_path: Where to write tag_index.json

    Returns:
        The tag index dict.
    """
    # schem_name -> {tags, page_url}
    schem_to_meta: dict = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            processed_paths = row.get("PROCESSED_PATHS", "").strip()
            if not processed_paths:
                continue

            # Parse the Python list literal from CSV
            try:
                schem_list = ast.literal_eval(processed_paths)
            except (ValueError, SyntaxError):
                continue

            if not isinstance(schem_list, list):
                continue

            # Parse tags
            tags_str = row.get("TAGS", "[]").strip()
            try:
                tags = ast.literal_eval(tags_str)
            except (ValueError, SyntaxError):
                tags = []

            page_url = row.get("PAGE_URL", "")

            for schem_name in schem_list:
                schem_to_meta[schem_name] = {
                    "tags": tags,
                    "page_url": page_url,
                }

    # Convert .schem filenames to .h5 filenames
    tag_index = {}
    for schem_name, meta in schem_to_meta.items():
        h5_name = schem_name.replace(".schem", ".h5")
        tag_index[h5_name] = meta

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(tag_index, f, indent=2)

    return tag_index


def main():
    csv_path = REPO_ROOT / "kaggle" / "processed_build_dataframe.csv"
    output_path = PROJECT_ROOT / "data" / "output" / "tag_index.json"

    if not csv_path.exists():
        print(f"ERROR: CSV not found at {csv_path}")
        sys.exit(1)

    print(f"Reading CSV: {csv_path}")
    tag_index = build_tag_index(csv_path, output_path)

    # Summary stats
    all_tags = defaultdict(int)
    for meta in tag_index.values():
        for tag in meta["tags"]:
            all_tags[tag] += 1

    print(f"\nTag index built: {len(tag_index)} H5 files mapped")
    print(f"Output: {output_path}")
    print(f"\nTop 20 tags:")
    for tag, count in sorted(all_tags.items(), key=lambda x: -x[1])[:20]:
        print(f"  {tag}: {count}")


if __name__ == "__main__":
    main()
