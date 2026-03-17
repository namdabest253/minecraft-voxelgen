"""Export v6b post-processed structures to Minecraft schematics."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.export.schematic_exporter import export_to_schematic, export_batch_to_schematic


def main():
    # Paths
    data_path = project_root / "data"
    v6b_output = data_path / "output" / "prior" / "diffusion_v6b"
    vocab_path = data_path / "vocabulary" / "tok2block.json"

    # Output directory for schematics (in server plugins folder for easy access)
    schematic_output = Path(r"C:\Users\namda\OneDrive\Desktop\Claude_Server\server\plugins\WorldEdit\schematics")
    schematic_output.mkdir(parents=True, exist_ok=True)

    # Load vocabulary
    print("Loading vocabulary...")
    with open(vocab_path, 'r') as f:
        tok2block = json.load(f)
    print(f"  {len(tok2block)} block types")

    # Air tokens
    air_tokens = {102, 576, 3352}

    # Load processed structures
    print(f"\nLoading structures from {v6b_output}...")
    processed_path = v6b_output / "processed_structures.pt"
    raw_path = v6b_output / "raw_structures.pt"

    if not processed_path.exists():
        print(f"ERROR: {processed_path} not found!")
        print("Make sure v6b has finished running and saved its outputs.")
        return

    processed = torch.load(processed_path, weights_only=False)
    print(f"  Loaded {processed.shape[0]} processed structures")

    # Also load raw for comparison
    if raw_path.exists():
        raw = torch.load(raw_path, weights_only=False)
        print(f"  Loaded {raw.shape[0]} raw structures")
    else:
        raw = None

    # Export processed structures
    print(f"\nExporting processed structures to {schematic_output}...")
    for i in range(processed.shape[0]):
        struct = processed[i]
        if isinstance(struct, torch.Tensor):
            struct = struct.numpy()

        # Count non-air blocks
        non_air = ~np.isin(struct, list(air_tokens))
        if non_air.sum() == 0:
            print(f"  Skipping structure {i} (empty)")
            continue

        output_path = schematic_output / f"v6b_processed_{i}.schem"
        export_to_schematic(
            struct,
            tok2block,
            output_path,
            air_tokens=air_tokens
        )

    # Export raw structures for comparison
    if raw is not None:
        print(f"\nExporting raw (v4) structures for comparison...")
        for i in range(min(4, raw.shape[0])):  # Just export first 4 raw for comparison
            struct = raw[i]
            if isinstance(struct, torch.Tensor):
                struct = struct.numpy()

            non_air = ~np.isin(struct, list(air_tokens))
            if non_air.sum() == 0:
                continue

            output_path = schematic_output / f"v6b_raw_{i}.schem"
            export_to_schematic(
                struct,
                tok2block,
                output_path,
                air_tokens=air_tokens
            )

    print("\n" + "="*60)
    print("DONE! Schematics exported to:")
    print(f"  {schematic_output}")
    print("\nTo test in Minecraft:")
    print("  1. Join your server")
    print("  2. Get a WorldEdit wand: //wand")
    print("  3. Load a schematic: //schem load v6b_processed_0")
    print("  4. Position yourself where you want it")
    print("  5. Paste: //paste")
    print("\nCompare processed vs raw:")
    print("  //schem load v6b_processed_0  (post-processed)")
    print("  //schem load v6b_raw_0        (raw v4 output)")
    print("="*60)


if __name__ == "__main__":
    main()
