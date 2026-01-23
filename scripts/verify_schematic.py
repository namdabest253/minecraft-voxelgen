"""Verify a schematic file is readable by comparing with a known working one."""
import nbtlib
from pathlib import Path

def examine_schem(path):
    print(f"\n=== {path.name} ===")
    try:
        nbt = nbtlib.load(path)
        print("Keys:", list(nbt.keys()))
        print(f"Version: {nbt.get('Version', 'N/A')}")
        print(f"DataVersion: {nbt.get('DataVersion', 'N/A')}")
        print(f"Width: {nbt.get('Width', 'N/A')}")
        print(f"Height: {nbt.get('Height', 'N/A')}")
        print(f"Length: {nbt.get('Length', 'N/A')}")
        print(f"PaletteMax: {nbt.get('PaletteMax', 'N/A')}")

        palette = nbt.get('Palette', {})
        print(f"Palette size: {len(palette)}")
        print("First 3 palette entries:", list(palette.items())[:3])

        block_data = nbt.get('BlockData', [])
        print(f"BlockData length: {len(block_data)}")

        expected = int(nbt['Width']) * int(nbt['Height']) * int(nbt['Length'])
        print(f"Expected (W*H*L): {expected}")

        if len(block_data) != expected:
            print(f"WARNING: BlockData length mismatch!")

        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False

# Check original and reconstruction
schematics_dir = Path(r"C:\Users\namda\OneDrive\Desktop\Claude_Server\server\plugins\FastAsyncWorldEdit\schematics")

original = schematics_dir / "sample_10_original.schem"
recon = schematics_dir / "sample_10_original_v51_recon.schem"

examine_schem(original)
examine_schem(recon)
