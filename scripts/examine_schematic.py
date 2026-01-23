"""Examine a working schematic's NBT structure."""
import nbtlib
from pathlib import Path

schem_path = Path(r"C:\Users\namda\OneDrive\Desktop\Claude_Server\server\plugins\FastAsyncWorldEdit\schematics\sample_10_original.schem")
nbt = nbtlib.load(schem_path)

print("=== NBT Root Structure ===")
for key in nbt.keys():
    val = nbt[key]
    print(f"{key}: type={type(val).__name__}")

print()
print("=== Dimensions ===")
print(f"Width (X): {nbt['Width']}")
print(f"Height (Y): {nbt['Height']}")
print(f"Length (Z): {nbt['Length']}")

print()
print("=== Palette ===")
palette = nbt["Palette"]
print(f"Type: {type(palette).__name__}")
print(f"Size: {len(palette)}")
print("Sample entries:")
for i, (k, v) in enumerate(list(palette.items())[:5]):
    print(f"  {k}: {v}")

print()
print("=== BlockData ===")
block_data = nbt["BlockData"]
print(f"Type: {type(block_data).__name__}")
print(f"Length: {len(block_data)}")
expected_size = int(nbt['Width']) * int(nbt['Height']) * int(nbt['Length'])
print(f"Expected size (W*H*L): {expected_size}")
print(f"First 20 bytes: {list(block_data[:20])}")

print()
print("=== Other keys ===")
for key in nbt.keys():
    if key not in ['Width', 'Height', 'Length', 'Palette', 'BlockData']:
        val = nbt[key]
        print(f"{key}: {repr(val)[:200]}")
