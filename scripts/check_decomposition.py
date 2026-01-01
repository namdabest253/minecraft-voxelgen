"""Check block decomposition coverage."""
import json
import sys
from pathlib import Path

# Import directly from block_decomposition to avoid package init
script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir / 'src' / 'data'))

from block_decomposition import decompose_block

# Load vocabulary
data_path = Path(__file__).parent.parent / 'data/kaggle/output/block2vec/v2/tok2block_collapsed.json'
with open(data_path) as f:
    tok2block = json.load(f)

# Find blocks with no material and generic 'block' shape (potentially missed)
missed = []
shapes = set()
materials = set()

for tok_id, block_name in tok2block.items():
    if block_name == 'UNKNOWN_BLOCK':
        continue

    comp = decompose_block(block_name)
    shapes.add(comp.shape)
    if comp.material:
        materials.add(comp.material)

    # Blocks that are just 'block' shape with no material might be missed
    if comp.shape == 'block' and comp.material is None:
        missed.append(block_name)

print(f"Total blocks in vocabulary: {len(tok2block)}")
print(f"Unique shapes detected: {len(shapes)}")
print(f"Unique materials detected: {len(materials)}")
print(f"\nBlocks with shape='block' and material=None: {len(missed)}")
print("\nThese blocks may need better decomposition rules:")
for name in sorted(missed):
    print(f"  {name}")

print(f"\n\nAll shapes detected ({len(shapes)}):")
print(sorted(shapes))

print(f"\n\nAll materials detected ({len(materials)}):")
print(sorted(materials))
