"""Extract block IDs from Minecraft Wiki markdown export and compare with vocabulary."""
import re
import json
from pathlib import Path

# Read the wiki markdown file
wiki_path = Path("C:/Users/namda/OneDrive/Desktop/Claude_Server/data/minecraft/blocks.md")
with open(wiki_path, encoding='utf-8') as f:
    content = f.read()

# The wiki table format is:
# |[[File:Block.png|30px]]||{{code|block_id}}||[[Block Name]]||...
# We want the SECOND {{code|...}} in each row - that's the resource location

# Split by table rows
rows = content.split('|-')

block_ids = set()
for row in rows:
    # Find all {{code|...}} in this row
    matches = re.findall(r'\{\{code\|([a-z0-9_]+)\}\}', row)
    # The first match after the image is the block ID (resource location)
    if matches and len(matches) >= 1:
        # First code match is the resource location
        block_id = matches[0]
        # Filter out obvious texture names (contain _top, _side, _bottom, _front, etc.)
        texture_suffixes = ['_top', '_side', '_bottom', '_front', '_back', '_inner', '_outer',
                           '_particle', '_on', '_off', '_lit', '_unlit', '_open', '_closed']
        is_texture = any(block_id.endswith(s) for s in texture_suffixes)
        if not is_texture:
            block_ids.add(block_id)

print(f"Block IDs extracted: {len(block_ids)}")

# Load our vocabulary
vocab_path = Path("C:/Users/namda/OneDrive/Desktop/Claude_Server/minecraft_ai/data/kaggle/output/block2vec/v2/tok2block_collapsed.json")
with open(vocab_path) as f:
    tok2block = json.load(f)

our_blocks = set()
for block_name in tok2block.values():
    if block_name != 'UNKNOWN_BLOCK':
        # Remove minecraft: prefix
        clean = block_name.replace('minecraft:', '')
        our_blocks.add(clean)

print(f"Our vocabulary: {len(our_blocks)} blocks")

# Find blocks in wiki but not in our vocab
wiki_only = block_ids - our_blocks
print(f"\nBlocks in Wiki but NOT in our vocabulary: {len(wiki_only)}")
for b in sorted(wiki_only)[:50]:
    print(f"  {b}")
if len(wiki_only) > 50:
    print(f"  ... and {len(wiki_only) - 50} more")

# Find blocks in our vocab but not in wiki (might be old/renamed blocks)
our_only = our_blocks - block_ids
print(f"\nBlocks in our vocabulary but NOT in Wiki: {len(our_only)}")
for b in sorted(our_only):
    print(f"  {b}")

# Save the complete wiki block list
output_path = Path("C:/Users/namda/OneDrive/Desktop/Claude_Server/data/minecraft/all_blocks.json")
with open(output_path, 'w') as f:
    json.dump(sorted(block_ids), f, indent=2)

print(f"\nSaved {len(block_ids)} block IDs to {output_path}")
