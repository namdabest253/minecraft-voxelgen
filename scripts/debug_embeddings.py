"""Debug script to check embedding/vocabulary mismatch."""
import numpy as np
import json
from pathlib import Path

BASE = Path("/mnt/c/Users/namda/OneDrive/Desktop/Claude_Server/minecraft_ai")

# Load embeddings
emb = np.load(BASE / "data/kaggle/output/block2vec/v3/block_embeddings_v3.npy")
print(f"Embeddings shape: {emb.shape}")

# Load FULL vocab (what V3 was trained on)
with open(BASE / "data/vocabulary/tok2block.json") as f:
    full_vocab = {int(k): v for k, v in json.load(f).items()}
print(f"Full vocab size: {len(full_vocab)}")

# Load collapsed vocab (used by visualization)
with open(BASE / "data/kaggle/output/block2vec/v2/tok2block_collapsed.json") as f:
    collapsed = {int(k): v for k, v in json.load(f).items()}
print(f"Collapsed vocab size: {len(collapsed)}")

# Load V3 vocab info
with open(BASE / "data/kaggle/output/block2vec/v3/vocab_info_v3.json") as f:
    v3_info = json.load(f)
print(f"V3 shapes: {len(v3_info['shapes'])}")
print(f"V3 materials: {len(v3_info['materials'])}")

# Check some specific blocks and their embeddings
print("\n=== Checking specific blocks ===")
blocks_to_check = [
    'coal_ore', 'copper_ore', 'iron_ore', 'diamond_ore',
    'dark_oak_fence', 'oak_fence', 'spruce_fence',
    'cobblestone_slab', 'stone_slab', 'oak_slab',
    'dark_oak_log', 'oak_log', 'spruce_log'
]

# Find in COLLAPSED vocab (what visualization uses - WRONG!)
found_collapsed = {}
for tok, block in collapsed.items():
    clean = block.replace("minecraft:", "").split("[")[0]
    for name in blocks_to_check:
        if clean == name:
            found_collapsed[name] = tok

# Find in FULL vocab (what V3 was trained on - CORRECT!)
found_full = {}
for tok, block in full_vocab.items():
    clean = block.replace("minecraft:", "").split("[")[0]
    for name in blocks_to_check:
        if clean == name and name not in found_full:  # First match only
            found_full[name] = tok

print("\nToken ID comparison (COLLAPSED vs FULL vocab):")
print("Block Name          | Collapsed ID | Full Vocab ID | MATCH?")
print("-" * 60)
for name in blocks_to_check:
    col_id = found_collapsed.get(name, "N/A")
    full_id = found_full.get(name, "N/A")
    match = "✓" if col_id == full_id else "✗ MISMATCH"
    print(f"{name:18} | {str(col_id):12} | {str(full_id):13} | {match}")

# Check if embeddings at these indices are reasonable
print("\n=== Embedding similarity with WRONG indices (collapsed vocab) ===")

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Using WRONG (collapsed) indices - this is what visualization currently does
if all(x in found_collapsed for x in ['coal_ore', 'iron_ore', 'diamond_ore']):
    coal_emb = emb[found_collapsed['coal_ore']]
    iron_emb = emb[found_collapsed['iron_ore']]
    diamond_emb = emb[found_collapsed['diamond_ore']]
    print(f"coal_ore ↔ iron_ore: {cosine_sim(coal_emb, iron_emb):.3f}")
    print(f"coal_ore ↔ diamond_ore: {cosine_sim(coal_emb, diamond_emb):.3f}")
    print(f"iron_ore ↔ diamond_ore: {cosine_sim(iron_emb, diamond_emb):.3f}")

# Check: Is the issue that embeddings are for FULL vocab but we're using COLLAPSED vocab indices?
print("\n=== CRITICAL CHECK: Vocabulary mismatch ===")
print(f"Embeddings has {emb.shape[0]} entries")
print(f"Collapsed vocab has {len(collapsed)} entries")
print(f"Full vocab has {len(full_vocab)} entries")
print(f"Max token ID in collapsed: {max(collapsed.keys())}")
print(f"Max token ID in full: {max(full_vocab.keys())}")

if emb.shape[0] == len(full_vocab):
    print("\n✓ Embeddings match FULL vocab size!")
    print("The visualization MUST use full vocab, not collapsed!")
    
# Now compare similarities using CORRECT (full vocab) indices
print("\n=== Embedding similarity with CORRECT indices (full vocab) ===")

if all(x in found_full for x in ['coal_ore', 'iron_ore', 'diamond_ore']):
    coal_emb = emb[found_full['coal_ore']]
    iron_emb = emb[found_full['iron_ore']]
    diamond_emb = emb[found_full['diamond_ore']]
    print(f"coal_ore ↔ iron_ore: {cosine_sim(coal_emb, iron_emb):.3f}")
    print(f"coal_ore ↔ diamond_ore: {cosine_sim(coal_emb, diamond_emb):.3f}")
    print(f"iron_ore ↔ diamond_ore: {cosine_sim(iron_emb, diamond_emb):.3f}")

if all(x in found_full for x in ['oak_fence', 'dark_oak_fence', 'spruce_fence']):
    oak_f = emb[found_full['oak_fence']]
    dark_f = emb[found_full['dark_oak_fence']]
    spruce_f = emb[found_full['spruce_fence']]
    print(f"\noak_fence ↔ dark_oak_fence: {cosine_sim(oak_f, dark_f):.3f}")
    print(f"oak_fence ↔ spruce_fence: {cosine_sim(oak_f, spruce_f):.3f}")

if all(x in found_full for x in ['oak_slab', 'cobblestone_slab', 'stone_slab']):
    oak_s = emb[found_full['oak_slab']]
    cob_s = emb[found_full['cobblestone_slab']]
    stone_s = emb[found_full['stone_slab']]
    print(f"\noak_slab ↔ cobblestone_slab: {cosine_sim(oak_s, cob_s):.3f}")
    print(f"oak_slab ↔ stone_slab: {cosine_sim(oak_s, stone_s):.3f}")

if all(x in found_full for x in ['oak_log', 'dark_oak_log', 'spruce_log']):
    oak_l = emb[found_full['oak_log']]
    dark_l = emb[found_full['dark_oak_log']]
    spruce_l = emb[found_full['spruce_log']]
    print(f"\noak_log ↔ dark_oak_log: {cosine_sim(oak_l, dark_l):.3f}")
    print(f"oak_log ↔ spruce_log: {cosine_sim(oak_l, spruce_l):.3f}")

