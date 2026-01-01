"""Create a V3-only validation notebook."""
import json

# Read the existing validation notebook
with open(r'C:\Users\namda\OneDrive\Desktop\Claude_Server\minecraft_ai\data\kaggle\notebooks\vqvae_embedding_validation.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 0: Update title
nb['cells'][0]['source'] = """# VQ-VAE Embedding Validation: V3 Only

## Purpose
Validate Block2Vec V3 (compositional) embeddings for VQ-VAE structure reconstruction.

Compare against the V2 validation results:
- V1: 44.3% structure accuracy
- V2: 43.0% structure accuracy
- Random: 39.0% structure accuracy

**Key Question**: Can V3 compositional embeddings beat V1's 44.3%?
"""

# Cell 2: Update config for V3 only
nb['cells'][2]['source'] = """# ============================================================
# CELL 2: Configuration (V3 ONLY)
# ============================================================

# === Data Paths ===
DATA_DIR = "/kaggle/input/minecraft-schematics/minecraft_splits/splits/train"
VAL_DIR = "/kaggle/input/minecraft-schematics/minecraft_splits/splits/val"
VOCAB_PATH = "/kaggle/input/minecraft-schematics/tok2block.json"

# V3 embeddings path
V3_EMBEDDINGS_PATH = "/kaggle/input/block2vec-v3/block_embeddings_v3.npy"

OUTPUT_DIR = "/kaggle/working"

# === Mini Model Architecture (same as V2 validation) ===
BLOCK_EMBEDDING_DIM = 32
HIDDEN_DIMS = [32, 64, 128]
LATENT_DIM = 128
NUM_CODEBOOK_ENTRIES = 512
COMMITMENT_COST = 0.25

# === Training ===
EPOCHS = 10
BATCH_SIZE = 4
LEARNING_RATE = 3e-4
USE_AMP = True
GRAD_ACCUM_STEPS = 4

# === Other ===
SEED = 42
NUM_WORKERS = 2

print("V3 Validation Configuration:")
print(f"  Hidden dims: {HIDDEN_DIMS}")
print(f"  Latent dim: {LATENT_DIM}")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRAD_ACCUM_STEPS})")
"""

# Cell 3: Load only V3 embeddings
nb['cells'][3]['source'] = """# ============================================================
# CELL 3: Load Vocabulary and V3 Embeddings
# ============================================================

# Load vocabulary
with open(VOCAB_PATH, 'r') as f:
    tok2block = {int(k): v for k, v in json.load(f).items()}

VOCAB_SIZE = len(tok2block)
print(f"Vocabulary size: {VOCAB_SIZE} block types")

# ============================================================
# Find ALL air tokens (not just token 0!)
# ============================================================
AIR_TOKENS = set()
for tok, block in tok2block.items():
    block_lower = block.lower()
    if 'air' in block_lower and 'stair' not in block_lower:
        AIR_TOKENS.add(tok)
        print(f"  Found air token: {tok} = '{block}'")

AIR_TOKENS_TENSOR = torch.tensor(sorted(AIR_TOKENS), dtype=torch.long)
print(f"\\nAir tokens: {AIR_TOKENS_TENSOR.tolist()}")

# Load V3 embeddings (should be 3717 x 32)
v3_embeddings = np.load(V3_EMBEDDINGS_PATH)
print(f"\\nV3 embeddings shape: {v3_embeddings.shape}")

# Create random baseline for comparison
np.random.seed(SEED)
random_embeddings = np.random.randn(VOCAB_SIZE, BLOCK_EMBEDDING_DIM).astype(np.float32)
random_embeddings = random_embeddings / np.linalg.norm(random_embeddings, axis=1, keepdims=True)
print(f"Random embeddings shape: {random_embeddings.shape}")

# Embeddings dict - V3 and Random only
EMBEDDINGS = {
    "V3": torch.tensor(v3_embeddings, dtype=torch.float32),
    "Random": torch.tensor(random_embeddings, dtype=torch.float32),
}

print(f"\\nEmbeddings to test: {list(EMBEDDINGS.keys())}")

# Save air tokens info
air_info = {
    "air_tokens": sorted(AIR_TOKENS),
    "note": "These tokens were excluded from structure accuracy calculation"
}
with open(f"{OUTPUT_DIR}/air_tokens_used.json", 'w') as f:
    json.dump(air_info, f, indent=2)
"""

# Find and modify the training loop cell to only run V3 and Random
for i, cell in enumerate(nb['cells']):
    source = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])

    # Find the main training loop
    if 'for emb_name, embeddings in EMBEDDINGS.items()' in source:
        print(f"Found training loop in cell {i}")
        # This cell should work as-is since EMBEDDINGS now only has V3 and Random

    # Update results saving cell
    if 'embedding_validation_results.json' in source:
        print(f"Found results saving in cell {i}")

# Find and update the final comparison/summary cell
for i, cell in enumerate(nb['cells']):
    source = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
    if 'V1' in source and 'V2' in source and 'comparison' in source.lower():
        # Update to compare V3 vs previous results
        nb['cells'][i]['source'] = source.replace(
            'for emb_name in ["V1", "V2", "Random"]',
            'for emb_name in ["V3", "Random"]'
        )

# Update any plotting code that references V1/V2
for i, cell in enumerate(nb['cells']):
    source = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
    if 'plt.' in source and ('V1' in source or 'V2' in source):
        # Replace V1/V2 references with V3
        new_source = source
        new_source = new_source.replace('"V1", "V2", "Random"', '"V3", "Random"')
        new_source = new_source.replace("['V1', 'V2', 'Random']", "['V3', 'Random']")
        if new_source != source:
            nb['cells'][i]['source'] = new_source
            print(f"Updated plotting in cell {i}")

# Save the modified notebook
with open(r'C:\Users\namda\OneDrive\Desktop\Claude_Server\minecraft_ai\data\kaggle\notebooks\vqvae_v3_validation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("\nCreated vqvae_v3_validation.ipynb")
print("This notebook will only validate V3 and Random (not V1/V2)")
print("\nYou'll need to upload block2vec-v3 as a Kaggle dataset containing:")
print("  - block_embeddings_v3.npy")
