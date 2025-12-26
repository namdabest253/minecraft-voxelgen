# Minecraft AI Structure Generation - Project Context

**Read this file at the start of every session.**

## Project Overview

Training an AI model to generate Minecraft structures from text prompts. Uses VQ-VAE with block2vec embeddings, inspired by World-GAN and text2mc research.

## Current Phase

**Phase 1: Data Preparation** - Not started

## Progress Tracker

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1: Data Preparation | Not started | Parse 10,963 schematics |
| Phase 2: Block2Vec | Not started | Train block embeddings |
| Phase 3: VQ-VAE | Not started | Train encoder/decoder |
| Phase 4: Text Encoder | Not started | Optional - CLIP integration |
| Phase 5: Generation | Not started | Export to Minecraft |

## Key Locations

```
Project root:     C:\Users\namda\OneDrive\Desktop\Claude_Server\minecraft_ai\
Schematics:       C:\Users\namda\OneDrive\Desktop\Claude_Server\schematics\ (10,963 files)
Documentation:    C:\Users\namda\OneDrive\Desktop\Claude_Server\docs\minecraft_ai_generation.md
Minecraft server: C:\Users\namda\OneDrive\Desktop\Claude_Server\server\
NBT output:       C:\Users\namda\OneDrive\Desktop\Claude_Server\server\world\generated\minecraft\structures\
```

## Code Conventions

### Python Style
- **Formatter**: black (line length 88)
- **Linter**: ruff
- **Type hints**: Required for all function signatures
- **Docstrings**: Google style

### Naming Conventions
```python
# Files: snake_case
schematic_parser.py
block2vec.py

# Classes: PascalCase
class VQVAEEncoder:
class BlockVocabulary:

# Functions/variables: snake_case
def parse_schematic(path: str) -> np.ndarray:
embedding_dim = 32

# Constants: UPPER_SNAKE_CASE
BLOCK_EMBEDDING_DIM = 32
CODEBOOK_SIZE = 512
```

### Import Order
```python
# 1. Standard library
import os
from pathlib import Path

# 2. Third-party
import numpy as np
import torch
import torch.nn as nn

# 3. Local
from src.data import schematic_parser
from src.models import block2vec
```

## Architecture Decisions

See `DECISIONS.md` for detailed rationale.

| Decision | Value | Reason |
|----------|-------|--------|
| Block embedding dim | 32 | Balance of expressiveness vs memory |
| Codebook size | 512 | Enough patterns without over-fragmentation |
| Structure size | 32x32x32 | Fits in memory, captures most builds |
| Batch size | 16 | Fits on 8GB GPU |

## Current Session

See `SESSION_STATUS.md` for detailed current state.

## Config Files

All hyperparameters are in `configs/*.yaml`. Never hardcode values.

- `configs/block2vec_config.yaml` - Block embedding training
- `configs/vqvae_config.yaml` - VQ-VAE architecture and training
- `configs/training_config.yaml` - General training settings
- `configs/generation_config.yaml` - Inference settings

## Testing

Run tests before committing:
```bash
pytest tests/ -v
```

## Common Commands

```bash
# Format code
black src/ scripts/ tests/

# Lint
ruff check src/ scripts/ tests/

# Type check
mypy src/

# Run all checks
pre-commit run --all-files
```
