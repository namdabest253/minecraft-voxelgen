# Architecture Decisions Log

This document tracks key architectural and design decisions made during the project. Each entry explains what was decided, why, and what alternatives were considered.

---

## Template

```markdown
### YYYY-MM-DD: Decision Title

**Decision**: What we decided

**Context**: Why this decision was needed

**Alternatives Considered**:
1. Alternative A - why rejected
2. Alternative B - why rejected

**Consequences**: What this means for the project

**Status**: Active / Superseded by [link]
```

---

## Decisions

### 2024-12-26: Block Embedding Dimension

**Decision**: Use 32-dimensional embeddings for block2vec

**Context**: Need to represent ~150-200 unique Minecraft blocks in a dense vector space that captures semantic similarity.

**Alternatives Considered**:
1. 16 dimensions - Too small, may not capture nuanced differences between similar blocks
2. 64 dimensions - More expressive but doubles memory usage and computation
3. 128 dimensions - Overkill for ~200 blocks, diminishing returns

**Consequences**:
- Memory efficient (32 floats per block vs 200 for one-hot)
- Proven effective in World-GAN paper
- Good balance for our 10,963 structure dataset

**Status**: Active

---

### 2024-12-26: VQ-VAE Codebook Size

**Decision**: Use 512 codebook entries

**Context**: The codebook determines how many distinct "pattern atoms" the model can learn.

**Alternatives Considered**:
1. 256 entries - May be too few to capture variety in 10k structures
2. 1024 entries - More patterns but harder to train, codes may go unused
3. 2048 entries - Likely overkill, training instability risk

**Consequences**:
- 512 is standard in VQ-VAE literature
- Provides good coverage without codebook collapse issues
- Can adjust if we see unused codes

**Status**: Active

---

### 2024-12-26: Structure Normalization Size

**Decision**: Normalize all structures to 32x32x32 voxels

**Context**: Neural networks need consistent input sizes. Our schematics vary widely in size.

**Alternatives Considered**:
1. 16x16x16 - Too small, loses detail on larger structures
2. 64x64x64 - 8x more memory, may not fit in GPU memory
3. Variable size with padding - Complicates batching

**Consequences**:
- 32,768 voxels per structure (manageable)
- Larger structures will be chunked or downsampled
- Smaller structures padded with air

**Status**: Active

---

### 2024-12-26: Development Tools

**Decision**: Use black, ruff, mypy, pytest with pre-commit hooks

**Context**: Need consistent code style and quality across sessions.

**Alternatives Considered**:
1. No tooling - Inconsistent code, harder to maintain
2. Just black - Missing linting and type checking
3. flake8 + isort instead of ruff - Slower, more dependencies

**Consequences**:
- All code formatted consistently
- Type hints catch errors early
- Tests ensure functionality works
- Pre-commit prevents bad commits

**Status**: Active

---

### 2024-12-26: Config-Driven Hyperparameters

**Decision**: All hyperparameters in YAML config files, never hardcoded

**Context**: Need consistent values across sessions and easy experimentation.

**Alternatives Considered**:
1. Hardcoded in Python - Easy to forget values, inconsistent
2. Environment variables - Harder to version control
3. JSON configs - Less readable than YAML

**Consequences**:
- Easy to change experiments
- Version controlled
- Claude can read configs to know current settings

**Status**: Active

---

## Future Decisions to Make

- [ ] How to handle structures larger than 32x32x32 (chunk vs downsample)
- [ ] Training batch size based on available GPU memory
- [ ] Whether to use data augmentation (rotations, flips)
- [ ] Text description strategy (manual vs auto-generated)
- [ ] Transformer vs diffusion for code generation (Phase 4)
