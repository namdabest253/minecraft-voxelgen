# Minecraft AI Structure Generation

AI-powered Minecraft structure generation from text prompts using VQ-VAE with block2vec embeddings.

## Overview

This project trains a neural network to generate Minecraft structures. The pipeline:

1. **Block2Vec**: Learn semantic embeddings for Minecraft blocks (similar to word2vec)
2. **VQ-VAE**: Encode structures into discrete latent codes and decode them back
3. **Text Encoder** (optional): Connect text descriptions to structure generation

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v
```

## Project Structure

```
minecraft_ai/
├── configs/           # Hyperparameter configs (YAML)
├── data/              # Data storage (not in git)
├── src/               # Source code
│   ├── data/          # Data loading and processing
│   ├── models/        # Neural network architectures
│   ├── training/      # Training loops
│   ├── generation/    # Structure generation
│   └── export/        # Export to Minecraft formats
├── scripts/           # Entry point scripts
├── notebooks/         # Jupyter notebooks
├── tests/             # Unit tests
├── checkpoints/       # Model weights (not in git)
├── logs/              # Training logs (not in git)
└── outputs/           # Generated structures (not in git)
```

## Training Pipeline

### Phase 1: Data Preparation
```bash
python scripts/prepare_data.py
```

### Phase 2: Block2Vec
```bash
python scripts/train.py --phase block2vec
```

### Phase 3: VQ-VAE
```bash
python scripts/train.py --phase vqvae
```

### Phase 4: Text Encoder (Optional)
```bash
python scripts/train.py --phase text_encoder
```

### Generate Structures
```bash
python scripts/generate.py --prompt "medieval castle with towers"
```

## Configuration

All hyperparameters are in `configs/`. Edit these files instead of hardcoding values:

- `block2vec_config.yaml` - Block embedding settings
- `vqvae_config.yaml` - VQ-VAE architecture and training
- `training_config.yaml` - General training settings
- `generation_config.yaml` - Generation and export settings

## Development

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

## Documentation

See `docs/minecraft_ai_generation.md` for detailed technical documentation.

## License

MIT
