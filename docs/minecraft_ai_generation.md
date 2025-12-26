# Minecraft AI Structure Generation - Technical Deep Dive

This document covers the technical methods for training AI models to generate Minecraft builds, based on research from World-GAN, VoxelCNN, text2mc, and related projects.

## Table of Contents
1. [Project Structure](#project-structure)
2. [Tools and Libraries](#tools-and-libraries)
3. [The Core Problem](#the-core-problem)
4. [Block Embeddings (block2vec)](#block-embeddings-block2vec)
5. [VAE Architecture](#vae-architecture)
6. [VQ-VAE (Discrete Latent Space)](#vq-vae-discrete-latent-space)
7. [GAN Architecture (World-GAN)](#gan-architecture-world-gan)
8. [Latent Space Interpolation](#latent-space-interpolation)
9. [Text-to-Structure Pipeline](#text-to-structure-pipeline)
10. [Existing Datasets](#existing-datasets)
11. [Key Papers & Resources](#key-papers--resources)
12. [Detailed Training Pipeline](#detailed-training-pipeline)

---

## Project Structure

This is the directory layout after completing all phases of the project:

```
minecraft_ai/
│
├── README.md                      # Project overview and quick start guide
├── requirements.txt               # Python dependencies
├── setup.py                       # Package installation script
├── .gitignore                     # Git ignore rules
│
├── configs/                       # Configuration files
│   ├── block2vec_config.yaml      # Block2Vec hyperparameters
│   ├── vqvae_config.yaml          # VQ-VAE hyperparameters
│   ├── training_config.yaml       # General training settings
│   └── generation_config.yaml     # Generation/inference settings
│
├── data/                          # All data-related files
│   ├── raw/                       # Original schematic files
│   │   └── schematics/            # Your 10,963 .schematic files
│   │
│   ├── processed/                 # Processed data ready for training
│   │   ├── structures/            # Parsed numpy arrays (.npy files)
│   │   ├── train/                 # Training split (80%)
│   │   ├── val/                   # Validation split (10%)
│   │   └── test/                  # Test split (10%)
│   │
│   ├── vocabulary/                # Block mappings
│   │   ├── block_to_id.json       # {"minecraft:stone": 1, ...}
│   │   ├── id_to_block.json       # {1: "minecraft:stone", ...}
│   │   └── block_metadata.json    # Additional block properties
│   │
│   └── embeddings/                # Trained embeddings
│       └── block2vec_32d.npy      # Block embeddings (N_blocks × 32)
│
├── src/                           # Source code
│   ├── __init__.py
│   │
│   ├── data/                      # Data processing modules
│   │   ├── __init__.py
│   │   ├── schematic_parser.py    # Parse .schematic files to numpy
│   │   ├── nbt_parser.py          # Parse .nbt structure files
│   │   ├── vocabulary.py          # Build and manage block vocabulary
│   │   ├── dataset.py             # PyTorch Dataset classes
│   │   └── dataloader.py          # DataLoader with batching
│   │
│   ├── models/                    # Neural network architectures
│   │   ├── __init__.py
│   │   ├── block2vec.py           # Skip-gram model for block embeddings
│   │   ├── encoder.py             # 3D convolutional encoder
│   │   ├── decoder.py             # 3D transposed conv decoder
│   │   ├── quantizer.py           # Vector quantization layer
│   │   ├── vqvae.py               # Complete VQ-VAE model
│   │   ├── text_encoder.py        # CLIP integration + mapping network
│   │   └── transformer_prior.py   # Autoregressive code generation
│   │
│   ├── training/                  # Training logic
│   │   ├── __init__.py
│   │   ├── train_block2vec.py     # Block2Vec training script
│   │   ├── train_vqvae.py         # VQ-VAE training script
│   │   ├── train_text_encoder.py  # Text conditioning training
│   │   ├── losses.py              # Loss function implementations
│   │   └── callbacks.py           # Training callbacks (checkpoints, logging)
│   │
│   ├── generation/                # Structure generation
│   │   ├── __init__.py
│   │   ├── generator.py           # Main generation pipeline
│   │   ├── sampler.py             # Sampling strategies (random, top-k, etc.)
│   │   └── interpolation.py       # Latent space interpolation
│   │
│   ├── export/                    # Export to Minecraft formats
│   │   ├── __init__.py
│   │   ├── to_schematic.py        # Export to .schematic (WorldEdit)
│   │   ├── to_nbt.py              # Export to .nbt (vanilla structures)
│   │   ├── to_litematic.py        # Export to .litematic (Litematica mod)
│   │   └── to_mcfunction.py       # Export to .mcfunction (commands)
│   │
│   └── utils/                     # Utility functions
│       ├── __init__.py
│       ├── visualization.py       # Visualize structures and embeddings
│       ├── metrics.py             # Evaluation metrics
│       └── helpers.py             # Misc helper functions
│
├── scripts/                       # Runnable scripts
│   ├── prepare_data.py            # Run full data preparation pipeline
│   ├── train.py                   # Main training entry point
│   ├── generate.py                # Generate structures from CLI
│   ├── evaluate.py                # Evaluate model quality
│   ├── visualize_embeddings.py    # Create t-SNE plots of block2vec
│   └── export_to_minecraft.py     # Batch export generated structures
│
├── notebooks/                     # Jupyter notebooks for exploration
│   ├── 01_explore_schematics.ipynb    # Analyze schematic collection
│   ├── 02_block2vec_analysis.ipynb    # Visualize block embeddings
│   ├── 03_vqvae_reconstruction.ipynb  # Test reconstruction quality
│   ├── 04_generation_examples.ipynb   # Generate and visualize structures
│   └── 05_interpolation_demo.ipynb    # Latent space interpolation demos
│
├── checkpoints/                   # Saved model weights
│   ├── block2vec/
│   │   ├── epoch_50.pt            # Final block2vec model
│   │   └── best.pt                # Best validation performance
│   │
│   ├── vqvae/
│   │   ├── epoch_100.pt           # Final VQ-VAE model
│   │   ├── best.pt                # Best reconstruction loss
│   │   └── codebook.pt            # Saved codebook vectors
│   │
│   └── text_encoder/
│       └── best.pt                # Text conditioning model
│
├── logs/                          # Training logs
│   ├── tensorboard/               # TensorBoard event files
│   │   ├── block2vec/
│   │   ├── vqvae/
│   │   └── text_encoder/
│   │
│   └── training_logs/             # Text logs
│       ├── block2vec.log
│       ├── vqvae.log
│       └── text_encoder.log
│
├── outputs/                       # Generated structures
│   ├── generated/                 # Raw generated numpy arrays
│   ├── schematics/                # Exported .schematic files
│   ├── nbt/                       # Exported .nbt files
│   └── visualizations/            # Rendered images of structures
│
└── tests/                         # Unit tests
    ├── __init__.py
    ├── test_parser.py             # Test schematic parsing
    ├── test_vocabulary.py         # Test block vocabulary
    ├── test_models.py             # Test model forward passes
    └── test_export.py             # Test export functions
```

### Folder Purposes

| Folder | Purpose |
|--------|---------|
| `configs/` | Store settings separately from code. Change hyperparameters without editing code. |
| `data/raw/` | Original untouched schematic files. Never modify these. |
| `data/processed/` | Cleaned, normalized data ready for training. |
| `data/vocabulary/` | Mappings between block names and numeric IDs. |
| `data/embeddings/` | Trained block2vec embeddings saved for reuse. |
| `src/data/` | Code for loading and processing data. |
| `src/models/` | Neural network architecture definitions. |
| `src/training/` | Training loops and loss functions. |
| `src/generation/` | Code for generating new structures. |
| `src/export/` | Convert numpy arrays to Minecraft file formats. |
| `src/utils/` | Helper functions used across the project. |
| `scripts/` | Entry points - run these from command line. |
| `notebooks/` | Interactive exploration and demos. |
| `checkpoints/` | Saved model weights during/after training. |
| `logs/` | Training progress logs for debugging and monitoring. |
| `outputs/` | Generated structures and visualizations. |
| `tests/` | Automated tests to verify code works correctly. |

---

## Tools and Libraries

### Core Deep Learning

| Tool | Version | Role | Simple Explanation |
|------|---------|------|-------------------|
| **PyTorch** | ≥2.0 | Deep learning framework | The main library for building neural networks. Like LEGO for AI - snap together layers to build models. |
| **PyTorch Lightning** | ≥2.0 | Training framework | Simplifies training loops, handles checkpoints and logging automatically. Less repetitive code. |
| **torchvision** | ≥0.15 | Vision utilities | Common image transforms. We adapt some for 3D. |

### Data Processing

| Tool | Version | Role | Simple Explanation |
|------|---------|------|-------------------|
| **NumPy** | ≥1.24 | Numerical arrays | Store and manipulate 3D voxel grids. Like Excel for programmers - grids of numbers with fast math. |
| **nbtlib** | ≥2.0 | NBT file parsing | Read/write Minecraft's special file format. Translator that speaks Minecraft's file language. |

### Text Understanding (Phase 4)

| Tool | Version | Role | Simple Explanation |
|------|---------|------|-------------------|
| **transformers** | ≥4.30 | Pre-trained models | Load CLIP and other AI models. HuggingFace's library of ready-to-use AI. |
| **CLIP** | - | Text-image AI | Already trained to connect text to visuals. Understands "medieval castle" means stone, towers, old. |

### Visualization

| Tool | Version | Role | Simple Explanation |
|------|---------|------|-------------------|
| **Matplotlib** | ≥3.7 | Basic plotting | Plot training curves and simple graphs. The standard Python plotting library. |
| **Plotly** | ≥5.15 | Interactive 3D plots | Visualize 3D structures you can rotate and zoom. Much better than static images. |
| **scikit-learn** | ≥1.2 | ML utilities | t-SNE for visualizing embeddings in 2D. Also basic ML algorithms and metrics. |
| **TensorBoard** | ≥2.13 | Training dashboard | Real-time training visualization. Like a car dashboard showing if training is going well. |

### Configuration & Utilities

| Tool | Version | Role | Simple Explanation |
|------|---------|------|-------------------|
| **PyYAML** | ≥6.0 | Config files | Load settings from .yaml files. Cleaner than hardcoding numbers in code. |
| **tqdm** | ≥4.65 | Progress bars | Show "Processing: 45% [████████░░░░░░░░░░]". Makes waiting less mysterious. |
| **click** | ≥8.1 | CLI arguments | Parse command-line arguments like `python train.py --epochs 100`. |

### Development & Testing

| Tool | Version | Role | Simple Explanation |
|------|---------|------|-------------------|
| **pytest** | ≥7.3 | Testing | Write and run automated tests. Ensures code changes don't break things. |
| **Jupyter** | ≥1.0 | Notebooks | Interactive coding environment. Great for exploration and demos. |

### Optional / Advanced

| Tool | Version | Role | When to Use |
|------|---------|------|-------------|
| **einops** | ≥0.6 | Tensor reshaping | Makes dimension manipulation readable. Use if you want cleaner code. |
| **Weights & Biases** | ≥0.15 | Experiment tracking | More advanced than TensorBoard. Use for serious hyperparameter tuning. |
| **MinkowskiEngine** | ≥0.5 | Sparse 3D convolutions | Handle huge structures efficiently. Use if memory becomes a problem. |
| **accelerate** | ≥0.20 | Distributed training | Multi-GPU training. Use if you have multiple GPUs. |

---

### How Tools Work Together

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DATA FLOW                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  .schematic files                                                        │
│        │                                                                 │
│        ▼                                                                 │
│  ┌──────────┐     ┌─────────┐     ┌──────────────┐                      │
│  │ nbtlib   │ ──▶ │ NumPy   │ ──▶ │ PyTorch      │                      │
│  │ (parse)  │     │ (store) │     │ Dataset      │                      │
│  └──────────┘     └─────────┘     └──────────────┘                      │
│                                          │                               │
│                                          ▼                               │
│                                   ┌──────────────┐                      │
│                                   │ PyTorch      │                      │
│                                   │ DataLoader   │                      │
│                                   └──────────────┘                      │
│                                          │                               │
│                                          ▼                               │
│  ┌────────────────────────────────────────────────────────────┐         │
│  │                    TRAINING                                 │         │
│  │  ┌──────────┐    ┌──────────┐    ┌──────────────────────┐  │         │
│  │  │ PyTorch  │───▶│ PyTorch  │───▶│ PyTorch Lightning    │  │         │
│  │  │ Models   │    │ Losses   │    │ (training loop)      │  │         │
│  │  └──────────┘    └──────────┘    └──────────────────────┘  │         │
│  │                                          │                  │         │
│  │                                          ▼                  │         │
│  │                                   ┌─────────────┐          │         │
│  │                                   │ TensorBoard │          │         │
│  │                                   │ (monitoring)│          │         │
│  │                                   └─────────────┘          │         │
│  └────────────────────────────────────────────────────────────┘         │
│                                          │                               │
│                                          ▼                               │
│                                   ┌──────────────┐                      │
│                                   │ Checkpoints  │                      │
│                                   │ (.pt files)  │                      │
│                                   └──────────────┘                      │
│                                          │                               │
│                                          ▼                               │
│  ┌────────────────────────────────────────────────────────────┐         │
│  │                    GENERATION                               │         │
│  │  ┌──────────────┐    ┌──────────┐    ┌────────────────┐    │         │
│  │  │ Load Model   │───▶│ Generate │───▶│ Export         │    │         │
│  │  │ (PyTorch)    │    │ (NumPy)  │    │ (nbtlib)       │    │         │
│  │  └──────────────┘    └──────────┘    └────────────────┘    │         │
│  └────────────────────────────────────────────────────────────┘         │
│                                          │                               │
│                                          ▼                               │
│                               .schematic / .nbt files                    │
│                               (use in Minecraft!)                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### Requirements.txt

```
# Core Deep Learning
torch>=2.0.0
pytorch-lightning>=2.0.0
torchvision>=0.15.0

# Data Processing
numpy>=1.24.0
nbtlib>=2.0.0

# Text Understanding
transformers>=4.30.0

# Visualization
matplotlib>=3.7.0
plotly>=5.15.0
scikit-learn>=1.2.0
tensorboard>=2.13.0

# Configuration & Utilities
pyyaml>=6.0
tqdm>=4.65.0
click>=8.1.0

# Development
pytest>=7.3.0
jupyter>=1.0.0
ipywidgets>=8.0.0

# Optional (uncomment if needed)
# einops>=0.6.0
# wandb>=0.15.0
# accelerate>=0.20.0
```

---

---

## The Core Problem

Minecraft has **hundreds of block types**. Using one-hot encoding (a vector of 0s with one 1) creates massive tensors:

- 71 block types → 71-dimensional vector per voxel
- A 50×50×50 structure = 125,000 voxels × 71 = **8.8 million values**

This is computationally expensive and doesn't capture semantic relationships between blocks (e.g., oak_planks and spruce_planks are functionally similar).

### Memory Comparison

| Encoding Method | Village Example (71 tokens) | Memory |
|-----------------|----------------------------|--------|
| One-hot encoding | 71 dimensions per voxel | 154.23 MB |
| block2vec (32-dim) | 32 dimensions per voxel | 69.51 MB |

---

## Block Embeddings (block2vec)

Inspired by **word2vec from NLP**, World-GAN introduced **block2vec** to create dense, learned representations of Minecraft blocks.

### The Intuition

Just like words that appear in similar contexts have similar meanings ("king" and "queen"), blocks that appear near each other in builds have similar functions:

```
"stone_bricks" often appears next to: stone, cobblestone, mossy_stone_bricks
"oak_planks" often appears next to: oak_log, oak_stairs, oak_door
```

### Training Process

1. **Context Window**: For each block in training data, examine its 3D neighbors (6 adjacent blocks or 26 in a 3×3×3 cube)

2. **Skip-gram Model Architecture**:
   ```
   Input Layer:  One-hot encoded block (N dimensions, N = number of block types)
        ↓
   Hidden Layer: 32 neurons (this becomes the embedding)
        ↓
   Output Layer: Predict probability of each neighbor block (N dimensions)
   ```

3. **Training Task**: Given a center block, predict what blocks surround it

4. **Extract Embeddings**: After training, the hidden layer weights become the block embeddings

### Handling Imbalanced Data

Air blocks dominate Minecraft structures (~75% of voxels). To prevent the model from just learning "predict air", apply subsampling:

```
P(block_i) = sqrt(f(block_i) / 0.001 + 1) × 0.001 / f(block_i)
```

Where `f(block_i)` is the frequency of block_i in the dataset.

### What Block2Vec Learns

After training, semantically similar blocks cluster together in the embedding space:

```
32-Dimensional Embedding Space (visualized in 2D):

     oak_planks ●  ● spruce_planks
                   ● dark_oak_planks
                   ● birch_planks

     cobblestone ●  ● stone_bricks
                    ● mossy_cobblestone
                    ● cracked_stone_bricks

     glass ●  ● glass_pane
              ● white_stained_glass

     oak_door ●  ● iron_door
                 ● spruce_door
```

### Key Benefit

The model learns that `oak_planks` and `spruce_planks` are interchangeable in most architectural contexts, so they're close in latent space. This enables:
- **Style transfer**: Swap all wood types by moving through embedding space
- **Rare block generation**: Rare blocks positioned near common similar blocks get generated appropriately

---

## VAE Architecture

Variational Autoencoders learn to compress structures into a latent space and reconstruct them.

### Encoder

```
Input: 3D voxel grid with block embeddings
       Shape: [batch, channels, depth, height, width]
       Example: [1, 32, 32, 32, 32] for 32³ structure with 32-dim embeddings
              ↓
       3D Convolutions (extract spatial patterns)
       Conv3D(32, 64, kernel=4, stride=2) → [1, 64, 16, 16, 16]
       Conv3D(64, 128, kernel=4, stride=2) → [1, 128, 8, 8, 8]
       Conv3D(128, 256, kernel=4, stride=2) → [1, 256, 4, 4, 4]
              ↓
       Flatten → Dense layers
       Flatten → [1, 16384]
       Dense(16384, 512)
              ↓
       Split into μ (mean) and σ (std dev)
       Dense(512, 128) → μ
       Dense(512, 128) → log(σ²)
```

### Reparameterization Trick

To enable backpropagation through sampling:
```
z = μ + σ × ε    where ε ~ N(0, 1)
```

### Decoder

```
Input: Latent vector z
       Shape: [1, 128]
              ↓
       Dense → Reshape to 3D
       Dense(128, 16384)
       Reshape → [1, 256, 4, 4, 4]
              ↓
       3D Transposed Convolutions (upscale)
       ConvTranspose3D(256, 128, kernel=4, stride=2) → [1, 128, 8, 8, 8]
       ConvTranspose3D(128, 64, kernel=4, stride=2) → [1, 64, 16, 16, 16]
       ConvTranspose3D(64, 32, kernel=4, stride=2) → [1, 32, 32, 32, 32]
              ↓
       Output: Continuous embeddings per voxel
       Shape: [1, 32, 32, 32, 32]
              ↓
       Nearest-neighbor lookup in block2vec codebook
              ↓
       Final: Discrete Minecraft blocks
       Shape: [32, 32, 32] block IDs
```

### VAE Loss Function

```
L = L_reconstruction + β × L_KL

L_reconstruction = MSE(input_embeddings, output_embeddings)
                   or CrossEntropy(input_blocks, output_blocks)

L_KL = -0.5 × sum(1 + log(σ²) - μ² - σ²)
```

The KL divergence term encourages the latent space to be smooth and continuous.

---

## VQ-VAE (Discrete Latent Space)

Vector Quantized VAE uses a **discrete** latent space instead of continuous, which better matches Minecraft's discrete nature.

### Key Difference from VAE

| Aspect | Regular VAE | VQ-VAE |
|--------|-------------|--------|
| Latent values | Continuous (any real number) | Discrete (indices into codebook) |
| Sampling | z = μ + σε | z = nearest codebook entry |
| Latent space | Gaussian distribution | Learned codebook vectors |

### The Codebook

```python
# Codebook: K learned embedding vectors (e.g., K=512)
codebook = [e_0, e_1, e_2, ..., e_511]  # Each e_i is D-dimensional

# During encoding:
z_continuous = encoder(input)  # Shape: [B, D, H, W, L]

# Quantization: Find nearest codebook entry for each spatial position
z_quantized = []
for each position (h, w, l):
    distances = ||z_continuous[:, :, h, w, l] - codebook||²
    nearest_idx = argmin(distances)
    z_quantized.append(codebook[nearest_idx])
```

### VQ-VAE Loss Function

```
L = L_reconstruction + L_codebook + β × L_commitment

L_reconstruction = ||x - decoder(z_quantized)||²
L_codebook = ||sg[z_continuous] - codebook||²     # Move codebook toward encoder output
L_commitment = ||z_continuous - sg[codebook]||²   # Keep encoder close to codebook

sg[] = stop gradient (no backprop through this term)
```

### Why VQ-VAE for Minecraft?

1. **Discrete outputs**: Minecraft blocks are discrete, not continuous
2. **Codebook as "style atoms"**: Each codebook entry can represent a building pattern
3. **Better generation quality**: Avoids blurry outputs common in VAE
4. **Compositionality**: Can combine codebook entries in new ways

---

## GAN Architecture (World-GAN)

World-GAN uses a 3D Generative Adversarial Network with multi-scale training.

### Generator Architecture

```
Input: Random noise z + upsampled previous scale
       z ~ N(0, 1), shape [batch, noise_dim]
              ↓
       Reshape to 3D spatial
       Dense → Reshape to [batch, channels, 4, 4, 4]
              ↓
       3D Convolution blocks (not transposed - maintains scale)
       Conv3D → BatchNorm → LeakyReLU
       Conv3D → BatchNorm → LeakyReLU
       Conv3D → BatchNorm → LeakyReLU
              ↓
       Output: Generated embeddings
       Shape: [batch, embed_dim, D, H, W]
```

### Discriminator Architecture

```
Input: Real or generated structure
       Shape: [batch, embed_dim, D, H, W]
              ↓
       3D Convolution blocks
       Conv3D → BatchNorm → LeakyReLU
       Conv3D → BatchNorm → LeakyReLU
       Conv3D → BatchNorm → LeakyReLU
              ↓
       Output: Real/Fake score (per patch)
       Shape: [batch, 1, d, h, w]
```

### Multi-Scale Training

World-GAN trains at multiple scales (inspired by SinGAN):

```
Scale 0: 8×8×8     (coarse structure)
Scale 1: 16×16×16  (medium details)
Scale 2: 32×32×32  (fine details)
Scale 3: 64×64×64  (full resolution)

Training order: Scale 0 → Scale 1 → Scale 2 → Scale 3

At each scale n:
x̃_n = upsample(x̃_{n-1}) + G_n(z_n + upsample(x̃_{n-1}))
```

### Loss Function (WGAN-GP)

```
L_D = E[D(fake)] - E[D(real)] + λ × GP
L_G = -E[D(fake)]

GP = gradient penalty = E[(||∇D(interpolated)||₂ - 1)²]
```

### Key Innovation: Single Example Training

World-GAN can train from a **single example structure**:
- Extract many overlapping patches from one large structure
- Train discriminator to recognize patches as "real"
- Generator learns to create new patches with same style

---

## Latent Space Interpolation

Once structures are encoded as latent vectors, we can blend them.

### Linear Interpolation

```python
# Encode two structures
z_castle = encoder(castle_voxels)  # [0.5, -0.3, 0.8, ...]
z_house = encoder(house_voxels)    # [-0.2, 0.7, 0.1, ...]

# Interpolate (α controls blend ratio)
for α in [0.0, 0.25, 0.5, 0.75, 1.0]:
    z_blend = (1 - α) * z_castle + α * z_house
    blended_structure = decoder(z_blend)
```

### Spherical Interpolation (SLERP)

Better for high-dimensional spaces:

```python
def slerp(z1, z2, α):
    """Spherical linear interpolation"""
    z1_norm = z1 / ||z1||
    z2_norm = z2 / ||z2||
    omega = arccos(dot(z1_norm, z2_norm))

    return (sin((1-α)*omega) * z1 + sin(α*omega) * z2) / sin(omega)
```

### Attribute Manipulation

If latent dimensions are disentangled:

```python
z = encoder(structure)

# Find "size" direction in latent space
z_large = encoder(large_structures).mean(axis=0)
z_small = encoder(small_structures).mean(axis=0)
size_direction = z_large - z_small

# Make structure bigger
z_bigger = z + 0.5 * size_direction
bigger_structure = decoder(z_bigger)
```

---

## Text-to-Structure Pipeline

Combining text understanding with structure generation.

### Architecture Overview

```
"medieval stone castle with towers"
            ↓
    ┌───────────────────────┐
    │    Text Encoder       │
    │   (CLIP or BERT)      │
    └───────────────────────┘
            ↓
    Text embedding [512 dimensions]
            ↓
    ┌───────────────────────┐
    │   Mapping Network     │
    │   (MLP: 512 → 256 → 128)│
    └───────────────────────┘
            ↓
    Latent vector z [128 dimensions]
            ↓
    ┌───────────────────────┐
    │    3D Decoder         │
    │ (VAE/VQ-VAE decoder)  │
    └───────────────────────┘
            ↓
    Block embeddings grid [32×32×32×32]
            ↓
    ┌───────────────────────┐
    │  Nearest-Neighbor     │
    │  Block Lookup         │
    └───────────────────────┘
            ↓
    Minecraft blocks [32×32×32]
```

### Training with CLIP

```python
# Contrastive learning between text and structures
text_features = CLIP.encode_text(descriptions)      # [batch, 512]
structure_features = structure_encoder(voxels)      # [batch, 512]

# Compute similarity matrix
similarity = text_features @ structure_features.T   # [batch, batch]

# Contrastive loss: matching pairs should have high similarity
labels = torch.arange(batch_size)
loss = CrossEntropy(similarity, labels) + CrossEntropy(similarity.T, labels)
```

---

## Existing Datasets

### Available Datasets

| Dataset | Size | Contents | Link |
|---------|------|----------|------|
| 3D-Craft | ~1000 houses | Block-by-block build sequences | [Facebook Research](https://github.com/facebookresearch/voxelcnn) |
| MineRL | 60M+ frames | Gameplay demonstrations | [MineRL](https://minerl.io/) |
| GDMC Submissions | Varies | Settlement generators | [GDMC](https://gendesignmc.wikidot.com/) |
| Planet Minecraft | 100k+ | Tagged schematics (scraping required) | [PMC](https://www.planetminecraft.com/) |

### Dataset Requirements

For training block2vec:
- Many structures with diverse block usage
- Context windows from 3D neighborhoods

For training VAE/GAN:
- Consistent structure sizes (or padding)
- Aligned orientations (or data augmentation)
- Labels for conditional generation (optional)

---

## Key Papers & Resources

### Core Papers

| Paper | Year | Contribution |
|-------|------|--------------|
| [World-GAN](https://arxiv.org/abs/2106.10155) | 2021 | block2vec + 3D GAN from single example |
| [VoxelCNN](https://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_Order-Aware_Generative_Modeling_Using_the_3D-Craft_Dataset_ICCV_2019_paper.pdf) | 2019 | Order-aware generation, 3D-Craft dataset |
| [STEVE-1](https://arxiv.org/abs/2306.00937) | 2023 | Text-to-behavior with MineCLIP |
| [DreamCraft](https://arxiv.org/abs/2404.15538) | 2024 | Text-to-3D with NeRF in Minecraft |

### Related Techniques

| Paper | Technique | Relevance |
|-------|-----------|-----------|
| [VQ-VAE](https://arxiv.org/abs/1711.00937) | Discrete latent space | Better for discrete blocks |
| [VQ-VAE-2](https://arxiv.org/abs/1906.00446) | Hierarchical VQ-VAE | Multi-scale generation |
| [DALL-E](https://arxiv.org/abs/2102.12092) | Text-to-image with VQ-VAE | Text conditioning approach |
| [SinGAN](https://arxiv.org/abs/1905.01164) | Single image GAN | World-GAN's inspiration |

### Code Repositories

- [World-GAN](https://github.com/Mawiszus/World-GAN)
- [VoxelCNN / 3D-Craft](https://github.com/facebookresearch/voxelcnn)
- [MineRL](https://github.com/minerllabs/minerl)

---

---

# Detailed Training Pipeline

This section provides a step-by-step guide with simple explanations for each phase.

## Overview: The 5 Phases

```
Phase 1: Data Preparation
         Parse schematics → Clean data → Create training sets
                                    ↓
Phase 2: Block2Vec Training
         Learn which blocks are "similar" to each other
                                    ↓
Phase 3: VQ-VAE Training
         Learn to compress structures into codes and reconstruct them
                                    ↓
Phase 4: Text Encoder Training (Optional)
         Learn to connect text descriptions to structures
                                    ↓
Phase 5: Generation & Deployment
         Generate new structures from text/random input
```

---

## Phase 1: Data Preparation

### Step 1.1: Parse Schematic Files

**What we're doing**: Converting `.schematic` files into 3D arrays of block IDs.

```
Input:  castle.schematic (binary file)
Output: 3D numpy array, shape [50, 30, 50]
        Where each cell contains a block ID (0=air, 1=stone, 2=dirt, etc.)
```

**Simple explanation**: A schematic file is like a ZIP file containing a 3D grid of blocks. We unpack it into a format Python can work with - just a big 3D grid of numbers where each number represents a block type.

### Step 1.2: Create Block Vocabulary

**What we're doing**: Making a list of all unique blocks and giving each one a number.

```
Block Vocabulary:
  0: minecraft:air
  1: minecraft:stone
  2: minecraft:dirt
  3: minecraft:oak_planks
  4: minecraft:cobblestone
  ...
  150: minecraft:oak_door
```

**Simple explanation**: Like making a dictionary - "stone = 1, dirt = 2, etc." This lets us work with numbers instead of text, which computers are much faster at processing.

### Step 1.3: Normalize Structure Sizes

**What we're doing**: Making all structures the same size so we can batch them together.

```
Options:
  A) Pad smaller structures with air to reach max size
  B) Crop larger structures to a fixed size
  C) Split large structures into overlapping chunks

Recommended: 32×32×32 chunks (32,768 blocks each)
```

**Simple explanation**: Neural networks need inputs of the same size. If one castle is 100×50×80 and another is 20×15×20, we need to standardize them - like resizing photos to all be the same dimensions.

### Step 1.4: Create Training Splits

```
Total schematics: 10,963

Training set:   8,770 (80%) - Used to train the model
Validation set: 1,096 (10%) - Used to check if model is learning well
Test set:       1,097 (10%) - Used only at the end to evaluate final model
```

**Simple explanation**: We hide some data from the model during training, then test it on that hidden data to make sure it actually learned general patterns (didn't just memorize the specific examples it saw).

---

## Phase 2: Block2Vec Training

### What Is Block2Vec? (Simple Explanation)

Imagine teaching a computer that "oak_planks" and "spruce_planks" are similar (both are wood for building), but "oak_planks" and "lava" are very different (you don't build houses with lava!).

Block2Vec learns these relationships automatically by looking at which blocks appear near each other in real builds. If oak_planks usually appears next to oak_logs and oak_doors, and spruce_planks also appears next to similar things, the model learns they're related.

### Step 2.1: Extract Training Pairs

**What we're doing**: For each block in our structures, we look at its neighbors and create (center_block, neighbor_block) pairs.

```
Example: A 3×3×3 section of a house wall

Center block: air (inside the room)
Neighbors: stone (walls around it)

Training pairs generated:
  (air, stone), (air, stone), (air, stone),
  (air, stone), (air, stone), (air, stone)
```

**Simple explanation**: We're teaching the model "when you see air inside a house, stone walls are often nearby." After seeing millions of examples, it learns patterns like "doors are usually next to wood planks, not floating in the sky."

### Step 2.2: Train Skip-Gram Model

**The neural network** (very simple - just 2 layers):

```
Layer 1 (Input):   One block ID (e.g., "oak_door")
                         ↓
Layer 2 (Hidden):  32 neurons - this becomes our embedding!
                         ↓
Layer 3 (Output):  Probability for each block type
                   "What blocks are likely neighbors?"
```

**Training process in plain English**:
1. Show the model a block (say, "oak_door")
2. Ask it to guess what blocks are usually nearby
3. Tell it the right answer ("oak_planks was actually next to it")
4. It adjusts its understanding slightly
5. Repeat millions of times
6. Eventually it "gets" that doors go with wood, not with lava

### Step 2.3: Extract Embeddings

**What we're doing**: After training, we save the 32 numbers that represent each block.

```
Trained embeddings (simplified example):
  minecraft:air         → [0.01, -0.02, 0.00, ..., 0.03]   (32 numbers)
  minecraft:oak_planks  → [0.22, 0.78, 0.45, ..., -0.12]
  minecraft:spruce_planks → [0.25, 0.81, 0.42, ..., -0.09]  ← Very similar to oak!
  minecraft:lava        → [-0.89, 0.12, 0.67, ..., 0.91]   ← Very different!
```

**Simple explanation**: These 32 numbers are like a "fingerprint" or "personality profile" for each block. Similar blocks have similar fingerprints. We'll use these fingerprints in the next phase instead of just block IDs.

---

## Phase 3: VQ-VAE Training

### What Is VQ-VAE? (Simple Explanation)

Think of it like this:
- **Encoder**: A machine that looks at a full building and writes down a short "recipe" for it
- **Codebook**: A cookbook with 512 pre-defined "ingredient patterns"
- **Decoder**: A machine that reads the recipe and rebuilds the structure

The "VQ" (Vector Quantized) part means: instead of allowing any possible recipe, we force it to use only ingredients from our 512-item cookbook. This makes the recipes more consistent and meaningful.

**Analogy**: Instead of describing a house as "slightly warm brownish wood with a bit of gray stone and some transparent glass", VQ forces you to say "Pattern #42 (cozy cabin) + Pattern #156 (stone foundation)".

### Step 3.1: Prepare Input Data

**What we're doing**: Converting block IDs into the rich embeddings we created in Phase 2.

```
Before: structure[x,y,z] = 3  (just the ID number for oak_planks)
After:  structure[x,y,z] = [0.22, 0.78, 0.45, ..., -0.12]  (32 descriptive numbers)
```

**Simple explanation**: Instead of just knowing "this is block #3", we now have 32 numbers describing the "essence" of that block - what it's used for, what it's similar to, etc. This richer description helps the neural network learn better patterns.

### Step 3.2: Encoder (Compression)

**What it does**: Takes a full 32×32×32 structure and squeezes it down to a tiny representation.

```
Input:  32×32×32 structure (32,768 blocks)
          ↓
        Shrink by half (3D convolution, stride 2)
          ↓
        16×16×16 (4,096 positions)
          ↓
        Shrink by half again
          ↓
        8×8×8 (512 positions)
          ↓
        Shrink one more time
          ↓
Output: 4×4×4 (64 positions)

Compression: 32,768 blocks → 64 code positions (512× smaller!)
```

**Simple explanation of 3D Convolution**: Imagine sliding a small 3D cube (like a 4×4×4 Rubik's cube) across your structure. At each position, it looks at all the blocks inside that cube and produces a single summary number. "Stride 2" means we skip every other position, halving the size.

### Step 3.3: Vector Quantization (The "VQ" Part)

**What it does**: Takes the encoder's continuous output and snaps it to the nearest entry in our cookbook.

```
Codebook: 512 learned patterns, each described by 256 numbers
          Pattern #0:   [0.12, -0.34, 0.56, ...]  "foundation corner"
          Pattern #1:   [0.78, 0.23, -0.11, ...]  "wooden wall section"
          Pattern #2:   [0.45, 0.67, 0.89, ...]  "window area"
          ...
          Pattern #511: [-0.22, 0.91, 0.33, ...] "rooftop peak"

For each of the 64 positions from the encoder:
  1. Look at the 256 numbers the encoder produced
  2. Find which cookbook pattern is most similar
  3. Write down that pattern's number (0-511)
```

**Simple explanation**: The encoder might output something like [0.13, -0.32, 0.58, ...], and the VQ step says "that's closest to Pattern #0 (foundation corner), so let's call it Pattern #0." This discretization (forcing choices from a menu) makes the codes more meaningful and easier to work with.

### Step 3.4: Decoder (Reconstruction)

**What it does**: Takes the small 4×4×4 code grid and expands it back to a full structure.

```
Input:  4×4×4 codes (looked up from codebook)
          ↓
        Expand by 2× (3D transposed convolution)
          ↓
        8×8×8
          ↓
        Expand by 2× again
          ↓
        16×16×16
          ↓
        Expand one more time
          ↓
Output: 32×32×32 embeddings
          ↓
        Find nearest block for each position
          ↓
Final:  32×32×32 block IDs (actual Minecraft blocks!)
```

**Simple explanation**: The decoder is like running the encoder backwards. It takes the compressed recipe and "inflates" it back to full size, adding details at each step.

### Step 3.5: Training (Teaching the Model)

**The goal**: Make the reconstructed structure match the original as closely as possible.

```
1. Take a real structure from our dataset
2. Compress it with the encoder → small code
3. Quantize the code → snap to cookbook
4. Decompress with decoder → reconstructed structure
5. Compare reconstruction to original
6. Calculate error: "How many blocks are wrong?"
7. Adjust the encoder, decoder, and codebook to reduce errors
8. Repeat millions of times
```

**What the model learns over time**:
- Epoch 1: Outputs look like random noise
- Epoch 10: Basic shapes are recognizable (boxy structures)
- Epoch 50: Most blocks are correct, some details wrong
- Epoch 100: Nearly perfect reconstruction

### Step 3.6: The Three Losses (What We're Optimizing)

1. **Reconstruction Loss**: "How different is the rebuilt structure from the original?"
   - Measures block-by-block accuracy
   - This is the main goal

2. **Codebook Loss**: "Are our cookbook patterns useful?"
   - Moves cookbook patterns toward what the encoder actually produces
   - Makes the cookbook adapt to our data

3. **Commitment Loss**: "Is the encoder being consistent?"
   - Encourages encoder to produce outputs close to existing cookbook patterns
   - Prevents the encoder from drifting away from the codebook

---

## Phase 4: Text Encoder Training (Optional)

### What This Phase Does (Simple Explanation)

We want to type "medieval castle with towers" and get a castle. This phase teaches the model to understand text and connect words to the visual patterns it learned in Phase 3.

### Step 4.1: Create Text-Structure Pairs

**What we need**: Descriptions for our structures.

```
castle_001.schematic → "medieval stone castle with towers"
house_042.schematic → "small wooden cottage with garden"
tower_007.schematic → "tall wizard tower made of stone bricks"
```

**Ways to get descriptions**:
- **Manual labeling**: You write descriptions (accurate but slow)
- **From filenames**: "medieval_castle_v2.schematic" → extract "medieval castle"
- **Existing tags**: Use tags from Planet Minecraft or other sources
- **AI-generated**: Use GPT/Claude to describe structures

### Step 4.2: Using CLIP (Pre-trained Text Understanding)

**What is CLIP?**: A model trained by OpenAI that already understands how text relates to images. We borrow its text understanding.

```
"medieval castle with towers"
         ↓
    CLIP (already trained on millions of image-text pairs)
         ↓
    512 numbers representing the meaning of this text
         ↓
    Our small mapping network (we train this part)
         ↓
    256 numbers that match our structure encoding format
```

**Simple explanation**: CLIP already knows that "medieval" relates to stone, towers, old architecture, etc. We just need to train a small translator to convert CLIP's understanding into our structure format.

### Step 4.3: Training the Connection

**What we're teaching**:
- "medieval castle" → should produce similar numbers to encoding an actual medieval castle
- "wooden cottage" → should produce similar numbers to encoding an actual wooden cottage

**The process**:
1. Take a text description and its matching structure
2. Get CLIP's understanding of the text
3. Get our VQ-VAE's understanding of the structure
4. Adjust the mapping network so these match up
5. Repeat for all text-structure pairs

---

## Phase 5: Generation & Deployment

### Step 5.1: Generate Random Structures

**Without text** (just exploring what the model learned):

```
1. Pick 64 random cookbook pattern numbers (0-511)
2. Arrange them in a 4×4×4 grid
3. Run the decoder to expand to 32×32×32
4. Convert embeddings to actual block IDs
5. Save as schematic file
```

**What you get**: Random but coherent structures - the model combines patterns it learned in ways it's never seen before.

### Step 5.2: Generate from Text

**With a text prompt**:

```
1. "medieval castle with towers"
         ↓
2. CLIP encodes this → 512 numbers
         ↓
3. Mapping network → 256 numbers
         ↓
4. Generate codes that match these numbers
   (This is the tricky part - needs a Transformer or diffusion model)
         ↓
5. Decoder expands codes → structure
         ↓
6. Save as schematic
```

### Step 5.3: Interpolation (Morphing Between Styles)

**Blending two structures**:

```
Structure A: Medieval castle  → Code A: [42, 156, 89, ...]
Structure B: Modern house     → Code B: [201, 34, 178, ...]

Blend 50%: [(42+201)/2, (156+34)/2, ...] → Decode → Medieval-modern hybrid!

Blend sequence:
  0%   = Pure castle
  25%  = Mostly castle, some modern elements
  50%  = Half and half
  75%  = Mostly modern, some castle elements
  100% = Pure modern house
```

### Step 5.4: Export to Minecraft

**Schematic format** (for WorldEdit):
```python
def save_schematic(blocks, filename):
    # blocks: 3D numpy array of block IDs
    # Create NBT structure
    # Save as .schematic file
```

**NBT Structure format** (for vanilla /place command):
```python
def save_nbt_structure(blocks, filename):
    # Convert to vanilla structure format
    # Save to server/world/generated/minecraft/structures/
    # Use with: /place template minecraft:filename
```

---

## Training Timeline

```
Week 1: Data Preparation
  Day 1-2: Write schematic parser
  Day 3-4: Build block vocabulary
  Day 5-7: Create data loaders, train/val/test splits

Week 2: Block2Vec
  Day 1-2: Extract neighbor pairs
  Day 3-4: Train embedding model
  Day 5-7: Evaluate and visualize embeddings

Week 3-4: VQ-VAE
  Day 1-3: Implement encoder
  Day 4-6: Implement decoder
  Day 7-10: Implement vector quantization
  Day 11-14: Train and debug

Week 5: Text Conditioning
  Day 1-2: Gather text descriptions
  Day 3-4: Integrate CLIP
  Day 5-7: Train mapping network

Week 6: Integration
  Day 1-2: Build generation pipeline
  Day 3-4: Export to Minecraft formats
  Day 5-7: Test in-game
```

---

## Glossary of ML Terms

| Term | Simple Explanation |
|------|-------------------|
| **Epoch** | One complete pass through all training data |
| **Batch** | A small group of examples processed together (e.g., 16 structures at once) |
| **Loss** | A number measuring how wrong the model is (lower = better) |
| **Gradient** | Direction to adjust weights to reduce loss |
| **Backpropagation** | Algorithm to calculate gradients for all weights |
| **Learning Rate** | How big of adjustments to make (too big = unstable, too small = slow) |
| **Overfitting** | Model memorizes training data but fails on new data |
| **Latent Space** | The compressed representation space (where codes live) |
| **Embedding** | A learned vector representation of something (block, word, etc.) |
| **Encoder** | Neural network that compresses input to latent space |
| **Decoder** | Neural network that expands latent space back to output |
| **Codebook** | Fixed set of vectors to quantize to (in VQ-VAE) |
| **Convolution** | Sliding window operation that detects patterns |
| **Stride** | How much the convolution window moves each step |

---

## Implementation Checklist

### Phase 1: Data Preparation
- [ ] Write schematic file parser (.schematic → numpy array)
- [ ] Build block vocabulary (block name → ID mapping)
- [ ] Handle special blocks (stairs, doors with orientations)
- [ ] Normalize sizes (pad/crop to 32×32×32)
- [ ] Create train/val/test splits (80/10/10)
- [ ] Build PyTorch DataLoader

### Phase 2: Block2Vec
- [ ] Extract (center, neighbor) pairs from structures
- [ ] Implement skip-gram model
- [ ] Add subsampling for frequent blocks (air)
- [ ] Train for 50 epochs
- [ ] Save embeddings to file
- [ ] Visualize with t-SNE (verify similar blocks cluster)

### Phase 3: VQ-VAE
- [ ] Implement 3D convolution encoder
- [ ] Implement codebook with 512 entries
- [ ] Implement vector quantization with straight-through gradient
- [ ] Implement 3D transposed convolution decoder
- [ ] Implement three losses (reconstruction, codebook, commitment)
- [ ] Train for 100 epochs
- [ ] Verify reconstruction quality (before/after comparison)

### Phase 4: Text Conditioning (Optional)
- [ ] Gather/generate text descriptions for structures
- [ ] Integrate CLIP text encoder
- [ ] Implement mapping network (MLP)
- [ ] Train mapping network
- [ ] Test text-to-structure generation

### Phase 5: Deployment
- [ ] Implement schematic export
- [ ] Implement NBT structure export
- [ ] Build generation API/script
- [ ] Integrate with Minecraft server
- [ ] Test generated structures in-game
