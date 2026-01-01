"""
Sparse Structure Dataset for Transformer-based models.

Converts dense 32x32x32 Minecraft structures to sparse sets of
(position, block_id) pairs, eliminating the 80% air block dominance
that plagues dense VQ-VAE approaches.

Key advantages:
- No air blocks in the representation (air is implicit)
- Each non-air block gets equal weight in training
- Directly uses Block2Vec embeddings as input/output targets
- Variable-length sequences that match actual structure complexity
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


# Air token IDs in the vocabulary
AIR_TOKENS: Set[int] = {102, 576, 3352}  # air, cave_air, void_air


class SparseStructureDataset(Dataset):
    """Dataset that converts dense structures to sparse (position, block) sets.

    Instead of representing a 32x32x32 grid with 32,768 values (80% air),
    this dataset extracts only non-air blocks as a set of (x, y, z, block_id)
    tuples.

    Args:
        data_dir: Directory containing H5 files
        embeddings_path: Path to Block2Vec embeddings .npy file
        vocab_path: Path to tok2block.json vocabulary file
        max_files: Optional limit on number of files (for debugging)
        max_blocks: Maximum non-air blocks per structure (for batching)
        cache_in_memory: If True, loads all structures into RAM
        augment: If True, apply random rotations and flips
        seed: Random seed for augmentation
    """

    def __init__(
        self,
        data_dir: str,
        embeddings_path: str,
        vocab_path: Optional[str] = None,
        max_files: Optional[int] = None,
        max_blocks: int = 2048,
        cache_in_memory: bool = False,
        augment: bool = False,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.max_blocks = max_blocks
        self.cache_in_memory = cache_in_memory
        self.augment = augment
        self.seed = seed
        self.rng = random.Random(seed)

        # Load Block2Vec embeddings
        self.embeddings = np.load(embeddings_path).astype(np.float32)
        self.embed_dim = self.embeddings.shape[1]
        self.vocab_size = self.embeddings.shape[0]
        print(f"Loaded embeddings: {self.embeddings.shape} (vocab={self.vocab_size}, dim={self.embed_dim})")

        # Load vocabulary for debugging/analysis
        self.tok2block: Optional[Dict[int, str]] = None
        if vocab_path and Path(vocab_path).exists():
            with open(vocab_path, 'r') as f:
                self.tok2block = {int(k): v for k, v in json.load(f).items()}

        # Find all H5 files
        self.h5_files = sorted(self.data_dir.glob("*.h5"))
        if not self.h5_files:
            raise ValueError(f"No H5 files found in {data_dir}")

        if max_files is not None:
            self.h5_files = self.h5_files[:max_files]

        print(f"Found {len(self.h5_files)} H5 files in {data_dir}")

        # Cache structures in memory if requested
        self._cache: Optional[List[np.ndarray]] = None
        if cache_in_memory:
            self._load_all_into_memory()

        # Precompute statistics
        self._compute_statistics()

    def _load_all_into_memory(self) -> None:
        """Load all structures into RAM for faster training."""
        print("Loading all structures into memory...")
        self._cache = []

        for i, h5_path in enumerate(self.h5_files):
            try:
                with h5py.File(h5_path, "r") as f:
                    key = list(f.keys())[0]
                    structure = f[key][:]
                    self._cache.append(structure.astype(np.int64))
            except Exception as e:
                print(f"Warning: Failed to load {h5_path}: {e}")
                continue

            if (i + 1) % 1000 == 0:
                print(f"  Loaded {i + 1}/{len(self.h5_files)} files...")

        print(f"Loaded {len(self._cache)} structures into memory")

    def _compute_statistics(self) -> None:
        """Compute dataset statistics for reporting."""
        sample_size = min(100, len(self))
        block_counts = []

        for i in range(sample_size):
            structure = self._load_structure(i)
            non_air_mask = ~np.isin(structure, list(AIR_TOKENS))
            block_counts.append(non_air_mask.sum())

        self.avg_blocks = np.mean(block_counts)
        self.max_observed_blocks = np.max(block_counts)
        self.min_observed_blocks = np.min(block_counts)

        print(f"Block count stats (sample of {sample_size}):")
        print(f"  Average: {self.avg_blocks:.1f}")
        print(f"  Min: {self.min_observed_blocks}, Max: {self.max_observed_blocks}")

    def _load_structure(self, idx: int) -> np.ndarray:
        """Load a single structure from disk or cache."""
        if self._cache is not None:
            return self._cache[idx]

        h5_path = self.h5_files[idx]
        with h5py.File(h5_path, "r") as f:
            key = list(f.keys())[0]
            structure = f[key][:]

        return structure.astype(np.int64)

    def _extract_sparse(
        self, structure: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract non-air blocks as sparse representation.

        Args:
            structure: Dense 3D array [D, H, W] of block IDs

        Returns:
            positions: [N, 3] array of (x, y, z) coordinates
            block_ids: [N] array of block token IDs
            embeddings: [N, embed_dim] array of block embeddings
        """
        # Find non-air positions
        non_air_mask = ~np.isin(structure, list(AIR_TOKENS))
        positions = np.argwhere(non_air_mask)  # [N, 3]

        # Get block IDs at those positions
        block_ids = structure[non_air_mask]  # [N]

        # Limit to max_blocks if needed (random sampling)
        n_blocks = len(block_ids)
        if n_blocks > self.max_blocks:
            indices = self.rng.sample(range(n_blocks), self.max_blocks)
            indices = sorted(indices)  # Keep spatial ordering
            positions = positions[indices]
            block_ids = block_ids[indices]

        # Get embeddings for these blocks
        embeddings = self.embeddings[block_ids]  # [N, embed_dim]

        return positions.astype(np.float32), block_ids, embeddings

    def _augment_sparse(
        self,
        positions: np.ndarray,
        block_ids: np.ndarray,
        grid_size: int = 32,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random augmentations to sparse representation.

        Augmentations:
        - Random rotation by 0, 90, 180, or 270 degrees around Y axis
        - Random horizontal flips

        Args:
            positions: [N, 3] coordinates
            block_ids: [N] block IDs (unchanged by spatial transforms)
            grid_size: Size of the grid (for flip coordinates)

        Returns:
            Augmented positions (block_ids unchanged)
        """
        positions = positions.copy()

        # Random rotation around Y axis (in XZ plane)
        k = self.rng.randint(0, 3)  # 0, 1, 2, or 3 * 90 degrees
        if k > 0:
            x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
            for _ in range(k):
                new_x = grid_size - 1 - z
                new_z = x.copy()
                x, z = new_x, new_z
            positions[:, 0] = x
            positions[:, 2] = z

        # Random horizontal flip in X direction
        if self.rng.random() > 0.5:
            positions[:, 0] = grid_size - 1 - positions[:, 0]

        # Random horizontal flip in Z direction
        if self.rng.random() > 0.5:
            positions[:, 2] = grid_size - 1 - positions[:, 2]

        return positions, block_ids

    def __len__(self) -> int:
        """Return number of structures in dataset."""
        if self._cache is not None:
            return len(self._cache)
        return len(self.h5_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load and return a sparse structure.

        Args:
            idx: Structure index

        Returns:
            Dictionary with:
                - positions: [N, 3] float tensor of (x, y, z) coordinates
                - block_ids: [N] long tensor of block token IDs
                - embeddings: [N, embed_dim] float tensor of block embeddings
                - num_blocks: scalar tensor with count of non-air blocks
        """
        structure = self._load_structure(idx)
        positions, block_ids, embeddings = self._extract_sparse(structure)

        if self.augment:
            positions, block_ids = self._augment_sparse(positions, block_ids)
            # Re-fetch embeddings after potential block ID changes
            embeddings = self.embeddings[block_ids]

        return {
            "positions": torch.from_numpy(positions).float(),
            "block_ids": torch.from_numpy(block_ids).long(),
            "embeddings": torch.from_numpy(embeddings).float(),
            "num_blocks": torch.tensor(len(block_ids), dtype=torch.long),
        }


def collate_sparse(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for sparse structure batches.

    Pads variable-length sequences to the maximum length in the batch
    and creates attention masks.

    Args:
        batch: List of dictionaries from SparseStructureDataset

    Returns:
        Dictionary with batched and padded tensors:
            - positions: [B, max_N, 3] padded positions
            - block_ids: [B, max_N] padded block IDs
            - embeddings: [B, max_N, embed_dim] padded embeddings
            - num_blocks: [B] actual block counts per sample
            - attention_mask: [B, max_N] boolean mask (True = valid, False = padding)
    """
    # Get individual tensors
    positions_list = [item["positions"] for item in batch]
    block_ids_list = [item["block_ids"] for item in batch]
    embeddings_list = [item["embeddings"] for item in batch]
    num_blocks = torch.stack([item["num_blocks"] for item in batch])

    # Pad sequences
    positions = pad_sequence(positions_list, batch_first=True, padding_value=0.0)
    block_ids = pad_sequence(block_ids_list, batch_first=True, padding_value=0)
    embeddings = pad_sequence(embeddings_list, batch_first=True, padding_value=0.0)

    # Create attention mask
    max_len = positions.size(1)
    attention_mask = torch.arange(max_len).unsqueeze(0) < num_blocks.unsqueeze(1)

    return {
        "positions": positions,
        "block_ids": block_ids,
        "embeddings": embeddings,
        "num_blocks": num_blocks,
        "attention_mask": attention_mask,
    }


class SparseStructureDatasetWithReconstruction(SparseStructureDataset):
    """Extended dataset that also provides dense reconstruction targets.

    Useful for:
    - Computing dense reconstruction accuracy
    - Comparing with dense VQ-VAE baseline
    - Visualization of reconstructions

    Args:
        Same as SparseStructureDataset
    """

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load sparse structure and dense target.

        Returns:
            Dictionary with sparse data plus:
                - dense_target: [32, 32, 32] full dense structure
        """
        # Get sparse representation
        result = super().__getitem__(idx)

        # Also return dense structure for evaluation
        structure = self._load_structure(idx)
        result["dense_target"] = torch.from_numpy(structure).long()

        return result


def collate_sparse_with_dense(
    batch: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """Collate function that also handles dense targets.

    Args:
        batch: List of dictionaries from SparseStructureDatasetWithReconstruction

    Returns:
        Same as collate_sparse plus:
            - dense_target: [B, 32, 32, 32] batched dense structures
    """
    # First do the sparse collation
    result = collate_sparse(batch)

    # Add dense targets
    dense_targets = torch.stack([item["dense_target"] for item in batch])
    result["dense_target"] = dense_targets

    return result


def get_block_category(block_name: str) -> str:
    """Categorize a block for per-category metrics.

    Args:
        block_name: Full block name like "minecraft:oak_stairs[facing=north]"

    Returns:
        Category string like "stairs", "slabs", "wood", etc.
    """
    name = block_name.replace("minecraft:", "").split("[")[0].lower()

    # Shape categories (most specific first)
    if "stair" in name:
        return "stairs"
    if "slab" in name:
        return "slabs"
    if "_wall" in name or name.endswith("_wall"):
        return "walls"
    if "fence_gate" in name:
        return "fence_gates"
    if "fence" in name:
        return "fences"
    if "door" in name:
        return "doors"
    if "trapdoor" in name:
        return "trapdoors"
    if "button" in name:
        return "buttons"
    if "pressure_plate" in name:
        return "pressure_plates"
    if "sign" in name:
        return "signs"

    # Material categories
    if any(w in name for w in ["oak", "spruce", "birch", "jungle", "acacia", "dark_oak", "mangrove", "cherry", "bamboo"]):
        if "planks" in name:
            return "planks"
        if "log" in name or "wood" in name:
            return "logs"
        return "wood"
    if any(w in name for w in ["stone", "cobble", "granite", "diorite", "andesite", "deepslate", "brick"]):
        return "stone"
    if "glass" in name:
        return "glass"
    if "wool" in name:
        return "wool"
    if "concrete" in name:
        return "concrete"
    if "terracotta" in name:
        return "terracotta"
    if any(w in name for w in ["torch", "lantern", "lamp", "light"]):
        return "lighting"
    if any(w in name for w in ["redstone", "piston", "observer", "comparator", "repeater", "lever"]):
        return "redstone"
    if any(w in name for w in ["flower", "grass", "fern", "leaves", "sapling", "vine", "moss"]):
        return "plants"
    if any(w in name for w in ["iron", "gold", "copper", "diamond", "emerald", "netherite"]):
        return "metal"
    if "ore" in name:
        return "ores"

    return "other"

