"""
Dataset for VQ-VAE training.

Loads 32x32x32 Minecraft structures from H5 files for
training the VQ-VAE encoder-decoder architecture.
"""

import random
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class VQVAEDataset(Dataset):
    """Dataset that loads full 32x32x32 structures for VQ-VAE training.

    Unlike Block2VecDataset which yields individual block pairs,
    this dataset yields complete structures for reconstruction learning.

    Args:
        data_dir: Directory containing H5 files
        max_files: Optional limit on number of files to load (for debugging)
        cache_in_memory: If True, loads all structures into RAM (faster but uses more memory)
        augment: If True, apply random rotations and flips
        seed: Random seed for augmentation
    """

    def __init__(
        self,
        data_dir: str,
        max_files: Optional[int] = None,
        cache_in_memory: bool = False,
        augment: bool = False,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.cache_in_memory = cache_in_memory
        self.augment = augment
        self.seed = seed
        self.rng = random.Random(seed)

        # Find all H5 files
        self.h5_files = sorted(self.data_dir.glob("*.h5"))
        if not self.h5_files:
            raise ValueError(f"No H5 files found in {data_dir}")

        # Optionally limit files (for quick testing)
        if max_files is not None:
            self.h5_files = self.h5_files[:max_files]

        print(f"Found {len(self.h5_files)} H5 files in {data_dir}")

        # Cache structures in memory if requested
        self._cache: Optional[List[np.ndarray]] = None
        if cache_in_memory:
            self._load_all_into_memory()

    def _load_all_into_memory(self) -> None:
        """Load all structures into RAM for faster training."""
        print("Loading all structures into memory...")
        self._cache = []

        for i, h5_path in enumerate(self.h5_files):
            try:
                with h5py.File(h5_path, "r") as f:
                    # H5 files have a single key containing the 3D array
                    key = list(f.keys())[0]
                    structure = f[key][:]
                    self._cache.append(structure.astype(np.int64))
            except Exception as e:
                print(f"Warning: Failed to load {h5_path}: {e}")
                continue

            if (i + 1) % 1000 == 0:
                print(f"  Loaded {i + 1}/{len(self.h5_files)} files...")

        print(f"Loaded {len(self._cache)} structures into memory")

    def _load_structure(self, idx: int) -> np.ndarray:
        """Load a single structure from disk or cache.

        Args:
            idx: Index into the file list

        Returns:
            3D numpy array of block token IDs [32, 32, 32]
        """
        if self._cache is not None:
            return self._cache[idx]

        h5_path = self.h5_files[idx]
        with h5py.File(h5_path, "r") as f:
            key = list(f.keys())[0]
            structure = f[key][:]

        return structure.astype(np.int64)

    def _augment_structure(self, structure: np.ndarray) -> np.ndarray:
        """Apply random augmentations to a structure.

        Augmentations (all preserve block semantics):
        - Random rotation by 0, 90, 180, or 270 degrees around Y axis
        - Random horizontal flips

        Args:
            structure: 3D array [depth, height, width]

        Returns:
            Augmented 3D array
        """
        # Random rotation around vertical (Y) axis
        # np.rot90 rotates in the plane of the given axes
        k = self.rng.randint(0, 3)  # 0, 1, 2, or 3 * 90 degrees
        if k > 0:
            # Rotate in X-Z plane (around Y axis)
            structure = np.rot90(structure, k=k, axes=(0, 2))

        # Random horizontal flip in X direction
        if self.rng.random() > 0.5:
            structure = np.flip(structure, axis=2)

        # Random horizontal flip in Z direction
        if self.rng.random() > 0.5:
            structure = np.flip(structure, axis=0)

        # Make contiguous (required after transformations)
        return np.ascontiguousarray(structure)

    def __len__(self) -> int:
        """Return number of structures in dataset."""
        if self._cache is not None:
            return len(self._cache)
        return len(self.h5_files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Load and return a single structure.

        Args:
            idx: Structure index

        Returns:
            Tensor of block IDs [32, 32, 32] as torch.long
        """
        structure = self._load_structure(idx)

        if self.augment:
            structure = self._augment_structure(structure)

        return torch.from_numpy(structure).long()


class VQVAEDatasetWithWeights(VQVAEDataset):
    """VQ-VAE Dataset that also provides per-position weights.

    Useful for:
    - Ignoring air blocks in loss computation
    - Weighting rare blocks more heavily
    - Focusing on structure surfaces

    Args:
        data_dir: Directory containing H5 files
        air_token: Token ID for air blocks (will be weighted lower)
        air_weight: Weight for air blocks (0.0 = ignore, 1.0 = full weight)
        surface_weight: Extra weight for blocks adjacent to air
        **kwargs: Additional arguments passed to VQVAEDataset
    """

    def __init__(
        self,
        data_dir: str,
        air_token: int = 0,
        air_weight: float = 0.1,
        surface_weight: float = 2.0,
        **kwargs,
    ):
        super().__init__(data_dir, **kwargs)
        self.air_token = air_token
        self.air_weight = air_weight
        self.surface_weight = surface_weight

    def _compute_weights(self, structure: np.ndarray) -> np.ndarray:
        """Compute per-position weights based on block context.

        Args:
            structure: 3D array of block IDs

        Returns:
            3D array of weights (same shape)
        """
        weights = np.ones_like(structure, dtype=np.float32)

        # Lower weight for air blocks
        weights[structure == self.air_token] = self.air_weight

        # Higher weight for surface blocks (adjacent to air)
        is_air = structure == self.air_token
        is_non_air = ~is_air

        # Check each direction for air neighbors
        for axis in [0, 1, 2]:
            # Shift in positive direction
            shifted = np.roll(is_air, 1, axis=axis)
            weights[is_non_air & shifted] = self.surface_weight

            # Shift in negative direction
            shifted = np.roll(is_air, -1, axis=axis)
            weights[is_non_air & shifted] = self.surface_weight

        return weights

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load structure and compute weights.

        Args:
            idx: Structure index

        Returns:
            Tuple of (structure tensor, weights tensor)
        """
        structure = self._load_structure(idx)
        weights = self._compute_weights(structure)

        if self.augment:
            structure = self._augment_structure(structure)
            weights = self._augment_structure(weights)

        return (
            torch.from_numpy(structure).long(),
            torch.from_numpy(weights).float(),
        )


def collate_vqvae(batch: List[torch.Tensor]) -> torch.Tensor:
    """Simple collate function for VQ-VAE batches.

    Args:
        batch: List of structure tensors

    Returns:
        Batched tensor [batch_size, 32, 32, 32]
    """
    return torch.stack(batch, dim=0)


def collate_vqvae_with_weights(
    batch: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate function for VQ-VAE batches with weights.

    Args:
        batch: List of (structure, weights) tuples

    Returns:
        Tuple of batched tensors
    """
    structures = torch.stack([b[0] for b in batch], dim=0)
    weights = torch.stack([b[1] for b in batch], dim=0)
    return structures, weights
