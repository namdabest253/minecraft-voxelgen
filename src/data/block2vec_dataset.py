"""
Dataset for Block2Vec training.

Extracts (center, context) block pairs from 3D Minecraft builds
for skip-gram training with negative sampling.
"""

import json
import random
from pathlib import Path
from typing import Iterator, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import IterableDataset


class Block2VecDataset(IterableDataset):
    """Iterable dataset that yields (center, context, negatives) tuples.

    Processes H5 files on-the-fly to avoid loading all builds into memory.
    Uses 6-neighbor or 26-neighbor context windows in 3D space.

    Args:
        data_dir: Directory containing H5 files
        vocab_size: Total vocabulary size for negative sampling
        context_type: "neighbors_6" or "neighbors_26"
        num_negative_samples: Number of negative samples per pair
        subsample_threshold: Threshold for subsampling frequent blocks
        air_token: Token ID for air (often subsampled heavily)
        seed: Random seed
    """

    # 6 neighbors: +/- in each axis (like a 3D cross)
    NEIGHBORS_6 = [
        (-1, 0, 0),
        (1, 0, 0),  # x-axis
        (0, -1, 0),
        (0, 1, 0),  # y-axis
        (0, 0, -1),
        (0, 0, 1),  # z-axis
    ]

    # 26 neighbors: all blocks in 3x3x3 cube except center
    NEIGHBORS_26 = [
        (dx, dy, dz)
        for dx in [-1, 0, 1]
        for dy in [-1, 0, 1]
        for dz in [-1, 0, 1]
        if not (dx == 0 and dy == 0 and dz == 0)
    ]

    def __init__(
        self,
        data_dir: str,
        vocab_size: int,
        context_type: str = "neighbors_6",
        num_negative_samples: int = 5,
        subsample_threshold: float = 0.001,
        negative_sampling_power: float = 0.75,
        air_token: int = 102,
        include_air: bool = True,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.vocab_size = vocab_size
        self.num_negative_samples = num_negative_samples
        self.subsample_threshold = subsample_threshold
        self.negative_sampling_power = negative_sampling_power
        self.air_token = air_token
        self.include_air = include_air
        self.seed = seed

        # Set neighbor offsets based on context type
        if context_type == "neighbors_6":
            self.neighbor_offsets = self.NEIGHBORS_6
        elif context_type == "neighbors_26":
            self.neighbor_offsets = self.NEIGHBORS_26
        else:
            raise ValueError(f"Unknown context_type: {context_type}")

        # Find all H5 files
        self.h5_files = sorted(self.data_dir.glob("*.h5"))
        if not self.h5_files:
            raise ValueError(f"No H5 files found in {data_dir}")

        # Initialize block frequencies (will be computed on first pass)
        self._block_freqs: Optional[np.ndarray] = None
        self._negative_table: Optional[np.ndarray] = None
        self._subsample_probs: Optional[np.ndarray] = None

    def _compute_frequencies(self) -> None:
        """Compute block frequencies across all data for sampling."""
        print("Computing block frequencies...")
        freqs = np.zeros(self.vocab_size, dtype=np.float64)

        for h5_path in self.h5_files:
            try:
                with h5py.File(h5_path, "r") as f:
                    key = list(f.keys())[0]
                    build = f[key][:]
                    unique, counts = np.unique(build, return_counts=True)
                    for token, count in zip(unique, counts):
                        if token < self.vocab_size:
                            freqs[token] += count
            except Exception:
                continue

        # Normalize to probabilities
        total = freqs.sum()
        if total > 0:
            freqs /= total

        self._block_freqs = freqs

        # Build negative sampling table (weighted by freq^0.75)
        weighted = np.power(freqs, self.negative_sampling_power)
        weighted /= weighted.sum()
        # Create table for efficient sampling
        table_size = 100_000_000  # 100M entries for smooth distribution
        self._negative_table = np.random.choice(
            self.vocab_size, size=table_size, p=weighted
        )

        # Compute subsampling probabilities
        # P(keep) = sqrt(threshold / freq) for freq > threshold
        self._subsample_probs = np.ones(self.vocab_size, dtype=np.float32)
        for i, freq in enumerate(freqs):
            if freq > self.subsample_threshold:
                self._subsample_probs[i] = np.sqrt(
                    self.subsample_threshold / freq
                )

        print(f"Block frequencies computed. Air frequency: {freqs[self.air_token]:.4f}")

    def _sample_negatives(self, n: int, rng: random.Random) -> np.ndarray:
        """Sample negative examples from the noise distribution."""
        if self._negative_table is None:
            self._compute_frequencies()

        indices = rng.sample(range(len(self._negative_table)), n)
        return self._negative_table[indices]

    def _should_keep(self, token: int, rng: random.Random) -> bool:
        """Determine if a token should be kept (subsampling)."""
        if self._subsample_probs is None:
            self._compute_frequencies()

        return rng.random() < self._subsample_probs[token]

    def __iter__(self) -> Iterator[tuple[int, int, np.ndarray]]:
        """Yield (center, context, negatives) tuples."""
        # Initialize frequencies if needed
        if self._block_freqs is None:
            self._compute_frequencies()

        # Get worker info for distributed loading
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single process
            files = self.h5_files
            worker_id = 0
        else:
            # Multi-process: split files among workers
            per_worker = len(self.h5_files) // worker_info.num_workers
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = start + per_worker if worker_id < worker_info.num_workers - 1 else len(self.h5_files)
            files = self.h5_files[start:end]

        rng = random.Random(self.seed + worker_id)

        for h5_path in files:
            try:
                with h5py.File(h5_path, "r") as f:
                    key = list(f.keys())[0]
                    build = f[key][:]
            except Exception:
                continue

            h, w, d = build.shape

            # Iterate through all positions
            for y in range(h):
                for x in range(w):
                    for z in range(d):
                        center = int(build[y, x, z])

                        # Skip air if configured
                        if not self.include_air and center == self.air_token:
                            continue

                        # Subsampling check
                        if not self._should_keep(center, rng):
                            continue

                        # Get context blocks
                        for dy, dx, dz in self.neighbor_offsets:
                            ny, nx, nz = y + dy, x + dx, z + dz

                            # Bounds check
                            if not (0 <= ny < h and 0 <= nx < w and 0 <= nz < d):
                                continue

                            context = int(build[ny, nx, nz])

                            # Skip air context if configured
                            if not self.include_air and context == self.air_token:
                                continue

                            # Sample negatives
                            negatives = self._sample_negatives(
                                self.num_negative_samples, rng
                            )

                            yield center, context, negatives


def collate_block2vec(
    batch: list[tuple[int, int, np.ndarray]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function for Block2Vec batches.

    Args:
        batch: List of (center, context, negatives) tuples

    Returns:
        Tuple of (center_ids, context_ids, negative_ids) tensors
    """
    centers = torch.tensor([b[0] for b in batch], dtype=torch.long)
    contexts = torch.tensor([b[1] for b in batch], dtype=torch.long)
    negatives = torch.tensor(np.array([b[2] for b in batch]), dtype=torch.long)

    return centers, contexts, negatives
