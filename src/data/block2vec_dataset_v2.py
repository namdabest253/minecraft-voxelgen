"""
Dataset for Block2Vec V2 training.

V2 changes from V1:
- Collapses block states during loading
- Returns all contexts for CBOW training (not just one)
- Supports both 6 and 26 neighbor contexts
"""

import json
import random
from pathlib import Path
from typing import Iterator, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import IterableDataset


class Block2VecDatasetV2(IterableDataset):
    """Iterable dataset for Block2Vec V2 training.

    Yields batches with center, all contexts, and negatives for
    hybrid skip-gram + CBOW training.

    Args:
        data_dir: Directory containing H5 files
        vocab_size: Collapsed vocabulary size
        original_to_collapsed: Mapping from original token IDs to collapsed IDs
        context_type: "neighbors_6" or "neighbors_26"
        num_negative_samples: Number of negative samples per pair
        subsample_threshold: Threshold for subsampling frequent blocks
        negative_sampling_power: Power for frequency-based negative sampling
        air_token: Token ID for air in COLLAPSED vocabulary
        include_air: Whether to include air blocks
        seed: Random seed
    """

    # 6 neighbors: +/- in each axis
    NEIGHBORS_6 = [
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
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
        original_to_collapsed: dict[int, int],
        context_type: str = "neighbors_6",
        num_negative_samples: int = 15,
        subsample_threshold: float = 0.0001,
        negative_sampling_power: float = 0.75,
        air_token: int = 0,  # Will be set correctly for collapsed vocab
        include_air: bool = False,  # Default to excluding air in V2
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.vocab_size = vocab_size
        self.original_to_collapsed = original_to_collapsed
        self.num_negative_samples = num_negative_samples
        self.subsample_threshold = subsample_threshold
        self.negative_sampling_power = negative_sampling_power
        self.air_token = air_token
        self.include_air = include_air
        self.seed = seed

        # Set neighbor offsets
        if context_type == "neighbors_6":
            self.neighbor_offsets = self.NEIGHBORS_6
        elif context_type == "neighbors_26":
            self.neighbor_offsets = self.NEIGHBORS_26
        else:
            raise ValueError(f"Unknown context_type: {context_type}")

        self.num_contexts = len(self.neighbor_offsets)

        # Find all H5 files
        self.h5_files = sorted(self.data_dir.glob("*.h5"))
        if not self.h5_files:
            raise ValueError(f"No H5 files found in {data_dir}")

        # Will be computed on first pass
        self._block_freqs: Optional[np.ndarray] = None
        self._negative_table: Optional[np.ndarray] = None
        self._subsample_probs: Optional[np.ndarray] = None

    def _collapse_build(self, build: np.ndarray) -> np.ndarray:
        """Collapse block states in a build array.

        Args:
            build: Original build array with original token IDs

        Returns:
            Build array with collapsed token IDs
        """
        collapsed = np.zeros_like(build)
        for orig_id, collapsed_id in self.original_to_collapsed.items():
            collapsed[build == orig_id] = collapsed_id
        return collapsed

    def _compute_frequencies(self) -> None:
        """Compute collapsed block frequencies across all data."""
        print("Computing collapsed block frequencies...")
        freqs = np.zeros(self.vocab_size, dtype=np.float64)

        for h5_path in self.h5_files:
            try:
                with h5py.File(h5_path, "r") as f:
                    key = list(f.keys())[0]
                    build = f[key][:]

                # Collapse block states
                collapsed = self._collapse_build(build)

                unique, counts = np.unique(collapsed, return_counts=True)
                for token, count in zip(unique, counts):
                    if token < self.vocab_size:
                        freqs[token] += count
            except Exception:
                continue

        # Normalize
        total = freqs.sum()
        if total > 0:
            freqs /= total

        self._block_freqs = freqs

        # Build negative sampling table
        weighted = np.power(freqs + 1e-10, self.negative_sampling_power)
        weighted /= weighted.sum()
        table_size = 100_000_000
        self._negative_table = np.random.choice(
            self.vocab_size, size=table_size, p=weighted
        )

        # Compute subsampling probabilities
        self._subsample_probs = np.ones(self.vocab_size, dtype=np.float32)
        for i, freq in enumerate(freqs):
            if freq > self.subsample_threshold:
                self._subsample_probs[i] = np.sqrt(self.subsample_threshold / freq)

        print(f"Collapsed vocab frequencies computed.")
        print(f"  Air frequency: {freqs[self.air_token]:.4f}")
        print(f"  Non-zero blocks: {(freqs > 0).sum()}/{self.vocab_size}")

    def _sample_negatives(self, n: int, rng: random.Random) -> np.ndarray:
        """Sample negative examples."""
        if self._negative_table is None:
            self._compute_frequencies()

        indices = rng.sample(range(len(self._negative_table)), n)
        return self._negative_table[indices]

    def _should_keep(self, token: int, rng: random.Random) -> bool:
        """Determine if a token should be kept (subsampling)."""
        if self._subsample_probs is None:
            self._compute_frequencies()

        return rng.random() < self._subsample_probs[token]

    def __iter__(self) -> Iterator[tuple]:
        """Yield (center, all_contexts, context_mask, negatives) tuples."""
        if self._block_freqs is None:
            self._compute_frequencies()

        # Worker info for distributed loading
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            files = self.h5_files
            worker_id = 0
        else:
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

            # Collapse block states
            build = self._collapse_build(build)
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

                        # Get all context blocks
                        contexts = []
                        context_mask = []

                        for dy, dx, dz in self.neighbor_offsets:
                            ny, nx, nz = y + dy, x + dx, z + dz

                            # Bounds check
                            if 0 <= ny < h and 0 <= nx < w and 0 <= nz < d:
                                ctx = int(build[ny, nx, nz])

                                # Skip air context if configured
                                if not self.include_air and ctx == self.air_token:
                                    contexts.append(0)  # Placeholder
                                    context_mask.append(False)
                                else:
                                    contexts.append(ctx)
                                    context_mask.append(True)
                            else:
                                contexts.append(0)
                                context_mask.append(False)

                        # Skip if no valid contexts
                        if not any(context_mask):
                            continue

                        # Sample negatives
                        negatives = self._sample_negatives(self.num_negative_samples, rng)

                        yield (
                            center,
                            np.array(contexts, dtype=np.int64),
                            np.array(context_mask, dtype=bool),
                            negatives,
                        )


def collate_block2vec_v2(
    batch: list[tuple]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function for Block2Vec V2 batches.

    Args:
        batch: List of (center, contexts, mask, negatives) tuples

    Returns:
        Tuple of (center_ids, context_ids, context_mask, negative_ids) tensors
    """
    centers = torch.tensor([b[0] for b in batch], dtype=torch.long)
    contexts = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.long)
    masks = torch.tensor(np.stack([b[2] for b in batch]), dtype=torch.bool)
    negatives = torch.tensor(np.stack([b[3] for b in batch]), dtype=torch.long)

    return centers, contexts, masks, negatives


class Block2VecDatasetV2Simple(IterableDataset):
    """Simplified V2 dataset that yields one context at a time.

    More similar to V1 but includes all contexts for CBOW.
    This is easier to integrate with existing training loops.
    """

    NEIGHBORS_6 = [
        (-1, 0, 0), (1, 0, 0),
        (0, -1, 0), (0, 1, 0),
        (0, 0, -1), (0, 0, 1),
    ]

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
        original_to_collapsed: dict[int, int],
        context_type: str = "neighbors_6",
        num_negative_samples: int = 15,
        subsample_threshold: float = 0.0001,
        negative_sampling_power: float = 0.75,
        air_token: int = 0,
        include_air: bool = False,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.vocab_size = vocab_size
        self.original_to_collapsed = original_to_collapsed
        self.num_negative_samples = num_negative_samples
        self.subsample_threshold = subsample_threshold
        self.negative_sampling_power = negative_sampling_power
        self.air_token = air_token
        self.include_air = include_air
        self.seed = seed

        if context_type == "neighbors_6":
            self.neighbor_offsets = self.NEIGHBORS_6
        elif context_type == "neighbors_26":
            self.neighbor_offsets = self.NEIGHBORS_26
        else:
            raise ValueError(f"Unknown context_type: {context_type}")

        self.num_contexts = len(self.neighbor_offsets)
        self.h5_files = sorted(self.data_dir.glob("*.h5"))

        if not self.h5_files:
            raise ValueError(f"No H5 files found in {data_dir}")

        self._block_freqs = None
        self._negative_table = None
        self._subsample_probs = None

    def _collapse_build(self, build: np.ndarray) -> np.ndarray:
        collapsed = np.zeros_like(build)
        for orig_id, collapsed_id in self.original_to_collapsed.items():
            collapsed[build == orig_id] = collapsed_id
        return collapsed

    def _compute_frequencies(self) -> None:
        print("Computing collapsed block frequencies...")
        freqs = np.zeros(self.vocab_size, dtype=np.float64)

        for h5_path in self.h5_files:
            try:
                with h5py.File(h5_path, "r") as f:
                    key = list(f.keys())[0]
                    build = f[key][:]
                collapsed = self._collapse_build(build)
                unique, counts = np.unique(collapsed, return_counts=True)
                for token, count in zip(unique, counts):
                    if token < self.vocab_size:
                        freqs[token] += count
            except Exception:
                continue

        total = freqs.sum()
        if total > 0:
            freqs /= total

        self._block_freqs = freqs

        weighted = np.power(freqs + 1e-10, self.negative_sampling_power)
        weighted /= weighted.sum()
        self._negative_table = np.random.choice(
            self.vocab_size, size=100_000_000, p=weighted
        )

        self._subsample_probs = np.ones(self.vocab_size, dtype=np.float32)
        for i, freq in enumerate(freqs):
            if freq > self.subsample_threshold:
                self._subsample_probs[i] = np.sqrt(self.subsample_threshold / freq)

        print(f"Air frequency: {freqs[self.air_token]:.4f}")

    def _sample_negatives(self, n: int, rng: random.Random) -> np.ndarray:
        if self._negative_table is None:
            self._compute_frequencies()
        indices = rng.sample(range(len(self._negative_table)), n)
        return self._negative_table[indices]

    def _should_keep(self, token: int, rng: random.Random) -> bool:
        if self._subsample_probs is None:
            self._compute_frequencies()
        return rng.random() < self._subsample_probs[token]

    def __iter__(self):
        if self._block_freqs is None:
            self._compute_frequencies()

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            files = self.h5_files
            worker_id = 0
        else:
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

            build = self._collapse_build(build)
            h, w, d = build.shape

            for y in range(h):
                for x in range(w):
                    for z in range(d):
                        center = int(build[y, x, z])

                        if not self.include_air and center == self.air_token:
                            continue

                        if not self._should_keep(center, rng):
                            continue

                        # Collect all contexts
                        all_contexts = []
                        context_mask = []

                        for dy, dx, dz in self.neighbor_offsets:
                            ny, nx, nz = y + dy, x + dx, z + dz

                            if 0 <= ny < h and 0 <= nx < w and 0 <= nz < d:
                                ctx = int(build[ny, nx, nz])
                                if not self.include_air and ctx == self.air_token:
                                    all_contexts.append(0)
                                    context_mask.append(False)
                                else:
                                    all_contexts.append(ctx)
                                    context_mask.append(True)
                            else:
                                all_contexts.append(0)
                                context_mask.append(False)

                        if not any(context_mask):
                            continue

                        negatives = self._sample_negatives(self.num_negative_samples, rng)

                        # Yield one entry per valid context (for skip-gram)
                        # but include all contexts (for CBOW)
                        for ctx_idx, (ctx, valid) in enumerate(zip(all_contexts, context_mask)):
                            if valid:
                                yield (
                                    center,
                                    ctx,
                                    negatives,
                                    np.array(all_contexts, dtype=np.int64),
                                    np.array(context_mask, dtype=bool),
                                )


def collate_block2vec_v2_simple(batch):
    """Collate for simple V2 dataset."""
    centers = torch.tensor([b[0] for b in batch], dtype=torch.long)
    contexts = torch.tensor([b[1] for b in batch], dtype=torch.long)
    negatives = torch.tensor(np.stack([b[2] for b in batch]), dtype=torch.long)
    all_contexts = torch.tensor(np.stack([b[3] for b in batch]), dtype=torch.long)
    masks = torch.tensor(np.stack([b[4] for b in batch]), dtype=torch.bool)

    return centers, contexts, negatives, all_contexts, masks
