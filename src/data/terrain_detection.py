"""
Terrain Detection Utilities.

Detects terrain blocks (dirt, grass, stone, etc.) vs building blocks
for more accurate metrics and weighted loss.

The key insight is that many Minecraft builds sit on a terrain base.
Reconstructing flat terrain is trivial, but it inflates our metrics.
By separating terrain from building blocks, we get a truer picture
of reconstruction quality.
"""

from typing import Dict, Set, Optional, Tuple
import torch
from torch import Tensor
import numpy as np


# Terrain blocks - natural blocks that often form the base of builds
TERRAIN_BLOCKS: Set[str] = {
    # Dirt family
    'minecraft:dirt',
    'minecraft:grass_block',
    'minecraft:coarse_dirt',
    'minecraft:podzol',
    'minecraft:mycelium',
    'minecraft:rooted_dirt',
    'minecraft:dirt_path',
    'minecraft:farmland',
    'minecraft:mud',
    'minecraft:muddy_mangrove_roots',

    # Stone family
    'minecraft:stone',
    'minecraft:cobblestone',
    'minecraft:mossy_cobblestone',
    'minecraft:bedrock',
    'minecraft:deepslate',
    'minecraft:cobbled_deepslate',
    'minecraft:tuff',
    'minecraft:granite',
    'minecraft:diorite',
    'minecraft:andesite',
    'minecraft:calcite',
    'minecraft:smooth_basalt',
    'minecraft:basalt',

    # Sand family
    'minecraft:sand',
    'minecraft:red_sand',
    'minecraft:gravel',
    'minecraft:clay',
    'minecraft:soul_sand',
    'minecraft:soul_soil',

    # Water and lava
    'minecraft:water',
    'minecraft:lava',

    # Terracotta (often natural terrain)
    'minecraft:terracotta',
    'minecraft:white_terracotta',
    'minecraft:orange_terracotta',
    'minecraft:magenta_terracotta',
    'minecraft:light_blue_terracotta',
    'minecraft:yellow_terracotta',
    'minecraft:lime_terracotta',
    'minecraft:pink_terracotta',
    'minecraft:gray_terracotta',
    'minecraft:light_gray_terracotta',
    'minecraft:cyan_terracotta',
    'minecraft:purple_terracotta',
    'minecraft:blue_terracotta',
    'minecraft:brown_terracotta',
    'minecraft:green_terracotta',
    'minecraft:red_terracotta',
    'minecraft:black_terracotta',

    # Netherrack family
    'minecraft:netherrack',
    'minecraft:crimson_nylium',
    'minecraft:warped_nylium',
    'minecraft:blackstone',

    # End stone
    'minecraft:end_stone',

    # Snow and ice
    'minecraft:snow_block',
    'minecraft:ice',
    'minecraft:packed_ice',
    'minecraft:blue_ice',

    # Moss
    'minecraft:moss_block',
}

# Air tokens for the standard vocabulary
AIR_TOKENS: Set[int] = {102, 576, 3352}


def build_terrain_token_set(tok2block: Dict[int, str]) -> Set[int]:
    """
    Build a set of token IDs that correspond to terrain blocks.

    Args:
        tok2block: Mapping from token ID to block name.

    Returns:
        Set of token IDs for terrain blocks.
    """
    terrain_tokens = set()

    for token_id, block_name in tok2block.items():
        # Handle block states: "minecraft:grass_block[snowy=false]" -> "minecraft:grass_block"
        base_name = block_name.split('[')[0] if '[' in block_name else block_name

        if base_name in TERRAIN_BLOCKS:
            terrain_tokens.add(token_id)

    return terrain_tokens


def detect_terrain(
    block_ids: Tensor,
    terrain_tokens: Set[int],
    y_threshold: Optional[int] = None,
    air_tokens: Set[int] = AIR_TOKENS,
) -> Tensor:
    """
    Detect terrain blocks in a structure.

    Uses block type classification. Optionally can also use Y-level
    (bottom layers are often terrain).

    Args:
        block_ids: Tensor of block IDs, shape [B, X, Y, Z] or [X, Y, Z].
        terrain_tokens: Set of token IDs that are terrain blocks.
        y_threshold: If provided, blocks at Y < threshold are also considered terrain.
        air_tokens: Set of air token IDs (excluded from terrain mask).

    Returns:
        Boolean mask where True = terrain block (excludes air).
    """
    device = block_ids.device

    # Convert terrain tokens to tensor for isin
    terrain_tensor = torch.tensor(list(terrain_tokens), dtype=torch.long, device=device)

    # Identify terrain by block type
    is_terrain = torch.isin(block_ids, terrain_tensor)

    # Optionally add Y-level based detection
    if y_threshold is not None:
        # Determine Y axis position (assuming shape is [B, X, Y, Z] or [X, Y, Z])
        if block_ids.dim() == 4:
            # [B, X, Y, Z] - Y is dim 2
            y_indices = torch.arange(block_ids.shape[2], device=device)
            y_indices = y_indices.view(1, 1, -1, 1).expand_as(block_ids)
        else:
            # [X, Y, Z] - Y is dim 1
            y_indices = torch.arange(block_ids.shape[1], device=device)
            y_indices = y_indices.view(1, -1, 1).expand_as(block_ids)

        is_low_y = y_indices < y_threshold
        is_terrain = is_terrain | is_low_y

    # Exclude air from terrain mask
    air_tensor = torch.tensor(list(air_tokens), dtype=torch.long, device=device)
    is_air = torch.isin(block_ids, air_tensor)
    is_terrain = is_terrain & ~is_air

    return is_terrain


def compute_terrain_aware_metrics(
    original: Tensor,
    reconstructed: Tensor,
    terrain_tokens: Set[int],
    air_tokens: Set[int] = AIR_TOKENS,
    y_threshold: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute separate metrics for terrain vs building blocks.

    Args:
        original: Ground truth block IDs, shape [B, X, Y, Z] or [X, Y, Z].
        reconstructed: Predicted block IDs, same shape.
        terrain_tokens: Set of terrain token IDs.
        air_tokens: Set of air token IDs.
        y_threshold: Optional Y-level threshold for terrain detection.

    Returns:
        Dictionary with terrain and building metrics.
    """
    device = original.device

    # Detect masks
    terrain_mask = detect_terrain(original, terrain_tokens, y_threshold, air_tokens)

    air_tensor = torch.tensor(list(air_tokens), dtype=torch.long, device=device)
    is_air = torch.isin(original, air_tensor)

    # Building = non-terrain and non-air
    building_mask = ~terrain_mask & ~is_air

    # Also track what the reconstruction produced
    is_air_recon = torch.isin(reconstructed, air_tensor)

    metrics = {}

    # Overall accuracy (excluding air)
    non_air_mask = ~is_air
    if non_air_mask.any():
        correct = (original == reconstructed) & non_air_mask
        metrics['overall_accuracy'] = (correct.sum() / non_air_mask.sum()).item()
    else:
        metrics['overall_accuracy'] = 0.0

    # Terrain metrics
    if terrain_mask.any():
        terrain_correct = (original[terrain_mask] == reconstructed[terrain_mask])
        metrics['terrain_accuracy'] = terrain_correct.float().mean().item()
        metrics['terrain_count'] = terrain_mask.sum().item()
    else:
        metrics['terrain_accuracy'] = 0.0
        metrics['terrain_count'] = 0

    # Building metrics (THE KEY METRICS)
    if building_mask.any():
        building_correct = (original[building_mask] == reconstructed[building_mask])
        metrics['building_accuracy'] = building_correct.float().mean().item()
        metrics['building_count'] = building_mask.sum().item()

        # Building recall: of original building blocks, how many are still non-air?
        building_preserved = building_mask & ~is_air_recon
        metrics['building_recall'] = (building_preserved.sum() / building_mask.sum()).item()

        # Building false air: of original building blocks, how many became air?
        building_erased = building_mask & is_air_recon
        metrics['building_false_air'] = (building_erased.sum() / building_mask.sum()).item()
    else:
        metrics['building_accuracy'] = 0.0
        metrics['building_recall'] = 0.0
        metrics['building_false_air'] = 0.0
        metrics['building_count'] = 0

    # Composition stats
    total_non_air = non_air_mask.sum().item()
    if total_non_air > 0:
        metrics['terrain_fraction'] = terrain_mask.sum().item() / total_non_air
        metrics['building_fraction'] = building_mask.sum().item() / total_non_air
    else:
        metrics['terrain_fraction'] = 0.0
        metrics['building_fraction'] = 0.0

    return metrics


class TerrainWeightedLoss(torch.nn.Module):
    """
    Cross-entropy loss with lower weight for terrain blocks.

    The intuition is that terrain blocks (dirt, grass, stone) are easy
    to reconstruct and shouldn't dominate the loss. We want the model
    to focus on building blocks which are more varied and interesting.

    Args:
        terrain_weight: Weight for terrain blocks (default 0.2).
        building_weight: Weight for building blocks (default 1.0).
        air_weight: Weight for air blocks (default 0.1).
    """

    def __init__(
        self,
        terrain_weight: float = 0.2,
        building_weight: float = 1.0,
        air_weight: float = 0.1,
    ):
        super().__init__()
        self.terrain_weight = terrain_weight
        self.building_weight = building_weight
        self.air_weight = air_weight

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        terrain_mask: Tensor,
        air_mask: Tensor,
    ) -> Tensor:
        """
        Compute terrain-weighted cross-entropy loss.

        Args:
            logits: Predicted logits, shape [B, num_classes, X, Y, Z].
            targets: Ground truth block IDs, shape [B, X, Y, Z].
            terrain_mask: Boolean mask, True = terrain block.
            air_mask: Boolean mask, True = air block.

        Returns:
            Weighted loss scalar.
        """
        # Compute per-voxel cross-entropy
        ce_loss = torch.nn.functional.cross_entropy(
            logits, targets, reduction='none'
        )

        # Build weight tensor
        weights = torch.full_like(ce_loss, self.building_weight)
        weights[terrain_mask] = self.terrain_weight
        weights[air_mask] = self.air_weight

        # Weighted mean
        weighted_loss = (ce_loss * weights).sum() / weights.sum()

        return weighted_loss


def analyze_dataset_terrain(
    dataloader,
    tok2block: Dict[int, str],
    air_tokens: Set[int] = AIR_TOKENS,
    max_batches: int = 100,
) -> Dict[str, float]:
    """
    Analyze terrain vs building composition across a dataset.

    Useful for understanding how much of the data is "easy" terrain.

    Args:
        dataloader: DataLoader yielding batches of block IDs.
        tok2block: Token to block name mapping.
        air_tokens: Set of air token IDs.
        max_batches: Maximum batches to analyze.

    Returns:
        Statistics about terrain composition.
    """
    terrain_tokens = build_terrain_token_set(tok2block)

    total_voxels = 0
    total_air = 0
    total_terrain = 0
    total_building = 0

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break

        if isinstance(batch, dict):
            block_ids = batch.get('blocks', batch.get('block_ids'))
        else:
            block_ids = batch

        air_tensor = torch.tensor(list(air_tokens), dtype=torch.long, device=block_ids.device)
        is_air = torch.isin(block_ids, air_tensor)

        terrain_mask = detect_terrain(block_ids, terrain_tokens, air_tokens=air_tokens)
        building_mask = ~terrain_mask & ~is_air

        total_voxels += block_ids.numel()
        total_air += is_air.sum().item()
        total_terrain += terrain_mask.sum().item()
        total_building += building_mask.sum().item()

    return {
        'total_voxels': total_voxels,
        'air_fraction': total_air / total_voxels,
        'terrain_fraction': total_terrain / total_voxels,
        'building_fraction': total_building / total_voxels,
        'terrain_of_non_air': total_terrain / max(total_terrain + total_building, 1),
        'building_of_non_air': total_building / max(total_terrain + total_building, 1),
    }
