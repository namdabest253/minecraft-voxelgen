"""
Block2Vec V3: Compositional Embeddings.

Instead of learning one embedding per block, V3 decomposes blocks into:
- Material embedding (16 dims): oak, spruce, stone, iron, etc.
- Shape embedding (16 dims): planks, stairs, slab, block, etc.
- Property embedding (8 dims): solid, transparent, light_emitting, etc.

Final block embedding = material_emb + shape_emb + property_emb

This GUARANTEES that:
- oak_planks and spruce_planks are similar (share shape_emb)
- oak_planks and oak_stairs are similar (share material_emb)
- All transparent blocks cluster (share property_emb)

The model still uses skip-gram training on neighbor prediction,
but the gradients flow through the component embeddings.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CompositionalBlock2Vec(nn.Module):
    """
    Compositional block embeddings using material + shape + properties.

    Architecture:
        block_emb = material_emb + shape_emb + sum(property_embs)

    Training uses skip-gram objective: predict neighbors from center block.
    """

    def __init__(
        self,
        num_blocks: int,
        num_materials: int,
        num_shapes: int,
        num_properties: int,
        material_dim: int = 16,
        shape_dim: int = 16,
        property_dim: int = 8,
        block_to_material: Optional[torch.Tensor] = None,
        block_to_shape: Optional[torch.Tensor] = None,
        block_to_properties: Optional[torch.Tensor] = None,
    ):
        """
        Initialize compositional embeddings.

        Args:
            num_blocks: Total number of blocks in vocabulary
            num_materials: Number of unique materials (+ 1 for _none_)
            num_shapes: Number of unique shapes
            num_properties: Number of unique properties
            material_dim: Dimensions for material embedding
            shape_dim: Dimensions for shape embedding
            property_dim: Dimensions for property embedding (per property)
            block_to_material: [num_blocks] tensor mapping block -> material idx
            block_to_shape: [num_blocks] tensor mapping block -> shape idx
            block_to_properties: [num_blocks, num_properties] binary tensor
        """
        super().__init__()

        self.num_blocks = num_blocks
        self.num_materials = num_materials
        self.num_shapes = num_shapes
        self.num_properties = num_properties
        self.material_dim = material_dim
        self.shape_dim = shape_dim
        self.property_dim = property_dim

        # Total embedding dimension
        self.embedding_dim = material_dim + shape_dim + property_dim

        # Component embeddings
        self.material_emb = nn.Embedding(num_materials, material_dim)
        self.shape_emb = nn.Embedding(num_shapes, shape_dim)
        self.property_emb = nn.Embedding(num_properties, property_dim)

        # Context embedding for skip-gram (separate from input embedding)
        self.context_emb = nn.Embedding(num_blocks, self.embedding_dim)

        # Register block -> component mappings as buffers (not parameters)
        if block_to_material is not None:
            self.register_buffer('block_to_material', block_to_material)
        else:
            self.register_buffer('block_to_material', torch.zeros(num_blocks, dtype=torch.long))

        if block_to_shape is not None:
            self.register_buffer('block_to_shape', block_to_shape)
        else:
            self.register_buffer('block_to_shape', torch.zeros(num_blocks, dtype=torch.long))

        if block_to_properties is not None:
            self.register_buffer('block_to_properties', block_to_properties.float())
        else:
            self.register_buffer('block_to_properties', torch.zeros(num_blocks, num_properties))

        # Initialize embeddings
        self._init_embeddings()

    def _init_embeddings(self):
        """Initialize embeddings with small random values."""
        nn.init.uniform_(self.material_emb.weight, -0.1, 0.1)
        nn.init.uniform_(self.shape_emb.weight, -0.1, 0.1)
        nn.init.uniform_(self.property_emb.weight, -0.1, 0.1)
        nn.init.uniform_(self.context_emb.weight, -0.1, 0.1)

    def get_block_embedding(self, block_ids: torch.Tensor) -> torch.Tensor:
        """
        Get compositional embeddings for blocks.

        Args:
            block_ids: [batch_size] or [batch_size, ...] tensor of block IDs

        Returns:
            [batch_size, ..., embedding_dim] tensor of embeddings
        """
        original_shape = block_ids.shape

        # Flatten for lookup
        flat_ids = block_ids.view(-1)

        # Get component indices for each block
        material_ids = self.block_to_material[flat_ids]  # [N]
        shape_ids = self.block_to_shape[flat_ids]  # [N]
        property_mask = self.block_to_properties[flat_ids]  # [N, num_properties]

        # Get component embeddings
        mat_emb = self.material_emb(material_ids)  # [N, material_dim]
        shp_emb = self.shape_emb(shape_ids)  # [N, shape_dim]

        # Property embedding: weighted sum based on which properties the block has
        # property_emb.weight: [num_properties, property_dim]
        # property_mask: [N, num_properties]
        # Result: [N, property_dim]
        prop_emb = torch.matmul(property_mask, self.property_emb.weight)

        # Normalize by number of properties (avoid zero division)
        num_props = property_mask.sum(dim=1, keepdim=True).clamp(min=1)
        prop_emb = prop_emb / num_props

        # Combine: concatenate components
        combined = torch.cat([mat_emb, shp_emb, prop_emb], dim=-1)  # [N, embedding_dim]

        # Reshape to original
        return combined.view(*original_shape, self.embedding_dim)

    def forward(
        self,
        center_ids: torch.Tensor,
        context_ids: torch.Tensor,
        negative_ids: torch.Tensor,
    ) -> dict:
        """
        Forward pass for skip-gram training.

        Args:
            center_ids: [batch_size] center block IDs
            context_ids: [batch_size] positive context block IDs
            negative_ids: [batch_size, num_neg] negative sample IDs

        Returns:
            Dictionary with 'loss' and other metrics
        """
        batch_size = center_ids.shape[0]
        num_neg = negative_ids.shape[1]

        # Get center block embeddings (compositional)
        center_emb = self.get_block_embedding(center_ids)  # [B, D]

        # Get context embeddings (direct lookup, not compositional)
        pos_ctx = self.context_emb(context_ids)  # [B, D]
        neg_ctx = self.context_emb(negative_ids)  # [B, num_neg, D]

        # Positive scores: dot product of center and positive context
        pos_scores = (center_emb * pos_ctx).sum(dim=1)  # [B]

        # Negative scores: dot product of center and negative contexts
        neg_scores = torch.bmm(neg_ctx, center_emb.unsqueeze(2)).squeeze(2)  # [B, num_neg]

        # Skip-gram loss: maximize positive, minimize negative
        pos_loss = F.logsigmoid(pos_scores).mean()
        neg_loss = F.logsigmoid(-neg_scores).mean()

        loss = -(pos_loss + neg_loss)

        return {
            'loss': loss,
            'pos_loss': -pos_loss.item(),
            'neg_loss': -neg_loss.item(),
        }

    def get_all_embeddings(self) -> np.ndarray:
        """Get embeddings for all blocks as numpy array."""
        with torch.no_grad():
            all_ids = torch.arange(self.num_blocks, device=self.material_emb.weight.device)
            embeddings = self.get_block_embedding(all_ids)
            return embeddings.cpu().numpy()

    def get_component_embeddings(self) -> dict:
        """Get the learned component embeddings."""
        return {
            'material': self.material_emb.weight.detach().cpu().numpy(),
            'shape': self.shape_emb.weight.detach().cpu().numpy(),
            'property': self.property_emb.weight.detach().cpu().numpy(),
        }


class CompositionalBlock2VecWithCBOW(CompositionalBlock2Vec):
    """
    Compositional Block2Vec with both Skip-gram and CBOW objectives.

    CBOW: Predict center block from context blocks.
    Skip-gram: Predict context blocks from center block.
    """

    def __init__(self, *args, cbow_weight: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.cbow_weight = cbow_weight
        self.skipgram_weight = 1.0 - cbow_weight

        # Output projection for CBOW (predicts block from context average)
        self.cbow_output = nn.Linear(self.embedding_dim, self.num_blocks, bias=False)

    def forward_cbow(
        self,
        center_ids: torch.Tensor,
        context_ids: torch.Tensor,
    ) -> dict:
        """
        CBOW forward pass: predict center from context.

        Args:
            center_ids: [batch_size] center block IDs (target)
            context_ids: [batch_size, num_context] context block IDs

        Returns:
            Dictionary with CBOW loss
        """
        # Get context embeddings and average
        ctx_emb = self.get_block_embedding(context_ids)  # [B, num_ctx, D]
        ctx_avg = ctx_emb.mean(dim=1)  # [B, D]

        # Predict center block
        logits = self.cbow_output(ctx_avg)  # [B, num_blocks]

        # Cross-entropy loss
        loss = F.cross_entropy(logits, center_ids)

        return {'cbow_loss': loss}

    def forward(
        self,
        center_ids: torch.Tensor,
        context_ids: torch.Tensor,
        negative_ids: torch.Tensor,
        all_context_ids: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Combined forward pass with skip-gram and optionally CBOW.

        Args:
            center_ids: [batch_size] center block IDs
            context_ids: [batch_size] single positive context (for skip-gram)
            negative_ids: [batch_size, num_neg] negative samples
            all_context_ids: [batch_size, num_context] all context blocks (for CBOW)
        """
        # Skip-gram loss
        sg_out = super().forward(center_ids, context_ids, negative_ids)

        if all_context_ids is not None and self.cbow_weight > 0:
            # CBOW loss
            cbow_out = self.forward_cbow(center_ids, all_context_ids)

            # Combine losses
            total_loss = (self.skipgram_weight * sg_out['loss'] +
                          self.cbow_weight * cbow_out['cbow_loss'])

            return {
                'loss': total_loss,
                'skipgram_loss': sg_out['loss'].item(),
                'cbow_loss': cbow_out['cbow_loss'].item(),
                'pos_loss': sg_out['pos_loss'],
                'neg_loss': sg_out['neg_loss'],
            }
        else:
            return sg_out


def create_block_mappings(
    vocab_data: dict,
    num_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create block -> component index mappings from vocabulary data.

    Args:
        vocab_data: Output from create_component_vocabularies()
        num_blocks: Total vocabulary size

    Returns:
        Tuple of (block_to_material, block_to_shape, block_to_properties)
    """
    material2idx = vocab_data['material2idx']
    shape2idx = vocab_data['shape2idx']
    property2idx = vocab_data['property2idx']
    block_components = vocab_data['block_components']

    num_properties = len(property2idx)

    block_to_material = torch.zeros(num_blocks, dtype=torch.long)
    block_to_shape = torch.zeros(num_blocks, dtype=torch.long)
    block_to_properties = torch.zeros(num_blocks, num_properties)

    for token_id, components in block_components.items():
        tid = int(token_id)
        if tid >= num_blocks:
            continue

        # Material (use _none_ index if no material)
        if components.material:
            block_to_material[tid] = material2idx.get(components.material, 0)
        else:
            block_to_material[tid] = 0  # _none_

        # Shape
        block_to_shape[tid] = shape2idx.get(components.shape, 0)

        # Properties (multi-hot)
        for prop in components.properties:
            if prop in property2idx:
                block_to_properties[tid, property2idx[prop]] = 1.0

    return block_to_material, block_to_shape, block_to_properties


def create_model_from_vocab(
    tok2block: dict[int, str],
    material_dim: int = 16,
    shape_dim: int = 16,
    property_dim: int = 8,
    cbow_weight: float = 0.0,
) -> CompositionalBlock2Vec:
    """
    Create a compositional Block2Vec model from vocabulary.

    Args:
        tok2block: Token ID to block name mapping
        material_dim: Embedding dimension for materials
        shape_dim: Embedding dimension for shapes
        property_dim: Embedding dimension for properties
        cbow_weight: Weight for CBOW loss (0 = skip-gram only)

    Returns:
        Initialized CompositionalBlock2Vec model
    """
    from src.data.block_decomposition import create_component_vocabularies

    # Create component vocabularies
    vocab_data = create_component_vocabularies(tok2block)

    num_blocks = len(tok2block)
    num_materials = len(vocab_data['materials'])
    num_shapes = len(vocab_data['shapes'])
    num_properties = len(vocab_data['properties'])

    # Create mappings
    block_to_material, block_to_shape, block_to_properties = create_block_mappings(
        vocab_data, num_blocks
    )

    # Create model
    if cbow_weight > 0:
        model = CompositionalBlock2VecWithCBOW(
            num_blocks=num_blocks,
            num_materials=num_materials,
            num_shapes=num_shapes,
            num_properties=num_properties,
            material_dim=material_dim,
            shape_dim=shape_dim,
            property_dim=property_dim,
            block_to_material=block_to_material,
            block_to_shape=block_to_shape,
            block_to_properties=block_to_properties,
            cbow_weight=cbow_weight,
        )
    else:
        model = CompositionalBlock2Vec(
            num_blocks=num_blocks,
            num_materials=num_materials,
            num_shapes=num_shapes,
            num_properties=num_properties,
            material_dim=material_dim,
            shape_dim=shape_dim,
            property_dim=property_dim,
            block_to_material=block_to_material,
            block_to_shape=block_to_shape,
            block_to_properties=block_to_properties,
        )

    return model, vocab_data
