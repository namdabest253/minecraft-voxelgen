"""
Block2Vec: Skip-gram model for learning Minecraft block embeddings.

Learns 32-dimensional embeddings for each block type by predicting
neighboring blocks in 3D structures. Similar to Word2Vec but for
spatial relationships instead of sequential text.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class Block2Vec(nn.Module):
    """Skip-gram model for block embeddings.

    Architecture:
        - Center embedding: Maps block token to embedding
        - Context embedding: Separate embedding for prediction
        - Dot product between center and context for similarity

    Args:
        vocab_size: Number of unique block tokens
        embedding_dim: Dimension of embeddings (default 32)
    """

    def __init__(self, vocab_size: int, embedding_dim: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Two separate embeddings: center and context
        # This is standard for skip-gram models
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Initialize with small random values
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize embeddings with uniform distribution."""
        init_range = 0.5 / self.embedding_dim
        self.center_embeddings.weight.data.uniform_(-init_range, init_range)
        self.context_embeddings.weight.data.uniform_(-init_range, init_range)

    def forward(
        self,
        center_ids: torch.Tensor,
        context_ids: torch.Tensor,
        negative_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute skip-gram loss with negative sampling.

        Args:
            center_ids: Center block tokens [batch_size]
            context_ids: Positive context tokens [batch_size]
            negative_ids: Negative sample tokens [batch_size, num_negatives]

        Returns:
            Loss tensor (scalar)
        """
        # Get embeddings
        center_emb = self.center_embeddings(center_ids)  # [B, D]
        context_emb = self.context_embeddings(context_ids)  # [B, D]

        # Positive score: dot product of center and context
        pos_score = torch.sum(center_emb * context_emb, dim=1)  # [B]
        pos_loss = F.logsigmoid(pos_score)  # [B]

        # Negative sampling loss
        if negative_ids is not None:
            neg_emb = self.context_embeddings(negative_ids)  # [B, N, D]
            # Broadcast center embedding for dot product
            center_emb_expanded = center_emb.unsqueeze(1)  # [B, 1, D]
            neg_score = torch.sum(center_emb_expanded * neg_emb, dim=2)  # [B, N]
            neg_loss = F.logsigmoid(-neg_score).sum(dim=1)  # [B]
        else:
            neg_loss = 0

        # Total loss (negative because we maximize log likelihood)
        loss = -(pos_loss + neg_loss).mean()
        return loss

    def get_embeddings(self) -> torch.Tensor:
        """Get the learned block embeddings.

        Returns center embeddings as the final representation.

        Returns:
            Embedding matrix [vocab_size, embedding_dim]
        """
        return self.center_embeddings.weight.data.clone()

    def get_similar_blocks(
        self, block_id: int, top_k: int = 10
    ) -> list[tuple[int, float]]:
        """Find most similar blocks to a given block.

        Args:
            block_id: Token ID of query block
            top_k: Number of similar blocks to return

        Returns:
            List of (token_id, similarity_score) tuples
        """
        embeddings = self.get_embeddings()

        # Normalize embeddings for cosine similarity
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)

        # Get query embedding
        query = embeddings_norm[block_id].unsqueeze(0)

        # Compute cosine similarity
        similarities = torch.mm(query, embeddings_norm.t()).squeeze(0)

        # Get top-k (excluding the query itself)
        values, indices = similarities.topk(top_k + 1)

        results = []
        for idx, sim in zip(indices.tolist(), values.tolist()):
            if idx != block_id:
                results.append((idx, sim))
            if len(results) >= top_k:
                break

        return results
