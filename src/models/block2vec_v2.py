"""
Block2Vec V2: Hybrid Skip-gram + CBOW for Minecraft block embeddings.

V2 addresses V1 issues:
- Combines Skip-gram (co-occurrence) with CBOW (distributional similarity)
- Works with collapsed block states
- Tracks separate losses for debugging

Architecture:
    Skip-gram: center -> predicts each context block
    CBOW: average of contexts -> predicts center block
    Total loss = alpha * sg_loss + beta * cbow_loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class Block2VecV2(nn.Module):
    """Hybrid Skip-gram + CBOW model for block embeddings.

    This model learns embeddings that capture:
    - Co-occurrence: blocks that appear together (skip-gram)
    - Distributional similarity: blocks with similar contexts (CBOW)

    Args:
        vocab_size: Number of unique block tokens
        embedding_dim: Dimension of embeddings (default 32)
        alpha: Weight for skip-gram loss (default 1.0)
        beta: Weight for CBOW loss (default 1.0)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 32,
        alpha: float = 1.0,
        beta: float = 1.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.alpha = alpha
        self.beta = beta

        # Two separate embeddings: center (input) and context (output)
        # This is standard for both skip-gram and CBOW
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Initialize with small random values
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize embeddings with uniform distribution."""
        init_range = 0.5 / self.embedding_dim
        self.center_embeddings.weight.data.uniform_(-init_range, init_range)
        self.context_embeddings.weight.data.uniform_(-init_range, init_range)

    def skipgram_loss(
        self,
        center_ids: torch.Tensor,
        context_ids: torch.Tensor,
        negative_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute skip-gram loss: center predicts context.

        Args:
            center_ids: Center block tokens [batch_size]
            context_ids: Positive context tokens [batch_size]
            negative_ids: Negative sample tokens [batch_size, num_negatives]

        Returns:
            Skip-gram loss (scalar)
        """
        # Get embeddings
        center_emb = self.center_embeddings(center_ids)  # [B, D]
        context_emb = self.context_embeddings(context_ids)  # [B, D]

        # Positive score: dot product
        pos_score = torch.sum(center_emb * context_emb, dim=1)  # [B]
        pos_loss = F.logsigmoid(pos_score)  # [B]

        # Negative sampling loss
        neg_emb = self.context_embeddings(negative_ids)  # [B, N, D]
        center_emb_expanded = center_emb.unsqueeze(1)  # [B, 1, D]
        neg_score = torch.sum(center_emb_expanded * neg_emb, dim=2)  # [B, N]
        neg_loss = F.logsigmoid(-neg_score).sum(dim=1)  # [B]

        # Total loss (negative because we maximize log likelihood)
        loss = -(pos_loss + neg_loss).mean()
        return loss

    def cbow_loss(
        self,
        center_ids: torch.Tensor,
        context_ids: torch.Tensor,
        context_mask: torch.Tensor,
        negative_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CBOW loss: context average predicts center.

        Args:
            center_ids: Center block tokens [batch_size]
            context_ids: Context tokens [batch_size, num_contexts]
            context_mask: Mask for valid contexts [batch_size, num_contexts]
            negative_ids: Negative sample tokens [batch_size, num_negatives]

        Returns:
            CBOW loss (scalar)
        """
        # Get context embeddings and average them
        context_emb = self.center_embeddings(context_ids)  # [B, C, D]

        # Apply mask and compute mean (handle variable number of contexts)
        mask_expanded = context_mask.unsqueeze(-1).float()  # [B, C, 1]
        context_sum = (context_emb * mask_expanded).sum(dim=1)  # [B, D]
        context_count = context_mask.sum(dim=1, keepdim=True).float().clamp(min=1)  # [B, 1]
        context_avg = context_sum / context_count  # [B, D]

        # Center embedding (as output/context embedding for CBOW)
        center_emb = self.context_embeddings(center_ids)  # [B, D]

        # Positive score: dot product of context avg with center
        pos_score = torch.sum(context_avg * center_emb, dim=1)  # [B]
        pos_loss = F.logsigmoid(pos_score)  # [B]

        # Negative sampling loss
        neg_emb = self.context_embeddings(negative_ids)  # [B, N, D]
        context_avg_expanded = context_avg.unsqueeze(1)  # [B, 1, D]
        neg_score = torch.sum(context_avg_expanded * neg_emb, dim=2)  # [B, N]
        neg_loss = F.logsigmoid(-neg_score).sum(dim=1)  # [B]

        # Total loss
        loss = -(pos_loss + neg_loss).mean()
        return loss

    def forward(
        self,
        center_ids: torch.Tensor,
        context_ids: torch.Tensor,
        context_mask: torch.Tensor,
        negative_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute combined skip-gram + CBOW loss.

        For skip-gram, we iterate through each context position.
        For CBOW, we use all contexts together.

        Args:
            center_ids: Center block tokens [batch_size]
            context_ids: All context tokens [batch_size, num_contexts]
            context_mask: Mask for valid contexts [batch_size, num_contexts]
            negative_ids: Negative sample tokens [batch_size, num_negatives]

        Returns:
            Dict with 'total_loss', 'skipgram_loss', 'cbow_loss'
        """
        batch_size = center_ids.shape[0]
        num_contexts = context_ids.shape[1]
        device = center_ids.device

        # Skip-gram loss: for each valid context
        sg_losses = []
        for ctx_idx in range(num_contexts):
            ctx = context_ids[:, ctx_idx]  # [B]
            mask = context_mask[:, ctx_idx]  # [B]

            if mask.sum() == 0:
                continue

            # Only compute for valid contexts
            valid_center = center_ids[mask]
            valid_ctx = ctx[mask]
            valid_neg = negative_ids[mask]

            if valid_center.shape[0] > 0:
                sg_loss = self.skipgram_loss(valid_center, valid_ctx, valid_neg)
                sg_losses.append(sg_loss * mask.sum() / batch_size)

        if sg_losses:
            skipgram_loss = torch.stack(sg_losses).sum() / num_contexts
        else:
            skipgram_loss = torch.tensor(0.0, device=device)

        # CBOW loss: all contexts together
        cbow_loss = self.cbow_loss(center_ids, context_ids, context_mask, negative_ids)

        # Combined loss
        total_loss = self.alpha * skipgram_loss + self.beta * cbow_loss

        return {
            "total_loss": total_loss,
            "skipgram_loss": skipgram_loss,
            "cbow_loss": cbow_loss,
        }

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


class Block2VecV2Simple(nn.Module):
    """Simplified V2 model for easier training.

    Instead of iterating through contexts, this version takes
    a single context per training sample (like V1) but adds
    a CBOW loss term using a sliding window.

    This is more memory efficient and easier to train.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 32,
        alpha: float = 1.0,
        beta: float = 1.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.alpha = alpha
        self.beta = beta

        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        init_range = 0.5 / self.embedding_dim
        self.center_embeddings.weight.data.uniform_(-init_range, init_range)
        self.context_embeddings.weight.data.uniform_(-init_range, init_range)

    def forward(
        self,
        center_ids: torch.Tensor,
        context_id: torch.Tensor,
        negative_ids: torch.Tensor,
        all_context_ids: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute loss.

        Args:
            center_ids: Center block tokens [batch_size]
            context_id: Single positive context [batch_size]
            negative_ids: Negative samples [batch_size, num_negatives]
            all_context_ids: All context tokens for CBOW [batch_size, num_contexts]
            context_mask: Mask for valid contexts [batch_size, num_contexts]

        Returns:
            Dict with losses
        """
        # Skip-gram loss
        center_emb = self.center_embeddings(center_ids)  # [B, D]
        context_emb = self.context_embeddings(context_id)  # [B, D]

        pos_score = torch.sum(center_emb * context_emb, dim=1)
        pos_loss = F.logsigmoid(pos_score)

        neg_emb = self.context_embeddings(negative_ids)  # [B, N, D]
        neg_score = torch.sum(center_emb.unsqueeze(1) * neg_emb, dim=2)
        neg_loss = F.logsigmoid(-neg_score).sum(dim=1)

        skipgram_loss = -(pos_loss + neg_loss).mean()

        # CBOW loss (if context provided)
        if all_context_ids is not None and context_mask is not None:
            ctx_emb = self.center_embeddings(all_context_ids)  # [B, C, D]
            mask_exp = context_mask.unsqueeze(-1).float()
            ctx_sum = (ctx_emb * mask_exp).sum(dim=1)
            ctx_count = context_mask.sum(dim=1, keepdim=True).float().clamp(min=1)
            ctx_avg = ctx_sum / ctx_count

            center_out = self.context_embeddings(center_ids)
            cbow_pos = torch.sum(ctx_avg * center_out, dim=1)
            cbow_pos_loss = F.logsigmoid(cbow_pos)

            cbow_neg_score = torch.sum(ctx_avg.unsqueeze(1) * neg_emb, dim=2)
            cbow_neg_loss = F.logsigmoid(-cbow_neg_score).sum(dim=1)

            cbow_loss = -(cbow_pos_loss + cbow_neg_loss).mean()
        else:
            cbow_loss = torch.tensor(0.0, device=center_ids.device)

        total_loss = self.alpha * skipgram_loss + self.beta * cbow_loss

        return {
            "total_loss": total_loss,
            "skipgram_loss": skipgram_loss,
            "cbow_loss": cbow_loss,
        }

    def get_embeddings(self) -> torch.Tensor:
        return self.center_embeddings.weight.data.clone()
