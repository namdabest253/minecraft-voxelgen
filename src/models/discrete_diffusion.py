"""
Masked Discrete Diffusion Transformer with CLIP Conditioning.

Operates on 1024 RFSQ token indices (512 stage1 + 512 stage2, flattened
from 8x8x8 grids). Uses absorbing-state (MASK token) diffusion with a
cosine masking schedule.

Conditioning: CLIP text embeddings via AdaLN-Zero (DiT-style).

Architecture:
    Text prompt → CLIP text encoder (frozen) → [512] embedding
                                                  ↓
                                             Linear → [d_model]
                                                  ↓
    Masked tokens [1024] → Token Embed → + 3D Pos Embed
        → Transformer (AdaLN-Zero) → Token logits [1024 × 625]

Reference: "Structured Denoising Diffusion Models in Discrete State-Spaces"
           (Austin et al., 2021) — D3PM / absorbing state variant.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# AdaLN-Zero Transformer Block (DiT-style)
# ---------------------------------------------------------------------------

class AdaLNZeroBlock(nn.Module):
    """Transformer block with Adaptive Layer Norm Zero conditioning.

    The conditioning vector modulates the layer norm parameters:
        γ1, β1, α1, γ2, β2, α2 = MLP(cond)
        x = x + α1 * Attention(LN(x, γ1, β1))
        x = x + α2 * FFN(LN(x, γ2, β2))

    α (scale) is initialized to zero so the block starts as identity.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        d_ff: Feed-forward hidden dimension.
        dropout: Dropout rate.
        cond_dim: Conditioning dimension (d_model by default).
    """

    def __init__(
        self,
        d_model: int = 384,
        n_heads: int = 6,
        d_ff: int = 1536,
        dropout: float = 0.1,
        cond_dim: Optional[int] = None,
    ):
        super().__init__()

        if cond_dim is None:
            cond_dim = d_model

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)

        # Self-attention
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        # AdaLN-Zero: project conditioning to 6 modulation params
        # (γ1, β1, α1, γ2, β2, α2)
        self.adaLN_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * d_model),
        )

        # Initialize α parameters to zero (identity at init)
        nn.init.zeros_(self.adaLN_proj[1].weight)
        nn.init.zeros_(self.adaLN_proj[1].bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Forward pass with AdaLN-Zero conditioning.

        Args:
            x: Token sequence [B, L, d_model].
            cond: Conditioning vector [B, d_model].

        Returns:
            Modulated output [B, L, d_model].
        """
        # Compute modulation parameters from conditioning
        params = self.adaLN_proj(cond)  # [B, 6*d_model]
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = params.chunk(6, dim=-1)

        # Reshape for broadcasting: [B, 1, d_model]
        gamma1 = gamma1.unsqueeze(1)
        beta1 = beta1.unsqueeze(1)
        alpha1 = alpha1.unsqueeze(1)
        gamma2 = gamma2.unsqueeze(1)
        beta2 = beta2.unsqueeze(1)
        alpha2 = alpha2.unsqueeze(1)

        # Modulated self-attention
        h = self.norm1(x) * (1 + gamma1) + beta1
        h, _ = self.attn(h, h, h)
        x = x + alpha1 * h

        # Modulated feed-forward
        h = self.norm2(x) * (1 + gamma2) + beta2
        h = self.ff(h)
        x = x + alpha2 * h

        return x


# ---------------------------------------------------------------------------
# 3D Positional Encoding
# ---------------------------------------------------------------------------

class LearnedPositionalEncoding3D(nn.Module):
    """Learned 3D positional embeddings for 8x8x8 grid positions.

    Each position gets a unique learned embedding. Stage indicator is added
    to distinguish stage1 vs stage2 tokens at the same spatial position.

    Args:
        grid_size: Spatial grid size (8 for 8x8x8).
        d_model: Embedding dimension.
        num_stages: Number of RFSQ stages (2).
    """

    def __init__(self, grid_size: int = 8, d_model: int = 384, num_stages: int = 2):
        super().__init__()
        self.grid_size = grid_size
        self.num_stages = num_stages

        # Factored 3D position: x, y, z each get their own embedding
        self.pos_x = nn.Embedding(grid_size, d_model)
        self.pos_y = nn.Embedding(grid_size, d_model)
        self.pos_z = nn.Embedding(grid_size, d_model)

        # Stage indicator embedding
        self.stage_emb = nn.Embedding(num_stages, d_model)

        # Precompute position indices for the flattened sequence
        # Sequence order: stage1 x-major [0..511] then stage2 [512..1023]
        positions = []
        for stage in range(num_stages):
            for x in range(grid_size):
                for y in range(grid_size):
                    for z in range(grid_size):
                        positions.append((stage, x, y, z))

        stages = torch.tensor([p[0] for p in positions])
        xs = torch.tensor([p[1] for p in positions])
        ys = torch.tensor([p[2] for p in positions])
        zs = torch.tensor([p[3] for p in positions])

        self.register_buffer("_stages", stages)
        self.register_buffer("_xs", xs)
        self.register_buffer("_ys", ys)
        self.register_buffer("_zs", zs)

    def forward(self, batch_size: int) -> Tensor:
        """Generate positional encodings for the full sequence.

        Args:
            batch_size: Batch size B.

        Returns:
            [B, 1024, d_model] positional embeddings.
        """
        pe = (
            self.pos_x(self._xs)
            + self.pos_y(self._ys)
            + self.pos_z(self._zs)
            + self.stage_emb(self._stages)
        )  # [1024, d_model]

        return pe.unsqueeze(0).expand(batch_size, -1, -1)


# ---------------------------------------------------------------------------
# Masking Schedule
# ---------------------------------------------------------------------------

def cosine_mask_schedule(t: Tensor, s: float = 0.008) -> Tensor:
    """Cosine masking schedule: fraction of tokens to mask at timestep t.

    Args:
        t: Timestep values in [0, 1]. t=0 means no masking, t=1 means full mask.
        s: Small offset to avoid singularity.

    Returns:
        Mask ratio in [0, 1] for each timestep.
    """
    return torch.cos(((t + s) / (1 + s)) * (math.pi / 2)).clamp(0, 1)


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class MaskedDiffusionTransformer(nn.Module):
    """Masked discrete diffusion transformer for Minecraft structure generation.

    Operates on 1024 RFSQ token indices. During training, random tokens are
    replaced with a MASK token; the model predicts the original token at
    each masked position. At inference, iteratively unmasks tokens from
    fully masked to fully unmasked.

    Args:
        codebook_size: Tokens per RFSQ stage (625 for [5,5,5,5]).
        d_model: Transformer hidden dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer layers.
        clip_dim: CLIP embedding dimension (512 for ViT-B/32).
        grid_size: Spatial grid size (8).
        num_stages: Number of RFSQ stages (2).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        codebook_size: int = 625,
        d_model: int = 384,
        n_heads: int = 6,
        n_layers: int = 12,
        clip_dim: int = 512,
        grid_size: int = 8,
        num_stages: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.codebook_size = codebook_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.num_tokens = grid_size ** 3 * num_stages  # 1024
        self.mask_token_id = codebook_size  # Use index 625 as MASK

        # Token embedding: codebook_size + 1 (for MASK token)
        self.token_emb = nn.Embedding(codebook_size + 1, d_model)

        # 3D positional encoding
        self.pos_enc = LearnedPositionalEncoding3D(grid_size, d_model, num_stages)

        # Project CLIP embedding to d_model
        self.clip_proj = nn.Sequential(
            nn.Linear(clip_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Timestep embedding (sinusoidal + MLP)
        self.time_emb = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Null conditioning embedding (for CFG dropout)
        self.null_cond = nn.Parameter(torch.randn(d_model) * 0.02)

        # Transformer layers
        d_ff = d_model * 4
        self.layers = nn.ModuleList([
            AdaLNZeroBlock(d_model, n_heads, d_ff, dropout, cond_dim=d_model)
            for _ in range(n_layers)
        ])

        # Final layer norm + output projection
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, codebook_size)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following DiT conventions."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

        # Zero-init output projection for stable training start
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def _sinusoidal_timestep_emb(self, t: Tensor) -> Tensor:
        """Create sinusoidal timestep embeddings.

        Args:
            t: Timestep values [B] in [0, 1].

        Returns:
            [B, d_model] timestep embeddings.
        """
        half_dim = self.d_model // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)  # [B, half_dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # [B, d_model]
        return emb

    def forward(
        self,
        masked_tokens: Tensor,
        t: Tensor,
        clip_embedding: Tensor,
        has_condition: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass: predict original tokens at masked positions.

        Args:
            masked_tokens: [B, 1024] token indices with some replaced by mask_token_id.
            t: [B] timestep values in [0, 1].
            clip_embedding: [B, d_model] projected CLIP conditioning.
            has_condition: [B] bool tensor. If False, use null conditioning (for CFG).

        Returns:
            [B, 1024, codebook_size] logits for each position.
        """
        B = masked_tokens.shape[0]

        # Token embeddings + positional encoding
        x = self.token_emb(masked_tokens) + self.pos_enc(B)

        # Conditioning: CLIP + timestep
        t_emb = self.time_emb(self._sinusoidal_timestep_emb(t))
        cond = clip_embedding + t_emb  # [B, d_model]

        # CFG: replace conditioning with null for unconditional samples
        if has_condition is not None:
            null_cond = self.null_cond.unsqueeze(0).expand(B, -1) + t_emb
            mask = has_condition.unsqueeze(1).float()  # [B, 1]
            cond = cond * mask + null_cond * (1 - mask)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, cond)

        # Output projection
        x = self.final_norm(x)
        logits = self.output_proj(x)  # [B, 1024, codebook_size]

        return logits

    @torch.no_grad()
    def sample(
        self,
        clip_embedding: Tensor,
        num_steps: int = 50,
        cfg_scale: float = 3.0,
        temperature: float = 1.0,
        device: str = "cuda",
    ) -> Tensor:
        """Generate structures via iterative unmasking.

        Args:
            clip_embedding: [B, d_model] CLIP conditioning.
            num_steps: Number of unmasking steps.
            cfg_scale: Classifier-free guidance scale.
            temperature: Sampling temperature.
            device: Torch device.

        Returns:
            [B, 1024] generated token indices.
        """
        B = clip_embedding.shape[0]

        # Start fully masked
        tokens = torch.full(
            (B, self.num_tokens), self.mask_token_id,
            dtype=torch.long, device=device,
        )

        # Iteratively unmask
        for step in range(num_steps):
            # Current mask ratio (decreasing from 1 to 0)
            t_now = 1.0 - step / num_steps
            t_next = 1.0 - (step + 1) / num_steps

            t_tensor = torch.full((B,), t_now, device=device)

            # Get predictions with CFG
            if cfg_scale > 1.0:
                # Conditional
                logits_cond = self.forward(
                    tokens, t_tensor, clip_embedding,
                    has_condition=torch.ones(B, dtype=torch.bool, device=device),
                )
                # Unconditional
                logits_uncond = self.forward(
                    tokens, t_tensor, clip_embedding,
                    has_condition=torch.zeros(B, dtype=torch.bool, device=device),
                )
                logits = logits_uncond + cfg_scale * (logits_cond - logits_uncond)
            else:
                logits = self.forward(
                    tokens, t_tensor, clip_embedding,
                    has_condition=torch.ones(B, dtype=torch.bool, device=device),
                )

            # Apply temperature
            logits = logits / temperature

            # Compute confidence (max probability) at each position
            probs = F.softmax(logits, dim=-1)
            max_probs, predicted = probs.max(dim=-1)  # [B, 1024]

            # Only consider currently masked positions
            is_masked = (tokens == self.mask_token_id)

            # How many tokens to unmask this step
            n_masked = is_masked.sum(dim=1).float()  # [B]
            n_to_unmask = (
                cosine_mask_schedule(torch.tensor(t_next, device=device))
                - cosine_mask_schedule(torch.tensor(t_now, device=device))
            ).abs()
            n_to_unmask = (n_masked * n_to_unmask / max(t_now, 1e-8)).long().clamp(min=1)

            # Unmask highest-confidence masked positions
            # Set confidence to -inf for non-masked positions
            confidence = max_probs.clone()
            confidence[~is_masked] = -float("inf")

            for b in range(B):
                k = min(n_to_unmask[b].item(), is_masked[b].sum().item())
                if k > 0:
                    _, top_idx = confidence[b].topk(k)
                    # Sample from categorical at these positions
                    for idx in top_idx:
                        token_probs = probs[b, idx]
                        tokens[b, idx] = torch.multinomial(token_probs, 1).squeeze()

        # Fill any remaining masked tokens greedily
        remaining_mask = (tokens == self.mask_token_id)
        if remaining_mask.any():
            t_tensor = torch.zeros(B, device=device)
            logits = self.forward(
                tokens, t_tensor, clip_embedding,
                has_condition=torch.ones(B, dtype=torch.bool, device=device),
            )
            final_tokens = logits.argmax(dim=-1)
            tokens[remaining_mask] = final_tokens[remaining_mask]

        return tokens

    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Training Utilities
# ---------------------------------------------------------------------------

def apply_mask(
    tokens: Tensor,
    mask_token_id: int,
    mask_ratio: float,
) -> Tuple[Tensor, Tensor]:
    """Apply random masking to token sequence.

    Args:
        tokens: [B, L] original token indices.
        mask_token_id: ID of the MASK token.
        mask_ratio: Fraction of tokens to mask (0 to 1).

    Returns:
        masked_tokens: [B, L] tokens with some replaced by mask_token_id.
        mask: [B, L] boolean mask (True = masked position).
    """
    B, L = tokens.shape
    n_mask = max(1, int(L * mask_ratio))

    # Random mask for each sample
    noise = torch.rand(B, L, device=tokens.device)
    # Get indices of top-k random values (positions to mask)
    _, indices = noise.topk(n_mask, dim=1)

    mask = torch.zeros(B, L, dtype=torch.bool, device=tokens.device)
    mask.scatter_(1, indices, True)

    masked_tokens = tokens.clone()
    masked_tokens[mask] = mask_token_id

    return masked_tokens, mask


def training_step(
    model: MaskedDiffusionTransformer,
    tokens: Tensor,
    clip_embedding: Tensor,
    has_condition: Tensor,
) -> Tuple[Tensor, Dict[str, float]]:
    """Single training step for masked diffusion.

    1. Sample random timestep t ~ U(0, 1)
    2. Compute mask ratio from cosine schedule
    3. Apply mask
    4. Predict original tokens at masked positions
    5. Cross-entropy loss on masked positions only

    Args:
        model: The diffusion model.
        tokens: [B, 1024] ground truth token indices.
        clip_embedding: [B, d_model] conditioning vectors.
        has_condition: [B] bool — False for CFG dropout samples.

    Returns:
        (loss, metrics_dict)
    """
    B = tokens.shape[0]
    device = tokens.device

    # Sample random timestep
    t = torch.rand(B, device=device)

    # Compute mask ratio from cosine schedule
    mask_ratios = 1.0 - cosine_mask_schedule(t)  # Higher t → more masking

    # Apply masking (per-sample mask ratio)
    masked_tokens = tokens.clone()
    all_masks = torch.zeros_like(tokens, dtype=torch.bool)

    for b in range(B):
        ratio = mask_ratios[b].item()
        n_mask = max(1, int(model.num_tokens * ratio))
        perm = torch.randperm(model.num_tokens, device=device)[:n_mask]
        all_masks[b, perm] = True
        masked_tokens[b, perm] = model.mask_token_id

    # Forward pass
    logits = model(masked_tokens, t, clip_embedding, has_condition)

    # Loss on masked positions only
    logits_masked = logits[all_masks]  # [N_masked, codebook_size]
    targets_masked = tokens[all_masks]  # [N_masked]

    loss = F.cross_entropy(logits_masked, targets_masked)

    # Metrics
    with torch.no_grad():
        preds = logits_masked.argmax(dim=-1)
        accuracy = (preds == targets_masked).float().mean().item()
        avg_mask_ratio = mask_ratios.mean().item()

    metrics = {
        "loss": loss.item(),
        "accuracy": accuracy,
        "avg_mask_ratio": avg_mask_ratio,
        "n_masked": all_masks.sum().item(),
    }

    return loss, metrics


# ---------------------------------------------------------------------------
# Model Configurations
# ---------------------------------------------------------------------------

DISCRETE_DIFFUSION_CONFIGS = {
    "base": {
        "codebook_size": 625,
        "d_model": 384,
        "n_heads": 6,
        "n_layers": 12,
        "clip_dim": 512,
        "grid_size": 8,
        "num_stages": 2,
        "dropout": 0.1,
    },
    "small": {
        "codebook_size": 625,
        "d_model": 256,
        "n_heads": 4,
        "n_layers": 8,
        "clip_dim": 512,
        "grid_size": 8,
        "num_stages": 2,
        "dropout": 0.1,
    },
}


def create_model(config_name: str = "base", **overrides) -> MaskedDiffusionTransformer:
    """Create a model from a named configuration.

    Args:
        config_name: Configuration name ("base" or "small").
        **overrides: Override any config parameter.

    Returns:
        Initialized model.
    """
    if config_name not in DISCRETE_DIFFUSION_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. "
                         f"Available: {list(DISCRETE_DIFFUSION_CONFIGS.keys())}")
    config = {**DISCRETE_DIFFUSION_CONFIGS[config_name], **overrides}
    return MaskedDiffusionTransformer(**config)
