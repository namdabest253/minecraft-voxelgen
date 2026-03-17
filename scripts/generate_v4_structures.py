"""Generate structures using Diffusion Prior v4 and export to schematics."""

import json
import math
import sys
from pathlib import Path
from typing import List, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.export.schematic_exporter import export_to_schematic


# =============================================================================
# FSQ & RFSQ Quantization
# =============================================================================

class FSQ(nn.Module):
    def __init__(self, levels: List[int], eps: float = 1e-3):
        super().__init__()
        self.levels = levels
        self.dim = len(levels)
        self.eps = eps
        self.codebook_size = int(np.prod(levels))
        self.register_buffer('_levels', torch.tensor(levels, dtype=torch.float32))
        basis = []
        acc = 1
        for L in reversed(levels):
            basis.append(acc)
            acc *= L
        self.register_buffer('_basis', torch.tensor(list(reversed(basis)), dtype=torch.long))
        half_levels = [(L - 1) / 2 for L in levels]
        self.register_buffer('_half_levels', torch.tensor(half_levels, dtype=torch.float32))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_bounded = torch.tanh(z)
        z_q = self._quantize(z_bounded)
        z_q = z_bounded + (z_q - z_bounded).detach()
        indices = self._to_indices(z_q)
        return z_q, indices

    def _quantize(self, z: torch.Tensor) -> torch.Tensor:
        z_q_list = []
        for i in range(self.dim):
            half_L = self._half_levels[i]
            z_i = z[..., i] * half_L
            z_i = torch.round(z_i).clamp(-half_L, half_L) / half_L
            z_q_list.append(z_i)
        return torch.stack(z_q_list, dim=-1)

    def _to_indices(self, z_q: torch.Tensor) -> torch.Tensor:
        indices = torch.zeros(z_q.shape[:-1], dtype=torch.long, device=z_q.device)
        for i in range(self.dim):
            L = self._levels[i].long()
            half_L = self._half_levels[i]
            level_idx = ((z_q[..., i] * half_L) + half_L).round().long().clamp(0, L - 1)
            indices = indices + level_idx * self._basis[i]
        return indices


class InvertibleLayerNorm(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('stored_mean', None, persistent=False)
        self.register_buffer('stored_std', None, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.stored_mean = x.mean(dim=(1, 2, 3), keepdim=True)
        self.stored_std = x.std(dim=(1, 2, 3), keepdim=True) + self.eps
        x_norm = (x - self.stored_mean) / self.stored_std
        return x_norm * self.weight + self.bias

    def inverse(self, x_norm: torch.Tensor) -> torch.Tensor:
        x = (x_norm - self.bias) / self.weight
        return x * self.stored_std + self.stored_mean


class RFSQStage(nn.Module):
    def __init__(self, levels: List[int]):
        super().__init__()
        self.fsq = FSQ(levels)
        self.layernorm = InvertibleLayerNorm(len(levels))

    def forward(self, residual: torch.Tensor):
        z_norm = self.layernorm(residual)
        z_q_norm, indices = self.fsq(z_norm)
        z_q = self.layernorm.inverse(z_q_norm)
        new_residual = residual - z_q
        return z_q, new_residual, indices


class RFSQ(nn.Module):
    def __init__(self, levels_per_stage: List[int], num_stages: int = 2):
        super().__init__()
        self.num_stages = num_stages
        self.levels_per_stage = levels_per_stage
        self.stages = nn.ModuleList([RFSQStage(levels_per_stage) for _ in range(num_stages)])

    def forward(self, z: torch.Tensor):
        residual = z
        z_q_sum = torch.zeros_like(z)
        all_indices = []
        for stage in self.stages:
            z_q, residual, indices = stage(residual)
            z_q_sum = z_q_sum + z_q
            all_indices.append(indices)
        return z_q_sum, all_indices


# =============================================================================
# VQ-VAE Components
# =============================================================================

class ResidualBlock3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class EncoderV16b(nn.Module):
    def __init__(self, in_channels: int = 40, hidden_dim: int = 128, rfsq_dim: int = 4):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.down1 = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.res1 = ResidualBlock3D(hidden_dim)
        self.down2 = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.res2 = nn.Sequential(*[ResidualBlock3D(hidden_dim) for _ in range(4)])
        self.proj = nn.Conv3d(hidden_dim, rfsq_dim, 3, padding=1)

    def forward(self, x):
        x = self.initial(x)
        x = self.res1(self.down1(x))
        x = self.res2(self.down2(x))
        return self.proj(x)


class DualHeadDecoderV16b(nn.Module):
    def __init__(self, rfsq_dim: int = 4, hidden_dim: int = 128, num_blocks: int = 3717):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv3d(rfsq_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.res1 = nn.Sequential(*[ResidualBlock3D(hidden_dim) for _ in range(4)])
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.res2 = ResidualBlock3D(hidden_dim)
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.trunk_final = ResidualBlock3D(hidden_dim)
        self.binary_head = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.BatchNorm3d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dim // 2, 2, 3, padding=1)
        )
        self.block_head = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dim, num_blocks, 3, padding=1)
        )

    def forward(self, z_q):
        x = self.initial(z_q)
        x = self.res1(x)
        x = self.res2(self.up1(x))
        x = self.up2(x)
        features = self.trunk_final(x)
        return self.binary_head(features), self.block_head(features)


class VQVAEv16b(nn.Module):
    def __init__(
        self,
        vocab_size: int = 3717,
        emb_dim: int = 40,
        hidden_dim: int = 128,
        rfsq_levels: List[int] = None,
        num_stages: int = 2,
        pretrained_embeddings: torch.Tensor = None,
        air_tokens: Set[int] = None,
    ):
        super().__init__()
        if rfsq_levels is None:
            rfsq_levels = [5, 5, 5, 5]
        self.rfsq_dim = len(rfsq_levels)
        self.vocab_size = vocab_size
        self.air_tokens = air_tokens or {102, 576, 3352}

        self.block_emb = nn.Embedding(vocab_size, emb_dim)
        if pretrained_embeddings is not None:
            self.block_emb.weight.data.copy_(pretrained_embeddings)
            self.block_emb.weight.requires_grad = False

        self.encoder = EncoderV16b(emb_dim, hidden_dim, self.rfsq_dim)
        self.quantizer = RFSQ(rfsq_levels, num_stages)
        self.decoder = DualHeadDecoderV16b(self.rfsq_dim, hidden_dim, vocab_size)

    def quantize_and_decode(self, z: torch.Tensor) -> torch.Tensor:
        """Quantize continuous z and decode to block IDs."""
        z_q, indices = self.quantizer(z)
        z_q_permuted = z_q.permute(0, 4, 1, 2, 3)
        binary_logits, block_logits = self.decoder(z_q_permuted)
        binary_pred = binary_logits.argmax(dim=1)
        block_pred = block_logits.argmax(dim=1)
        air_token = 102
        return torch.where(binary_pred == 0, air_token, block_pred)


# =============================================================================
# Diffusion Model Components
# =============================================================================

def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class DiffusionSchedule:
    def __init__(self, num_timesteps: int = 1000, device: torch.device = None):
        self.num_timesteps = num_timesteps
        self.device = device or torch.device('cpu')

        betas = cosine_beta_schedule(num_timesteps)
        self.betas = betas.to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int = None):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)

        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_ch * 2),
            )
        else:
            self.time_mlp = None

        self.res_conv = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor = None) -> torch.Tensor:
        h = self.norm1(F.silu(self.conv1(x)))

        if self.time_mlp is not None and t_emb is not None:
            t_emb = self.time_mlp(t_emb)[:, :, None, None, None]
            scale, shift = t_emb.chunk(2, dim=1)
            h = h * (1 + scale) + shift

        h = self.norm2(F.silu(self.conv2(h)))
        return h + self.res_conv(x)


class Downsample3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class DiffusionUNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        channels: List[int] = None,
        time_emb_dim: int = 128,
    ):
        super().__init__()
        if channels is None:
            channels = [64, 128, 256]

        self.channels = channels
        self.num_levels = len(channels)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.GELU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        self.init_conv = nn.Conv3d(in_channels, channels[0], 3, padding=1)

        self.enc_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i in range(self.num_levels):
            in_ch = channels[i-1] if i > 0 else channels[0]
            out_ch = channels[i]
            self.enc_blocks.append(ConvBlock3D(in_ch, out_ch, time_emb_dim))
            self.enc_blocks.append(ConvBlock3D(out_ch, out_ch, time_emb_dim))
            if i < self.num_levels - 1:
                self.downsamples.append(Downsample3D(out_ch))

        self.mid_block1 = ConvBlock3D(channels[-1], channels[-1], time_emb_dim)
        self.mid_block2 = ConvBlock3D(channels[-1], channels[-1], time_emb_dim)

        self.dec_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        reversed_channels = list(reversed(channels))
        for i in range(self.num_levels):
            out_ch = reversed_channels[i]
            if i == 0:
                in_ch = reversed_channels[0]
            else:
                in_ch = reversed_channels[i-1] + reversed_channels[i]
                self.upsamples.append(Upsample3D(reversed_channels[i-1]))
            self.dec_blocks.append(ConvBlock3D(in_ch, out_ch, time_emb_dim))
            self.dec_blocks.append(ConvBlock3D(out_ch, out_ch, time_emb_dim))

        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, channels[0]),
            nn.SiLU(),
            nn.Conv3d(channels[0], out_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t.float())
        x = self.init_conv(x)

        skips = []
        block_idx = 0
        for i in range(self.num_levels):
            x = self.enc_blocks[block_idx](x, t_emb)
            block_idx += 1
            x = self.enc_blocks[block_idx](x, t_emb)
            block_idx += 1
            if i < self.num_levels - 1:
                skips.append(x)
                x = self.downsamples[i](x)

        x = self.mid_block1(x, t_emb)
        x = self.mid_block2(x, t_emb)

        block_idx = 0
        for i in range(self.num_levels):
            if i > 0:
                x = self.upsamples[i-1](x)
                skip = skips.pop()
                x = torch.cat([x, skip], dim=1)
            x = self.dec_blocks[block_idx](x, t_emb)
            block_idx += 1
            x = self.dec_blocks[block_idx](x, t_emb)
            block_idx += 1

        return self.final_conv(x)


# =============================================================================
# Generation Functions
# =============================================================================

@torch.no_grad()
def generate_latents_ddim(
    model: DiffusionUNet3D,
    schedule: DiffusionSchedule,
    num_samples: int = 1,
    num_inference_steps: int = 100,
    device: torch.device = None,
) -> torch.Tensor:
    """Generate latent codes using DDIM sampling."""
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    step_size = schedule.num_timesteps // num_inference_steps
    timesteps = list(range(0, schedule.num_timesteps, step_size))[::-1]

    # Start from random noise
    z = torch.randn(num_samples, 4, 8, 8, 8, device=device)

    print(f"Generating {num_samples} samples with {num_inference_steps} DDIM steps...")
    for i, t in enumerate(timesteps):
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
        noise_pred = model(z, t_batch)

        alpha_t = schedule.alphas_cumprod[t]
        alpha_t_prev = schedule.alphas_cumprod[timesteps[i + 1]] if i < len(timesteps) - 1 else torch.tensor(1.0)

        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
        x_0_pred = (z - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
        x_0_pred = torch.clamp(x_0_pred, -10, 10)

        direction = torch.sqrt(1 - alpha_t_prev) * noise_pred
        z = torch.sqrt(alpha_t_prev) * x_0_pred + direction

    # Return in [B, 8, 8, 8, 4] format for VQ-VAE
    return z.permute(0, 2, 3, 4, 1)


@torch.no_grad()
def generate_structures(
    diffusion_model: DiffusionUNet3D,
    vqvae: VQVAEv16b,
    schedule: DiffusionSchedule,
    num_samples: int = 1,
    num_inference_steps: int = 100,
    device: torch.device = None,
) -> torch.Tensor:
    """Generate full structures from noise."""
    z = generate_latents_ddim(diffusion_model, schedule, num_samples, num_inference_steps, device)
    print("Decoding latents through VQ-VAE...")
    structures = vqvae.quantize_and_decode(z)
    return structures


# =============================================================================
# Main
# =============================================================================

def main():
    # Paths
    BASE_PATH = Path(__file__).parent.parent
    DATA_PATH = BASE_PATH / "data"

    DIFFUSION_MODEL_PATH = DATA_PATH / "output" / "prior" / "diffusion_v4" / "diffusion_prior_v4_best.pt"
    VQVAE_MODEL_PATH = DATA_PATH / "output" / "vqvae" / "v16b" / "vqvae_v16b_best.pt"
    EMBEDDINGS_PATH = DATA_PATH / "output" / "block2vec" / "v3" / "block_embeddings_v3.npy"
    VOCAB_PATH = DATA_PATH / "vocabulary" / "tok2block.json"

    OUTPUT_DIR = Path(r"C:\Users\namda\OneDrive\Desktop\Claude_Server\server\plugins\FastAsyncWorldEdit\schematics")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load vocabulary
    print(f"\nLoading vocabulary from {VOCAB_PATH}")
    with open(VOCAB_PATH, 'r') as f:
        tok2block = {int(k): v for k, v in json.load(f).items()}
    vocab_size = len(tok2block)
    print(f"Vocabulary size: {vocab_size}")

    air_tokens = {102, 576, 3352}

    # Load embeddings
    print(f"Loading embeddings from {EMBEDDINGS_PATH}")
    embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)
    embeddings_tensor = torch.from_numpy(embeddings)
    print(f"Embeddings shape: {embeddings.shape}")

    # Load VQ-VAE
    print(f"\nLoading VQ-VAE v16b from {VQVAE_MODEL_PATH}")
    vqvae = VQVAEv16b(
        vocab_size=vocab_size,
        emb_dim=40,
        hidden_dim=128,
        rfsq_levels=[5, 5, 5, 5],
        num_stages=2,
        pretrained_embeddings=embeddings_tensor,
        air_tokens=air_tokens,
    ).to(device)

    vqvae.load_state_dict(torch.load(VQVAE_MODEL_PATH, map_location=device, weights_only=False))
    vqvae.eval()
    print("VQ-VAE loaded!")

    # Load Diffusion model
    print(f"\nLoading Diffusion Prior v4 from {DIFFUSION_MODEL_PATH}")
    diffusion_model = DiffusionUNet3D(
        in_channels=4,
        out_channels=4,
        channels=[64, 128, 256],
        time_emb_dim=128,
    ).to(device)

    diffusion_model.load_state_dict(torch.load(DIFFUSION_MODEL_PATH, map_location=device, weights_only=False))
    diffusion_model.eval()
    print("Diffusion model loaded!")

    # Create diffusion schedule
    schedule = DiffusionSchedule(num_timesteps=1000, device=device)

    # Generate structures
    num_samples = 10
    print(f"\n{'='*60}")
    print(f"GENERATING {num_samples} STRUCTURES")
    print(f"{'='*60}")

    structures = generate_structures(
        diffusion_model, vqvae, schedule,
        num_samples=num_samples,
        num_inference_steps=100,
        device=device,
    )

    print(f"\nGenerated {structures.shape[0]} structures of shape {structures.shape[1:]}")

    # Analyze structures
    print("\n" + "="*60)
    print("ANALYZING GENERATED STRUCTURES")
    print("="*60)

    air_tokens_tensor = torch.tensor(list(air_tokens), device=device)

    for i in range(structures.shape[0]):
        struct = structures[i]
        is_air = torch.isin(struct, air_tokens_tensor)
        non_air_count = (~is_air).sum().item()
        total = struct.numel()

        unique_blocks = torch.unique(struct[~is_air])

        print(f"\nStructure {i+1}:")
        print(f"  Non-air blocks: {non_air_count:,} / {total:,} ({100*non_air_count/total:.1f}%)")
        print(f"  Unique block types: {len(unique_blocks)}")

        # Show sample block names
        if len(unique_blocks) > 0:
            sample_blocks = [tok2block.get(b.item(), 'unknown')[:40] for b in unique_blocks[:5]]
            print(f"  Sample blocks: {sample_blocks}")

    # Export to schematics
    print("\n" + "="*60)
    print("EXPORTING TO SCHEMATICS")
    print("="*60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(structures.shape[0]):
        output_path = OUTPUT_DIR / f"diffusion_v4_gen_{i+1}.schem"
        export_to_schematic(
            structures[i],
            tok2block,
            output_path,
            width=32,
            height=32,
            length=32,
            air_tokens=air_tokens,
        )

    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print(f"\nGenerated {num_samples} schematics in:")
    print(f"  {OUTPUT_DIR}")
    print(f"\nTo use in Minecraft:")
    print(f"  1. //schem load diffusion_v4_gen_1")
    print(f"  2. //paste")


if __name__ == "__main__":
    main()
