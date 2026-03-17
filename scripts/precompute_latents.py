"""
Precompute RFSQ latent indices for all filtered structures.

Runs each H5 file through the frozen VQ-VAE v16b encoder + RFSQ quantizer
to produce stage1 and stage2 token indices for discrete diffusion training.

Input: Filtered H5 files (from clip_filter_and_label.py)
Output: minecraft_ai/data/output/latent_indices/{filename}.pt
        Each .pt file contains {"stage1": [8,8,8], "stage2": [8,8,8]}

Usage:
    python minecraft_ai/scripts/precompute_latents.py
    python minecraft_ai/scripts/precompute_latents.py --batch-size 16 --max-files 100

Requires: VQ-VAE v16b checkpoint + layernorm_stats.pt
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_vqvae_v16b(
    checkpoint_path: Path,
    layernorm_stats_path: Path,
    device: str = "cuda",
) -> torch.nn.Module:
    """Load frozen VQ-VAE v16b with RFSQ fixed stats.

    The v16b model architecture is defined in the training notebook.
    We reconstruct it here from the checkpoint's state dict.

    Args:
        checkpoint_path: Path to vqvae_v16b_best.pt
        layernorm_stats_path: Path to layernorm_stats.pt
        device: Torch device.

    Returns:
        Frozen VQ-VAE model.
    """
    # Load checkpoint to inspect architecture
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract state dict (may be nested under 'model_state_dict')
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Detect architecture from state dict keys
    has_dual_head = any("binary_head" in k or "dual" in k.lower() for k in state_dict.keys())

    # Build model — import from the models module
    # v16b uses custom architecture; we need to build it matching the checkpoint
    from src.models.rfsq import RFSQ

    # Infer dimensions from state dict
    emb_weight = state_dict.get("block_emb.weight", state_dict.get("block_embeddings.weight"))
    vocab_size, emb_dim = emb_weight.shape

    # Find hidden dim from first conv layer
    for k, v in state_dict.items():
        if "encoder" in k and "conv" in k.lower() and v.dim() == 5:
            hidden_dim = v.shape[0]
            break
    else:
        hidden_dim = 128  # default for v16b

    # Find RFSQ dim from projection layer
    for k, v in state_dict.items():
        if "proj" in k and "encoder" in k:
            rfsq_dim = v.shape[0]
            break
    else:
        rfsq_dim = 4

    print(f"Detected architecture: vocab={vocab_size}, emb={emb_dim}, "
          f"hidden={hidden_dim}, rfsq_dim={rfsq_dim}")

    # Try loading with VQVAEv4 first (it has RFSQ support in some versions)
    # If that fails, we build a minimal encoder-only model
    try:
        from src.models.vqvae import VQVAEv4

        model = VQVAEv4(
            vocab_size=vocab_size,
            block_embedding_dim=emb_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            latent_dim=rfsq_dim,
        )
        model.load_state_dict(state_dict, strict=False)
    except Exception:
        # Build a minimal wrapper that just needs encoder + quantizer
        model = _build_minimal_v16b(state_dict, vocab_size, emb_dim, hidden_dim, rfsq_dim)

    model = model.to(device)
    model.eval()

    # Load LayerNorm fixed stats
    if layernorm_stats_path.exists():
        stats = torch.load(layernorm_stats_path, map_location=device, weights_only=False)
        if hasattr(model, "quantizer") and hasattr(model.quantizer, "set_fixed_stats"):
            model.quantizer.set_fixed_stats(stats)
            model.quantizer.enable_fixed_stats(True)
            print("Loaded LayerNorm fixed stats")

    return model


def _build_minimal_v16b(state_dict, vocab_size, emb_dim, hidden_dim, rfsq_dim):
    """Build minimal v16b model from state dict when standard loading fails."""
    import torch.nn as nn

    from src.models.rfsq import RFSQ

    class MinimalV16b(nn.Module):
        def __init__(self):
            super().__init__()
            self.block_emb = nn.Embedding(vocab_size, emb_dim)

            # Reconstruct encoder from state dict structure
            self.encoder = nn.Sequential(
                nn.Conv3d(emb_dim, hidden_dim, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(hidden_dim, rfsq_dim, 3, padding=1),
            )

            self.quantizer = RFSQ([5, 5, 5, 5], num_stages=2)

        def encode(self, block_ids):
            embedded = self.block_emb(block_ids)
            embedded = embedded.permute(0, 4, 1, 2, 3).contiguous()
            return self.encoder(embedded)

    model = MinimalV16b()
    # Load matching keys
    model_dict = model.state_dict()
    matched = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(matched)
    model.load_state_dict(model_dict, strict=False)
    print(f"Loaded {len(matched)}/{len(model_dict)} parameters")
    return model


@torch.no_grad()
def precompute_batch(
    model: torch.nn.Module,
    block_ids_batch: torch.Tensor,
    device: str = "cuda",
) -> List[Dict[str, torch.Tensor]]:
    """Run encoder + RFSQ on a batch.

    Args:
        model: Frozen VQ-VAE v16b.
        block_ids_batch: [B, 32, 32, 32] block IDs.
        device: Torch device.

    Returns:
        List of dicts, one per sample: {"stage1": [8,8,8], "stage2": [8,8,8]}
    """
    block_ids_batch = block_ids_batch.to(device)

    # Encode to continuous latent
    z_e = model.encode(block_ids_batch)

    # Convert to channels-last for RFSQ: [B, C, 8, 8, 8] -> [B, 8, 8, 8, C]
    z_e_cl = z_e.permute(0, 2, 3, 4, 1).contiguous()

    # Quantize with RFSQ
    _, all_indices = model.quantizer(z_e_cl)

    # all_indices is a list of 2 tensors, each [B, 8, 8, 8]
    results = []
    batch_size = block_ids_batch.shape[0]
    for i in range(batch_size):
        results.append({
            "stage1": all_indices[0][i].cpu(),  # [8, 8, 8]
            "stage2": all_indices[1][i].cpu(),  # [8, 8, 8]
        })

    return results


def precompute_all(
    h5_files: List[Path],
    output_dir: Path,
    model: torch.nn.Module,
    device: str = "cuda",
    batch_size: int = 32,
) -> Dict:
    """Precompute RFSQ indices for all H5 files.

    Args:
        h5_files: List of H5 file paths.
        output_dir: Where to save .pt files.
        model: Frozen VQ-VAE model.
        device: Torch device.
        batch_size: Batch size for inference.

    Returns:
        Stats dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {"total": len(h5_files), "computed": 0, "skipped": 0, "errors": 0}
    start_time = time.time()

    # Process in batches
    batch_paths = []
    batch_blocks = []

    def flush_batch():
        if not batch_blocks:
            return
        block_tensor = torch.stack(batch_blocks)
        results = precompute_batch(model, block_tensor, device)

        for path, result in zip(batch_paths, results):
            out_path = output_dir / f"{path.stem}.pt"
            torch.save(result, out_path)
            stats["computed"] += 1

        batch_paths.clear()
        batch_blocks.clear()

    for h5_path in tqdm(h5_files, desc="Precomputing latents"):
        # Skip if already computed
        out_path = output_dir / f"{h5_path.stem}.pt"
        if out_path.exists():
            stats["skipped"] += 1
            continue

        try:
            with h5py.File(h5_path, "r") as f:
                key = list(f.keys())[0]
                block_ids = f[key][:].astype(np.int64)

            if block_ids.shape != (32, 32, 32):
                stats["errors"] += 1
                continue

            batch_paths.append(h5_path)
            batch_blocks.append(torch.from_numpy(block_ids).long())

            if len(batch_blocks) >= batch_size:
                flush_batch()

        except Exception as e:
            stats["errors"] += 1
            if stats["errors"] <= 5:
                print(f"\n  Error: {h5_path.name}: {e}")

    # Flush remaining
    flush_batch()

    elapsed = time.time() - start_time
    stats["elapsed_seconds"] = elapsed
    print(f"\nPrecomputation complete in {elapsed:.1f}s")
    print(f"  Computed: {stats['computed']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Errors: {stats['errors']}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Precompute RFSQ latent indices")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--splits-file", type=str, default=None,
                        help="Path to filtered_splits.json (uses all H5 dirs if not specified)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_path = PROJECT_ROOT / "data" / "output" / "vqvae" / "v16b" / "vqvae_v16b_best.pt"
    layernorm_path = PROJECT_ROOT / "data" / "output" / "vqvae" / "v16b" / "layernorm_stats.pt"
    output_dir = PROJECT_ROOT / "data" / "output" / "latent_indices"

    if not checkpoint_path.exists():
        print(f"ERROR: VQ-VAE checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Load model
    print("Loading VQ-VAE v16b...")
    model = load_vqvae_v16b(checkpoint_path, layernorm_path, device)

    # Collect H5 files
    if args.splits_file:
        splits_path = Path(args.splits_file)
        with open(splits_path) as f:
            splits = json.load(f)
        # Find H5 files from filtered splits
        h5_dirs = [
            PROJECT_ROOT / "data" / "splits" / "train",
            PROJECT_ROOT / "data" / "splits" / "val",
            PROJECT_ROOT / "data" / "splits" / "test",
            PROJECT_ROOT.parent / "kaggle" / "processed_h5",
        ]
        h5_index = {}
        for d in h5_dirs:
            if d.exists():
                for f in d.glob("*.h5"):
                    h5_index[f.name] = f

        filenames = splits.get("train", []) + splits.get("val", []) + splits.get("test", [])
        h5_files = [h5_index[fn] for fn in filenames if fn in h5_index]
    else:
        h5_dirs = [
            PROJECT_ROOT / "data" / "splits" / "train",
            PROJECT_ROOT / "data" / "splits" / "val",
            PROJECT_ROOT / "data" / "splits" / "test",
            PROJECT_ROOT.parent / "kaggle" / "processed_h5",
        ]
        h5_files = []
        for d in h5_dirs:
            if d.exists():
                h5_files.extend(sorted(d.glob("*.h5")))

    if args.max_files > 0:
        h5_files = h5_files[:args.max_files]

    print(f"Files to process: {len(h5_files)}")

    precompute_all(h5_files, output_dir, model, device, args.batch_size)


if __name__ == "__main__":
    main()
