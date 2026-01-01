#!/usr/bin/env python3
"""
Training script for Sparse Structure Transformer.

A fundamentally different approach from dense VQ-VAE:
- No air blocks in representation
- Equal weight per non-air block
- Embedding prediction instead of class logits
- Directly leverages Block2Vec semantic structure

Usage:
    python scripts/train_sparse_transformer.py --epochs 20 --batch-size 16

    # Lite model for quick testing
    python scripts/train_sparse_transformer.py --lite --epochs 5

    # With VQ for generation pipeline
    python scripts/train_sparse_transformer.py --vq-codes 1024 --epochs 30
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.sparse_dataset import (
    SparseStructureDataset,
    SparseStructureDatasetWithReconstruction,
    collate_sparse,
    collate_sparse_with_dense,
    AIR_TOKENS,
    get_block_category,
)
from models.sparse_transformer import (
    SparseStructureTransformer,
    SparseStructureTransformerLite,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Sparse Structure Transformer")

    # Data paths
    parser.add_argument("--train-dir", type=str, default="data/splits/train",
                        help="Training data directory")
    parser.add_argument("--val-dir", type=str, default="data/splits/val",
                        help="Validation data directory")
    parser.add_argument("--embeddings", type=str,
                        default="data/kaggle/output/block2vec/v3/block_embeddings_v3.npy",
                        help="Path to Block2Vec embeddings")
    parser.add_argument("--vocab", type=str, default="data/vocabulary/tok2block.json",
                        help="Path to vocabulary file")

    # Model architecture
    parser.add_argument("--lite", action="store_true",
                        help="Use lite model (faster, simpler)")
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="Transformer hidden dimension")
    parser.add_argument("--n-encoder-layers", type=int, default=6,
                        help="Number of encoder layers")
    parser.add_argument("--n-decoder-layers", type=int, default=6,
                        help="Number of decoder layers")
    parser.add_argument("--n-heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--num-latent-codes", type=int, default=16,
                        help="Number of latent codes after pooling")
    parser.add_argument("--vq-codes", type=int, default=0,
                        help="VQ codebook size (0 = no VQ)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")

    # Training settings
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--aux-weight", type=float, default=0.1,
                        help="Weight for auxiliary classification loss")
    parser.add_argument("--max-blocks", type=int, default=2048,
                        help="Maximum blocks per structure")

    # Other settings
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output-dir", type=str, default="outputs/sparse_transformer",
                        help="Output directory")
    parser.add_argument("--save-every", type=int, default=5,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Limit files for debugging")

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_model(args, embed_dim: int, device: torch.device) -> nn.Module:
    """Create model based on arguments."""
    if args.lite:
        model = SparseStructureTransformerLite(
            embed_dim=embed_dim,
            hidden_dim=args.hidden_dim // 2,  # Smaller for lite
            n_layers=4,
            n_heads=4,
            dropout=args.dropout,
        )
    else:
        model = SparseStructureTransformer(
            embed_dim=embed_dim,
            hidden_dim=args.hidden_dim,
            n_encoder_layers=args.n_encoder_layers,
            n_decoder_layers=args.n_decoder_layers,
            n_heads=args.n_heads,
            num_latent_codes=args.num_latent_codes,
            vq_num_embeddings=args.vq_codes,
            dropout=args.dropout,
        )

    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    return model


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    all_embeddings: torch.Tensor,
    device: torch.device,
    aux_weight: float = 0.1,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_embed_loss = 0
    total_vq_loss = 0
    total_aux_loss = 0
    total_accuracy = 0
    n_batches = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        positions = batch["positions"].to(device)
        embeddings = batch["embeddings"].to(device)
        block_ids = batch["block_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        optimizer.zero_grad()

        # Forward pass with loss computation
        if hasattr(model, "compute_loss"):
            losses = model.compute_loss(
                positions=positions,
                embeddings=embeddings,
                block_ids=block_ids,
                attention_mask=attention_mask,
                all_embeddings=all_embeddings,
                aux_weight=aux_weight,
            )
        else:
            # Lite model doesn't have compute_loss
            outputs = model(positions, embeddings, attention_mask)
            pred_embeddings = outputs["pred_embeddings"]

            # MSE loss
            mask = attention_mask.unsqueeze(-1)
            embed_diff = (pred_embeddings - embeddings) ** 2
            embed_loss = (embed_diff * mask).sum() / mask.sum() / pred_embeddings.size(-1)

            losses = {
                "loss": embed_loss,
                "embed_loss": embed_loss,
                "vq_loss": torch.tensor(0.0),
                "aux_loss": torch.tensor(0.0),
                "accuracy": torch.tensor(0.0),
            }

        loss = losses["loss"]
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        # Track metrics
        total_loss += losses["loss"].item()
        total_embed_loss += losses["embed_loss"].item()
        total_vq_loss += losses["vq_loss"].item() if isinstance(losses["vq_loss"], torch.Tensor) else losses["vq_loss"]
        total_aux_loss += losses["aux_loss"].item() if isinstance(losses["aux_loss"], torch.Tensor) else losses["aux_loss"]
        total_accuracy += losses["accuracy"].item() if isinstance(losses["accuracy"], torch.Tensor) else losses["accuracy"]
        n_batches += 1

        pbar.set_postfix({
            "loss": f"{losses['loss'].item():.4f}",
            "acc": f"{losses['accuracy'].item():.2%}" if isinstance(losses["accuracy"], torch.Tensor) else "N/A",
        })

    return {
        "loss": total_loss / n_batches,
        "embed_loss": total_embed_loss / n_batches,
        "vq_loss": total_vq_loss / n_batches,
        "aux_loss": total_aux_loss / n_batches,
        "accuracy": total_accuracy / n_batches,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    all_embeddings: torch.Tensor,
    device: torch.device,
    tok2block: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    """Validate model and compute detailed metrics."""
    model.eval()

    total_loss = 0
    total_embed_loss = 0
    total_accuracy = 0
    total_blocks = 0
    n_batches = 0

    # Per-category tracking
    category_correct: Dict[str, int] = {}
    category_total: Dict[str, int] = {}

    # Embedding similarity tracking
    cosine_similarities = []

    pbar = tqdm(loader, desc="Validate", leave=False)
    for batch in pbar:
        positions = batch["positions"].to(device)
        embeddings = batch["embeddings"].to(device)
        block_ids = batch["block_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        num_blocks = batch["num_blocks"]

        # Forward pass
        outputs = model(positions, embeddings, attention_mask)
        pred_embeddings = outputs["pred_embeddings"]

        # Embedding loss
        mask = attention_mask.unsqueeze(-1)
        embed_diff = (pred_embeddings - embeddings) ** 2
        embed_loss = (embed_diff * mask).sum() / mask.sum() / pred_embeddings.size(-1)

        total_embed_loss += embed_loss.item()
        total_loss += embed_loss.item()

        # Predict blocks via nearest neighbor
        B, N, D = pred_embeddings.shape
        pred_flat = pred_embeddings.view(-1, D)
        distances = torch.cdist(pred_flat, all_embeddings)
        pred_ids = distances.argmin(dim=1).view(B, N)

        # Accuracy
        correct = (pred_ids == block_ids) & attention_mask
        total_accuracy += correct.sum().item()
        total_blocks += attention_mask.sum().item()

        # Cosine similarity
        pred_norm = F.normalize(pred_embeddings, dim=-1)
        target_norm = F.normalize(embeddings, dim=-1)
        cos_sim = (pred_norm * target_norm).sum(dim=-1)  # [B, N]
        valid_cos_sim = cos_sim[attention_mask]
        cosine_similarities.append(valid_cos_sim.cpu())

        # Per-category accuracy
        if tok2block is not None:
            for b in range(B):
                for i in range(num_blocks[b].item()):
                    true_id = block_ids[b, i].item()
                    pred_id = pred_ids[b, i].item()

                    if true_id in tok2block:
                        category = get_block_category(tok2block[true_id])
                        if category not in category_total:
                            category_correct[category] = 0
                            category_total[category] = 0

                        category_total[category] += 1
                        if pred_id == true_id:
                            category_correct[category] += 1

        n_batches += 1

    # Compute final metrics
    avg_accuracy = total_accuracy / total_blocks if total_blocks > 0 else 0
    all_cos_sim = torch.cat(cosine_similarities)

    # Category accuracy
    category_accuracy = {}
    for cat in category_total:
        if category_total[cat] > 0:
            category_accuracy[cat] = category_correct[cat] / category_total[cat]

    return {
        "loss": total_loss / n_batches,
        "embed_loss": total_embed_loss / n_batches,
        "accuracy": avg_accuracy,
        "cosine_similarity": all_cos_sim.mean().item(),
        "cosine_similarity_std": all_cos_sim.std().item(),
        "category_accuracy": category_accuracy,
        "total_blocks": total_blocks,
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load embeddings
    embeddings_path = Path(args.embeddings)
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")

    all_embeddings = np.load(embeddings_path).astype(np.float32)
    all_embeddings_tensor = torch.from_numpy(all_embeddings).to(device)
    embed_dim = all_embeddings.shape[1]
    vocab_size = all_embeddings.shape[0]
    print(f"Loaded embeddings: {all_embeddings.shape}")

    # Load vocabulary
    tok2block = None
    vocab_path = Path(args.vocab)
    if vocab_path.exists():
        with open(vocab_path) as f:
            tok2block = {int(k): v for k, v in json.load(f).items()}
        print(f"Loaded vocabulary: {len(tok2block)} blocks")

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = SparseStructureDataset(
        data_dir=args.train_dir,
        embeddings_path=args.embeddings,
        vocab_path=args.vocab,
        max_files=args.max_files,
        max_blocks=args.max_blocks,
        augment=True,
        seed=args.seed,
    )

    val_dataset = SparseStructureDataset(
        data_dir=args.val_dir,
        embeddings_path=args.embeddings,
        vocab_path=args.vocab,
        max_files=args.max_files // 5 if args.max_files else None,
        max_blocks=args.max_blocks,
        augment=False,
        seed=args.seed,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_sparse,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_sparse,
        pin_memory=True,
    )

    print(f"Train: {len(train_dataset)} structures, {len(train_loader)} batches")
    print(f"Val: {len(val_dataset)} structures, {len(val_loader)} batches")

    # Create model
    print("\nCreating model...")
    model = create_model(args, embed_dim, device)

    # Create optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr / 10,
    )

    # Training history
    history = {
        "train_loss": [],
        "train_embed_loss": [],
        "train_vq_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_cosine_sim": [],
        "learning_rate": [],
    }
    best_val_acc = 0.0

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, all_embeddings_tensor, device, args.aux_weight
        )

        # Validate
        val_metrics = validate(model, val_loader, all_embeddings_tensor, device, tok2block)

        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Record history
        history["train_loss"].append(train_metrics["loss"])
        history["train_embed_loss"].append(train_metrics["embed_loss"])
        history["train_vq_loss"].append(train_metrics["vq_loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_cosine_sim"].append(val_metrics["cosine_similarity"])
        history["learning_rate"].append(current_lr)

        # Print progress
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch + 1}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train: loss={train_metrics['loss']:.4f}, acc={train_metrics['accuracy']:.2%}")
        print(f"  Val:   loss={val_metrics['loss']:.4f}, acc={val_metrics['accuracy']:.2%}, "
              f"cos_sim={val_metrics['cosine_similarity']:.4f}")

        # Print category accuracy
        if val_metrics["category_accuracy"]:
            cats = sorted(val_metrics["category_accuracy"].items(), key=lambda x: -x[1])
            print("  Categories (top 5):")
            for cat, acc in cats[:5]:
                print(f"    {cat}: {acc:.2%}")

        # Save best model
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": best_val_acc,
                "args": vars(args),
            }, output_dir / "best_model.pt")
            print(f"  ✓ New best model saved (acc={best_val_acc:.2%})")

        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "args": vars(args),
            }, output_dir / f"checkpoint_epoch_{epoch + 1}.pt")

    # Training complete
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Training complete in {total_time / 60:.1f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2%}")
    print("=" * 60)

    # Save final results
    results = {
        "best_val_accuracy": best_val_acc,
        "final_val_accuracy": history["val_accuracy"][-1],
        "final_train_accuracy": history["train_accuracy"][-1],
        "training_time_minutes": total_time / 60,
        "num_epochs": args.epochs,
        "args": vars(args),
        "history": history,
    }

    with open(output_dir / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save history separately for plotting
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nResults saved to {output_dir}")


# Import F for validation
import torch.nn.functional as F


if __name__ == "__main__":
    main()

