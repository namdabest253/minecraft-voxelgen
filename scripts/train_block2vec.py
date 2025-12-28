"""
Training script for Block2Vec.

Usage:
    python scripts/train_block2vec.py
    python scripts/train_block2vec.py --epochs 100 --batch-size 8192
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.block2vec import Block2Vec
from src.data.block2vec_dataset import Block2VecDataset, collate_block2vec


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_block2vec(
    data_dir: str,
    vocab_path: str,
    output_dir: str,
    embedding_dim: int = 32,
    epochs: int = 50,
    batch_size: int = 4096,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    context_type: str = "neighbors_6",
    num_negative_samples: int = 5,
    subsample_threshold: float = 0.001,
    negative_sampling_power: float = 0.75,
    include_air: bool = True,
    save_every_epochs: int = 10,
    log_every_steps: int = 1000,
    num_workers: int = 0,
    seed: int = 42,
    device: str = "auto",
) -> dict:
    """Train Block2Vec model.

    Args:
        data_dir: Directory with H5 training files
        vocab_path: Path to tok2block.json vocabulary
        output_dir: Directory to save model and embeddings
        embedding_dim: Dimension of embeddings
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate for AdamW
        weight_decay: L2 regularization
        context_type: "neighbors_6" or "neighbors_26"
        num_negative_samples: Negatives per positive pair
        subsample_threshold: Threshold for subsampling frequent blocks
        negative_sampling_power: Power for negative sampling distribution
        include_air: Whether to include air blocks
        save_every_epochs: Checkpoint frequency
        log_every_steps: Logging frequency
        num_workers: DataLoader workers
        seed: Random seed
        device: "auto", "cuda", or "cpu"

    Returns:
        Training statistics dict
    """
    # Set device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Set random seed
    torch.manual_seed(seed)

    # Load vocabulary
    with open(vocab_path, "r") as f:
        tok2block = json.load(f)
    vocab_size = len(tok2block)
    print(f"Vocabulary size: {vocab_size}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create dataset
    print(f"Loading data from {data_dir}...")
    dataset = Block2VecDataset(
        data_dir=data_dir,
        vocab_size=vocab_size,
        context_type=context_type,
        num_negative_samples=num_negative_samples,
        subsample_threshold=subsample_threshold,
        negative_sampling_power=negative_sampling_power,
        include_air=include_air,
        seed=seed,
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_block2vec,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    # Create model
    model = Block2Vec(vocab_size=vocab_size, embedding_dim=embedding_dim)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer with cosine learning rate schedule
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Training loop
    stats = {
        "epoch_losses": [],
        "step_losses": [],
        "total_pairs": 0,
        "training_time": 0,
    }

    print(f"\nStarting training for {epochs} epochs...")
    print("=" * 60)

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_pairs = 0
        step = 0

        epoch_start = time.time()

        # Progress bar for this epoch
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

        for center_ids, context_ids, negative_ids in pbar:
            # Move to device
            center_ids = center_ids.to(device)
            context_ids = context_ids.to(device)
            negative_ids = negative_ids.to(device)

            # Forward pass
            loss = model(center_ids, context_ids, negative_ids)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track stats
            batch_size_actual = center_ids.size(0)
            epoch_loss += loss.item() * batch_size_actual
            epoch_pairs += batch_size_actual
            step += 1

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Log periodically
            if step % log_every_steps == 0:
                stats["step_losses"].append(
                    {"epoch": epoch + 1, "step": step, "loss": loss.item()}
                )

        # Epoch complete
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / max(epoch_pairs, 1)
        stats["epoch_losses"].append(avg_loss)
        stats["total_pairs"] += epoch_pairs

        print(
            f"Epoch {epoch + 1}: loss={avg_loss:.4f}, "
            f"pairs={epoch_pairs:,}, time={epoch_time:.1f}s"
        )

        # Save checkpoint
        if (epoch + 1) % save_every_epochs == 0:
            checkpoint_path = output_path / f"block2vec_epoch_{epoch + 1}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                },
                checkpoint_path,
            )
            print(f"  Saved checkpoint: {checkpoint_path}")

    # Training complete
    stats["training_time"] = time.time() - start_time

    print("=" * 60)
    print(f"Training complete in {stats['training_time']:.1f}s")
    print(f"Total pairs processed: {stats['total_pairs']:,}")

    # Save final model
    final_model_path = output_path / "block2vec_final.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model: {final_model_path}")

    # Save embeddings as numpy file for easy loading
    embeddings = model.get_embeddings().cpu().numpy()
    embeddings_path = output_path / "block_embeddings.npy"
    import numpy as np
    np.save(embeddings_path, embeddings)
    print(f"Saved embeddings: {embeddings_path} (shape: {embeddings.shape})")

    # Save training stats
    stats_path = output_path / "training_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved training stats: {stats_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Train Block2Vec embeddings")

    # Data paths
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/splits/train",
        help="Directory with H5 training files",
    )
    parser.add_argument(
        "--vocab-path",
        type=str,
        default="data/vocabulary/tok2block.json",
        help="Path to vocabulary file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/block2vec",
        help="Output directory for model and embeddings",
    )

    # Model
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=32,
        help="Embedding dimension",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--weight-decay", type=float, default=0.0001, help="Weight decay"
    )

    # Context
    parser.add_argument(
        "--context-type",
        type=str,
        default="neighbors_6",
        choices=["neighbors_6", "neighbors_26"],
        help="Context window type",
    )
    parser.add_argument(
        "--num-negatives", type=int, default=5, help="Negative samples per pair"
    )

    # Subsampling
    parser.add_argument(
        "--subsample-threshold",
        type=float,
        default=0.001,
        help="Subsampling threshold for frequent blocks",
    )
    parser.add_argument(
        "--include-air",
        action="store_true",
        default=True,
        help="Include air blocks in training",
    )
    parser.add_argument(
        "--no-air",
        action="store_true",
        help="Exclude air blocks from training",
    )

    # Misc
    parser.add_argument(
        "--save-every", type=int, default=10, help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--log-every", type=int, default=1000, help="Log every N steps"
    )
    parser.add_argument(
        "--num-workers", type=int, default=0, help="DataLoader workers"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use",
    )

    args = parser.parse_args()

    # Resolve paths relative to project root
    data_dir = PROJECT_ROOT / args.data_dir
    vocab_path = PROJECT_ROOT / args.vocab_path
    output_dir = PROJECT_ROOT / args.output_dir

    # Handle air flag
    include_air = args.include_air and not args.no_air

    train_block2vec(
        data_dir=str(data_dir),
        vocab_path=str(vocab_path),
        output_dir=str(output_dir),
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        context_type=args.context_type,
        num_negative_samples=args.num_negatives,
        subsample_threshold=args.subsample_threshold,
        include_air=include_air,
        save_every_epochs=args.save_every,
        log_every_steps=args.log_every,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
