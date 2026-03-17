"""
Precompute per-caption CLIP text embeddings for all captioned builds.

Reads captions.json (filename -> {caption, ...}), encodes each caption
through CLIP ViT-B/32 text encoder, and saves a flat dict of
filename -> [512] normalized embedding tensors.

Also computes a null embedding (empty string) for CFG unconditional training.

Output: minecraft_ai/data/output/clip_embeddings/caption_embeddings.pt

Usage:
    cd minecraft_ai
    python scripts/precompute_caption_embeddings.py
    python scripts/precompute_caption_embeddings.py --batch-size 64
    python scripts/precompute_caption_embeddings.py --captions path/to/captions.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def precompute_caption_embeddings(
    captions_path: Path,
    output_path: Path,
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    batch_size: int = 128,
    device: str | None = None,
) -> dict:
    """Encode each caption with CLIP and save per-file embeddings.

    Args:
        captions_path: Path to captions.json.
        output_path: Where to save caption_embeddings.pt.
        model_name: OpenCLIP model name.
        pretrained: Pretrained weights.
        batch_size: Captions per batch for encoding.
        device: Torch device (auto-detects if None).

    Returns:
        Stats dict.
    """
    import open_clip

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load captions
    with open(captions_path) as f:
        all_captions = json.load(f)

    # Filter to entries that have a caption
    captioned = {
        fn: entry["caption"]
        for fn, entry in all_captions.items()
        if entry.get("caption")
    }
    print(f"Captioned builds: {len(captioned)} / {len(all_captions)} total")

    if not captioned:
        print("ERROR: No captions found. Run generate_captions.py first.")
        sys.exit(1)

    # Load CLIP model
    print(f"Loading CLIP {model_name} ({pretrained})...")
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    # Encode all captions in batches
    filenames = list(captioned.keys())
    texts = [captioned[fn] for fn in filenames]
    all_embeddings = []

    print(f"Encoding {len(texts)} captions (batch_size={batch_size})...")
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        tokens = tokenizer(batch_texts).to(device)
        with torch.no_grad():
            features = model.encode_text(tokens)
            features = F.normalize(features, dim=-1)
        all_embeddings.append(features.cpu())

        done = min(i + batch_size, len(texts))
        if done % 1000 == 0 or done == len(texts):
            print(f"  {done}/{len(texts)}")

    all_embeddings = torch.cat(all_embeddings, dim=0)  # [N, 512]

    # Build filename -> embedding dict
    embeddings = {}
    for idx, fn in enumerate(filenames):
        embeddings[fn] = all_embeddings[idx]

    # Null embedding for CFG unconditional
    with torch.no_grad():
        null_tokens = tokenizer([""]).to(device)
        null_emb = model.encode_text(null_tokens)
        null_emb = F.normalize(null_emb, dim=-1)
    embeddings["_null"] = null_emb.cpu().squeeze(0)

    embed_dim = all_embeddings.shape[1]
    print(f"\nEmbedding dim: {embed_dim}")
    print(f"Total entries: {len(embeddings)} ({len(filenames)} captions + _null)")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, output_path)
    print(f"Saved to {output_path}")

    return {
        "captioned": len(filenames),
        "skipped": len(all_captions) - len(captioned),
        "embed_dim": embed_dim,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Precompute per-caption CLIP text embeddings"
    )
    parser.add_argument(
        "--captions", type=str, default=None,
        help="Path to captions.json (default: data/output/captions/captions.json)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output .pt path (default: data/output/clip_embeddings/caption_embeddings.pt)",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    captions_path = (
        Path(args.captions) if args.captions
        else PROJECT_ROOT / "data" / "output" / "captions" / "captions.json"
    )
    output_path = (
        Path(args.output) if args.output
        else PROJECT_ROOT / "data" / "output" / "clip_embeddings" / "caption_embeddings.pt"
    )

    precompute_caption_embeddings(
        captions_path=captions_path,
        output_path=output_path,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
