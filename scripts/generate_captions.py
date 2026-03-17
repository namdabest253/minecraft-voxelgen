"""
Batch caption generation for Minecraft structures.

Renders each H5 file to a temporary PNG, captions it with a vision LLM, then
deletes the render immediately. No permanent render storage needed.

Three backends (auto-selected by priority):
  1. OpenAI: fast (~1-2s/caption). Set OPENAI_API_KEY env var.
             Uses gpt-4o-mini. Cost: ~$0.15/1M input tokens.
  2. Gemini REST API: fast (~2s/caption). Set GEMINI_API_KEY env var.
             Free tier: 15 req/min, 1500 req/day (gemini-2.0-flash).
  3. Gemini CLI fallback: slow (~37s/caption). Uses `gemini` CLI with OAuth.

Priority: OPENAI_API_KEY > GEMINI_API_KEY > Gemini CLI

Runs multiple workers in parallel for throughput:
  - API backends: default 10 workers → ~5-10 min for 4,462 builds
  - CLI backend:  default 3 workers  → ~7-8 hours for 4,462 builds

Usage:
    export OPENAI_API_KEY="your-key"   # preferred
    export GEMINI_API_KEY="your-key"   # alternative
    cd minecraft_ai
    python scripts/generate_captions.py                    # all splits
    python scripts/generate_captions.py --max-files 20    # test run
    python scripts/generate_captions.py --workers 5       # tune parallelism
    python scripts/generate_captions.py --resume          # skip done files
    python scripts/generate_captions.py --backend openai  # force backend
"""

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env file from project root (no external deps)
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _key, _, _val = _line.partition("=")
            _key, _val = _key.strip(), _val.strip()
            if _val and _key not in os.environ:
                os.environ[_key] = _val

from src.data.block_colors import load_block_colors
from src.data.voxel_renderer import render_structure_multiview

AIR_TOKENS = {102, 576, 3352}

TERRAIN_BLOCK_NAMES = {
    "grass_block", "dirt", "coarse_dirt", "rooted_dirt", "podzol",
    "sand", "red_sand", "gravel", "stone", "deepslate", "bedrock",
    "netherrack", "soul_sand", "soul_soil", "magma_block",
    "water", "lava",
}

PROMPT_TEMPLATE = """\
You are captioning images for a dataset that trains a CLIP-conditioned generative model. \
Captions must be natural English sentences that a person might type to request this structure — \
because at inference time users will type natural language prompts, and CLIP was pretrained on \
natural image-text pairs, not tag lists.

The image shows four views of the same voxel build: top-down (top-left), front (top-right), \
right side (bottom-left), and isometric (bottom-right). Synthesize all four views to understand \
the full 3D shape. Do NOT mention the grid, views, or camera angles in your caption.

VERIFIED MATERIALS (extracted from the build file, ordered by how many voxels use each block):
{top_blocks}

Only refer to materials using these exact names. Do not invent materials not in this list.

Scale: {scale_tag}  (small = <500 voxels, medium = 500-2000, large = >2000)

Rules:
1. Write 1-2 natural English sentences. Use "a", "an", "the" normally.
2. Lead with the structure type and scale: "a small...", "a large...", "a medium-sized..."
3. Describe the overall shape and roof/top form if visible.
4. Name the 2-4 most visually dominant materials from the list above, woven naturally into the description.
5. Mention one or two distinctive secondary features if clearly visible (archways, towers, fencing, etc.).
6. Do NOT include: "Minecraft", "voxel", "render", "isometric", "block", "build", "structure" (use the specific type instead).
7. Do NOT describe incidental ground (grass_block, dirt, coarse_dirt) unless it is clearly intentional flooring.
8. Keep it under 40 words.

Examples:
- "a small stone_bricks ruin with an uneven base and crumbling spruce_log pillars"
- "a large medieval castle with thick stone_bricks walls, a central oak_planks keep, and iron_bars portcullis"
- "a medium red barn with a steep gabled roof in red_concrete, white_concrete trim, and spruce_fence enclosure"

Output ONLY the caption sentence — no labels, no explanation, no quotes.
"""

GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta"
    "/models/gemini-2.0-flash:generateContent"
)

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"


# ---------------------------------------------------------------------------
# H5 helpers
# ---------------------------------------------------------------------------

def clean_block_name(raw_name: str) -> str:
    """Strip minecraft: prefix and block state brackets."""
    return raw_name.split("[")[0].replace("minecraft:", "").strip()


def load_structure(h5_path: Path) -> np.ndarray:
    """Load [32,32,32] int64 block_ids from H5 file."""
    with h5py.File(h5_path, "r") as f:
        key = list(f.keys())[0]
        return f[key][:].astype(np.int64)


def extract_metadata(
    block_ids: np.ndarray, tok2block: dict, top_n: int = 10
) -> tuple[list[str], str, int]:
    """Single-pass extraction of top blocks, scale tag, and non-air count."""
    flat = block_ids.flatten()
    non_air_mask = ~np.isin(flat, list(AIR_TOKENS))
    non_air_count = int(non_air_mask.sum())

    counts = Counter(int(b) for b in flat[non_air_mask])
    seen: dict[str, int] = {}
    for tok, count in counts.most_common(top_n * 2):
        clean = clean_block_name(tok2block.get(str(tok), f"unknown_{tok}"))
        if clean not in seen:
            seen[clean] = count
        if len(seen) >= top_n:
            break

    if non_air_count < 500:
        scale = "small"
    elif non_air_count < 2000:
        scale = "medium"
    else:
        scale = "large"

    return list(seen.keys()), scale, non_air_count


def is_terrain(top_blocks: list[str], non_air_count: int) -> bool:
    """True if structure is mostly terrain blocks or nearly empty."""
    if non_air_count < 100:
        return True
    return sum(1 for b in top_blocks[:5] if b in TERRAIN_BLOCK_NAMES) >= 4


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_to_temp(block_ids: np.ndarray, block_colors: dict) -> Path:
    """Render 4-view composite to a temp PNG.

    For Gemini CLI, files must be in ~/.gemini/tmp/. For OpenAI/Gemini REST,
    any path works (read via base64). We use the Gemini CLI dir for
    compatibility with all backends.
    """
    composite = render_structure_multiview(block_ids, block_colors)
    gemini_tmp = Path.home() / ".gemini" / "tmp" / "minecraft-ai"
    gemini_tmp.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", dir=gemini_tmp, delete=False)
    Image.fromarray(composite).save(tmp.name)
    tmp.close()
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------

def _call_openai_api(
    prompt: str, image_path: Path, api_key: str, timeout: int = 30
) -> str | None:
    """OpenAI GPT-4o-mini vision API with base64 image (~1-2s)."""
    b64_image = base64.b64encode(image_path.read_bytes()).decode()

    body = json.dumps({
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64_image}",
                            "detail": "low",
                        },
                    },
                ],
            }
        ],
        "max_tokens": 80,
        "temperature": 0.3,
    }).encode()

    req = urllib.request.Request(
        OPENAI_API_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read())
        text = result["choices"][0]["message"]["content"].strip()
        # Strip surrounding quotes if the model adds them
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        return text
    except urllib.error.HTTPError as e:
        tqdm.write(f"  OpenAI error {e.code}: {e.read().decode()[:120]}")
        return None
    except Exception as e:
        tqdm.write(f"  OpenAI error: {e}")
        return None


# ---------------------------------------------------------------------------
# Gemini backends
# ---------------------------------------------------------------------------

def _call_gemini_api(
    prompt: str, image_path: Path, api_key: str, timeout: int = 30
) -> str | None:
    """Fast path: Gemini REST API with base64 image (~2s)."""
    body = json.dumps({
        "contents": [{"parts": [
            {"text": prompt},
            {"inline_data": {
                "mime_type": "image/png",
                "data": base64.b64encode(image_path.read_bytes()).decode(),
            }},
        ]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 80},
    }).encode()

    req = urllib.request.Request(
        f"{GEMINI_API_URL}?key={api_key}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read())
        return result["candidates"][0]["content"]["parts"][0]["text"].strip()
    except urllib.error.HTTPError as e:
        tqdm.write(f"  API error {e.code}: {e.read().decode()[:120]}")
        return None
    except Exception as e:
        tqdm.write(f"  API error: {e}")
        return None


def _call_gemini_cli(
    prompt: str, image_path: Path, timeout: int = 90
) -> str | None:
    """Slow fallback: Gemini CLI subprocess (~37s). image_path must be in ~/.gemini/tmp/."""
    full_prompt = f"{prompt}\n\nImage to caption: {image_path}"
    try:
        result = subprocess.run(
            ["gemini", "-p", full_prompt],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0:
            return None
        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        return lines[-1] if lines else None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


# ---------------------------------------------------------------------------
# Backend dispatcher
# ---------------------------------------------------------------------------

# Module-level backend override; set by --backend CLI flag or auto-detected.
_forced_backend: str | None = None


def call_caption_api(prompt: str, image_path: Path) -> str | None:
    """Caption image via the best available backend.

    Priority: --backend flag > OPENAI_API_KEY > GEMINI_API_KEY > Gemini CLI.
    """
    backend = _forced_backend

    if backend == "openai" or (backend is None and os.environ.get("OPENAI_API_KEY")):
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            return _call_openai_api(prompt, image_path, api_key)

    if backend == "gemini" or (backend is None and os.environ.get("GEMINI_API_KEY")):
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            return _call_gemini_api(prompt, image_path, api_key)

    return _call_gemini_cli(prompt, image_path)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_caption(caption: str) -> bool:
    """Accept natural language captions 6-60 words, reject refusals."""
    if not caption:
        return False
    words = caption.split()
    if not (6 <= len(words) <= 60):
        return False
    lowered = caption.lower()
    bad = ("i cannot", "i'm unable", "i can't", "no image", "caption:", "output:")
    return not any(p in lowered for p in bad)


# ---------------------------------------------------------------------------
# Per-file worker (runs in thread)
# ---------------------------------------------------------------------------

def caption_one(
    h5_path: Path, tok2block: dict, block_colors: dict
) -> tuple[str, dict]:
    """Load, render, caption, delete temp. Fully self-contained for threading."""
    key = h5_path.name
    tmp_path = None
    try:
        block_ids = load_structure(h5_path)
        if block_ids.shape != (32, 32, 32):
            return key, {"caption": None, "filtered": "bad_shape"}

        top_blocks, scale_tag, non_air = extract_metadata(block_ids, tok2block)

        if is_terrain(top_blocks, non_air):
            return key, {
                "caption": None, "top_blocks": top_blocks,
                "scale": scale_tag, "non_air_voxels": non_air,
                "filtered": "terrain",
            }

        tmp_path = render_to_temp(block_ids, block_colors)
        prompt = PROMPT_TEMPLATE.format(
            top_blocks=", ".join(top_blocks), scale_tag=scale_tag
        )
        caption = call_caption_api(prompt, tmp_path)

        if not caption or not validate_caption(caption):
            return key, {
                "caption": None, "top_blocks": top_blocks,
                "scale": scale_tag, "non_air_voxels": non_air,
                "filtered": "invalid_caption", "raw_output": caption,
            }

        return key, {
            "caption": caption, "top_blocks": top_blocks,
            "scale": scale_tag, "non_air_voxels": non_air,
        }

    except Exception as e:
        return key, {"caption": None, "filtered": "error", "error": str(e)}
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()


# ---------------------------------------------------------------------------
# Batch orchestrator
# ---------------------------------------------------------------------------

def generate_captions(
    h5_dirs: list[Path],
    output_path: Path,
    tok2block: dict,
    block_colors: dict,
    max_files: int = 0,
    resume: bool = True,
    workers: int = 4,
) -> dict:
    """Run parallel captioning across all H5 files.

    Args:
        h5_dirs: Directories containing H5 files.
        output_path: Path to write/update captions.json.
        tok2block: Vocab dict str(token_id) -> block_name.
        block_colors: Dict token_id -> (R, G, B) for rendering.
        max_files: If >0, process only this many files.
        resume: Skip files already present in output_path.
        workers: Parallel Gemini calls. CLI: 3-5. API: up to 15.
    """
    existing = {}
    if resume and output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
        print(f"Resuming: {len(existing)} already done")

    h5_files = []
    for h5_dir in h5_dirs:
        if not h5_dir.exists():
            print(f"  WARNING: {h5_dir} not found, skipping")
            continue
        found = sorted(h5_dir.glob("*.h5"))
        print(f"  {len(found)} H5 files in {h5_dir.name}")
        h5_files.extend(found)

    if resume:
        h5_files = [f for f in h5_files if f.name not in existing]
    if max_files > 0:
        h5_files = h5_files[:max_files]

    print(f"To caption: {len(h5_files)}  |  Workers: {workers}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    captions = dict(existing)
    save_lock = threading.Lock()
    stats = {"captioned": 0, "skipped": len(existing), "terrain": 0, "errors": 0}
    done_count = 0

    def save():
        with save_lock:
            with open(output_path, "w") as f:
                json.dump(captions, f, indent=2)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(caption_one, p, tok2block, block_colors): p
            for p in h5_files
        }
        with tqdm(total=len(h5_files), desc="Captioning") as bar:
            for future in as_completed(futures):
                key, result = future.result()
                with save_lock:
                    captions[key] = result

                filtered = result.get("filtered")
                if filtered == "terrain":
                    stats["terrain"] += 1
                elif result.get("caption"):
                    stats["captioned"] += 1
                else:
                    stats["errors"] += 1
                    err = result.get("error") or result.get("raw_output") or ""
                    if stats["errors"] <= 5 and err:
                        tqdm.write(f"  Error {key}: {str(err)[:100]}")

                done_count += 1
                bar.update(1)
                if done_count % 50 == 0:
                    save()

    save()
    print(f"\nDone — captioned: {stats['captioned']}  "
          f"skipped: {stats['skipped']}  "
          f"terrain: {stats['terrain']}  "
          f"errors: {stats['errors']}")
    print(f"Output: {output_path}")
    return stats


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _detect_backend() -> tuple[str, int]:
    """Detect best available backend and return (label, default_workers)."""
    if os.environ.get("OPENAI_API_KEY"):
        return "OpenAI gpt-4o-mini (~1-2s/caption)", 10
    if os.environ.get("GEMINI_API_KEY"):
        return "Gemini REST API (~2s/caption)", 10
    return "Gemini CLI fallback (~37s/caption)", 3


def main():
    global _forced_backend

    parser = argparse.ArgumentParser(
        description="Batch caption Minecraft structures (parallel, render-on-the-fly)"
    )
    parser.add_argument(
        "--input-dir", type=str, nargs="+", default=None,
        help="H5 input directories (default: splits/train + val + test)",
    )
    parser.add_argument("--max-files", type=int, default=0,
                        help="Limit files processed (0 = all)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-captioned files")
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel workers (default: 10 for API, 3 for CLI)")
    parser.add_argument(
        "--backend", type=str, choices=["openai", "gemini", "gemini-cli"],
        default=None,
        help="Force a specific backend (default: auto-detect from env vars)",
    )
    args = parser.parse_args()

    if args.backend:
        _forced_backend = args.backend

    tok2block_path = PROJECT_ROOT / "data" / "vocabulary" / "tok2block.json"
    output_path = PROJECT_ROOT / "data" / "output" / "captions" / "captions.json"

    print("Loading vocabulary and block colors...")
    with open(tok2block_path) as f:
        tok2block = json.load(f)
    block_colors = load_block_colors(str(tok2block_path))

    backend_label, default_workers = _detect_backend()
    if _forced_backend:
        backend_label = f"{_forced_backend} (forced)"
    workers = args.workers if args.workers > 0 else default_workers

    print(f"Backend: {backend_label}  |  Workers: {workers}")
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        print("  Tip: set OPENAI_API_KEY or GEMINI_API_KEY for fast captioning")

    h5_dirs = (
        [Path(d) for d in args.input_dir]
        if args.input_dir
        else [
            PROJECT_ROOT / "data" / "splits" / "train",
            PROJECT_ROOT / "data" / "splits" / "val",
            PROJECT_ROOT / "data" / "splits" / "test",
            PROJECT_ROOT / "data" / "processed_new",
            PROJECT_ROOT / "data" / "procedural",
        ]
    )

    generate_captions(
        h5_dirs=h5_dirs,
        output_path=output_path,
        tok2block=tok2block,
        block_colors=block_colors,
        max_files=args.max_files,
        resume=args.resume,
        workers=workers,
    )


if __name__ == "__main__":
    main()
