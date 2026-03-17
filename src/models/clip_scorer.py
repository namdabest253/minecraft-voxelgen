"""
CLIP Scorer — OpenCLIP wrapper for zero-shot structure classification.

Uses OpenCLIP ViT-B/32 (LAION-2B pretrained) to score rendered structure
images against text prompts for filtering and labeling.

Requires: pip install open_clip_torch
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Text prompts for zero-shot classification
STRUCTURE_PROMPTS: Dict[str, List[str]] = {
    "house": [
        "a Minecraft house made of blocks",
        "a blocky house with walls and a roof",
    ],
    "castle": [
        "a Minecraft castle with towers",
        "a blocky medieval castle",
    ],
    "tower": [
        "a tall Minecraft tower",
        "a tall blocky tower structure",
    ],
    "ship": [
        "a Minecraft ship or boat",
        "a blocky ship on water",
    ],
    "bridge": [
        "a Minecraft bridge",
        "a blocky bridge structure",
    ],
    "temple": [
        "a Minecraft temple or church",
        "a blocky religious building",
    ],
    "generic_structure": [
        "a Minecraft building",
        "a constructed structure made of blocks",
        "a blocky building or construction",
    ],
}

NEGATIVE_PROMPTS: Dict[str, List[str]] = {
    "terrain": [
        "natural terrain with dirt and stone",
        "raw Minecraft terrain",
        "natural landscape with grass and dirt",
    ],
    "random": [
        "random scattered blocks",
        "noise pattern of colored blocks",
    ],
    "empty": [
        "mostly empty space with a few blocks",
        "sparse scattered blocks in empty space",
    ],
}


class CLIPScorer:
    """OpenCLIP wrapper for scoring Minecraft structure renders.

    Args:
        model_name: OpenCLIP model name (default ViT-B-32).
        pretrained: Pretrained weights name.
        device: Torch device (auto-detects if None).
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: Optional[str] = None,
    ):
        import open_clip

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # Cache text embeddings for all prompts
        self._text_cache: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text prompts to CLIP embeddings.

        Args:
            texts: List of text strings.

        Returns:
            [n_texts, embed_dim] normalized embeddings.
        """
        tokens = self.tokenizer(texts).to(self.device)
        features = self.model.encode_text(tokens)
        features = F.normalize(features, dim=-1)
        return features

    @torch.no_grad()
    def encode_images(self, images: List[np.ndarray]) -> torch.Tensor:
        """Encode images to CLIP embeddings.

        Args:
            images: List of [H, W, 3] uint8 numpy arrays.

        Returns:
            [n_images, embed_dim] normalized embeddings.
        """
        from PIL import Image

        processed = []
        for img in images:
            pil_img = Image.fromarray(img)
            processed.append(self.preprocess(pil_img))

        batch = torch.stack(processed).to(self.device)
        features = self.model.encode_image(batch)
        features = F.normalize(features, dim=-1)
        return features

    @torch.no_grad()
    def score_images(
        self,
        images: List[np.ndarray],
        text_prompts: List[str],
    ) -> np.ndarray:
        """Compute cosine similarity between images and text prompts.

        Args:
            images: List of [H, W, 3] uint8 numpy arrays.
            text_prompts: List of text strings.

        Returns:
            [n_images, n_prompts] cosine similarity matrix.
        """
        image_features = self.encode_images(images)
        text_features = self.encode_text(text_prompts)
        similarity = (image_features @ text_features.T).cpu().numpy()
        return similarity

    def _get_category_embeddings(self) -> Tuple[List[str], torch.Tensor]:
        """Get averaged text embeddings per category (cached).

        Returns:
            (category_names, [n_categories, embed_dim] embeddings)
        """
        all_prompts = {**STRUCTURE_PROMPTS, **NEGATIVE_PROMPTS}
        categories = list(all_prompts.keys())

        embeddings = []
        for cat in categories:
            cache_key = f"cat_{cat}"
            if cache_key not in self._text_cache:
                prompts = all_prompts[cat]
                feats = self.encode_text(prompts)
                avg_feat = F.normalize(feats.mean(dim=0, keepdim=True), dim=-1)
                self._text_cache[cache_key] = avg_feat
            embeddings.append(self._text_cache[cache_key])

        return categories, torch.cat(embeddings, dim=0)

    @torch.no_grad()
    def classify_structure(
        self,
        composite_image: np.ndarray,
    ) -> Dict[str, float]:
        """Classify a structure using its multi-view composite.

        Args:
            composite_image: [224, 224, 3] uint8 composite of 4 views.

        Returns:
            {category: score} dict with cosine similarities.
        """
        categories, text_features = self._get_category_embeddings()
        image_features = self.encode_images([composite_image])
        scores = (image_features @ text_features.T).cpu().numpy()[0]
        return {cat: float(score) for cat, score in zip(categories, scores)}

    @torch.no_grad()
    def get_text_embedding(self, category: str) -> np.ndarray:
        """Get the averaged CLIP text embedding for a category.

        Args:
            category: Category name from STRUCTURE_PROMPTS.

        Returns:
            [embed_dim] numpy array (normalized).
        """
        all_prompts = {**STRUCTURE_PROMPTS, **NEGATIVE_PROMPTS}
        if category not in all_prompts:
            # Single prompt
            feats = self.encode_text([category])
            return feats.cpu().numpy()[0]

        prompts = all_prompts[category]
        feats = self.encode_text(prompts)
        avg_feat = F.normalize(feats.mean(dim=0, keepdim=True), dim=-1)
        return avg_feat.cpu().numpy()[0]
