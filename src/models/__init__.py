"""Neural network model architectures."""

from src.models.block2vec import Block2Vec
from src.models.vqvae import (
    VQVAE,
    VQVAEv4,
    VectorQuantizer,
    VectorQuantizerEMA,
    Encoder,
    Decoder,
    EncoderV4,
    DecoderV4,
    EmbeddingAwareLoss,
    ShapePreservationLoss,
    AsymmetricStructureLoss,
    compute_similarity_matrix,
    similarity_weighted_accuracy,
)
from src.models.sparse_transformer import (
    SparseStructureTransformer,
    SparseStructureTransformerLite,
    FourierPositionalEncoding,
    SetPooling,
)

__all__ = [
    # Block2Vec
    "Block2Vec",
    # VQ-VAE v3 (original)
    "VQVAE",
    "VectorQuantizer",
    "VectorQuantizerEMA",
    "Encoder",
    "Decoder",
    # VQ-VAE v4 (improved)
    "VQVAEv4",
    "EncoderV4",
    "DecoderV4",
    "EmbeddingAwareLoss",
    "ShapePreservationLoss",
    "AsymmetricStructureLoss",
    "compute_similarity_matrix",
    "similarity_weighted_accuracy",
    # Sparse Transformer
    "SparseStructureTransformer",
    "SparseStructureTransformerLite",
    "FourierPositionalEncoding",
    "SetPooling",
]
