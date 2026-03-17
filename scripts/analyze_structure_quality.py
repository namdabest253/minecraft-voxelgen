"""Analyze structure quality metrics for real vs generated builds."""

import json
from pathlib import Path
from collections import defaultdict
import h5py
import numpy as np
from scipy import ndimage


def load_structure(path: Path) -> np.ndarray:
    """Load structure from H5 file."""
    with h5py.File(path, 'r') as f:
        key = list(f.keys())[0]
        return f[key][:].astype(np.int64)


def get_binary_mask(structure: np.ndarray, air_tokens: set) -> np.ndarray:
    """Convert structure to binary (0=air, 1=structure)."""
    mask = np.ones_like(structure, dtype=np.uint8)
    for air_token in air_tokens:
        mask[structure == air_token] = 0
    return mask


def count_connected_components(binary_mask: np.ndarray) -> tuple:
    """Count connected components and return (count, largest_component_ratio)."""
    labeled, num_features = ndimage.label(binary_mask)
    if num_features == 0:
        return 0, 0.0

    # Find largest component
    component_sizes = ndimage.sum(binary_mask, labeled, range(1, num_features + 1))
    largest_size = max(component_sizes) if len(component_sizes) > 0 else 0
    total_blocks = binary_mask.sum()

    largest_ratio = largest_size / total_blocks if total_blocks > 0 else 0
    return num_features, largest_ratio


def compute_support_ratio(binary_mask: np.ndarray) -> float:
    """Compute ratio of blocks that have support below them."""
    if binary_mask.sum() == 0:
        return 0.0

    # A block has support if the block below it (Y-1) is also solid, or it's at Y=0
    supported = np.zeros_like(binary_mask)

    # Bottom layer is always supported
    supported[:, 0, :] = binary_mask[:, 0, :]

    # Other layers: supported if block below
    for y in range(1, binary_mask.shape[1]):
        supported[:, y, :] = binary_mask[:, y, :] & binary_mask[:, y-1, :]

    total_blocks = binary_mask.sum()
    supported_blocks = (binary_mask & supported).sum()

    return supported_blocks / total_blocks if total_blocks > 0 else 0.0


def compute_ground_contact(binary_mask: np.ndarray) -> float:
    """Compute ratio of bottom layer that has structure."""
    bottom_layer = binary_mask[:, 0, :]
    return bottom_layer.sum() / bottom_layer.size


def compute_density(binary_mask: np.ndarray) -> float:
    """Compute overall density (non-air ratio)."""
    return binary_mask.sum() / binary_mask.size


def compute_surface_to_volume(binary_mask: np.ndarray) -> float:
    """Compute surface area to volume ratio."""
    if binary_mask.sum() == 0:
        return 0.0

    volume = binary_mask.sum()

    # Count exposed faces (faces adjacent to air)
    surface = 0
    padded = np.pad(binary_mask, 1, mode='constant', constant_values=0)

    # Check all 6 directions
    for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
        shifted = np.roll(np.roll(np.roll(padded, dx, axis=0), dy, axis=1), dz, axis=2)
        # Exposed face: block is solid, neighbor is air
        exposed = binary_mask & (shifted[1:-1, 1:-1, 1:-1] == 0)
        surface += exposed.sum()

    return surface / volume if volume > 0 else 0.0


def compute_layer_density_variance(binary_mask: np.ndarray) -> float:
    """Compute variance of per-layer density (low = consistent floors)."""
    layer_densities = []
    for y in range(binary_mask.shape[1]):
        layer = binary_mask[:, y, :]
        layer_densities.append(layer.sum() / layer.size)

    return np.std(layer_densities)


def compute_block_type_entropy(structure: np.ndarray, binary_mask: np.ndarray) -> float:
    """Compute entropy of block type distribution."""
    if binary_mask.sum() == 0:
        return 0.0

    # Get block types in structure regions
    block_types = structure[binary_mask == 1]
    unique, counts = np.unique(block_types, return_counts=True)

    # Compute entropy
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs + 1e-10))

    return entropy


def compute_local_coherence(structure: np.ndarray, binary_mask: np.ndarray) -> float:
    """Compute local coherence (how often adjacent blocks are same type)."""
    if binary_mask.sum() < 2:
        return 0.0

    same_neighbor_count = 0
    total_neighbor_count = 0

    # Check horizontal neighbors (X and Z directions)
    for dx, dz in [(1, 0), (0, 1)]:
        for x in range(structure.shape[0] - dx):
            for y in range(structure.shape[1]):
                for z in range(structure.shape[2] - dz):
                    if binary_mask[x, y, z] and binary_mask[x+dx, y, z+dz]:
                        total_neighbor_count += 1
                        if structure[x, y, z] == structure[x+dx, y, z+dz]:
                            same_neighbor_count += 1

    return same_neighbor_count / total_neighbor_count if total_neighbor_count > 0 else 0.0


def compute_unique_block_types(structure: np.ndarray, binary_mask: np.ndarray) -> int:
    """Count unique block types in structure."""
    if binary_mask.sum() == 0:
        return 0
    block_types = structure[binary_mask == 1]
    return len(np.unique(block_types))


def analyze_structure(structure: np.ndarray, air_tokens: set) -> dict:
    """Compute all metrics for a structure."""
    binary_mask = get_binary_mask(structure, air_tokens)

    num_components, largest_ratio = count_connected_components(binary_mask)

    return {
        'density': compute_density(binary_mask),
        'num_components': num_components,
        'largest_component_ratio': largest_ratio,
        'support_ratio': compute_support_ratio(binary_mask),
        'ground_contact': compute_ground_contact(binary_mask),
        'surface_to_volume': compute_surface_to_volume(binary_mask),
        'layer_density_std': compute_layer_density_variance(binary_mask),
        'block_entropy': compute_block_type_entropy(structure, binary_mask),
        'local_coherence': compute_local_coherence(structure, binary_mask),
        'unique_blocks': compute_unique_block_types(structure, binary_mask),
        'total_blocks': int(binary_mask.sum()),
    }


def main():
    # Paths
    BASE_PATH = Path(__file__).parent.parent
    TRAIN_DIR = BASE_PATH / "data" / "splits" / "train"
    GEN_SAMPLES_PATH = BASE_PATH / "data" / "output" / "prior" / "diffusion_v4" / "generated_samples.pt"

    air_tokens = {102, 576, 3352}

    # Analyze real structures (sample 50)
    print("="*70)
    print("ANALYZING REAL TRAINING STRUCTURES")
    print("="*70)

    h5_files = sorted(TRAIN_DIR.glob("*.h5"))[:50]
    real_metrics = []

    for h5_file in h5_files:
        try:
            structure = load_structure(h5_file)
            metrics = analyze_structure(structure, air_tokens)
            real_metrics.append(metrics)
        except Exception as e:
            print(f"Error loading {h5_file}: {e}")

    print(f"Analyzed {len(real_metrics)} real structures")

    # Analyze generated structures
    print("\n" + "="*70)
    print("ANALYZING GENERATED STRUCTURES (v4)")
    print("="*70)

    import torch
    generated = torch.load(GEN_SAMPLES_PATH).numpy()
    gen_metrics = []

    for i in range(generated.shape[0]):
        structure = generated[i]
        metrics = analyze_structure(structure, air_tokens)
        gen_metrics.append(metrics)

    print(f"Analyzed {len(gen_metrics)} generated structures")

    # Compute statistics
    def compute_stats(metrics_list, key):
        values = [m[key] for m in metrics_list]
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
        }

    print("\n" + "="*70)
    print("COMPARISON: REAL vs GENERATED")
    print("="*70)

    metric_keys = [
        'density', 'num_components', 'largest_component_ratio',
        'support_ratio', 'ground_contact', 'surface_to_volume',
        'layer_density_std', 'block_entropy', 'local_coherence',
        'unique_blocks', 'total_blocks'
    ]

    print(f"\n{'Metric':<25} {'Real (mean±std)':<25} {'Generated (mean±std)':<25} {'Gap':<15}")
    print("-"*90)

    comparison_data = {}

    for key in metric_keys:
        real_stats = compute_stats(real_metrics, key)
        gen_stats = compute_stats(gen_metrics, key)

        gap = gen_stats['mean'] - real_stats['mean']
        gap_pct = (gap / real_stats['mean'] * 100) if real_stats['mean'] != 0 else 0

        print(f"{key:<25} {real_stats['mean']:.3f}±{real_stats['std']:.3f}{'':>10} "
              f"{gen_stats['mean']:.3f}±{gen_stats['std']:.3f}{'':>10} "
              f"{gap_pct:+.1f}%")

        comparison_data[key] = {
            'real': real_stats,
            'generated': gen_stats,
            'gap_pct': gap_pct
        }

    # Save results
    output_path = BASE_PATH / "data" / "output" / "prior" / "diffusion_v4" / "quality_metrics.json"
    with open(output_path, 'w') as f:
        json.dump(comparison_data, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Key findings
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)

    print(f"\n1. CONNECTIVITY:")
    print(f"   Real: {comparison_data['num_components']['real']['mean']:.1f} components")
    print(f"   Generated: {comparison_data['num_components']['generated']['mean']:.1f} components")
    print(f"   -> Generated structures are {'more' if comparison_data['num_components']['gap_pct'] > 0 else 'less'} fragmented")

    print(f"\n2. SUPPORT/GROUNDING:")
    print(f"   Real support ratio: {comparison_data['support_ratio']['real']['mean']*100:.1f}%")
    print(f"   Generated support ratio: {comparison_data['support_ratio']['generated']['mean']*100:.1f}%")
    print(f"   Real ground contact: {comparison_data['ground_contact']['real']['mean']*100:.1f}%")
    print(f"   Generated ground contact: {comparison_data['ground_contact']['generated']['mean']*100:.1f}%")

    print(f"\n3. LOCAL COHERENCE:")
    print(f"   Real: {comparison_data['local_coherence']['real']['mean']*100:.1f}% same-neighbor")
    print(f"   Generated: {comparison_data['local_coherence']['generated']['mean']*100:.1f}% same-neighbor")
    print(f"   -> Generated blocks are {'more' if comparison_data['local_coherence']['gap_pct'] > 0 else 'less'} coherent")

    print(f"\n4. COMPLEXITY:")
    print(f"   Real surface/volume: {comparison_data['surface_to_volume']['real']['mean']:.2f}")
    print(f"   Generated surface/volume: {comparison_data['surface_to_volume']['generated']['mean']:.2f}")
    print(f"   -> Generated structures are {'more' if comparison_data['surface_to_volume']['gap_pct'] > 0 else 'less'} complex/sparse")


if __name__ == "__main__":
    main()
