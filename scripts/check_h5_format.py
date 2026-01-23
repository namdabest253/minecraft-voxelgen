"""Check VQ-VAE training data coordinate ordering."""
import h5py
import numpy as np
from pathlib import Path

# Check a training file to see coordinate ordering
train_dir = Path(r"C:\Users\namda\OneDrive\Desktop\Claude_Server\minecraft_ai\data\splits\test")
h5_files = list(train_dir.glob("*.h5"))[:3]

for h5_file in h5_files:
    with h5py.File(h5_file, "r") as f:
        print(f"H5 file: {h5_file.name}")
        print(f"  Keys: {list(f.keys())}")

        # Get the structure data (try different key names)
        for key in f.keys():
            data = f[key][:]
            print(f"  {key} shape: {data.shape}, dtype: {data.dtype}")

            if len(data.shape) == 3:
                structure = data

                # Check which axis is "height" (Y) - should have most air at top
                non_air = (structure != 102) & (structure != 576) & (structure != 3352)

                # Sum along each axis
                sum_axis0 = non_air.sum(axis=(1, 2))  # Per slice along axis 0
                sum_axis1 = non_air.sum(axis=(0, 2))  # Per slice along axis 1
                sum_axis2 = non_air.sum(axis=(0, 1))  # Per slice along axis 2

                print(f"  Axis 0 pattern (first 5 vs last 5): {sum_axis0[:5].mean():.0f} vs {sum_axis0[-5:].mean():.0f}")
                print(f"  Axis 1 pattern (first 5 vs last 5): {sum_axis1[:5].mean():.0f} vs {sum_axis1[-5:].mean():.0f}")
                print(f"  Axis 2 pattern (first 5 vs last 5): {sum_axis2[:5].mean():.0f} vs {sum_axis2[-5:].mean():.0f}")
        print()
