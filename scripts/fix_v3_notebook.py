"""Fix the V3 validation notebook - rewrite cells 9-12 to work correctly."""
import json

with open(r'C:\Users\namda\OneDrive\Desktop\Claude_Server\minecraft_ai\data\kaggle\notebooks\vqvae_v3_validation.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 9: Load previous results and merge with V3
cell9_code = '''# ============================================================
# CELL 9: Load Previous Results and Merge
# ============================================================

# Load previous results (V1, V2, Random from previous validation run)
print("Loading previous validation results...")
with open(PREVIOUS_RESULTS_PATH, 'r') as f:
    previous_results = json.load(f)

print(f"Loaded: {list(previous_results.keys())}")

# The V3 results are in all_results from cell 8
# We need to convert to same format as previous results
v3_data = all_results["V3"]

# Create V3 entry in same format as previous results
v3_formatted = {
    "history": v3_data["history"],
    "training_time": v3_data["training_time"]
}

# Merge all results
combined_results = {**previous_results}
combined_results["V3"] = v3_formatted

print("\\nAll results combined:")
for name in ["V1", "V2", "V3", "Random"]:
    final_struct = combined_results[name]["history"]["val_struct_acc"][-1]
    print(f"  {name}: {final_struct:.1%} structure accuracy")

# Store for use in later cells
all_results_combined = combined_results
'''

# Cell 10: Plot all results
cell10_code = '''# ============================================================
# CELL 10: Plot All Results (V1, V2, V3, Random)
# ============================================================

# Plot comparison
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

colors = {"V1": "blue", "V2": "green", "V3": "purple", "Random": "red"}
names = ["V1", "V2", "V3", "Random"]

# Validation Loss
ax = axes[0, 0]
for name in names:
    ax.plot(all_results_combined[name]["history"]["val_loss"], label=name, color=colors[name], linewidth=2)
ax.set_title("Validation Loss by Embedding Type", fontsize=12)
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation Loss")
ax.legend()
ax.grid(True, alpha=0.3)

# Overall Accuracy
ax = axes[0, 1]
for name in names:
    ax.plot(all_results_combined[name]["history"]["val_acc"], label=name, color=colors[name], linewidth=2)
ax.set_title("Overall Accuracy (includes ~80% air)", fontsize=12)
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation Accuracy")
ax.legend()
ax.grid(True, alpha=0.3)

# Structure Accuracy (KEY METRIC)
ax = axes[0, 2]
for name in names:
    ax.plot(all_results_combined[name]["history"]["val_struct_acc"], label=name, color=colors[name], linewidth=2)
ax.set_title("STRUCTURE Accuracy (KEY METRIC)", fontsize=12, fontweight="bold")
ax.set_xlabel("Epoch")
ax.set_ylabel("Structure Accuracy (non-air)")
ax.legend()
ax.grid(True, alpha=0.3)

# Air Accuracy
ax = axes[1, 0]
for name in names:
    ax.plot(all_results_combined[name]["history"]["val_air_acc"], label=name, color=colors[name], linewidth=2)
ax.set_title("Air Block Accuracy", fontsize=12)
ax.set_xlabel("Epoch")
ax.set_ylabel("Air Accuracy")
ax.legend()
ax.grid(True, alpha=0.3)

# Air Percentage (sanity check)
ax = axes[1, 1]
for name in names:
    ax.plot(all_results_combined[name]["history"]["val_air_pct"], label=name, color=colors[name], linewidth=2)
ax.set_title("Air Block % in Data (~should be constant)", fontsize=12)
ax.set_xlabel("Epoch")
ax.set_ylabel("Air Percentage")
ax.set_ylim(0.78, 0.82)
ax.legend()
ax.grid(True, alpha=0.3)

# Bar chart comparison
ax = axes[1, 2]
struct_accs = [all_results_combined[name]["history"]["val_struct_acc"][-1] for name in names]
bar_colors = [colors[n] for n in names]
bars = ax.bar(names, struct_accs, color=bar_colors, edgecolor="black", linewidth=1.5)
ax.set_title("Best Structure Accuracy Comparison", fontsize=12, fontweight="bold")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 0.55)

# Add value labels on bars
for bar, acc in zip(bars, struct_accs):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
            f"{acc:.1%}", ha="center", fontsize=14, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/embedding_comparison_all.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"Saved plot to {OUTPUT_DIR}/embedding_comparison_all.png")
'''

# Cell 11: Save results
cell11_code = '''# ============================================================
# CELL 11: Save Final Results
# ============================================================

# Get V3 final results
v3_hist = all_results_combined["V3"]["history"]

v3_summary = {
    "V3": {
        "final_val_loss": v3_hist["val_loss"][-1],
        "final_val_acc": v3_hist["val_acc"][-1],
        "final_val_struct_acc": v3_hist["val_struct_acc"][-1],
        "final_val_air_acc": v3_hist["val_air_acc"][-1],
        "training_time_minutes": all_results_combined["V3"]["training_time"] / 60,
    }
}

with open(f"{OUTPUT_DIR}/v3_validation_results.json", "w") as f:
    json.dump(v3_summary, f, indent=2)

# Save full V3 history
with open(f"{OUTPUT_DIR}/v3_validation_full.json", "w") as f:
    json.dump({"V3": all_results_combined["V3"]}, f, indent=2)

# Print comparison
print("="*70)
print("FINAL COMPARISON - Structure Accuracy (non-air blocks)")
print("="*70)

v3_struct = v3_hist["val_struct_acc"][-1]
v1_struct = all_results_combined["V1"]["history"]["val_struct_acc"][-1]
v2_struct = all_results_combined["V2"]["history"]["val_struct_acc"][-1]
random_struct = all_results_combined["Random"]["history"]["val_struct_acc"][-1]

print(f"  V1:     {v1_struct:.1%}  (previous)")
print(f"  V3:     {v3_struct:.1%}  <- NEW")
print(f"  V2:     {v2_struct:.1%}  (previous)")
print(f"  Random: {random_struct:.1%}  (previous)")
print()

if v3_struct > v1_struct:
    print("V3 BEATS V1! Use V3 embeddings for VQ-VAE.")
elif v3_struct > v2_struct:
    print("V3 is between V1 and V2. V1 is still best.")
else:
    print("V3 is worse than V1 and V2. Stick with V1 embeddings.")
'''

# Cell 12: Final summary
cell12_code = '''# ============================================================
# CELL 12: Final Summary
# ============================================================

print()
print("="*70)
print("PHASE 0 COMPLETE: VQ-VAE EMBEDDING VALIDATION (V3)")
print("="*70)

# Get accuracies
v1_struct = all_results_combined["V1"]["history"]["val_struct_acc"][-1]
v2_struct = all_results_combined["V2"]["history"]["val_struct_acc"][-1]
v3_struct = all_results_combined["V3"]["history"]["val_struct_acc"][-1]
random_struct = all_results_combined["Random"]["history"]["val_struct_acc"][-1]

# Calculate improvements
v1_improvement = (v1_struct - random_struct) / random_struct * 100
v2_improvement = (v2_struct - random_struct) / random_struct * 100
v3_improvement = (v3_struct - random_struct) / random_struct * 100

print()
print("Structure accuracy improvement over random baseline:")
print(f"  V1: {v1_improvement:+.1f}% ({v1_struct:.1%})")
print(f"  V3: {v3_improvement:+.1f}% ({v3_struct:.1%}) <- NEW")
print(f"  V2: {v2_improvement:+.1f}% ({v2_struct:.1%})")

print()
print("="*70)
print("CONCLUSION")
print("="*70)

# Rank them
rankings = sorted([
    ("V1", v1_struct),
    ("V2", v2_struct),
    ("V3", v3_struct),
    ("Random", random_struct)
], key=lambda x: -x[1])

print()
print("Ranking by structure accuracy:")
for i, (name, acc) in enumerate(rankings, 1):
    marker = "<- BEST" if i == 1 else ""
    print(f"  {i}. {name}: {acc:.1%} {marker}")

print()
best_name = rankings[0][0]
print(f"RECOMMENDATION: Use {best_name} embeddings for full VQ-VAE training (Phase 3)")

print()
print("="*70)
print("Files saved:")
print(f"  - {OUTPUT_DIR}/embedding_comparison_all.png")
print(f"  - {OUTPUT_DIR}/v3_validation_results.json")
print(f"  - {OUTPUT_DIR}/v3_validation_full.json")
print(f"  - {OUTPUT_DIR}/air_tokens_used.json")
print("="*70)
'''

# Update the cells
nb['cells'][9]['source'] = cell9_code
nb['cells'][10]['source'] = cell10_code
nb['cells'][11]['source'] = cell11_code
nb['cells'][12]['source'] = cell12_code

# Save
with open(r'C:\Users\namda\OneDrive\Desktop\Claude_Server\minecraft_ai\data\kaggle\notebooks\vqvae_v3_validation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Fixed cells 9, 10, 11, 12")
print("Changes:")
print("  - Cell 9: Loads previous results and merges with V3")
print("  - Cell 10: Plots all 4 embeddings (V1, V2, V3, Random)")
print("  - Cell 11: Saves V3 results and prints comparison")
print("  - Cell 12: Final summary with rankings")
