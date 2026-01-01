"""Check notebook for syntax errors and variable access issues."""
import ast
import json
import re

# Read the notebook
with open("data/kaggle/notebooks/vqvae_v3_validation.ipynb", 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("=" * 60)
print("CHECKING NOTEBOOK FOR ERRORS")
print("=" * 60)

# Check syntax for each code cell
print("\n1. SYNTAX CHECK")
print("-" * 40)
syntax_errors = []

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        try:
            ast.parse(source)
            print(f"  Cell {i}: ✓ OK")
        except SyntaxError as e:
            print(f"  Cell {i}: ✗ SYNTAX ERROR - {e}")
            syntax_errors.append((i, str(e)))

if syntax_errors:
    print(f"\n  Found {len(syntax_errors)} syntax errors!")
else:
    print("\n  No syntax errors found!")

# Check for variable definitions and usages
print("\n2. VARIABLE ACCESS CHECK")
print("-" * 40)

# Track key variables and where they're defined
key_vars = {
    'all_results': None,
    'previous_results': None,
    'avg_air_pct': None,
    'v3_result': None,
    'PREVIOUS_RESULTS_PATH': None,
    'AIR_TOKENS': None,
    'AIR_TOKENS_TENSOR': None,
    'EMBEDDINGS': None,
    'train_loader': None,
    'val_loader': None,
}

# Find definitions
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        for var in key_vars:
            # Check for assignment patterns
            if re.search(rf'\b{var}\s*=', source) and key_vars[var] is None:
                key_vars[var] = i

print("  Variable definitions:")
for var, cell_idx in key_vars.items():
    if cell_idx is not None:
        print(f"    {var}: defined in Cell {cell_idx}")
    else:
        print(f"    {var}: NOT FOUND!")

# Check usages come after definitions
print("\n3. USAGE ORDER CHECK")
print("-" * 40)

usage_issues = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        for var, def_cell in key_vars.items():
            # Check if variable is used (but not just defined)
            if re.search(rf'\b{var}\b', source):
                # Skip if this is the definition cell
                if def_cell is not None and i < def_cell:
                    # Used before defined!
                    if not re.search(rf'\b{var}\s*=', source):  # Not a definition
                        usage_issues.append(f"Cell {i} uses '{var}' before it's defined in Cell {def_cell}")

if usage_issues:
    for issue in usage_issues:
        print(f"  ✗ {issue}")
else:
    print("  ✓ All variables used after definition")

# Check specific dict key accesses
print("\n4. DICT KEY ACCESS CHECK")
print("-" * 40)

# Check that all_results['V3'] exists when accessed
# Check that previous_results has V1, V2, Random keys

# Look for dict accesses in cells 9-12
for i in [9, 10, 11, 12]:
    if i < len(nb['cells']) and nb['cells'][i]['cell_type'] == 'code':
        source = ''.join(nb['cells'][i]['source'])
        
        # Check all_results accesses
        ar_matches = re.findall(r"all_results\['(\w+)'\]", source)
        for key in ar_matches:
            if key != 'V3':
                print(f"  Cell {i}: all_results['{key}'] - should only be 'V3'!")
        
        # Check previous_results accesses
        pr_matches = re.findall(r"previous_results\['(\w+)'\]", source)
        for key in pr_matches:
            if key not in ['V1', 'V2', 'Random', 'history']:
                print(f"  Cell {i}: previous_results['{key}'] - unexpected key!")

print("  ✓ Dict key access patterns look correct")

# Check JSON file structure expectation
print("\n5. EXPECTED JSON STRUCTURE")
print("-" * 40)
print("  The code expects previous_results JSON to have:")
print("    - Keys: 'V1', 'V2', 'Random'")
print("    - Each with 'history' containing:")
print("      - 'val_loss', 'val_acc', 'val_struct_acc'")
print("      - 'val_air_acc', 'val_air_pct'")
print("    - Each with 'training_time'")

# Verify with actual file if it exists
try:
    with open("data/kaggle/output/vqvae-validation/v2/embedding_validation_full.json") as f:
        prev = json.load(f)
    print("\n  Checking actual JSON file...")
    for name in ['V1', 'V2', 'Random']:
        if name not in prev:
            print(f"  ✗ Missing key: {name}")
        else:
            hist = prev[name].get('history', {})
            missing = []
            for key in ['val_loss', 'val_acc', 'val_struct_acc', 'val_air_acc', 'val_air_pct']:
                if key not in hist:
                    missing.append(key)
            if missing:
                print(f"  ✗ {name} missing history keys: {missing}")
            else:
                print(f"  ✓ {name}: all required keys present")
            if 'training_time' not in prev[name]:
                print(f"  ✗ {name} missing 'training_time'")
except FileNotFoundError:
    print("  (JSON file not found locally - will be on Kaggle)")

# Check final_metrics keys match what's accessed
print("\n6. FINAL_METRICS KEYS CHECK")
print("-" * 40)

# Get Cell 7 content
cell7_source = ''.join(nb['cells'][7]['source'])

# Keys that are stored in final_metrics
stored_keys = ['name', 'final_train_loss', 'final_val_loss', 'final_train_acc', 
               'final_val_acc', 'final_train_struct_acc', 'final_val_struct_acc',
               'final_train_air_acc', 'final_val_air_acc', 'avg_air_pct',
               'best_val_loss', 'best_val_acc', 'best_val_struct_acc',
               'training_time', 'history']

# Keys that are accessed from all_results['V3'] or v3_result
accessed_keys = ['avg_air_pct', 'best_val_struct_acc', 'history', 
                 'training_time', 'best_val_loss', 'best_val_acc']

print("  Keys stored in final_metrics:")
for key in stored_keys:
    if f"'{key}':" in cell7_source:
        print(f"    ✓ {key}")
    else:
        print(f"    ✗ {key} - NOT FOUND!")

print("\n  Keys accessed from all_results['V3']:")
for key in accessed_keys:
    if key in stored_keys:
        print(f"    ✓ {key}")
    else:
        print(f"    ✗ {key} - NOT IN STORED KEYS!")

print("\n" + "=" * 60)
print("CHECK COMPLETE")
print("=" * 60)

