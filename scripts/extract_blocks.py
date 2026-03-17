import h5py
import numpy as np
import json
import argparse

def get_build_blocks(h5_path, tok2block_path):
    # 1. Load the block mapping dictionary
    with open(tok2block_path, 'r') as f:
        tok2block = json.load(f)
        
    # 2. Load the 3D array from the .h5 file
    with h5py.File(h5_path, 'r') as f:
        build_array = np.array(f['build'])
        
    # 3. Find all unique integers used in the build
    unique_ids = np.unique(build_array)
    
    # 4. Map integers to names, ignoring 'minecraft:air' (ID 102) and UNKNOWN_BLOCK
    block_names = set()
    for block_id in unique_ids:
        block_str = str(block_id)
        if block_str in tok2block:
            name = tok2block[block_str]
            # Clean up the name: remove 'minecraft:' prefix and any block states '[...]'
            clean_name = name.replace('minecraft:', '').split('[')[0].replace('_', ' ')
            
            if clean_name not in ['air', 'void air', 'cave air', 'UNKNOWN BLOCK']:
                block_names.add(clean_name)
                
    return list(block_names)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract block vocabulary from an h5 build.")
    parser.add_argument("h5_file", help="Path to the .h5 build file")
    parser.add_argument("--map", default="/mnt/c/Users/namda/OneDrive/Desktop/Claude_Server/minecraft_ai/data/vocabulary/tok2block.json", help="Path to tok2block.json")
    args = parser.parse_args()
    
    blocks = get_build_blocks(args.h5_file, args.map)
    print(", ".join(blocks))
