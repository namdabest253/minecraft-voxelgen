"""Create interactive t-SNE visualization with toggleable shapes using Plotly."""
import json
import numpy as np
from sklearn.manifold import TSNE
from pathlib import Path
from PIL import Image
from collections import defaultdict
import base64
from io import BytesIO
import platform
import warnings
warnings.filterwarnings('ignore')

# Check for plotly
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("Installing plotly...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'plotly'])
    import plotly.graph_objects as go

# Paths - detect WSL vs Windows
if platform.system() == "Linux" and "microsoft" in platform.release().lower():
    BASE_PATH = Path("/mnt/c/Users/namda/OneDrive/Desktop/Claude_Server/minecraft_ai")
else:
    BASE_PATH = Path("C:/Users/namda/OneDrive/Desktop/Claude_Server/minecraft_ai")

EMBEDDINGS_PATH = BASE_PATH / "data/kaggle/output/block2vec/v3/block_embeddings_v3.npy"
# CRITICAL: Must use FULL vocabulary (3717 entries) that matches embeddings, NOT collapsed (1007)
VOCAB_PATH = BASE_PATH / "data/vocabulary/tok2block.json"
TEXTURES_PATH = BASE_PATH / "data/kaggle/plot/block_textures"
OUTPUT_PATH = BASE_PATH / "data/kaggle/output/block2vec/v3/tsne_interactive_fixed.html"

# Shape configuration with colors
SHAPE_CONFIG = {
    'planks':      {'color': '#FF6B6B', 'name': 'Planks'},
    'stairs':      {'color': '#4ECDC4', 'name': 'Stairs'},
    'slab':        {'color': '#45B7D1', 'name': 'Slabs'},
    'fence':       {'color': '#96CEB4', 'name': 'Fence'},
    'fence_gate':  {'color': '#88D8B0', 'name': 'Fence Gate'},
    'door':        {'color': '#FFEAA7', 'name': 'Doors'},
    'trapdoor':    {'color': '#FFD93D', 'name': 'Trapdoors'},
    'wall':        {'color': '#F4A460', 'name': 'Walls'},
    'log':         {'color': '#DDA0DD', 'name': 'Logs'},
    'wood':        {'color': '#DA70D6', 'name': 'Wood'},
    'stripped_log':{'color': '#EE82EE', 'name': 'Stripped Log'},
    'stripped_wood':{'color': '#FF00FF', 'name': 'Stripped Wood'},
    'leaves':      {'color': '#90EE90', 'name': 'Leaves'},
    'wool':        {'color': '#FFB6C1', 'name': 'Wool'},
    'concrete':    {'color': '#87CEEB', 'name': 'Concrete'},
    'concrete_powder': {'color': '#ADD8E6', 'name': 'Concrete Powder'},
    'terracotta':  {'color': '#CD853F', 'name': 'Terracotta'},
    'glazed_terracotta': {'color': '#D2691E', 'name': 'Glazed Terracotta'},
    'carpet':      {'color': '#FFC0CB', 'name': 'Carpet'},
    'bed':         {'color': '#FF69B4', 'name': 'Beds'},
    'glass':       {'color': '#E0FFFF', 'name': 'Glass'},
    'pane':        {'color': '#B0E0E6', 'name': 'Glass Pane'},
    'stained_glass': {'color': '#AFEEEE', 'name': 'Stained Glass'},
    'stained_glass_pane': {'color': '#98FB98', 'name': 'Stained Glass Pane'},
    'block':       {'color': '#A9A9A9', 'name': 'Blocks'},
    'bricks':      {'color': '#B22222', 'name': 'Bricks'},
    'ore':         {'color': '#FFD700', 'name': 'Ores'},
    'button':      {'color': '#FF4500', 'name': 'Buttons'},
    'pressure_plate': {'color': '#FF6347', 'name': 'Pressure Plates'},
    'sign':        {'color': '#DEB887', 'name': 'Signs'},
    'banner':      {'color': '#9932CC', 'name': 'Banners'},
    'candle':      {'color': '#FFA07A', 'name': 'Candles'},
    'lantern':     {'color': '#FFE4B5', 'name': 'Lanterns'},
    'torch':       {'color': '#FF8C00', 'name': 'Torches'},
    'flower':      {'color': '#FF1493', 'name': 'Flowers'},
    'sapling':     {'color': '#228B22', 'name': 'Saplings'},
    'potted_plant':{'color': '#8B4513', 'name': 'Potted Plants'},
    'chest':       {'color': '#8B4513', 'name': 'Chests'},
    'shulker_box': {'color': '#9370DB', 'name': 'Shulker Boxes'},
    'coral':       {'color': '#FF7F50', 'name': 'Coral'},
    'coral_block': {'color': '#FF6347', 'name': 'Coral Block'},
}

DEFAULT_CONFIG = {'color': '#666666', 'name': 'Other'}


def identify_shape(block_name: str) -> str:
    """Identify the shape from a block name."""
    name = block_name.replace("minecraft:", "")
    if "[" in name:
        name = name.split("[")[0]
    
    if name.startswith("stripped_"):
        if "_log" in name:
            return "stripped_log"
        elif "_wood" in name:
            return "stripped_wood"
    
    if name.startswith("potted_"):
        return "potted_plant"
    
    shape_patterns = [
        ("_stained_glass_pane", "stained_glass_pane"),
        ("_stained_glass", "stained_glass"),
        ("_glazed_terracotta", "glazed_terracotta"),
        ("_concrete_powder", "concrete_powder"),
        ("_pressure_plate", "pressure_plate"),
        ("_hanging_sign", "sign"),
        ("_wall_sign", "sign"),
        ("_fence_gate", "fence_gate"),
        ("_coral_block", "coral_block"),
        ("_shulker_box", "shulker_box"),
        ("_trapdoor", "trapdoor"),
        ("_concrete", "concrete"),
        ("_terracotta", "terracotta"),
        ("_planks", "planks"),
        ("_stairs", "stairs"),
        ("_button", "button"),
        ("_carpet", "carpet"),
        ("_slab", "slab"),
        ("_fence", "fence"),
        ("_door", "door"),
        ("_wall", "wall"),
        ("_sign", "sign"),
        ("_wool", "wool"),
        ("_log", "log"),
        ("_wood", "wood"),
        ("_bed", "bed"),
        ("_ore", "ore"),
        ("_pane", "pane"),
        ("_glass", "glass"),
        ("_leaves", "leaves"),
        ("_sapling", "sapling"),
        ("_banner", "banner"),
        ("_candle", "candle"),
        ("_coral", "coral"),
        ("bricks", "bricks"),
        ("lantern", "lantern"),
        ("torch", "torch"),
        ("chest", "chest"),
    ]
    
    for pattern, shape in shape_patterns:
        if pattern in name:
            return shape
    
    return "block"


def get_texture_path(block_name: str) -> Path:
    """Find texture file for a block."""
    name = block_name.replace("minecraft:", "")
    if "[" in name:
        name = name.split("[")[0]
    
    patterns = [
        f"{name}.png",
        f"{name}_side.png",
        f"{name}_front.png",
        f"{name}_top.png",
        f"{name}s.png",
    ]
    
    if "_stairs" in name:
        base = name.replace("_stairs", "")
        patterns.extend([f"{base}.png", f"{base}_planks.png", f"{base}s.png"])
    elif "_slab" in name:
        base = name.replace("_slab", "")
        patterns.extend([f"{base}.png", f"{base}_planks.png", f"{base}s.png"])
    elif "_fence" in name:
        base = name.replace("_fence", "").replace("_gate", "")
        patterns.extend([f"{base}.png", f"{base}_planks.png"])
    elif "_door" in name:
        patterns.extend([f"{name}_bottom.png", f"{name}_top.png"])
    elif "_wall" in name:
        base = name.replace("_wall", "")
        patterns.extend([f"{base}.png", f"{base}s.png"])
    elif "_wool" in name:
        patterns.append(f"{name}.png")
    elif "_concrete" in name and "_powder" not in name:
        patterns.append(f"{name}.png")
    elif "_button" in name:
        base = name.replace("_button", "")
        patterns.extend([f"{base}.png", f"{base}_planks.png"])
    elif "_sign" in name:
        base = name.replace("_sign", "").replace("_wall", "").replace("_hanging", "")
        patterns.extend([f"{base}.png", f"{base}_planks.png"])
    elif "ore" in name:
        patterns.append(f"{name}.png")
    
    for pattern in patterns:
        path = TEXTURES_PATH / pattern
        if path.exists():
            return path
    
    return None


def load_texture_as_base64(path: Path, size: int = 16) -> str:
    """Load texture and convert to base64 for embedding in HTML."""
    try:
        img = Image.open(path).convert("RGBA")
        img = img.resize((size, size), Image.Resampling.NEAREST)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    except Exception:
        return None


def main():
    print("Loading embeddings...")
    embeddings = np.load(EMBEDDINGS_PATH)
    print(f"Embeddings shape: {embeddings.shape}")
    
    print("Loading vocabulary...")
    with open(VOCAB_PATH) as f:
        tok2block = {int(k): v for k, v in json.load(f).items()}
    print(f"Vocabulary size: {len(tok2block)}")
    
    # Find textures and shapes for all blocks
    # IMPORTANT: The full vocabulary has block states (e.g., oak_log[axis=x], oak_log[axis=y])
    # We deduplicate by taking the FIRST token for each base block name
    print("\nProcessing blocks...")
    block_data = {}
    seen_blocks = set()  # Track which base block names we've already added
    
    for tok, block_name in sorted(tok2block.items()):  # Sort to get consistent first token
        if block_name == "UNKNOWN_BLOCK":
            continue
        if "air" in block_name.lower() and "stair" not in block_name.lower():
            continue
        
        # Get the base block name (without state)
        clean_name = block_name.replace("minecraft:", "").split("[")[0]
        
        # Skip if we've already added this base block
        if clean_name in seen_blocks:
            continue
        
        tex_path = get_texture_path(block_name)
        if tex_path:
            tex_b64 = load_texture_as_base64(tex_path, size=32)
            if tex_b64:
                seen_blocks.add(clean_name)
                block_data[tok] = {
                    'name': clean_name,
                    'shape': identify_shape(block_name),
                    'texture_b64': tex_b64
                }
    
    print(f"Found {len(block_data)} unique blocks with textures (from {len(tok2block)} vocab entries)")
    
    # Get valid tokens and embeddings
    valid_tokens = list(block_data.keys())
    valid_embeddings = embeddings[valid_tokens]
    
    # Run t-SNE
    print("\nRunning t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(valid_tokens)-1))
    coords = tsne.fit_transform(valid_embeddings)
    print("t-SNE complete!")
    
    # Group blocks by shape
    shape_groups = defaultdict(list)
    for i, tok in enumerate(valid_tokens):
        shape = block_data[tok]['shape']
        shape_groups[shape].append({
            'token': tok,
            'index': i,
            'name': block_data[tok]['name'],
            'x': float(coords[i, 0]),
            'y': float(coords[i, 1]),
            'texture_b64': block_data[tok]['texture_b64']
        })
    
    # Sort shapes by count
    sorted_shapes = sorted(shape_groups.items(), key=lambda x: -len(x[1]))
    
    print(f"\nShape distribution:")
    for shape, blocks in sorted_shapes[:15]:
        config = SHAPE_CONFIG.get(shape, DEFAULT_CONFIG)
        print(f"  {shape}: {len(blocks)} blocks")
    
    # Create Plotly figure
    print("\nCreating interactive visualization...")
    fig = go.Figure()
    
    # Calculate image size based on data range
    all_x = [b['x'] for blocks in shape_groups.values() for b in blocks]
    all_y = [b['y'] for blocks in shape_groups.values() for b in blocks]
    x_range = max(all_x) - min(all_x)
    y_range = max(all_y) - min(all_y)
    img_size = min(x_range, y_range) / 50  # Smaller textures to reduce overlap
    
    # Add textures as layout images
    print("Adding texture images...")
    images_list = []
    for shape, blocks in sorted_shapes:
        for b in blocks:
            images_list.append(dict(
                source=f"data:image/png;base64,{b['texture_b64']}",
                xref="x",
                yref="y",
                x=b['x'],
                y=b['y'],
                sizex=img_size,
                sizey=img_size,
                xanchor="center",
                yanchor="middle",
                layer="above",
            ))
    
    print(f"Added {len(images_list)} texture images")
    
    # Add scatter traces for interactivity (invisible markers for hover/legend)
    for shape, blocks in sorted_shapes:
        config = SHAPE_CONFIG.get(shape, DEFAULT_CONFIG)
        
        x_vals = [b['x'] for b in blocks]
        y_vals = [b['y'] for b in blocks]
        names = [b['name'] for b in blocks]
        
        # Create hover text with block name
        hover_text = [f"<b>{name}</b><br>Shape: {shape}" for name in names]
        
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='markers',
            name=f"{config['name']} ({len(blocks)})",
            marker=dict(
                size=8,  # Small clickable area
                color=config['color'],
                opacity=0.0,  # Invisible - just for hover detection
                symbol='square'
            ),
            text=hover_text,
            hoverinfo='text',
            hoverlabel=dict(bgcolor='#2d2d44', font_size=14),
            legendgroup=shape,
        ))
    
    # Update layout with images
    fig.update_layout(
        title=dict(
            text='Block2Vec V3 t-SNE - Interactive Shape Visualization',
            font=dict(size=20, color='white'),
            x=0.5
        ),
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#1a1a2e',
        font=dict(color='white'),
        legend=dict(
            title=dict(text='Shapes (click to toggle)', font=dict(size=14)),
            bgcolor='rgba(45, 45, 68, 0.9)',
            bordercolor='white',
            borderwidth=1,
            font=dict(size=11),
            itemsizing='constant',
            yanchor='top',
            y=0.99,
            xanchor='left',
            x=1.02,
        ),
        xaxis=dict(
            title='t-SNE 1',
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            zeroline=False,
            color='white',
            scaleanchor='y',  # Keep aspect ratio
        ),
        yaxis=dict(
            title='t-SNE 2',
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            zeroline=False,
            color='white'
        ),
        hovermode='closest',
        width=1400,
        height=900,
        margin=dict(r=250),
        images=images_list,  # Add all texture images
    )
    
    # Add instructions
    fig.add_annotation(
        text="💡 Click legend items to toggle shapes | Double-click to isolate | Drag to zoom",
        xref="paper", yref="paper",
        x=0.5, y=-0.08,
        showarrow=False,
        font=dict(size=12, color='#888888'),
    )
    
    # Save as HTML
    fig.write_html(
        OUTPUT_PATH,
        include_plotlyjs=True,
        full_html=True,
        config={
            'displayModeBar': True,
            'scrollZoom': True,
            'modeBarButtonsToAdd': ['toggleSpikelines'],
        }
    )
    print(f"\nSaved interactive visualization to: {OUTPUT_PATH}")
    print("\nOpen the HTML file in a browser to use the interactive features!")
    print("  - Click legend items to show/hide shapes")
    print("  - Double-click a legend item to isolate that shape")
    print("  - Drag to zoom, double-click to reset")
    print("  - Hover over points to see block names")


if __name__ == "__main__":
    main()

