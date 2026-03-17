"""
Voxel Renderer — 3D structures to 2D orthographic images.

Renders 32x32x32 Minecraft structures as 2D images for CLIP scoring.
Pure NumPy implementation — no GPU or OpenGL dependencies.

Produces 4 views per structure:
- Top-down (-Y axis): floor plan / roof layout
- Front (-Z axis): facade, height profile
- Right side (-X axis): side profile
- Isometric (oblique from upper corner): overall 3D shape

Each view is 224x224 RGB (CLIP's native input size).
Ambient occlusion and edge outlines are applied by default
to improve 3D depth cues for CLIP scoring.
"""

from typing import Dict, Optional, Tuple

import numpy as np


def _compute_ao_map(
    block_ids: np.ndarray,
    air_tokens: set = frozenset({102, 576, 3352}),
) -> np.ndarray:
    """Compute ambient occlusion factors for each voxel.

    For each non-air voxel, counts how many of its 6 neighbors are also
    non-air (occluded). More neighbors = darker (more occluded).

    Args:
        block_ids: [X, Y, Z] voxel grid.
        air_tokens: Set of air token IDs.

    Returns:
        [X, Y, Z] float32 array of AO factors (0.6–1.0).
        Air voxels get 1.0 (no darkening).
    """
    occupied = np.ones(block_ids.shape, dtype=bool)
    for tok in air_tokens:
        occupied &= block_ids != tok
    occupied_f = occupied.astype(np.float32)

    # Count occupied neighbors along all 6 directions
    neighbor_count = np.zeros_like(occupied_f)
    for ax in range(3):
        # Shift forward (+1)
        neighbor_count += np.roll(occupied_f, -1, axis=ax)
        # Shift backward (-1)
        neighbor_count += np.roll(occupied_f, 1, axis=ax)

    # AO factor: 1.0 - 0.08 * count, clamped to [0.6, 1.0]
    # Max neighbors = 6 -> 1.0 - 0.48 = 0.52, but we clamp at 0.6
    ao_map = 1.0 - 0.08 * neighbor_count
    ao_map = np.clip(ao_map, 0.6, 1.0)

    # Air voxels should not be darkened
    ao_map[~occupied] = 1.0

    return ao_map


def _project_orthographic(
    block_ids: np.ndarray,
    block_colors: Dict[int, Tuple[int, int, int]],
    axis: int,
    direction: int = -1,
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    air_tokens: set = frozenset({102, 576, 3352}),
    ao_map: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Render an orthographic projection along one axis.

    Casts rays along the given axis direction. For each pixel, finds the
    first non-air block and uses its color, optionally darkened by AO.

    Args:
        block_ids: [X, Y, Z] voxel grid (32x32x32).
        block_colors: {token_id: (R, G, B)} mapping.
        axis: Projection axis (0=X, 1=Y, 2=Z).
        direction: -1 for negative direction, +1 for positive.
        bg_color: Background color for air-only rays.
        air_tokens: Set of air token IDs.
        ao_map: Optional [X, Y, Z] float32 AO factors from _compute_ao_map.

    Returns:
        [H, W, 3] uint8 image where H and W depend on the projection axis.
    """
    shape = block_ids.shape  # (X, Y, Z) = (32, 32, 32)

    # Determine the 2D image axes based on projection axis
    # axis=0 (X): image is (Z, Y) — looking from side
    # axis=1 (Y): image is (X, Z) — looking from top
    # axis=2 (Z): image is (X, Y) — looking from front
    if axis == 0:  # Right side view (-X)
        h_size, w_size = shape[2], shape[1]  # Z, Y
    elif axis == 1:  # Top-down view (-Y)
        h_size, w_size = shape[0], shape[2]  # X, Z
    else:  # Front view (-Z)
        h_size, w_size = shape[0], shape[1]  # X, Y

    # RGBA: 4 channels, alpha=0 for transparent background
    image = np.zeros((h_size, w_size, 4), dtype=np.float32)

    # Iterate along ray direction
    depth_range = range(shape[axis])
    if direction < 0:
        depth_range = reversed(list(depth_range))

    # Build a mask of which pixels have been filled
    filled = np.zeros((h_size, w_size), dtype=bool)

    for d in depth_range:
        if filled.all():
            break

        # Extract the slice at depth d along the projection axis
        if axis == 0:
            slab = block_ids[d, :, :]  # [Y, Z]
            slab = slab.T  # [Z, Y] for image coords
            ao_slab = ao_map[d, :, :].T if ao_map is not None else None
        elif axis == 1:
            slab = block_ids[:, d, :]  # [X, Z]
            ao_slab = ao_map[:, d, :] if ao_map is not None else None
        else:
            slab = block_ids[:, :, d]  # [X, Y]
            ao_slab = ao_map[:, :, d] if ao_map is not None else None

        # Find non-air blocks in this slab that haven't been filled
        for token_id in np.unique(slab):
            if int(token_id) in air_tokens:
                continue
            mask = (slab == token_id) & ~filled
            if not mask.any():
                continue
            color = np.array(
                block_colors.get(int(token_id), (128, 128, 128)), dtype=np.float32
            )
            if ao_slab is not None:
                ao_factors = ao_slab[mask]  # [N] array of AO factors
                image[mask, :3] = color[None, :] * ao_factors[:, None]
            else:
                image[mask, :3] = color
            image[mask, 3] = 255.0  # opaque
            filled |= mask

    return np.clip(image, 0, 255).astype(np.uint8)


def _render_isometric(
    block_ids: np.ndarray,
    block_colors: Dict[int, Tuple[int, int, int]],
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    air_tokens: set = frozenset({102, 576, 3352}),
    enable_face_shading: bool = True,
) -> np.ndarray:
    """Render an isometric (oblique) projection with face shading.

    Uses a simple oblique projection from the upper-right-front corner.
    Each voxel projects to a 2D position based on its (x, y, z) coordinates.
    Face shading applies directional lighting: top=100%, left=80%, right=60%.

    Args:
        block_ids: [X, Y, Z] voxel grid (32x32x32).
        block_colors: {token_id: (R, G, B)} mapping.
        bg_color: Background color.
        air_tokens: Set of air token IDs.
        enable_face_shading: Whether to apply directional face shading.

    Returns:
        [size, size, 3] uint8 image.
    """
    sx, sy, sz = block_ids.shape

    # Isometric projection: x_2d = (x - z) * cos(30), y_2d = -y + (x + z) * sin(30)
    cos30 = 0.866
    sin30 = 0.5

    # Calculate image bounds (4x for larger voxels in the output)
    img_size = int(max(sx, sy, sz) * 4)
    cx, cy = img_size // 2, img_size // 2

    # RGBA: 4 channels, alpha=0 for transparent background
    image = np.zeros((img_size, img_size, 4), dtype=np.float32)
    depth_buffer = np.full((img_size, img_size), -np.inf)

    # Build occupancy grid for face visibility checks
    air_set = set(air_tokens)
    occupied = np.ones(block_ids.shape, dtype=bool)
    for tok in air_set:
        occupied &= block_ids != tok

    non_air = np.argwhere(occupied)

    if len(non_air) == 0:
        return image.astype(np.uint8)

    for x, y, z in non_air:
        token_id = int(block_ids[x, y, z])

        # Isometric projection (top-down view, looking from upper-front-right)
        px = int(cx + (x - z) * cos30)
        py = int(cy + y * 1.2 - (x + z) * sin30)

        # Depth for painter's algorithm (closer to camera = higher depth)
        depth = x + z - y

        if 0 <= px < img_size and 0 <= py < img_size:
            if depth > depth_buffer[py, px]:
                depth_buffer[py, px] = depth
                color = np.array(
                    block_colors.get(token_id, (128, 128, 128)), dtype=np.float32
                )

                if enable_face_shading:
                    color = _apply_face_shading(
                        color, x, y, z, occupied, block_ids.shape
                    )

                image[py, px, :3] = color
                image[py, px, 3] = 255.0  # opaque

    return np.clip(image, 0, 255).astype(np.uint8)


def _apply_face_shading(
    color: np.ndarray,
    x: int,
    y: int,
    z: int,
    occupied: np.ndarray,
    shape: Tuple[int, ...],
) -> np.ndarray:
    """Apply directional face shading to an isometric voxel.

    Determines the most visible face and applies brightness:
    - Top face (y-1 is air): 100% brightness
    - Left face (z+1 is air): 80% brightness
    - Right face (x+1 is air): 60% brightness

    Args:
        color: [3] float32 base color.
        x, y, z: Voxel position.
        occupied: [X, Y, Z] bool occupancy grid.
        shape: Grid dimensions.

    Returns:
        [3] float32 shaded color.
    """
    sx, sy, sz = shape

    # Check which faces are exposed (neighbor is air / out of bounds)
    top_exposed = y == 0 or not occupied[x, y - 1, z]
    left_exposed = z == sz - 1 or not occupied[x, y, z + 1]
    right_exposed = x == sx - 1 or not occupied[x + 1, y, z]

    # Pick the brightest exposed face (priority: top > left > right)
    if top_exposed:
        factor = 1.0
    elif left_exposed:
        factor = 0.8
    elif right_exposed:
        factor = 0.6
    else:
        # Interior voxel visible due to projection — darken more
        factor = 0.5

    return color * factor


def _crop_to_content(
    image: np.ndarray,
    padding: int = 2,
) -> np.ndarray:
    """Crop RGBA image to its non-transparent bounding box with padding.

    Args:
        image: [H, W, 4] uint8 RGBA image.
        padding: Pixels of padding around content.

    Returns:
        Cropped [H', W', 4] uint8 RGBA image (square, transparent padding).
    """
    # Use alpha channel to find content
    mask = image[:, :, 3] > 0

    if not mask.any():
        return image

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols)[0][[0, -1]]

    # Add padding
    h, w = image.shape[:2]
    r_min = max(0, r_min - padding)
    r_max = min(h - 1, r_max + padding)
    c_min = max(0, c_min - padding)
    c_max = min(w - 1, c_max + padding)

    cropped = image[r_min : r_max + 1, c_min : c_max + 1]

    # Make square by padding the shorter dimension
    ch, cw = cropped.shape[:2]
    size = max(ch, cw)
    square = np.zeros((size, size, 4), dtype=image.dtype)  # transparent
    y_off = (size - ch) // 2
    x_off = (size - cw) // 2
    square[y_off : y_off + ch, x_off : x_off + cw] = cropped

    return square


def _apply_edge_outlines(
    image: np.ndarray,
    blend: float = 0.4,
) -> np.ndarray:
    """Apply dark edge outlines at color boundaries.

    For each pixel, checks 4 neighbors (up/down/left/right). If any neighbor
    has a different color and neither pixel is transparent, the pixel is
    darkened by blending with black.

    Args:
        image: [H, W, 4] uint8 RGBA image.
        blend: Blend factor with black (0=no change, 1=fully black).

    Returns:
        [H, W, 4] uint8 RGBA image with edge outlines.
    """
    h, w = image.shape[:2]

    # Detect transparent (background) pixels via alpha channel
    is_bg = image[:, :, 3] == 0  # [H, W]

    # Compare only RGB channels for color differences
    rgb = image[:, :, :3]

    # Check 4-connected neighbors for color differences
    edge_mask = np.zeros((h, w), dtype=bool)

    for shift_axis, shift_dir in [(0, 1), (0, -1), (1, 1), (1, -1)]:
        shifted = np.roll(rgb, shift_dir, axis=shift_axis)
        shifted_bg = np.roll(is_bg, shift_dir, axis=shift_axis)

        # Pixels differ from their neighbor
        differs = np.any(rgb != shifted, axis=2)
        # Neither this pixel nor the neighbor is transparent
        neither_bg = ~is_bg & ~shifted_bg

        edge_mask |= differs & neither_bg

    # Darken edge pixels by blending RGB with black (preserve alpha)
    result = image.astype(np.float32)
    result[edge_mask, :3] *= 1.0 - blend

    return np.clip(result, 0, 255).astype(np.uint8)


def _upscale_nearest(image: np.ndarray, target_size: int) -> np.ndarray:
    """Upscale image using nearest-neighbor interpolation.

    Args:
        image: [H, W, C] uint8 image (RGB or RGBA).
        target_size: Target height and width.

    Returns:
        [target_size, target_size, C] uint8 image.
    """
    h, w = image.shape[:2]
    row_indices = (np.arange(target_size) * h / target_size).astype(int)
    col_indices = (np.arange(target_size) * w / target_size).astype(int)
    row_indices = np.clip(row_indices, 0, h - 1)
    col_indices = np.clip(col_indices, 0, w - 1)
    return image[np.ix_(row_indices, col_indices)]


def render_structure_multiview(
    block_ids: np.ndarray,
    block_colors: Dict[int, Tuple[int, int, int]],
    image_size: int = 224,
    enable_ao: bool = True,
    enable_outlines: bool = True,
) -> np.ndarray:
    """Render a 2x2 composite of 4 views of a 3D structure.

    Layout: top-left=top, top-right=front, bottom-left=right, bottom-right=iso.

    Args:
        block_ids: [32, 32, 32] voxel grid of block token IDs.
        block_colors: {token_id: (R, G, B)} color mapping.
        image_size: Output image size (default 224 for CLIP).
        enable_ao: Apply ambient occlusion darkening (default True).
        enable_outlines: Apply edge outlines at block boundaries (default True).

    Returns:
        [image_size, image_size, 4] uint8 RGBA composite image.
    """
    air_tokens = frozenset({102, 576, 3352})

    # Compute AO map once for all orthographic views
    ao_map = _compute_ao_map(block_ids, air_tokens) if enable_ao else None

    # Render 4 views
    top = _project_orthographic(
        block_ids, block_colors, axis=1, direction=-1,
        air_tokens=air_tokens, ao_map=ao_map,
    )
    front = _project_orthographic(
        block_ids, block_colors, axis=2, direction=-1,
        air_tokens=air_tokens, ao_map=ao_map,
    )
    right = _project_orthographic(
        block_ids, block_colors, axis=0, direction=-1,
        air_tokens=air_tokens, ao_map=ao_map,
    )
    iso = _render_isometric(
        block_ids, block_colors, air_tokens=air_tokens,
        enable_face_shading=enable_ao,
    )

    # Crop isometric to content so structure fills the frame
    iso = _crop_to_content(iso)

    # Orientation fixes:
    # Front/Right: rotate 90° CCW so ground is at the bottom
    front = np.rot90(front, k=1)
    right = np.rot90(right, k=1)
    # Isometric: rotate 180° (was upside down after top-down POV change)
    iso = np.rot90(iso, k=2)

    # Upscale each view directly to half the composite size
    half = image_size // 2
    top = _upscale_nearest(top, half)
    front = _upscale_nearest(front, half)
    right = _upscale_nearest(right, half)
    iso = _upscale_nearest(iso, half)

    # Apply edge outlines after upscale
    if enable_outlines:
        top = _apply_edge_outlines(top)
        front = _apply_edge_outlines(front)
        right = _apply_edge_outlines(right)
        iso = _apply_edge_outlines(iso)

    # Assemble 2x2 composite grid
    composite = np.zeros((image_size, image_size, 4), dtype=np.uint8)
    composite[:half, :half] = top
    composite[:half, half:image_size] = front
    composite[half:image_size, :half] = right
    composite[half:image_size, half:image_size] = iso

    return composite
