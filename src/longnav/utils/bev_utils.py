import numpy as np
from PIL import Image as PILImage
from pathlib import Path
import einops


def load_depth(
    path: Path,
) -> np.ndarray:
    """
    Reverse the save_depth encoding to retrieve depth in meters.
    
    Inverse Logic:
      y = x / (x + k)  -->  x = (y * k) / (1 - y)
      where x is depth_meters and y is normalized depth [0, 1).
    """
    # 1. Load the 16-bit PNG
    # distinct from standard PNGs, we interpret pixels as 16-bit integers
    img = PILImage.open(path)
    depth_u16 = np.asarray(img, dtype=float)

    # 2. Re-normalize to [0, 1]
    depth_meters = depth_u16 / 1000

    # 3. Invert the monotonic curve
    # Note: If a pixel value is exactly 65535 (saturated), the denominator
    # becomes 0. np.divide handles this by returning inf, which is mathematically correct
    # (x -> infinity as norm -> 1). We suppress the 'divide by zero' warning here.

    return depth_meters

def load_depths(example):
    depths = []
    for path in example['depth_sequence']:
        depths.append(load_depth(path))
    return np.array(depths)

def patch_average_einops(points: np.ndarray, patch_size: int) -> np.ndarray:
    """
    Downsamples the point cloud by averaging points within patches.
    
    Args:
        points: Shape (..., H, W, 3)
        patch_size: int (e.g., 14 or 16)
        
    Returns:
        Shape (..., h, w, 3) where h=H/patch_size, w=W/patch_size
    """
    # '... (h p1) (w p2) c' captures the batch dim in '...'
    # p1 and p2 capture the patch height/width
    return einops.reduce(
        points, 
        '... (h p1) (w p2) c -> ... h w c', 
        'mean', 
        p1=patch_size, 
        p2=patch_size
    )

def depth_to_pointcloud(depth: np.ndarray, fov_degrees: float = 90.0) -> np.ndarray:
    """
    Vectorized unprojection.
    
    Args:
        depth: Shape (H, W) or (B, H, W)
        fov_degrees: Horizontal Field of View
        
    Returns:
        points: Shape (..., H, W, 3) where ... matches input batch dims.
        Coordinates are CV standard: X Right, Y Down, Z Forward.
    """
    # 1. Determine shapes and intrinsics
    # We assume depth is at least 2D. The last two dims are H, W.
    H, W = depth.shape[-2:]
    
    # Calculate focal length from FOV
    # f = (W / 2) / tan(FOV / 2)
    f = (W / 2) / np.tan(np.deg2rad(fov_degrees / 2))
    
    # Principal point (center)
    cx, cy = W / 2, H / 2
    
    # 2. Create the Grid (Vectorized)
    # create generic [0...W-1] and [0...H-1]
    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v)  # shapes: (H, W)
    
    # 3. Unproject
    # Note: we subtract cx, cy then multiply by depth.
    # NumPy Broadcasting Magic:
    # uu (H, W) - cx (scalar) -> (H, W)
    # (H, W) * depth (B, H, W) -> (B, H, W) automatically aligns trailing dims.
    
    z = depth
    x = (uu - cx) * z / f
    y = (vv - cy) * z / f
    
    # 4. Stack
    # We stack along the LAST axis to get (..., H, W, 3)
    points = np.stack([x, y, z], axis=-1)
    
    return points

from scipy.spatial.transform import Rotation as R

def pos_rots_to_matrix(pos_rots: np.ndarray) -> np.ndarray:
    """
    Converts 7D vector [x, y, z, qx, qy, qz, qw] to 4x4 transformation matrix.
    
    Args:
        pos_rots: Numpy array of shape (..., 7)
    Returns:
        Numpy array of shape (..., 4, 4)
    """
    # Split position and rotation
    pos = pos_rots[..., :3]
    quat = pos_rots[..., 3:] # [qx, qy, qz, qw]
    
    # Create the 4x4 matrix container
    # Shape logic: if input is (N, 7), output is (N, 4, 4)
    batch_shape = pos_rots.shape[:-1]
    T = np.eye(4)
    # Broadcast identity if we have batch dims
    if batch_shape:
        T = np.tile(T, (*batch_shape, 1, 1))
        
    # Scipy handles the batch quaternion to rotation matrix conversion automatically
    # Note: Scipy expects scalar-last quaternions [x, y, z, w], which matches your format
    rot_matrices = R.from_quat(quat).as_matrix()
    
    T[..., :3, :3] = rot_matrices
    T[..., :3, 3] = pos
    
    return T

def get_cv_to_habitat_correction() -> np.ndarray:
    """
    Returns the static 4x4 matrix converting CV Frame -> Habitat Frame.
    Rotate 180 degrees around X-axis (flip Y and Z).
    """
    correction = np.eye(4)
    correction[1, 1] = -1
    correction[2, 2] = -1
    return correction

def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    Applies 4x4 transform to points (..., N, 3).
    """
    # 1. Homogeneous coordinates
    # Append 1s to the last dimension
    ones = np.ones((*points.shape[:-1], 1))
    points_hom = np.concatenate([points, ones], axis=-1)
    print(points_hom.shape)
    # 2. Matrix Multiplication
    # (..., N, 4) @ (..., 4, 4).T -> (..., N, 4)
    # We transpose the last two dims of the transform
    points_transformed = points_hom @ np.swapaxes(transform, -1, -2)
    
    # 3. Crop back to 3D
    return points_transformed[..., :3]

def transform_points_batch(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    Applies a batch of transforms to a batch of point clouds.
    
    Args:
        points: Shape (B, ..., 3)  <-- Can be (B, N, 3) or (B, H, W, 3)
        transform: Shape (B, 4, 4)
        
    Returns:
        points_transformed: Shape (B, ..., 3)
    """
    # 1. Homogeneous Coordinates
    # Append 1 to the last dim. 
    # Result shape: (B, ..., 4)
    ones = np.ones((*points.shape[:-1], 1))
    points_hom = np.concatenate([points, ones], axis=-1)
    
    points_transformed = np.einsum('b...i, bji -> b...j', points_hom, transform)
    
    # 3. Crop back to 3D
    return points_transformed[..., :3]
# --- Usage ---

def convert_cv_points_to_world(points_cv: np.ndarray, pos_rots: np.ndarray) -> np.ndarray:
    """
    Main entry point.
    points_cv: (..., N, 3)
    pos_rots: (..., 7)
    """
    # 1. Agent Pose (Habitat Frame)
    T_world_agent = pos_rots_to_matrix(pos_rots)
    
    # 2. Camera -> Agent Correction
    T_agent_cv = get_cv_to_habitat_correction()
    
    # 3. Combined Transform: Camera -> World
    # If dealing with batches, T_agent_cv broadcasts automatically
    T_world_cv = T_world_agent @ T_agent_cv
    
    # 4. Apply
    return transform_points(points_cv, T_world_cv)

def create_colored_pointcloud(points, rgb_image):
    import open3d as o3d
    # ... (Same reshaping/normalization logic as before) ...
    if rgb_image.ndim == 3:
        colors = rgb_image.reshape(-1, 3)
    else:
        colors = rgb_image
        
    if colors.max() > 1.001: 
        colors = colors.astype(np.float64) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # # Flip orientation for easier viewing (Camera to World)
    # # Rotating 180 deg around X-axis
    # R = pcd.get_rotation_matrix_from_xyz((np.pi, 0, 0))
    # pcd.rotate(R, center=(0, 0, 0))
    return pcd

def create_bev_mosaic(
    rgbs: np.ndarray, 
    patch_coords_discrete: np.ndarray, 
    patch_size: int = 32
):
    """
    Creates a Birds-Eye View (BEV) mosaic from image patches.
    
    Args:
        rgbs: Raw RGB images (B, H_img, W_img, 3)
        patch_coords_discrete: Discrete world coords (B, h_grid, w_grid, 3)
        patch_size: Size of the patches (e.g., 32)
        
    Returns:
        canvas: The resulting BEV image (H_map, W_map, 3)
    """
    # 1. Extract High-Res Patches from RGB images
    # We rearrange (B, H, W, C) -> (N, P, P, C)
    # This aligns perfectly with the flattened patch_coords
    patches = einops.rearrange(
        rgbs, 
        'b (h p1) (w p2) c -> (b h w) p1 p2 c', 
        p1=patch_size, 
        p2=patch_size
    )
    
    # 2. Flatten Coordinates and Project to 2D
    # We only care about X (right) and Z (back/forward) for BEV
    # Shape: (N, 3) -> (N, 2) taking columns 0 (x) and 2 (z)
    flat_coords = patch_coords_discrete.reshape(-1, 3)
    xz_coords = flat_coords[:, [0, 2]] 
    
    # 3. Vectorized Averaging of Overlapping Patches
    # Find unique (x, z) coordinates and the inverse mapping
    # unique_xz: (U, 2) - The list of unique grid locations
    # inverse_indices: (N,) - Maps every original patch to a unique location index
    unique_xz, inverse_indices = np.unique(xz_coords, axis=0, return_inverse=True)
    
    # Initialize container for averaged patches: (U, 32, 32, 3)
    unique_patches_sum = np.zeros((len(unique_xz), patch_size, patch_size, 3), dtype=np.float32)
    counts = np.zeros((len(unique_xz),), dtype=np.int32)
    
    # "Scatter Add" logic using numpy
    # This sums all patches that map to the same unique index
    np.add.at(unique_patches_sum, inverse_indices, patches)
    np.add.at(counts, inverse_indices, 1)
    
    # Compute mean (broadcasting the count division)
    unique_patches_avg = unique_patches_sum / counts[:, None, None, None]
    unique_patches_avg = np.clip(unique_patches_avg, 0, 255).astype(np.uint8)

    # 4. Construct the Canvas
    # Determine bounds
    min_x, min_z = unique_xz.min(axis=0)
    max_x, max_z = unique_xz.max(axis=0)
    
    grid_h = (max_z - min_z) + 1
    grid_w = (max_x - min_x) + 1
    
    # Canvas size in pixels
    canvas_h = grid_h * patch_size
    canvas_w = grid_w * patch_size
    
    # Initialize canvas (black background)
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    
    # 5. Paint the Canvas
    # Iterate over the UNIQUE patches (much faster than iterating over all N)
    for i, (gx, gz) in enumerate(unique_xz):
        # Calculate pixel offsets
        # Note: We flip Z logic if necessary. Here we map World Z -> Image Row
        row_idx = (gz - min_z)
        col_idx = (gx - min_x)
        
        y_start = row_idx * patch_size
        x_start = col_idx * patch_size
        
        canvas[y_start : y_start+patch_size, x_start : x_start+patch_size] = unique_patches_avg[i]
        
    return canvas

def get_thw(patch_coords_discrete):
    import torch
    flat_coords = patch_coords_discrete.view(-1, 3)
    # Find global mins to shift origin to (0,0,0)
    # Note: We compute min across the entire set to maintain relative structure
    min_vals = flat_coords.min(dim=0).values # (x_min, y_min, z_min)
    norm_coords = flat_coords - min_vals

    x_col = norm_coords[:, 0]
    y_col = norm_coords[:, 1]
    z_col = norm_coords[:, 2]
    
    # Qwen expects position_ids shape: [3, Batch, Seq_Len]
    # Dimension 0 is (Temporal, Height, Width)
    t_vals = y_col
    h_vals = z_col
    w_vals = x_col
    
    mapped_coords = torch.stack([t_vals, h_vals+t_vals, w_vals+t_vals], dim=0) # (3, Total_Patches)
    return mapped_coords

def get_pos_id(input_ids,map_coords,processor,canvas_size=2000):
    import torch
    '''
    input_ids: B by Seq
    map_coords: B by 3 by SeqImg, (possibly ragged)
    '''
    pos_ids = torch.arange(input_ids.shape[1]).repeat(3,1,1)
    for b in range(input_ids.shape[0]):
        seq = input_ids[b]
        pos_id = pos_ids[:,b,:]
        mc = map_coords[b]
        try:
            vision_origin = torch.where(seq==processor.vision_start_token_id)[0][0]+1
        except:
            continue
        img_mask = (seq == processor.image_token_id)
        pos_id[:,img_mask]=mc+vision_origin

        trailing_text_mask = ~img_mask[vision_origin:]
        trailing_text_num = torch.sum(trailing_text_mask)
        # print(pos_id.shape)
        pos_id[:,vision_origin:][:,trailing_text_mask] = torch.arange(trailing_text_num).repeat(3,1)+vision_origin+canvas_size
    return pos_ids

def get_patch_coords(pos_rots,depths,patch_size=32,resolution=0.15,fov_degrees=79):
    '''
    maps square patches of the depth image to voxel grid positions in world frame.
    note: patch size MUST divide the image dimensions.
    - pos_rots: B by 7 poses
    - depths: B by H by W depth image
    - resolution: size of each voxel
    '''
    mats = pos_rots_to_matrix(pos_rots) @ get_cv_to_habitat_correction()
    points = depth_to_pointcloud(depths,fov_degrees=fov_degrees)
    patch_coords = patch_average_einops(points,patch_size=patch_size)
    patch_coords_world = transform_points_batch(patch_coords,mats)
    patch_coords_discrete = np.round(patch_coords_world/resolution).astype(int) #TODO: need to round to closest grid. 
    return patch_coords_discrete