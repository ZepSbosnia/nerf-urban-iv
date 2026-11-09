import torch
import torch.nn.functional as F
#import astra
import numpy as np
import plotly.graph_objects as go
import numpy as np 
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from phantominator import shepp_logan
import os
import matplotlib.pyplot as plt
import json 
import math 
from scipy.spatial.transform import Rotation as R
from packaging import version as pver

def plot_rgb_and_density_slices(rgb_volume, density_volume, slices_z, log_scale=False, figsize=(12, 6)):
    n_slices = len(slices_z)
    fig, axes = plt.subplots(2, n_slices, figsize=figsize)

    epsilon = 1e-6  # For log scale to avoid log(0)

    for i, z in enumerate(slices_z):
        # RGB slice
        axes[0, i].imshow(rgb_volume[z])
        axes[0, i].set_title(f"RGB z={z}")
        axes[0, i].axis('off')

        # Density slice
        dens_slice = density_volume[z]
        if log_scale:
            dens_slice = np.log(dens_slice + epsilon)

        axes[1, i].imshow(dens_slice, cmap='gray')
        axes[1, i].set_title(f"Density z={z}")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()
    
def plot_rgb_slices(rgb_volume, slices_z, figsize=(12, 6)):
    n_slices = len(slices_z)
    fig, axes = plt.subplots(1, n_slices, figsize=figsize)

    for i, z in enumerate(slices_z):
        ax = axes[i]
        ax.imshow(rgb_volume[z])  # Slice at z (shape: [Y, X, 3])
        ax.set_title(f"Slice z={z}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    
    


def get_custom_3Dphantom_RGB(res=128, scale=1e0, type='fake_palace'):
    if type == 'fake_palace': 
         
        phantom = np.zeros((res, res, res))
        center = res // 2
        cube_size = res // 4  # size of the main cube

        # Step 1: Main cube
        x_min, x_max = center - cube_size, center + cube_size
        y_min, y_max = center - cube_size, center + cube_size
        z_min, z_max = center - cube_size, center + cube_size
        phantom[x_min:x_max, y_min:y_max, z_min:z_max] = 1.0

        # Step 2: Second square prism, sticking out from the upper face
        small_cube_size = cube_size // 2
        shift = cube_size  # shift to stick outside

        # 3D grid
        X, Y, Z = np.meshgrid(np.arange(res), np.arange(res), np.arange(res), indexing='ij')

        # Prism center: move along +z to be on top of main cube
        xc_shift, yc_shift, zc_shift = center, center, center + shift

        # Prism mask
        prism_mask = (
            (np.abs(X - xc_shift) < small_cube_size) &
            (np.abs(Y - yc_shift) < small_cube_size) &
            (np.abs(Z - zc_shift) < small_cube_size)
        )

        phantom[prism_mask] = 0.8  # slightly lower density for the protruding square

        # Step 3: Rotated smaller cube as a hole (inside main cube)
        small_cube_size = cube_size // 2 - 5
        Xc, Yc, Zc = X - center, Y - center, Z - center


        angle_z = np.deg2rad(30)
        angle_x = np.deg2rad(20)
        angle_y = np.deg2rad(15)  # Add y-axis rotation

        cos_z, sin_z = np.cos(angle_z), np.sin(angle_z)
        cos_x, sin_x = np.cos(angle_x), np.sin(angle_x)
        cos_y, sin_y = np.cos(angle_y), np.sin(angle_y)

        Rz = np.array([
            [cos_z, -sin_z, 0],
            [sin_z,  cos_z, 0],
            [0,      0,     1]
        ])

        Rx = np.array([
            [1, 0,      0],
            [0, cos_x, -sin_x],
            [0, sin_x,  cos_x]
        ])

        Ry = np.array([
            [cos_y,  0, sin_y],
            [0,      1, 0],
            [-sin_y, 0, cos_y]
        ])

        # Combined rotation (apply in order: z, y, x)
        R = Rz @ Ry @ Rx

        coords = np.stack([Xc.flatten(), Yc.flatten(), Zc.flatten()], axis=0)
        rotated_coords = R @ coords
        Xr, Yr, Zr = rotated_coords.reshape(3, res, res, res)

        xc_shift, yc_shift, zc_shift = 0, 0, 0  # centered
        rotated_cube_mask = (
            (np.abs(Xr - xc_shift) < small_cube_size) &
            (np.abs(Yr - yc_shift) < small_cube_size) &
            (np.abs(Zr - zc_shift) < small_cube_size)
        )

        phantom[rotated_cube_mask] = 0.0  # hole (black background)

        # Convert to torch tensor
        phantom = torch.from_numpy(phantom).float()

        # Step 4: Assign RGB colors
        RGB = torch.zeros(res, res, res, 3) 
        shape_cube = torch.where(phantom == 1.0)
        RGB[shape_cube] = torch.Tensor([1, 0, 0.2])  # gray main cube

        shape_prism = torch.where(phantom == 0.8)
        RGB[shape_prism] = torch.Tensor([0.0, 1.0, 0.0])  # green square prism
 

        shape_hole = torch.where(phantom == 0)
        RGB[shape_hole] = torch.Tensor([0.0, 0.0, 0.0])  # black hole/background

        # Apply scaling 
        phantom *= scale
    elif type == 'fake_palace_air':
        phantom = np.zeros((res, res, res))
        center = res // 2
        cube_size = res // 4

        # Step 1: Main cube
        x_min, x_max = center - cube_size, center + cube_size
        y_min, y_max = center - cube_size, center + cube_size
        z_min, z_max = center - cube_size, center + cube_size
        phantom[x_min:x_max, y_min:y_max, z_min:z_max] = 1.0

        # Step 2: Second square prism on top
        small_cube_size = cube_size // 2
        shift = cube_size
        X, Y, Z = np.meshgrid(np.arange(res), np.arange(res), np.arange(res), indexing='ij')
        xc_shift, yc_shift, zc_shift = center, center, center + shift
        
        
        prism_mask = (
            (np.abs(X - xc_shift) < small_cube_size) &
            (np.abs(Y - yc_shift) < small_cube_size) &
            (np.abs(Z - zc_shift) < small_cube_size)
        )
        phantom[prism_mask] = 0.8

        # Step 3: Rotated smaller cube hole
        small_cube_size = cube_size // 2 - 5
        Xc, Yc, Zc = X - center, Y - center, Z - center

        angle_z = np.deg2rad(30)
        angle_x = np.deg2rad(20)
        angle_y = np.deg2rad(15)  # Add y-axis rotation

        cos_z, sin_z = np.cos(angle_z), np.sin(angle_z)
        cos_x, sin_x = np.cos(angle_x), np.sin(angle_x)
        cos_y, sin_y = np.cos(angle_y), np.sin(angle_y)

        Rz = np.array([
            [cos_z, -sin_z, 0],
            [sin_z,  cos_z, 0],
            [0,      0,     1]
        ])

        Rx = np.array([
            [1, 0,      0],
            [0, cos_x, -sin_x],
            [0, sin_x,  cos_x]
        ])

        Ry = np.array([
            [cos_y,  0, sin_y],
            [0,      1, 0],
            [-sin_y, 0, cos_y]
        ])

        # Combined rotation (apply in order: z, y, x)
        R = Rz @ Ry @ Rx

        # Rotate the coordinates
        coords = np.stack([Xc.flatten(), Yc.flatten(), Zc.flatten()], axis=0)
        rotated_coords = R @ coords
        Xr, Yr, Zr = rotated_coords.reshape(3, res, res, res)

        xc_shift, yc_shift, zc_shift = 0, 0, 0
        rotated_cube_mask = (
            (np.abs(Xr - xc_shift) < small_cube_size) &
            (np.abs(Yr - yc_shift) < small_cube_size) &
            (np.abs(Zr - zc_shift) < small_cube_size)
        )

                

        # Step 4: Fill lower density in zero regions
        phantom[phantom == 0.0] = 0.1

        # Convert to torch tensor
        phantom = torch.from_numpy(phantom).float()

        # Step 5: Assign RGB colors
        RGB = torch.zeros(res, res, res, 3)
        shape_cube = torch.where(phantom == 1.0)
        RGB[shape_cube] = torch.Tensor([1, 0, 0.2])  # gray main cube

        shape_prism = torch.where(phantom == 0.8)
        RGB[shape_prism] = torch.Tensor([0.0, 1.0, 0.0])  # green square prism

        shape_low_density = torch.where(phantom == 0.1)
        RGB[shape_low_density] = torch.Tensor([0, .2, 1])  # dark gray for lower density

        phantom[rotated_cube_mask] = 0.0  # hole
        # Apply scaling 
        phantom *= scale
    else:
        phantom = np.zeros((res, res, res))
        center = res // 2
        cube_size = res // 4  # size of the main cube

        # Step 1: Create main cube (density 1.0)
        x_min, x_max = center - cube_size, center + cube_size
        y_min, y_max = center - cube_size, center + cube_size
        z_min, z_max = center - cube_size, center + cube_size
        phantom[x_min:x_max, y_min:y_max, z_min:z_max] = 1.0

        # Step 2: Create rotated and shifted second cube (hole with zero density)
        small_cube_size = cube_size // 1.5 -15
        shift = cube_size // 2  # shift to make it intersect

        # Create 3D grid
        X, Y, Z = np.meshgrid(np.arange(res), np.arange(res), np.arange(res), indexing='ij')
        Xc, Yc, Zc = X - center, Y - center, Z - center  # center coordinates

        # Apply rotation matrix (rotation around z-axis + slight around x-axis)
        angle_z = np.deg2rad(30)  # 30 degrees
        angle_x = np.deg2rad(20)  # 20 degrees

        cos_z, sin_z = np.cos(angle_z), np.sin(angle_z)
        cos_x, sin_x = np.cos(angle_x), np.sin(angle_x)

        # Rotation matrices
        Rz = np.array([
            [cos_z, -sin_z, 0],
            [sin_z,  cos_z, 0],
            [0,      0,     1]
        ])

        Rx = np.array([
            [1, 0,      0],
            [0, cos_x, -sin_x],
            [0, sin_x,  cos_x]
        ])

        # Combined rotation
        R = Rz @ Rx

        # Flatten grid for rotation
        coords = np.stack([Xc.flatten(), Yc.flatten(), Zc.flatten()], axis=0)
        rotated_coords = R @ coords
        Xr, Yr, Zr = rotated_coords.reshape(3, res, res, res)

        # Define rotated & shifted cube mask
        xc_shift, yc_shift, zc_shift = -shift, shift // 2, shift // 3
        rotated_cube_mask = (
            (np.abs(Xr - xc_shift) < small_cube_size) &
            (np.abs(Yr - yc_shift) < small_cube_size) &
            (np.abs(Zr - zc_shift) < small_cube_size)
        )

        phantom[rotated_cube_mask] = 0.0  # hole (zero density)

        # Step 3: Add ellipse (density 0.8), shifted
        a, b, c = cube_size, cube_size * 0.8, cube_size * 0.6
        ellipse_center = (center + cube_size // 2, center, center)
        eq = ((X - ellipse_center[0]) ** 2) / a ** 2 + ((Y - ellipse_center[1]) ** 2) / b ** 2 + ((Z - ellipse_center[2]) ** 2) / c ** 2
        ellipse_mask = eq <= 1

        # Assign ellipse density only where not zero (so hole remains zero)
        phantom[ellipse_mask  != 0] = 0.8

        # Convert to torch tensor
        phantom = torch.from_numpy(phantom).float()

        # Step 4: Assign RGB colors
        RGB = torch.zeros(res, res, res, 3)
        shape_cube = torch.where(phantom == 1.0)
        RGB[shape_cube] = torch.Tensor([0.6, 0.6, 0.6])  # gray main cube

        shape_ellipse = torch.where(phantom == 0.8)
        RGB[shape_ellipse] = torch.Tensor([0.0, 1.0, 0.0])  # green ellipse

        shape_hole = torch.where(phantom == 0)
        RGB[shape_hole] = torch.Tensor([0.0, 0.0, 0.0])  # black hole/background

        phantom *= scale

    return phantom, RGB




def get_3Dphantom_RGB(res = 128,scale = 1e0):
    phantom , T1, T2 = shepp_logan((res, res, res), MR=True)
    phantom = torch.from_numpy(phantom)
    
    values = torch.unique(phantom) #[0.0000, 0.0012, 0.0062, 0.0074, 0.0080, 0.0082, 0.0085, 0.0093, 0.0095,0.0098, 0.0118]
    
    RGB = torch.zeros(res,res,res,3)
    shape_one = torch.where(phantom == .12 )
    shape_two = torch.where((phantom > 0.6) & (phantom < 0.73))
    shape_three = torch.where((phantom > 0.73) & (phantom < 0.88))
    shape_four = torch.where((phantom > 0.89) & (phantom < 0.99))
    shape_five = torch.where(phantom > 1)
    
    RGB[shape_one] = torch.Tensor([.75,0,0])
    RGB[shape_two] = torch.Tensor([0,.9,0])
    RGB[shape_three] = torch.Tensor([0.5,0.2,1])
    RGB[shape_four] = torch.Tensor([.1,.4,0])
    RGB[shape_five] = torch.Tensor([0,0,1])
    
    phantom *= scale
    return phantom,RGB 

def pc_to_ply(grid,thresh,file_path,subset =100, up_thresh = None):

    if isinstance(grid, torch.Tensor):
        voxel_grid = grid.cpu().numpy()

    # Get the coordinates of all non-zero voxels
    if up_thresh :
        non_zero_indices = np.where((grid > thresh) & (grid < up_thresh)) 
    else: 
        non_zero_indices = np.where(grid > thresh)
    subset_size = len(non_zero_indices[0]) // subset  # 10% of the total points
    sampled_indices = np.random.choice(len(non_zero_indices[0]), subset_size, replace=False)


    x = non_zero_indices[0][sampled_indices]
    y = non_zero_indices[1][sampled_indices]
    z = non_zero_indices[2][sampled_indices]
        
    sampled_colors = grid[x,y,z]
    

    # Normalize colors
    colors_normalized = (sampled_colors - sampled_colors.min()) / (sampled_colors.max() - sampled_colors.min())

    # Apply Jet colormap
    jet_cmap = cm.get_cmap('jet')
    colors_normalized = colors_normalized.detach().cpu().numpy()
    colors_mapped = jet_cmap(colors_normalized)  # Extract RGB values and ignore alpha

    # Convert to [0, 255] range for PLY format
    colors_mapped = (colors_mapped * 255).astype(np.uint8)


    # Create header for PLY file
    header = f"""ply
                format ascii 1.0
                element vertex {x.shape[0]}
                property float x
                property float y
                property float z
                property uchar red
                property uchar green
                property uchar blue
                end_header
                """

    # Write to file
    with open(file_path, 'w') as f:
        f.write(header)
        for xi,yi,zi , c in zip(x,y,z, colors_mapped):
            f.write(f"{xi} {yi} {zi} {c[0]} {c[1]} {c[2]}\n")
#pass the 3d grid density and color 3d grid, save the ply with density above the thres 
def density_rgb_to_ply(density, rgb, thresh=0, outpath=".", up_thresh = None):
    mask = density > thresh  # Keep only points above threshold
    # Get the coordinates of all non-zero voxels
    if up_thresh :
        mask = (density > thresh) & (density < up_thresh)
    else: 
        mask = density > thresh
    # Get (Z, Y, X) coordinates
    z_idx, y_idx, x_idx = np.where(mask)
    coords = np.stack([x_idx, y_idx, z_idx], axis=1)  # Shape: (N, 3)

    # Get colors at valid points
    colors = rgb[mask]  # Shape: (N, 3), values in [0,1]
    colors = (255 * colors).astype(np.uint8)  # Convert to 0-255

    # Make sure output path exists
    #os.makedirs(outpath, exist_ok=True)
    file_path = os.path.join(outpath)

    # Create PLY header
    header = f"""ply
        format ascii 1.0
        element vertex {coords.shape[0]}
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        """

    # Write to PLY file
    with open(file_path, 'w') as f:
        f.write(header)
        for point, c in zip(coords, colors):
            f.write(f"{point[0]} {point[1]} {point[2]} {c[0]} {c[1]} {c[2]}\n")

    print(f"Saved {coords.shape[0]} points to {file_path}")

def pcRGB_to_ply(grid,thresh=1,file_path=None,subset =100):

    
    if isinstance(grid, torch.Tensor):
        grid = grid.cpu().numpy()

    # Get the coordinates of all non-zero voxels
    #non_zero_indices = np.where(grid > thresh)
    #subset_size = len(non_zero_indices[0]) // subset  # 10% of the total points
    #sampled_indices = np.random.choice(len(non_zero_indices[0]), subset_size, replace=False)
    """
    Save a 3D numpy array (n, n, n, 3) as a PLY point cloud file.
    
    Parameters:
    - voxel_grid (numpy.ndarray): The input array of shape (n, n, n, 3) with RGB values.
    - filename (str): The name of the output PLY file.
    """
    n_t = grid.shape[0]
    n_x = grid.shape[1]
    n_y = grid.shape[2]
    points = []
    colors = []

    # Iterate through the 3D grid to collect non-zero points and their colors
    for t in range(n_t):
        for x in range(n_x):
            for y in range(n_y):
                color = grid[t,x,y]
                if not np.all(color == 0):  # Skip points with RGB = (0, 0, 0)
                    points.append([t,x,y])
                    colors.append(color )  # Normalize RGB to [0, 1] for open3d

    # Convert to numpy arrays
    points = np.array(points)
    colors =(255* np.array(colors)).astype(np.uint8)

    # Create header for PLY file
    header = f"""ply
                format ascii 1.0
                element vertex {points.shape[0]}
                property float x
                property float y
                property float z
                property uchar red
                property uchar green
                property uchar blue
                end_header
                """

    # Write to file
    with open(file_path, 'w') as f:
        f.write(header)
        for point , c in zip(points, colors):
            f.write(f"{point[0]} {point[1]} {point[2]} {c[0]} {c[1]} {c[2]}\n")





def x_ray_transform_3d_old(volume, angles):
    """
    Perform a simple 3D X-ray transform (parallel beam) using PyTorch.
    Args:
        volume (torch.Tensor): 3D tensor of shape (D, H, W) representing the volume.
        angles (torch.Tensor): 1D tensor of projection angles in radians.
    Returns:
        torch.Tensor: Sinogram (projections) of shape (num_angles, H, W).
    """
    D, H, W = volume.shape
    num_angles = len(angles)
    sinogram = torch.zeros((num_angles, H, W), device=volume.device)

    # Rotation matrix and affine grid generation
    for i, angle in enumerate(angles):
        rotation_matrix = torch.tensor([
            [torch.cos(angle), -torch.sin(angle), 0],
            [torch.sin(angle),  torch.cos(angle), 0]
        ], device=volume.device)

        grid = F.affine_grid(rotation_matrix.unsqueeze(0), volume.unsqueeze(0).size(), align_corners=False)
        rotated_volume = F.grid_sample(volume.unsqueeze(0), grid, align_corners=False)
        
        # Sum along one axis to simulate line integrals (projection)
        sinogram[i] = rotated_volume.sum(dim=1).squeeze(0)

    return sinogram



def x_ray_transform_3d(volume, angles_x,angles_y):
    """
    Perform a simple 3D X-ray transform (parallel beam) using PyTorch.
    Args:
        volume (torch.Tensor): 3D tensor of shape (D, H, W) representing the volume.
        angles (torch.Tensor): 1D tensor of projection angles in radians.
    Returns:
        torch.Tensor: Sinogram (projections) of shape (num_angles, H, W).
    """
    D, H, W = volume.shape
    
    num_angles_x = len(angles_x)
    num_angles_y = len(angles_y)
    sinogram = torch.zeros((num_angles_x*num_angles_y, H, W), device=volume.device)

    it = 0 
    # Rotation matrix and affine grid generation
    for i, angle_y in enumerate(angles_y):
        for j , angle_x in enumerate(angles_x):
            rotated_volume = rotate(volume, angle_x,angle_y) 
            # Sum along one axis to simulate line integrals (projection)
            sinogram[it] = rotated_volume.sum(dim=1).squeeze(0)
            it += 1

    return sinogram





def VR_transform_3d(volume,RGB, angles_x,angles_y):
    """
    Perform a simple 3D X-ray transform (parallel beam) using PyTorch.
    Args:
        volume (torch.Tensor): 3D tensor of shape (D, H, W) representing the volume.
        angles (torch.Tensor): 1D tensor of projection angles in radians.
    Returns:
        torch.Tensor: Sinogram (projections) of shape (num_angles, H, W).
    """
    D, H, W = volume.shape
    
    num_angles_x = len(angles_x)
    num_angles_y = len(angles_y)
    sinogram = torch.zeros((num_angles_x*num_angles_y, H, W,3), device=volume.device)
#    T = torch.zeros((num_angles_x*num_angles_y, D, H, W, 1), device=volume.device)
    T = torch.zeros(( D, H, W, 1), device=volume.device)

    it = 0 
    # Rotation matrix and affine grid generation
    for i, angle_y in enumerate(angles_y):
        for j , angle_x in enumerate(angles_x):
            rotated_volume = rotate(volume, angle_x,angle_y).unsqueeze(-1)
            rotated_RGB = rotate_RGB(RGB, angle_x,angle_y)
            # Sum along one axis to simulate line integrals (projection)
            T = torch.exp(-torch.cumsum(rotated_volume, dim=1)) 
            sinogram[it] = torch.sum(rotated_volume * rotated_RGB* T , dim=1)
            it += 1
    return sinogram# , rotated_RGB, T
 

def get_rotation_matrix(camera_center, camera_direction, world_up=torch.tensor([0.0, 0.0, 1.0])):
    """
    Computes camera-to-world rotation matrices from center and forward direction.

    Args:
        camera_center: (N, 3) tensor, camera positions
        camera_direction: (N, 3) tensor, forward direction the camera is pointing (not necessarily toward origin)
        world_up: (3,) world up direction (default: z-up)

    Returns:
        R: (N, 3, 3) rotation matrices from camera to world coordinates
    """
    device = camera_center.device
    N = camera_center.shape[0]

    # Z-axis: forward direction (pointing away from camera)
    z = camera_direction / torch.norm(camera_direction, dim=1, keepdim=True)

    # Handle edge case when view_dir || up
    up = world_up.to(device).expand(N, 3)
    colinear = torch.abs((z * up).sum(dim=1)) > 0.99
    if colinear.any():
        alt_up = torch.tensor([0.0, 1.0, 0.0], device=device)
        up[colinear] = alt_up

    # X-axis: right
    x = torch.cross(up, z, dim=1)
    x = x / torch.norm(x, dim=1, keepdim=True)

    # Y-axis: up
    y = torch.cross(z, x, dim=1)

    # Rotation matrix: [x, y, z]
    R = torch.stack([x, y, z], dim=2)  # (N, 3, 3)
    return R



def get_camera_transform(cam_pos, cam_dir, up=torch.tensor([0, 1, 0], dtype=torch.float32)):
    z = cam_dir / torch.norm(cam_dir)           # forward
    x = torch.cross(z, up)
    x = x / torch.norm(x)                        # right
    y = torch.cross(x, z)                        # real up

    # Camera-to-world rotation (columns are right, up, -forward)
    R = torch.stack([x, y, -z], dim=1)  # shape (3, 3)

    # Build 4x4 transform matrix
    T = torch.eye(4)
    T[:3, :3] = R
    T[:3, 3] = cam_pos
    return T

def write_transform_json(camera_transforms, fov_x, fov_y, image_res, out_path='/volume/VR_res/transforms.json'):
    fl_x = 0.5 * image_res[0] / torch.tan(fov_x / 2)
    fl_y = 0.5 * image_res[1] / torch.tan(fov_y / 2)

    out = {
        "camera_angle_x": float(fov_x),
        "camera_angle_y": float(fov_y),
        "fl_x": float(fl_x),
        "fl_y": float(fl_y),
        "cx": image_res[0] / 2,
        "cy": image_res[1] / 2,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "w": image_res[0],
        "h": image_res[1],
        "aabb_scale": 4,
        "frames": []
    }

    for i, T in enumerate(camera_transforms):
        out['frames'].append({
            "file_path": f"images/image_{i:03d}.png",
            "transform_matrix": T.tolist()
        })

    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

def quat_tvec_to_matrix4x4(quat: torch.Tensor, tvec: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion + translation to 4x4 world-to-camera transformation matrix.

    Args:
        quat: (N, 4) tensor of quaternions (w, x, y, z) in Hamilton convention.
        tvec: (N, 3) tensor of translations (TX, TY, TZ).

    Returns:
        mat4x4: (N, 4, 4) homogeneous transformation matrices.
    """
    assert quat.shape[0] == tvec.shape[0]
    device = quat.device
    N = quat.shape[0]

    # Convert to SciPy format: (x, y, z, w)
    quat_np = quat[:, [1, 2, 3, 0]].cpu().numpy()  # (N, 4)

    # Use scipy to get rotation matrices
    Rs = R.from_quat(quat_np).as_matrix()  # (N, 3, 3)

    # Build the 4x4 matrices
    mats = torch.eye(4, device=device).unsqueeze(0).repeat(N, 1, 1)  # (N, 4, 4)
    mats[:, :3, :3] = torch.tensor(Rs, dtype=torch.float32, device=device)
    mats[:, :3, 3] = tvec

    return mats

def normalize(v):
    return v / (v.norm(dim=-1, keepdim=True) + 1e-8)

def rotation_matrix_to_quaternion(R):
    """
    Convert a batch of rotation matrices (N x 3 x 3) to quaternions (N x 4)
    Quaternion format: (w, x, y, z)
    """
    batch_size = R.shape[0]
    q = torch.zeros(batch_size, 4, device=R.device)

    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    # Compute w
    q[:, 0] = torch.sqrt(torch.clamp(1 + trace, min=0)) / 2

    # Compute x, y, z
    q[:, 1] = (R[:, 2, 1] - R[:, 1, 2]) / (4 * q[:, 0] + 1e-8)
    q[:, 2] = (R[:, 0, 2] - R[:, 2, 0]) / (4 * q[:, 0] + 1e-8)
    q[:, 3] = (R[:, 1, 0] - R[:, 0, 1]) / (4 * q[:, 0] + 1e-8)

    return q

def dir_to_transform(camera_center, target, up=torch.tensor([0.,1.,  0.], device='cuda')):
    """
    Build a COLMAP-style camera-to-world transform matrix.

    Parameters:
    - camera_center: (N, 3) camera positions
    - target: (N, 3) points the camera is looking at
    - up: (3,) up direction (default: [0, 1, 0])

    Returns:
    - (N, 4, 4) camera-to-world transformation matrices
    """
    forward = F.normalize(target - camera_center, dim=-1)  # Camera looks toward -Z

    if up.ndim == 1:
        up = up.unsqueeze(0).expand_as(forward)

    right = F.normalize(torch.cross(up, forward, dim=-1), dim=-1)
    up_corrected = torch.cross(forward, right, dim=-1)

    R = torch.stack([right, up_corrected, -forward], dim=-1)  # (N, 3, 3)
    RT = torch.cat([R, camera_center.unsqueeze(-1)], dim=-1)  # (N, 3, 4)

    bottom = torch.tensor([0, 0, 0, 1], dtype=R.dtype, device=R.device).expand(R.shape[0], 1, 4)
    return torch.cat([RT, bottom], dim=1)  # (N, 4, 4)


def random_angles_on_sphere(n, device='cuda'):
    theta = torch.rand(n, device=device) * 2 * torch.pi  # θ ∈ [0, 2π) 
    phi = torch.rand(n, device=device) * 2 * torch.pi                  
    return torch.stack((theta, phi), dim=1)  

def perturb_directions(ray_dirs_cam, epsilon: float = 0.01):
    # ray_dirs_cam: (N, 3), already normalized
    noise = torch.randn_like(ray_dirs_cam) * epsilon
    perturbed = ray_dirs_cam + noise
    perturbed = perturbed / torch.norm(perturbed, dim=-1, keepdim=True) 
    return perturbed


def qvec2rotmat(qvec):
    """
    Convert quaternion(s) to rotation matrix/matrices.

    Args:
        qvec: Tensor of shape (..., 4), where last dimension is (x, y, z, w)
    
    Returns:
        Tensor of shape (..., 3, 3) with rotation matrices.
    """
    w, x, y, z= qvec.unbind(-1)

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    # Rotation matrix elements (broadcasted)
    row0 = torch.stack([1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)], dim=-1)
    row1 = torch.stack([2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)], dim=-1)
    row2 = torch.stack([2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)], dim=-1)

    rot_mats = torch.stack([row0, row1, row2], dim=-2)  # (..., 3, 3)

    return rot_mats


def rotmat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Computes rotation matrices that rotate vectors `a` to vectors `b`.
    Both `a` and `b` must be of shape (N, 3). Returns a tensor of shape (N, 3, 3).
    """
    a = a / a.norm(dim=1, keepdim=True)
    b = b / b.norm(dim=1, keepdim=True)

    v = torch.cross(a, b, dim=1)
    c = (a * b).sum(dim=1, keepdim=True)  # (N, 1)
    s = v.norm(dim=1, keepdim=True)      # (N, 1)

    # Handle edge case: vectors in opposite direction
    mask = (c < -1 + 1e-10).squeeze()

    if mask.any():
        eps = torch.randn_like(a[mask]) * 1e-2
        return rotmat_torch(a[mask] + eps, b[mask])

    # Skew-symmetric cross-product matrix for each vector v
    zero = torch.zeros_like(v[:, 0])
    kmat = torch.stack([
        torch.stack([zero, -v[:, 2],  v[:, 1]], dim=1),
        torch.stack([v[:, 2],  zero, -v[:, 0]], dim=1),
        torch.stack([-v[:, 1], v[:, 0],  zero], dim=1),
    ], dim=1)  # (N, 3, 3)

    eye = torch.eye(3, device=a.device, dtype=a.dtype).expand(a.shape[0], 3, 3)
    outer_k = torch.bmm(kmat, kmat)
    denom = (s ** 2 + 1e-10).view(-1, 1, 1)
    R = eye + kmat + outer_k * ((1 - c).view(-1, 1, 1) / denom)

    return R



def create_transform(dirs, tvec, device='cuda', apply_axis_adjustments=True):
    """
    Creates a camera-to-world transformation matrix.

    Args:
        qvec: Tensor of shape (N, 4), quaternion (x, y, z, w)
        tvec: Tensor of shape (N, 3), translation
        device: torch device
        apply_axis_adjustments: whether to flip/swap axes for coordinate convention

    Returns:
        c2w: (N, 4, 4) camera-to-world transformation matrices
    """
    N = dirs.shape[0]
    b = torch.tensor([0., 0., 1.], device=dirs.device).expand(N, 3)
    R = rotmat(dirs,b )
    t = tvec.view(N, 3, 1)

    Rt = torch.cat([R, t], dim=2)  # (N, 3, 4)
    bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).view(1, 1, 4).expand(N, -1, -1)
    m = torch.cat([Rt, bottom], dim=1)  # (N, 4, 4)
    
    c2w = torch.linalg.inv(m)

    if apply_axis_adjustments:
        # Optional flipping and axis convention adjustments
        c2w[:, 0:3, 2] *= -1  # flip z axis column
        c2w[:, 0:3, 1] *= -1  # flip y axis column
        c2w = c2w[:, [1, 0, 2, 3], :]  # swap x and y rows
        c2w[:, 2, :] *= -1  # flip entire z row (world up/down)
    
    return c2w

    

def angles_to_quaternion(theta_dirs, reference=torch.tensor([0.0, 0.0, 1.0], device='cuda')):
    """
    Converts azimuth and elevation angles to quaternions.
    
    Args:
        theta_dirs: (N, 2) tensor of angles [theta_x, theta_y]
        reference: (3,) reference vector (default: [0,0,1])
        
    Returns:
        quaternions: (N, 4) in (x, y, z, w) format
    """
    theta_x = theta_dirs[:, 0]
    theta_y = theta_dirs[:, 1]

    # Convert angles to direction vector
    cam_dirs = torch.stack([
        torch.cos(theta_y) * torch.cos(theta_x),
        torch.cos(theta_y) * torch.sin(theta_x),
        torch.sin(theta_y)
    ], dim=1)  # (N, 3)

    # Normalize input vectors (should already be normalized, but safe)
    cam_dirs = F.normalize(cam_dirs, dim=-1)
    ref = F.normalize(reference, dim=0).expand_as(cam_dirs)

    # Quaternion from two directions (ref to cam_dirs)
    dot = (ref * cam_dirs).sum(dim=-1, keepdim=True)  # (N, 1)
    axis = torch.cross(ref, cam_dirs, dim=-1)
    axis_norm = torch.norm(axis, dim=-1, keepdim=True)

    # Handle near-parallel edge case
    near_parallel = axis_norm < 1e-6
    axis[near_parallel.squeeze()] = torch.tensor([1.0, 0.0, 0.0], device=cam_dirs.device)

    axis = axis / (axis_norm + 1e-8)
    angle = torch.acos(torch.clamp(dot, -1.0, 1.0))  # (N, 1)

    sin_half_angle = torch.sin(angle / 2)
    q_xyz = axis * sin_half_angle
    q_w = torch.cos(angle / 2)

    return torch.cat([q_xyz, q_w], dim=-1)  # (N, 4)
    
    

def build_cam2world_matrices(cam_centers: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """
    Compute camera-to-world matrices without a predefined up vector.
    Uses a dynamic up vector orthogonal to the view direction.

    Args:
        cam_centers: (N, 3) tensor of camera positions
        directions: (N, 3) tensor of camera directions (pointing into the scene)

    Returns:
        cam2world: (N, 4, 4) tensor of camera-to-world matrices
    """
    N = cam_centers.shape[0]
    device = cam_centers.device

    forward = F.normalize(directions, dim=-1)  # -Z axis in world space
    world_up = torch.tensor([0.0, 1.0, 0.0], device=device).expand(N, 3)

    # Project world_up onto the plane orthogonal to forward
    dot = (world_up * forward).sum(dim=-1, keepdim=True)  # shape (N, 1)
    up_proj = F.normalize(world_up - dot * forward, dim=-1)  # shape (N, 3)
 
    right = torch.cross(forward, up_proj)  # Y axis (already orthogonal)

    # Compose rotation matrix
    rot = torch.stack([right, up_proj, forward], dim=-1)  # shape (N, 3, 3)
    trans = cam_centers.unsqueeze(-1)                     # shape (N, 3, 1)

    cam2world = torch.cat([rot, trans], dim=-1)           # shape (N, 3, 4)
    bottom = torch.tensor([0, 0, 0, 1], device=device).view(1, 1, 4).expand(N, -1, -1)
    cam2world = torch.cat([cam2world, bottom], dim=1)     # shape (N, 4, 4)

    return cam2world



def get_pose_dir(pose, scale=0.33, offset=(0, 0, 0) ):
    """
    pose: (N, 3, 3) or (N, 3, 4) camera pose matrix
          assumed to be in the format [right, up, back, T] as columns (COLMAP/NeRF style)
    
    Returns:
        new_pose: (N, 4, 4)
        dir: (N, 3) — normalized direction vectors
    """
    N = pose.shape[0]

    if pose.shape[-1] == 3:  # (N, 3, 3) → add dummy translation
        T = torch.zeros(N, 3, 1, device=pose.device, dtype=pose.dtype)
        pose = torch.cat([pose, T], dim=-1)  # (N, 3, 4)

    # Apply reordering and sign flips
    new_pose = torch.zeros((N, 4, 4), device=pose.device, dtype=pose.dtype)
    new_pose[:, 0, :3] =  pose[:, 1, :3].clone().detach()
    new_pose[:, 1, :3] =  pose[:, 2, :3].clone().detach()
    new_pose[:, 2, :3] =  pose[:, 0, :3].clone().detach()
    new_pose[:, 3, 3] = 1.0

    # Apply sign flip and translation scale/offset
    new_pose[:, 1, :3] *= -1
    new_pose[:, 2, :3] *= -1

    new_pose[:, 0, 3] = pose[:, 1, 3].clone() * scale + offset[0]
    new_pose[:, 1, 3] = pose[:, 2, 3].clone() * scale + offset[1]
    new_pose[:, 2, 3] = pose[:, 0, 3].clone() * scale + offset[2]
 

    return new_pose 



def get_pose_dir_inv(pose, scale=0.33, offset=(0, 0, 0) ):
    """
    pose: (N, 3, 3) or (N, 3, 4) camera pose matrix
          assumed to be in the format [right, up, back, T] as columns (COLMAP/NeRF style)
    
    Returns:
        new_pose: (N, 4, 4)
        dir: (N, 3) — normalized direction vectors
    """
    N = pose.shape[0]

    if pose.shape[-1] == 3:  # (N, 3, 3) → add dummy translation
        T = torch.zeros(N, 3, 1, device=pose.device, dtype=pose.dtype)
        pose = torch.cat([pose, T], dim=-1)  # (N, 3, 4)

    new_pose = torch.zeros((N, 4, 4), device=pose.device, dtype=pose.dtype)

    new_pose[:, 0, 3] = pose[:, 2, 3].clone() / scale - offset[0]
    new_pose[:, 1, 3] = pose[:, 0, 3].clone() / scale - offset[1]
    new_pose[:, 2, 3] = pose[:, 1, 3].clone() / scale - offset[2]
    
        # Apply sign flip and translation scale/offset
    new_pose[:, 1, :3] *= -1
    new_pose[:, 0, :3] *= -1
    
    
    # Apply reordering and sign flips
    new_pose = torch.zeros((N, 4, 4), device=pose.device, dtype=pose.dtype)
    new_pose[:, 0, :3] =  pose[:, 2, :3].clone().detach()
    new_pose[:, 1, :3] =  pose[:, 0, :3].clone().detach()
    new_pose[:, 2, :3] =  pose[:, 1, :3].clone().detach()
    new_pose[:, 3, 3] = 1.0




 

    return new_pose 



def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')
    
    

def DP_fan_camera_poses(FOV_x, FOV_y, S_dist, focal_lenght=0.4, res=128, n_theta=[180, 1], rand_angle = None,
                        thetas_x=None, thetas_y=None, rand_S=None, rand_dir=None, device='cuda' 
                        , output_json_path = '/volume/torch-ngp_NODIR/VR_res/house/transforms_all.json', perturb_dir = True, scale=.33):

    # --- Convert FOVs to radians ---
    
    FOV_x = torch.deg2rad(torch.tensor(FOV_x, device=device))
    FOV_y = torch.deg2rad(torch.tensor(FOV_y, device=device))

    # --- Pose angles ---
    if n_theta[0] > 1:
        thetas_x = torch.linspace(0, 2 * torch.pi, n_theta[0], device=device)[:-1]
    else:
        thetas_x = torch.tensor([0.0], device=device)

    if n_theta[1] > 1:
        thetas_y = torch.linspace(0, 2 * torch.pi, n_theta[1] + 1, device=device)[:-1]
    else:
        thetas_y = torch.tensor([0.0], device=device)

    if rand_angle is not None:
        thetas = random_angles_on_sphere(rand_angle)
        theta_x_flat = thetas[:,0]
        theta_y_flat = thetas[:,1]
        n_poses = thetas.shape[0]
    else:
        thetas_x_grid, thetas_y_grid = torch.meshgrid(thetas_x, thetas_y, indexing='ij')
        n_poses = thetas_x_grid.numel()

        theta_x_flat = thetas_x_grid.reshape(-1)
        theta_y_flat = thetas_y_grid.reshape(-1)

    # --- Camera center directions ---
    camera_centers = torch.stack([
        torch.cos(theta_y_flat) * torch.cos(theta_x_flat),
        torch.cos(theta_y_flat) * torch.sin(theta_x_flat),
        torch.sin(theta_y_flat)
    ], dim=1)  # shape (N, 3)
    
    if rand_S is not None:
        S_dist = torch.empty(n_poses, device=device).uniform_(S_dist * rand_S[0], S_dist * rand_S[1])
    else:
        S_dist = torch.full((n_poses,), float(S_dist), device=device)
        
    
    cam_dirs = -1 * camera_centers.clone() 
    
    if perturb_dir:
        cam_dirs = perturb_directions( cam_dirs , epsilon=0.1) 
 
 
    camera_centers =  camera_centers * S_dist.unsqueeze(-1)     

    # c2w = build_cam2world_matrices(camera_centers, cam_dirs)
    
    c2w = dir_to_transform( camera_centers.clone().detach(), camera_centers.clone() +cam_dirs.clone().detach())
    c2w_clone = c2w.clone()
    '''
    R = qvec2rotmat(qvec) 
    bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device)
    Rt = torch.cat([R, camera_centers.unsqueeze(-1)], dim=2)  # (N, 3, 4)
 
    # bottom row: (N, 1, 4) filled with [0, 0, 0, 1]
    bottom = torch.tensor([0., 0., 0., 1.], device=R.device).view(1, 1, 4).expand(R.shape[0], -1, -1)

    # final result: (N, 4, 4)
    m = torch.cat([Rt, bottom], dim=1)
 
    c2w = torch.linalg.inv(m) 
 
    R = c2w[:,:3,:3].clone() 
    
    '''
    c2w[:,0:3, 2] *= -1 # flip the y and z axis
    c2w[:,0:3, 1] *= -1
    c2w = c2w[:,[1, 0, 2, 3],:] # swap y and z
    c2w[:,2, :] *= -1 
    '''
    N = c2w.shape[0]
    avglen = 0.0
    for entry in c2w:  # assuming tensor.shape = (N, 4, 4)
        avglen += entry[0:3, 3].norm()
    avglen /= N
 
    
    '''
    #c2w[:,0:3,3] *= 4.0 / avglen 
    
    #c2w = quat_tvec_to_matrix4x4(qvec, camera_centers)
     
    #c2w = create_transform(cam_dirs, -1*camera_centers)   
    # --- Fan-beam pixel directions in camera frame ---
    #focal_lenght *= scale
    fl_x = 0.5 * res / torch.tan(FOV_x / 2) 
    fl_y = 0.5 * res / torch.tan(FOV_y / 2) 
    intrinsics = fl_x, fl_y, res/2, res/2
    rays = get_rays(c2w,intrinsics,res,res)
    
    rays_o = rays['rays_o'] 
    ray_dirs = rays['rays_d']
     
    half_width = focal_lenght * torch.tan(FOV_x / 2)
    half_height = focal_lenght * torch.tan(FOV_y / 2)

    '''

    xs = torch.linspace(-half_width, half_width, res, device=device)
    ys = torch.linspace(-half_height, half_height, res, device=device)
    x_grid, y_grid = torch.meshgrid(xs, ys, indexing='ij')
    
    z_grid = torch.full_like(x_grid, 1.0)
    
    ray_dirs_cam = torch.stack([x_grid, y_grid, z_grid], dim=-1)  # (res, res, 3)
    
    ray_dirs_flat = ray_dirs_cam.reshape(-1, 3)  # (M, 3)
    ray_dirs_flat = ray_dirs_flat / ray_dirs_flat.norm(dim=-1, keepdim=True)  # normalize

    n_pixels = ray_dirs_flat.shape[0]

    # --- Compute rotation matrices ---
    #R = get_rotation_matrix(camera_center=camera_center, camera_direction= cam_dirs)  # (N, 3, 3) 
    ray_dirs_expanded = ray_dirs_flat.unsqueeze(0).expand(n_poses, n_pixels, 3).unsqueeze(-1)  # (N, M, 3, 1)

    #R = c2w_orig[:, :3, :3] 
    #R[:,:,2] *= -1
    #c2w[:,:,2] *= -1
    #R = R[: , [1,2,0],: ]
    
    #R[:, 0:3, 0] *= -1  # negate second column
    #R[:, 0:3, 2] *= -1    
    #R_expanded = R.unsqueeze(1)  # (180, 1, 3, 3)
    ray_dirs_expanded = ray_dirs_flat.unsqueeze(0).expand(n_poses, -1, -1)  # (N, M, 3)
 
    # Apply rotation: (N, M, 3)
    RT = c2w.clone().detach() 
      
    RT = RT[:,[1,2,0],:]
    RT[:,:,1] *= -1 
    RT[:,:,2] *= -1 
    
     
    RT[:,:,3] *= scale
    RT[:,:,2] *= -1 
    R = RT[:,:3,:3]
    #R = R.transpose(-2, -1)
     
    T = RT[:,:3, 3].clone().detach() 
       
    R = c2w[:,:3,:3].clone()
    result = torch.matmul(R[:, None,:,:], ray_dirs[None, :, :, None])  # (N, M, 3, 1)
    
# Remove the last dimension to get shape (N, M, 3)
    ray_dirs_world = result.squeeze(-1)
    
    ray_dirs_world = F.normalize(ray_dirs_world, p=2, dim=-1, eps=1e-8)
# Repeat camera centers for all pixels 
    
    x = R[:,:,0]
    y = R[:,:,1]
    z = R[:,:,2]
    
    orientations = torch.stack([x,y,z],dim = 1)  
     

    # --- Repeat camera center ---
    camera_center_expanded = T[:, None, :].expand(-1, n_pixels, -1)  # (N, M, 3)
    #camera_center_expanded = camera_center_expanded[:, : ,[1, 2, 0]]
    #transforms = [get_camera_transform(pos, dir, up).cpu() for pos, dir in zip(camera_center, cam_dirs)]
    ''' 
    # Write JSON 
    fl_x = 0.5 * res / torch.tan(FOV_x / 2) 
    fl_y = 0.5 * res / torch.tan(FOV_y / 2)  
    angle_x = FOV_x
    angle_y = FOV_y
    metadata = {
        "camera_angle_x": float(angle_x),
        "camera_angle_y": float(angle_y),
        "fl_x": float(fl_x),
        "fl_y": float(fl_y),
        "cx": res / 2,
        "cy": res / 2,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "w": res,
        "h": res, 
        "frames": []
    }

    for i, T in enumerate(c2w_clone):
        metadata["frames"].append({
            "file_path": f"images/image_{i:03d}.png",
            "transform_matrix": T.tolist()
        })

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return rays_o, ray_dirs


def get_rays(poses, intrinsics, H, W, N=-1, error_map=None, patch_size=1):
    ''' get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    i, j = custom_meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device)) # float
    i = i.t().reshape([1, H*W]).expand([B, H*W]) + 0.5
    j = j.t().reshape([1, H*W]).expand([B, H*W]) + 0.5

    results = {}

    if N > 0:
        N = min(N, H*W)

        # if use patch-based sampling, ignore error_map
        if patch_size > 1:

            # random sample left-top cores.
            # NOTE: this impl will lead to less sampling on the image corner pixels... but I don't have other ideas.
            num_patch = N // (patch_size ** 2)
            inds_x = torch.randint(0, H - patch_size, size=[num_patch], device=device)
            inds_y = torch.randint(0, W - patch_size, size=[num_patch], device=device)
            inds = torch.stack([inds_x, inds_y], dim=-1) # [np, 2]

            # create meshgrid for each patch
            pi, pj = custom_meshgrid(torch.arange(patch_size, device=device), torch.arange(patch_size, device=device))
            offsets = torch.stack([pi.reshape(-1), pj.reshape(-1)], dim=-1) # [p^2, 2]

            inds = inds.unsqueeze(1) + offsets.unsqueeze(0) # [np, p^2, 2]
            inds = inds.view(-1, 2) # [N, 2]
            inds = inds[:, 0] * W + inds[:, 1] # [N], flatten

            inds = inds.expand([B, N])

        elif error_map is None:
            inds = torch.randint(0, H*W, size=[N], device=device) # may duplicate
            inds = inds.expand([B, N])
        else:

            # weighted sample on a low-reso grid
            inds_coarse = torch.multinomial(error_map.to(device), N, replacement=False) # [B, N], but in [0, 128*128)

            # map to the original resolution with random perturb.
            inds_x, inds_y = inds_coarse // 128, inds_coarse % 128 # `//` will throw a warning in torch 1.10... anyway.
            sx, sy = H / 128, W / 128
            inds_x = (inds_x * sx + torch.rand(B, N, device=device) * sx).long().clamp(max=H - 1)
            inds_y = (inds_y * sy + torch.rand(B, N, device=device) * sy).long().clamp(max=W - 1)
            inds = inds_x * W + inds_y

            results['inds_coarse'] = inds_coarse # need this when updating error_map

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results['inds'] = inds

    else:
        inds = torch.arange(H*W, device=device).expand([B, H*W])

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2) # (B, N, 3)

    rays_o = poses[..., :3, 3] # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d) # [B, N, 3]
    
    results['rays_o'] = rays_o
    results['rays_d'] = rays_d

    return results


def VR_fanbeam_transform_3d(volume,RGB, FOV_x, FOV_y, S_dist,n_theta, thetas=None ):
    """
    Perform a simple 3D X-ray transform (parallel beam) using PyTorch.
    Args:
        volume (torch.Tensor): 3D tensor of shape (D, H, W) representing the volume.Bolzaneto, Genoa, Metropolitan City of Genoa
        angles (torch.Tensor): 1D tensor of projection angles in radians.
    Returns:
        torch.Tensor: Sinogram (projections) of shape (num_angles, H, W).
    """
    D, H, W = volume.shape 
    camera_center, angles_x, angles_y = DP_fan_camera_poses(FOV_x=FOV_x,FOV_y=FOV_y,S_dist=S_dist,res=128, n_theta=n_theta)
    
    num_angles_x = len(angles_x)
    num_angles_y = len(angles_y)
    sinogram = torch.zeros((num_angles_x*num_angles_y, H, W,3), device=volume.device)
#    T = torch.zeros((num_angles_x*num_angles_y, D, H, W, 1), device=volume.device)
    T = torch.zeros(( D, H, W, 1), device=volume.device)

    it = 0 
    # Rotation matrix and affine grid generation
    for i, angle_y in enumerate(angles_y):
        for j , angle_x in enumerate(angles_x):
            rotated_volume = rotate(volume, angle_x,angle_y).unsqueeze(-1)
            rotated_RGB = rotate_RGB(RGB, angle_x,angle_y)
            # Sum along one axis to simulate line integrals (projection)
            T = torch.exp(-torch.cumsum(rotated_volume, dim=1)) 
            sinogram[it] = torch.sum(rotated_volume * rotated_RGB* T , dim=1)
            it += 1
    return sinogram# , rotated_RGB, T
 
def DP_fan_Volume_Rendering(sigma, RGB, S_pos=None, S_dist = None, S_dir=None, S_angles=None,  preserve_range=False,circle = True):
    """
    Calculates the radon transform of an image given specified
    projection angles.

    Parameters
    ----------
    image : array_like
        Input image. The rotation axis will be located in the pixel with
        indices ``(image.shape[0] // 2, image.shape[1] // 2)``.
    theta : array_like, optional
        Projection angles (in degrees). If `None`, the value is set to
        np.arange(180).
    circle : boolean, optional
        Assume image is zero outside the inscribed circle, making the
        width of each projection (the first dimension of the sinogram)
        equal to ``min(image.shape)``.
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of `img_as_float`.
        Also see https://scikit-image.org/docs/dev/user_guide/data_types.html

    Returns
    -------
    radon_image : ndarray
        Radon transform (sinogram).  The tomography rotation axis will lie
        at the pixel index ``radon_image.shape[0] // 2`` along the 0th
        dimension of ``radon_image``.

    References
    ----------
    .. [1] AC Kak, M Slaney, "Principles of Computerized Tomographic
           Imaging", IEEE Press 1988.
    .. [2] B.R. Ramesh, N. Srinivasa, K. Rajgopal, "An Algorithm for Computing
           the Discrete Radon Transform With Some Applications", Proceedings of
           the Fourth IEEE Region 10 International Conference, TENCON '89, 1989

    Notes
    -----
    Based on code of Justin K. Romberg
    (https://www.clear.rice.edu/elec431/projects96/DSP/bpanalysis.html)

    """ 
    if sigma.ndim ==2:
        sigma = convert_to_float(sigma, preserve_range)

        shape_min = min(sigma.shape)
        radius = shape_min // 2
        img_shape = np.array(sigma.shape)
        coords = np.array(np.ogrid[:sigma.shape[0], :sigma.shape[1]],
                            dtype=object)
        dist = ((coords - img_shape // 2) ** 2).sum(0)
        outside_reconstruction_circle = dist > radius ** 2
        if np.any(sigma[outside_reconstruction_circle]):
            warn('Radon transform: image must be zero outside the '
                    'reconstruction circle')
        # Crop image to make it square
        slices = tuple(slice(int(np.ceil(excess / 2)),
                                int(np.ceil(excess / 2) + shape_min))
                        if excess > 0 else slice(None)
                        for excess in (img_shape - shape_min))
        #padded_sigma = sigma[slices]
        
        
        
        if circle:
            shape_min = min(sigma.shape)
            radius = shape_min // 2
            img_shape = np.array(sigma.shape)
            coords = np.array(np.ogrid[:sigma.shape[0], :sigma.shape[1]],
                            dtype=object)
            dist = ((coords - img_shape // 2) ** 2).sum(0)
            outside_reconstruction_circle = dist > radius ** 2
            if np.any(sigma[outside_reconstruction_circle]):
                warn('Radon transform: image must be zero outside the '
                    'reconstruction circle')
            # Crop image to make it square
            slices = tuple(slice(int(np.ceil(excess / 2)),
                                int(np.ceil(excess / 2) + shape_min))
                        if excess > 0 else slice(None)
                        for excess in (img_shape - shape_min))
            padded_sigma = sigma[slices]
            padded_RGB = RGB[slices]
        else:
            diagonal = np.sqrt(2) * max(sigma.shape)
            pad = [int(np.ceil(diagonal - s)) for s in sigma.shape]
            new_center = [(s + p) // 2 for s, p in zip(sigma.shape, pad)]
            old_center = [s // 2 for s in sigma.shape]
            pad_before = [nc - oc for oc, nc in zip(old_center, new_center)]
            pad_width = [(pb, p - pb) for pb, p in zip(pad_before, pad)]
            pad_width_RGB = [(pb, p - pb) for pb, p in zip(pad_before, pad)] + [(0, 0)]
            padded_sigma = np.pad(sigma, pad_width, mode='constant',
                                constant_values=0)
            padded_RGB = np.pad(RGB, pad_width_RGB, mode='constant',
                            constant_values=0)

        #padded_RGB = RGB[slices]
        
        # padded_image is always square
        if padded_sigma.shape[0] != padded_sigma.shape[1]:
            raise ValueError('padded_image must be a square')
        center = padded_sigma.shape[0] // 2

        VR = np.zeros(((len(S_angles))* len(S_pos)  ,3), 
                    dtype=sigma.dtype)

        par_thetas = np.zeros(((len(S_angles))* len(S_pos)  ), 
                    dtype=sigma.dtype)
        par_s = np.zeros(((len(S_angles))* len(S_pos)  ), 
                    dtype=sigma.dtype)
        
        k = 0 
        #T_cum_thetas = np.zeros((len(theta),padded_sigma.shape[0],padded_sigma.shape[1]),  dtype=sigma.dtype)
        for i, angle_S in enumerate( S_pos ):
            for j, angle_r in enumerate( S_angles ):

                angle = angle_S + angle_r + S_dir[i]
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                R = np.array([[cos_a, sin_a, -center * (cos_a + sin_a - 1)],
                            [-sin_a, cos_a, -center * (cos_a - sin_a - 1)],
                            [0, 0, 1]])
                rotated_sigma = warp(padded_sigma, R, clip=False)

                s =  round( S_dist[i] * np.sin(angle_r + S_dir[i]) )  
                s_ind = s + rotated_sigma.shape[1]/2   
                if s_ind < 0 or s_ind> rotated_sigma.shape[1] -1:
                    s_ind = rotated_sigma.shape[1] -1 
                s_ind =  round(s_ind)
                rotated_sigma = rotated_sigma[:, s_ind ]
                rotated_RGB = warp(padded_RGB, R, clip=False)[:,s_ind]  
                T_cum = np.exp(-np.cumsum(rotated_sigma, axis=0))#np.roll(rotated,1,axis=0))
                integrand = (rotated_sigma*T_cum )[:, np.newaxis]* rotated_RGB

                
                #T_cum_thetas[i,:]=T_cum
                par_thetas[k] = angle
                par_s[k] = s
                VR[k] = integrand.sum(0) 

                k+=1
        
    return  VR , par_thetas, par_s 

def rotate(density_grid, theta_x, theta_y):
    # Rotation matrix around the x-axis 
    n  = density_grid.shape[0] 
    
    x_ = torch.linspace(-1, 1, n )
    y_ = torch.linspace(-1, 1, n )
    z_ = torch.linspace(-1, 1, n )

    # Create a meshgrid for the 3D space (Shape: (n, n, n, 3))
    grid = torch.stack(torch.meshgrid(x_, y_, z_), dim=-1) #*128
    R_x = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(theta_x), -torch.sin(theta_x)],
        [0, torch.sin(theta_x), torch.cos(theta_x)]
    ])

    # Rotation matrix around the y-axis
    R_y = torch.tensor([
        [torch.cos(theta_y), 0, torch.sin(theta_y)],
        [0, 1, 0],
        [-torch.sin(theta_y), 0, torch.cos(theta_y)]
    ])

    # Flatten the grid to shape (n*n*n, 3)
    grid_flat = grid.view(-1, 3)  # Shape: (n*n*n, 3)

    # Apply rotations to the grid (batch-wise)
    rotated_grid = torch.matmul(grid_flat, R_y.T)  # Apply y-axis rotation
    rotated_grid = torch.matmul(rotated_grid, R_x.T)  # Apply x-axis rotation

    # Round the coordinates of the rotated grid to the nearest integer to map them back to the original grid
    rotated_grid_rounded = torch.round((rotated_grid +1)* ((n-1)/2) ).long()

    # Ensure the coordinates are within bounds
    rotated_grid_rounded = torch.clamp(rotated_grid_rounded, 0, n-1 )

    # Now, we will map the rotated grid to the corresponding density values
    rotated_density = density_grid[rotated_grid_rounded[..., 0], rotated_grid_rounded[..., 1], rotated_grid_rounded[..., 2]]

    # Reshape the rotated_density back to the original shape of the grid
    rotated_density = rotated_density.view(n, n, n) 
    return rotated_density

def rotate_RGB(RGB_grid, theta_x, theta_y):
    # Rotation matrix around the x-axis 
    n  = RGB_grid.shape[0] 
    
    x_ = torch.linspace(-1, 1, n )
    y_ = torch.linspace(-1, 1, n )
    z_ = torch.linspace(-1, 1, n )

    # Create a meshgrid for the 3D space (Shape: (n, n, n, 3))
    grid = torch.stack(torch.meshgrid(x_, y_, z_), dim=-1) #*128
    R_x = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(theta_x), -torch.sin(theta_x)],
        [0, torch.sin(theta_x), torch.cos(theta_x)]
    ])

    # Rotation matrix around the y-axis
    R_y = torch.tensor([
        [torch.cos(theta_y), 0, torch.sin(theta_y)],
        [0, 1, 0],
        [-torch.sin(theta_y), 0, torch.cos(theta_y)]
    ])

    # Flatten the grid to shape (n*n*n, 3)
    grid_flat = grid.view(-1, 3)  # Shape: (n*n*n, 3)

    # Apply rotations to the grid (batch-wise)
    rotated_grid = torch.matmul(grid_flat, R_y.T)  # Apply y-axis rotation
    rotated_grid = torch.matmul(rotated_grid, R_x.T)  # Apply x-axis rotation

    # Round the coordinates of the rotated grid to the nearest integer to map them back to the original grid
    rotated_grid_rounded = torch.round((rotated_grid +1)* ((n-1)/2) ).long()

    # Ensure the coordinates are within bounds
    rotated_grid_rounded = torch.clamp(rotated_grid_rounded, 0, n-1 )

    # Now, we will map the rotated grid to the corresponding density values
    rotated_RGB = RGB_grid[rotated_grid_rounded[..., 0], rotated_grid_rounded[..., 1], rotated_grid_rounded[..., 2]]

    # Reshape the rotated_density back to the original shape of the grid
    rotated_RGB = rotated_RGB.view(n, n, n,3) 
    return rotated_RGB

# Assume sinogram is a NumPy array of shape (num_angles, height, width)
#sinogram = np.load('3d_sinogram.npy')  # Load your sinogram here
def x_ray_inverse(sinogram, angles):
        
    # Create a 3D projection geometry (parallel beam)
    geom = astra.create_proj_geom('parallel3d', angles, 128, 128)

    # Create a volume geometry based on the sinogram dimensions
    vol_geom = astra.create_vol_geom((128, 128, 128))  # Reconstructed volume shape

    # Create sinogram data structure in ASTRA
    sinogram_id = astra.data3d.create('-sino', sinogram)

    # Create a volume in ASTRA
    reconstruction_id = astra.data3d.create('-vol', vol_geom)

    # Configure and run the FBP algorithm
    cfg = astra.astra_dict('FBP3D_CUDA')
    cfg['ProjectionDataId'] = sinogram_id
    cfg['ReconstructionDataId'] = reconstruction_id

    astra.algorithm.run(cfg)

    # Retrieve the reconstructed volume
    reconstructed_volume = astra.data3d.get(reconstruction_id)

    # Clean up
    astra.data3d.delete([sinogram_id, reconstruction_id])

    print("Reconstructed volume shape:", reconstructed_volume.shape)
    return reconstructed_volume



def visualize_voxel_grid_interactive(voxel_grid,thresh = 0, subset = 5):
    """
    Visualize a 3D voxel grid interactively using Plotly, excluding zero values.

    Args:
        voxel_grid (torch.Tensor or np.ndarray): 3D tensor of shape (D, H, W) containing voxel values.
                                                 Values should be in the range [0, max_value].
    """
    # Convert to numpy if it's a PyTorch tensor
    if isinstance(voxel_grid, torch.Tensor):
        voxel_grid = voxel_grid.cpu().numpy()

    # Get the coordinates of all non-zero voxels
    non_zero_indices = np.where(voxel_grid > thresh)
    subset_size = len(non_zero_indices[0]) // subset  # 10% of the total points
    sampled_indices = np.random.choice(len(non_zero_indices[0]), subset_size, replace=False)


    x = non_zero_indices[0][sampled_indices]
    y = non_zero_indices[1][sampled_indices]
    z = non_zero_indices[2][sampled_indices]
    
    values = voxel_grid[x,y,z]  # Corresponding voxel values
    print( values.min())
    #print(len(non_zero_indices))
    # Normalize the values for the color mapping (0 -> white, max -> black)
    normalized_values = (values - values.min()) / (values.max() - values.min())
    #colors = (1 - normalized_values)  # Invert: 0 -> white, 1 -> black
    colors_rgb = [[c, c, c] for c in normalized_values]  # Grayscale colors

    # Create the scatter plot
    fig = go.Figure(
        data=go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=dict(
                size=3,
                color=normalized_values,  # Intensity of the voxel values
                colorscale="viridis",  # Map 0 -> white, 1 -> black
                cmin=0,
                cmax=1,
                colorbar=dict(
                    title="Voxel Value",
                    tickvals=[0, 1],
                    #ticktext=["White (0)", "Black (max)"],
                ),
            ),
        )
    )

    # Update the layout for better visualization
    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="cube",  # Keep aspect ratio cubic
        ),
        title="Interactive 3D Voxel Grid",
    )

    # Show the plot
    fig.show()
