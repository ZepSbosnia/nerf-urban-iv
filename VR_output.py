import math
import trimesh
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import raymarching
from nerf import utils
from nerf.utils import custom_meshgrid

from PIL import Image 
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import stat 
 
import my_utils_x_ray as my_utils

def density_fun(xyzs: torch.Tensor):
    r2 = xyzs[:, 0]**2 + xyzs[:, 1]**2 + xyzs[:, 2]**2
    inside = r2 <= 1.0

    return torch.where(
        inside,
        torch.full_like(r2, 10.0, dtype=torch.float32, device=xyzs.device),
        torch.zeros_like(r2, dtype=torch.float32, device=xyzs.device)
    )

def density_fun_cube(xyzs: torch.Tensor):
    inside = (
        (xyzs[:, 0].abs() <= 1.0) &
        (xyzs[:, 1].abs() <= 1.0) &
        (xyzs[:, 2].abs() <= 1.0)
    )

    return torch.where(
        inside,
        torch.full((xyzs.shape[0],), 10.0, dtype=torch.float32, device=xyzs.device),
        torch.zeros((xyzs.shape[0],), dtype=torch.float32, device=xyzs.device)
    )

def density_fun_multi(xyzs: torch.Tensor):
    x, y, z = xyzs[:, 0], xyzs[:, 1], xyzs[:, 2]

    # Shape 1: sphere centered at origin, radius .9
    sphere = (x**2 + y**2 + z**2) <= 0.9**2

    # Shape 2: cube centered at (1, 0, 0), side 0.6
    cube = ((x - 1).abs() <= 0.3) & (y.abs() <= 0.3) & (z.abs() <= 0.3)

    # Shape 3: cylinder along z-axis, radius 0.25, height 1.0, centered at (-1.2, 0, 0)
    cyl = ((x + 1.2)**2 + y**2 <= 0.25**2) & (z.abs() <= 0.5)

    # Shape 4: ellipsoid centered at (0, 1.2, 0), radii (0.3, 0.15, 0.4)
    ellipsoid = (((x)**2 / 0.3**2) + ((y - 1.2)**2 / 0.15**2) + (z**2 / 0.4**2)) <= 1

    density = torch.zeros(xyzs.shape[0], device=xyzs.device) #* .1

    # Assign density = 10 inside any shape
    density[sphere | cube | cyl | ellipsoid] = 10.0

    return density

def density_house(xyzs: torch.Tensor) -> torch.Tensor:
    """
    Create a 3D density field shaped like a house: cube base, roof, and a front rectangular extrusion.

    Args:
        xyzs: (N, 3) tensor of 3D coordinates

    Returns:
        (N,) tensor of densities
    """
    x, y, z = xyzs[:, 0], xyzs[:, 1], xyzs[:, 2]

    # Base cube: from -0.5 to 0.5 in x, z and -0.5 to 0.0 in y
    in_base = (
        (x >= -0.5) & (x <= 0.5) &
        (z >= -0.5) & (z <= 0.5) &
        (y >= -1.5) & (y <= 0.0)
    )

    # Roof (triangular prism): y in [0.0, 0.5]
    height = y - 0.0
    roof_half_width = 0.5 * (1.0 - height * 2.0)
    in_roof = (
        (y >= 0.0) & (y <= 0.5) &
        (x >= -roof_half_width) & (x <= roof_half_width) &
        (z >= -0.5) & (z <= 0.5)
    )

    # Front rectangular extrusion ("door" or "chimney"), centered at z = 0.5
    in_extrusion = (
        (x >= -0.1) & (x <= 0.1) &
        (y >= -0.35) & (y <= -0.15) &
        (z >= 0.5) & (z <= 0.6)
    )

    inside = in_base | in_roof | in_extrusion

    return torch.where(
        inside,
        torch.tensor(10.0, dtype=torch.float32, device=xyzs.device),
        torch.tensor(0.0, dtype=torch.float32, device=xyzs.device)
    )
    
def color_house(xyzs: torch.Tensor) -> torch.Tensor:
    """
    Assign RGB colors to parts of a house shape:
    - Gray base (cube)
    - Red roof (triangular prism)
    - Blue extrusion (e.g., door or chimney)

    Args:
        xyzs: (N, 3) tensor of 3D coordinates

    Returns:
        (N, 3) RGB colors
    """
    x, y, z = xyzs[:, 0], xyzs[:, 1], xyzs[:, 2]

    # Base cube
    in_base = (
        (x >= -0.5) & (x <= 0.5) &
        (z >= -0.5) & (z <= 0.5) &
        (y >= -1.5) & (y <= 0.0)
    )

    # Roof (triangular prism)
    height = y - 0.0
    roof_half_width = 0.5 * (1.0 - height * 2.0)
    in_roof = (
        (y >= 0.0) & (y <= 0.5) &
        (x >= -roof_half_width) & (x <= roof_half_width) &
        (z >= -0.5) & (z <= 0.5)
    )

    # Extrusion (e.g., door or chimney)
    in_extrusion = (
        (x >= -0.1) & (x <= 0.1) &
        (y >= -0.35) & (y <= -0.15) &
        (z >= 0.5) & (z <= 0.6)
    )

    # Default: black background
    colors = torch.zeros_like(xyzs)

    # Assign colors
    colors[in_base] = torch.tensor([1, 0.77, 0.8], device=xyzs.device)     # pink base
    colors[in_roof] = torch.tensor([1.0, 0.0, 0.0], device=xyzs.device)     # red roof
    colors[in_extrusion] = torch.tensor([0.2, 0.2, 1.0], device=xyzs.device) # blue door/chimney

    return colors


def density_virus(xyzs: torch.Tensor, num_spikes=100, spike_length=0.2, spike_radius=0.1, seed=42):
    """
    Generate a volumetric density resembling a virus (sphere with spikes).
    
    Args:
        xyzs: (N, 3) coordinates in space
        num_spikes: number of radial protrusions
        spike_length: how far spikes reach out from the sphere
        spike_radius: thickness of each spike
        seed: for reproducibility
    
    Returns:
        (N,) density values
    """
    torch.manual_seed(seed)
    r2 = (xyzs ** 2).sum(dim=1)
    density = torch.zeros_like(r2)

    # Base sphere
    inside_sphere = r2 <= 1.0
    density[inside_sphere] = 10.0

    # Random spike directions (unit vectors)
    dirs = torch.randn(num_spikes, 3, device=xyzs.device)
    dirs = dirs / (dirs.norm(dim=1, keepdim=True) + 1e-8)

    for i in range(num_spikes):
        center_line = dirs[i].flatten()  # Make sure it's (3,)
        proj = torch.sum(xyzs * center_line[None, :], dim=1)  # (N,)
        spike_length = np.random.uniform(0.3, 0.5)
        spike_radius = np.random.uniform(0.02,0.1)
        mask = (proj >= 1.0) & (proj <= 1.0 + spike_length)
        orth_dist2 = ((xyzs - proj[:, None] * center_line[None, :]) ** 2).sum(dim=1)
        spike_mask = mask & (orth_dist2 <= spike_radius**2)

        density[spike_mask] += 20.0 #* torch.exp(-orth_dist2[spike_mask] / (2 * spike_radius**2))

    return torch.clamp(density, min=0.0)



def color_virus(xyzs: torch.Tensor):
    """
    Color points red if inside the sphere,
    and assign varying colors to spike regions based on angle.
    
    Args:
        xyzs: (N, 3) points
    
    Returns:
        (N, 3) RGB colors
    """
    x, y, z = xyzs[:, 0], xyzs[:, 1], xyzs[:, 2]
    r = torch.sqrt(x**2 + y**2 + z**2 + 1e-5)

    # Define inside the main sphere (radius ~0.8)
    inside_sphere = r <= 1

    # Compute spherical coordinates
    theta = torch.acos(torch.clamp(z / r, -1.0, 1.0))  # polar
    phi = torch.atan2(y, x)                            # azimuth

    # Colors based on angles (gives patterns on spikes)
    r_col = (torch.sin(3 * phi) * 0.5 + 0.5)
    g_col = (torch.sin(5 * theta) * 0.5 + 0.5)
    b_col = (r - r.min()) / (r.max() - r.min() + 1e-6)
    spike_colors = torch.stack([r_col, g_col, b_col], dim=-1).clamp(0, 1)

    # Final colors: red inside, varied outside
    colors = torch.where(
        inside_sphere.unsqueeze(-1),
        torch.tensor([1.0, 0.0, 0.0], device=xyzs.device),
        spike_colors
    )

    return colors



def color_fun_multi(xyzs: torch.Tensor):
    x, y, z = xyzs[:, 0], xyzs[:, 1], xyzs[:, 2]

    # Reuse shape masks from density function
    sphere = (x**2 + y**2 + z**2) <= 0.9**2
    cube = ((x - 1).abs() <= 0.3) & (y.abs() <= 0.3) & (z.abs() <= 0.3)
    cyl = ((x + 1.2)**2 + y**2 <= 0.25**2) & (z.abs() <= 0.5)
    ellipsoid = (((x)**2 / 0.3**2) + ((y - 1.2)**2 / 0.15**2) + (z**2 / 0.4**2)) <= 1

    colors = torch.zeros_like(xyzs)#*100

    # Background color (outside all shapes)
    inside_any = sphere | cube | cyl | ellipsoid
    colors[~inside_any] = torch.tensor([0.1, 0.1, 0.1], device=xyzs.device)  # dark gray

    colors[sphere] = torch.tensor([1.0, 0.0, 0.0], device=xyzs.device)      # red
    colors[cube] = torch.tensor([0.0, 1.0, 0.0], device=xyzs.device)        # green
    colors[cyl] = torch.tensor([0.0, 0.0, 1.0], device=xyzs.device)         # blue
    colors[ellipsoid] = torch.tensor([1.0, 1.0, 0.0], device=xyzs.device)   # yellow

    return colors
    
def color_fun(xyzs: torch.Tensor):
    r2 = xyzs[:, 0]**2 + xyzs[:, 1]**2 + xyzs[:, 2]**2
    inside = r2 <= 1.0

    colors = torch.zeros_like(xyzs)

    # Assign low blue outside the sphere
    colors[~inside] = torch.tensor([0.1, 0.1, 0.5], dtype=torch.float32, device=xyzs.device)

    # Apply original logic inside the sphere
    mask_pos = (xyzs[:, 0] > 0.3) & inside
    mask_neg = (xyzs[:, 0] <= 0.3) & inside

    colors[mask_pos] = torch.tensor([1.0, 0.3, 1.0], dtype=torch.float32, device=xyzs.device)
    colors[mask_neg] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=xyzs.device)

    return colors

def color_fun_cube(xyzs: torch.Tensor):
    # Define a cube: |x| <= 1, |y| <= 1, |z| <= 1
    inside = (
        (xyzs[:, 0].abs() <= 1.0) &
        (xyzs[:, 1].abs() <= 1.0) &
        (xyzs[:, 2].abs() <= 1.0)
    )

    colors = torch.zeros_like(xyzs)

    # Assign default bluish color outside the cube
    colors[~inside] = torch.tensor([0.1, 0.1, 0.5], dtype=torch.float32, device=xyzs.device)

    # Left half of the cube (x <= 0)
    mask_left = (xyzs[:, 0] <= 0) & inside
    colors[mask_left] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=xyzs.device)  # red

    # Right half of the cube (x > 0)
    mask_right = (xyzs[:, 0] > 0) & inside
    colors[mask_right] = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=xyzs.device)  # green

    return colors


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # bins: [B, T], old_z_vals
    # weights: [B, T - 1], bin weights.
    # return: [B, n_samples], new_z_vals

    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples).to(weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).to(weights.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (B, n_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples



def save_rays_as_ply(camera_centers: torch.Tensor, rays_dirs: torch.Tensor,
                         ray_length=.4, save_path="/volume/torch-ngp/VR_res/rays.ply"):
    """
    Save rays for all poses in separate PLY files.

    Args:
        camera_centers: (N, M, 3) tensor of ray origins
        rays_dirs: (N, M, 3) tensor of ray directions
        ray_length: length to scale ray directions
        save_path_prefix: prefix for output filenames, e.g. '.../rays' 
                          results in files like rays_0.ply, rays_1.ply, ...
    """
    N, M, _ = camera_centers.shape

    # Flatten all origins and directions
    origins = camera_centers.reshape(-1, 3).cpu().numpy()   # (N*M, 3)
    directions = rays_dirs.reshape(-1, 3).cpu().numpy()    # (N*M, 3)

    points_start = origins
    points_end = origins + directions * ray_length

    vertices = np.vstack([points_start, points_end])
    num_vertices = vertices.shape[0]

    edges = [[i, i + N * M] for i in range(N * M)]
    num_edges = len(edges)

    with open(save_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_vertices}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element edge {num_edges}\n")
        f.write("property int vertex1\nproperty int vertex2\n")
        f.write("end_header\n")

        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")

        for e in edges:
            f.write(f"{e[0]} {e[1]}\n")

    print(f"Saved all rays for {N} poses with {M} rays each to {save_path}")
        




def save_rays_as_ply_rgb(camera_center: torch.Tensor, rays_dirs: torch.Tensor, colors: torch.Tensor,
                         ray_length=0.4, save_path="/volume/torch-ngp/VR_res/rays_rgb.ply", count=None,
                         axes: torch.Tensor = None):
    """
    Save a point cloud of ray points starting at camera_center, going along rays_dirs,
    coloring each point with the corresponding color.
    Optionally add coordinate axes as colored lines from camera_center.

    Args:
        camera_center: (3,) tensor for camera origin
        rays_dirs: (N, 3) tensor for directions (assumed normalized)
        colors: (N, 3) tensor of uint8 RGB colors [0-255]
        ray_length: length of each ray (distance from camera_center to point)
        save_path: output PLY filename
        count: optional integer to append to filename
        axes: optional (3,3) tensor of orientation axes vectors from camera_center
              (each axis should be scaled to desired length, e.g. unit vectors * 0.1)
    """
    N = rays_dirs.shape[0]

    # Compute points along rays
    points = camera_center.unsqueeze(0) + rays_dirs * ray_length  # (N, 3)

    # Convert to numpy
    points_np = points.cpu().numpy()
    colors_np = colors.cpu().numpy().astype(np.uint8)

    if count is not None:
        save_path = save_path.replace('.ply', f'_{count}.ply')

    # Prepare axis points and colors if axes are provided
    axis_points_np = None
    axis_colors_np = None
    axis_edges = []
    if axes is not None:
        # Ensure shape (3,3)
        assert axes.shape == (3, 3)
        # Axes are vectors from camera_center: add to origin to get tips
        origin_np = camera_center.cpu().numpy().reshape(1, 3)
        axis_tips_np = origin_np + axes.cpu().numpy()  # shape (3, 3)

        # Points = origin + axis tips
        # We'll store origin once, and tips
        axis_points_np = np.vstack([origin_np, axis_tips_np])  # (4,3)

        # Colors for axes: Red, Green, Blue for X, Y, Z
        axis_colors_np = np.array([
            [255, 255, 255],  # origin white
            [255, 0, 0],      # X red
            [0, 255, 0],      # Y green
            [0, 0, 255],      # Z blue
        ], dtype=np.uint8)

        # Edges connecting origin to each axis tip (line segments)
        # origin is vertex 0, tips are 1,2,3
        axis_edges = [[0, 1], [0, 2], [0, 3]]

    # Combine points and colors
    if axes is not None:
        all_points = np.vstack([points_np, axis_points_np])
        all_colors = np.vstack([colors_np, axis_colors_np])
    else:
        all_points = points_np
        all_colors = colors_np

    num_vertices = all_points.shape[0]

    # Prepare edges (optional)
    edges = []  # No edges for rays here, but can add if you want
    if axes is not None:
        # offset edges by N (number of rays points)
        edges.extend([[e[0] + N, e[1] + N] for e in axis_edges])
    num_edges = len(edges)

    # Write PLY
    with open(save_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_vertices}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        if num_edges > 0:
            f.write(f"element edge {num_edges}\n")
            f.write("property int vertex1\nproperty int vertex2\n")
        f.write("end_header\n")

        for p, c in zip(all_points, all_colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")

        for e in edges:
            f.write(f"{e[0]} {e[1]}\n")

    print(f"Saved {N} colored points + axes to {save_path}")
    
    
    
    
def run( rays_o, rays_d, num_steps=128, upsample_steps=128, bg_color=None, perturb=False):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # bg_color: [3] in range [0, 1]
        # return: image: [B, N, 3], depth: [B, N]

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0] # N = B * N, in fact
        device = rays_o.device

        # choose aabb
        aabb = torch.tensor([-2.0, -2.0, -2.0,2.0, 2.0, 2.0], device=device)
            
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, aabb, 0.0)
        nears.unsqueeze_(-1)
        fars.unsqueeze_(-1) 
 

        #print(f'nears = {nears.min().item()} ~ {nears.max().item()}, fars = {fars.min().item()} ~ {fars.max().item()}')

        z_vals = torch.linspace(0.0, 1.0, num_steps, device=device).unsqueeze(0) # [1, T]
        z_vals = z_vals.expand((N, num_steps)) # [N, T]
        z_vals = nears + (fars - nears) * z_vals # [N, T], in [nears, fars]

        # perturb z_vals
        sample_dist = (fars - nears) / num_steps
        if perturb:
            z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist
            #z_vals = z_vals.clamp(nears, fars) # avoid out of bounds xyzs.

        # generate xyzs
        xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1) # [N, 1, 3] * [N, T, 1] -> [N, T, 3]
        xyzs = torch.min(torch.max(xyzs, aabb[:3]), aabb[3:]) # a manual clip.

        #plot_pointcloud(xyzs.reshape(-1, 3).detach().cpu().numpy())

        # query SDF and RGB
        density_outputs =  density_house(xyzs.reshape(-1, 3))
 
        density_outputs  = density_outputs.view(N, num_steps)
 

        # upsample z_vals (nerf-like)
        if upsample_steps > 0:
            with torch.no_grad():

                deltas = z_vals[..., 1:] - z_vals[..., :-1] # [N, T-1]
                deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)

                alphas = 1 - torch.exp(-deltas  * density_outputs.squeeze(-1)) # [N, T]
                alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1) # [N, T+1]
                weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # [N, T]

        alphas = 1 - torch.exp(-deltas *  density_outputs.squeeze(-1)) # [N, T+t]
        alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1) # [N, T+t+1]
        weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # [N, T+t]
  
        rgbs = color_house(xyzs.reshape(-1, 3))#, dirs.reshape(-1, 3), mask=mask.reshape(-1), **density_outputs)
        rgbs = rgbs.view(N, -1, 3) # [N, T+t, 3]

        #print(xyzs.shape, 'valid_rgb:', mask.sum().item())

        # calculate weight_sum (mask)
        weights_sum = weights.sum(dim=-1) # [N]
        
        # calculate depth 
        ori_z_vals = ((z_vals - nears) / (fars - nears)).clamp(0, 1)
        depth = torch.sum(weights * ori_z_vals, dim=-1)

        # calculate color
        image = torch.sum(weights.unsqueeze(-1) * rgbs, dim=-2) # [N, 3], in [0, 1]

        bg_color = 1
            
        image = image + (1 - weights_sum).unsqueeze(-1) * bg_color

        alpha = (weights_sum > 0).float().unsqueeze(-1)  # shape: (*prefix, 1)

        # Concatenate alpha as 4th channel
        image = torch.cat([image, alpha], dim=-1)  # shape: (*prefix, 4)

        image = image.view(*prefix, 4)
        depth = depth.view(*prefix)

        # tmp: reg loss in mip-nerf 360
        # z_vals_shifted = torch.cat([z_vals[..., 1:], sample_dist * torch.ones_like(z_vals[..., :1])], dim=-1)
        # mid_zs = (z_vals + z_vals_shifted) / 2 # [N, T]
        # loss_dist = (torch.abs(mid_zs.unsqueeze(1) - mid_zs.unsqueeze(2)) * (weights.unsqueeze(1) * weights.unsqueeze(2))).sum() + 1/3 * ((z_vals_shifted - z_vals_shifted) * (weights ** 2)).sum()

        return {
            'depth': depth,
            'image': image,
            'weights_sum': weights_sum,
        }

  
 
resolution = 800
n_photos = 250

rays_o , rays_dirs  = my_utils.DP_fan_camera_poses(40,40,6,res=resolution, focal_lenght=2, rand_angle=n_photos, perturb_dir=True, rand_S=[.95,1.05] )

#rays_dirs *= -1
 

#save_rays_as_ply(rays_o,rays_dirs)
folder = '/volume/torch-ngp_NODIR/'
os.chmod(folder , stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
j=0
for i in range(n_photos):
    res = run( rays_o[i ,...].clone(), rays_dirs[i ,...].clone(),num_steps=128)
    img = res['image']  # Assuming shape (H, W) or (H, W, 3)
 
    #save_rays_as_ply_rgb(rays_o[ i ,0].clone(), rays_dirs[ i ,...].clone(), 255*img[:,:3], axes = orientations[i,...], count=i )
    img_np = img.detach().cpu().numpy()  # to numpy
    img_np = img_np.reshape(resolution, resolution, 4)  # reshape to (H, W, C)

    # Convert to 0–255 and uint8
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)

    # Save as RGB PNG
    img_pil = Image.fromarray(img_np, mode='RGBA')
    folder = '/volume/torch-ngp_NODIR/VR_res/house/images'
    os.makedirs(folder, exist_ok=True)  # creates the folder and intermediate ones if needed
    
    img_pil.save(f'{folder}/image_{i:03d}.png') 
    

    # Save 
print(res['image'].shape)