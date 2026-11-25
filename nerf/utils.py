import os
import glob
import tqdm
import math
import imageio
import random
import warnings
import tensorboardX

import numpy as np
import pandas as pd

import time
from datetime import datetime

import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

import tensorflow as tf 
import trimesh
import mcubes
from rich.console import Console
from torch_ema import ExponentialMovingAverage

from packaging import version as pver
import lpips
from torchmetrics.functional import structural_similarity_index_measure

import scipy.ndimage as ndi

import colmap_loader 
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider

import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms

def sobel_operator(image):
    # Step 1: Convert RGB image to grayscale
    grayscale_image = TF.rgb_to_grayscale(image.permute(2, 0, 1), num_output_channels=1)

    # Step 2: Define Sobel kernels
    sobel_x = torch.tensor([[-1, 0, 1], 
                            [-2, 0, 2], 
                            [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    sobel_y = torch.tensor([[-1, -2, -1], 
                            [0, 0, 0], 
                            [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Ensure the Sobel kernels are on the same device and data type as the input image
    sobel_x = sobel_x.to(grayscale_image.device, dtype=grayscale_image.dtype)
    sobel_y = sobel_y.to(grayscale_image.device, dtype=grayscale_image.dtype)

    # Reshape grayscale_image to match the expected input shape for conv2d
    image = grayscale_image.unsqueeze(0)  # Shape: (1, 1, n, m)

    # Apply Sobel filters
    sobel_filtered_x  = F.conv2d(image, sobel_x, padding=1)
    sobel_filtered_y  = F.conv2d(image, sobel_y, padding=1)

    # Compute gradient magnitude
    gradient_magnitude = torch.sqrt(sobel_filtered_x ** 2 + sobel_filtered_y ** 2)

    # Remove batch dimension
    gradient_magnitude = gradient_magnitude.squeeze(0)

    return gradient_magnitude

def DP_Mininmum_Gradient_Support(dens,deltas,lambda_TV,eps):

    density = torch.as_tensor(dens)
    mask_delt = deltas> 0

    # Step 3: Select the indices where the condition is True
    selected_indices_delt = torch.nonzero(mask_delt, as_tuple=True)[0]
    diff = density[selected_indices_delt]  - torch.roll(density[selected_indices_delt] , shifts=1, dims=0)
    deltas_sel= deltas[selected_indices_delt] 
    mask = torch.abs(diff) > eps
    selected_indices = torch.nonzero(mask, as_tuple=True)[0]
    deltas_sel= deltas_sel[selected_indices] 
    diff = diff[selected_indices] 
    grad = diff / deltas_sel
    Supp = torch.sum(deltas_sel)
    val = deltas_sel * 1 /(grad**2+eps**2)
    loss = lambda_TV * (Supp - eps**2 *torch.sum(val))
    #res =  (torch.sum(x_diff_relu) + torch.sum(y_diff_relu) + torch.sum(z_diff_relu) )
    return loss

def DP_Mininmum_Support(dens,deltas,lambda_TV,eps):

    density = torch.as_tensor(dens)
    val = deltas * density**2 /(density**2+eps**2)
    loss = lambda_TV * torch.sum(val)/(torch.sum(deltas) + eps)
    #res =  (torch.sum(x_diff_relu) + torch.sum(y_diff_relu) + torch.sum(z_diff_relu) )
    return loss

def DP_Mininmum_Support_grid(dens,lambda_MS,eps):

    density = torch.as_tensor(dens)
    val =  density**2 /(density**2+eps**2)
    loss = lambda_MS * torch.sum(val) / np.prod(density.shape)
    #res =  (torch.sum(x_diff_relu) + torch.sum(y_diff_relu) + torch.sum(z_diff_relu) )
    return loss

# define your custom loss
def DP_total_variation_loss(dens,lambda_TV,eps):
    density = torch.as_tensor(dens)
    x_diff = density - torch.roll(density, shifts=1, dims=0)
    y_diff = density - torch.roll(density, shifts=1, dims=1)
    z_diff = density - torch.roll(density, shifts=1, dims=2)
    
    ''' 
    x_diff_relu = torch.relu(x_diff) + torch.relu(- x_diff)
    y_diff_relu = torch.relu(y_diff) + torch.relu(- y_diff)
    z_diff_relu = torch.relu(z_diff) + torch.relu(- z_diff)
    '''

    x_diff_huber = torch.nn.functional.huber_loss(density ,torch.roll(density, shifts=1, dims=0), reduction='mean', delta=eps)
    y_diff_huber = torch.nn.functional.huber_loss(density ,torch.roll(density, shifts=1, dims=1), reduction='mean', delta=eps)
    z_diff_huber = torch.nn.functional.huber_loss(density ,torch.roll(density, shifts=1, dims=2), reduction='mean', delta=eps)

    sum = x_diff_huber+y_diff_huber+z_diff_huber
    #res =  (torch.sum(x_diff_relu) + torch.sum(y_diff_relu) + torch.sum(z_diff_relu) )
    loss = lambda_TV * sum
    return loss


def chamfer(point_cloud1, point_cloud2):
    # Calculate pairwise distances
    pairwise_distances = torch.norm(
        torch.unsqueeze(point_cloud1, dim=1) - torch.unsqueeze(point_cloud2, dim=0),
        dim=-1
    )
    # Find the minimum distance for each point in the first set
    chamfer = torch.min(pairwise_distances, dim=1).values
    # Calculate the mean of the minimum distances
    chamfer_dist = torch.mean(chamfer)
    '''
    print(torch.isnan(chamfer).any())
    print(torch.isinf(chamfer).any())

    print(chamfer_dist)
    print(chamfer_dist.item())
    print(pairwise_distances)'''
    # Convert the result to a NumPy array if needed
    return chamfer_dist.item() 

def DP_loss_distortion(t, w):
    """Compute iint w[i] w[j] |t[i] - t[j]| di dj."""
    # The loss incurred between all pairs of intervals.
    ut = (t[..., 1:] + t[..., :-1]) / 2
    dut = torch.abs(ut[..., :, None] - ut[..., None, :])
    loss_inter = torch.sum(w * torch.sum(w[..., None, :] * dut, dim=-1), dim=-1)

    # The loss incurred within each individual interval with itself.
    loss_intra = torch.sum(w**2 * (t[..., 1:] - t[..., :-1]), dim=-1) / 3

    return loss_inter + loss_intra


class DP_distortion_loss(nn.Module):
    def __init__(self, lambda_dist):
        super().__init__()
        self.dist_loss = lambda_dist
    def forward(self, t,w,lambda_dist):
        ut = (t[..., 1:] + t[..., :-1]) / 2
        dut = torch.abs(ut[..., :, None] - ut[..., None, :])
        loss_inter = torch.sum(w * torch.sum(w[..., None, :] * dut, dim=-1), dim=-1)

        # The loss incurred within each individual interval with itself.
        loss_intra = torch.sum(w**2 * (t[..., 1:] - t[..., :-1]), dim=-1) / 3
        loss = loss_inter + loss_intra
        loss = lambda_dist * loss.mean()
        return loss 
    
DP_loss_dist = DP_distortion_loss(lambda_dist=0)   

# define your custom loss
def DP_total_variation_loss_1D(dens,lambda_TV,eps):
    density = torch.as_tensor(dens)
    diff = density - torch.roll(density, shifts=1, dims=0)
    diff_huber = torch.nn.functional.huber_loss(density ,torch.roll(density, shifts=1, dims=0), reduction='mean', delta=eps)

    #res =  (torch.sum(x_diff_relu) + torch.sum(y_diff_relu) + torch.sum(z_diff_relu) )
    loss = lambda_TV * diff_huber
    return loss




class DP_TV_loss(nn.Module):
    def __init__(self, lambda_TV,eps):
        super().__init__()
        self.lambda_TV = lambda_TV

    def forward(self, predicted, lambda_TV,eps):
        loss = DP_total_variation_loss(predicted, lambda_TV,eps) 
        return loss
loss_TV = DP_TV_loss(lambda_TV=0,eps = 1.0)

class DP_TV_loss_1D(nn.Module):
    def __init__(self, lambda_TV,eps):
        super().__init__()
        self.lambda_TV = lambda_TV

    def forward(self, predicted, lambda_TV,eps):
        loss = DP_total_variation_loss_1D(predicted, lambda_TV,eps) 
        return loss
loss_TV_1D = DP_TV_loss_1D(lambda_TV=0,eps = 1.0)

  
class DP_MGS(nn.Module):
    def __init__(self, lambda_TV,eps):
        super().__init__()
        self.lambda_TV = lambda_TV
    def forward(self, dens,delt, lambda_TV,eps):
        loss = DP_Mininmum_Gradient_Support(dens,delt, lambda_TV,eps) 
        return loss
loss_MGS = DP_MGS(lambda_TV=0,eps = 1.0)

class DP_MS(nn.Module):
    def __init__(self, lambda_TV,eps):
        super().__init__()
        self.lambda_TV = lambda_TV
    def forward(self, dens,delt, lambda_TV,eps):
        loss = DP_Mininmum_Support(dens,delt, lambda_TV,eps) 
        return loss
loss_MS = DP_MS(lambda_TV=0,eps = 1.0)

class DP_MS_grid(nn.Module):
    def __init__(self, lambda_TV,eps):
        super().__init__()
        self.lambda_TV = lambda_TV
    def forward(self, dens,lambda_TV,eps):
        loss = DP_Mininmum_Support_grid(dens, lambda_TV,eps) 
        return loss
loss_MS_grid = DP_MS_grid(lambda_TV=0,eps = 1.0)

class DP_chamfer_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predicted, colmap_pc):
        loss = chamfer(predicted,colmap_pc) 
        return loss
loss_chamfer = DP_chamfer_loss()

# define your custom loss

def DP_distortion_loss_old(xyzs,rays,dens):
    loss_inter = torch.tensor([0.0]).to("cuda")
    loss_intra = torch.tensor([0.0]).to("cuda")

    for r in rays:
        ray = xyzs[r[1]:r[2]]
        ray_dens = dens[r[1]:r[2]]
        for i,w_i in enumerate(ray_dens[:-1]) : 
            for j,w_j in enumerate(ray_dens[:-1]):
                loss_inter += w_i*w_j *  torch.norm( (ray[i] + ray[i+1] )/2 - (ray[j] + ray[j+1] )/2 )
                loss_intra += 1/3 * w_i**2 *  torch.norm(ray[i] - ray[i+1] ) 
        '''ut = (ray[ 1:] + ray[ :-1]) / 2
        print(ut[5])
        dut = torch.abs(ut[..., :, None] - ut[..., None, :])
        loss_inter = torch.sum(ray_dens.unsqueeze(-1).unsqueeze(-1)  * torch.sum(ray_dens.unsqueeze(-1).unsqueeze(-1)[..., None, :] * dut))
        loss_intra = torch.sum(ray_dens[..., None]**2 * (ray[..., 1:] - ray[..., :-1])) / 3'''
    res =  loss_inter + loss_intra
    return res/rays.shape[0]

def DP_extreme_loss( y_pred):
    # Calculate additional loss terms to encourage extreme values
    near_zero_loss = torch.mean(torch.abs(y_pred - 0))*.01
    near_five_loss = torch.mean(torch.abs(y_pred - 5)) *.0001
    
    # Combine the losses with appropriate weights
    total_loss = near_zero_loss #+ near_five_loss
    
    return total_loss

def DP_distortion_loss(xyzs,dens):
    loss_inter = torch.tensor([0.0]).to("cuda")
    loss_intra = torch.tensor([0.0]).to("cuda")


    for ind_x,rays_x in enumerate(xyzs):
        for ind_y,ray in enumerate(rays_x):
            ray_dens = dens[ind_x,ind_y,:]
            for i,w_i in enumerate(ray_dens[:-1]) : 
                for j,w_j in enumerate(ray_dens[:-1]):
                    loss_inter += w_i*w_j *  torch.norm( (ray[i] + ray[i+1] )/2 - (ray[j] + ray[j+1] )/2 )
                    loss_intra += 1/3 * w_i**2 *  torch.norm(ray[i] - ray[i+1] ) 
        '''ut = (ray[ 1:] + ray[ :-1]) / 2
        print(ut[5])
        dut = torch.abs(ut[..., :, None] - ut[..., None, :])
        loss_inter = torch.sum(ray_dens.unsqueeze(-1).unsqueeze(-1)  * torch.sum(ray_dens.unsqueeze(-1).unsqueeze(-1)[..., None, :] * dut))
        loss_intra = torch.sum(ray_dens[..., None]**2 * (ray[..., 1:] - ray[..., :-1])) / 3'''
    res =  loss_inter + loss_intra
    return res/rays.shape[0]

class DP_dist_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, xyzs,dens):
        loss = DP_distortion_loss(xyzs,dens) 
        return loss

loss_distortion = DP_dist_loss()


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


@torch.jit.script
def linear_to_srgb(x):
    return torch.where(x < 0.0031308, 12.92 * x, 1.055 * x ** 0.41666 - 0.055)


@torch.jit.script
def srgb_to_linear(x):
    return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


@torch.cuda.amp.autocast(enabled=False)
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


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True


def torch_vis_2d(x, renormalize=False):
    # x: [3, H, W] or [1, H, W] or [H, W]
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    
    if isinstance(x, torch.Tensor):
        if len(x.shape) == 3:
            x = x.permute(1,2,0).squeeze()
        x = x.detach().cpu().numpy()
        
    print(f'[torch_vis_2d] {x.shape}, {x.dtype}, {x.min()} ~ {x.max()}')
    
    x = x.astype(np.float32)
    
    # renormalize
    if renormalize:
        x = (x - x.min(axis=0, keepdims=True)) / (x.max(axis=0, keepdims=True) - x.min(axis=0, keepdims=True) + 1e-8)

    plt.imshow(x)
    plt.show()

#DP !!! comment: returns the evaluation of the density into a meshgrid bounded by the input
def extract_fields(bound_min, bound_max, resolution, query_func, S=128):

    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(S)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(S)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(S)

    #DP !!!
    ''' 
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(S)
    Y = torch.linspace(-1,2, resolution).split(S)
    Z = torch.linspace(-2,-.5, resolution).split(S)
    '''

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [S, 3]
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy() # [S, 1] --> [x, y, z]  #DP !!! query_func passata è la funzione che dai punti restituisce la densita
                    u[xi * S: xi * S + len(xs), yi * S: yi * S + len(ys), zi * S: zi * S + len(zs)] = val
    return u

import cv2

def DP_find_density_edges(volume, threshold):
    # Calculate gradient magnitude using the Sobel operator
    gradient = np.gradient(volume)
    gradient_magnitude = np.sqrt(np.sum(np.square(gradient), axis=0))

    # Apply a threshold to find the edges
    edges = gradient_magnitude > threshold

    return edges


def DP_remove_isolated_points(matrix, min_points=10):
    # Create a binary mask where non-zero values represent the points
    binary_mask = matrix != 0

    # Perform morphological opening to remove isolated points
    structuring_element = ndi.generate_binary_structure(3, 2)  # 3D structuring element
    opened_mask = ndi.binary_opening(binary_mask, structure=structuring_element)

    # Label connected components in the opened mask
    labeled_mask, num_labels = ndi.label(opened_mask)

    # Find the size of each connected component
    component_sizes = np.bincount(labeled_mask.ravel())

    # Identify the labels corresponding to isolated points based on the minimum number of points threshold
    isolated_point_labels = np.where(component_sizes <= min_points)[0]
    # Remove the isolated points from the labeled mask
    labeled_mask[np.isin(labeled_mask, isolated_point_labels)] = 0

    # Reconstruct the matrix by multiplying the labeled mask with the original matrix
    filtered_matrix = labeled_mask * matrix

    return filtered_matrix

def DP_extract_custom_fields(v1,v2,v3,resolution, query_func,area_bound, S=32): #S=128
    print(v1,v2,v3)
    T = torch.column_stack((torch.from_numpy(v1), torch.from_numpy(v3), torch.from_numpy(v2)))

    '''X = torch.linspace(area_bound[0,0], area_bound[0,1], resolution).split(S)
    Y = torch.linspace(area_bound[1,0], area_bound[1,1], resolution).split(S)
    Z = torch.linspace(area_bound[2,0], area_bound[2,1], resolution).split(S)'''

    X = torch.linspace(-2,2, resolution).split(S)
    Y = torch.linspace(-2,2, resolution).split(S)
    Z = torch.linspace(-2,2, resolution).split(S)
    #area_bound = np.array([[-2,2],[-.9,.5],[-.7,3]])#[[0,1],[-.9,.1],[-.7,.3]]
    
    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    #mask =   (xx > area_bound[0,0])&(xx < area_bound[0,1]) & (yy > area_bound[1,0])  & (yy < area_bound[1,1]) & (zz > area_bound[2,0])  & (yy < area_bound[2,1]) 
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [S, 3]
                    mask = (pts[:,0] > area_bound[0,0])&(pts[:,0] < area_bound[0,1]) & (pts[:,1] > area_bound[1,0]) & (pts[:,1] < area_bound[1,1]) & (pts[:,2] > area_bound[2,0])  & (pts[:,2] < area_bound[2,1]) 
                    pts_rot = torch.matmul(pts.to(dtype=torch.float32)  , T.to(dtype=torch.float32)  ) #to consider the points rotated by T 
                    val = torch.where(mask.to("cuda"),query_func(pts_rot), torch.zeros_like(mask).to("cuda")).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy() # [S, 1] --> [x, y, z]  #DP !!! query_func passata è la funzione che dai punti restituisce la densita
                    #val = query_func(pts_rot).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy() # [S, 1] --> [x, y, z]  #DP !!! query_func passata è la funzione che dai punti restituisce la densita
                    u[xi * S: xi * S + len(xs), yi * S: yi * S + len(ys), zi * S: zi * S + len(zs)] = val
     
    '''u[u< 3] =0
    edges = DP_find_density_edges(u,5)
    np.save( "/volume/torch-ngp_NODIR/results/360C_nosky_denoised_r4/exp_Huber_1e-2/TV_1e-2_grid_101/edges.npy", edges)
    u = DP_remove_isolated_points(u,20)
    u = edges *6'''
    return u, pts_rot, T
#DP !!! comment: create u vector with evaluation of density in a grid of points then determine 
# the vertices and traingles over threshold using marching_cubes func
def extract_geometry(bound_min, bound_max, resolution, threshold, query_func, DP_rotation = False,v1=None,v2=None,v3=None,b=None):
    #print('threshold: {}'.format(threshold))
    if not DP_rotation:
        '''bound_min = torch.ones(3) * -1.3
        bound_max = torch.ones(3) * 1.3'''
        u = extract_fields(bound_min, bound_max, resolution, query_func)
        vertices, triangles = mcubes.marching_cubes(u,threshold) #threshold
        #DP !!! by gpt:
        # Each row in vertices contains the x, y, and z coordinates of a single vertex. 
        #Thus, if vertices has dimensions (n, 3), there are n vertices in the mesh.
        #triangles is also a numpy ndarray containing the indices of the vertices that 
        # define each triangular face in the mesh. Each row in triangles contains the indices of 
        # the three vertices that define a single triangular face. If triangles has dimensions (m, 3), 
        # there are m triangular faces in the mesh.
        b_max_np = bound_max.detach().cpu().numpy()
        b_min_np = bound_min.detach().cpu().numpy()
        vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :] 
    else: 
        u,pts_rot, T = DP_extract_custom_fields(v1,v2,v3,resolution,query_func,b)

    #print(u.shape, u.max(), u.min(), np.percentile(u, 50))
        # mcubes.smooth(binary_sphere)
        vertices, triangles = mcubes.marching_cubes(u, threshold)
        #DP !!! by gpt:
        # Each row in vertices contains the x, y, and z coordinates of a single vertex. 
        #Thus, if vertices has dimensions (n, 3), there are n vertices in the mesh.
        #triangles is also a numpy ndarray containing the indices of the vertices that 
        # define each triangular face in the mesh. Each row in triangles contains the indices of 
        # the three vertices that define a single triangular face. If triangles has dimensions (m, 3), 
        # there are m triangular faces in the mesh.

        b_min_np = b[:,0]
        b_max_np = b[:,1]

        b_min_np = np.ones_like(b[:,0])*(-2)
        b_max_np = np.ones_like(b[:,1])*2
    
        vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :] 
        vertices = np.dot(vertices , T.detach().cpu().numpy())
    #DP !!! comment: rescaled the values of vertex in the original bounding box
    return vertices, triangles

#DP !!!
def DP_get_triangle_colors(vertices, triangles, color_func):
    # Compute the colors of the vertices
    vertex_colors = color_func(vertices)

    # Initialize an array to hold the colors of the triangles
    num_triangles = len(triangles)
    triangle_colors = np.zeros((num_triangles, 3))
    


    # Interpolate the colors of the triangles from the colors of their vertices
    for i, tri in enumerate(triangles):
        vertex_indices = tri.tolist()
        tri_colors = vertex_colors[vertex_indices]
        avg_color = np.mean(tri_colors, axis=0)
        triangle_colors[i] = avg_color
    alpha_array = np.ones((len(triangle_colors), 1), dtype=np.float32)
    triangle_colors =  np.hstack((triangle_colors, alpha_array))
    return vertex_colors, triangle_colors

#DP !!!
def DP_get_triangle_colors_2(Normals, baricentrics, color_func):
    # Compute the colors of the vertices
    b_colors = color_func(baricentrics,Normals)

    alpha_array = np.ones((len(b_colors), 1), dtype=np.float32)
    triangle_colors =  np.hstack((b_colors, alpha_array))
    return triangle_colors

def DP_create_rays(r_o, r_d, nears , fars ,n=100):
                    # Create a 1D tensor of n elements ranging from -b to b
        r_o = r_o.cuda()
        r_d = r_d.cuda()
        nears = nears.unsqueeze(-1).unsqueeze(-1)
        fars = fars.unsqueeze(-1).unsqueeze(-1)
# Create t using linspace
        def calculate_t(near, far, n):
            t = torch.linspace(near.item(), far.item(), n).cuda()  # Create t for a single pair of near and far
            return t

        # Apply the custom function to each pair of nears and fars using list comprehension
        t_list = [calculate_t(near, far, n) for near, far in zip(nears, fars)]

        # Stack the results to create the final t tensor
        t = torch.stack(t_list)
        #t = torch.linspace(nears, fars, n).cuda()
        t = t.unsqueeze(0)
        # Expand the dimensions of t to match rays_o and rays_dir
        #t = t.view(1, 1, n, 1)
        r_o_expanded = r_o.unsqueeze(-2)
        r_d_expanded = r_d.unsqueeze(-2)

        # Create the points tensor by element-wise multiplication
        points = r_o_expanded + r_d_expanded * t.unsqueeze(-1)
        # Create a tensor to store the points
        #points = r_o[:, :, None, :] + t[:, :, None, None] * r_d[:, :, None, :]

        return points
def DP_get_weight(density,rays):

    diff = rays[:,:-1] - rays[:,1:]
    delta = torch.norm(diff, dim=2)
    product = - density * delta
    
    exp_prod = torch.exp(product)
    alpha = 1 - exp_prod
    # Calculate the cumulative sum along the second dimension (dimension 1)

    # transmittance must be exclusive: T_i = exp(-sum_{j<i} sigma_j * delta_j), so T_0 = 1
    T = torch.exp(torch.cumsum(product, dim=1) - product)

    w = alpha * T
    return w

class PSNRMeter:
    def __init__(self):
        self.V = 0
        self.N = 0

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths) # [B, N, 3] or [B, H, W, 3], range[0, 1]
          
        # simplified since max_pixel_value is 1 here.
        psnr = -10 * np.log10(np.mean((preds - truths) ** 2))
        
        self.V += psnr
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "PSNR"), self.measure(), global_step)

    def report(self):
        return f'PSNR = {self.measure():.6f}'



class TotalVariation:
    def __init__(self):
        self.V = 0
        self.N = 0

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths) # [B, N, 3] or [B, H, W, 3], range[0, 1]
          
        # simplified since max_pixel_value is 1 here.
        psnr = -10 * np.log10(np.mean((preds - truths) ** 2))
        
        self.V += psnr
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "PSNR"), self.measure(), global_step)

    def report(self):
        return f'TotalVariation = {self.measure():.6f}'


class SSIMMeter:
    def __init__(self, device=None):
        self.V = 0
        self.N = 0

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            inp = inp.permute(0, 3, 1, 2).contiguous() # [B, 3, H, W]
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths) # [B, H, W, 3] --> [B, 3, H, W], range in [0, 1]

        ssim = structural_similarity_index_measure(preds, truths)

        self.V += ssim
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "SSIM"), self.measure(), global_step)

    def report(self):
        return f'SSIM = {self.measure():.6f}'


class LPIPSMeter:
    def __init__(self, net='alex', device=None):
        self.V = 0
        self.N = 0
        self.net = net

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fn = lpips.LPIPS(net=net).eval().to(self.device)

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            inp = inp.permute(0, 3, 1, 2).contiguous() # [B, 3, H, W]
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs
    
    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths) # [B, H, W, 3] --> [B, 3, H, W], range in [0, 1]
        v = self.fn(truths, preds, normalize=True).item() # normalize=True: [0, 1] to [-1, 1]
        self.V += v
        self.N += 1
    
    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, f"LPIPS ({self.net})"), self.measure(), global_step)

    def report(self):
        return f'LPIPS ({self.net}) = {self.measure():.6f}'

class Trainer(object):
    def __init__(self, 
                 name, # name of this experiment
                 opt, # extra conf
                 model, # network 
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 optimizer=None, # optimizer
                 ema_decay=None, # if use EMA, set the decay
                 lr_scheduler=None, # scheduler
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 fp16=False, # amp optimize level
                 eval_interval=1, # eval once every $ epoch
                 max_keep_ckpt=2, # max num of saved ckpts in disk
                 workspace='workspace', # workspace to save logs & ckpts
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=True, # use loss as the first metric
                 report_metric_at_train=False, # also report metrics at training
                 use_checkpoint="latest", # which ckpt to use at init time
                 use_tensorboardX=True, # whether to use tensorboard for logging
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                 TV_lambda=0, #DP !!!
                 dist_loss=0, #DP !!!
                 Huber_eps=1 ,#DP !!!
                 MS_lambda=0,
                 sobel_mse = False,
                 mesh_thresh = 2.5,#DP !!!
                 l1_loss = 0,
                 extreme_loss = 0
                 ):
        
        self.name = name
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        
        #DP !!!
        self.MS_lambda = MS_lambda
        self.TV_lambda = TV_lambda
        self.Huber_eps = Huber_eps
        self.thresh = mesh_thresh
        self.l1_loss = l1_loss
        self.dist_loss = dist_loss 
        self.extreme_loss = extreme_loss
        self.sobel_mse = sobel_mse
        #DP !!! load the colmap pointcloud
       # colmap_pc = np.load( os.path.join(self.opt.path, "xyz.npy"))

        self.console = Console()

        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        # optionally use LPIPS loss for patch-based training
        if self.opt.patch_size > 1:
            import lpips
            self.criterion_lpips = lpips.LPIPS(net='alex').to(self.device)

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4) # naive adam
        else:
            self.optimizer = optimizer(self.model)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1) # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [], # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
            }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)        
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)
            
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else: # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)
        
        # clip loss prepare
        if opt.rand_pose >= 0: # =0 means only using CLIP loss, >0 means a hybrid mode.
            from nerf.clip_utils import CLIPLoss
            self.clip_loss = CLIPLoss(self.device)
            self.clip_loss.prepare_text([self.opt.clip_text]) # only support one text prompt now...


    def __del__(self):
        if self.log_ptr: 
            self.log_ptr.close()


    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute: 
                #print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr: 
                print(*args, file=self.log_ptr)
                self.log_ptr.flush() # write immediately to file

    ### ------------------------------	




    def train_step(self, data):
        
        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]

        # if there is no gt image, we train with CLIP loss.
        if 'images' not in data:

            B, N = rays_o.shape[:2]
            H, W = data['H'], data['W']

            # currently fix white bg, MUST force all rays!
            outputs = self.model.render(rays_o,self.model.render, staged=False, bg_color=None, perturb=True, force_all_rays=True, **vars(self.opt))
            pred_rgb = outputs['image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()
            

            # [debug] uncomment to plot the images used in train_step
            #torch_vis_2d(pred_rgb[0])

            loss = self.clip_loss(pred_rgb)
            
            return pred_rgb, None, loss

        images = data['images'] # [B, N, 3/4]
        B, N, C = images.shape

        if self.opt.color_space == 'linear':
            images[..., :3] = srgb_to_linear(images[..., :3])

        if C == 3 or self.model.bg_radius > 0:
            bg_color = 1
        # train with random background color if not using a bg model and has alpha channel.
        else:
            #bg_color = torch.ones(3, device=self.device) # [3], fixed white background
            #bg_color = torch.rand(3, device=self.device) # [3], frame-wise random.
            bg_color = torch.rand_like(images[..., :3]) # [N, 3], pixel-wise random.

        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images

        outputs = self.model.render(rays_o, rays_d, staged=False, bg_color=bg_color, perturb=True, force_all_rays=False if self.opt.patch_size == 1 else True, **vars(self.opt))
        # outputs = self.model.render(rays_o, rays_d, staged=False, bg_color=bg_color, perturb=True, force_all_rays=True, **vars(self.opt))
    
        # DP !!! 
        #pred_density = self.density(pts.float())['sigma'].cpu().numpy()
        dens = outputs['density']
        pred_rgb = outputs['image']
        xyzs = outputs["points"]
        rays = outputs["rays"]
        deltas = outputs['deltas']#  [M, 2], all generated points' deltas. (first for RGB, second for Depth)
        
        '''reg_strength = 0.05

        meshgrid_points = meshgrid_points = self.DP_create_grid(self.opt.bound,101).to('cuda')
        density_outputs = self.model.density(meshgrid_points.float())['sigma']

        # Compute the L1 regularization term
        l1_regularization = reg_strength * torch.norm(density_outputs, p=1)/density_outputs.numel()
        loss += l1_regularization'''

        #print(xyz.shape)
        #DP !!! qua usa la loss!!!!!
        # MSE loss
        loss = self.criterion(pred_rgb, gt_rgb).mean(-1) # [B, N, 3] --> [B, N]
        loss = loss.mean() 
       # np_loss = loss.cpu().detach().numpy() 
        if self.sobel_mse:
            sobel_gt = sobel_operator(gt_rgb)
        # sobel_gt = -1*sobel_gt + 1
            sob_diff = sobel_gt.permute(0, 2, 1).expand(-1, -1, 3) * (gt_rgb - pred_rgb)
            l_sob  = torch.norm(sob_diff, dim=2).mean(-1)
            loss += l_sob.mean()# .item()
        #from raymarching function _march_rays_train
        # rays: int32, [N, 3], all rays' (index, point_offset, point_count), e.g., xyzs[rays[i, 1]:rays[i, 2]] --> points belonging to rays[i, 0]
       # extreme_loss = False
        if self.extreme_loss != 0 : 
            reg_strength = float(self.extreme_loss)
            extreme_loss = DP_extreme_loss(dens)
            loss += reg_strength * extreme_loss
       # l1_loss = True
        if self.l1_loss != 0 :
            reg_strength = float(self.l1_loss)
            #dens_ = torch.where(dens*deltas[:,0] < 4.5 , torch.tensor(0), torch.tensor(1)).float()

            # Compute the L1 regularization term
            l1_regularization = reg_strength * torch.norm(dens * deltas[:,0] , p=1)/ (rays.shape[0] -1 )
            loss += l1_regularization

        """if dist_loss != "0" :
            reg_strength = float(self.extreme_loss)
            meshgrid_points = meshgrid_points = self.DP_create_grid(self.opt.bound,self.opt.TV_density_dim).to('cuda')
            density_outputs = self.model.density(meshgrid_points.float())['sigma']
            loss_dist = loss_distortion(meshgrid_points ,density_outputs)
            #loss_dens = 0.01 * isolated_points_loss(dens)
            #print(loss_dist)
            loss += reg_strength *  loss_dist #DP !!! uncomment to add new terms to the loss"""
        # extra loss
        # pred_weights_sum = outputs['weights_sum'] + 1e-8
        # loss_ws = - 1e-1 * pred_weights_sum * torch.log(pred_weights_sum) # entropy to encourage weights_sum to be 0 or 1.
        # loss += loss_ws.mean()

        # patch-based rendering
        if self.opt.patch_size > 1:
            gt_rgb = gt_rgb.view(-1, self.opt.patch_size, self.opt.patch_size, 3).permute(0, 3, 1, 2).contiguous()
            pred_rgb = pred_rgb.view(-1, self.opt.patch_size, self.opt.patch_size, 3).permute(0, 3, 1, 2).contiguous()

            # torch_vis_2d(gt_rgb[0])
            # torch_vis_2d(pred_rgb[0])

            # LPIPS loss [not useful...]
            loss +=  1e-3 * self.criterion_lpips(pred_rgb, gt_rgb)

        # special case for CCNeRF's rank-residual training
        if len(loss.shape) == 3: # [K, B, N]
            loss = loss.mean(0)

        # update error_map
        if self.error_map is not None:
            index = data['index'] # [B]
            inds = data['inds_coarse'] # [B, N]

            # take out, this is an advanced indexing and the copy is unavoidable.
            error_map = self.error_map[index] # [B, H * W]

            # [debug] uncomment to save and visualize error map
            # if self.global_step % 1001 == 0:
            #     tmp = error_map[0].view(128, 128).cpu().numpy()
            #     print(f'[write error map] {tmp.shape} {tmp.min()} ~ {tmp.max()}')
            #     tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())
            #     cv2.imwrite(os.path.join(self.workspace, f'{self.global_step}.jpg'), (tmp * 255).astype(np.uint8))

            error = loss.detach().to(error_map.device) # [B, N], already in [0, 1]
            
            # ema update
            ema_error = 0.1 * error_map.gather(1, inds) + 0.9 * error
            error_map.scatter_(1, inds, ema_error)

            # put back
            self.error_map[index] = error_map

        
        #DP !!! mse + total_variation loss
            
        if self.TV_lambda != 0 :   
          #  meshgrid_points = self.DP_create_grid(self.opt.bound,self.opt.TV_density_dim).to('cuda')
          #  density_outputs = self.model.density(meshgrid_points.float())['sigma']
          #  pc_ind = torch.where(density_outputs > self.opt.mesh_thresh) 
           # nerf_pc = meshgrid_points[pc_ind] 
            total_var = loss_TV_1D(dens , float(self.TV_lambda),float(self.Huber_eps) )
           # total_var = loss_TV(density_outputs , float(self.TV_lambda),float(self.Huber_eps) )# DP !!! original
           # binary_mesh = set_1 = torch.where(density_outputs > .5,float(1),float( 0))
          #  total_var = loss_TV(binary_mesh , float(self.TV_lambda),float(self.Huber_eps) )# DP !!! original
           #total_var_rec = loss_TV(1/(.001+density_outputs ), float(self.TV_lambda),float(self.Huber_eps))
            #loss_dens = 0.01 * isolated_points_loss(dens)
            loss += total_var # keep it a tensor: .item() would detach it and give zero gradient

        
        if self.MS_lambda != 0 :
            # loss +=  loss_MGS(dens,deltas[:,0] ,1e-3,1e-2 )
            loss +=  loss_MS(dens,deltas[:,0] ,float(self.MS_lambda),float(self.Huber_eps))# .item()

        #DP !!! set False use minium support over a uniform sampled grid of the scene
        if False:
            meshgrid_points = self.DP_create_grid(self.opt.bound,self.opt.TV_density_dim).to('cuda')
            density_outputs = self.model.density(meshgrid_points.float())['sigma']
            #loss += loss_TV(density_outputs , 1e-2,float(self.Huber_eps)).item()
            loss += loss_MS_grid(density_outputs , 1e-2,float(self.Huber_eps))#.item()

        chamfer_loss = False

        if chamfer_loss:
            print("CHAMFERRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR")
            meshgrid_points = self.DP_create_grid(self.opt.bound,256).to('cuda')
            density_outputs = self.model.density(meshgrid_points.float())['sigma']
            pc_ind = torch.where(density_outputs >  self.opt.mesh_thresh)


            colmap_pc_nmpy = np.load( os.path.join(self.opt.path, "xyz.npy"))
            # Apply the mask to the original arraynerf_pc
            
            colmap_pc = torch.tensor(colmap_pc_nmpy).to('cuda')
            colmap_num_total_rows = colmap_pc.size(0)
            colmap_num_rows_to_select = int(colmap_num_total_rows * .01)
            colmap_indices = torch.randperm(colmap_num_total_rows)[:colmap_num_rows_to_select]
            colmap_pc_perc = colmap_pc[colmap_indices] 
            

            nerf_pc = torch.tensor(meshgrid_points[pc_ind]).to('cuda')
            '''print(nerf_pc.shape)'''
            num_total_rows = nerf_pc.size(0)
            num_rows_to_select = int(num_total_rows * .0001)
            indices = torch.randperm(num_total_rows)[:num_rows_to_select]
    
            # Select rows using the generated indices
            nerf_pc_perc = nerf_pc[indices]

           # print(colmap_pc_nmpy[-1,:])
            has_nan = torch.isnan(colmap_pc).any()

            # Check for inf values
            has_inf = torch.isinf(colmap_pc).any() 
           # print(has_nan,has_inf)
            has_nan = torch.isnan(nerf_pc).any()

            # Check for inf values
            has_inf = torch.isinf(nerf_pc).any() 
           # print(has_nan,has_inf)
            '''
            print(colmap_pc.shape)
            print(nerf_pc_perc.shape)
            print(colmap_pc_perc.shape)
            '''
            loss_ch = loss_chamfer(nerf_pc_perc,colmap_pc)#colmap_pc_perc 
           # print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
           # print(loss_ch )
            loss +=  loss_ch.item()
            l1_loss = False
            if l1_loss:
                count_loss = num_total_rows/256**3
                loss += 100* count_loss.item()
               # print(count_loss)
        
        if self.dist_loss != 0 :
            nears = outputs['nears']
            fars = outputs['fars']
            num_rows = nears.shape[0]

            # Calculate the number of rows to select (10 percent)
            num_rows_to_select = int(0.10 * num_rows)
            # Generate random indices to select rows
            random_indices = np.random.choice(num_rows, num_rows_to_select, replace=False)

            n = 512
            #meshgrid_points = meshgrid_points = self.DP_create_grid(self.opt.bound,self.opt.TV_density_dim).to('cuda')
            rays = DP_create_rays(rays_o[0,random_indices,:], rays_d[0,random_indices,:], nears[random_indices], fars[random_indices], n=n )
            density_outputs = self.model.density(rays.float())['sigma']
            #diff = rays[0,:, 1:] - rays[0,:, :-1]
            #diff = torch.cumsum(diff, dim=1)
            
            # Calculate the norm of the difference for each row along the second dimension
            #t = torch.norm(rays[0,:, :], dim=2)
            t = torch.linspace(0, 1, n).cuda()
            
            t = t.unsqueeze(0).repeat(num_rows_to_select, 1) 
            w = DP_get_weight(density_outputs[0,:,:-1], rays[0,:] )
            #dist_loss = self.DP_loss_distortion(t, w)
            dist_loss_val = DP_loss_dist(t,w, float(self.dist_loss))
            loss += dist_loss_val # keep it a tensor: .item() would detach it and give zero gradient
        # extra loss
        # pred_weights_sum = outputs['weights_sum'] + 1e-8
        # loss_ws = - 1e-1 * pred_weights_sum * torch.log(pred_weights_sum) # entropy to encourage weights_sum to be 0 or 1.
        # loss += loss_ws.mean()

        '''reg_strength = 0.05

        meshgrid_points = meshgrid_points = self.DP_create_grid(self.opt.bound,101).to('cuda')
        density_outputs = self.model.density(meshgrid_points.float())['sigma']

        # Compute the L1 regularization term
        l1_regularization = reg_strength * torch.norm(density_outputs, p=1)/density_outputs.numel()
        loss += l1_regularization'''
        return pred_rgb, gt_rgb, loss

    #DP !!!
    def DP_create_grid(self,b, n):
            # Create a 1D tensor of n elements ranging from -b to b
            coords = torch.linspace(-b, b, n)

            # Use torch.meshgrid to create a 3D tensor where each element is a tuple of the 3D coordinates
            x, y, z = torch.meshgrid(coords, coords, coords)

            # Stack the x, y, and z tensors along a new dimension to create a 4D tensor of shape (n, n, n, 3)
            points = torch.stack([x, y, z], dim=3)


            return points
    def DP_create_rand_grid(self, b, n):
            # Generate (n^3) random points in 3D space within [-b, b]
            points = (2 * b) * torch.rand(n**3, 3) - b

            # Optionally reshape into a 3D grid shape (n, n, n, 3) if needed
            points = points.view(n, n, n, 3)

            return points 
        
    def eval_step(self, data):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        images = data['images'] # [B, H, W, 3/4]
        B, H, W, C = images.shape

        if self.opt.color_space == 'linear':
            images[..., :3] = srgb_to_linear(images[..., :3])

        # eval with fixed background color
        bg_color = 1
        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images
        #print("Before self.model.render() call")
        self.opt.DP_val = False
        outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=False, **vars(self.opt))#original 
        #outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color,force_all_rays=False if self.opt.patch_size == 1 else True, perturb=False, **vars(self.opt)) 
        #outputs = self.model.render(rays_o, rays_d, staged=False, bg_color=bg_color, perturb=True, force_all_rays=False if self.opt.patch_size == 1 else True, **vars(self.opt))
        self.opt.training = True
        #o_2 = self.model.render(rays_o, rays_d, staged=False, bg_color=bg_color, perturb=True, force_all_rays=False if self.opt.patch_size == 1 else True, **vars(self.opt))
        self.opt.training = False 
        self.opt.DP_val = False
        #print(outputs)
        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W)
        loss = self.criterion(pred_rgb, gt_rgb).mean()
        
        

        #DP !!! 
        
        dens = outputs['density']
        deltas = outputs['deltas']
        rays = outputs['rays']
        """
       # extreme_loss = False
        if extreme_loss: 
            extreme_loss = DP_extreme_loss(dens)
            loss += extreme_loss
        l1_loss = True
        if l1_loss != "0": 
            reg_strength = float(self.l1_loss)
        #dens_ = torch.where(dens < 2.5 , torch.tensor(0), torch.tensor(1)).float()
            l1_regularization = reg_strength * torch.norm(dens/5 * deltas[:,0] , p=1)/ (rays.shape[0] -1 )

            loss += l1_regularization"""

        return pred_rgb, pred_depth, gt_rgb, loss

    # moved out bg_color and perturb for more flexible control...
    def test_step(self, data, bg_color=None, perturb=False,**kwargs):  

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        H, W = data['H'], data['W']

        if bg_color is not None:
            bg_color = bg_color.to(self.device)

        outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=perturb, **vars(self.opt))

        #DP !!! if get density i have different outputs : 
        #if kwargs['get_density']: 
        #    return outputs
        
        pred_rgb = outputs['image'].reshape(-1, H, W, 3)
        pred_depth = outputs['depth'].reshape(-1, H, W)

        return pred_rgb, pred_depth

    def DP_distance_imgs(self,loader, save_path=None,file_name = None, n_samples =512,ray_start = 0,threshold =9):
        def DP_first_point(arr, threshold,n_samples ):
            # Create a boolean mask where the condition is met
            mask = arr > threshold
            
            # Use np.argmax on the masked array
            # np.argmax returns the first occurrence of the maximum value
            first_indices = np.argmax(mask, axis=1)
            
            # If no values exceed the threshold in a row, np.argmax will return 0 (the first index)
            # Check if the first value of the mask is False to identify these cases
            first_indices[~mask.any(axis=1)] = n_samples-1 # or some other value to indicate no match found
            
            return first_indices
        if save_path is None:
            save_path = os.path.join(self.workspace, 'dist_imgs')

        name = f'{file_name}_{self.epoch:04d}.png'

        os.makedirs(save_path, exist_ok=True)
        self.log(f"==> Saving distance images to {save_path}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        deltas = torch.linspace(ray_start, 4*np.sqrt(3), n_samples).unsqueeze(1).cuda()

        with torch.no_grad():
            pbar = tqdm.tqdm(total=10 )
            
            for ind, data in enumerate(loader):  
                if ind %10 ==0  :
                    dists =[] 
                    rgbs=[]
                    rays_o = data['rays_o'] # [B, N, 3]
                    rays_d = data['rays_d'] # [B, N, 3]

                    # Generate random indices between 0 and N-1     v
                # num_indices = 60000  # Number of random indices to select
                # random_indices = np.random.choice(rays_o.shape[1] , num_indices, replace=False)

                    # Convert indices to torch tensor
                # random_indices_tensor = torch.tensor(random_indices)

                    # Select elements along the second dimension using the random indices
                  #  rays_o = rays_o[:, :, :]
                   # rays_d = rays_d[:, :, :]
                    H = data["H"]  
                    W = data["W"]  
                    B, N, _ = rays_d.shape
                    part = 10
                    part_size = N // part  # Size of each part
                    for i in range(part):
                        start_idx = i * part_size
                        end_idx = (i + 1) * part_size

                        # Handle the case where N is not perfectly divisible by 10
                        if i == part-1:  # Ensure we cover all elements in the last split
                            end_idx = N

                        rays_o_split = rays_o[:, start_idx:end_idx, :]  # Shape: [B, part_size, 3]
                        rays_d_split = rays_d[:, start_idx:end_idx, :]  # Shape: [B, part_size, 3]

                        rays_o_expanded = rays_o_split[0, :, :].unsqueeze(1)  # Shape: (part_size, 1, 3)
                        rays_d_expanded = rays_d_split[0, :, :].unsqueeze(1)  # Shape: (part_size, 1, 3)

                        # Calculate the points along each ray
                        points = rays_o_expanded + deltas * rays_d_expanded
                    #s print(points.shape)
                        density = self.model.density(points)
                        dens = density['sigma']
                       # geo_feat = density['geo_feat'].detach().cpu().numpy()

                        indices = DP_first_point(dens.cpu().numpy(),threshold,n_samples)

                        # Exclude points with value -1 from indices
                        valid_indices = indices           
                        d = np.array([ 0, -1, 0])
                        ds = np.tile(d, (valid_indices.shape[0],1))
                        #sigmas__, rgbs = self.model(vertices, torch.from_numpy(ds).cuda().float())
                        ds = torch.tensor(ds).to(dtype=torch.float32, device='cuda')

                        vertices = torch.tensor(points[torch.arange(valid_indices.shape[0] ), valid_indices])

    #                    directions = torch.tensor(rays_d_expanded[torch.arange(valid_indices.shape[0] ), valid_indices])
                        geo_feat = self.model.density(vertices.to(self.device).float())['geo_feat'].detach().cpu().numpy()
                        geo_feat = torch.tensor(geo_feat).to(dtype=torch.float32, device='cuda')
                        rgb = self.model.color(vertices.to(self.device).float(),geo_feat = geo_feat.to(self.device).float())
                        dists.append(deltas[valid_indices])
                        rgbs.append(rgb)
                        # Append the points corresponding to valid_indices to xyzs
                    # xyzs.append(points[torch.arange(valid_indices.shape[0] ) , valid_indices])
                    
                    pbar.update(1)
                    dists = torch.cat(dists, dim=0)# dists = torch.stack(dists)
                    dists = dists.cpu().numpy()/(3-ray_start)*255
                    cv2.imwrite(os.path.join(save_path, f'{ind}_{i:04d}_dists.png'),dists.reshape(H, W))
                    
                    rgbs = torch.cat(rgbs, dim=0)
                    rgbs = (rgbs.cpu().numpy()*255).astype(np.uint8)
                    cv2.imwrite(os.path.join(save_path, f'{ind}_{i:04d}_rgb.png'), cv2.cvtColor(rgbs.reshape(H, W,3), cv2.COLOR_RGB2BGR))


    def DP_get_histo(self):
        def query_func(pts):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    sigma = self.model.density(pts.to(self.device))['sigma']
            return sigma

        save_path = os.path.join(self.workspace, 'graph', "mesh_TV_" + str(self.TV_lambda) +"_" + f'{self.epoch}.png') #DP !!!

        self.log(f"==> Saving histo to {save_path}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        density_outputs = extract_fields(self.model.aabb_infer[:3], self.model.aabb_infer[3:], resolution=512, query_func=query_func)
         
        density_outputs = density_outputs.ravel()

        hist, bins = np.histogram(density_outputs, bins=1000)
        plt.hist(density_outputs, bins=bins, alpha=1)
        plt.title('Histogram of density')
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.yscale('log')
        plt.savefig(save_path.replace(".png","all.png"))
        plt.clf()
        behind10 = density_outputs[np.where(density_outputs<10)] 
        hist, bins = np.histogram(behind10, bins=1000)
        plt.hist(behind10, bins=bins, alpha=1)
        plt.title('Histogram of density')
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.yscale('log')
        plt.savefig(save_path.replace(".png","<10.png"))
        plt.clf()
        print(np.max(density_outputs))
        above10 = density_outputs[np.where(10<=density_outputs ) ]#and np.where(density_outputs <100) 
        hist, bins = np.histogram(above10, bins=1000)
        plt.hist(above10, bins=bins, alpha=1)
        plt.title('Histogram of density')
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.yscale('log')
        plt.savefig(save_path.replace(".png",">10.png"))

    def DP_plot_density_rays(self,loader, n_samples =1024,ray_start = 0, dens_clip_value= 2):
        random_indeces_path= os.path.join( os.path.dirname(self.workspace), 'random_indeces_rays.npy')
        save_dir = os.path.join(self.workspace, 'graph') 
        save_path = os.path.join(self.workspace, 'graph', "density_plot" + str(self.TV_lambda) +"_" + f'{self.epoch}.png') #DP !!!

        xyzs = []
        dens =[] 
        num_indices = 10
        self.log(f"==> Saving density plots to {save_path}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        deltas = torch.linspace(0, 3*np.sqrt(3), n_samples).unsqueeze(1).cuda()
        
        
        print(random_indeces_path) 
        # Check if file exists
        if os.path.exists(random_indeces_path):
            print(f"Loading existing indices from {random_indeces_path}")
            all_rand_indices = np.load(random_indeces_path)
        else:
            print(f"No indices found at {random_indeces_path}, creating new ones")
            rand_indices_list = []
            with torch.no_grad():
                for ind, data in enumerate(loader): 
                    rand_indices_load = np.random.choice(data['rays_o'].shape[1], num_indices, replace=False)
                    rand_indices_list.append(rand_indices_load)
                    
            all_rand_indices = np.stack(rand_indices_list, axis=0)  # shape: (num_batches, num_indices)
            
            np.save(random_indeces_path, all_rand_indices)
            print(f"Saved new indices to {random_indeces_path}")
                    
        with torch.no_grad():
            for ind, data in enumerate(loader): 
                 
                rand_indices = all_rand_indices[ind,...]
                
                rays_o = data['rays_o'][:,rand_indices,:]  # [B, N, 3]
                rays_d = data['rays_d'][:,rand_indices,:]  # [B, N, 3]        

                rays_o_expanded = rays_o[0, :, :].unsqueeze(1)  # Shape: (part_size, 1, 3)
                rays_d_expanded = rays_d[0, :, :].unsqueeze(1)  # Shape: (part_size, 1, 3)

                # Calculate the points along each ray
                points = rays_o_expanded + deltas * rays_d_expanded
                
                density_outputs = self.model.density(points)['sigma']# .detach().cpu().numpy()
                dens.append(density_outputs)
                
                if ind == 0 :
                    rays_o_all = data['rays_o']
                    rays_d_all = data['rays_d']  
                    selected_indices = torch.randperm(rays_d_all.shape[1])[:10000]
                    rays_d_all_sampled = rays_d_all[0, selected_indices, :]  # shape: (10000, 3)

                    print(rays_d_all_sampled.shape)

                    # Step 1: Compute mean direction (camera direction)
                    mean_direction = rays_d_all_sampled.mean(dim=0)
                    mean_direction = mean_direction / mean_direction.norm()  # Normalize
                    print(mean_direction)

                    # Step 2: Compute deviations from mean direction (dot product)
                    dot_products = torch.matmul(rays_d_all_sampled, mean_direction)  # shape: (10000,)

                    # Step 3: Find min and max deviations
                    min_deviation = dot_products.min()
                    max_deviation = dot_products.max()

                    # Step 4: Select rays within ±1/4 of max deviation
                    selection_range = max_deviation / 25
                    selected_rays_indices = torch.where(
                        (dot_products >= (max_deviation - selection_range)) &
                        (dot_products <= (max_deviation + selection_range))
                    )[0]

                    # Step 5: Extract selected rays
                    selected_rays = rays_d_all_sampled[selected_rays_indices].unsqueeze(1)

                    # Optionally: Visualize the selected rays, if desired
                    print(f"Selected rays shape: {selected_rays.shape}")
                    rays_o_all_expanded = rays_o_all[0, selected_rays_indices, :].unsqueeze(1)  # Shape: (part_size, 1, 3)

                    # Calculate the points along each ray
                    points_all = rays_o_all_expanded + deltas * selected_rays
                    
                    density_outputs_all = self.model.density(points_all)['sigma'].detach().cpu().numpy() 

       # dens = torch.cat([tensor.view(-1, 2) for tensor in dens], dim=0)
        dens = torch.cat(dens, dim=0).cpu().numpy() 
        x_points = deltas.cpu().numpy()
        np_save_path = os.path.join(self.workspace, 'graph')
        
        np.save(np_save_path+ "/points", x_points)
        np.save(np_save_path + "/densities", dens)
        
        np.save(np_save_path+ "/points_all", points_all.cpu())
        np.save(np_save_path + "/densities_all", density_outputs_all)
        print(np_save_path)
        
        for i in range(dens.shape[0]):
            plt.plot(x_points, dens[i, :], alpha=0.5)

        # Add labels and title
        
        plt.xlabel('distance')
        plt.ylabel('density value')
        plt.title('density rays')
        plt.axvline(x=ray_start)
        plt.savefig(save_path)
        
        plt.yscale('log')
        plt.xlabel('distance')
        plt.ylabel('log-density value')
        plt.title('log-density rays')
        plt.axvline(x=ray_start)
        plt.savefig(save_path.replace(".png","_log.png"))
        plt.clf()

        dens_clip = np.clip(dens ,a_min = None,  a_max=dens_clip_value )
        for i in range(dens_clip.shape[0]):
            plt.plot(x_points, dens_clip[i, :], alpha=0.25)
                
        plt.xlabel('distance')
        plt.ylabel('density value')
        plt.title('density rays')
        plt.axvline(x=ray_start)
        plt.savefig(save_path.replace(".png","_clamped.png"))
        plt.clf()
        interval = dens.shape[0]//5
        rand_dens = dens[ ::interval,:]
        for dens_ray in rand_dens:
            plt.plot(x_points, dens_ray, linewidth=0.5)

        plt.yscale('log')
        plt.xlabel('distance')
        plt.ylabel('log-density value')
        plt.title('log-density rays')
        plt.axvline(x=ray_start)
        plt.savefig(save_path.replace(".png","log_few_rays.png"))


        rand_dens_clip = np.clip(rand_dens ,a_min = None,  a_max=dens_clip_value )
        plt.figure(figsize=(10, 6))
        for dens_ray in rand_dens_clip:
            plt.plot(x_points, dens_ray, linewidth=0.5)

        plt.xlabel('distance')
        plt.ylabel('density value')
        plt.title('density rays clamped')
        plt.axvline(x=ray_start)
        plt.savefig(save_dir + f'/density_plot_clamped_few_rays_{str(dens_clip_value)}.png')

        plt.yscale('log')
        plt.xlabel('distance')
        plt.ylabel('log-density value')
        plt.title('log-density rays')
        plt.axvline(x=ray_start)
        plt.savefig(save_dir + f'/density_plot_clamped_log_few_rays_{str(dens_clip_value)}.png')
        plt.clf()
        
               # xyzs.append(points[torch.arange(valid_indices.shape[0] ) , valid_indices])

    def save_mesh_boundary(self,loader, save_path=None,file_name = None, threshold=9, threshold_up=np.inf,n_samples = 1024, ray_start = 0):

        def DP_first_point(arr, threshold,threshold_up):
            
            # Create a boolean mask where the condition is met
            mask = (arr > threshold) & (arr < threshold_up)

            # Use np.argmax on the masked array
            # np.argmax returns the first occurrence of the maximum value
            first_indices = np.argmax(mask, axis=1)
            
            # If no values exceed the threshold in a row, np.argmax will return 0 (the first index)
            # Check if the first value of the mask is False to identify these cases
            first_indices[~mask.any(axis=1)] = -1  # or some other value to indicate no match found
            

            return first_indices
        if save_path is None:
            save_path = os.path.join(self.workspace, 'meshes')

        name = f'{file_name}boundary_ep{self.epoch:04d}.ply'

        os.makedirs(save_path, exist_ok=True)
        self.log(f"==> Saving boundary mesh to {save_path}")
        xyzs = []
        rgbs=[] 
        dens =[] 
        deltas = torch.linspace(ray_start, 4*np.sqrt(3), n_samples).unsqueeze(1).cuda()
        with torch.no_grad():
            pbar = tqdm.tqdm(total=len(loader) )
            for ind, data in enumerate(loader):  
                
                rays_o = data['rays_o'] # [B, N, 3]
                rays_d = data['rays_d'] # [B, N, 3]
                 
                # Generate random indices between 0 and N-1
                num_indices = 60000  # Number of random indices to select
                random_indices = np.random.choice(rays_o.shape[1] , num_indices, replace=False)

                # Convert indices to torch tensor
                random_indices_tensor = torch.tensor(random_indices)

                # Select elements along the second dimension using the random indices
                rays_o = rays_o[:, random_indices_tensor, :]
                rays_d = rays_d[:, random_indices_tensor, :]
                
                B, N, _ = rays_d.shape
                part = 20
                part_size = N // part  # Size of each part


                

                for i in range(part):
                    try :
                            
                        start_idx = i * part_size
                        end_idx = (i + 1) * part_size

                        # Handle the case where N is not perfectly divisible by 10
                        if i == part-1:  # Ensure we cover all elements in the last split
                            end_idx = N

                        rays_o_split = rays_o[:, start_idx:end_idx, :]  # Shape: [B, part_size, 3]
                        rays_d_split = rays_d[:, start_idx:end_idx, :]  # Shape: [B, part_size, 3]

                        rays_o_expanded = rays_o_split[0, :, :].unsqueeze(1)  # Shape: (part_size, 1, 3)
                        rays_d_expanded = rays_d_split[0, :, :].unsqueeze(1)  # Shape: (part_size, 1, 3)

                        # Calculate the points along each ray
                        points = rays_o_expanded + deltas * rays_d_expanded
                    #s print(points.shape)
                        density = self.model.density(points)
                        dens = density['sigma']
                    #  geo_feat = density['geo_feat'].detach().cpu().numpy()
                        
                        indices = DP_first_point(dens.cpu().numpy(),threshold,threshold_up)
                        non_none_indices = [i for i, index in enumerate(indices) if index>=0]
                        filtered_indices = [index for index in indices if index>= 0 ]
                        # Exclude points with value -1 from indices
                        valid_indices = indices# [indices != 0]

                        d = np.array([ 0, -1, 0])
                        '''
                        ds = np.tile(d, (valid_indices.shape[0],1))
                        #sigmas__, rgbs = self.model(vertices, torch.from_numpy(ds).cuda().float())
                        ds = torch.tensor(ds).to(dtype=torch.float32, device='cuda')'''
                        vertices = torch.tensor(points[non_none_indices, filtered_indices])
                    
    #                    directions = torch.tensor(rays_d_expanded[torch.arange(valid_indices.shape[0] ), valid_indices])
                        geo_feat = self.model.density(vertices.to(self.device).float())['geo_feat'].detach().cpu().numpy()
                        geo_feat = torch.tensor(geo_feat).to(dtype=torch.float32, device='cuda')
                        rgb = self.model.color(vertices.to(self.device).float(),geo_feat = geo_feat.to(self.device).float())
                        
                        # Append the points corresponding to valid_indices to xyzs
                        xyzs.append(vertices)
                        rgbs.append(rgb)
                    except RuntimeError as e:
                        print(f"RuntimeError occurred in iteration {i}: {e}")
                        # Optionally log the error to a file or take other actions
                        continue                         
                pbar.update(1)

        xyzs = torch.cat([tensor.view(-1, 3) for tensor in xyzs], dim=0)
        rgbs = torch.cat([tensor.view(-1, 3) for tensor in rgbs], dim=0)
       #s print(torch.cat(xyzs, dim=0))
        mesh = trimesh.Trimesh(vertices=xyzs.cpu(),vertex_colors = rgbs.cpu())
        mesh.export(save_path+"/"+name)
        return save_path+"/"+name
    
    def DP_save_plane(self,loader, save_path=None,file_name = None, threshold=9, threshold_up=np.inf,n_samples = 1024, z_val =0.0):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'meshes')

        name = f'{file_name}plane_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)
        self.log(f"==> Saving boundary mesh to {save_path}")
        xyzs = []
        rgbs=[] 
        dens =[] 
        x = torch.linspace(self.model.aabb_infer[0], self.model.aabb_infer[3], n_samples).cuda()
        y = torch.linspace(self.model.aabb_infer[1], self.model.aabb_infer[4], n_samples).cuda()
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        zz = torch.full_like(xx, z_val)
        pts = torch.stack([xx,zz, yy ], dim=-1).reshape(-1, 3) 
        deltas = torch.linspace(self.model.aabb_infer[0], self.model.aabb_infer[4], n_samples).unsqueeze(1).cuda()
        with torch.no_grad():
            
            density = self.model.density(pts)
            dens = density['sigma']
            mask = dens > threshold
            xyzs = pts[mask]
            #filtered_densities = dens[mask]
            geo_feat = self.model.density(xyzs.to(self.device).float())['geo_feat'].detach().cpu().numpy()
            geo_feat = torch.tensor(geo_feat).to(dtype=torch.float32, device='cuda')
                    
            rgbs = self.model.color(xyzs.to(self.device).float(),geo_feat = geo_feat.to(self.device).float())    
                                           
                
        xyzs = torch.cat([tensor.view(-1, 3) for tensor in xyzs], dim=0)
        rgbs = torch.cat([tensor.view(-1, 3) for tensor in rgbs], dim=0)
       #s print(torch.cat(xyzs, dim=0))
       
        np.savez_compressed(os.path.join(save_path, name), xyz=xyzs.detach().cpu().numpy(), rgb=rgbs.detach().cpu().numpy())
        mesh = trimesh.Trimesh(vertices=xyzs.cpu(),vertex_colors = rgbs.cpu())
        mesh.export(save_path+"/"+name+ ".ply")
        
    def save_mesh(self, save_path=None, resolution=256, threshold=10,v1=None,v2=None,v3=None):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'meshes', "mesh_TV_" + str(self.TV_lambda) +"_" + f'{self.epoch}.ply') #DP !!!

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        def query_func(pts):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    sigma = self.model.density(pts.to(self.device))['sigma']
                    #DP !!!
                    '''
                    d = np.array([-7.992268800735473633e-01, 1.765568703413009644e-01, 5.745120048522949219e-01])
                #[  -4.127612709999084473e-01, 7.848503589630126953e-01, -4.622099697589874268e-01]) # y il cielo x 360? 
                    ds = np.tile(d, (pts.shape[0],pts.shape[1],pts.shape[2],1))
                    sigmas__, rgbs = self(pts.to(self.device), torch.from_numpy(ds).cuda().float())'''
            return sigma
        #DP !!!
        def DP_color_func(vertices):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    d = np.array([ 0, -1, 0])
                    ds = np.tile(d, (vertices.shape[0],1))
                    #sigmas__, rgbs = self.model(vertices, torch.from_numpy(ds).cuda().float())
                    ds = torch.tensor(ds).to(dtype=torch.float32, device='cuda')

                    vertices = torch.tensor(vertices)
                    geo_feat = self.model.density(vertices.to(self.device).float())['geo_feat'].detach().cpu().numpy()
                    geo_feat = torch.tensor(geo_feat).to(dtype=torch.float32, device='cuda')
                    rgbs = self.model.color(vertices.to(self.device).float(),geo_feat = geo_feat.to(self.device).float())
                    #colors = rgbs['image'].cpu().numpy()
                    colors = rgbs.cpu().numpy() 
                    #sigmas__, colors = self.model(vertices,ds)
            return colors
        def DP_color_func_2(vertices, dirs):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    #sigmas__, rgbs = self.model(vertices, torch.from_numpy(ds).cuda().float())
                    ds = torch.tensor(dirs)

            # Use the to() method to convert the data type or move to a different device
                    ds = ds.to(dtype=torch.float32, device='cuda')
                    vertices_ = torch.tensor(vertices)
            # Use the to() method to convert the data type or move to a different device
                    vertices_ = vertices_.to(dtype=torch.float32, device='cuda')
                    
                    rgbs = self.model.DP_render(vertices_, ds, staged=False, perturb=True, force_all_rays=False if self.opt.patch_size == 1 else True, **vars(self.opt))
                    colors = rgbs['image'].cpu().numpy()
            return colors
        '''
        mask = np.logical_and(sigma > threshold)
        rgb_ = rgb[mask,:]
        '''
        
        if self.opt.mesh_principal:
            vertices, triangles = extract_geometry(self.model.aabb_infer[:3], self.model.aabb_infer[3:], resolution=512, threshold=threshold, query_func=query_func)
        elif self.opt.mesh_partial: 
            area_bound = np.array([[-2,2],[-.9,.5],[-.7,.1]])
            vertices, triangles = extract_geometry(self.model.aabb_infer[:3], self.model.aabb_infer[3:], resolution=resolution, threshold=threshold, query_func=query_func,DP_rotation=True, v1=v1,v2=v2,v3=v3,b=area_bound)
        else : 
            vertices, triangles = extract_geometry(self.model.aabb_infer[:3], self.model.aabb_infer[3:], resolution=resolution, threshold=threshold, query_func=query_func)
        
        
        
        #DP !!!

        '''U = vertices[triangles[:, 1]] - vertices[triangles[:, 0]]
        V = vertices[triangles[:, 2]] - vertices[triangles[:, 0]]
        Norm = np.cross(U, V)
        norms = np.sqrt(np.sum(Norm ** 2, axis=1))
        Normal = (Norm.T / norms).T

        # Calculate barycentric coordinates for each triangle
        A = vertices[triangles[:, 0]]
        B = vertices[triangles[:, 1]]
        C = vertices[triangles[:, 2]]
        P = (A + B + C) / 3.0
        alpha = np.sum(np.cross((B - C), (P - C)) * Normal, axis=1) / np.sum(Normal * Normal, axis=1)
        beta = np.sum(np.cross((C - A), (P - A)) * Normal, axis=1) / np.sum(Normal * Normal, axis=1)
        gamma = 1 - alpha - beta

        # Combine normal vectors and barycentric coordinates into a single output array
        baricentrics = np.hstack(( alpha[:, None], beta[:, None], gamma[:, None]))
        '''


        #b_colors = DP_get_triangle_colors_2(-Normal,baricentrics,DP_color_func_2 )
        
        #b_colors[np.logical_and(b_colors[:,0]>.95,b_colors[:,1]>.95,b_colors[:,2]>.95,)] = [1,1,0,0]
        
        
        v_colors, t_colors = DP_get_triangle_colors(vertices, triangles, DP_color_func)
        
        #mesh = trimesh.Trimesh(vertices=vertices, process=False, vertex_colors= v_colors) # vertex_colors= v_colors,  important, process=True leads to seg fault...
        mesh = trimesh.Trimesh(vertices=vertices, faces= triangles, process=False, vertex_colors= v_colors,  face_colors=t_colors) # vertex_colors= v_colors,  important, process=True leads to seg fault...
        #mesh = trimesh.Trimesh(vertices=vertices,  process=False, vertex_colors= v_colors) # vertex_colors= v_colors,  important, process=True leads to seg fault...

        if self.opt.mesh_principal:
            # Find connected components
            components = mesh.split()

            # Find the largest connected component
            largest_component = max(components, key=lambda x: len(x.vertices))

            # Extract vertices and faces of the largest component
            largest_vertices = largest_component.vertices
            largest_faces = largest_component.faces

            # Create a new mesh from the largest component
            largest_mesh = trimesh.Trimesh(vertices=largest_vertices, faces=largest_faces)

           # largest_mesh.export(save_path.replace(".ply","largest_cc.ply"))
            # Compute bounding box of the largest component
            bbox_min, bbox_max = largest_mesh.bounds
            vertices, triangles = extract_geometry(torch.tensor(bbox_min), torch.tensor(bbox_max), resolution=resolution, threshold=threshold, query_func=query_func)
            v_colors, t_colors = DP_get_triangle_colors(vertices, triangles, DP_color_func)
            mesh = trimesh.Trimesh(vertices=vertices, faces= triangles, process=False, vertex_colors= v_colors,  face_colors=t_colors) # vertex_colors= v_colors,  important, process=True leads to seg fault...
            save_path = save_path.replace(".ply","_largest_cc.ply")

        np.savetxt('vertices.txt',vertices) 
        np.savetxt('t_colors.txt',t_colors)
        mesh.export(save_path)

        self.log(f"==> Finished saving mesh.")
        return save_path

    ### ------------------------------

    def train(self, train_loader, valid_loader, max_epochs):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        # mark untrained region (i.e., not covered by any camera from the training dataset)
        if self.model.cuda_ray:
            self.model.mark_untrained_grid(train_loader._data.poses, train_loader._data.intrinsics)

        # get a ref to error_map
        self.error_map = train_loader._data.error_map
        #DP !!! initialize np to save the density progresses
        if self.opt.density_prog:
            meshgrid_points = self.DP_create_grid(self.opt.bound,self.opt.TV_density_dim).to('cuda')
            density_prog = []

        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch

            self.train_one_epoch(train_loader)

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=False, best=True)

            #DP !!! saving the the density each 10 epochs 
            if self.epoch % 1 == 0 and self.opt.density_prog:
                density_outputs = self.model.density(meshgrid_points.float())['sigma'].detach().cpu().numpy()
                density_prog.append(density_outputs)
        if self.opt.density_prog:
            np.save( self.opt.workspace + "/density_prog.npy", density_prog)

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader, name=None):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name)
        self.use_tensorboardX = use_tensorboardX

    def test(self, loader, save_path=None, name=None, write_video=True):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'results')

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)
        
        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()

        if write_video:
            all_preds = []
            all_preds_depth = []

        with torch.no_grad():

            for i, data in enumerate(loader):
                
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth = self.test_step(data)

                if self.opt.color_space == 'linear':
                    preds = linear_to_srgb(preds)

                pred = preds[0].detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)

                pred_depth = preds_depth[0].detach().cpu().numpy()
                pred_depth = (pred_depth * 255).astype(np.uint8)

                if write_video:
                    all_preds.append(pred)
                    all_preds_depth.append(pred_depth)
                else:
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_rgb.png'), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_depth.png'), pred_depth)

                pbar.update(loader.batch_size)
        
        if write_video:
            all_preds = np.stack(all_preds, axis=0)
            all_preds_depth = np.stack(all_preds_depth, axis=0)
            imageio.mimwrite(os.path.join(save_path, f'{name}_rgb.mp4'), all_preds, fps=25, quality=8, macro_block_size=1)
            imageio.mimwrite(os.path.join(save_path, f'{name}_depth.mp4'), all_preds_depth, fps=25, quality=8, macro_block_size=1)

        self.log(f"==> Finished Test.")
    
    # [GUI] just train for 16 steps, without any other overhead that may slow down rendering.
    def train_gui(self, train_loader, step=16):

        self.model.train()

        total_loss = torch.tensor([0], dtype=torch.float32, device=self.device)
        
        loader = iter(train_loader)

        # mark untrained grid
        if self.global_step == 0:
            self.model.mark_untrained_grid(train_loader._data.poses, train_loader._data.intrinsics)

        for _ in range(step):
            
            # mimic an infinite loop dataloader (in case the total dataset is smaller than step)
            try:
                data = next(loader)
            except StopIteration:
                loader = iter(train_loader)
                data = next(loader)

            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()
            
            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss = self.train_step(data)
         
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            total_loss += loss.detach()

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss.item() / step

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        outputs = {
            'loss': average_loss,
            'lr': self.optimizer.param_groups[0]['lr'],
        }
        
        return outputs

    
    # [GUI] test on a single image
    def test_gui(self, pose, intrinsics, W, H, bg_color=None, spp=1, downscale=1):
        
        # render resolution (may need downscale to for better frame rate)
        rH = int(H * downscale)
        rW = int(W * downscale)
        intrinsics = intrinsics * downscale

        pose = torch.from_numpy(pose).unsqueeze(0).to(self.device)

        rays = test_gui(pose, intrinsics, rH, rW, -1)

        data = {
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'H': rH,
            'W': rW,
        }
        
        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.fp16):
                # here spp is used as perturb random seed! (but not perturb the first sample)
                preds, preds_depth = self.test_step(data, bg_color=bg_color, perturb=False if spp == 1 else spp)

        if self.ema is not None:
            self.ema.restore()

        # interpolation to the original resolution
        if downscale != 1:
            # TODO: have to permute twice with torch...
            preds = F.interpolate(preds.permute(0, 3, 1, 2), size=(H, W), mode='nearest').permute(0, 2, 3, 1).contiguous()
            preds_depth = F.interpolate(preds_depth.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)

        if self.opt.color_space == 'linear':
            preds = linear_to_srgb(preds)

        pred = preds[0].detach().cpu().numpy()
        pred_depth = preds_depth[0].detach().cpu().numpy()

        outputs = {
            'image': pred,
            'depth': pred_depth,
        }

        return outputs

    def train_one_epoch(self, loader):
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)
        
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:

            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()
                    
            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss = self.train_step(data)
                #DP!!!add loss for uniform grid sampling Minimum Support

                        
         
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(preds, truths)
                        
                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(f"loss={loss_val:.4f} avg loss=({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} avg loss=({total_loss/self.local_step:.4f})")
                pbar.update(loader.batch_size)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")


    def evaluate_one_epoch(self, loader, name=None):
        self.log(f"++> Evaluate at epoch {self.epoch} ...")

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0

            for data in loader:    
                self.local_step += 1

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, truths, loss = self.eval_step(data)

                # all_gather/reduce the statistics (NCCL only support all_*)
                if self.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / self.world_size
                    
                    preds_list = [torch.zeros_like(preds).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)

                    preds_depth_list = [torch.zeros_like(preds_depth).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_depth_list, preds_depth)
                    preds_depth = torch.cat(preds_depth_list, dim=0)

                    truths_list = [torch.zeros_like(truths).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(truths_list, truths)
                    truths = torch.cat(truths_list, dim=0)
                
                loss_val = loss.item()
                total_loss += loss_val

                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:

                    for metric in self.metrics:
                        metric.update(preds, truths)

                    # save image
                    save_path = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_rgb.png')
                    save_path_depth = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_depth.png')

                    #self.log(f"==> Saving validation image to {save_path}")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    if self.opt.color_space == 'linear':
                        preds = linear_to_srgb(preds)

                    pred = preds[0].detach().cpu().numpy()
                    pred = (pred * 255).astype(np.uint8)

                    pred_depth = preds_depth[0].detach().cpu().numpy()
                    pred_depth = (pred_depth * 255).astype(np.uint8)
                    
                    cv2.imwrite(save_path, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(save_path_depth, pred_depth)

                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                    pbar.update(loader.batch_size)


        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(result if self.best_mode == 'min' else - result) # if max mode, use -result
            else:
                self.stats["results"].append(average_loss) # if no metric, choose best by min loss

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def save_checkpoint(self, name=None, full=False, best=False, remove_old=True):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
        }

        if self.model.cuda_ray:
            state['mean_count'] = self.model.mean_count
            state['mean_density'] = self.model.mean_density

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()
        
        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{self.ckpt_path}/{name}.pth"

            if remove_old:
                self.stats["checkpoints"].append(file_path)

                if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                    old_ckpt = self.stats["checkpoints"].pop(0)
                    if os.path.exists(old_ckpt):
                        os.remove(old_ckpt)

            torch.save(state, file_path)

        else:    
            if len(self.stats["results"]) > 0:
                if self.stats["best_result"] is None or self.stats["results"][-1] < self.stats["best_result"]:
                    self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results 
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    # we don't consider continued training from the best ckpt, so we discard the unneeded density_grid to save some storage (especially important for dnerf)
                    if 'density_grid' in state['model']:
                        del state['model']['density_grid']

                    if self.ema is not None:
                        self.ema.restore()
                    
                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")
            
    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/{self.name}_ep*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        
        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")   

        if self.ema is not None and 'ema' in checkpoint_dict:
            self.ema.load_state_dict(checkpoint_dict['ema'])

        if self.model.cuda_ray:
            if 'mean_count' in checkpoint_dict:
                self.model.mean_count = checkpoint_dict['mean_count']
            if 'mean_density' in checkpoint_dict:
                self.model.mean_density = checkpoint_dict['mean_density']
        
        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")
        
        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")
        
        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")
        
        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")