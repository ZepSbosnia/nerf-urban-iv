import os
import cv2
import glob
import json
from cv2 import transform
import tqdm
import numpy as np
from scipy.spatial.transform import Slerp, Rotation

import trimesh

import torch
from torch.utils.data import DataLoader

from .utils import get_rays

def DP_Rtp_to_xyz(R,theta,phi,scale=.33 , offset=[0, 0, 0]):

    x = (R * np.sin(theta) * np.cos(phi))* scale 
    y = (R * np.cos(theta) * np.cos(phi) )* scale 
    z = (-R * np.sin(phi))* scale 

    x += offset[0]
    y += offset[1]
    z += offset[2]

    #DP !!! shifted x,y,z because nerf_matrix_to_ngp funct shift the rows of transofmration matrix in this way 
    res = np.array([y,z,x]) * scale 
    return res  

def DP_road_versors(road_versor):
    y_perp = np.sqrt(1/(1+(road_versor[2]/road_versor[0])**2))
    road_perp = np.array([-road_versor[2]/road_versor[0]* y_perp,road_versor[1],y_perp]) #get the perpendicular vector of road_versor in the plane with second entry fixed
    road_perp = road_perp/ np.linalg.norm(road_perp) 
    v1= road_versor
    v2=road_perp
    v3 = np.cross(v1 ,v2)
    v3 = v3/ np.linalg.norm(v3) 
    return v1,v2,v3


def DP_get_axis(pose, scale=0.33, offset=[0, 0, 0]):
    
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    
    R_n = new_pose[0,3]/new_pose[0,2]
    R_o = pose[0,3]/pose[0,2]
    theta_n = np.arctan(new_pose[1,0]/-new_pose[0,0]) *180/np.pi
    theta_o = np.arctan(pose[1,0]/-pose[0,0])*180/np.pi
    phi_n = np.arctan(-new_pose[2,2]/new_pose[2,1]) *180/np.pi
    phi_o = np.arctan(-pose[2,2]/pose[2,1])*180/np.pi
    print("touli")
    return [R_o* scale,theta_o,phi_o, R_n,theta_n,phi_n]
# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
#DP comment: mette la prima riga come la terza, moltiplica seconda e terza colonna per -1, l'ultima colonna e riscalata e sommata se offset non zero, quarta riga 0,0,0,1
def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0]):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose

def DP_change_permission(folder_path,uid=2001,gid=2001):

    # Walk through the directory tree
    components = folder_path.split(os.sep)
    for i in range(1, len(components) + 1):
        partial_path = os.sep.join(components[:i])
        if partial_path:  # Skip empty path elements (e.g., leading '/')
            os.chown(partial_path, uid, gid)

    for root, dirs, files in os.walk(folder_path):
        for d in dirs:
            dir_path = os.path.join(root, d)
            os.chown(dir_path, uid, gid)
        for f in files:
            file_path = os.path.join(root, f)
            os.chown(file_path, uid, gid)
  
def DP_get_coord( opt, device,downscale =1 ):

    opt = opt
    device = device
    downscale = downscale
    root_path = opt.path
    preload = opt.preload # preload data into GPU
    scale = opt.scale # camera radius scale to make sure camera are inside the bounding box.
    offset = opt.offset # camera offset
    bound = opt.bound # bounding box half length, also used as the radius to random sample poses.
    fp16 = opt.fp16 # if preload, load into fp16.
    workspace = opt.workspace


    rand_pose = opt.rand_pose
    downscale = downscale
    if os.path.exists(os.path.join(root_path, 'transforms.json')):
        mode = 'colmap' # manually split, use view-interpolation for test.
    elif os.path.exists(os.path.join(root_path, 'transforms_train.json')):
        mode = 'blender' # provided split
    else:
        raise NotImplementedError(f'[NeRFDataset] Cannot find transforms*.json under {self.root_path}')

    # load nerf-compatible format data.
    if mode == 'colmap':
        with open(os.path.join(root_path, 'transforms.json'), 'r') as f:
            transform = json.load(f)
    elif mode == 'blender':
        
        with open(os.path.join(root_path, f'transforms_all.json'), 'r') as f:
            transform = json.load(f)

    else:
        raise NotImplementedError(f'unknown dataset mode: {mode}')

    # load image size
    if 'h' in transform and 'w' in transform:
        H = int(transform['h']) // downscale
        W = int(transform['w']) // downscale
    else:
        # we have to actually read an image to get H and W later.
        H = W = None
    
    # read images
    transform["frames"]
    frames = transform["frames"]
    frames.sort(key=lambda x: int(x["file_path"].split(".png")[0].split(".jpg")[0].split("/")[-1]) )
    #print(type(frames))
    
    #frames = sorted(frames, key=lambda d: d['file_path']) # why do I sort...
    

    #DP!!! takes two random frames (positions) and constructs a sinusoidal path between them with n_steps +1 frames
    # for colmap, manually interpolate a test set.

    # for colmap, manually split a valid set (the first frame).
    if mode == 'colmap':
        if type == 'train':
            frames = frames[1:]
        elif type == 'val':
            frames = frames[:1]
        # else 'all' or 'trainval' : use all frames
    
    poses = []
    images = []
    #DP !!!
    frames_name = []
    for f in tqdm.tqdm(frames, desc=f'Loading {type} data'):
        f_path = os.path.join(root_path, f['file_path'])
        if mode == 'blender' and '.' not in os.path.basename(f_path):
            f_path += '.png' # so silly...

        # there are non-exist paths in fox...
        if not os.path.exists(f_path):
            continue
        
        pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
        pose = nerf_matrix_to_ngp(pose, scale=scale, offset=offset) #DP comment: mette la prima riga come la terza, moltiplica seconda e terza colonna per -1, l'ultima colonna e riscalata e sommata se offset non zero, quarta riga 0,0,0,1

        image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
        if H is None or W is None:
            H = image.shape[0] // downscale
            W = image.shape[1] // downscale

        # add support for the alpha channel as a mask.
        if image.shape[-1] == 3: 
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

        if image.shape[0] != H or image.shape[1] != W:
            image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
            
        image = image.astype(np.float32) / 255 # [H, W, 3/4]

        poses.append(pose)
        images.append(image)

        #DP !!!
        frames_name.append(f['file_path'].split("/")[-1].split(".jpg")[0])
        
    poses = torch.from_numpy(np.stack(poses, axis=0)) # [N, 4, 4]
    
    if images is not None:
        images = torch.from_numpy(np.stack(images, axis=0)) # [N, H, W, C]
    
    # calculate mean radius of all camera poses
    radius = poses[:, :3, 3].norm(dim=-1).mean(0).item()
    #print(f'[INFO] dataset camera poses: radius = {self.radius:.4f}, bound = {self.bound}')


    # [debug] uncomment to view all training poses.
    # visualize_poses(self.poses.numpy())

    # [debug] uncomment to view examples of randomly generated poses.
    # visualize_poses(rand_poses(100, self.device, radius=self.radius).cpu().numpy())



    if preload:
        poses = poses.to(device)
        if images is not None:
            # TODO: linear use pow, but pow for half is only available for torch >= 1.10 ?
            if fp16 and opt.color_space != 'linear':
                dtype = torch.half
            else:
                dtype = torch.float
            images = images.to(dtype).to(device)
#        if error_map is not None:
#            error_map = error_map.to(device)

    # load intrinsics
    if 'fl_x' in transform or 'fl_y' in transform:
        fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / downscale
        fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / downscale
    elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
        # blender, assert in radians. already downscaled since we use H/W
        fl_x = W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
        fl_y = H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
        if fl_x is None: fl_x = fl_y
        if fl_y is None: fl_y = fl_x
    else:
        raise RuntimeError('Failed to load focal length, please check the transforms.json!')

    cx = (transform['cx'] / downscale) if 'cx' in transform else (W / 2)
    cy = (transform['cy'] / downscale) if 'cy' in transform else (H / 2)

    
    intrinsics = np.array([fl_x, fl_y, cx, cy])



    rays_o = poses[..., :3, 3] # [B, 3]
    print(workspace)
    np.save(workspace + "/camera_origins.npy",rays_o.detach().cpu().numpy() )

    road_dirs = []
    for ind, pt in enumerate(rays_o.detach().cpu().numpy()):
        for pt2 in rays_o[ind+1:].detach().cpu().numpy():
            if np.linalg.norm(pt - pt2) > .1:
                road_dirs.append(pt2-pt)
    road_dir = np.mean(road_dirs,axis=0)
    road_dir = road_dir / np.linalg.norm(road_dir)      

    #ind = frames_name.index("10")'''
    #rays = get_rays(rays_o, intrinsics, H, W, opt.patch_size)

    #res_0 = DP_get_axis(poses[0, :, :].detach().cpu().numpy())
    #res_176 = DP_get_axis(poses[176, :, :].detach().cpu().numpy())

    v1,v2,v3 = DP_road_versors(road_dir)

    return v1,v2,v3

# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0]):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose


def visualize_poses(poses, size=0.1):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 3

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a], [pos, o]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()


def rand_poses(size, device, radius=1, theta_range=[np.pi/3, 2*np.pi/3], phi_range=[0, 2*np.pi]):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, \pi]
        phi_range: [min, max], should be in [0, 2\pi]
    Return:
        poses: [size, 4, 4]
    '''
    
    def normalize(vectors):
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)

    thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
    phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1) # [B, 3]

    # lookat
    forward_vector = - normalize(centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0).repeat(size, 1) # confused at the coordinate system...
    right_vector = normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    return poses


class NeRFDataset:
    def __init__(self, opt, device, type='train', downscale=1, n_test=10, plot = False):#DP !!! plot added 
        super().__init__()
        
        self.opt = opt
        self.device = device
        self.type = type # train, val, test
        self.downscale = downscale
        self.root_path = opt.path
        self.preload = opt.preload # preload data into GPU
        self.scale = opt.scale # camera radius scale to make sure camera are inside the bounding box.
        self.offset = opt.offset # camera offset
        self.bound = opt.bound # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = opt.fp16 # if preload, load into fp16.
        self.first_point = opt.first_point#DP !!! added to load all rays for training photos, used to generate a good mesh 

        self.training = self.type in ['train', 'all', 'trainval']
        self.num_rays = self.opt.num_rays if self.training else -1
        self.plot = plot#DP !!! 
        if plot : 
            self.num_rays = -1

        '''
        if opt.first_point :#DP !!! 
            self.num_rays = -1 '''

        self.rand_pose = opt.rand_pose

        # auto-detect transforms.json and split mode.
        if os.path.exists(os.path.join(self.root_path, 'transforms.json')):
            self.mode = 'colmap' # manually split, use view-interpolation for test.
        elif os.path.exists(os.path.join(self.root_path, 'transforms_train.json')):
            self.mode = 'blender' # provided split
        else:
            raise NotImplementedError(f'[NeRFDataset] Cannot find transforms*.json under {self.root_path}')

        # load nerf-compatible format data.
        if self.mode == 'colmap':
            with open(os.path.join(self.root_path, 'transforms.json'), 'r') as f:
                transform = json.load(f)
        elif self.mode == 'blender':
            # load all splits (train/valid/test), this is what instant-ngp in fact does...
            if type == 'all':
                transform_paths = glob.glob(os.path.join(self.root_path, '*.json'))
                transform = None
                for transform_path in transform_paths:
                    with open(transform_path, 'r') as f:
                        tmp_transform = json.load(f)
                        if transform is None:
                            transform = tmp_transform
                        else:
                            transform['frames'].extend(tmp_transform['frames'])
            # load train and val split
            elif type == 'trainval':
                with open(os.path.join(self.root_path, f'transforms_train.json'), 'r') as f:
                    transform = json.load(f)
                with open(os.path.join(self.root_path, f'transforms_val.json'), 'r') as f:
                    transform_val = json.load(f)
                transform['frames'].extend(transform_val['frames'])
            # only load one specified split
            elif type == 'random_video':  #DP !!!
                with open(os.path.join(self.root_path, f'transforms_test.json'), 'r') as f:
                    transform = json.load(f)
            else:
                with open(os.path.join(self.root_path, f'transforms_{type}.json'), 'r') as f:
                    transform = json.load(f)

        else:
            raise NotImplementedError(f'unknown dataset mode: {self.mode}')

        # load image size
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h']) // downscale
            self.W = int(transform['w']) // downscale
        else:
            # we have to actually read an image to get H and W later.
            self.H = self.W = None
        
        # read images
        frames = transform["frames"]
        #frames = sorted(frames, key=lambda d: d['file_path']) # why do I sort...
        

        #DP!!! takes two random frames (positions) and constructs a sinusoidal path between them with n_steps +1 frames
        # for colmap, manually interpolate a test set.
        if self.mode == 'colmap' and type == 'test':
            
            print('ENTRATOOOOOOOOOOOOOOOOOOOOOOOOO')
            # choose two random poses, and interpolate between.
            f0, f1 = np.random.choice(frames, 2, replace=False)
            pose0 = nerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
            pose1 = nerf_matrix_to_ngp(np.array(f1['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
            rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
            slerp = Slerp([0, 1], rots)

            self.poses = []
            self.images = None
            for i in range(n_test + 1):
                ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = slerp(ratio).as_matrix()
                pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                self.poses.append(pose)

        elif type == 'random_video':  #DP !!!
            
            # choose two random poses, and interpolate between.
            f0, f1 = np.random.choice(frames, 2, replace=False)
            sorted_frames = sorted(frames, key=lambda x: tuple(map(int, (x['file_path'].replace(".jpg","").replace(".png","").replace("./images/","")).split("_"))))
            self.poses = []
            for ind in range(len(sorted_frames)-1):
                f0 = sorted_frames[ind]
                f1 = sorted_frames[ind+1]
                pose0 = nerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
                pose1 = nerf_matrix_to_ngp(np.array(f1['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
                rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
                slerp = Slerp([0, 1], rots)
                
                self.images = None
                for i in range(n_test + 1):
                    ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
                    pose = np.eye(4, dtype=np.float32)
                    pose[:3, :3] = slerp(ratio).as_matrix()
                    pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                    self.poses.append(pose)

        else:
            # for colmap, manually split a valid set (the first frame).
            if self.mode == 'colmap':
                if type == 'train':
                    frames = frames[1:]
                elif type == 'val':
                    frames = frames[:1]
                # else 'all' or 'trainval' : use all frames
            
            self.poses = []
            self.images = []
            for f in tqdm.tqdm(frames, desc=f'Loading {type} data'):

                f_path = os.path.join(self.root_path, f['file_path'])
                if self.mode == 'blender' and '.' not in os.path.basename(f_path):
                    f_path += '.png' # so silly...

                
                
                pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
                pose = nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset)
                self.poses.append(pose)
                # there are non-exist paths in fox...
                if not os.path.exists(f_path):
                    continue
                image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
                if self.H is None or self.W is None:
                    self.H = image.shape[0] // downscale
                    self.W = image.shape[1] // downscale

                # add support for the alpha channel as a mask.
                if image.shape[-1] == 3: 
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

                if image.shape[0] != self.H or image.shape[1] != self.W:
                    image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
                    
                image = image.astype(np.float32) / 255 # [H, W, 3/4]
                self.images.append(image)
            
        self.poses = torch.from_numpy(np.stack(self.poses, axis=0)) # [N, 4, 4]
        if self.images !=[] :
            self.images = torch.from_numpy(np.stack(self.images, axis=0)) # [N, H, W, C]
        
        # calculate mean radius of all camera poses
        self.radius = self.poses[:, :3, 3].norm(dim=-1).mean(0).item()
        #print(f'[INFO] dataset camera poses: radius = {self.radius:.4f}, bound = {self.bound}')

        # initialize error_map
        if self.training and self.opt.error_map:
            self.error_map = torch.ones([self.images.shape[0], 128 * 128], dtype=torch.float) # [B, 128 * 128], flattened for easy indexing, fixed resolution...
        else:
            self.error_map = None

        # [debug] uncomment to view all training poses.
        # visualize_poses(self.poses.numpy())

        # [debug] uncomment to view examples of randomly generated poses.
        # visualize_poses(rand_poses(100, self.device, radius=self.radius).cpu().numpy())

        if self.preload:
            self.poses = self.poses.to(self.device)
            if self.images is not None:
                # TODO: linear use pow, but pow for half is only available for torch >= 1.10 ?
                if self.fp16 and self.opt.color_space != 'linear':
                    dtype = torch.half
                else:
                    dtype = torch.float
                if self.images !=[] :
                    self.images = self.images.to(dtype).to(self.device)
            if self.error_map is not None:
                self.error_map = self.error_map.to(self.device)

        # load intrinsics
        if 'fl_x' in transform or 'fl_y' in transform:
            fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / downscale
            fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / downscale
        elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = self.W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
            fl_y = self.H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
            if fl_x is None: fl_x = fl_y
            if fl_y is None: fl_y = fl_x
        else:
            raise RuntimeError('Failed to load focal length, please check the transforms.json!')

        cx = (transform['cx'] / downscale) if 'cx' in transform else (self.W / 2)
        cy = (transform['cy'] / downscale) if 'cy' in transform else (self.H / 2)
    
        self.intrinsics = np.array([fl_x, fl_y, cx, cy])


    def collate(self, index):

        B = len(index) # a list of length 1

        # random pose without gt images.
        if self.rand_pose == 0 or index[0] >= len(self.poses):

            poses = rand_poses(B, self.device, radius=self.radius)

            # sample a low-resolution but full image for CLIP
            s = np.sqrt(self.H * self.W / self.num_rays) # only in training, assert num_rays > 0
            rH, rW = int(self.H / s), int(self.W / s)
            rays = get_rays(poses, self.intrinsics / s, rH, rW, -1)

            return {
                'H': rH,
                'W': rW,
                'rays_o': rays['rays_o'],
                'rays_d': rays['rays_d'],    
            }

        poses = self.poses[index].to(self.device) # [B, 4, 4]

        error_map = None if self.error_map is None else self.error_map[index]
        
        rays = get_rays(poses, self.intrinsics, self.H, self.W, self.num_rays, error_map, self.opt.patch_size)

        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
        }

        if self.images !=[] :
            images = self.images[index].to(self.device) # [B, H, W, 3/4]
            if self.training and not self.plot:#DP !!! added  and not self.first_point
                C = images.shape[-1]
                images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
            results['images'] = images
        
        # need inds to update error_map
        if error_map is not None:
            results['index'] = index
            results['inds_coarse'] = rays['inds_coarse']
            
        return results

    def dataloader(self):
        size = len(self.poses)
        if self.training and self.rand_pose > 0:
            size += size // self.rand_pose # index >= size means we use random pose.
        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = self.images is not None
        return loader