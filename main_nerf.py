import torch
import argparse
import os 
import sys 

from nerf.provider import DP_get_coord
from nerf.provider import DP_change_permission
from nerf.provider import NeRFDataset
from nerf.gui import NeRFGUI
from nerf.utils import *

from functools import partial
from loss import huber_loss

# from colmap_loader import read_points3D_binary, read_points3D_text

# os.environ["CUDA_VISIBLE_DEVICES"] = "3" 

#torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --preload")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)

    ### training options
    parser.add_argument('--iters', type=int, default=30000, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--num_rays', type=int, default=4096, help="num rays sampled per image for each training step")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=1024, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=512, help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0, help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")
    parser.add_argument('--patch_size', type=int, default=1, help="[experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable")

    ### network backbone options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
    parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")

    ### dataset options
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('--preload', action='store_true', help="preload all data into GPU, accelerate training but use more GPU memory")
    # (the default value is for the fox dataset)
    parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=0.33, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--dt_gamma', type=float, default=1/128, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.2, help="minimum near distance for camera")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    parser.add_argument('--bg_radius', type=float, default=-1, help="if positive, use a background model at sphere(bg_radius)")

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=1920, help="GUI width")
    parser.add_argument('--H', type=int, default=1080, help="GUI height")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=64, help="GUI rendering max sample per pixel")

    ### experimental
    parser.add_argument('--error_map', action='store_true', help="use error map to sample rays")
    parser.add_argument('--clip_text', type=str, default='', help="text input for CLIP guidance")
    parser.add_argument('--rand_pose', type=int, default=-1, help="<0 uses no rand pose, =0 only uses rand pose, >0 sample one rand pose every $ known poses")

    #DP !!! 
    # Density options
    parser.add_argument('--get_density', action='store_true', help="get values for cube grid of [-bound,bound]^3")
    parser.add_argument('--get_density_rgb', action='store_true', help="get values for cube grid of [-bound,bound]^3 and color")
    # save density progress
    parser.add_argument('--density_prog',  action='store_true', help="save density progress each ep")
    #TV_parameter
    parser.add_argument('--dist_loss', default=0, help="fix the lambda value for distortion loss, if not set is zero")
    parser.add_argument('--TV_lambda', default=0, help="fix the lambda value for totalvariation, if not set is zero")
    parser.add_argument('--TV_density_dim', type=int,  default=101 , help="set the n of points (for each dimension ) on which evaluate TV")
    parser.add_argument('--l1_loss', default=0, help="fix the value for l1 loss, if not set is zero")
    parser.add_argument('--extreme_loss', default=0, help="fix the value for extreme loss, if not set is zero")
    parser.add_argument('--MS_lambda', default=0, help="fix the lambda value for minimum support, if not set is zero")
    parser.add_argument('--sobel_mse', action='store_true',help="set if u want tocalculate mse weighted on sobel ")
    
    # huber parameter
    parser.add_argument('--Huber_eps', default=1e-2, help="fix the epsilon value for totalvariation, if not set is one")
    # generate a random video 
    parser.add_argument('--random_video',  action='store_true', help="set true if want to generate a video with frame that transit from random 'real' poses")
    #threshold for saving mesh
    parser.add_argument('--mesh_thresh', type=float, default=9, help="fix the threshold for the resulting mesh.ply")
    #set mesh resolution
    parser.add_argument('--mesh_res',  type=int, default=512, help="set if u want save a file with 3d coordinates of camera origin for each photo")
    parser.add_argument('--mesh_thresh_up', type=float, default=float('inf'), help="fix the threshold for the upper bound of  resulting mesh.ply")    
    
    #generate file with coordinates of each Oc of frames 
    parser.add_argument('--mesh_partial', action='store_true', help="set if u want a mesh of a specific location of the scene")
    parser.add_argument('--mesh_principal', action='store_true', help="set if u want to obtain the mesh of solely the principal connected component of the mesh")
    parser.add_argument('--get_histo', action='store_true', help="set if want to save the histogram of the density" )
    parser.add_argument('--first_point', action='store_true', help="set if u want to obtain the mesh of first point with positive density along training views")
    parser.add_argument('--rays_plot', action='store_true', help="set if want to save the histogram of the density along training rays" )
    parser.add_argument('--dens_clip', type=float, default=2, help="set if the tresh for density plot clipped" )    
    parser.add_argument('--binary_sigma', action='store_true', help="use 10+sigma as activation function for the density")
    parser.add_argument('--binary_learn', action='store_true', help="use 10+sigma as activation function for the density")
    parser.add_argument('--sigmoid_sigma', action='store_true', help="use sigmoid sigma as activation function for the density")
    parser.add_argument('--max_epoch', default=0, help="set number of epoch u want to process")
    parser.add_argument('--plot', action='store_true',help="set if u want to have plot starting from latest checkpoint")
    parser.add_argument('--dist_img', action='store_true',help="set if u want to have images of distances ")
    parser.add_argument('--plane', action='store_true',help="set if u want to have a plane mesh saved ")
    
 
    opt = parser.parse_args()

    dens_clip = opt.dens_clip 
    threshold = opt.mesh_thresh 
    resolution = opt.mesh_res
    opt.workspace  = opt.workspace  +"/"
    if opt.binary_sigma:
        opt.workspace = opt.workspace + "binary10+sigma/"
    elif opt.sigmoid_sigma:
        opt.workspace = opt.workspace + "sigmoid_sigma/"
    elif opt.binary_learn:
        opt.workspace = opt.workspace + "binary_learn/"
    else:
        opt.workspace = opt.workspace + "original/"
    
    default_loss = True
    if opt.sobel_mse :
        opt.workspace = opt.workspace + "/sobel_x_mse/" 
        default_loss = False
    if opt.TV_lambda != 0 :
        opt.workspace = opt.workspace + "_TV_" + str(opt.TV_lambda) + "_grid_" + str(opt.TV_density_dim)
        default_loss = False
    if opt.MS_lambda != 0 :
        opt.workspace = opt.workspace + "_MS_" + str(opt.MS_lambda) 
        default_loss = False
    if opt.dist_loss != 0: 
        opt.workspace = opt.workspace   +"_dist_" + str(opt.dist_loss) 
        default_loss = False
    if opt.l1_loss != 0: 
        opt.workspace = opt.workspace + "_l1_" + str(opt.l1_loss)
        default_loss = False
    if opt.extreme_loss != 0: 
        opt.workspace = opt.workspace + "_extreme_" + str(opt.extreme_loss)
        default_loss = False
    if default_loss:
        opt.workspace = opt.workspace + "default"

    
    
    opt.DP_val = False 

    prior_pc = False

    bin_path = os.path.join(opt.path, "colmap_sparse/0/points3D.bin")
    txt_path = os.path.join(opt.path, "colmap_sparse/0/points3D.txt")
    if prior_pc :
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)

   # np.save(os.path.join(opt.path, "xyz"),xyz)

    if opt.O:
        opt.fp16 = True
        opt.cuda_ray = True
        opt.preload = True
    
    if opt.patch_size > 1:
        opt.error_map = False # do not use error_map if use patch-based training
        # assert opt.patch_size > 16, "patch_size should > 16 to run LPIPS loss."
        assert opt.num_rays % (opt.patch_size ** 2) == 0, "patch_size ** 2 should be dividable by num_rays."


    if opt.ff:
        opt.fp16 = True
        assert opt.bg_radius <= 0, "background model is not implemented for --ff"
        from nerf.network_ff import NeRFNetwork
        print("FF")
    elif opt.tcnn:
        opt.fp16 = True
        assert opt.bg_radius <= 0, "background model is not implemented for --tcnn"
        from nerf.network_tcnn import NeRFNetwork
        print("fp16")
    elif opt.binary_sigma:
        print("binary_sigma")
        from nerf.network_binary import NeRFNetwork
    elif opt.binary_learn:
        print("binary_sigma")
        from nerf.network_binarylearn import NeRFNetwork
    elif opt.sigmoid_sigma:
        print("sigmoid sigma")
        from nerf.network_sigmoid import NeRFNetwork
    else:
        print("original")
        from nerf.network import NeRFNetwork

    print(opt)
    
    seed_everything(opt.seed)

    model = NeRFNetwork(
        encoding="hashgrid",
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        density_scale=1,
        min_near=opt.min_near,
        density_thresh=opt.density_thresh,
        bg_radius=opt.bg_radius,
    )

    import torch

    
    print(model)

    criterion = torch.nn.MSELoss(reduction='none') #DP !!! original uncomment
    #criterion = partial(huber_loss, reduction='none')
    #criterion = torch.nn.HuberLoss(reduction='none', beta=0.1) # only available after torch 1.10 ?

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if opt.mesh_partial:
        v1,v2,v3 = DP_get_coord(opt, device=device) #(opt, device=device, type='get_coordinates').dataloader()
        np.save(opt.workspace+"/road_versors.npy", [v1, v2, v3])
        print(v1,v2,v3)

    if opt.plot : 
        print("SAVING PLOTS")
        metrics = [PSNRMeter(), LPIPSMeter(device=device)]
        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16, metrics=metrics, use_checkpoint=opt.ckpt)
                                
        if opt.rays_plot: 
            train_loader = NeRFDataset(opt, device=device, type='train',downscale=1,plot = opt.plot).dataloader()
            trainer.DP_plot_density_rays(train_loader,ray_start=opt.min_near,dens_clip_value=dens_clip)
        if opt.get_histo:
            trainer.DP_get_histo()
        if opt.first_point :
            train_loader = NeRFDataset(opt, device=device, type='train',downscale=1,plot = opt.plot).dataloader()
            first_point_name =  opt.workspace.replace("/","_").replace("results_","")
            if opt.mesh_thresh != 9 : 
                first_point_name = first_point_name + "_mesh_thresh_"+str(opt.mesh_thresh)
            if opt.mesh_thresh_up != float('inf') : 
                first_point_name = first_point_name + "_up_"+str(opt.mesh_thresh_up)
            trainer.save_mesh_boundary(train_loader,file_name =first_point_name,ray_start=opt.min_near,threshold=opt.mesh_thresh,threshold_up=opt.mesh_thresh_up)
        if opt.dist_img:
            train_loader = NeRFDataset(opt, device=device, type='train',downscale=2,plot = opt.plot).dataloader()
            trainer.DP_distance_imgs(train_loader, ray_start=opt.min_near)
        if opt.plane : 
            train_loader = NeRFDataset(opt, device=device, type='train',downscale=2,plot = opt.plot).dataloader()
            trainer.DP_save_plane(train_loader,file_name = opt.workspace.replace("/","_").replace("results_","_plane"), threshold=opt.mesh_thresh, z_val =0.0)
            
        DP_change_permission(str(opt.workspace))
        

    elif opt.random_video:
        
        metrics = [PSNRMeter(), LPIPSMeter(device=device)]
        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16, metrics=metrics, use_checkpoint=opt.ckpt)
        
        test_loader = NeRFDataset(opt, device=device, type='random_video').dataloader()  #DP!!! added n_test , n_test = 250

        if test_loader.has_gt:
            trainer.evaluate(test_loader) # blender has gt, so evaluate it.

        trainer.test(test_loader, write_video=True) # test and save video
        
        trainer.save_mesh(resolution=resolution, threshold=threshold)
        '''
    if opt.get_density:
        
        metrics = [PSNRMeter(), LPIPSMeter(device=device)]
        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16, metrics=metrics, use_checkpoint=opt.ckpt)

        test_loader = NeRFDataset(opt, device=device, type='random_video').dataloader()  #DP!!! added n_test , n_test = 250

        if test_loader.has_gt:
            trainer.evaluate(test_loader) # blender has gt, so evaluate it.
        self.'''
    elif opt.test:
        metrics = [PSNRMeter(), LPIPSMeter(device=device)]
        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16, metrics=metrics, use_checkpoint=opt.ckpt)

        if opt.gui:
            gui = NeRFGUI(opt, trainer)
            gui.render()
        
        else:
            test_loader = NeRFDataset(opt, device=device, type='test').dataloader()  #DP !!! , n_test = 50added n_test , n_test = 250
            #DP !!!
            #trainer.test(test_loader, write_video=False) # test and save video
            DP_change_permission(str(opt.workspace))
            trainer.test(test_loader, write_video=True) # test and save video
            if opt.mesh_partial:
                trainer.save_mesh(resolution=resolution, threshold=threshold,v1=v1 , v2 = v2 , v3 = v3) 
            else:
                trainer.save_mesh(resolution=resolution, threshold=threshold)
            if test_loader.has_gt:
                trainer.evaluate(test_loader) # blender has gt, so evaluate it.
    

            
            

    else:

        optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)

        
        train_loader = NeRFDataset(opt, device=device, type='train').dataloader()
        
        # decay to 0.1 * init_lr at last iter step
        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

        metrics = [PSNRMeter(), LPIPSMeter(device=device)]
        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, optimizer=optimizer, criterion=criterion, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, scheduler_update_every_step=True, metrics=metrics, use_checkpoint=opt.ckpt, eval_interval=50, TV_lambda=opt.TV_lambda,MS_lambda = opt.MS_lambda,sobel_mse = opt.sobel_mse, dist_loss=opt.dist_loss, Huber_eps=opt.Huber_eps, mesh_thresh=opt.mesh_thresh, l1_loss = opt.l1_loss, extreme_loss=opt.extreme_loss)

        if opt.gui:
            gui = NeRFGUI(opt, trainer, train_loader)
            gui.render()
        
        else:
            valid_loader = NeRFDataset(opt, device=device, type='val', downscale=1).dataloader()
            if opt.max_epoch == 0 :
                max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
            else :
                max_epoch = np.int32(opt.max_epoch)
            trainer.train(train_loader, valid_loader, max_epoch)
            
            if opt.mesh_partial:
                v1,v2,v3 = DP_get_coord(opt, device=device) #(opt, device=device, type='get_coordinates').dataloader()
                np.save(opt.workspace+"/road_versors.npy", [v1, v2, v3])
                print(v1,v2,v3)
                     
            train_load = NeRFDataset(opt, device=device, type='train',plot=True).dataloader()
            if opt.rays_plot: 
                trainer.DP_plot_density_rays(train_load,ray_start=opt.min_near, dens_clip_value=dens_clip)
            if opt.get_histo:
                trainer.DP_get_histo()
            if opt.first_point :
                first_point_name =  opt.workspace.replace("/","_").replace("results_","")
                if opt.mesh_thresh != 9 : 
                    first_point_name = first_point_name + "_mesh_thresh_"+str(opt.mesh_thresh)
                if opt.mesh_thresh_up != float('inf') : 
                    first_point_name = first_point_name + "_up_"+str(opt.mesh_thresh_up)
                trainer.save_mesh_boundary(train_load,file_name =first_point_name,ray_start=opt.min_near,threshold=opt.mesh_thresh,threshold_up=opt.mesh_thresh_up)
            if opt.dist_img:
                train_loader = NeRFDataset(opt, device=device, type='train',downscale=2,plot=True).dataloader()
                trainer.DP_distance_imgs(train_loader, ray_start=opt.min_near)
            DP_change_permission(str(opt.workspace))
            # also test
           # test_loader = NeRFDataset(opt, device=device, type='test', n_test = 25).dataloader() #DP !!! added n_test
            
           # if test_loader.has_gt:
             #   trainer.evaluate(test_loader) # blender has gt, so evaluate it.
            
           # trainer.test(test_loader, write_video=True ) # test and save video
            

           # trainer.save_mesh(resolution=resolution, threshold=threshold)