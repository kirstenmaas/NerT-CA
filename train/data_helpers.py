import numpy as np
from datetime import datetime
import wandb
import os
import copy

from proj_helpers import get_ray_values_tigre

# argparse
def config_parser_hybrid(config_file='temporal/hybrid.txt', sweep_file='temporal/sweep-hybrid.yaml'):

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path', default=config_file)
    parser.add_argument('--wandb_sweep_yaml', type=str, default=sweep_file)
    parser.add_argument('--use_wandb', default=True, type=lambda x: (str(x).lower() == 'true'))

    # general run info
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--x_ray_type", type=str, default='roadmap')
    parser.add_argument('--data_size', type=int)
    parser.add_argument('--model_name', type=str)

    # data args
    parser.add_argument('--use_experiment_name', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument('--data_limited_range', type=float)
    parser.add_argument('--data_step_size', type=float)
    parser.add_argument('--data_numb_angles', type=int, default=None)
    parser.add_argument('--data_time_range_start', type=int)
    parser.add_argument('--data_time_range_end', type=int)
    parser.add_argument('--data_limited_range_test', type=int, default=None)
    parser.add_argument('--data_step_size_test', type=float, default=None)

    parser.add_argument('--only_prepare_data', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--debug_mode', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--visualize_geometry', default=False, type=lambda x: (str(x).lower() == 'true'))
    
    parser.add_argument('--export_mesh', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--log_dir', type=str, default=None, help='specific weights model file to reload for coarse network')

    # run info
    parser.add_argument('--n_iters', type=int)
    parser.add_argument('--display_every', type=int, default=500)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--batch_size_start', type=int, default=1024)
    parser.add_argument('--batch_size_decay_steps', type=int, default=15000)
    parser.add_argument('--batch_size_end', type=int, default=8192)
    
    # parameters nerf
    parser.add_argument('--depth_samples_min', type=int)
    parser.add_argument('--depth_samples_max', type=int)
    parser.add_argument('--depth_samples_increase_every', type=int)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_end_factor', type=float, default=0.1)
    parser.add_argument('--lr_decay_steps', type=int, default=100000)
    parser.add_argument('--img_sample_size', type=int, default=64**2)

    # TensoRF
    parser.add_argument('--N_voxel_init', type=int, default=2097156, help='Initial voxels') # 128 ** 3
    parser.add_argument('--N_voxel_final', type=int, default=262144000, help='Final voxels') # 640 ** 3
    parser.add_argument('--step_ratio', type=float, default=1, help='step ratio for the voxels')
    parser.add_argument('--N_components', type=int, default=16, help='Number of components')
    parser.add_argument("--density_shift", type=float, default=-10,
                        help='shift density in softplus; making density = 0  when feature == 0')
    parser.add_argument('--bb_thresh', type=float, default=1.5)
    parser.add_argument('--upsamp_list', type=int, action='append')
    parser.add_argument('--update_density_list', type=int, action='append')
    parser.add_argument('--l1_weight_init', type=float, default=0)
    parser.add_argument('--l1_weight_final', type=float, default=0)
    parser.add_argument('--tv_weight_init', type=float, default=1e-2)
    parser.add_argument('--tv_weight_final', type=float, default=1e-2)

    parser.add_argument('--mlp_num_time_dim', type=int, default=8)
    parser.add_argument('--mlp_num_channels', type=int, default=128)
    parser.add_argument('--mlp_num_early_layers', type=int, default=4)
    parser.add_argument('--mlp_num_late_layers', type=int, default=0)
    parser.add_argument('--use_time_latents', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--per_layer_input', default=False, type=lambda x: (str(x).lower() == 'true'))

    # FourieRF
    parser.add_argument("--increase_feature_cap_every", type=int, default=1,
                        help='Increase the frequency_cap every certain number of iterations')
    parser.add_argument("--increase_frequency_cap_until", type=int, default=30000,
                        help='Increase the frequency_cap until a certain number of iterations')
    parser.add_argument("--density_clip", type=float, default=100.0, help='In percentages of parameters used, so 1 is using 1 percent of available parameters')
    parser.add_argument("--max_freq", type=float, default=1.0)

    parser.add_argument('--pos_enc', type=str)
    parser.add_argument('--pos_enc_basis', type=int)
    parser.add_argument('--pos_enc_window_start', type=int, default=0)
    parser.add_argument('--pos_enc_window_decay_steps', type=int)
    
    parser.add_argument('--lr_sparse', type=float, default=1e-3)
    parser.add_argument('--lr_sparse_end_factor', type=float, default=0.1)
    parser.add_argument('--lr_sparse_decay_steps', type=int, default=100000)

    parser.add_argument('--model_delay', type=int, default=0)

    parser.add_argument('--occl_weight_start', type=float, default=0)
    parser.add_argument('--occl_weight_end', type=float, default=0)
    parser.add_argument('--occl_reg_perc', type=float, default=0)

    parser.add_argument('--entro_weight_start', type=float, default=0)
    parser.add_argument('--entro_weight_end', type=float, default=0)

    parser.add_argument('--param_delay', type=float, default=0)
    parser.add_argument('--hyperparam_decay_steps', type=int)

    parser.add_argument('--var_sample_perc', type=float, default=0.)
    parser.add_argument('--var_sample_thre', type=float, default=0.)

    return parser

def denormalize_image(image, img_width, img_height, img_min_max):
    image = image.reshape((img_width, img_height)).T

    if int(np.min(image)) == 0 and int(np.max(image)) == 1:
        denormalized_image = image * (img_min_max[1] - img_min_max[0]) + img_min_max[0]
    else:
        denormalized_image = image
    
    return denormalized_image

# loads the data and prepares it for training
def prepare_data_for_loader_tigre(data, geo_info, img_width, img_height, depth_samples_per_ray, weighted_loss_max, device, use_weighting=True):
    rays = np.stack([get_ray_values_tigre(row['theta'], row['phi'], row['larm'], geo_info, device) for row in data], 0) #[N_img, ro+rd, W, H, 3]

    images = np.stack([denormalize_image(np.load(row['file_path']), img_width, img_height, row['img_min_max']) for row in data], 0) #[N_img, W, H]

    images = np.repeat(images[:, None, :, :, None], 3, axis=-1) #[N_img, 1, W, H, 3]

    weighted_images = np.ones((images.shape[0], img_width, img_height))
    if use_weighting:
        weighted_images = np.stack([np.load(row['weighted_file_path']).reshape((img_width, img_height)).T for row in data], 0) #[N_img, W, H]

    weighted_images = (weighted_images - 1) * weighted_loss_max + 1
    weighted_images = np.repeat(weighted_images[:, None, :, :, None], rays.shape[-1], axis=-1) #[N_img, 1, W, H, 3]

    masks = np.stack([np.load(row['mask_file_path']).reshape((img_width, img_height)).T for row in data], 0) #[N_img, W, H]
    masks = np.repeat(masks[:, None, :, :, None], 3, axis=-1) #[N_img, 1, W, H, 3]

    heart_phases = np.array([row['heart_phase'] for row in data]) #[N_img]
    heart_phases = np.tile(heart_phases[:, None, None], (img_width, img_height)) #[N_img, W, H]
    phases_train = np.reshape(heart_phases, [-1]) #[N_img*W*H]

    rays_train = np.concatenate([rays, images, weighted_images, masks], 1) #[N_img, ro+rd+img+wimg+mimg, W, H, 3]
    rays_train = np.transpose(rays_train, [0,2,3,1,4]) #[N_img, W, H, ro+rd+img, 3]
    rays_train = np.reshape(rays_train, [-1, rays_train.shape[-2], rays_train.shape[-1]]) #[N_img*W*H, ro+rd+img, 3]

    return rays_train, phases_train

def initialize_wandb(extra_chars=''):
    exp_name = datetime.now().strftime("%Y-%m-%d-%H%M") + extra_chars
    wandb.init(
        notes=exp_name,
        # mode='offline'
    )
    return exp_name

def initialize_save_folder(folder_name, exp_name):
    log_dir = folder_name + 'runs/' + exp_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    return log_dir

def overwrite_args_wandb(run_args, wandb_args):
    # we want to overwrite the args based on the sweep args
    new_args = copy.deepcopy(run_args)
    for key in wandb_args.keys():
        setattr(new_args, key, wandb_args[key])
    
    return new_args