import torch
import numpy as np
import sys
import json
import traceback
import yaml
from tqdm.auto import tqdm
import gc
import time

sys.path.append('.')
torch.cuda.empty_cache()

torch.set_printoptions(precision=10)

from data_helpers import *
from preprocess.datatoray import datatoray
from train_helpers import *
from model.tensoRF import TensorVM
from model.Temporal import Temporal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Main wrapper function to optionally enable debug mode
def main():
    parser = config_parser_hybrid()
    run_args = parser.parse_args()
    debug_mode = run_args.debug_mode

    if debug_mode:
        print('DEBUG MODE ON')
        try:
            train()
        except Exception as e:
            print(traceback.print_exc(), file=sys.stderr)
            exit(1)
    else:
        train()

# Function to generate projection data using datatoray (e.g., forward projections)
def generate_data(run_args):
    datatoray(run_args)

def train():
    parser = config_parser_hybrid()
    run_args = parser.parse_args()

    exp_name = initialize_wandb('-hybrid')
    run_args = overwrite_args_wandb(run_args, wandb.config)
    wandb.log(vars(run_args))

    store_folder_name = f'cases/{run_args.data_name}/'
    log_dir = initialize_save_folder(store_folder_name, exp_name)

    # log all the hyperparameters in a txt file
    if run_args.config is not None:
        f = f'{log_dir}config.json'
        with open(f, 'w') as file:
            file.write(json.dumps(vars(run_args)))

    data_folder_name = f'data/{run_args.data_name}/{run_args.data_size}/'
    general_file_name = f'{data_folder_name}general.json'

    if run_args.use_experiment_name:
        train_file_name = f'{data_folder_name}train-{run_args.experiment_name}.json'
        test_file_name = f'{data_folder_name}test-{run_args.experiment_name}.json'
    else:
        train_file_name = f'{data_folder_name}train-{float(run_args.data_limited_range)}-{float(run_args.data_step_size)}-{run_args.data_time_range_start}-{run_args.data_time_range_end}.json'
        test_file_name = f'{data_folder_name}test-{float(run_args.data_limited_range)}-{float(run_args.data_step_size)}-{run_args.data_time_range_start}-{run_args.data_time_range_end}.json'

    # if not (os.path.exists(train_file_name) and os.path.exists(test_file_name)):
    generate_data(run_args)

    with open(general_file_name) as f:
        data_info = json.load(f)

    with open(train_file_name) as f:
        train_data = json.load(f)['frames']

    with open(test_file_name) as f:
        test_data = json.load(f)['frames']

        # always only use one test image
        if len(test_data) > 0:
            test_data = [test_data[0]]

    # TIGRE CODE
    img_width, img_height = data_info['nDetector']
    near_thresh = data_info['near_thresh']
    far_thresh = data_info['far_thresh']
    max_pixel_value = data_info['max_pixel_value']
    
    # Convert train rays to tensors, extract phase information
    rays_train, phases_train = prepare_data_for_loader_tigre(train_data, data_info, img_width, img_height, run_args.depth_samples_min, 1, device)
    phases_min = np.min(phases_train)
    phases_max = np.max(phases_train)
    phases_train = torch.from_numpy(phases_train).to(device).int()

    # Sample variance rays more densely
    var_ray_ids = np.argwhere(rays_train[:, 2, 0] > 1. + run_args.var_sample_thre/100.).flatten()
    all_ray_ids = np.arange(0, rays_train.shape[0])
    non_var_ray_ids = np.setxor1d(var_ray_ids, all_ray_ids)

    if run_args.var_sample_perc > 0:
        nb_var_rays = 0
        if run_args.var_sample_perc > 0:
            nb_var_rays = int((run_args.var_sample_perc / 100.) * run_args.img_sample_size)
        
        nb_non_var_rays = run_args.img_sample_size - nb_var_rays
    
    rays_train = torch.from_numpy(rays_train).to(device)
    training_sampler = SimpleSampler(rays_train.shape[0], run_args.img_sample_size)
    
    rays_test, phases_test = prepare_data_for_loader_tigre(test_data, data_info, img_width, img_height, run_args.depth_samples_min, 1, device)
    rays_test = torch.from_numpy(rays_test).to(device)
    phases_test = torch.from_numpy(phases_test).to(device).int()

    # Initialize axis-aligned bounding box and sampling resolution
    aabb = torch.tensor([[-run_args.bb_thresh, -run_args.bb_thresh, -run_args.bb_thresh], [run_args.bb_thresh, run_args.bb_thresh, run_args.bb_thresh]]).to(device)
    curr_reso = N_to_reso(run_args.N_voxel_init**3, aabb)
    depth_samples_min = max(run_args.depth_samples_min, cal_n_samples(curr_reso,run_args.step_ratio))
    depth_samples_max = max(run_args.depth_samples_max, depth_samples_min)
    initial_intensities = torch.Tensor([max_pixel_value for _ in np.arange(0, run_args.img_sample_size)]).to(device)

    # voxel list according to upsampling schedule
    upsamp_list = run_args.upsamp_list or []
    if isinstance(upsamp_list, str):
        upsamp_list = json.loads(upsamp_list)

    update_density_list = run_args.update_density_list or []
    if isinstance(update_density_list, str):
        update_density_list = json.loads(update_density_list)

    N_voxel_list = (torch.round(torch.exp(torch.linspace(np.log(run_args.N_voxel_init**3), np.log(run_args.N_voxel_final**3), len(upsamp_list)+1))).long()).tolist()
    if run_args.N_voxel_init == run_args.N_voxel_final:
        N_voxel_list = [np.exp(np.log(run_args.N_voxel_init**3))]
    print(N_voxel_list)
    
    N_voxel_list = N_voxel_list[1:]

    reso_list = [N_to_reso(N_voxel, aabb) for N_voxel in N_voxel_list]
    print(reso_list)

    low_model = TensorVM(aabb, curr_reso, max_pixel_value, device, density_n_comp=[run_args.N_components, run_args.N_components, run_args.N_components], 
                    near_far=[near_thresh, far_thresh], density_shift=run_args.density_shift, step_ratio=run_args.step_ratio)

    sparse_params = {
        'num_early_layers': run_args.mlp_num_early_layers,
        'num_late_layers': run_args.mlp_num_late_layers,
        'num_filters': run_args.mlp_num_channels,
        'num_input_channels': 3,
        'num_input_times': 1,
        'num_output_channels': 1,
        'per_layer_input': run_args.per_layer_input,  
        'use_bias': True,
        'use_time_latents': run_args.use_time_latents,
        'num_time_dim': run_args.mlp_num_time_dim,
        'pos_enc': run_args.pos_enc,
        'pos_enc_window_start': run_args.pos_enc_window_start,
        'pos_enc_basis': run_args.pos_enc_basis,
        'act_func': 'relu',
        'device': device,
        'max_pixel_value': max_pixel_value,
        'near_far': [near_thresh, far_thresh],
    }
    
    sparse_params = dict(sparse_params)
    sparse_model = Temporal(sparse_params)
    sparse_model.to(device)
    
    grad_vars = low_model.get_optparam_groups(run_args.lr)
    grad_vars += [{ 'params': sparse_model.parameters(), 'lr': float(run_args.lr_sparse) }]

    optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))
    lr_factor = run_args.lr_end_factor**(1/run_args.lr_decay_steps)
    lr_factor_sparse = run_args.lr_sparse_end_factor**(1/run_args.lr_sparse_decay_steps)

    renderer = render_hybrid

    tvreg = TVLoss()

    torch.cuda.empty_cache()
    PSNRs = []

    cum_train_time = 0
    occl_weight = float(run_args.occl_weight_start)
    tv_weight = float(run_args.tv_weight_init)
    depth_samples_per_ray = depth_samples_min
    batch_size = run_args.batch_size_start

    training_times = np.array([])

    pbar = tqdm(range(run_args.n_iters+1), miniters=10, file=sys.stdout)

    for n_iter in pbar:
        start_iter_time = time.time()

        low_model.train()
        sparse_model.train()

        if run_args.pos_enc == 'free_windowed' and n_iter >= run_args.model_delay:
            sparse_model.update_freq_mask_alpha(n_iter - run_args.model_delay, run_args.pos_enc_window_decay_steps)
           
        # sample a random amount of ids within the range of the rays
        random_ray_ids = training_sampler.nextids()

        batch_rays = rays_train[random_ray_ids] #(sample_size, or+dr+pix, 3)
        batch_phases = phases_train[random_ray_ids]
        batch_phases_samples = batch_phases[:,None].repeat(1, depth_samples_per_ray) #(sample_size, depth_samples_per_ray)

        int_train = batch_rays[:,2,0] #(sample_size)

        # sample both the low-rank and sparse model
        if n_iter >= run_args.model_delay:
            _, l_sigmas, dists, _, s_sigmas, _ = renderer(batch_rays[:, :2, :], batch_phases_samples, low_model, sparse_model, chunk=batch_size, \
                                                          sample_sparse=True, N_samples=depth_samples_per_ray, device=device, is_train=True)
            sigmas = l_sigmas + s_sigmas
            int_map = initial_intensities - torch.sum(sigmas * dists, dim=-1)
        # initially only optimize the low-rank model
        else:
            int_map, l_sigmas, _, _, _, _ = renderer(batch_rays[:, :2, :], batch_phases_samples, low_model, sparse_model, chunk=batch_size, N_samples=depth_samples_per_ray, device=device, is_train=True)

        loss = torch.mean((int_map - int_train) ** 2)

        total_loss = loss.clone()
        
        loss_tv = tv_weight * low_model.TV_loss_density(tvreg)
        total_loss += loss_tv

        # computes the occlusion regularization for the sparse model
        if n_iter >= run_args.model_delay:
            occl_loss = occl_weight * compute_occl_loss(s_sigmas, dists, reg_perc=run_args.occl_reg_perc)
            total_loss += occl_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        loss = loss.detach().item()
        psnr = -10.0 * np.log(loss) / np.log(10.0)
        PSNRs.append(psnr)

        for i, param_group in enumerate(optimizer.param_groups):
            if i < len(optimizer.param_groups) - 1:
                param_group['lr'] = param_group['lr'] * lr_factor
            elif n_iter >= run_args.model_delay and i == 2:
                param_group['lr'] = param_group['lr'] * lr_factor_sparse

        if n_iter % run_args.log_every == 0:
            train_time = time.time() - start_iter_time
            cum_train_time = cum_train_time + train_time

            with torch.no_grad():
                log_dict = {
                    "train_loss": total_loss, 
                    "train_psnr": psnr,
                    "train_pixel_loss": loss,
                    "train_time": train_time,
                    "cum_train_time": cum_train_time / 60, # in minutes
                    "curr_reso": curr_reso[0],
                    "sigma_l_max": torch.max(l_sigmas),
                    "depth_samples_per_ray": depth_samples_per_ray,
                    "batch_size": batch_size,
                }

                log_dict['loss_tv'] = loss_tv
                log_dict['tv_weight'] = tv_weight

                if n_iter >= run_args.model_delay:
                    log_dict['sigma_s_max'] = torch.max(s_sigmas)

                    if n_iter >= run_args.model_delay > 0:
                        log_dict['loss_occl'] = occl_loss
                        log_dict['occl_weight'] = occl_weight

                if 'windowed' in run_args.pos_enc:
                    log_dict['sparse_windowed'] = sparse_model.position_encoder.ptr

                for i, param_group in enumerate(optimizer.param_groups):
                    log_dict[f'lr_{i}'] = param_group['lr']

                wandb.log(log_dict)

        if n_iter % run_args.display_every == 0:
            low_model.eval()
            
            with torch.no_grad():
                test_phases_samples = phases_test[:,None].repeat(1, depth_samples_per_ray)
                int_map_test, int_map_gt, test_loss, test_psnr, s_int_map_test, d_int_map_test, bin_map_test, dice_score, bin_map_gt = evaluation_hybrid(rays_test, test_phases_samples, low_model, sparse_model, renderer, chunk=batch_size, N_samples=depth_samples_per_ray, data_size=run_args.data_size, device=device, is_train=False)
                
                log_dict = {
                    "test_loss": test_loss, 
                    "test_psnr": test_psnr,
                    "test_pixel_loss": test_psnr,
                    "dice_score": dice_score,
                }

                wandb.log(log_dict)

                int_map_test_np = normalize(int_map_test.cpu().numpy())
                int_map_gt_np = normalize(int_map_gt.cpu().numpy())
                s_int_map_test_np = normalize(s_int_map_test.cpu().numpy())
                d_int_map_test_np = normalize(d_int_map_test.cpu().numpy())
                bin_map_test_np = normalize(bin_map_test.cpu().numpy())
                bin_map_gt_np = normalize(bin_map_gt.cpu().numpy())

                image_log = {
                    'prediction': wandb.Image(int_map_test_np),
                    'low': wandb.Image(s_int_map_test_np),
                    'sparse': wandb.Image(d_int_map_test_np),
                    'original': wandb.Image(int_map_gt_np),
                    'difference': wandb.Image(np.abs(int_map_test_np-int_map_gt_np)),
                    'binary': wandb.Image(bin_map_test_np),
                    'binary_original': wandb.Image(bin_map_gt_np)
                }

                wandb.log(image_log)

                print_string = f'Iteration {n_iter:05d}:' \
                    + f' train_psnr = {float(np.mean(PSNRs)):.2f}' \
                    + f' test_psnr = {float(test_psnr):.2f}' \
                    + f' mse = {loss:.6f}'
                pbar.set_description(print_string)

                PSNRs = []

                torch.cuda.empty_cache()
                gc.collect()

        if n_iter == run_args.model_delay and n_iter > 0:
            # reset learning rates
            grad_vars = low_model.get_optparam_groups(run_args.lr)
            grad_vars += [{ 'params': sparse_model.parameters(), 'lr': float(run_args.lr_sparse) }]

            optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))
        
        if n_iter in update_density_list:
            sparse_model.update_density_mask()

        if n_iter % run_args.save_every == 0:
            with torch.no_grad():
                low_model.save(f'{log_dir}low_model.pth')
                sparse_model.save(f'{log_dir}sparse_model.pth', {})

                low_model.save(f'{log_dir}low_model-{n_iter}.pth')
                sparse_model.save(f'{log_dir}sparse_model-{n_iter}.pth', {})

                # store current cumulative training time
                training_times = np.append(training_times, cum_train_time)
                np.savetxt(f'{log_dir}train.txt', np.around(training_times / 60, decimals=5))

            torch.cuda.empty_cache()
            gc.collect()

    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    torch.manual_seed(20250302) # 02-03-2025
    np.random.seed(20250302)

    parser = config_parser_hybrid()
    run_args = parser.parse_args()

    # render mesh based on setup
    wandb.login()

    project_name = 'Hybrid'
    if run_args.use_wandb:
        with open(run_args.wandb_sweep_yaml, 'r') as f:
            sweep_configuration = yaml.load(f, Loader=yaml.FullLoader)

        sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)
        wandb.agent(sweep_id, function=main)
    else:
        main()