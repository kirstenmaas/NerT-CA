import numpy as np
import matplotlib.pyplot as plt
import torch
import pyvista as pv
import json
import os
import tigre
from skimage.filters import frangi, threshold_otsu

from proj_helpers import source_matrix

# Forward project a volume using TIGRE with specified geometry and angles
def forward_project_geo(vol, geo, angles):
    if geo.nVoxel[0] != vol.shape[0]:
        vol = np.transpose(vol, (2, 1, 0)).copy()

    projections = tigre.Ax(vol, geo, angles, projection_type='interpolated')[0]
    return projections

# Compute and save TIGRE projection images, store metadata
def obtain_img_and_store_tigre(image_id, vol, geo, theta, phi, larm, hrt_phase, int_hrt_phase, resp_phase, max_pixel_value, images_weighted_dict, train_folder_name):
    view_point_key = f'{theta}-{phi}'
    image_id_str = f'image-hrt={int_hrt_phase}-resp={int(resp_phase)}-angles={view_point_key}'

    angles = np.deg2rad(np.vstack([-theta, phi, 0]).T)

    img = forward_project_geo(vol, geo, angles)

    img = np.flipud(img)

    img_t = max_pixel_value * np.exp(-img)
    log_img_t = np.log(img_t)

    norm_img, img_min, img_max = normalize(log_img_t)

    plt.imsave(f'{train_folder_name}{image_id_str}.png', norm_img, cmap='gray')
    np.save(f'{train_folder_name}{image_id_str}.npy', norm_img)

    images_weighted_dict = maintain_weighted_img_dict(log_img_t, view_point_key, images_weighted_dict)

    data_frame = store_train_img(image_id, image_id_str, img_min, img_max, view_point_key, resp_phase, int_hrt_phase, hrt_phase, theta, phi, larm, train_folder_name)

    return data_frame, images_weighted_dict, log_img_t

# Create a dictionary with image metadata for storage and retrieval
def store_train_img(image_id, image_id_str, img_min, img_max, view_point_key, resp_phase, int_hrt_phase, hrt_phase, theta, phi, larm, train_folder_name):
    data_frame = {}

    data_frame['image_id_str'] = image_id_str
    data_frame['image_id'] = image_id
    data_frame['file_path'] = f'{train_folder_name}{image_id_str}.npy'
    data_frame['img_min_max'] = [img_min.astype('float64'), img_max.astype('float64')]
    data_frame['weighted_file_path'] = f'{train_folder_name}image-{view_point_key}-var.npy'
    data_frame['mask_file_path'] = f'{train_folder_name}{image_id_str}-mask.npy'
    data_frame['resp_phase'] = resp_phase
    data_frame['heart_phase'] = int_hrt_phase 
    data_frame['org_heart_phase'] = int(hrt_phase)
    data_frame['theta'] = float(theta)
    data_frame['phi'] = float(phi)
    data_frame['larm'] = float(larm)

    return data_frame

# Generate ray origins and directions from TIGRE geometry and viewing angless
def get_ray_values_tigre(theta, phi, larm, geo, device):
    src_pt = np.array([0, 0, -geo.DSO])

    src_matrix = source_matrix(src_pt, theta, phi, larm)
    pose = torch.from_numpy(src_matrix).to(device).float()

    img_width, img_height = geo.nDetector

    ii, jj = torch.meshgrid(
        torch.linspace(0, img_width-1, img_width).to(device),
        torch.linspace(0, img_height-1, img_height).to(device),
        indexing='xy'
    )

    uu = (ii.t() + 0.5 - img_width / 2) * geo.dDetector[0] + geo.offDetector[0]
    vv = (jj.t() + 0.5 - img_height / 2) * geo.dDetector[1] + geo.offDetector[1]
    dirs = torch.stack([uu / geo.DSD, vv / geo.DSD, torch.ones_like(uu)], -1)

    ray_directions = torch.sum(torch.matmul(pose[:3,:3], dirs[..., None]).to(device), -1) # pose[:3, :3] * 
    ray_origins = pose[:3, -1].expand(ray_directions.shape)

    return ray_origins, ray_directions, src_matrix, ii, jj

# Sample depth values along rays for volume rendering or projection
def get_depth_values(near_thresh, far_thresh, depth_samples_per_ray, device, stratified=True):
    t_vals = torch.linspace(0., 1., depth_samples_per_ray)
    z_vals = near_thresh * (1.-t_vals) + far_thresh * (t_vals)

    mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.concat([mids, z_vals[..., -1:]], -1)
    lower = torch.concat([z_vals[..., :1], mids], -1)

    if stratified:
        t_rand = torch.rand(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand
    depth_values = z_vals.to(device)
    return depth_values

# Perform ray tracing by querying interpolated volume along sampled rays
def ray_tracing(interpolator, angles, ray_origins, ray_directions, depth_values, img_width, img_height, ii, jj, batch_size, device, proj_folder_name, type='ct', max_pixel_value=8.670397):
    img = torch.ones((int(np.ceil(img_width)), int(np.ceil(img_height))))

    if batch_size > img_width or batch_size > img_height:
        batch_size = min(img_width, img_height)

    for i_index in range(0, ii.shape[0], batch_size):
        for j_index in range(0, jj.shape[0], batch_size):

            query_points = ray_origins[..., None, :][i_index:i_index+batch_size, j_index:j_index+batch_size] + ray_directions[..., None, :][i_index:i_index+batch_size, j_index:j_index+batch_size] * depth_values[..., :, None]

            one_e_10 = torch.tensor([1e-10], dtype=depth_values.dtype, device=depth_values.device)
            dists = torch.cat((depth_values[..., 1:] - depth_values[..., :-1], one_e_10.expand(depth_values[..., :1].shape)), dim=-1)

            ray_points = query_points.cpu().numpy().reshape((-1, 3))

            interp_vals = torch.from_numpy(interpolator(ray_points)).to(device).reshape(query_points.shape[:-1])

            if type == 'ct':
                norm_dists = dists
                weights = interp_vals * norm_dists
                base_intensity = max_pixel_value
                img_val = base_intensity * torch.exp(-torch.sum(weights, dim=-1))
            elif type == 'mip':
                img_val = torch.max(interp_vals, dim=-1).values

            img[i_index:i_index+batch_size,j_index:j_index+batch_size] = img_val

    return img

# Maintain dictionary of images grouped by view point
def maintain_weighted_img_dict(img, view_point_key, images_weighted_dict):
    if view_point_key not in images_weighted_dict:
        images_weighted_dict[view_point_key] = [img]
    else:
        new_images_weighted = images_weighted_dict[view_point_key]
        new_images_weighted.append(img)

        images_weighted_dict[view_point_key] = new_images_weighted
    return images_weighted_dict

# Compute variance-weighted projections from grouped images
def obtain_weighted_imgs(view_point_keys, images_weighted_dict, weighted_pixel_values, weighted_method, img_width, img_height, train_folder_name):
    images_weighted = []
    view_point_vars = {}
    for view_point_key in view_point_keys:
        if not view_point_key in view_point_vars:
            view_point_imgs = np.array(images_weighted_dict[view_point_key])
            view_point_imgs = view_point_imgs.reshape((-1, img_width*img_height))
            
            view_point_var_pix = np.ones((img_width, img_height))
            if weighted_method == 'var' and len(view_point_imgs) > 1:
                view_point_imgs = np.exp(-view_point_imgs)
                view_point_var = np.var(view_point_imgs, axis=0)
                view_point_var_pix = view_point_var.reshape((img_width, img_height))
                view_point_var_pix = (view_point_var_pix - np.min(view_point_var_pix)) / (np.max(view_point_var_pix) - np.min(view_point_var_pix)+1e-10)

            plt.imsave(f'{train_folder_name}image-{view_point_key}-var.png', view_point_var_pix, cmap='Reds')

            view_point_vars[view_point_key] = view_point_var_pix + 1
            np.save(f'{train_folder_name}image-{view_point_key}-var.npy', view_point_var_pix+1)

            images_weighted.append(view_point_vars[view_point_key].tolist())
            weighted_pixel_values.append(view_point_vars[view_point_key].flatten())
    plt.close()
    
    return images_weighted, weighted_pixel_values

# Generate vessel masks from projection images using Frangi filters
def obtain_vessel_masks(view_point_keys, images_weighted_dict, max_pixel_value, img_width, img_height, train_folder_name, dataset):
    for view_point_key in view_point_keys:
        view_point_imgs = np.array(images_weighted_dict[view_point_key])

        view_point_imgs = view_point_imgs.reshape((-1, img_width, img_height))

        for i in range(view_point_imgs.shape[0]):
            image = view_point_imgs[i]
            image_id_str = f'image-hrt={i}-resp={0}-angles={view_point_key}'
            norm_img = (image - np.min(image)) / (np.max(image) - np.min(image))
            frangi_gamma = 1e-2 if dataset == 'XCAT-V1' else 3e-3
            mask = frangi(norm_img, gamma=frangi_gamma)
            thresh = threshold_otsu(mask)
            binary = (mask > thresh)
            dynamic_mask = np.zeros_like(image)
            dynamic_mask[dynamic_mask == 0] = max_pixel_value
            dynamic_mask[binary == 1] = image[binary == 1]
            plt.imsave(f'{train_folder_name}{image_id_str}-mask.png', dynamic_mask, cmap='gray')
            np.save(f'{train_folder_name}{image_id_str}-mask.npy', dynamic_mask)
    return None

# Visualize camera geometry, rays, and sample query points in 3D
def visualize_geometry_tigre(train_view_points, grid, geo, depth_values, interpolator, device):
    fig = plt.figure(figsize=(8,8))
    
    sub_fig = fig.add_subplot(projection='3d')
    sub_fig.set_xlabel('X Label')
    sub_fig.set_ylabel('Y Label')
    sub_fig.set_zlabel('Z Labefl')

    ax_boundary = 15
    sub_fig.set_xlim3d(-ax_boundary, ax_boundary)
    sub_fig.set_ylim3d(-ax_boundary, ax_boundary)
    sub_fig.set_zlim3d(-ax_boundary, ax_boundary)

    colors = ['red', 'green', 'blue']
    x_pt = np.array([1, 0, 0, 1])
    y_pt = np.array([0, 1, 0, 1])
    z_pt = np.array([0, 0, 1, 1])
    pts = [x_pt, y_pt, z_pt]

    for i, pt in enumerate(pts):
        points = np.array([[0,0,0], pt[:3]]).T
        sub_fig.plot(points[0, :], points[1, :], points[2, :], c=colors[i])

    img_width, img_height = geo.nDetector

    sub_fig = visualize_volume_bounds(sub_fig, grid.bounds)

    for j, viewpoint in enumerate(train_view_points):
        theta, phi = viewpoint

        ray_origins, ray_directions, src_matrix, ii, jj = get_ray_values_tigre(theta, phi, 0., geo, device)
    
        source_o = src_matrix.dot(np.array([0, 0, 0, 1]))

        source_x = src_matrix.dot(x_pt)
        source_y = src_matrix.dot(y_pt)
        source_z = src_matrix.dot(z_pt)
        source_pts = [source_x, source_y, source_z]

        sub_fig.scatter(source_o[0], source_o[1], source_o[2], c='black')
        sub_fig.text(source_o[0], source_o[1], source_o[2], f'{theta}-{phi}', size=20, zorder=1, color='k')
        
        for i, pt in enumerate(source_pts):
            points = np.array([source_o[:3], pt[:3]]).T
            sub_fig.plot(points[0, :], points[1, :], points[2, :], c=colors[i])
        
        detector_o = ray_origins[0,0] + ray_directions[0,0] * geo.DSD
        detector_y = ray_origins[0,0] + ray_directions[0,img_height-1] * geo.DSD
        detector_x = ray_origins[0,0] + ray_directions[img_width-1,0] * geo.DSD
        detector_pts = [detector_x, detector_y]
        for i, pt in enumerate(detector_pts):
            points = np.array([detector_o[:3].cpu().numpy(), pt[:3].cpu().numpy()]).T
            sub_fig.plot(points[0, :], points[1, :], points[2, :], c=colors[i])
        
        visualize_query_points(sub_fig, img_width, img_height, ray_origins, ray_directions, depth_values, grid_scaling_factor=1)

        if j == 0:
            query_points = (ray_origins[..., None, :] + ray_directions[..., None, :] * depth_values[..., :, None]).cpu().numpy()
            query_points = query_points.reshape((-1, 3))
            query_scalars = interpolator(query_points)
            volume_scalar_ids = np.argwhere(query_scalars > 0)
            for k in range(0, volume_scalar_ids.shape[0], int(volume_scalar_ids.shape[0]*0.01)):
                vol_id = volume_scalar_ids[k]
                vol_query_point = query_points[vol_id].reshape(3)
                sub_fig.scatter(vol_query_point[0], vol_query_point[1], vol_query_point[2], color='grey')
            artery_scalar_ids = np.argwhere(query_scalars >= 0.05)
            for k in range(0, artery_scalar_ids.shape[0], 25):
                artery_id = artery_scalar_ids[k]
                artery_query_point = query_points[artery_id].reshape(3)
                sub_fig.scatter(artery_query_point[0], artery_query_point[1], artery_query_point[2], color='purple')

    plt.show()

# Visualize the boundary of the 3D volume
def visualize_volume_bounds(sub_fig, grid_bounds):
    x_bounds = grid_bounds[0:2]
    y_bounds = grid_bounds[2:4]
    z_bounds = grid_bounds[4:6]
    for x in x_bounds:
        for y in y_bounds:
            for z in z_bounds:
                sub_fig.scatter(x, y, z, color='black')
    return sub_fig

# Plot sample rays and their intersection with the volume
def visualize_query_points(sub_fig, img_width, img_height, ray_origins, ray_directions, depth_values, grid_scaling_factor=1):
    reg_perc = 0
    cum_dists = torch.cumsum(depth_values, dim=0)
    
    dists_range_front = reg_perc * cum_dists[-1]

    mask_front = torch.where(cum_dists < dists_range_front, 1., 0.).int()
    
    last_ray_index = torch.argmin(mask_front) - 1

    x_pixs = [0, img_width//2-1, img_width-1]
    y_pixs = [0, img_height//2-1, img_height-1]
    for x in x_pixs:
        for y in y_pixs:
            query_points = (ray_origins[..., None, :][x, y] + ray_directions[..., None, :][x, y] * depth_values[..., :, None]).cpu().numpy()

            point1 = query_points[0]
            point2 = query_points[-1]
            points = np.array([point1, point2]).T

            sub_fig.plot(points[0, :], points[1, :], points[2, :], c='grey', alpha=0.5)

            point3 = query_points[last_ray_index]
            points = np.array([point1, point3]).T
            sub_fig.plot(points[0, :], points[1, :], points[2, :], c='red')
    return sub_fig

# Load a volume and convert it to a VTK grid for visualization
def load_vol_grid(vol_file_name, dimensions, case_folder_name):
    vol_np = np.load(f'{case_folder_name}/{vol_file_name}')
    vol_vtk = np_to_vtk(vol_np, dimensions)

    return vol_vtk, vol_np

# Convert a NumPy volume into a VTK structured grids
def np_to_vtk(np_vol, dimensions):
    xs = np.linspace(0, dimensions[0], dimensions[0])
    ys = np.linspace(0, dimensions[1], dimensions[1])
    zs = np.linspace(0, dimensions[2], dimensions[2])
    mesh_grid = np.array(np.meshgrid(xs, ys, zs))

    grid = pv.StructuredGrid(mesh_grid[0], mesh_grid[1], mesh_grid[2])
    grid.point_data['scalars'] = np_vol.flatten()

    return grid

# Normalize an image to range [0, 1]
def normalize(img):
    img_max = np.max(img)
    img_min = np.min(img)
    norm_img = (img - img_min) / (img_max - img_min)
    return norm_img, img_min, img_max

# Return TIGRE projection geometry for XCAT data based on resolution
def get_xcat_properties_tigre(data_size, vol_dimensions):
    if data_size == 200:
        geo_data = {
            "DSD": 2500,
            "DSO": 450, #4,
            "nDetector": [200, 200],
            "dDetector": [1, 1],
            "nVoxel": vol_dimensions,
            "dVoxel": [0.25,0.25,0.25],
            "offOrigin": [10,-25,25],
            "offDetector": [0,0],
            "accuracy": 0.5,
            "mode": "cone",
            "filter": None,
        }
    elif data_size == 50:
        geo_data = {
            "DSD": 2500,
            "DSO": 450, #4,
            "nDetector": [50, 50],
            "dDetector": [4, 4],
            "nVoxel": vol_dimensions,
            "dVoxel": [0.25,0.25,0.25],
            "offOrigin": [10,-25,25],
            "offDetector": [0,0],
            "accuracy": 0.5,
            "mode": "cone",
            "filter": None,
        }
    else:
        print('UNKNOWN DATA SIZE')
        return
    return geo_data

# Return TIGRE projection geometry for CCTA data based on resolution
def get_ccta_properties_tigre(data_size, vol_dimensions):
    if data_size == 200:
        geo_data = {
            "DSD": 2000,
            "DSO": 600,
            "nDetector": [200, 200],
            "dDetector": [1,1],
            "nVoxel": vol_dimensions,
            "dVoxel": [0.9, 0.9, 0.9],
            "offOrigin": [0, 0, 0],
            "offDetector": [0,0],
            "accuracy": 0.5,
            "mode": "cone",
            "filter": None,
        }
    elif data_size == 50:
        geo_data = {
            "DSD": 2000,
            "DSO": 600,
            "nDetector": [50, 50],
            "dDetector": [4, 4],
            "nVoxel": vol_dimensions,
            "dVoxel": [0.9, 0.9, 0.9],
            "offOrigin": [0, 0, 0],
            "offDetector": [0,0],
            "accuracy": 0.5,
            "mode": "cone",
            "filter": None,
        }
    else:
        print('UNKNOWN DATA SIZE')
        return
    return geo_data

# Setup training/testing projection viewpoints based on parameters or experiment definition
def setup_experiment_type(data_args, train_folder_name):
    if data_args.use_experiment_name:
        train_file_name = f"{train_folder_name}train-{data_args.experiment_name}.json"
        test_file_name = f"{train_folder_name}test-{data_args.experiment_name}.json"

        experiment_info_path = f"preprocess/xcat/{data_args.experiment_name}.json"
        with open(experiment_info_path) as f:
            phase_volume_lst = json.load(f)
    else:
        time_steps = np.arange(data_args.data_time_range_start, data_args.data_time_range_end) / 10

        if data_args.data_limited_range_test and data_args.data_step_size_test:
            limited_range_test = data_args.data_limited_range_test
            step_size_test = data_args.data_step_size_test

            test_theta_angles = np.arange(-limited_range_test, limited_range_test+1, step_size_test)
            test_phi_angles = np.arange(-limited_range_test, limited_range_test+1, step_size_test)
            test_angles = np.array(np.meshgrid(test_theta_angles, test_phi_angles, indexing='ij')).reshape((2, -1)).T
            test_angles = np.insert(test_angles, 0, [0, -90], axis=0)
        else:
            # LAO = +theta, RAO = -theta
            # CRA = +phi, CAU = -phi
            test_angles = np.array([[-5, 40], [-5, -40], [90, 0], [-30, 0]])

        limited_range = data_args.data_limited_range
        step_size = data_args.data_step_size #15
        numb_angles = data_args.data_numb_angles

        if step_size <= limited_range:
            theta_angles = np.arange(-limited_range, limited_range+1, step_size)
            phi_angles = np.arange(-limited_range, limited_range+1, step_size)
            set_angle_comb = np.array(np.meshgrid(theta_angles, phi_angles, indexing='ij')).reshape((2, -1)).T

            close_thresh = 15
            angle_comb = []
            for train_angle in set_angle_comb:
                far_away = True
                for test_angle in test_angles:
                    diff_angle = np.abs(np.array(train_angle) - np.array(test_angle))
                    if np.sum(diff_angle) <= close_thresh:
                        far_away = False
                if far_away:
                    angle_comb.append(train_angle)
            angle_comb = np.array(angle_comb)
            
            if angle_comb.shape[0] != set_angle_comb.shape[0]:
                print('Removed some to avoid too close to validation angles!')

            if angle_comb.shape[0] == 4:
                four_angles = np.array([[-30, 30], [-30, -30], [60, -30], [60, 30]])
                angle_comb = four_angles
        elif numb_angles != None:
            if numb_angles == 4:
                predf_angles = np.array([[-30, 30], [-30, -30], [60, -30], [60, 30]])
            elif numb_angles == 3:
                predf_angles = np.array([[-30, -30], [60, -30], [60, 30]])
            elif numb_angles == 2:
                predf_angles = np.array([[-30, -30], [60, 30]])
            angle_comb = predf_angles
        else:
            return Exception()

        train_file_name = f"{train_folder_name}train-{float(limited_range)}-{float(step_size)}-{data_args.data_time_range_start}-{data_args.data_time_range_end}.json"
        test_file_name = f"{train_folder_name}test-{float(limited_range)}-{float(step_size)}-{data_args.data_time_range_start}-{data_args.data_time_range_end}.json"
        
        phase_volume_lst = []
        for i, time_step in enumerate(time_steps):
            phase_obj = {
                "hrt_phase": time_step,
                "resp_phase": 0, 
                "train_viewpoints": angle_comb, 
                "test_viewpoints": [],
            }

            phase_obj['test_viewpoints'] = test_angles
            
            phase_volume_lst.append(phase_obj)

    render_train_imgs = not os.path.isfile(train_file_name)
    render_test_imgs = not os.path.isfile(test_file_name)

    data_per_view = {}
    data_per_view['frames'] = []

    data_per_view_test = {}
    data_per_view_test['frames'] = []

    return train_file_name, render_train_imgs, data_per_view, test_file_name, render_test_imgs, data_per_view_test, phase_volume_lst