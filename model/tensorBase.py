import torch
import torch.nn.functional as F

import time
import numpy as np

# Helper function to perturb depth values (for stratified sampling during training)
def randomize_depth(z_vals):
    mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.concat([mids, z_vals[..., -1:]], -1)
    lower = torch.concat([z_vals[..., :1], mids], -1)
    t_rand = torch.rand(z_vals.shape).to(z_vals)
    z_vals = lower + (upper - lower) * t_rand

    return z_vals

# Class for masking 3D space using a precomputed density volume
class DensityGridMask(torch.nn.Module):
    def __init__(self, device, aabb, density_volume):
        super(DensityGridMask, self).__init__()
        self.device = device

        self.aabb = aabb.to(self.device)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invgridSize = 1.0/self.aabbSize * 2 # scale to [-1, 1]
        self.density_volume = density_volume.view(1,1,*density_volume.shape[-3:]) # format for grid_sample
        self.gridSize = torch.LongTensor([density_volume.shape[-1],density_volume.shape[-2],density_volume.shape[-3]]).to(self.device)

    def sample_density(self, xyz_sampled):
        xyz_sampled = self.normalize_coord(xyz_sampled)
        # Use trilinear sampling to get density
        density_vals = F.grid_sample(self.density_volume, xyz_sampled.view(1,-1,1,1,3), align_corners=True).view(-1)

        return density_vals

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled-self.aabb[0]) * self.invgridSize - 1

# Simple passthrough rendering (optional override with MLPRender)
def DensityRender(features):
    sigma = features
    return sigma

# Optional MLP for rendering from feature vector
class MLPRender(torch.nn.Module):
    def __init__(self, num_input, num_channels=128):
        super(MLPRender, self).__init__()
        
        layer1 = torch.nn.Linear(num_input, num_channels)
        layer2 = torch.nn.Linear(num_channels, num_channels)
        layer3 = torch.nn.Linear(num_channels, 1)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, features):
        sigma = self.mlp(features.T)
        return sigma.flatten()

# Base class for a tensor field representation
class TensorBase(torch.nn.Module):
    def __init__(self, aabb, gridSize, max_pixel_value, device, density_n_comp = 8, density_mask=None, near_far=[2.0, 6.0],
                 density_shift = -10, step_ratio=2.0, fea2denseAct = 'softplus'):
        super(TensorBase, self).__init__()

        self.density_n_comp = density_n_comp
        self.aabb = aabb
        self.density_mask = density_mask
        self.device = device
        
        self.max_pixel_value = max_pixel_value
        self.density_thresh = 1e-4

        self.density_shift = density_shift
        self.fea2denseAct = fea2denseAct

        self.near_far = near_far
        self.step_ratio = step_ratio

        self.update_stepSize(gridSize)

        # Volume modes
        self.matMode = [[0,1], [0,2], [1,2]]
        self.vecMode =  [2, 1, 0]
        self.comp_w = [1,1,1]

        self.init_svd_volume(gridSize[0], device)
        self.init_render_func('') # default = simple sigma return

    # Initialize the rendering function based on the selected mode
    def init_render_func(self, mode):
        if mode == 'MLP':
            self.renderModule = MLPRender(self.density_n_comp[0], num_channels=8).to(self.device)
        else:
            self.renderModule = DensityRender

    # Update step size and grid resolution parameters
    def update_stepSize(self, gridSize):
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0/self.aabbSize
        self.gridSize= torch.LongTensor(gridSize).to(self.device)

        self.units=self.aabbSize / (self.gridSize-1)
        self.stepSize=torch.mean(self.units)*self.step_ratio
        self.aabbDiag = torch.sqrt(torch.sum(torch.square(self.aabbSize)))
        self.nSamples=int((self.aabbDiag / self.stepSize).item()) + 1
        
    def init_svd_volume(self, res, device):
        pass

    def compute_features(self, xyz_sampled):
        pass
    
    def compute_densityfeature(self, xyz_sampled):
        pass
    
    def is_fourier_model(self):
        return False
    
    def is_rank_model(self):
        return False

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled-self.aabb[0]) * self.invaabbSize - 1

    def get_optparam_groups(self, lr_init_spatial = 0.02, lr_init_network = 0.001):
        pass

    def get_kwargs(self):
        return {
            'max_pixel_value': self.max_pixel_value,
            'aabb': self.aabb,
            'gridSize':self.gridSize.tolist(),
            'density_n_comp': self.density_n_comp,

            'density_shift': self.density_shift,
            'fea2denseAct': self.fea2denseAct,

            'near_far': self.near_far,
            'step_ratio': self.step_ratio,
        }

    def save(self, path):
        kwargs = self.get_kwargs()
        ckpt = {'kwargs': kwargs, 'state_dict': self.state_dict()}

        if self.density_mask is not None:
            density_volume = self.density_mask.density_volume.bool().cpu().numpy()
            ckpt.update({'density_mask.shape': density_volume.shape})
            ckpt.update({'density_mask.mask':np.packbits(density_volume.reshape(-1))})
            ckpt.update({'density_mask.aabb': self.density_mask.aabb.cpu()})
        torch.save(ckpt, path)

    def load(self, ckpt):
        if 'density_mask.aabb' in ckpt.keys():
            length = np.prod(ckpt['density_mask.shape'])
            density_volume = torch.from_numpy(np.unpackbits(ckpt['density_mask.mask'])[:length].reshape(ckpt['density_mask.shape']))
            self.density_mask = DensityGridMask(self.device, ckpt['density_mask.aabb'].to(self.device), density_volume.float().to(self.device))
        self.load_state_dict(ckpt['state_dict'])

    # Generate sampling depths (interpx) for ray marching
    def get_interpx(self, is_train = True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        near, far = self.near_far

        t_vals = torch.linspace(0., 1., N_samples)
        interpx = near * (1.-t_vals) + far * (t_vals)
        interpx = interpx.to(self.device)

        if is_train:
            interpx = randomize_depth(interpx) # Add stratified noise

        return interpx

    # Sample 3D points along rays given origin, direction, and depth values
    def sample_ray(self, rays_o, rays_d, interpx):
        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., :, None]
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(dim=-1)

        return rays_pts, interpx, ~mask_outbbox

    def shrink(self, new_aabb, voxel_size):
        pass

    # Generate a dense grid of points and compute sigma for all voxels
    @torch.no_grad()
    def get_grid(self, gridSize=None, aabb=None):
        gridSize = self.gridSize if gridSize is None else gridSize

        aabb = self.aabb.cpu().numpy() if aabb is None else aabb

        xs = torch.linspace(aabb[0][0], aabb[1][0], gridSize[0])
        ys = torch.linspace(aabb[0][1], aabb[1][1], gridSize[1])
        zs = torch.linspace(aabb[0][2], aabb[1][2], gridSize[2])

        dense_xyz = np.array(np.meshgrid(xs, ys, zs))

        xyz = torch.meshgrid(xs, ys, zs)
        dense_samples = torch.stack(xyz, -1).to(self.device)

        sigma = torch.zeros_like(dense_samples[...,0])
        for i in range(gridSize[0]):
            sigma[i] = self.compute_sigma(dense_samples[i].view(-1,3)).view((gridSize[1], gridSize[2]))

        return sigma, dense_xyz, dense_samples
    
    # Compute densities at a set of 3D locations
    def compute_sigma(self, xyz_locs):
        if self.density_mask is not None:
            densities = self.density_mask.sample_density(xyz_locs)
            density_mask = densities > 0
        else:
            density_mask = torch.ones_like(xyz_locs[:,0], dtype=bool)

        sigma = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device)
        if density_mask.any():
            xyz_sampled = self.normalize_coord(xyz_locs[density_mask])
            sigma_feature = self.compute_densityfeature(xyz_sampled)
            render_feature = self.renderModule(sigma_feature)
            valid_sigma = self.feature2density(render_feature)
            sigma[density_mask] = valid_sigma

        sigma = sigma.view(xyz_locs.shape[:-1])
        return sigma
    
    # Update density mask by max-pooling and thresholding a 3D sigma grid
    @torch.no_grad()
    def update_density_mask(self, gridSize=(200,200,200)):

        densities, _, dense_xyz = self.get_grid(gridSize)
        dense_xyz = dense_xyz.transpose(0,2).contiguous()
        densities = densities.clamp(0,1).transpose(0,2).contiguous()[None,None]
        total_voxels = gridSize[0] * gridSize[1] * gridSize[2]

        ks = 3
        densities = F.max_pool3d(densities, kernel_size=ks, padding=ks // 2, stride=1).view(gridSize[::-1])
        densities[densities >= self.density_thresh] = 1
        densities[densities < self.density_thresh] = 0

        self.density_mask = DensityGridMask(self.device, self.aabb, densities)

        valid_xyz = dense_xyz[densities>0.5]

        xyz_min = valid_xyz.amin(0)
        xyz_max = valid_xyz.amax(0)

        new_aabb = torch.stack((xyz_min, xyz_max))

        total = torch.sum(densities)
        print(f"bbox: {xyz_min, xyz_max} density rest %%%f"%(total/total_voxels*100))
        return new_aabb

    # Remove rays that don't intersect the density volume
    @torch.no_grad()
    def filtering_rays(self, all_rays, N_samples=256, chunk=10240*5, bbox_only=False):
        print('========> filtering rays ...')
        tt = time.time()

        N = torch.tensor(all_rays.shape[0])

        mask_filtered = []
        idx_chunks = torch.split(torch.arange(N), chunk)
        for idx_chunk in idx_chunks:
            rays_chunk = all_rays[idx_chunk].to(self.device)

            rays_o, rays_d = rays_chunk[:, 0], rays_chunk[:, 1]
            interpx = interpx if len(interpx) > 0 else self.get_interpx(is_train=False, N_samples=N_samples)
            xyz_sampled, _,_ = self.sample_ray(rays_o, rays_d, interpx)

            # Keep rays that intersect a non-zero density region
            mask_inbbox= (self.density_mask.sample_density(xyz_sampled).view(xyz_sampled.shape[:-1]) > 0).any(-1)
            
            mask_filtered.append(mask_inbbox.cpu())

        mask_filtered = torch.cat(mask_filtered).tolist()

        print(f'Ray filtering done! takes {time.time()-tt} s. ray mask ratio: {np.sum(mask_filtered) / N}')
        return all_rays[mask_filtered]

    def feature2density(self, density_features):
        if self.fea2denseAct == "softplus":
            return F.softplus(density_features+self.density_shift)
        elif self.fea2denseAct == "relu":
            return F.relu(density_features)

    def forward(self, rays_chunk, interpx=[], is_train=True, N_samples=-1):
        # sample points
        viewdirs = rays_chunk[:, 1]

        # Sample 3D points along each ray
        interpx = interpx if len(interpx) > 0 else self.get_interpx(is_train=is_train, N_samples=N_samples)
        xyz_sampled, z_vals, ray_valid = self.sample_ray(rays_chunk[:, 0], viewdirs, interpx)
        
        # Compute distances between sampled points
        one_e_10 = torch.tensor([1e-10], dtype=viewdirs.dtype, device=viewdirs.device)
        dists = torch.cat((z_vals[..., 1:] - z_vals[..., :-1], one_e_10.expand(z_vals[..., :1].shape)), dim=-1)
        rays_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
        dists = dists * rays_norm
        viewdirs = viewdirs / rays_norm

        # Mask out points outside valid density regions
        if self.density_mask is not None:
            densities = self.density_mask.sample_density(xyz_sampled[ray_valid])
            density_mask = densities > 0
            ray_invalid = ~ray_valid
            ray_invalid[ray_valid] |= (~density_mask)
            ray_valid = ~ray_invalid

        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)

        # Compute sigma only for valid rays
        if ray_valid.any():
            xyz_sampled = self.normalize_coord(xyz_sampled)
            sigma_feature = self.compute_densityfeature(xyz_sampled[ray_valid])
            render_feature = self.renderModule(sigma_feature)
            validsigma = self.feature2density(render_feature)
            sigma[ray_valid] = validsigma

        # Perform absorption-based rendering
        initial_intensities = torch.tensor([self.max_pixel_value], dtype=viewdirs.dtype, device=viewdirs.device).expand(sigma.shape[0])
        int_map = initial_intensities - torch.sum(sigma * dists, dim=-1)

        return int_map, sigma, dists