import torch
import torch.nn as nn
import torch.nn.functional as F

from .Encoder import FourierEncoding, NoEncoding, FreeEncoding, SimpleEncoding

# Density grid mask used for pruning 3D space during volume rendering
class DensityGridMask(torch.nn.Module):
    def __init__(self, device, aabb, density_volume):
        super(DensityGridMask, self).__init__()
        self.device = device

        # Define bounding box and scale factors
        self.aabb = aabb.to(self.device)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invgridSize = 1.0/self.aabbSize * 2

        # Prepare density volume for sampling
        self.density_volume = density_volume.view(1,1,*density_volume.shape[-3:])
        self.gridSize = torch.LongTensor([density_volume.shape[-1],density_volume.shape[-2],density_volume.shape[-3]]).to(self.device)

    def sample_density(self, xyz_sampled):
        # Normalize and sample densities using grid_sample
        xyz_sampled = self.normalize_coord(xyz_sampled)
        density_vals = F.grid_sample(self.density_volume, xyz_sampled.view(1,-1,1,1,3), align_corners=True).view(-1)
        return density_vals

    def normalize_coord(self, xyz_sampled):
        # Normalize coordinates from [aabb] to [-1, 1]
        return (xyz_sampled-self.aabb[0]) * self.invgridSize - 1

def randomize_depth(z_vals):
    # Stratified sampling of depth values for rendering
    mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.concat([mids, z_vals[..., -1:]], -1)
    lower = torch.concat([z_vals[..., :1], mids], -1)
    t_rand = torch.rand(z_vals.shape).to(z_vals)
    z_vals = lower + (upper - lower) * t_rand
    return z_vals

class Temporal(nn.Module):
    def __init__(self, model_definition: dict) -> None:
        super().__init__()
        self.version = "v0.00"
        self.model_definition = model_definition
        self.device = model_definition['device']

        # Network architecture and configuration
        self.num_early_layers = model_definition['num_early_layers']
        self.num_late_layers = model_definition['num_late_layers']
        self.num_filters = model_definition['num_filters']
        self.num_input_channels = model_definition['num_input_channels'] 
        self.num_input_times = model_definition['num_input_times'] 
        self.num_output_channels = model_definition['num_output_channels']
        self.use_bias = model_definition['use_bias']
        self.per_layer_input = model_definition['per_layer_input']

        self.use_time_latents = model_definition['use_time_latents']
        self.act_func = model_definition['act_func']
        self.max_pixel_value = model_definition['max_pixel_value']
        self.near_far = model_definition['near_far']

        # Time latents
        self.fixed_frame_ids = torch.arange(0, 10) #for CAVAREV
        if self.use_time_latents:
            self.num_time_dim = model_definition['num_time_dim']
            self.time_latents = nn.Parameter(torch.rand((self.fixed_frame_ids.shape[0], self.num_time_dim)))

        self.first_act_func = nn.ReLU()
        self.act_func = nn.ReLU()
        
        self.use_pos_enc = model_definition['pos_enc']

        self.density_shift = -10

        # Positional encoding
        self.position_encoder = NoEncoding(self.num_input_channels)
        if self.use_pos_enc != 'none':
            self.pos_enc_basis = model_definition['pos_enc_basis']
            if self.use_pos_enc == 'simple':
                self.position_encoder = SimpleEncoding(self.num_input_channels, self.pos_enc_basis, self.device)
            if self.use_pos_enc == 'fourier':
                self.position_encoder = FourierEncoding(self.num_input_channels, self.pos_enc_basis, model_definition['fourier_sigma'], self.device)
            elif self.use_pos_enc == 'free_windowed':
                self.position_encoder = FreeEncoding(self.num_input_channels, self.pos_enc_basis, model_definition['pos_enc_window_start'], self.device)
        
        # Determine input feature size
        self.input_features = self.position_encoder.encoding_size + self.num_input_times * 2
        if self.use_time_latents:
            self.input_features = self.position_encoder.encoding_size + self.num_time_dim

        self.store_activations = False
        self.activation_dictionary = {}

        self.aabb = torch.tensor([[-1, -1, -1], [1, 1, 1]]).to(self.device)
        self.density_mask = None
        self.density_thresh = 1e-5

        self.create_time_net()

    def create_time_net(self):
        # Build MLP layers

        input_features = self.input_features
        num_filters = self.num_filters
        use_bias = self.use_bias
        num_output_channels = self.num_output_channels
        per_layer_input = self.per_layer_input

        early_pts_layers = []
        early_pts_layers += self.__create_layer(input_features, num_filters,
                                           use_bias, activation=self.first_act_func)
        for _ in range(self.num_early_layers):
            layer_input = num_filters + input_features if per_layer_input else num_filters
            early_pts_layers += self.__create_layer(layer_input, num_filters,
                                               use_bias, activation=self.act_func)

        self.early_pts_layers = nn.ModuleList(early_pts_layers)
        if self.num_late_layers > 0:
            self.skip_connection = self.__create_layer(num_filters + input_features, num_filters,
                                                use_bias, activation=self.act_func)

            late_pts_layers = []
            for _ in range(self.num_late_layers - 1):
                late_pts_layers += self.__create_layer(num_filters, num_filters,
                                                use_bias, activation=self.act_func)

            self.late_pts_layers = nn.ModuleList(late_pts_layers)
        self.output_linear = self.__create_layer(num_filters, num_output_channels,
                                        use_bias, activation=None)

    @staticmethod
    def __create_layer(num_in_filters: int, num_out_filters: int,
                       use_bias: bool, activation=nn.ReLU(), dropout=0.5) -> nn.Sequential:
        block = []
        block.append(nn.Linear(num_in_filters, num_out_filters, bias=use_bias))
        if activation:
            block.append(activation)
        block = nn.Sequential(*block)

        return block

    def activations(self, store_activations: bool) -> None:
        # Utility to store intermediate activations if needed
        self.store_activations = store_activations

        if not store_activations:
            self.activation_dictionary = {}

    def query_time(self, xs: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        # Query network without per-layer inputs
        input_pts = xs
        time_pts = ts

        pts_encoded = self.position_encoder(input_pts)

        time_encoded = time_pts
        values = torch.cat([pts_encoded, time_encoded], dim=-1)
        for _, pts_layer in enumerate(self.early_pts_layers):
            values = pts_layer(values)

        if self.num_late_layers > 0:
            values = self.skip_connection(torch.cat([pts_encoded, time_encoded, values], dim=-1))
            for _, pts_layer in enumerate(self.late_pts_layers):
                values = pts_layer(values)
        outputs = self.output_linear(values)

        return outputs
    
    def query_time_layer(self, xs, ts):
        # Query network with per-layer input (used for better gradient flow)
        per_layer_input = self.per_layer_input

        input_pts = xs
        time_pts = ts

        pts_encoded = self.position_encoder(input_pts)

        time_encoded = time_pts
        values = torch.cat([pts_encoded, time_encoded], dim=-1)

        for i, pts_layer in enumerate(self.early_pts_layers):
            values = torch.cat([values, pts_encoded, time_encoded], dim=-1) if (per_layer_input and i > 0 and i % 2 == 0) else values
            values = pts_layer(values)

        if self.num_late_layers > 0:
            values = self.skip_connection(torch.cat([pts_encoded, time_encoded, values], dim=-1))

            for i, pts_layer in enumerate(self.late_pts_layers):
                values = torch.cat([values, pts_encoded, time_encoded], dim=-1) if (per_layer_input and i > 0 and i % 2 == 0) else values
                values = pts_layer(values)
        outputs = self.output_linear(values)

        return outputs
    
    def forward_composite(self, x: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        # Forward input points and time
        input_pts = x
        time_pts = ts

        ts_int = time_pts.flatten()
        learned_time_pts = self.time_latents[ts_int.long()]

        if self.per_layer_input:
            outputs = self.query_time_layer(input_pts, learned_time_pts)
        else:
            outputs = self.query_time(input_pts, learned_time_pts)

        return outputs
    
    def get_interpx(self, is_train = True, N_samples=-1):
        # Generate interpolation steps (stratified during training)
        N_samples = N_samples if N_samples > 0 else self.nSamples
        near, far = self.near_far

        t_vals = torch.linspace(0., 1., N_samples)
        interpx = near * (1.-t_vals) + far * (t_vals)
        interpx = interpx.to(self.device)

        if is_train:
            interpx = randomize_depth(interpx)

        return interpx

    def sample_ray(self, rays_o, rays_d, interpx, aabb):
        # Compute 3D points along ray
        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., :, None]
        mask_outbbox = ((aabb[0] > rays_pts) | (rays_pts > aabb[1])).any(dim=-1)

        return rays_pts, interpx, ~mask_outbbox

    def forward(self, rays_chunk, time_chunk, interpx=[], is_train=True, N_samples=-1):
        # Full forward pass for rendering

        # Extract view directions from the ray bundle
        viewdirs = rays_chunk[:, 1]

        # Generate sampling interpolation points if not already provided
        interpx = interpx if len(interpx) > 0 else self.get_interpx(is_train=is_train, N_samples=N_samples)
        # Sample 3D points along the rays
        xyz_sampled, z_vals, ray_valid = self.sample_ray(rays_chunk[:, 0], viewdirs, interpx, self.aabb)
        # Compute distances between consecutive depth samples
        one_e_10 = torch.tensor([1e-10], dtype=viewdirs.dtype, device=viewdirs.device)
        # Normalize view directions and apply normalization to distances
        dists = torch.cat((z_vals[..., 1:] - z_vals[..., :-1], one_e_10.expand(z_vals[..., :1].shape)), dim=-1)
        rays_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
        dists = dists * rays_norm
        viewdirs = viewdirs / rays_norm

        # Optional density masking to remove empty space
        if self.density_mask is not None:
            densities = self.density_mask.sample_density(xyz_sampled[ray_valid])
            density_mask = densities > 0
            ray_invalid = ~ray_valid
            ray_invalid[ray_valid] |= (~density_mask)
            ray_valid = ~ray_invalid

        # Initialize density volume and compute values where rays are valid
        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        features = self.forward_composite(xyz_sampled[ray_valid], time_chunk[ray_valid])
        sigma[ray_valid] = F.softplus(features+self.density_shift).flatten()

        # Use a simple attenuation model to compute intensity map
        initial_intensities = torch.tensor([self.max_pixel_value], dtype=viewdirs.dtype, device=viewdirs.device).expand(sigma.shape[0])
        int_map = initial_intensities - torch.sum(sigma * dists, dim=-1)

        return int_map, sigma, dists

    def update_freq_mask_alpha(self, current_iter, max_iter):
        # Update frequency mask for progressive positional encoding (FreeNeRF)
        self.position_encoder.update_alpha(current_iter, max_iter)
    
    @torch.no_grad()
    def save(self, filename: str, training_information: dict) -> None:
        # Save model checkpoint and related configuration
        save_parameters = {
            'version': self.version,
            'parameters': self.model_definition,
            'training_information': training_information,
            'model': self.state_dict(),
        }
        
        # Save encoding-specific state if using FreeEncoding
        if 'free_windowed' in self.use_pos_enc:
            save_parameters['freq_mask_alpha'] = self.position_encoder.alpha

        torch.save(
            save_parameters,
            f=filename)
    
    @torch.no_grad()
    def update_density_mask(self, grid_size=(256,256,256)):
        # Update spatial pruning mask by evaluating maximum density over all frames

        densities, dense_xyz = self.get_grid(grid_size)

        # Format tensors to fit 3D operations
        dense_xyz = dense_xyz.transpose(0,2).contiguous()
        densities = densities.clamp(0,1).transpose(0,2).contiguous()[None,None]
        total_voxels = grid_size[0] * grid_size[1] * grid_size[2]

        # Apply 3D max pooling and threshold to create binary density mask
        ks = 3
        densities = F.max_pool3d(densities, kernel_size=ks, padding=ks // 2, stride=1).view(grid_size[::-1])
        densities[densities >= self.density_thresh] = 1
        densities[densities < self.density_thresh] = 0

        # Store the binary mask
        self.density_mask = DensityGridMask(self.device, self.aabb, densities)

        # Determine new tight bounding box from valid density voxels
        valid_xyz = dense_xyz[densities > 0.5]
        xyz_min = valid_xyz.amin(0)
        xyz_max = valid_xyz.amax(0)
        new_aabb = torch.stack((xyz_min, xyz_max))

        total = torch.sum(densities)
        print(f"bbox: {xyz_min, xyz_max} density rest %%%f" % (total / total_voxels * 100))
        return new_aabb
    
    @torch.no_grad()
    def get_grid(self, grid_size):
        # Generate a 3D grid and compute maximum density across fixed time frames
        aabb = self.aabb
        xs = torch.linspace(aabb[0][0], aabb[1][0], grid_size[0])
        ys = torch.linspace(aabb[0][1], aabb[1][1], grid_size[1])
        zs = torch.linspace(aabb[0][2], aabb[1][2], grid_size[2])

        # Create a 3D meshgrid of points inside AABB
        pts_flatten = torch.stack(torch.meshgrid(xs, ys, zs), -1).to(self.device)

        # Allocate tensor to store densities for each frame
        sigmas = torch.zeros(len(self.fixed_frame_ids), grid_size[0], grid_size[1], grid_size[2]).to(self.device)

        # Compute densities per time step and accumulate the max
        for phase in self.fixed_frame_ids:
            time_flatten = torch.Tensor([phase]).repeat_interleave(grid_size[1]*grid_size[2]).unsqueeze(-1).to(self.device)
            for i in range(grid_size[0]):
                raw = self.forward_composite(pts_flatten[i].view(-1,3), time_flatten).view((grid_size[1], grid_size[2]))
                sigma = torch.nn.Softplus()(raw + self.density_shift)
                sigmas[phase, i] = sigma

        sigmas = torch.max(sigmas, dim=0).values  # Max over all time steps

        return sigmas, pts_flatten