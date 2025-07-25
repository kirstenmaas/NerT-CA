import torch
import torch.nn.functional as F
import pdb

from .tensorBase import TensorBase

class TensorCP(TensorBase):
    def __init__(self, aabb, gridSize, max_pixel_value, device, **kargs):
        super(TensorCP, self).__init__(aabb, gridSize, max_pixel_value, device, **kargs)

    def init_svd_volume(self, res, device):
        self.density_line = self.init_one_svd(self.density_n_comp[0], self.gridSize, 0.2, device)

    def init_one_svd(self, n_component, gridSize, scale, device):
        # Initialize 1D line coefficients along each spatial axis
        line_coef = []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            line_coef.append(
                torch.nn.Parameter(scale * torch.randn((1, n_component, gridSize[vec_id], 1))))
        return torch.nn.ParameterList(line_coef).to(device) # (dim, 1, n_component, gridsize, 1)
    
    def get_optparam_groups(self, lr_init_spatialxyz = 0.02):
        grad_vars = [{'params': self.density_line, 'lr': lr_init_spatialxyz}]
        return grad_vars

    def compute_densityfeature(self, xyz_sampled):
        # Sample from the 1D CP-decomposed fields using trilinear interpolation
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        # Trilinear multiplication of samples from three directions
        line_coef_point = F.grid_sample(self.density_line[0], coordinate_line[[0]], align_corners=True).view(-1, *xyz_sampled.shape[:1])
        line_coef_point = line_coef_point * F.grid_sample(self.density_line[1], coordinate_line[[1]],
                                        align_corners=True).view(-1, *xyz_sampled.shape[:1])
        line_coef_point = line_coef_point * F.grid_sample(self.density_line[2], coordinate_line[[2]],
                                        align_corners=True).view(-1, *xyz_sampled.shape[:1])

        sigma_feature = torch.sum(line_coef_point, dim=0) # Final feature after product
        
        return sigma_feature
    
    @torch.no_grad()
    def up_sampling_vector(self, density_line_coef, res_target):
        # Upsample 1D coefficients to a higher grid resolution
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            density_line_coef[i] = torch.nn.Parameter(
                F.interpolate(density_line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))

        return density_line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        # Apply the upsampling and update step sizes accordingly
        self.density_line = self.up_sampling_vector(self.density_line, res_target)

        self.update_stepSize(res_target)
        print(f'upsamping to {res_target}')

    @torch.no_grad()
    def shrink(self, new_aabb):
        # Crop the grid to a smaller AABB (shrink it)
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units

        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)

        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]
            self.density_line[i] = torch.nn.Parameter(
                self.density_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )

        newSize = b_r - t_l
        self.aabb = new_aabb
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))
    
    def density_L1(self):
        # L1 regularization for sparsity
        total = 0
        for idx in range(len(self.density_line)):
            total = total + torch.mean(torch.abs(self.density_line[idx]))
        return total

    def TV_loss_density(self, reg):
        # Total variation loss to enforce spatial smoothness
        total = 0
        for idx in range(len(self.density_line)):
            total = total + reg(self.density_line[idx]) * 1e-3
        return total
    
class TensorVM(TensorBase):
    def __init__(self, aabb, gridSize, max_pixel_value, device, **kargs):
        super(TensorVM, self).__init__(aabb, gridSize, max_pixel_value, device, **kargs)

    def init_svd_volume(self, res, device):
        # Initialize both 2D plane and 1D line coefficients (Vector-Matrix decomposition)
        self.density_plane, self.density_line = self.init_one_svd(self.density_n_comp, self.gridSize, 0.01, device)

    def init_one_svd(self, n_component, gridSize, scale, device):
        # Create learnable parameters for plane (2D) and line (1D) coefficients
        plane_coef, line_coef = [], []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef.append(torch.nn.Parameter(
                scale * torch.randn((1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]))))  #
            line_coef.append(
                torch.nn.Parameter(scale * torch.randn((1, n_component[i], gridSize[vec_id], 1))))

        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(line_coef).to(device)

    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        grad_vars = [{'params': self.density_line, 'lr': lr_init_spatialxyz}, 
                     {'params': self.density_plane, 'lr': lr_init_spatialxyz}]
        
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params':self.renderModule.parameters(), 'lr':lr_init_network}]
        return grad_vars
    
    def density_L1(self):
        # L1 regularization combining line and plane coefficients
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + torch.mean(torch.abs(self.density_plane[idx])) * 1e-2 + torch.mean(torch.abs(self.density_line[idx])) * 1e-3
        return total
    
    def TV_loss_density(self, reg):
        # Total variation loss on both components
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + reg(self.density_plane[idx]) * 1e-2 + reg(self.density_line[idx]) * 1e-3
        return total

    def compute_densityfeature(self, xyz_sampled):
        # Compute interpolated features from 2D planes and 1D lines for each axis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        plane_feat = []
        line_feat = []
        for idx_plane in range(len(self.density_plane)):
            plane_feat.append(F.grid_sample(self.density_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1]))
            line_feat.append(F.grid_sample(self.density_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1]))
        plane_feat, line_feat = torch.stack(plane_feat, dim=0), torch.stack(line_feat, dim=0)

        inter = plane_feat * line_feat
        inter = torch.sum(torch.sum(inter, dim=0), dim=0)

        return inter

    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):
        # Upsample both plane and line coefficients to a higher resolution
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef[i] = torch.nn.Parameter(
                F.interpolate(plane_coef[i].data, size=(res_target[mat_id_1], res_target[mat_id_0]), mode='bilinear',
                              align_corners=True))
            line_coef[i] = torch.nn.Parameter(
                F.interpolate(line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))

        return plane_coef, line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.density_plane, self.density_line = self.up_sampling_VM(self.density_plane, self.density_line, res_target)

        self.update_stepSize(res_target)
        print(f'upsamping to {res_target}')

    @torch.no_grad()
    def shrink(self, new_aabb):
        # Shrink the voxel volume to new AABB and adjust parameters accordingly
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units
        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)

        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]
            self.density_line[i] = torch.nn.Parameter(
                self.density_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            mode0, mode1 = self.matMode[i]
            self.density_plane[i] = torch.nn.Parameter(
                self.density_plane[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]]
            )
        
        if not torch.all(self.density_mask.gridSize == self.gridSize):
            t_l_r, b_r_r = t_l / (self.gridSize-1), (b_r-1) / (self.gridSize-1)
            correct_aabb = torch.zeros_like(new_aabb)
            correct_aabb[0] = (1-t_l_r)*self.aabb[0] + t_l_r*self.aabb[1]
            correct_aabb[1] = (1-b_r_r)*self.aabb[0] + b_r_r*self.aabb[1]
            print("aabb", new_aabb, "\ncorrect aabb", correct_aabb)
            new_aabb = correct_aabb

        newSize = b_r - t_l
        self.aabb = new_aabb
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))