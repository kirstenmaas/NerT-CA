import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import pyvista as pv
from torchmetrics.classification import MulticlassF1Score

# Calculate the resolution (number of voxels per axis) needed to represent a volume within a bounding box given a target number of total voxels.
def N_to_reso(n_voxels, bbox):
    xyz_min, xyz_max = bbox
    dim = len(xyz_min)
    voxel_size = ((xyz_max - xyz_min).prod() / n_voxels).pow(1 / dim)
    return ((xyz_max - xyz_min) / voxel_size).long().tolist()

# Estimate the number of depth samples needed for ray marching given the spatial resolution.
def cal_n_samples(reso, step_ratio=0.5):
    return int(np.linalg.norm(reso)/step_ratio)

def normalize(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-10)

class TVLoss(torch.nn.Module):
    def __init__(self, TVLoss_weight_dim1=1.0, TVLoss_weight_dim2=1.0):
        super(TVLoss, self).__init__()
        self.TVLoss_weight_dim1 = TVLoss_weight_dim1
        self.TVLoss_weight_dim2 = TVLoss_weight_dim2

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = (
            torch.pow((x[:, :, 1:, :] - x[:, :, : h_x - 1, :]), 2).sum()
            * self.TVLoss_weight_dim1
        )
        w_tv = (
            torch.pow((x[:, :, :, 1:] - x[:, :, :, : w_x - 1]), 2).sum()
            * self.TVLoss_weight_dim2
        )

        val = h_tv / count_h
        if count_w > 0:
            val = val + w_tv / count_w

        return 2 * (val) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

# render function for training
def render_hybrid(rays, phases, low_model, sparse_model, interpx=[], chunk=4096, N_samples=-1, is_train=False, device='cuda', sample_sparse=False):
    l_ints, l_sigmas, l_dists = [], [], []
    s_ints, s_sigmas, s_dists = [], [], []
    N_rays_all = rays.shape[0]

    interpx = low_model.get_interpx(is_train=is_train, N_samples=N_samples) if len(interpx) == 0 else interpx
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
        rays_phases = phases[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
    
        l_int_map, l_sigma, l_dist = low_model(rays_chunk, interpx, is_train=is_train, N_samples=N_samples)
        
        l_ints.append(l_int_map)
        l_sigmas.append(l_sigma)
        l_dists.append(l_dist)

        if sample_sparse:
            s_int_map, s_sigma, s_dist = sparse_model(rays_chunk, rays_phases, interpx, is_train=is_train, N_samples=N_samples)
            s_ints.append(s_int_map)
            s_sigmas.append(s_sigma)
            s_dists.append(s_dist)
    
    l_ints = torch.cat(l_ints)
    l_sigmas = torch.cat(l_sigmas)
    l_dists = torch.cat(l_dists)
    if sample_sparse:
        s_ints = torch.cat(s_ints)
        s_sigmas = torch.cat(s_sigmas)
        s_dists = torch.cat(s_dists)

    return l_ints, l_sigmas, l_dists, s_ints, s_sigmas, s_dists

# computes the occlusion regularization loss. Optionally also from the back of the rays.
def compute_occl_loss(sigma_s, dists, reg_perc=0.1, use_back=False):
  cum_dists = torch.cumsum(dists, dim=-1)
  dists_range_front = reg_perc * cum_dists[-1, -1]
  dists_range_back = (1-reg_perc) * cum_dists[-1, -1]

  mask = torch.where(cum_dists < dists_range_front, 1., 0.).int()

  if use_back:
    mask_back = torch.where(cum_dists > dists_range_back, 1., 0.).int()
    mask = torch.bitwise_or(mask, mask_back)

  loss = torch.sum(sigma_s * dists * mask, dim=-1)
  return loss.sum()

# Test evaluation for the hybrid model
def evaluation_hybrid(rays_test, phases_test, tensorf, temp_model, renderer, chunk, N_samples, data_size, device, is_train=False):
    s_int_map, s_sigmas, dists, d_int_map, d_sigmas, _ = renderer(rays_test[:, :2, :], phases_test, tensorf, temp_model, sample_sparse=True, \
                                                                  chunk=chunk, N_samples=N_samples, device=device, is_train=is_train)
    sigmas = s_sigmas + d_sigmas

    initial_intensities = torch.tensor([tensorf.max_pixel_value], device=device).expand(s_sigmas.shape[0])
    int_map = initial_intensities - torch.sum(sigmas * dists, dim=-1)

    mip_map = torch.max(sigmas, dim=-1).values.reshape((data_size, data_size))
    
    bin_map = torch.clone(mip_map)
    
    bin_map[bin_map < 0.05] = 0
    bin_map[bin_map >= 0.05] = 1 
    
    int_map = int_map.reshape((data_size, data_size))
    s_int_map = s_int_map.reshape((data_size, data_size))
    d_int_map = d_int_map.reshape((data_size, data_size))

    int_map_gt = rays_test[:,2,0].reshape((data_size, data_size))
    int_map_gt = int_map_gt.reshape(int_map.shape)

    bin_map_gt = rays_test[:, -1, 0].reshape((data_size, data_size))
    bin_map_gt = 1 - torch.clamp(bin_map_gt / torch.max(bin_map_gt), 0, 1).int()

    loss = torch.mean((int_map - int_map_gt)**2)
    psnr = -10.0 * np.log(loss.item()) / np.log(10.0)

    dice_f = MulticlassF1Score(num_classes=2, average='none').to(device)

    _, dice_score = dice_f(bin_map.int(), bin_map_gt.int())

    return int_map, int_map_gt, loss, psnr, s_int_map, d_int_map, bin_map, dice_score, bin_map_gt

# Training sampler
class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr += self.batch

        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr+self.batch]