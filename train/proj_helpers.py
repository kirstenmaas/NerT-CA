import numpy as np
import torch

def x_rotation_matrix(angle):
    return np.array([
        [1, 0, 0, 0],
        [0, np.cos(angle), -np.sin(angle), 0], 
        [0, np.sin(angle), np.cos(angle), 0],
        [0, 0, 0, 1]
    ])

def y_rotation_matrix(angle):
    return np.array([
        [np.cos(angle), 0, np.sin(angle), 0],
        [0, 1, 0, 0],
        [-np.sin(angle), 0, np.cos(angle), 0],
        [0, 0, 0, 1]
    ])

def z_rotation_matrix(angle):
    return np.array([
        [np.cos(angle), -np.sin(angle), 0, 0],
        [np.sin(angle), np.cos(angle), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def translation_matrix(vec):
    m = np.identity(4)
    m[:3, 3] = vec[:3]
    return m

def get_rotation_matrix_tigre(theta, phi, larm):
    R1 = x_rotation_matrix(-np.pi/2)
    R2 = x_rotation_matrix(np.deg2rad(phi))
    R3 = z_rotation_matrix(np.pi/2)
    R4 = z_rotation_matrix(-np.deg2rad(theta))

    R = np.dot(np.dot(R4, np.dot(R3, R2)), R1)
    return R

def source_matrix_tigre(source_pt, theta, phi, larm=0, translation=[0,0,0], type='rotation'):
    rot = get_rotation_matrix_tigre(theta, phi, larm)
    worldtocam = rot.dot(translation_matrix(source_pt))

    return worldtocam

# cast rays based on tigre geometry
def get_ray_values_tigre(theta, phi, larm, geo, device):
    src_pt = np.array([0, 0, -geo['DSO']])

    pose = torch.from_numpy(source_matrix_tigre(src_pt, theta, phi, larm)).to(device).float()

    img_width, img_height = geo['nDetector']

    ii, jj = torch.meshgrid(
        torch.linspace(0, img_width-1, img_width).to(device),
        torch.linspace(0, img_height-1, img_height).to(device),
        indexing='xy'
    )

    uu = (ii.t() + 0.5 - img_width / 2) * geo['dDetector'][0] + geo['offDetector'][0]
    vv = (jj.t() + 0.5 - img_height / 2) * geo['dDetector'][1] + geo['offDetector'][1]
    dirs = torch.stack([uu / geo['DSD'], vv / geo['DSD'], torch.ones_like(uu)], -1)

    ray_directions = torch.sum(torch.matmul(pose[:3,:3], dirs[..., None]).to(device), -1)
    ray_origins = pose[:3, -1].expand(ray_directions.shape)

    ray_origins = ray_origins.cpu().numpy()
    ray_directions = ray_directions.cpu().numpy()

    return ray_origins, ray_directions