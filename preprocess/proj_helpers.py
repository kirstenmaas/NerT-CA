import pandas as pd
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

def get_rotation_matrix(theta, phi, larm):
    R1 = x_rotation_matrix(-np.pi/2)
    R2 = x_rotation_matrix(np.deg2rad(phi))
    R3 = z_rotation_matrix(np.pi/2)
    R4 = z_rotation_matrix(np.deg2rad(theta))   

    R = np.dot(np.dot(R4, np.dot(R3, R2)), R1)
    return R

# definition of source matrix based on TIGRE geometry
def source_matrix(source_pt, theta, phi, larm=0, translation=[0,0,0], type='rotation'):
    rot = get_rotation_matrix(theta, phi, larm)
    worldtocam = rot.dot(translation_matrix(source_pt))

    return worldtocam