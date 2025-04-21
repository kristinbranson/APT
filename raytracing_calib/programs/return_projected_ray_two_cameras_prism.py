import matplotlib.pyplot as plt
import numpy as np
import torch
import scipy.io as sio
import os
from matplotlib.widgets import Cursor
import sys

new_path = '/groups/branson/bransonlab/aniket/APT/raytracing_calib/programs/'
sys.path.append(new_path)
from prism_arenas import Arena_reprojection_loss_two_cameras_prism_grid_distances
from ray_tracing_simulator_nnModules_grad import get_rot_mat

#%%
#config_path = 'config.yaml'
#with open(config_path, 'r') as f:
#    data = yaml.safe_load(f)

#dividing_col = config['dividing_column'] # Read dividing path from a yaml path
#dividing_col = 1336 - 1
#image_width = 1920
rotmat = 1
#rotmat = config['rotmat']

#print(f'Loading model from: {PATH}')
checkpoint = torch.load(PATH, weights_only=True)
arena = Arena_reprojection_loss_two_cameras_prism_grid_distances(
            principal_point_pixel_cam_0=torch.tensor([0.,0.]).to(torch.float64),
            principal_point_pixel_cam_1=torch.tensor([0.,0.]).to(torch.float64), 
            focal_length_cam_0=torch.tensor(0.).to(torch.float64), 
            focal_length_cam_1=torch.tensor(0.).to(torch.float64),
            R_stereo_cam=torch.eye(3).to(torch.float64),
            T_stereo_cam=torch.zeros(3,1).to(torch.float64), 
            prism_angles=torch.zeros(3,).to(torch.float64), 
            prism_center=torch.zeros(3,1).to(torch.float64), 
            )
arena.load_state_dict(checkpoint)


# %%
#NOTE: This is wrong. Needs to be fixed. Start with providing the correct R and T 
def get_annotations_curve(arena, user_annotation, 
                          camera_real, cam_real_dist_coeff,
                          camera_virtual, cam_virtual_dist_coeff):
    with torch.no_grad():
        undistorted_annotations = camera_virtual.undistort_pixels_classical(user_annotation[:2],
        cam_real_dist_coeff)
        cam_1_ray = camera_virtual(undistorted_annotations)
        _, _, emergent_ray, _ = arena.prism(cam_1_ray)
        origin = emergent_ray.origin[:,0][:,None]
        direction = emergent_ray.direction[:,0][:,None]
        s = torch.linspace(0, 3, 100).to(torch.float64)
        r = origin + s[None, :] * direction

        R1 = torch.eye(3, 3).to(torch.float64)
        T1 = torch.zeros(3, 1).to(torch.float64)
        R_stereo_cam = get_rot_mat(
            arena.stereo_camera_angles[0],
            arena.stereo_camera_angles[1],
            arena.stereo_camera_angles[2],
            )
        
        annotations_curve_real = camera_real.reproject(r, R1, T1)
        annotations_curve_virtual = camera_virtual.reproject(r, R_stereo_cam, arena.T_stereo_cam)
        annotations_curve_real = camera_real.distort_pixels_classical(
                                                            annotations_curve_real,
                                                            cam_real_dist_coeff)
        annotations_curve_virtual = camera_virtual.distort_pixels_classical(
                                                            annotations_curve_virtual,
                                                            cam_virtual_dist_coeff)
        return annotations_curve_real, annotations_curve_virtual
    

def get_epipolar_line(arena, user_annotation, 
                          cam_label):
    if "primary" in cam_label:
        labelled_camera = arena.camera1
        cam_labelled_dist_coeff = arena.radial_dist_coeffs_cam_0
        unlabelled_camera = get_secondary_camera(arena)
        cam_unlabelled_dist_coeff = arena.radial_dist_coeffs_cam_1
        R_labelled = torch.eye(3, 3).to(torch.float64)
        T_labelled = torch.zeros(3, 1).to(torch.float64)
        R_unlabelled = get_rot_mat(
            arena.stereo_camera_angles[0],
            arena.stereo_camera_angles[1],
            arena.stereo_camera_angles[2],
            ) 
        T_unlabelled = arena.T_stereo_cam

    elif "secondary" in cam_label:
        labelled_camera = get_secondary_camera(arena)
        cam_labelled_dist_coeff = arena.radial_dist_coeffs_cam_1
        unlabelled_camera = arena.camera1
        cam_unlabelled_dist_coeff = arena.radial_dist_coeffs_cam_0    
        R_unlabelled = torch.eye(3, 3).to(torch.float64)
        T_unlabelled = torch.zeros(3, 1).to(torch.float64)
        R_labelled = get_rot_mat(
            arena.stereo_camera_angles[0],
            arena.stereo_camera_angles[1],
            arena.stereo_camera_angles[2],
            ) 
        T_labelled = arena.T_stereo_cam

    else:
        print("Please provide appropriate camera labels")


    if cam_label in ["primary_virtual", "secondary_virtual"]:
        if cam_label == "primary_virtual":
            """
            Update crop_coor based on virtual/real camera
            """
        

        """"
        Read crop_coor, rot_angle from config file
        Add crop_coor to the annotation
        Rotate annotation by rot_angle
        """

        with torch.no_grad():
            undistorted_annotations = labelled_camera.undistort_pixels_classical(user_annotation[:2],
                                                                                cam_labelled_dist_coeff)
            cam_1_ray = labelled_camera(undistorted_annotations)
            _, _, emergent_ray, _ = arena.prism(cam_1_ray)
            origin = emergent_ray.origin[:,0][:,None]
            
            direction = emergent_ray.direction[:,0][:,None]
            s = torch.linspace(0, 8, 50).to(torch.float64)
            r = origin + s[None, :] * direction        
            annotations_curve_unlabelled_camera = unlabelled_camera.reproject(r, R_unlabelled, T_unlabelled)
            annotations_curve_labelled_camera = labelled_camera.reproject(r, R_labelled, T_labelled)
            annotations_curve_unlabelled_camera = unlabelled_camera.distort_pixels_classical(
                                                                annotations_curve_unlabelled_camera,
                                                                cam_unlabelled_dist_coeff)
            annotations_curve_labelled_camera = labelled_camera.distort_pixels_classical(
                                                                annotations_curve_labelled_camera,
                                                                cam_labelled_dist_coeff)
            return annotations_curve_unlabelled_camera, annotations_curve_labelled_camera
    

    elif cam_label in ["primary_real", "secondary_real"]:
        with torch.no_grad():
            prism_distance = torch.norm(arena.prism_center)
            undistorted_annotations = labelled_camera.undistort_pixels_classical(user_annotation[:2],
                                                                                cam_labelled_dist_coeff)
            cam_1_ray = labelled_camera(undistorted_annotations)
            origin = cam_1_ray.origin[:,0][:,None]
            direction = cam_1_ray.direction[:,0][:,None]
            s = torch.linspace(prism_distance, prism_distance + 10, 50).to(torch.float64)
            r = origin + s[None, :] * direction       
            
            annotations_curve_unlabelled_camera = unlabelled_camera.reproject(r, R_unlabelled, T_unlabelled)            
            annotations_curve_unlabelled_camera = unlabelled_camera.distort_pixels_classical(
                                                                annotations_curve_unlabelled_camera,
                                                                cam_unlabelled_dist_coeff)
            return annotations_curve_unlabelled_camera


def get_secondary_camera(arena):
    """
    Returns secondary camera given the arena model parameters
    """
    R_stereo_cam = get_rot_mat(
            arena.stereo_camera_angles[0],
            arena.stereo_camera_angles[1],
            arena.stereo_camera_angles[2],
            )
    return arena.get_stereo_camera(arena.principal_point_pixel_cam_1,
                                arena.focal_length_cam_1,
                                R_stereo_cam,
                                arena.T_stereo_cam,
                                r1=arena.stereocam_r1,
                                radial_dist_coeffs=arena.radial_dist_coeffs_cam_1)


if not(type(user_annotation) == torch.Tensor):
    user_annotation = torch.tensor(user_annotation).to(torch.float64)[:, None]

if rotmat:
    theta = torch.tensor(torch.pi / 2)
    R_theta_inv = torch.tensor([[torch.cos(-theta), -torch.sin(-theta)], [torch.sin(-theta), torch.cos(-theta)]]).to(torch.float64)
    R_theta = torch.tensor([[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]]).to(torch.float64)
    user_annotation = R_theta_inv @ user_annotation
    user_annotation[0, :] = -user_annotation[0, :] + dividing_col  # 1-based indexing for dividing_col 
    user_annotation[1, :] = -user_annotation[1, :]

if "virtual" in cam_label:
    epipolar_line_unlabelled, epipolar_line_labelled = get_epipolar_line(arena, user_annotation, cam_label)
    if rotmat:
        user_annotation = R_theta @ user_annotation
        epipolar_line_unlabelled = R_theta @ epipolar_line_unlabelled
        epipolar_line_unlabelled[1, :] = -epipolar_line_unlabelled[1, :] + image_width - 1
        epipolar_line_unlabelled[0, :] *= -1
        epipolar_line_labelled = R_theta @ epipolar_line_labelled
        epipolar_line_labelled[1, :] = -epipolar_line_labelled[1, :] + image_width - 1
        epipolar_line_labelled[0, :] *= -1
    epipolar_line_unlabelled = epipolar_line_unlabelled.numpy()
    epipolar_line_labelled = epipolar_line_labelled.numpy()

elif "real" in cam_label:
    epipolar_line_unlabelled = get_epipolar_line(arena, user_annotation, cam_label)
    epipolar_line_unlabelled = epipolar_line_unlabelled.numpy()
    epipolar_line_labelled = np.array([])

# %%
