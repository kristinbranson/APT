# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: APT_raytracing
#     language: python
#     name: python3
# ---

# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
import scipy.io as sio
import os
from matplotlib.widgets import Cursor
import sys
import math
raytracing_lib_path = os.path.dirname(os.path.abspath(sys.argv[0]))
APT_path = os.path.dirname(os.path.dirname(raytracing_lib_path))
# Aniket's APT path /groups/branson/bransonlab/aniket/APT/
print(f"raytracing_lib_path: {raytracing_lib_path}")
print(f"APT_path: {APT_path}")
sys.path.append(raytracing_lib_path)
import importlib
import scipy.io as sio
from prism_arenas_nan_masks import Arena_reprojection_loss_two_cameras_prism_grid_distances
from utils import euclidean_distance
rotmat = 1
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

"""
This python script is accessed by APT's Multiview triangulation program from MATLAB.
Note that dividing_col, image_width, APT_path, PATH come from MATLAB and do not need to be defined in this script
"""


def triangulate(xp, 
                image_width=[1920, 1920], 
                image_height=[1200, 1200], 
                dividing_col=[1331, 1331], 
                view_labels=["primary_virtual"], 
                APT_path=APT_path, 
                PATH='/groups/branson/bransonlab/PTR_fly_annotations_3D/exp46/best_model_weights_only.pth', 
                num_cams=2):
    """
    Triangulates 3-D points for the prism arena, given the 2-D points in pixel coordinates from APT.
    Doesn't necessarily require pixel coordinates from all cameras (or all views)
    Parameters
    ----------
    xp : tensor or numpy array or memoryview object (passed from MATLAB)
        2D points in pixel coordinates with shape (2, num_points, num_views).
    image_width : list of float
        Width of the images in pixels for each camera.
    image_height : list of float
        Height of the images in pixels for each camera.
    dividing_col : list of float
        Dividing column for each camera, used to adjust the x-coordinates of the virtual cameras.
    view_labels : list of str
        Labels for the views whose pixel coordinates are provided, e.g., ['primary_virtual', 'secondary_virtual'].
    APT_path : str
        Path to the APT root directory that has the raytracing calibration programs.
    PATH : str
        Path to the model weights file for the triangulation.
    num_cams : int, optional
        Number of cameras used in the triangulation. Default is 2.
    
    Returns
    -------
    recon_3D : numpy array
        Triangulated 3D points with shape (3, num_points).
    recon_pixels : numpy array
        Reprojected pixel coordinates of the triangulated points with shape (2, num_points, num_cams * 2).
    reprojection_error : numpy array
        Reprojection error for the triangulated points with shape (num_points, num_cams * 2).
    
    Notes
    -----
    Example inputs
    xp = np.array([
        [[523.4, 457.97]],
     [[399.92, 410.98]]]
    )
    image_width = [1920, 1920]  # Width of the images in pixels
    view_labels = ['primary_virtual', 'secondary_virtual']  # Labels for the views
    PATH = '/groups/branson/bransonlab/PTR_fly_annotations_3D/exp46/best_model_weights_only.pth'
    
    dividing_col = [1331.0, 1331.0]
    image_width = [1920, 1920]
    image_height = [1200, 1200]
    """

    view_labels = list(view_labels)
    xp = np.array(xp, copy=True, dtype=np.float64)  # Ensure xp is a numpy array of type float64
    all_views = ['primary_virtual', 'primary_real', 'secondary_virtual', 'secondary_real'] # This assumes that there are two cameras (on the fly prism rig)
    indices = [all_views.index(item) for item in view_labels]
    num_points = xp.shape[1] # Number of points in xp to be triangulated
    xp_all = np.zeros((2, num_points, num_cams * 2))  # 2D coordinates for all cameras
    xp_all = np.full((2, num_points, num_cams * 2), np.nan, dtype=np.float64)  # Initialize with NaN values
    for i in range(len(indices)):
        xp_all[:, :, indices[i]] = xp[:, :, i]
    xp = xp_all.reshape(2 * num_points, num_cams * 2, order='F')  # Reshape to (2 * num_points, num_cams). Order ensures that the first axis changes the fastest (x1,y1,x2,y2,..)
    rotmat = 1
    checkpoint = torch.load(PATH, weights_only=True, map_location=torch.device('cpu'))
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
    if isinstance(xp, memoryview):
        xp = np.array(xp)

    if not(type(xp) == torch.Tensor):
        xp = torch.tensor(xp).to(torch.float64)

    xp_APT_reference = xp.clone()
    if rotmat:
        theta = torch.tensor(torch.pi / 2)
        R_theta_inv = torch.tensor([[torch.cos(-theta), -torch.sin(-theta)], [torch.sin(-theta), torch.cos(-theta)]]).to(torch.float64)
        R_theta = torch.tensor([[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]]).to(torch.float64)
        xp = R_theta_inv @ xp
        
        for cam_label in range(num_cams):
            # For virtual views        
            if torch.isnan(xp[0, cam_label * 2]).any() or torch.isnan(xp[1, cam_label * 2]).any():
                continue
            xp[0, cam_label * 2] = -xp[0, cam_label * 2] +  dividing_col[cam_label] - 1  # 1-based indexing for dividing_col
            xp[1, cam_label * 2] *= -1

        for cam_label in range(num_cams):
            # For real views
            if torch.isnan(xp[0, cam_label * 2 + 1]).any() or torch.isnan(xp[1, cam_label * 2 + 1]).any():
                continue
            xp[0, cam_label * 2 + 1] = -xp[0, cam_label * 2] + image_width[cam_label] - 1 
            xp[1, cam_label * 2 + 1] *= -1

    recon_3D, recon_pixels_1_virtual, recon_pixels_1_real, recon_pixels_2_virtual, recon_pixels_2_real, _ = arena.triangulate(
        xp, 
        image_width,
        image_height,
    )
    recon_pixels = torch.vstack(
        [recon_pixels_1_virtual, recon_pixels_1_real, recon_pixels_2_virtual, recon_pixels_2_real]
    )

    # Get the coordinates in APT's reference frame
    recon_pixels = recon_pixels.detach().numpy()
    recon_pixels = recon_pixels.reshape(2, num_points, -1, order='F')  # Reshape to (2, num_points, num_cams * 2). Order ensures that the first axis changes the fastest (x1,y1,x2,y2,..)
    for pt in range(num_points):
        recon_pixels[:, pt, :] = R_theta @ recon_pixels[:, pt, :]

    for cam_label in range(num_cams):
        recon_pixels[1, :, cam_label * 2] = -recon_pixels[1, :, cam_label * 2] + dividing_col[cam_label] - 1  # Invert x-coordinates for virtual cameras
        recon_pixels[0, :, cam_label * 2] *= -1  # Invert y-coordinates for virtual cameras
        recon_pixels[1, :, cam_label * 2 + 1] = -recon_pixels[0, :, cam_label * 2 + 1] + image_width[cam_label] - 1  # Invert x-coordinates for real cameras
        recon_pixels[0, :, cam_label * 2 + 1] *= -1  # Invert y-coordinates for real cameras


    recon_pixels = recon_pixels[:, :, indices]  # Select only the views that were used for triangulation
    xp_APT_reference = xp_APT_reference.numpy().reshape(2, num_points, -1, order='F')[:, :, indices]  # Reshape to (2, num_points, num_cams * 2). Order ensures that the first axis changes the fastest (x1,y1,x2,y2,..)
    reprojection_error = euclidean_distance(recon_pixels, xp_APT_reference)
    reprojection_error = reprojection_error.detach().numpy()
    recon_3D = recon_3D.detach().numpy()

    return recon_3D, recon_pixels, reprojection_error
# %%
recon_3D, recon_pixels, reprojection_error = triangulate(
                                                        xp, 
                                                        image_width=image_width, 
                                                        image_height=[1200, 1200], 
                                                        dividing_col=dividing_col, 
                                                        view_labels=view_labels, 
                                                        APT_path=APT_path, 
                                                        PATH=PATH, 
                                                        num_cams=2
                                                        )

# %%
