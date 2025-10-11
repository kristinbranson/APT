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
# %matplotlib inline

# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
import scipy.io as sio
import os
from matplotlib.widgets import Cursor
import sys
import math

# %% [markdown]
# ##################################################################################################
# User inputs
# %%
checkpoint_path = '/groups/branson/bransonlab/PTR_fly_annotations_3D/exp32/best_model_weights_only.pth' # Path to the checkpoint file
APT_path = '/groups/branson/bransonlab/aniket/APT/' # Path to the APT repository
user_annotation = np.array([688.55168207, 368.61018184]) # User annotation (tensor)
cam_labelling = 'primary_virtual' #Annotation camera label (can be one of these: "primary_virtual", "primary_real", "secondary_virtual", "secondary_real")
cam_projecting = 'secondary_virtual' #Label of the camera in which epipolar line is drawn (can be one of these: "primary_virtual", "primary_real", "secondary_virtual", "secondary_real")
calibration_file_path = '/groups/branson/bransonlab/PTR_fly_annotations_3D/exp32/exp_32_calibration_data.mat' # Path to the calibration file
device = 'cpu'
###################################################################################################
# %%
cal_mat_data = sio.loadmat(calibration_file_path)
dividing_col = cal_mat_data['dividing_col'][0]
image_width = cal_mat_data['image_width'][0]
cam_label = cam_labelling

# %%

raytracing_lib_path = os.path.join(APT_path, 'raytracing_calib', 'programs')
sys.path.append(raytracing_lib_path)
from prism_arenas import Arena_reprojection_loss_two_cameras_prism_grid_distances
from ray_tracing_simulator_nnModules_grad import get_rot_mat
rotmat = 1
#if torch.cuda.is_available():
#    device = 'cuda'
#else:
#    device = 'cpu'

checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=torch.device('cpu'))
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


def get_epipolar_line(arena, user_annotation, labelling_cam, projecting_cam):
    """
    Project epipolar line in projecting_cam given a label in labeling cam
    arena: Prism arena object
    user_annotation: 2D point in the image
    labelling_cam (str): Can be one of these: "primary_virtual", "primary_real", "secondary_virtual", "secondary_real"    
    projecting_cam (str): Can be one of these: "primary_virtual", "primary_real", "secondary_virtual", "secondary_real"    
    """
    with torch.no_grad():
        num_epipolar_pts = 4 # This will only work if the calibration is highly acccurate. Else, consider adding more points for redundancy
        if "primary" in labelling_cam:
            labelling_cam_obj = arena.camera1
            labelling_cam_dist_coeff = arena.radial_dist_coeffs_cam_0            

        elif "secondary" in labelling_cam:
            labelling_cam_obj = get_secondary_camera(arena)
            labelling_cam_dist_coeff = arena.radial_dist_coeffs_cam_1

        else:
            print("Please provide appropriate camera labels: 'primary_virtual', 'primary_real', 'secondary_virtual', 'secondary_real'")
        

        if "primary" in projecting_cam:
            projecting_cam_obj = arena.camera1
            projecting_cam_dist_coeff = arena.radial_dist_coeffs_cam_0
            R_projecting = torch.eye(3, 3).to(torch.float64)
            T_projecting = torch.zeros(3, 1).to(torch.float64)

        
        elif "secondary" in projecting_cam:
            projecting_cam_obj = get_secondary_camera(arena)
            projecting_cam_dist_coeff = arena.radial_dist_coeffs_cam_1
            R_projecting = get_rot_mat(
                arena.stereo_camera_angles[0],
                arena.stereo_camera_angles[1],
                arena.stereo_camera_angles[2],
                )
            T_projecting = arena.T_stereo_cam
            

        # Get the 3-D epipolar line using the annotation in the labelling camera. Two cases: (1) Ray passes through the prism, (2) Ray passes through the air
        if "virtual" in labelling_cam:
            undistorted_annotations = labelling_cam_obj.undistort_pixels_classical(user_annotation[:2],
                                                                                labelling_cam_dist_coeff)
            cam_1_ray = labelling_cam_obj(undistorted_annotations)
            _, _, emergent_ray, _ = arena.prism(cam_1_ray)
            origin = emergent_ray.origin[:,0][:,None]
            
            direction = emergent_ray.direction[:,0][:,None]
            s = torch.linspace(0, 8, num_epipolar_pts).to(torch.float64)
            epipolar_line_3D = origin + s[None, :] * direction        
            
        if "real" in labelling_cam:
            prism_distance = torch.norm(arena.prism_center)
            undistorted_annotations = labelling_cam_obj.undistort_pixels_classical(user_annotation[:2],
                                                                                labelling_cam_dist_coeff)
            cam_1_ray = labelling_cam_obj(undistorted_annotations)
            origin = cam_1_ray.origin[:,0][:,None]
            direction = cam_1_ray.direction[:,0][:,None]
            s = torch.linspace(prism_distance + 5, prism_distance + 17, num_epipolar_pts).to(torch.float64)
            epipolar_line_3D = origin + s[None, :] * direction       
                
        # Reproject the epipolar line in the projecting camera
        if "virtual" in projecting_cam:
            epipolar_line_projecting_cam = torch.zeros(2, num_epipolar_pts).to(dtype=torch.float64, device=device)            
            for pt_id in range(num_epipolar_pts):
                if 'primary' in projecting_cam:
                    reprojection_pixel = closest_virtual_pixel_to_point(arena, epipolar_line_3D[:, pt_id].unsqueeze(-1), image_width, image_height=[1200, 1200], cam_label_projection='primary')
                elif 'secondary' in projecting_cam:
                    reprojection_pixel = closest_virtual_pixel_to_point(arena, epipolar_line_3D[:, pt_id].unsqueeze(-1), image_width, image_height=[1200, 1200], cam_label_projection='secondary')
                    print(reprojection_pixel)
                epipolar_line_projecting_cam[:, pt_id] = reprojection_pixel[:2, 0]
            

        if "real" in projecting_cam:
            epipolar_line_projecting_cam = projecting_cam_obj.reproject(epipolar_line_3D, R_projecting, T_projecting)
            #epipolar_line_labelling_cam = labelling_cam_obj.reproject(epipolar_line_3D, R_labelling, T_labelling)
            epipolar_line_projecting_cam = projecting_cam_obj.distort_pixels_classical(
                                                                epipolar_line_projecting_cam,
                                                                projecting_cam_dist_coeff)
            #epipolar_line_labelling_cam = labelling_cam_obj.distort_pixels_classical(
            #                                                    epipolar_line_labelling_cam,
            #                                                    labelling_cam_dist_coeff)
    return epipolar_line_projecting_cam

def closest_virtual_pixel_to_point(arena, point, image_width, image_height=[1200, 1200], cam_label_projection='primary'):    
    """
    cam_label (str): ["primary", "secondary", "both"]
    """    
    
    resolution_for_virtual_projection = 0.01 # pixel accuracy needed to get reprojection in the virtual camera
    grid_factor = 14    
    if cam_label_projection == 'primary':
        width = [image_width[0], image_width[0]] # Only to make the script compatible with the multiple camera case
        height = [image_height[0], image_height[0]] # Only to make the script compatible with the multiple camera case
        num_views = 1
    elif cam_label_projection == 'secondary':
        width = [image_width[1], image_width[1]] # Only to make the script compatible with the multiple camera case
        height = [image_height[1], image_height[1]] # Only to make the script compatible with the multiple camera case
        num_views = 1
    elif cam_label_projection == 'both':
        num_views = 2
        width = image_width
        height = image_height
    
    #virtual_pixels_grid = torch.zeros(2 * num_views, grid_factor * grid_factor).to(dtype=torch.float64, device=device)
    num_iterations = math.ceil(math.log(width[0] / resolution_for_virtual_projection, grid_factor)) # number of iterations needed to get the virtual pixel
    num_iterations = int(num_iterations)

    center = torch.zeros(2 * num_views, 1).to(dtype=torch.float64, device=device) 
    with torch.no_grad():
        for i in range(num_views):
            center[i*2:(i+1)*2, 0] = torch.tensor([width[i] / 2, height[i] / 2]).to(dtype=torch.float64, device=device)
        
        x = torch.linspace(-0.5 + 1/grid_factor/2, 0.5 - 1/grid_factor/2, grid_factor)
        y = torch.linspace(-0.5 + 1/grid_factor/2, 0.5 - 1/grid_factor/2, grid_factor) 
        [xx, yy] = torch.meshgrid(x, y)
        xx = xx.flatten().unsqueeze(-1).T
        yy = yy.flatten().unsqueeze(-1).T
        
        for _ in range(num_iterations):            
            virtual_pixels_grid = torch.vstack((xx * width[0] + center[0,0], yy * height[0] + center[1,0]))    
            for view in range(1, num_views):
                virtual_pixels_grid = torch.vstack((virtual_pixels_grid, 
                                                    torch.vstack((xx * width[view] + center[2*view,0], 
                                                                  yy * height[view] + center[2*view+1,0])
                                                                 )
                                                     )
                                                    ) 
            
            rays_dict = arena.pass_through_virtual_cam(virtual_pixels_grid, cam_label=cam_label_projection)
            if cam_label_projection == 'primary':
                ray = rays_dict["cam_1_ray_virtual"]
            elif cam_label_projection == 'secondary':
                ray = rays_dict["cam_2_ray_virtual"]
            
            if not(cam_label_projection == 'both'):
                distance = ray.distance_to_point(point)
                min_arg = torch.argmin(distance)
                center = virtual_pixels_grid[:, min_arg][:, None]
                width = [width_el / grid_factor for width_el in width]
                height = [height_el / grid_factor for height_el in height]

                
            elif cam_label_projection == 'both':                
                ray1, ray2 = rays_dict["cam_1_ray_virtual"], rays_dict["cam_2_ray_virtual"]                
                distance1 = ray1.distance_to_point(point)                
                min_arg1 = torch.argmin(distance1)
                distance2 = ray2.distance_to_point(point)
                min_arg2 = torch.argmin(distance2)            
                center = torch.vstack((virtual_pixels_grid[:2, min_arg1][:, None], virtual_pixels_grid[2:, min_arg2][:, None]))
                width = [width_el / grid_factor for width_el in width]
                height = [height_el / grid_factor for height_el in height]
    
    return center

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

dividing_col_dict = {}
dividing_col_dict['primary_real'] = dividing_col[0]
dividing_col_dict['primary_virtual'] = dividing_col[0]
dividing_col_dict['secondary_real'] = dividing_col[1]
dividing_col_dict['secondary_virtual'] = dividing_col[1]

image_width_dict = {}
image_width_dict['primary_real'] = image_width[0]
image_width_dict['primary_virtual'] = image_width[0]
image_width_dict['secondary_real'] = image_width[1]
image_width_dict['secondary_virtual'] = image_width[1]


if __name__ == "__main__":

    if not(type(user_annotation) == torch.Tensor):
        user_annotation = torch.tensor(user_annotation).to(torch.float64)[:, None]

    if not(user_annotation.dtype == torch.float64):
        user_annotation = user_annotation.to(torch.float64)

    if len(user_annotation.shape) == 1:
        user_annotation = user_annotation.unsqueeze(-1)

    if rotmat:
        theta = torch.tensor(torch.pi / 2)
        R_theta_inv = torch.tensor([[torch.cos(-theta), -torch.sin(-theta)], [torch.sin(-theta), torch.cos(-theta)]]).to(torch.float64)
        R_theta = torch.tensor([[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]]).to(torch.float64)
        user_annotation = R_theta_inv @ user_annotation
        if "virtual" in cam_label:
            user_annotation[0, :] = -user_annotation[0, :] + dividing_col_dict[cam_label] - 1  # 1-based indexing for dividing_col 
            user_annotation[1, :] = -user_annotation[1, :]
        if "real" in cam_label:
            user_annotation[0, :] = -user_annotation[0, :] + image_width_dict[cam_label] - 1
            user_annotation[1, :] = -user_annotation[1, :]


    epipolar_line_projecting_cam = get_epipolar_line(arena, user_annotation, cam_label, cam_projecting)
    if "virtual" in cam_projecting:
        if rotmat:
            epipolar_line_projecting_cam = R_theta @ epipolar_line_projecting_cam
            epipolar_line_projecting_cam[1, :] = -epipolar_line_projecting_cam[1, :] + dividing_col_dict[cam_projecting] - 1
            epipolar_line_projecting_cam[0, :] *= -1
        epipolar_line_projecting_cam = epipolar_line_projecting_cam.numpy()

    elif "real" in cam_projecting:
        if rotmat:
            epipolar_line_projecting_cam = R_theta @ epipolar_line_projecting_cam
            epipolar_line_projecting_cam[1, :] = -epipolar_line_projecting_cam[1, :] + image_width_dict[cam_projecting] - 1
            epipolar_line_projecting_cam[0, :] *= -1
        epipolar_line_projecting_cam = epipolar_line_projecting_cam.numpy()

print(f'End points of epipolar line in {cam_projecting} camera: {epipolar_line_projecting_cam[:,0]} and {epipolar_line_projecting_cam[:,1]}')

# %%
