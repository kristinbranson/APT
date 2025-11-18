import numpy as np
import torch
import torch.nn as nn
from ray_tracing_simulator_nnModules_grad_6d_rotations import Prism, Ray, Plane, ReflectingPlane, RefractingPlane, EfficientCamera, visualize_camera_configuration, closest_point, rotx, get_rot_mat, Rotation6D
from utils import euclidean_distance, rotation_matrix_to_quaternion
from pytorch3d.transforms import matrix_to_euler_angles
pi = torch.tensor(np.pi, dtype=torch.float64)
import math
import pdb

# 3-D reconstruction loss
class Arena_3D_loss(nn.Module):
    def __init__(self, 
    principal_point_pixel_cam_0, 
    principal_point_pixel_cam_1, 
    focal_length_cam_0, 
    focal_length_cam_1, 
    R, 
    T, 
    prism_distance=None,
    prism_angles=None,
    prism_center=None):
        super(Arena_3D_loss, self).__init__()
        
        if not isinstance(principal_point_pixel_cam_0, torch.Tensor):
            principal_point_pixel_cam_0 = torch.tensor(principal_point_pixel_cam_0)
        
        if not isinstance(principal_point_pixel_cam_1, torch.Tensor):
            principal_point_pixel_cam_1 = torch.tensor(principal_point_pixel_cam_1)

        if not isinstance(focal_length_cam_0, torch.Tensor):
            focal_length_cam_0 = torch.tensor(focal_length_cam_0)

        if not isinstance(focal_length_cam_1, torch.Tensor):
            focal_length_cam_1 = torch.tensor(focal_length_cam_1)

        # Camera initialization      
        principal_point_pixel_cam_0 = nn.Parameter(
            principal_point_pixel_cam_0.reshape(2,1),
            requires_grad=True,
        )  

        principal_point_pixel_cam_1 = nn.Parameter(
            principal_point_pixel_cam_1.reshape(2,1),
            requires_grad=True,
        )  

        focal_length_cam_0 = nn.Parameter(
            focal_length_cam_0,
            requires_grad=True,
        )

        focal_length_cam_1 = nn.Parameter(
            focal_length_cam_1,
            requires_grad=True,
        )

        r1_cam_0 = nn.Parameter(
            torch.tensor(1e-16, dtype=torch.float64),
            requires_grad=True,
        )

        self.r1_cam_1 = nn.Parameter(
            torch.tensor(1e-16, dtype=torch.float64),
            requires_grad=True,
        )


        self.camera1 = EfficientCamera(
            principal_point_pixel=principal_point_pixel_cam_0, 
            focal_length_pixels=focal_length_cam_0,
            r1=r1_cam_0)
        self.principal_point_pixel_cam_1 = principal_point_pixel_cam_1
        self.focal_length_cam_1 = focal_length_cam_1
        self.R = R
        self.T = nn.Parameter(torch.tensor(T, dtype=torch.float64), requires_grad=True)
        stereo_alpha, stereo_beta, stereo_gamma = self.get_stereo_camera_angles(R)
        self.stereo_camera_angles = nn.Parameter(
                                            torch.tensor([stereo_alpha, stereo_beta, stereo_gamma], 
                                                dtype=torch.float64,
                                                device=R.device),
                                            requires_grad=True
                                            )
        
        # Prism initialization
        prism_angles = nn.Parameter(prism_angles, requires_grad=True)
        refractive_index_glass = torch.tensor(1.51, dtype=torch.float64)
        refractive_index_glass = nn.Parameter(refractive_index_glass, requires_grad=True)
        if prism_center is None:
            prism_center = self.camera1.aperture.clone() + self.camera1.axes[:,0].unsqueeze(-1).clone() * prism_distance.clone()
            prism_center[0] = -5.

        prism_center = nn.Parameter(prism_center, requires_grad=True)
        prism_size = nn.Parameter(torch.tensor([20.,20.],dtype=torch.float64), requires_grad=True)
        #prism_size = torch.tensor([20.,20.,20.], dtype=torch.float64)
        self.prism = Prism(prism_size=prism_size, 
                        prism_center=prism_center, 
                        prism_angles=prism_angles,
                        refractive_index_glass=refractive_index_glass,
                        )
    
    def get_camera_2(self, 
                     principal_point_pixel_cam_1,
                     focal_length_cam_1,
                     stereo_camera_angles,
                     T,
                     r1):
        R = get_rot_mat(stereo_camera_angles[0], stereo_camera_angles[1], stereo_camera_angles[2])
        camera2 = EfficientCamera(
            principal_point_pixel=principal_point_pixel_cam_1, 
            focal_length_pixels=focal_length_cam_1,
            r1=r1)
        camera2.update_camera_pose(R, T)
        return camera2
    
    
    def get_stereo_camera_angles(self, 
                                 R_stereo_cam):
        axes = torch.eye(3,3).to(torch.float64)
        axes = torch.mm(R_stereo_cam, axes)
        plane = Plane(axes=axes)
        return plane.alpha, plane.beta, plane.gamma
    
    


    def forward(self, pixels_virtual_two_cams):
        camera2 = self.get_camera_2(self.principal_point_pixel_cam_1,
                                    self.focal_length_cam_1,
                                    self.stereo_camera_angles,
                                    self.T,
                                    self.r1_cam_1)
        undistorted_virtual_pixels_cam_0 = pixels_virtual_two_cams[:2,:]
        undistorted_virtual_pixels_cam_1 = pixels_virtual_two_cams[2:,:]
        undistorted_virtual_pixels_cam_0 = self.camera1.undistort_pixels(undistorted_virtual_pixels_cam_0)
        undistorted_virtual_pixels_cam_1 = camera2.undistort_pixels(undistorted_virtual_pixels_cam_1)
        #undistorted_virtual_pixels_cam_0 = pixels_virtual_two_cams[:2, :]
        #undistorted_virtual_pixels_cam_1 = pixels_virtual_two_cams[2:, :]
        #cam_1_ray = self.camera1.initialize_ray(undistorted_virtual_pixels_cam_0)
        cam_1_ray = self.camera1(undistorted_virtual_pixels_cam_0)
        #cam_2_ray = self.camera2.initialize_ray(undistorted_virtual_pixels_cam_1)
        cam_2_ray = camera2(undistorted_virtual_pixels_cam_1)
        prism_ray11, prism_ray12, emergent_ray_1, intersection_penalty_1 = self.prism(cam_1_ray)
        prism_ray21, prism_ray22, emergent_ray_2, intersection_penalty_2 = self.prism(cam_2_ray)
        recon_3D, closest_distance = closest_point(emergent_ray_1, emergent_ray_2)
        output = {}
        output['recon_3D'] = recon_3D
        output['closest_distance'] = closest_distance
        output['prism_ray11'] = prism_ray11
        output['prism_ray12'] = prism_ray12
        output['prism_ray21'] = prism_ray21
        output['prism_ray22'] = prism_ray22
        output['intersection_penalty_1'] = intersection_penalty_1
        output['intersection_penalty_2'] = intersection_penalty_2
        return output
    

    def visualize(self, pixels_virtual_two_cams, color_labels=None):
        num_samples = 10
        undistorted_virtual_pixels_cam_0 = pixels_virtual_two_cams[:2, :].clone()
        undistorted_virtual_pixels_cam_1 = pixels_virtual_two_cams[2:, :].clone()
        test_idx = torch.randperm(undistorted_virtual_pixels_cam_0.shape[1])[:num_samples]

        ray = self.camera1.initialize_ray(undistorted_virtual_pixels_cam_0[:, test_idx])
        ray.t *= 100
        fig, ax = self.camera1.visualize()
        fig, ax = ray.visualize(fig, ax, color_labels)
        prism_ray11, prism_ray12, emergent_ray1, _ = self.prism(ray)
        fig, ax = prism_ray11.visualize(fig, ax, color_labels)
        fig, ax = prism_ray12.visualize(fig, ax, color_labels)
        fig, ax = self.prism.visualize_prism(fig, ax)
        emergent_ray1.t *= 25
        fig, ax = emergent_ray1.visualize(fig, ax, color_labels)
        
        camera2 = self.get_camera_2(self.principal_point_pixel_cam_1,
                                self.focal_length_cam_1,
                                self.stereo_camera_angles,
                                self.T,
                                self.r1_cam_1)
        ray = camera2.initialize_ray(undistorted_virtual_pixels_cam_1[:, test_idx])
        ray.t *= 100
        fig, ax = camera2.visualize(fig, ax)
        fig, ax = ray.visualize(fig, ax, color_labels)
        prism_ray21, prism_ray22, emergent_ray2, _ = self.prism(ray)
        fig, ax = self.prism.visualize_prism(fig, ax)
        fig, ax = prism_ray21.visualize(fig, ax, color_labels)
        fig, ax = prism_ray22.visualize(fig, ax, color_labels)
        emergent_ray2.t *= 25
        emergent_ray2.visualize(fig, ax, color_labels)
        ax.set_aspect('equal', adjustable='datalim')   

# Pixel reprojection error
class Arena_reprojection_loss_two_cameras_prism(nn.Module):
    def __init__(self, 
    principal_point_pixel_cam_0, 
    principal_point_pixel_cam_1, 
    focal_length_cam_0, 
    focal_length_cam_1, 
    R_stereo_cam, 
    T_stereo_cam, 
    prism_distance=None,
    prism_angles=None,
    prism_center=None):
        super(Arena_reprojection_loss_two_cameras_prism, self).__init__()

        # Camera initialization      
        principal_point_pixel_cam_0 = nn.Parameter(
            torch.tensor(principal_point_pixel_cam_0, dtype=torch.float64).reshape(2,1),
            requires_grad=True,
        )  

        principal_point_pixel_cam_1 = nn.Parameter(
            torch.tensor(principal_point_pixel_cam_1, dtype=torch.float64).reshape(2,1),
            requires_grad=True,
        )  

        focal_length_cam_0 = nn.Parameter(
            torch.tensor(focal_length_cam_0, dtype=torch.float64),
            requires_grad=True,
        )

        focal_length_cam_1 = nn.Parameter(
            torch.tensor(focal_length_cam_1, dtype=torch.float64),
            requires_grad=True,
        )

        self.r1 = nn.Parameter(
            torch.tensor(1e-6, dtype=torch.float64),
            requires_grad=True,
        )
        self.stereocam_r1 = nn.Parameter(
            torch.tensor(1e-6, dtype=torch.float64),
            requires_grad=True,
        )

        self.camera1 = EfficientCamera(
            principal_point_pixel=principal_point_pixel_cam_0,
            focal_length_pixels=focal_length_cam_0,
            r1=self.r1)

        self.principal_point_pixel_cam_1 = principal_point_pixel_cam_1
        self.focal_length_cam_1 = focal_length_cam_1
        self.R_stereo_cam = R_stereo_cam
        self.T_stereo_cam = nn.Parameter(T_stereo_cam, requires_grad=True)
        stereo_alpha, stereo_beta, stereo_gamma = self.get_stereo_camera_angles(R_stereo_cam)
        self.stereo_camera_angles = nn.Parameter(
                                                torch.tensor([stereo_alpha, stereo_beta, stereo_gamma], 
                                                    dtype=torch.float64, 
                                                    device=R_stereo_cam.device), 
                                                requires_grad=True
                                                )


        # Prism initialization
        prism_angles = nn.Parameter(prism_angles, requires_grad=True)
        refractive_index_glass = torch.tensor(1.5, dtype=torch.float64)
        refractive_index_glass = nn.Parameter(refractive_index_glass, requires_grad=True)
        if prism_center is None:
            prism_center = self.camera1.aperture.clone() + self.camera1.axes[:,0].unsqueeze(-1).clone() * prism_distance.clone()
            prism_center[0] = -5.

        prism_center = nn.Parameter(prism_center, requires_grad=True)
        prism_size = nn.Parameter(torch.tensor([20.,20.,20.],dtype=torch.float64), requires_grad=True)

        self.prism = Prism(prism_size=prism_size, 
                        prism_center=prism_center, 
                        prism_angles=prism_angles,
                        refractive_index_glass=refractive_index_glass,
                        )
        

    def get_stereo_camera_angles(self, 
                                 R_stereo_cam):
        axes = torch.eye(3,3).to(torch.float64)
        axes = torch.mm(R_stereo_cam, axes)
        plane = Plane(axes=axes)
        return plane.alpha, plane.beta, plane.gamma
    

    def get_stereo_camera(self, 
                     principal_point_pixel_cam_1,
                     focal_length_cam_1,
                     R,
                     T,
                     r1):
        camera2 = EfficientCamera(
            principal_point_pixel=principal_point_pixel_cam_1, 
            focal_length_pixels=focal_length_cam_1, 
            r1=r1)
        camera2.update_camera_pose(R, T)
        return camera2



    def forward(self, pixels_virtual_two_cams, pixels_real_two_cams):
        R_stereo_cam = get_rot_mat(
            self.stereo_camera_angles[0],
            self.stereo_camera_angles[1],
            self.stereo_camera_angles[2],
            )
        R1 = torch.eye(3, 3).to(torch.float64)
        T1 = torch.zeros(3, 1).to(torch.float64)
        R2 = R_stereo_cam
        T2 = self.T_stereo_cam
        camera2 = self.get_stereo_camera(self.principal_point_pixel_cam_1,
                                    self.focal_length_cam_1,
                                    R_stereo_cam,
                                    self.T_stereo_cam,
                                    r1=self.stereocam_r1)
        distorted_virtual_pixels_cam_0 = pixels_virtual_two_cams[:2,:]
        distorted_virtual_pixels_cam_1 = pixels_virtual_two_cams[2:,:]
        undistorted_virtual_pixels_cam_0 = self.camera1.undistort_pixels(distorted_virtual_pixels_cam_0)
        undistorted_virtual_pixels_cam_1 = camera2.undistort_pixels(distorted_virtual_pixels_cam_1)
        distorted_real_pixels_cam_0 = pixels_real_two_cams[:2,:]
        distorted_real_pixels_cam_1 = pixels_real_two_cams[2:,:]
        undistorted_real_pixels_cam_0 = self.camera1.undistort_pixels(distorted_real_pixels_cam_0)
        undistorted_real_pixels_cam_1 = camera2.undistort_pixels(distorted_real_pixels_cam_1)
        cam_1_ray_real = self.camera1(undistorted_real_pixels_cam_0)
        cam_2_ray_real = camera2(undistorted_real_pixels_cam_1)
        recon_3D_real, closest_distance_real = closest_point(cam_1_ray_real, cam_2_ray_real)
        recon_undistorted_real_pixels_cam_0 = self.camera1.reproject(recon_3D_real, R1, T1)
        recon_undistorted_real_pixels_cam_1 = camera2.reproject(recon_3D_real, R2, T2)
        recon_distorted_real_pixels_cam_0 = self.camera1.distort_pixels(recon_undistorted_real_pixels_cam_0)
        recon_distorted_real_pixels_cam_1 = camera2.distort_pixels(recon_undistorted_real_pixels_cam_1)
        cam_1_ray_virtual = self.camera1(undistorted_virtual_pixels_cam_0)
        cam_2_ray_virtual = camera2(undistorted_virtual_pixels_cam_1)
        _, _, cam_1_ray_virtual, intersection_penalty_1 = self.prism(cam_1_ray_virtual)
        _, _, cam_2_ray_virtual, intersection_penalty_2 = self.prism(cam_2_ray_virtual)
        recon_3D_virtual, closest_distance_virtual = closest_point(cam_1_ray_virtual, cam_2_ray_virtual)
        recon_3D_1, closest_distance_12 = closest_point(cam_1_ray_virtual, cam_2_ray_real)
        recon_3D_2, closest_distance_21 = closest_point(cam_1_ray_virtual, cam_2_ray_real)
        recon_3D = (recon_3D_real + recon_3D_1 + recon_3D_2 + recon_3D_virtual) / 4
        closest_distance_mean = (closest_distance_12 + closest_distance_21 + closest_distance_real + closest_distance_virtual) / 4
        recon_pixels_1 = self.camera1.reproject(recon_3D, R1, T1)
        recon_pixels_1 = self.camera1.distort_pixels(recon_pixels_1)
        recon_pixels_2 = camera2.reproject(recon_3D, R2, T2)
        recon_pixels_2 = self.camera1.distort_pixels(recon_pixels_2)
        recon_undistorted_virtual_pixels_cam_0 = self.camera1.reproject(recon_3D_virtual, R1, T1)
        recon_undistorted_virtual_pixels_cam_1 = camera2.reproject(recon_3D_virtual, R2, T2)

        # Distortion loss (constraining the distortion and undistortion function to be the inverse of each other)
        recon_distorted_virtual_pixels_cam_0 = self.camera1.distort_pixels_MLP(recon_undistorted_virtual_pixels_cam_0)
        recon_distorted_virtual_pixels_cam_1 = camera2.distort_pixels_MLP(recon_undistorted_virtual_pixels_cam_1)        
        distortion_penalty_cam_0 = self.camera1.calculate_distortion_penalty(distorted_virtual_pixels_cam_0) + self.camera1.calculate_distortion_penalty(distorted_real_pixels_cam_0)
        distortion_penalty_cam_1 = camera2.calculate_distortion_penalty(distorted_virtual_pixels_cam_1) + camera2.calculate_distortion_penalty(distorted_real_pixels_cam_1)    
        output = {}
        output['recon_3D'] = recon_3D
        output['closest_distance_mean'] = closest_distance_mean
        output['closest_distance_virtual'] = closest_distance_virtual
        output['closest_distance_real'] = closest_distance_real
        output['closest_distance_12'] = closest_distance_12
        output['closest_distance_21'] = closest_distance_21
        output['recon_pixels_1'] = recon_pixels_1
        output['recon_pixels_2'] = recon_pixels_2
        output['recon_distorted_virtual_pixels_cam_0'] = recon_distorted_virtual_pixels_cam_0
        output['recon_distorted_virtual_pixels_cam_1'] = recon_distorted_virtual_pixels_cam_1
        output['recon_distorted_real_pixels_cam_0'] = recon_distorted_real_pixels_cam_0
        output['recon_distorted_real_pixels_cam_1'] = recon_distorted_real_pixels_cam_1
        output['intersection_penalty_1'] = intersection_penalty_1
        output['intersection_penalty_2'] = intersection_penalty_2
        output['distortion_penalty_cam_0'] = distortion_penalty_cam_0
        output['distortion_penalty_cam_1'] = distortion_penalty_cam_1
        return output


    def visualize(self, pixels_virtual_two_cams, color_labels=None):
        num_samples = 10
        undistorted_virtual_pixels_cam_0 = pixels_virtual_two_cams[:2, :].clone()
        undistorted_virtual_pixels_cam_1 = pixels_virtual_two_cams[2:, :].clone()
        test_idx = torch.randperm(undistorted_virtual_pixels_cam_0.shape[1])[:num_samples]

        ray = self.camera1.initialize_ray(undistorted_virtual_pixels_cam_0[:, test_idx])
        ray.t *= 100
        fig, ax = self.camera1.visualize()
        fig, ax = ray.visualize(fig, ax, color_labels)
        prism_ray11, prism_ray12, emergent_ray1, intersection_penalty_1 = self.prism(ray)
        fig, ax = prism_ray11.visualize(fig, ax, color_labels)
        fig, ax = prism_ray12.visualize(fig, ax, color_labels)
        
        fig, ax = self.prism.visualize_prism(fig, ax)
        emergent_ray1.t *= 25
        fig, ax = emergent_ray1.visualize(fig, ax, color_labels)
        R_stereo_cam = get_rot_mat(
            self.stereo_camera_angles[0],
            self.stereo_camera_angles[1],
            self.stereo_camera_angles[2],
        )
        camera2 = self.get_stereo_camera(self.principal_point_pixel_cam_1,
                                self.focal_length_cam_1,
                                R_stereo_cam,
                                self.T_stereo_cam,
                                self.stereocam_r1)
        ray = camera2.initialize_ray(undistorted_virtual_pixels_cam_1[:, test_idx])
        ray.t *= 100
        fig, ax = camera2.visualize(fig, ax)
        fig, ax = ray.visualize(fig, ax, color_labels)
        prism_ray21, prism_ray22, emergent_ray2, intersection_penalty_2 = self.prism(ray)
        fig, ax = self.prism.visualize_prism(fig, ax)
        fig, ax = prism_ray21.visualize(fig, ax, color_labels)
        fig, ax = prism_ray22.visualize(fig, ax, color_labels)
        emergent_ray2.t *= 25
        emergent_ray2.visualize(fig, ax, color_labels)
        ax.set_aspect('equal', adjustable='datalim')   


class Arena_reprojection_loss_two_cameras_prism_grid_distances(nn.Module):
    def __init__(self, 
    principal_point_pixel_cam_0, 
    principal_point_pixel_cam_1, 
    focal_length_cam_0, 
    focal_length_cam_1, 
    R_stereo_cam, 
    T_stereo_cam, 
    prism_distance=None,
    prism_angles=None,
    prism_center=None,
    outlier_threshold=1e3, # maximum closest distance error you're okay with
    ):
        super(Arena_reprojection_loss_two_cameras_prism_grid_distances, self).__init__()

        # Camera initialization      
        principal_point_pixel_cam_0 = nn.Parameter(
            torch.tensor(principal_point_pixel_cam_0, dtype=torch.float64).reshape(2,1),
            requires_grad=True,
        )  

        principal_point_pixel_cam_1 = nn.Parameter(
            torch.tensor(principal_point_pixel_cam_1, dtype=torch.float64).reshape(2,1),
            requires_grad=True,
        )  

        focal_length_cam_0 = nn.Parameter(
            torch.tensor(focal_length_cam_0, dtype=torch.float64),
            requires_grad=True,
        )

        focal_length_cam_1 = nn.Parameter(
            torch.tensor(focal_length_cam_1, dtype=torch.float64),
            requires_grad=True,
        )

        self.r1 = nn.Parameter(
            torch.tensor(1e-6, dtype=torch.float64),
            requires_grad=True,
        )
        self.stereocam_r1 = nn.Parameter(
            torch.tensor(1e-6, dtype=torch.float64),
            requires_grad=True,
        )
        self.virtual_proj_prob_thresh = 0. # Default is to not use virtual reprojection for reprojection error computation (since it is very expensive). Can be set to a value between 0 and 1

        self.radial_dist_coeffs_cam_0 = nn.Parameter(torch.tensor([0.,0.,0.]).unsqueeze(-1).to(torch.float64),
                                               requires_grad=True)
        
        self.radial_dist_coeffs_cam_1 = nn.Parameter(torch.tensor([0.,0.,0.]).unsqueeze(-1).to(torch.float64),
                                               requires_grad=True)

        self.camera1 = EfficientCamera(
            principal_point_pixel=principal_point_pixel_cam_0,
            focal_length_pixels=focal_length_cam_0,
            r1=self.r1,
            radial_dist_coeffs=self.radial_dist_coeffs_cam_0)

        self.principal_point_pixel_cam_1 = principal_point_pixel_cam_1
        self.focal_length_cam_1 = focal_length_cam_1
        self.R_stereo_cam = R_stereo_cam
        self.T_stereo_cam = nn.Parameter(T_stereo_cam, requires_grad=True)
        stereo_alpha, stereo_beta, stereo_gamma = self.get_stereo_camera_angles(R_stereo_cam)
        self.stereo_camera_rotation_6d = Rotation6D(
                stereo_gamma,
                stereo_beta,
                stereo_alpha
                )
        self.outlier_threshold = outlier_threshold
        #self.stereo_camera_angles = nn.Parameter(
        #                                        torch.tensor([stereo_alpha, stereo_beta, stereo_gamma], 
        #                                                     dtype=torch.float64,
        #                                                     device=R_stereo_cam.device),
        #                                        requires_grad=True
        #                                        ) # Deprecated


        # Prism initialization
        #self.prism_angles = nn.Parameter(prism_angles, requires_grad=True) 
        self.prism_rotation_6d = Rotation6D(prism_angles[2],
                                            prism_angles[1],
                                            prism_angles[0])

        refractive_index_glass = torch.tensor(1.51, dtype=torch.float64)
        self.refractive_index_glass = nn.Parameter(refractive_index_glass, requires_grad=True)
        if prism_center is None:
            prism_center = self.camera1.aperture.clone() + self.camera1.axes[:,0].unsqueeze(-1).clone() * prism_distance.clone()
            prism_center[0] = -5.

        self.prism_center = nn.Parameter(prism_center, requires_grad=True)
        self.prism_size = nn.Parameter(
                                        torch.tensor(
                                        [20.,20.,20.],
                                        dtype=torch.float64), 
                                        requires_grad=True
                                        )

        self.prism = Prism(
                        prism_size=self.prism_size, 
                        prism_center=self.prism_center, 
                        prism_rotation_6d=self.prism_rotation_6d,
                        refractive_index_glass=self.refractive_index_glass,
                        )
        

    def get_stereo_camera_angles(self, 
                                 R_stereo_cam):
        gamma, beta, alpha = matrix_to_euler_angles(R_stereo_cam, 'ZYX')
        return alpha, beta, gamma
        axes = torch.eye(3,3).to(device=R_stereo_cam.device, dtype=torch.float64)
        axes = torch.mm(R_stereo_cam, axes)
        plane = Plane(axes=axes)
        return plane.alpha, plane.beta, plane.gamma
    

    def get_stereo_camera(self, 
                     principal_point_pixel_cam_1,
                     focal_length_cam_1,
                     R,
                     T,
                     r1,
                     radial_dist_coeffs):
        camera2 = EfficientCamera(
            principal_point_pixel=principal_point_pixel_cam_1, 
            focal_length_pixels=focal_length_cam_1, 
            r1=r1,
            radial_dist_coeffs=radial_dist_coeffs)
        camera2.update_camera_pose(R, T)
        return camera2
    

    def get_prism_corners(self):
        front_plane, reflecting_plane, top_plane = self.prism.get_planes(prism_center=self.prism.prism_center,
                                                                         prism_rotation_6d=self.prism_rotation_6d)
        front_plane_corners = front_plane.get_plane_corners()
        reflecting_plane_corners = reflecting_plane.get_plane_corners()
        top_plane_corners = top_plane.get_plane_corners()
        return front_plane_corners, reflecting_plane_corners, top_plane_corners
    
    def detect_outliers_vectorized(self, stacked, threshold):
        """
        Fully vectorized version for maximum efficiency.
        If a point needs fallback (< 1 non-NaN values after outlier removal),
        keep all values for that point (don't remove any outliers) except for the extreme outliers defined by upper bound threshold.
        If a point needs fall back after this and all values are outliers, keep all original values.
        Returns:
            tuple: (cleaned_tensors, mask)
                - cleaned_tensors: List of tensors with outliers marked as NaN
                - mask: Tensor of shape (6, num_points) indicating which values are used (True = used, False = NaN)
        """
        num_tensors, num_points = stacked.shape    
        
        # Initialize per-point thresholds
        point_thresholds = torch.full((num_points,), threshold, device=stacked.device)
        
        keep_searching_outliers = True
        
        while keep_searching_outliers:
            # Apply per-point thresholds
            result = stacked.clone()
            
            # Create outlier mask using per-point thresholds
            outlier_mask = torch.abs(stacked) > point_thresholds.unsqueeze(0)  # Broadcast to (num_tensors, num_points)
            result[outlier_mask] = float('nan')
            
            # Count non-NaN values per point after outlier removal
            non_nan_count = (~torch.isnan(result)).sum(dim=0)
            needs_fallback = non_nan_count < 1  # no stereo pair with non nan value
            
            if not needs_fallback.any():
                keep_searching_outliers = False
            else:
                # Only increase threshold for points that need fallback
                point_thresholds[needs_fallback] *= 2
        
        # Create usage mask: True where values are not NaN
        usage_mask = ~torch.isnan(result)
        return [result[i] for i in range(num_tensors)], usage_mask
    
    
    def return_reprojected_pixels(self, recon_3D, R1, T1, R2, T2, camera2, image_width, image_height):
        """
        Returns the reprojected pixels in both real and virtual views for both cameras
        """
        recon_pixels_1_undistorted_real = self.camera1.reproject(recon_3D, R1, T1) # Note that this only reprojects in the real view
        recon_pixels_1_real = self.camera1.distort_pixels_classical(recon_pixels_1_undistorted_real, self.radial_dist_coeffs_cam_0)
        recon_pixels_2_undistorted_real = camera2.reproject(recon_3D, R2, T2) # Note that this only reprojects in the real view
        recon_pixels_2_real = camera2.distort_pixels_classical(recon_pixels_2_undistorted_real, self.radial_dist_coeffs_cam_1)
        recon_pixels_undistorted_virtual = self.pseudo_reprojection_to_virtual_view(recon_3D, image_width=image_width, image_height=image_height, cam_label_projection='both')
        recon_pixels_1_undistorted_virtual, recon_pixels_2_undistorted_real = recon_pixels_undistorted_virtual[:2,...], recon_pixels_undistorted_virtual[2:,...]
        recon_pixels_1_virtual = self.camera1.distort_pixels_classical(recon_pixels_1_undistorted_virtual, self.radial_dist_coeffs_cam_0)
        recon_pixels_2_virtual = camera2.distort_pixels_classical(recon_pixels_2_undistorted_real, self.radial_dist_coeffs_cam_1)
        return recon_pixels_1_virtual, recon_pixels_1_real, recon_pixels_2_virtual, recon_pixels_2_real

    def vote_best_views(
            self,
            v1_stacked_error, 
            r1_stacked_error, 
            v2_stacked_error, 
            r2_stacked_error, 
            recon_3D_virtual,
            recon_3D_real,
            recon_3D_1,
            recon_3D_2,
            recon_3D_12,
            recon_3D_21,
            threshold):
        n_keypoints = v1_stacked_error.shape[1]
        keypoint_thresholds = torch.full((n_keypoints,), threshold, device=v1_stacked_error.device)
        relax_threshold = True
        fly_body_length = 75 #pixels
        
        while relax_threshold:
            num_agreeing_views_needed = 2
            # So that the while loop does not continue infinitely. This is for cases where only 2 views have keypoint predictions available
            if (keypoint_thresholds > fly_body_length).any():
                num_agreeing_views_needed = 1 
            if (keypoint_thresholds > 1000).any(): # 1000 is of the order of the image dimension. Just walk out if this criteria is not met
                num_agreeing_views_needed = 0
            votes_1v = v1_stacked_error < keypoint_thresholds.unsqueeze(0)  # (6, N_keypoints) boolean mask
            votes_1r = r1_stacked_error < keypoint_thresholds.unsqueeze(0)
            votes_2v = v2_stacked_error < keypoint_thresholds.unsqueeze(0)
            votes_2r = r2_stacked_error < keypoint_thresholds.unsqueeze(0)
            # Count votes per keypoint per camera
            vote_counts_1v = votes_1v.sum(dim=0)  # (N_keypoints,)
            vote_counts_2v = votes_2v.sum(dim=0)
            vote_counts_1r = votes_1r.sum(dim=0)
            vote_counts_2r = votes_2r.sum(dim=0)

            virtual_mask = (vote_counts_1v >= num_agreeing_views_needed) & (vote_counts_2v >= num_agreeing_views_needed)
            real_mask = (vote_counts_1r >= num_agreeing_views_needed) & (vote_counts_2r >= num_agreeing_views_needed)
            cam1_mask = (vote_counts_1v >= num_agreeing_views_needed) & (vote_counts_1r >= num_agreeing_views_needed)
            cam2_mask = (vote_counts_2v >= num_agreeing_views_needed) & (vote_counts_2r >= num_agreeing_views_needed)
            virtual1_real2_mask = (vote_counts_1v >= num_agreeing_views_needed) & (vote_counts_2r >= num_agreeing_views_needed)
            virtual2_real1_mask = (vote_counts_2v >= num_agreeing_views_needed) & (vote_counts_1r >= num_agreeing_views_needed)

            stacked_masks = torch.vstack((
                virtual_mask,
                real_mask,
                cam1_mask,
                cam2_mask,
                virtual1_real2_mask,
                virtual2_real1_mask,
            ))
            # Ensure each keypoints is assigned to some category
            valid_keypoints = stacked_masks.any(dim=0) 
            invalid_keypoints = ~valid_keypoints
            if valid_keypoints.all():
                relax_threshold = False
            else:
                keypoint_thresholds[invalid_keypoints] *= 2

        final_recon_3D = (
            recon_3D_virtual * virtual_mask.unsqueeze(0) + recon_3D_real * real_mask.unsqueeze(0) + recon_3D_1 * cam1_mask.unsqueeze(0) + recon_3D_2 * cam2_mask.unsqueeze(0) + recon_3D_12 * virtual1_real2_mask.unsqueeze(0) + recon_3D_21 * virtual2_real1_mask.unsqueeze(0)
        ) / (virtual_mask.float().unsqueeze(0) + real_mask.float().unsqueeze(0) + cam1_mask.float().unsqueeze(0) + cam2_mask.float().unsqueeze(0) + virtual1_real2_mask.float().unsqueeze(0) + virtual2_real1_mask.float().unsqueeze(0) + 1e-6)

        return final_recon_3D


    def triangulate(self, pixels, image_width, image_height, discard_outliers=True):
        """
        Triangulate 3D points from pixel coordinates in two real and virtual cameras using the prism
        Coordinates can be nan if the pixel detections are missing for certain cameras
        Args:
        pixels: torch.Tensor of shape (2, N, npts) where N is the number of views, npts is the number of points. pixels are stacked in the order primary_virtual, primary_real, secondary_virtual, secondary_real
        image_width: list, width of the images for each camera
        image_height: list, height of the virtual images for each camera
        """
        if len(pixels.shape) == 2:
            # If only one point is triangulated, and the point (third) dimension does not exist
            pixels = pixels[..., None] # Add the point dimension

        self.prism = Prism(prism_size=self.prism_size,
                        prism_center=self.prism_center,
                        prism_rotation_6d=self.prism_rotation_6d,
                        refractive_index_glass=self.refractive_index_glass,
                        )

        R_stereo_cam = self.stereo_camera_rotation_6d.matrix()
        R1 = torch.eye(3, 3).to(device=pixels.device, dtype=torch.float64)
        T1 = torch.zeros(3, 1).to(device=pixels.device, dtype=torch.float64)
        R2 = R_stereo_cam
        T2 = self.T_stereo_cam
        camera2 = self.get_stereo_camera(self.principal_point_pixel_cam_1,
                                    self.focal_length_cam_1,
                                    R_stereo_cam,
                                    self.T_stereo_cam,
                                    r1=self.stereocam_r1,
                                    radial_dist_coeffs=self.radial_dist_coeffs_cam_1)
        # Read pixels for each view
        distorted_virtual_pixels_cam_0 = pixels[:, 0, :]
        distorted_real_pixels_cam_0 = pixels[:, 1, :]
        distorted_virtual_pixels_cam_1 = pixels[:, 2, :]
        distorted_real_pixels_cam_1 = pixels[:, 3, :]

        undistorted_real_pixels_cam_0 = self.camera1.undistort_pixels_classical(distorted_real_pixels_cam_0,
                                                                                self.radial_dist_coeffs_cam_0)
        undistorted_virtual_pixels_cam_0 = self.camera1.undistort_pixels_classical(distorted_virtual_pixels_cam_0,
                                                                                self.radial_dist_coeffs_cam_0)
        undistorted_real_pixels_cam_1 = camera2.undistort_pixels_classical(distorted_real_pixels_cam_1,
                                                                            self.radial_dist_coeffs_cam_1)
        undistorted_virtual_pixels_cam_1 = camera2.undistort_pixels_classical(distorted_virtual_pixels_cam_1,
                                                                            self.radial_dist_coeffs_cam_1)

        cam_1_ray_real = self.camera1(undistorted_real_pixels_cam_0)
        cam_2_ray_real = camera2(undistorted_real_pixels_cam_1)
        recon_3D_real, closest_distance_real = closest_point(cam_1_ray_real, cam_2_ray_real)

        cam_1_ray_virtual = self.camera1(undistorted_virtual_pixels_cam_0)
        cam_2_ray_virtual = camera2(undistorted_virtual_pixels_cam_1)
        _, _, cam_1_ray_virtual, intersection_penalty_1 = self.prism(cam_1_ray_virtual)
        _, _, cam_2_ray_virtual, intersection_penalty_2 = self.prism(cam_2_ray_virtual)
        recon_3D_virtual, closest_distance_virtual = closest_point(cam_1_ray_virtual, cam_2_ray_virtual)
        recon_3D_1, closest_distance_1 = closest_point(cam_1_ray_virtual, cam_1_ray_real)
        recon_3D_2, closest_distance_2 = closest_point(cam_2_ray_virtual, cam_2_ray_real)
        recon_3D_12, closest_distance_12 = closest_point(cam_1_ray_virtual, cam_2_ray_real)
        recon_3D_21, closest_distance_21 = closest_point(cam_2_ray_virtual, cam_1_ray_real)
        

        if discard_outliers:
            # Find outliers based on closest distance error
            #combined_tensors = torch.vstack((closest_distance_real, closest_distance_virtual, closest_distance_1, closest_distance_2, closest_distance_12, closest_distance_21))
            #combined_tensors, usage_mask = self.detect_outliers_vectorized(combined_tensors, self.outlier_threshold)
            #closest_distance_real, closest_distance_virtual, closest_distance_1, closest_distance_2, closest_distance_12, closest_distance_21 = combined_tensors
            # Find outliers based on reprojection error                    
            # Get reprojections from all pairs in all views
            rep_1v_real, rep_1r_real, rep_2v_real, rep_2r_real = self.return_reprojected_pixels(recon_3D_real, R1, T1, R2, T2, camera2, image_width, image_height)
            rep_1v_virtual, rep_1r_virtual, rep_2v_virtual, rep_2r_virtual = self.return_reprojected_pixels(recon_3D_virtual, R1, T1, R2, T2, camera2, image_width, image_height)
            rep_1v_1, rep_1r_1, rep_2v_1, rep_2r_1 = self.return_reprojected_pixels(recon_3D_1, R1, T1, R2, T2, camera2, image_width, image_height)
            rep_1v_2, rep_1r_2, rep_2v_2, rep_2r_2 = self.return_reprojected_pixels(recon_3D_2, R1, T1, R2, T2, camera2, image_width, image_height)
            rep_1v_12, rep_1r_12, rep_2v_12, rep_2r_12 = self.return_reprojected_pixels(recon_3D_12, R1, T1, R2, T2, camera2, image_width, image_height)
            rep_1v_21, rep_1r_21, rep_2v_21, rep_2r_21 = self.return_reprojected_pixels(recon_3D_21, R1, T1, R2, T2, camera2, image_width, image_height)
            # Calculate reprojection errors for each of the above views
            repr_error_1r_real = torch.norm(rep_1r_real - distorted_real_pixels_cam_0, dim=0)
            repr_error_1v_real = torch.norm(rep_1v_real - distorted_virtual_pixels_cam_0, dim=0)
            repr_error_2r_real = torch.norm(rep_2r_real - distorted_real_pixels_cam_1, dim=0)
            repr_error_2v_real = torch.norm(rep_2v_real - distorted_virtual_pixels_cam_1, dim=0)
            repr_error_r_real = torch.maximum(repr_error_1r_real, repr_error_2r_real)
            repr_error_1v_virtual = torch.norm(rep_1v_virtual - distorted_virtual_pixels_cam_0, dim=0)
            repr_error_1r_virtual = torch.norm(rep_1r_virtual - distorted_real_pixels_cam_0, dim=0)
            repr_error_2v_virtual = torch.norm(rep_2v_virtual - distorted_virtual_pixels_cam_1, dim=0)
            repr_error_2r_virtual = torch.norm(rep_2r_virtual - distorted_real_pixels_cam_1, dim=0)
            repr_error_virtual = torch.maximum(repr_error_1v_virtual, repr_error_2v_virtual)
            repr_error_1v_1 = torch.norm(rep_1v_1 - distorted_virtual_pixels_cam_0, dim=0)            
            repr_error_1r_1 = torch.norm(rep_1r_1 - distorted_real_pixels_cam_0, dim=0)
            repr_error_2v_1 = torch.norm(rep_2v_1 - distorted_virtual_pixels_cam_1, dim=0)
            repr_error_2r_1 = torch.norm(rep_2r_1 - distorted_real_pixels_cam_1, dim=0)
            repr_error_1 = torch.maximum(repr_error_1v_1, repr_error_1r_1)
            repr_error_1v_2 = torch.norm(rep_1v_2 - distorted_virtual_pixels_cam_0, dim=0)
            repr_error_1r_2 = torch.norm(rep_1r_2 - distorted_real_pixels_cam_0, dim=0)
            repr_error_2v_2 = torch.norm(rep_2v_2 - distorted_virtual_pixels_cam_1, dim=0)            
            repr_error_2r_2 = torch.norm(rep_2r_2 - distorted_real_pixels_cam_1, dim=0)
            repr_error_2 = torch.maximum(repr_error_2v_2, repr_error_2r_2)
            repr_error_1v_12 = torch.norm(rep_1v_12 - distorted_virtual_pixels_cam_0, dim=0)
            repr_error_1r_12 = torch.norm(rep_1r_12 - distorted_real_pixels_cam_0, dim=0)
            repr_error_2v_12 = torch.norm(rep_2v_12 - distorted_virtual_pixels_cam_1, dim=0)
            repr_error_2r_12 = torch.norm(rep_2r_12 - distorted_real_pixels_cam_1, dim=0)
            repr_error_12 = torch.maximum(repr_error_1v_12, repr_error_2r_12)
            repr_error_1v_21 = torch.norm(rep_1v_21 - distorted_virtual_pixels_cam_0, dim=0)
            repr_errors_1r_21 = torch.norm(rep_1r_21 - distorted_real_pixels_cam_0, dim=0)
            repr_error_2v_21 = torch.norm(rep_2v_21 - distorted_virtual_pixels_cam_1, dim=0)            
            repr_error_2r_21 = torch.norm(rep_2r_21 - distorted_real_pixels_cam_1, dim=0)
            repr_error_21 = torch.maximum(repr_error_2v_21, repr_errors_1r_21)
            v1_stacked_error = torch.vstack([repr_error_1v_virtual, repr_error_1v_real, repr_error_1v_1, repr_error_1v_2, repr_error_1v_12, repr_error_1v_21])
            r1_stacked_error = torch.vstack([repr_error_1r_virtual, repr_error_1r_real, repr_error_1r_1, repr_error_1r_2, repr_error_1r_12, repr_errors_1r_21])
            v2_stacked_error = torch.vstack([repr_error_2v_virtual, repr_error_2v_real, repr_error_2v_1, repr_error_2v_2, repr_error_2v_12, repr_error_2v_21])
            r2_stacked_error = torch.vstack([repr_error_2r_virtual, repr_error_2r_real, repr_error_2r_1, repr_error_2r_2, repr_error_2r_12, repr_error_2r_21])
            recon_3D = self.vote_best_views(
            v1_stacked_error, 
            r1_stacked_error, 
            v2_stacked_error, 
            r2_stacked_error, 
            recon_3D_virtual,
            recon_3D_real,
            recon_3D_1,
            recon_3D_2,
            recon_3D_12,
            recon_3D_21,
            self.outlier_threshold)

            """repr_errors_stacked = torch.stack([
                repr_error_r_real,
                repr_error_virtual,
                repr_error_1,
                repr_error_2,
                repr_error_12,
                repr_error_21,            
            ], dim=0)
            _, usage_mask = self.detect_outliers_vectorized(repr_errors_stacked, self.outlier_threshold)            
            combined_tensors_recon_3D = torch.stack([recon_3D_real, recon_3D_virtual, recon_3D_1, recon_3D_2, recon_3D_12, recon_3D_21], dim=0)
            masked_combined_tensors_3D = combined_tensors_recon_3D.clone()
            masked_combined_tensors_3D[~usage_mask.unsqueeze(1).expand(-1, 3, -1)] = float('nan')
            recon_3D_real, recon_3D_virtual, recon_3D_1, recon_3D_2, recon_3D_12, recon_3D_21 = masked_combined_tensors_3D

        recon_3D = torch.nanmean(
            torch.stack([recon_3D_real, recon_3D_virtual, recon_3D_1, recon_3D_2, recon_3D_12, recon_3D_21],
                        dim=0),
                    dim=0,
                )"""
        else:
            recon_3D = torch.nanmean(
            torch.stack([recon_3D_real, recon_3D_virtual, recon_3D_1, recon_3D_2, recon_3D_12, recon_3D_21],
                        dim=0),
                    dim=0,
                )
        closest_distance = torch.nanmean(
            torch.stack([closest_distance_real, closest_distance_virtual, closest_distance_1, closest_distance_2, closest_distance_12, closest_distance_21],
                        dim=0),
                    dim=0,
                )
        recon_pixels_1_virtual, recon_pixels_1_real, recon_pixels_2_virtual, recon_pixels_2_real = self.return_reprojected_pixels(recon_3D, R1, T1, R2, T2, camera2, image_width, image_height)
        
        return recon_3D, recon_pixels_1_virtual, recon_pixels_1_real, recon_pixels_2_virtual, recon_pixels_2_real, closest_distance


    def forward(self, pixels_virtual_two_cams, pixels_real_two_cams):
        self.prism = Prism(prism_size=self.prism_size, 
                        prism_center=self.prism_center, 
                        prism_rotation_6d=self.prism_rotation_6d,
                        refractive_index_glass=self.refractive_index_glass,
                        )
            
        #R_stereo_cam = get_rot_mat(
        #    self.stereo_camera_angles[0],
        #    self.stereo_camera_angles[1],
        #    self.stereo_camera_angles[2],
        #    )
        R1 = torch.eye(3, 3).to(device=pixels_virtual_two_cams.device, dtype=torch.float64)
        T1 = torch.zeros(3, 1).to(device=pixels_virtual_two_cams.device, dtype=torch.float64)
        R2 = self.stereo_camera_rotation_6d.matrix()
        T2 = self.T_stereo_cam
        camera2 = self.get_stereo_camera(self.principal_point_pixel_cam_1,
                                    self.focal_length_cam_1,
                                    R2,
                                    self.T_stereo_cam,
                                    r1=self.stereocam_r1,
                                    radial_dist_coeffs=self.radial_dist_coeffs_cam_1)
        distorted_real_pixels_cam_0 = torch.hstack((pixels_real_two_cams[:2,:], 
                                                    pixels_real_two_cams[2:4,:]))
        distorted_real_pixels_cam_1 = torch.hstack((pixels_real_two_cams[4:6,:], 
                                                    pixels_real_two_cams[6:,:]))
        distorted_virtual_pixels_cam_0 = torch.hstack((pixels_virtual_two_cams[:2,:], 
                                                    pixels_virtual_two_cams[2:4,:]))
        distorted_virtual_pixels_cam_1 = torch.hstack((pixels_virtual_two_cams[4:6,:], 
                                                    pixels_virtual_two_cams[6:,:]))
        
        undistorted_real_pixels_cam_0 = self.camera1.undistort_pixels_classical(distorted_real_pixels_cam_0, 
                                                                                self.radial_dist_coeffs_cam_0)
        undistorted_virtual_pixels_cam_0 = self.camera1.undistort_pixels_classical(distorted_virtual_pixels_cam_0,
                                                                                self.radial_dist_coeffs_cam_0)
        undistorted_real_pixels_cam_1 = camera2.undistort_pixels_classical(distorted_real_pixels_cam_1,
                                                                            self.radial_dist_coeffs_cam_1)
        undistorted_virtual_pixels_cam_1 = camera2.undistort_pixels_classical(distorted_virtual_pixels_cam_1,
                                                                            self.radial_dist_coeffs_cam_1)

        cam_1_ray_real = self.camera1(undistorted_real_pixels_cam_0)
        cam_2_ray_real = camera2(undistorted_real_pixels_cam_1)
        recon_3D_real, closest_distance_real = closest_point(cam_1_ray_real, cam_2_ray_real)
        
        cam_1_ray_virtual = self.camera1(undistorted_virtual_pixels_cam_0)
        cam_2_ray_virtual = camera2(undistorted_virtual_pixels_cam_1)
        _, _, cam_1_ray_virtual, intersection_penalty_1 = self.prism(cam_1_ray_virtual)
        _, _, cam_2_ray_virtual, intersection_penalty_2 = self.prism(cam_2_ray_virtual)        
        recon_3D_virtual, closest_distance_virtual = closest_point(cam_1_ray_virtual, cam_2_ray_virtual)        
        recon_3D_1, closest_distance_1 = closest_point(cam_1_ray_virtual, cam_1_ray_real)
        recon_3D_2, closest_distance_2 = closest_point(cam_2_ray_virtual, cam_2_ray_real)
        recon_3D_12, closest_distance_12 = closest_point(cam_1_ray_virtual, cam_2_ray_real)
        recon_3D_21, closest_distance_21 = closest_point(cam_2_ray_virtual, cam_1_ray_real)
        
        #recon_3D = (recon_3D_real + recon_3D_1 + recon_3D_2 + recon_3D_12 + recon_3D_21 + recon_3D_virtual) / 6
        recon_3D = torch.nanmean(
            torch.stack([recon_3D_real, recon_3D_virtual, recon_3D_1, recon_3D_2, recon_3D_12, recon_3D_21], 
                        dim=0),
            dim=0
        )
        num_points = recon_3D.shape[1] // 2
        #closest_distance = (closest_distance_real + closest_distance_virtual + closest_distance_1 + closest_distance_2 + closest_distance_12 + closest_distance_21) / 6
        closest_distance = torch.nanmean(
            torch.stack([closest_distance_real, closest_distance_virtual, closest_distance_1, closest_distance_2, closest_distance_12, closest_distance_21], 
                        dim=0),
            dim=0
        )
        closest_distance = closest_distance[:num_points]
        
        pairwise_distance = euclidean_distance(
            recon_3D[:, :num_points],
            recon_3D[:, num_points:]
        )        
        recon_pixels_1_undistorted = self.camera1.reproject(recon_3D, R1, T1)
        recon_pixels_1 = self.camera1.distort_pixels_classical(recon_pixels_1_undistorted, self.radial_dist_coeffs_cam_0)
        recon_pixels_2_undistorted = camera2.reproject(recon_3D, R2, T2)
        recon_pixels_2 = camera2.distort_pixels_classical(recon_pixels_2_undistorted, self.radial_dist_coeffs_cam_1)
      
        distortion_penalty_cam_0 = self.camera1.calculate_distortion_penalty(recon_pixels_1,
                                                                             self.radial_dist_coeffs_cam_0)
        distortion_penalty_cam_1 = camera2.calculate_distortion_penalty(recon_pixels_2,
                                                                        self.radial_dist_coeffs_cam_1)
        
        recon_pixels_0_from_virtual_undistorted = self.camera1.reproject(recon_3D_virtual, R1, T1)
        recon_pixels_0_from_virtual = self.camera1.distort_pixels_classical(recon_pixels_0_from_virtual_undistorted, self.radial_dist_coeffs_cam_0)
        recon_pixels_1_from_virtual_undistorted = camera2.reproject(recon_3D_virtual, R2, T2)
        recon_pixels_1_from_virtual = camera2.distort_pixels_classical(recon_pixels_1_from_virtual_undistorted, self.radial_dist_coeffs_cam_1)

        prob_virtual_reprojection = torch.rand(1)
        if prob_virtual_reprojection < self.virtual_proj_prob_thresh:
            recon_pixels_virtual = self.pseudo_reprojection_to_virtual_view(
                recon_3D, 
                image_width=[1920,1920], 
                image_height=[1200, 1200], 
                cam_label_projection='both')
            recon_pixels_1_to_virtual_undistorted, recon_pixels_2_to_virtual_undistorted = recon_pixels_virtual[:2,:], recon_pixels_virtual[2:,:]
            recon_pixels_1_to_virtual = self.camera1.distort_pixels_classical(recon_pixels_1_to_virtual_undistorted, self.radial_dist_coeffs_cam_0)
            recon_pixels_2_to_virtual = camera2.distort_pixels_classical(recon_pixels_2_to_virtual_undistorted, self.radial_dist_coeffs_cam_1)

        #distortion_penalty_cam_0 = self.camera1.calculate_distortion_penalty(recon_pixels_1, 
        #                                                                     self.radial_dist_coeffs_cam_0)
        #distortion_penalty_cam_1 = camera2.calculate_distortion_penalty(recon_pixels_2,
        #                                                                self.radial_dist_coeffs_cam_1)
        
        output = {}
        output['recon_3D'] = recon_3D
        output['recon_3D_real'] = recon_3D_real
        output['recon_3D_virtual'] = recon_3D_virtual
        output['recon_3D_12'] = recon_3D_12
        output['recon_3D_21'] = recon_3D_21
        output['recon_3D_1'] = recon_3D_1
        output['recon_3D_2'] = recon_3D_2
        output['closest_distance'] = closest_distance
        output['closest_distance_real'] = closest_distance_real
        output['closest_distance_virtual'] = closest_distance_virtual
        output['closest_distance_1'] = closest_distance_1
        output['closest_distance_2'] = closest_distance_2
        output['closest_distance_12'] = closest_distance_12
        output['closest_distance_21'] = closest_distance_21
        output['recon_pixels_1'] = recon_pixels_1
        output['recon_pixels_2'] = recon_pixels_2
        output['recon_pixels_0_virtual'] = recon_pixels_0_from_virtual
        output['recon_pixels_1_virtual'] = recon_pixels_1_from_virtual
        output['recon_pixels_1_to_virtual'] = recon_pixels_1_to_virtual if prob_virtual_reprojection < self.virtual_proj_prob_thresh else None
        output['recon_pixels_2_to_virtual'] = recon_pixels_2_to_virtual if prob_virtual_reprojection < self.virtual_proj_prob_thresh else None
        output['intersection_penalty_1'] = intersection_penalty_1
        output['intersection_penalty_2'] = intersection_penalty_2
        output['distortion_penalty_cam_0'] = distortion_penalty_cam_0
        output['distortion_penalty_cam_1'] = distortion_penalty_cam_1
        output['pairwise_distance'] = pairwise_distance    
        
        return output
    
    def pseudo_reprojection_to_virtual_view(self, point, image_width, image_height=[1200, 1200], cam_label_projection='primary'):
        """
        Reprojection in virtual view (cannot be done using inverse ray tracing)
        cam_label_projection (str): ["primary", "secondary", "both"]
        point: 3D point in the world coordinates
        Returns the closest virtual pixel to the point in the virtual camera.
        """    
        if len(point.shape) == 2:
            point = point.unsqueeze(1)
        num_points = point.shape[-1]
        device = point.device
        resolution_for_virtual_projection = 0.01 # pixel accuracy needed to get reprojection in the virtual camera
        grid_factor = 4  
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

        center = torch.zeros(2 * num_views, num_points).to(dtype=torch.float64, device=device) 
        with torch.no_grad():
            for i in range(num_views):
                center[i*2:(i+1)*2, :] = torch.tensor([width[i] / 2, height[i] / 2]).to(dtype=torch.float64, device=device)[:, None]
            
            x = torch.linspace(-0.5 + 1/grid_factor/2, 0.5 - 1/grid_factor/2, grid_factor)
            y = torch.linspace(-0.5 + 1/grid_factor/2, 0.5 - 1/grid_factor/2, grid_factor) 
            [xx, yy] = torch.meshgrid(x, y)
            xx = xx.flatten().unsqueeze(-1).T
            yy = yy.flatten().unsqueeze(-1).T
                    
            # Reshape for broadcasting
            xx = xx.unsqueeze(2) # (1, grid_factor, 1)
            yy = yy.unsqueeze(2) # (1, grid_factor, 1)
            for _ in range(num_iterations):            
                center = center.unsqueeze(1) # Reshape for broadcasting (4, 1, num_points)
                grid_points_x = xx * width[0] + center[0,:]
                grid_points_x = grid_points_x.reshape(1, -1)
                grid_points_y = yy * height[0] + center[1,:]
                grid_points_y = grid_points_y.reshape(1, -1)
                virtual_pixels_grid = torch.vstack(
                        (grid_points_x, 
                            grid_points_y)
                        )
                #virtual_pixels_grid = torch.vstack((xx * width[0] + center[0,:], yy * height[0] + center[1,:]))    
                for view in range(1, num_views):
                    grid_points_x = xx * width[view] + center[2*view,:]
                    grid_points_x = grid_points_x.reshape(1, -1)
                    grid_points_y = yy * height[view] + center[2*view+1,:]
                    grid_points_y = grid_points_y.reshape(1, -1)

                    virtual_pixels_grid = torch.vstack(
                                                        (virtual_pixels_grid, 
                                                            torch.vstack(
                                                                (grid_points_x, 
                                                                    grid_points_y)
                                                                    )
                                                                )
                                                            ) 
                rays_dict = self.pass_through_virtual_cam(virtual_pixels_grid, cam_label=cam_label_projection)
                if cam_label_projection == 'primary':
                    ray = rays_dict["cam_1_ray_virtual"]
                elif cam_label_projection == 'secondary':
                    ray = rays_dict["cam_2_ray_virtual"]
                
                
                if not(cam_label_projection == 'both'):
                    #distance = ray.distance_to_point(point) # (num_rays, num_points)
                    ray_direction = ray.direction.view(3, grid_factor, num_points)
                    ray_origin = ray.origin.view(3, grid_factor, num_points)
                    distance = self.vectorized_distance_between_rays_and_points(ray_origin, ray_direction, point)
                    min_arg = torch.argmin(distance, dim=0)
                    virtual_pixels_grid = virtual_pixels_grid.reshape(4, grid_factor**2, num_points)
                    center = virtual_pixels_grid[:, min_arg, torch.arange(num_points)]

                    if len(center.shape) == 1:
                        center = center[:, None]
                    width = [width_el / grid_factor for width_el in width]
                    height = [height_el / grid_factor for height_el in height]
                    
                elif cam_label_projection == 'both':                
                    ray1, ray2 = rays_dict["cam_1_ray_virtual"], rays_dict["cam_2_ray_virtual"]                
                    ray_direction = ray1.direction.reshape(3, grid_factor**2, num_points)
                    ray_origin = ray1.origin.reshape(3, grid_factor**2, num_points)
                    #distance1 = ray1.distance_to_point(point) # (num_rays, num_points)
                    distance1 = self.vectorized_distance_between_rays_and_points(ray_origin, ray_direction, point)
                    min_arg1 = torch.argmin(distance1, dim=0)

                    ray_direction = ray2.direction.reshape(3, grid_factor**2, num_points)
                    ray_origin = ray2.origin.reshape(3, grid_factor**2, num_points)
                    #distance2 = ray2.distance_to_point(point) # (num_rays, num_points)
                    distance2 = self.vectorized_distance_between_rays_and_points(ray_origin, ray_direction, point)
                    min_arg2 = torch.argmin(distance2, dim=0)
                    virtual_pixels_grid = virtual_pixels_grid.reshape(4, grid_factor**2, num_points) # (3, K, N)
                    center1 = virtual_pixels_grid[:2, min_arg1, torch.arange(num_points)]
                    if len(center1.shape) == 1:
                        center1 = center1[:, None]
                    center2 = virtual_pixels_grid[2:, min_arg2, torch.arange(num_points)]
                    if len(center2.shape) == 1:
                        center2 = center2[:, None]
                    center = torch.vstack((center1, center2))
                    width = [width_el / grid_factor for width_el in width]
                    height = [height_el / grid_factor for height_el in height]
        
        return center

    def vectorized_distance_between_rays_and_points(self, ray_origin, ray_direction, point):
        """
        This is used in reprojecting rays in virtual views
        ray_origin: (tensor) origin of the ray (3, K, N)
        ray_direction: (tensor) direction of the ray (3, K, N)
        pts: (tensor) points from which to compute distance to the subset (3,K) of rays (1,1,N)
        """
        point_shifted = point - ray_origin # (3, K, N)
        distance = torch.cross(ray_direction, point_shifted, dim=0)  # (3, K, N)
        distance = torch.linalg.norm(distance, dim=0) # (K, N)
        return distance

    def pass_through_virtual_cam(self, pixels_virtual_two_cams, cam_label='both'):
        # Input pixels aren't provided in pairs
        self.prism = Prism(prism_size=self.prism_size,
                        prism_center=self.prism_center,
                        prism_rotation_6d=self.prism_rotation_6d,
                        refractive_index_glass=self.refractive_index_glass,
                        )
        
        if cam_label == 'primary':
            distorted_virtual_pixels_cam_0 = pixels_virtual_two_cams
        elif cam_label == 'secondary':
            distorted_virtual_pixels_cam_1 = pixels_virtual_two_cams
        elif cam_label == 'both':
            distorted_virtual_pixels_cam_0 = pixels_virtual_two_cams[:2,:]
            distorted_virtual_pixels_cam_1 = pixels_virtual_two_cams[2:4,:]

        output = {}
        output['cam_1_ray_virtual'] = None
        output['cam_2_ray_virtual'] = None

        if cam_label == 'primary' or cam_label == 'both':
            undistorted_virtual_pixels_cam_0 = self.camera1.undistort_pixels_classical(distorted_virtual_pixels_cam_0,
                                                                                self.radial_dist_coeffs_cam_0)
            cam_1_ray_virtual = self.camera1(undistorted_virtual_pixels_cam_0)
            _, _, cam_1_ray_virtual, _ = self.prism(cam_1_ray_virtual)
            output['cam_1_ray_virtual'] = cam_1_ray_virtual
            
        if cam_label == 'secondary' or cam_label == 'both':
            """R_stereo_cam = get_rot_mat(
            self.stereo_camera_angles[0],
            self.stereo_camera_angles[1],
            self.stereo_camera_angles[2],
            )"""
            R_stereo_cam = self.stereo_camera_rotation_6d.matrix()
            camera2 = self.get_stereo_camera(self.principal_point_pixel_cam_1,
                                    self.focal_length_cam_1,
                                    R_stereo_cam,
                                    self.T_stereo_cam,
                                    r1=self.stereocam_r1,
                                    radial_dist_coeffs=self.radial_dist_coeffs_cam_1)
            undistorted_virtual_pixels_cam_1 = camera2.undistort_pixels_classical(distorted_virtual_pixels_cam_1,
                                                                            self.radial_dist_coeffs_cam_1)
            cam_2_ray_virtual = camera2(undistorted_virtual_pixels_cam_1)
            _, _, cam_2_ray_virtual, _ = self.prism(cam_2_ray_virtual)
            output['cam_2_ray_virtual'] = cam_2_ray_virtual

        return output


    def visualize(self, pixels_virtual_two_cams, color_labels=None):
        num_samples = 10

        undistorted_virtual_pixels_cam_0 = torch.hstack(
            (pixels_virtual_two_cams[:2, :].clone(), pixels_virtual_two_cams[2:4, :].clone())
        )
        undistorted_virtual_pixels_cam_1 = torch.hstack(
            (pixels_virtual_two_cams[4:6, :].clone(),
             pixels_virtual_two_cams[6:,:])
        )

        test_idx = torch.randperm(undistorted_virtual_pixels_cam_0.shape[1])[:num_samples]

        ray = self.camera1.initialize_ray(undistorted_virtual_pixels_cam_0[:, test_idx])
        ray.t *= 100
        fig, ax = self.camera1.visualize()
        fig, ax = ray.visualize(fig, ax, color_labels)
        prism_ray11, prism_ray12, emergent_ray1, intersection_penalty_1 = self.prism(ray)
        fig, ax = prism_ray11.visualize(fig, ax, color_labels)
        fig, ax = prism_ray12.visualize(fig, ax, color_labels)
        
        fig, ax = self.prism.visualize_prism(fig, ax)
        emergent_ray1.t *= 25
        fig, ax = emergent_ray1.visualize(fig, ax, color_labels)
        R_stereo_cam = self.stereo_camera_rotation_6d.matrix()
        camera2 = self.get_stereo_camera(self.principal_point_pixel_cam_1,
                                self.focal_length_cam_1,
                                R_stereo_cam,
                                self.T_stereo_cam,
                                self.stereocam_r1,
                                radial_dist_coeffs=self.radial_dist_coeffs_cam_1)
        ray = camera2.initialize_ray(undistorted_virtual_pixels_cam_1[:, test_idx])
        ray.t *= 100
        fig, ax = camera2.visualize(fig, ax)
        fig, ax = ray.visualize(fig, ax, color_labels)
        prism_ray21, prism_ray22, emergent_ray2, intersection_penalty_2 = self.prism(ray)
        fig, ax = self.prism.visualize_prism(fig, ax)
        fig, ax = prism_ray21.visualize(fig, ax, color_labels)
        fig, ax = prism_ray22.visualize(fig, ax, color_labels)
        emergent_ray2.t *= 25
        emergent_ray2.visualize(fig, ax, color_labels)
        ax.set_aspect('equal', adjustable='datalim')   

class Arena_reprojection_loss_single_camera_prism(nn.Module):
    def __init__(self, 
    principal_point_pixel_cam_0, 
    principal_point_pixel_cam_1, 
    focal_length_cam_0, 
    focal_length_cam_1, 
    R_stereo_cam, 
    T_stereo_cam, 
    prism_distance=None,
    prism_angles=None,
    prism_center=None):
        super(Arena_reprojection_loss_single_camera_prism, self).__init__()

        # Camera initialization      
        principal_point_pixel_cam_0 = nn.Parameter(
            torch.tensor(principal_point_pixel_cam_0, dtype=torch.float64).reshape(2,1),
            requires_grad=True,
        )  

        principal_point_pixel_cam_1 = nn.Parameter(
            torch.tensor(principal_point_pixel_cam_1, dtype=torch.float64).reshape(2,1),
            requires_grad=False,
        )  

        focal_length_cam_0 = nn.Parameter(
            torch.tensor(focal_length_cam_0, dtype=torch.float64),
            requires_grad=True,
        )

        focal_length_cam_1 = nn.Parameter(
            torch.tensor(focal_length_cam_1, dtype=torch.float64),
            requires_grad=False,
        )

        self.r1 = nn.Parameter(
            torch.tensor(1e-6, dtype=torch.float64),
            requires_grad=True,
        )
        self.stereocam_r1 = nn.Parameter(
            torch.tensor(1e-6, dtype=torch.float64),
            requires_grad=True,
        )

        self.camera1 = EfficientCamera(
            principal_point_pixel=principal_point_pixel_cam_0,
            focal_length_pixels=focal_length_cam_0,
            r1=self.r1)

        self.principal_point_pixel_cam_1 = principal_point_pixel_cam_1
        self.focal_length_cam_1 = focal_length_cam_1
        self.R_stereo_cam = R_stereo_cam
        self.T_stereo_cam = nn.Parameter(T_stereo_cam, requires_grad=False)
        stereo_alpha, stereo_beta, stereo_gamma = self.get_stereo_camera_angles(R_stereo_cam)
        self.stereo_camera_angles = nn.Parameter(
                                                torch.tensor([stereo_alpha, stereo_beta, stereo_gamma], 
                                                    dtype=torch.float64, 
                                                    requires_grad=False
                                                    )
                                                )


        # Prism initialization
        prism_angles = nn.Parameter(prism_angles, requires_grad=True)
        refractive_index_glass = torch.tensor(1.5, dtype=torch.float64)
        refractive_index_glass = nn.Parameter(refractive_index_glass, requires_grad=True)
        if prism_center is None:
            prism_center = self.camera1.aperture.clone() + self.camera1.axes[:,0].unsqueeze(-1).clone() * prism_distance.clone()
            prism_center[0] = -5.

        prism_center = nn.Parameter(prism_center, requires_grad=True)
        prism_size = nn.Parameter(torch.tensor([20.,20.,20.],dtype=torch.float64), requires_grad=True)

        self.prism = Prism(prism_size=prism_size, 
                        prism_center=prism_center, 
                        prism_angles=prism_angles,
                        refractive_index_glass=refractive_index_glass,
                        )
        

    def get_stereo_camera_angles(self, 
                                 R_stereo_cam):
        axes = torch.eye(3,3).to(device=R_stereo_cam.device, dtype=torch.float64)
        axes = torch.mm(R_stereo_cam, axes)
        plane = Plane(axes=axes)
        return plane.alpha, plane.beta, plane.gamma
    

    def get_stereo_camera(self, 
                     principal_point_pixel_cam_1,
                     focal_length_cam_1,
                     R,
                     T,
                     r1):
        camera2 = EfficientCamera(
            principal_point_pixel=principal_point_pixel_cam_1, 
            focal_length_pixels=focal_length_cam_1, 
            r1=r1)
        camera2.update_camera_pose(R, T)
        return camera2


    def forward(self, pixels_virtual_two_cams, pixels_real_two_cams):
        R_stereo_cam = get_rot_mat(
            self.stereo_camera_angles[0],
            self.stereo_camera_angles[1],
            self.stereo_camera_angles[2],
            )
        R1 = torch.eye(3, 3).to(device=pixels_virtual_two_cams.device, dtype=torch.float64)
        T1 = torch.zeros(3, 1).to(device=pixels_virtual_two_cams.device, dtype=torch.float64)
        
        distorted_virtual_pixels_cam_0 = pixels_virtual_two_cams[:2,:]
        undistorted_virtual_pixels_cam_0 = self.camera1.undistort_pixels(distorted_virtual_pixels_cam_0)
        distorted_real_pixels_cam_0 = pixels_real_two_cams[:2,:]
        #undistorted_real_pixels_cam_0 = distorted_real_pixels_cam_0
        undistorted_real_pixels_cam_0 = self.camera1.undistort_pixels(distorted_real_pixels_cam_0)
        cam_1_ray_real = self.camera1(undistorted_real_pixels_cam_0)
        cam_1_ray_virtual = self.camera1(undistorted_virtual_pixels_cam_0)
        _, _, emergent_1_ray_virtual, intersection_penalty_1 = self.prism(cam_1_ray_virtual)
        recon_3D, closest_distance = closest_point(cam_1_ray_real, emergent_1_ray_virtual)
        recon_undistorted_real_pixels_cam_0 = self.camera1.reproject(recon_3D, R1, T1)
        recon_distorted_pixels_cam_0 = self.camera1.distort_pixels(recon_undistorted_real_pixels_cam_0)
        #recon_distorted_pixels_cam_0 = recon_undistorted_real_pixels_cam_0
        distortion_penalty_cam_0 = self.camera1.calculate_distortion_penalty(distorted_virtual_pixels_cam_0) + self.camera1.calculate_distortion_penalty(distorted_real_pixels_cam_0)
        output = {}
        output['recon_3D'] = recon_3D
        output['closest_distance'] = closest_distance
        output['recon_distorted_pixels_cam_0'] = recon_distorted_pixels_cam_0
        output['intersection_penalty_1'] = intersection_penalty_1
        output['distortion_penalty_cam_0'] = distortion_penalty_cam_0
        return output


    def visualize(self, pixels_virtual_two_cams, color_labels=None):
        num_samples = 10
        undistorted_virtual_pixels_cam_0 = pixels_virtual_two_cams[:2, :].clone()
        undistorted_virtual_pixels_cam_1 = pixels_virtual_two_cams[2:, :].clone()
        test_idx = torch.randperm(undistorted_virtual_pixels_cam_0.shape[1])[:num_samples]

        ray = self.camera1.initialize_ray(undistorted_virtual_pixels_cam_0[:, test_idx])
        ray.t *= 100
        fig, ax = self.camera1.visualize()
        fig, ax = ray.visualize(fig, ax, color_labels)
        prism_ray11, prism_ray12, emergent_ray1, intersection_penalty_1 = self.prism(ray)
        fig, ax = prism_ray11.visualize(fig, ax, color_labels)
        fig, ax = prism_ray12.visualize(fig, ax, color_labels)
        
        fig, ax = self.prism.visualize_prism(fig, ax)
        emergent_ray1.t *= 25
        fig, ax = emergent_ray1.visualize(fig, ax, color_labels)
        ax.set_aspect('equal', adjustable='datalim')   



# Single camera grid distances
class Arena_single_camera_prism_grid_distance(nn.Module):
    def __init__(self, 
    principal_point_pixel_cam_0=None, 
    principal_point_pixel_cam_1=None, 
    focal_length_cam_0=None, 
    focal_length_cam_1=None, 
    R_stereo_cam=None, 
    T_stereo_cam=None, 
    prism_distance=None,
    prism_angles=None,
    prism_center=None,
    prism_size=None):
        super(Arena_single_camera_prism_grid_distance, self).__init__()

        # Camera initialization      
        principal_point_pixel_cam_0 = nn.Parameter(
            torch.tensor(principal_point_pixel_cam_0, dtype=torch.float64).reshape(2,1),
            requires_grad=True,
        )  

        principal_point_pixel_cam_1 = nn.Parameter(
            torch.tensor(principal_point_pixel_cam_1, dtype=torch.float64).reshape(2,1),
            requires_grad=True,
        )  

        focal_length_cam_0 = nn.Parameter(
            torch.tensor(focal_length_cam_0, dtype=torch.float64),
            requires_grad=True,
        )

        focal_length_cam_1 = nn.Parameter(
            torch.tensor(focal_length_cam_1, dtype=torch.float64),
            requires_grad=True,
        )

        self.r1 = nn.Parameter(
            torch.tensor(1e-6, dtype=torch.float64),
            requires_grad=True,
        )
        self.stereocam_r1 = nn.Parameter(
            torch.tensor(1e-6, dtype=torch.float64),
            requires_grad=True,
        )

        self.radial_dist_coeffs = nn.Parameter(torch.tensor([0.,0.,0.]).unsqueeze(-1).to(torch.float64),
                                               requires_grad=True)

        self.camera1 = EfficientCamera(
            principal_point_pixel=principal_point_pixel_cam_0,
            focal_length_pixels=focal_length_cam_0,
            r1=self.r1,
            radial_dist_coeffs=self.radial_dist_coeffs)

        self.principal_point_pixel_cam_1 = principal_point_pixel_cam_1
        self.focal_length_cam_1 = focal_length_cam_1
        self.R_stereo_cam = R_stereo_cam
        self.T_stereo_cam = nn.Parameter(T_stereo_cam, requires_grad=True)

        # Prism initialization
        prism_angles = nn.Parameter(prism_angles, requires_grad=True)
        refractive_index_glass = torch.tensor(1.51, dtype=torch.float64)
        refractive_index_glass = nn.Parameter(refractive_index_glass, requires_grad=True)
        if prism_center is None:
            prism_center = self.camera1.aperture.clone() + self.camera1.axes[:,0].unsqueeze(-1).clone() * prism_distance.clone()
            prism_center[0] = -5.

        prism_center = nn.Parameter(prism_center, requires_grad=True)
        prism_size = nn.Parameter(torch.tensor([prism_size, prism_size, prism_size],dtype=torch.float64), requires_grad=True)

        self.prism = Prism(prism_size=prism_size, 
                        prism_center=prism_center, 
                        prism_angles=prism_angles,
                        refractive_index_glass=refractive_index_glass,
                        )
        
        
    def forward(self, pixels_real_two_cams, pixels_virtual_two_cams):
        R1 = torch.eye(3, 3).to(torch.float64)
        T1 = torch.zeros(3, 1).to(torch.float64)
        distorted_real_pixels_cam_0 = torch.hstack((pixels_real_two_cams[:2,:], 
                                                    pixels_real_two_cams[2:4,:]))
        distorted_virtual_pixels_cam_0 = torch.hstack((pixels_virtual_two_cams[:2,:], 
                                                    pixels_virtual_two_cams[2:4,:])) 
        undistorted_real_pixels_cam_0 = self.camera1.undistort_pixels_classical(distorted_real_pixels_cam_0, 
                                                                                self.radial_dist_coeffs)
        undistorted_virtual_pixels_cam_0 = self.camera1.undistort_pixels_classical(distorted_virtual_pixels_cam_0,
                                                                                self.radial_dist_coeffs)
        
        #undistorted_real_pixels_cam_0 = self.camera1.undistort_pixels_MLP(distorted_real_pixels_cam_0)
        #undistorted_virtual_pixels_cam_0 = self.camera1.undistort_pixels_MLP(distorted_virtual_pixels_cam_0)
        #undistorted_real_pixels_cam_0 = distorted_real_pixels_cam_0        
        #undistorted_virtual_pixels_cam_0 = distorted_virtual_pixels_cam_0
        cam_1_ray_real = self.camera1(undistorted_real_pixels_cam_0)
        cam_1_ray_virtual = self.camera1(undistorted_virtual_pixels_cam_0)
        _, _, emergent_1_ray_virtual, intersection_penalty = self.prism(cam_1_ray_virtual)
        
        recon_3D, closest_distance = closest_point(cam_1_ray_real, emergent_1_ray_virtual)
        num_points = recon_3D.shape[1] // 2
        pairwise_distance = euclidean_distance(
            recon_3D[:, :num_points],
            recon_3D[:, num_points:]
        )
        recon_undistorted_real_pixels_cam_0 = self.camera1.reproject(recon_3D, R1, T1)
        recon_distorted_real_pixels_cam_0 = self.camera1.distort_pixels_classical(recon_undistorted_real_pixels_cam_0,
                                                                                  self.radial_dist_coeffs.data)
        #recon_distorted_real_pixels_cam_0 = self.camera1.distort_pixels_MLP(recon_undistorted_real_pixels_cam_0)
        #recon_distorted_real_pixels_cam_0 = recon_undistorted_real_pixels_cam_0
        #recon_distorted_real_pixels_cam_1 = recon_undistorted_real_pixels_cam_1
        #recon_3D = recon_3D[:, :num_points]
        #recon_distorted_real_pixels_cam_0 = recon_distorted_real_pixels_cam_0[:, :num_points]
        #recon_distorted_real_pixels_cam_1 = recon_distorted_real_pixels_cam_1[:, :num_points]
        recon_loss_real = (euclidean_distance(
            recon_distorted_real_pixels_cam_0,
            distorted_real_pixels_cam_0)) / 2
        
        # Distortion penalty
        distorted_real_pixels_cam_0_ = self.camera1.distort_pixels_classical(undistorted_real_pixels_cam_0,
                                                                             self.radial_dist_coeffs.data)
        distorted_virtual_pixels_cam_0_ = self.camera1.distort_pixels_classical(undistorted_virtual_pixels_cam_0,
                                                                      self.radial_dist_coeffs.data)
        
        #distorted_real_pixels_cam_0_ = self.camera1.distort_pixels_MLP(undistorted_real_pixels_cam_0)
        #distorted_virtual_pixels_cam_0_ = self.camera1.distort_pixels_MLP(undistorted_virtual_pixels_cam_0)
        distortion_penalty = (euclidean_distance(
            distorted_real_pixels_cam_0_, distorted_real_pixels_cam_0
        ) + euclidean_distance(
            distorted_virtual_pixels_cam_0_, distorted_virtual_pixels_cam_0
        )) / 2
        return recon_3D, closest_distance, recon_distorted_real_pixels_cam_0, recon_loss_real, pairwise_distance, intersection_penalty, distortion_penalty


    def visualize(self, pixels_real_two_cams, pixels_virtual_two_cams, color_labels=None):
            num_samples = 10
            undistorted_virtual_pixels_cam_0 = pixels_virtual_two_cams[:2, :].clone()
            undistorted_real_pixels_cam_0 = pixels_real_two_cams[:2, :].clone()
            test_idx = torch.randperm(undistorted_real_pixels_cam_0.shape[1])[:num_samples]
            ray = self.camera1.initialize_ray(undistorted_real_pixels_cam_0[:, test_idx])
            ray.t *= 100
            fig, ax = self.camera1.visualize()
            fig, ax = ray.visualize(fig, ax, color_labels)
            ray = self.camera1.initialize_ray(undistorted_virtual_pixels_cam_0[:, test_idx])
            fig, ax = ray.visualize(fig, ax, color_labels)
            prism_ray11, prism_ray12, ray, intersection_penalty_1 = self.prism(ray)
            ray.t *= 100
            fig, ax = self.prism.visualize_prism(fig, ax)
            fig, ax = ray.visualize(fig, ax, color_labels)        
            ax.set_aspect('equal', adjustable='datalim')  
            return fig, ax


# Single camera grid distances
class Arena_single_camera_prism_grid_image(nn.Module):
    def __init__(self, 
    principal_point_pixel_cam_0=None, 
    principal_point_pixel_cam_1=None, 
    focal_length_cam_0=None, 
    focal_length_cam_1=None, 
    R_stereo_cam=None, 
    T_stereo_cam=None, 
    prism_distance=None,
    prism_angles=None,
    prism_center=None):
        super(Arena_single_camera_prism_grid_image, self).__init__()

        # Camera initialization      
        principal_point_pixel_cam_0 = nn.Parameter(
            torch.tensor(principal_point_pixel_cam_0, dtype=torch.float64).reshape(2,1),
            requires_grad=True,
        )  

        principal_point_pixel_cam_1 = nn.Parameter(
            torch.tensor(principal_point_pixel_cam_1, dtype=torch.float64).reshape(2,1),
            requires_grad=True,
        )  

        focal_length_cam_0 = nn.Parameter(
            torch.tensor(focal_length_cam_0, dtype=torch.float64),
            requires_grad=True,
        )

        focal_length_cam_1 = nn.Parameter(
            torch.tensor(focal_length_cam_1, dtype=torch.float64),
            requires_grad=True,
        )

        self.r1 = nn.Parameter(
            torch.tensor(1e-6, dtype=torch.float64),
            requires_grad=True,
        )
        self.stereocam_r1 = nn.Parameter(
            torch.tensor(1e-6, dtype=torch.float64),
            requires_grad=True,
        )

        self.radial_dist_coeffs = nn.Parameter(torch.tensor([0.,0.,0.]).unsqueeze(-1).to(torch.float64),
                                               requires_grad=True)

        self.camera1 = EfficientCamera(
            principal_point_pixel=principal_point_pixel_cam_0,
            focal_length_pixels=focal_length_cam_0,
            r1=self.r1,
            radial_dist_coeffs=self.radial_dist_coeffs)

        self.principal_point_pixel_cam_1 = principal_point_pixel_cam_1
        self.focal_length_cam_1 = focal_length_cam_1
        self.R_stereo_cam = R_stereo_cam
        self.T_stereo_cam = nn.Parameter(T_stereo_cam, requires_grad=True)

        # Prism initialization
        prism_angles = nn.Parameter(prism_angles, requires_grad=True)
        refractive_index_glass = torch.tensor(1.5, dtype=torch.float64)
        refractive_index_glass = nn.Parameter(refractive_index_glass, requires_grad=True)
        if prism_center is None:
            prism_center = self.camera1.aperture.clone() + self.camera1.axes[:,0].unsqueeze(-1).clone() * prism_distance.clone()
            prism_center[0] = -5.

        prism_center = nn.Parameter(prism_center, requires_grad=True)
        prism_size = nn.Parameter(torch.tensor([20.,20.,20.],dtype=torch.float64), requires_grad=True)

        self.prism = Prism(prism_size=prism_size, 
                        prism_center=prism_center, 
                        prism_angles=prism_angles,
                        refractive_index_glass=refractive_index_glass,
                        )
        
        
    def forward(self, pixels_real_two_cams, pixels_virtual_two_cams):
        R1 = torch.eye(3, 3).to(torch.float64)
        T1 = torch.zeros(3, 1).to(torch.float64)
        distorted_real_pixels_cam_0 = pixels_real_two_cams[:2,:]
        distorted_virtual_pixels_cam_0 = pixels_virtual_two_cams[:2,:]
        undistorted_real_pixels_cam_0 = self.camera1.undistort_pixels_classical(distorted_real_pixels_cam_0, 
                                                                                self.radial_dist_coeffs)
        undistorted_virtual_pixels_cam_0 = self.camera1.undistort_pixels_classical(distorted_virtual_pixels_cam_0,
                                                                                self.radial_dist_coeffs)
        
        #undistorted_real_pixels_cam_0 = self.camera1.undistort_pixels_MLP(distorted_real_pixels_cam_0)
        #undistorted_virtual_pixels_cam_0 = self.camera1.undistort_pixels_MLP(distorted_virtual_pixels_cam_0)
        #undistorted_real_pixels_cam_0 = distorted_real_pixels_cam_0        
        #undistorted_virtual_pixels_cam_0 = distorted_virtual_pixels_cam_0
        cam_1_ray_real = self.camera1(undistorted_real_pixels_cam_0)
        cam_1_ray_virtual = self.camera1(undistorted_virtual_pixels_cam_0)
        _, _, emergent_1_ray_virtual, intersection_penalty = self.prism(cam_1_ray_virtual)
        recon_3D, closest_distance = closest_point(cam_1_ray_real, emergent_1_ray_virtual)
        recon_undistorted_real_pixels_cam_0 = self.camera1.reproject(recon_3D, R1, T1)
        recon_distorted_real_pixels_cam_0 = self.camera1.distort_pixels_classical(recon_undistorted_real_pixels_cam_0,
                                                                                  self.radial_dist_coeffs)
        #recon_distorted_real_pixels_cam_0 = self.camera1.distort_pixels_MLP(recon_undistorted_real_pixels_cam_0)
        #recon_distorted_real_pixels_cam_0 = recon_undistorted_real_pixels_cam_0
        #recon_distorted_real_pixels_cam_1 = recon_undistorted_real_pixels_cam_1
        #recon_3D = recon_3D[:, :num_points]
        #recon_distorted_real_pixels_cam_0 = recon_distorted_real_pixels_cam_0[:, :num_points]
        #recon_distorted_real_pixels_cam_1 = recon_distorted_real_pixels_cam_1[:, :num_points]
        recon_loss_real = (euclidean_distance(
            recon_distorted_real_pixels_cam_0,
            distorted_real_pixels_cam_0))
        
        # Distortion penalty
        distorted_real_pixels_cam_0_ = self.camera1.distort_pixels_classical(undistorted_real_pixels_cam_0,
                                                                             self.radial_dist_coeffs.data)
        distorted_virtual_pixels_cam_0_ = self.camera1.distort_pixels_classical(undistorted_virtual_pixels_cam_0,
                                                                      self.radial_dist_coeffs.data)
        
        #distorted_real_pixels_cam_0_ = self.camera1.distort_pixels_MLP(undistorted_real_pixels_cam_0)
        #distorted_virtual_pixels_cam_0_ = self.camera1.distort_pixels_MLP(undistorted_virtual_pixels_cam_0)
        distortion_penalty = (euclidean_distance(
            distorted_real_pixels_cam_0_, distorted_real_pixels_cam_0
        ) + euclidean_distance(
            distorted_virtual_pixels_cam_0_, distorted_virtual_pixels_cam_0
        )) / 2
        return recon_3D, closest_distance, recon_distorted_real_pixels_cam_0, recon_loss_real, intersection_penalty, distortion_penalty


    def visualize(self, pixels_real_two_cams, pixels_virtual_two_cams, color_labels=None):
        num_samples = 10
        undistorted_virtual_pixels_cam_0 = pixels_virtual_two_cams[:2, :].clone()
        undistorted_real_pixels_cam_0 = pixels_real_two_cams[:2, :].clone()
        test_idx = torch.randperm(undistorted_real_pixels_cam_0.shape[1])[:num_samples]
        ray = self.camera1.initialize_ray(undistorted_real_pixels_cam_0[:, test_idx])
        ray.t *= 100
        fig, ax = self.camera1.visualize()
        fig, ax = ray.visualize(fig, ax, color_labels)
        ray = self.camera1.initialize_ray(undistorted_virtual_pixels_cam_0[:, test_idx])
        prism_ray11, prism_ray12, ray, intersection_penalty_1 = self.prism(ray)
        ray.t *= 100
        fig, ax = self.prism.visualize_prism(fig, ax)
        fig, ax = ray.visualize(fig, ax, color_labels)        
        ax.set_aspect('equal', adjustable='datalim')  

# Adhesion layer
class Arena_adhesion_layer(nn.Module):
    def __init__(self, 
    principal_point_pixel_cam_0, 
    principal_point_pixel_cam_1, 
    focal_length_cam_0, 
    focal_length_cam_1, 
    R, 
    T, 
    prism_distance=None,
    prism_angles=None,
    prism_center=None,
    adhesion_thickness_factor=None,
    adhesion_refractive_index=None):
        super(Arena_adhesion_layer, self).__init__()

        # Camera initialization        
        self.camera1 = EfficientCamera(
            principal_point_pixel=principal_point_pixel_cam_0, 
            focal_length_pixels=focal_length_cam_0)
        self.camera2 = EfficientCamera(
            principal_point_pixel=principal_point_pixel_cam_1, 
            focal_length_pixels=focal_length_cam_1)
        self.camera2.update_camera_pose(R, T)
        
        # Prism initialization
        prism_angles = nn.Parameter(prism_angles, requires_grad=True)
        refractive_index_glass = torch.tensor(1.5, dtype=torch.float32)
        refractive_index_glass = nn.Parameter(refractive_index_glass, requires_grad=True)
        if prism_center is None:
            prism_center = self.camera1.aperture.clone() + self.camera1.axes[:,0].unsqueeze(-1).clone() * prism_distance.clone()
            prism_center[0] = -5.

        prism_center = nn.Parameter(prism_center, requires_grad=True)
        prism_size = nn.Parameter(torch.tensor([20.,20.,20.],dtype=torch.float32), requires_grad=False)
        adhesion_thickness_factor = nn.Parameter(torch.tensor(20.), requires_grad=True)
        refractive_index_adhesion = nn.Parameter(torch.tensor(1.5), requires_grad=True)
        #prism_size = torch.tensor([20.,20.,20.], dtype=torch.float32)
        self.prism = Prism(prism_size=prism_size, 
                        prism_center=prism_center, 
                        prism_angles=prism_angles,
                        refractive_index_glass=refractive_index_glass,
                        adhesion_thickness_factor=adhesion_thickness_factor,
                        refractive_index_adhesion=refractive_index_adhesion,
                        )
        #freeze_camera_parameters(self.camera1)
        #freeze_camera_parameters(self.camera2)
        #freeze_individual_planes(self.prism)
    

    def forward(self, pixels_virtual_two_cams):
        undistorted_virtual_pixels_cam_0 = pixels_virtual_two_cams[:2, :]
        undistorted_virtual_pixels_cam_1 = pixels_virtual_two_cams[2:, :]
        #cam_1_ray = self.camera1.initialize_ray(undistorted_virtual_pixels_cam_0)
        cam_1_ray = self.camera1(undistorted_virtual_pixels_cam_0)
        #cam_2_ray = self.camera2.initialize_ray(undistorted_virtual_pixels_cam_1)
        cam_2_ray = self.camera2(undistorted_virtual_pixels_cam_1)
        prism_ray11, prism_ray12, _, _, emergent_ray_1, intersection_penalty_1 = self.prism(cam_1_ray)
        prism_ray21, prism_ray22, _,  _, emergent_ray_2, intersection_penalty_2 = self.prism(cam_2_ray)
        recon_3D, closest_distance = closest_point(emergent_ray_1, emergent_ray_2)
        return recon_3D, closest_distance, prism_ray11, prism_ray12, prism_ray21, prism_ray22, intersection_penalty_1, intersection_penalty_2


    def visualize(self, pixels_virtual_two_cams, color_labels=None):
        num_samples = 10
        undistorted_virtual_pixels_cam_0 = pixels_virtual_two_cams[:2, :].clone()
        undistorted_virtual_pixels_cam_1 = pixels_virtual_two_cams[2:, :].clone()
        test_idx = torch.randperm(undistorted_virtual_pixels_cam_0.shape[1])[:num_samples]

        ray = self.camera1.initialize_ray(undistorted_virtual_pixels_cam_1[:, test_idx])
        ray.t *= 100
        fig, ax = self.camera1.visualize()
        fig, ax = ray.visualize(fig, ax, color_labels)
        prism_ray11, prism_ray12, _, _, emergent_ray1, intersection_penalty = self.prism(ray)
        fig, ax = prism_ray11.visualize(fig, ax, color_labels)
        fig, ax = prism_ray12.visualize(fig, ax, color_labels)
        fig, ax = self.prism.visualize_prism(fig, ax)
        emergent_ray1.t *= 25
        fig, ax = emergent_ray1.visualize(fig, ax, color_labels)

        ray = self.camera2.initialize_ray(undistorted_virtual_pixels_cam_1[:, test_idx])
        ray.t *= 100
        fig, ax = self.camera2.visualize(fig, ax)
        fig, ax = ray.visualize(fig, ax, color_labels)
        prism_ray21, prism_ray22, _, _, emergent_ray2, intersection_penalty = self.prism(ray)
        fig, ax = self.prism.visualize_prism(fig, ax)
        fig, ax = prism_ray21.visualize(fig, ax, color_labels)
        fig, ax = prism_ray22.visualize(fig, ax, color_labels)
        emergent_ray2.t *= 25
        fig, ax = emergent_ray2.visualize(fig, ax, color_labels)
        ax.set_aspect('equal', adjustable='datalim')   
