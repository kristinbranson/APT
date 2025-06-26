import numpy as np
import torch
import torch.nn as nn
from ray_tracing_simulator_nnModules_grad import Prism, Ray, Plane, ReflectingPlane, RefractingPlane, EfficientCamera, visualize_camera_configuration, closest_point, rotx, get_rot_mat
from utils import euclidean_distance, rotation_matrix_to_quaternion
pi = torch.tensor(np.pi, dtype=torch.float64)

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
                                            torch.tensor([stereo_alpha, stereo_beta, stereo_gamma], dtype=torch.float64),
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
        return recon_3D, closest_distance, prism_ray11, prism_ray12, prism_ray21, prism_ray22, intersection_penalty_1, intersection_penalty_2


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
                                                             dtype=torch.float64),
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
        recon_3D_real, _ = closest_point(cam_1_ray_real, cam_2_ray_real)
        recon_undistorted_real_pixels_cam_0 = self.camera1.reproject(recon_3D_real, R1, T1)
        recon_undistorted_real_pixels_cam_1 = camera2.reproject(recon_3D_real, R2, T2)
        recon_distorted_real_pixels_cam_0 = self.camera1.distort_pixels(recon_undistorted_real_pixels_cam_0)
        recon_distorted_real_pixels_cam_1 = camera2.distort_pixels(recon_undistorted_real_pixels_cam_1)
        cam_1_ray_virtual = self.camera1(undistorted_virtual_pixels_cam_0)
        cam_2_ray_virtual = camera2(undistorted_virtual_pixels_cam_1)
        _, _, cam_1_ray_virtual, intersection_penalty_1 = self.prism(cam_1_ray_virtual)
        _, _, cam_2_ray_virtual, intersection_penalty_2 = self.prism(cam_2_ray_virtual)
        recon_3D_virtual, closest_distance = closest_point(cam_1_ray_virtual, cam_2_ray_virtual)
        recon_3D_1, _ = closest_point(cam_1_ray_virtual, cam_2_ray_real)
        recon_3D_2, _ = closest_point(cam_1_ray_virtual, cam_2_ray_real)
        recon_3D = (recon_3D_real + recon_3D_1 + recon_3D_2) / 3
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
        return recon_3D, closest_distance, recon_pixels_1, recon_pixels_2, recon_distorted_virtual_pixels_cam_0, recon_distorted_virtual_pixels_cam_1, recon_distorted_real_pixels_cam_0, recon_distorted_real_pixels_cam_1, intersection_penalty_1, distortion_penalty_cam_0, distortion_penalty_cam_1, intersection_penalty_2


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
    prism_center=None):
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
        self.stereo_camera_angles = nn.Parameter(
                                                torch.tensor([stereo_alpha, stereo_beta, stereo_gamma], 
                                                             dtype=torch.float64),
                                                requires_grad=True
                                                )


        # Prism initialization
        self.prism_angles = nn.Parameter(prism_angles, requires_grad=True)
        refractive_index_glass = torch.tensor(1.51, dtype=torch.float64)
        self.refractive_index_glass = nn.Parameter(refractive_index_glass, requires_grad=True)
        if prism_center is None:
            prism_center = self.camera1.aperture.clone() + self.camera1.axes[:,0].unsqueeze(-1).clone() * prism_distance.clone()
            prism_center[0] = -5.

        self.prism_center = nn.Parameter(prism_center, requires_grad=True)
        self.prism_size = nn.Parameter(torch.tensor([20.,20.,20.],dtype=torch.float64), requires_grad=True)

        self.prism = Prism(prism_size=self.prism_size, 
                        prism_center=self.prism_center, 
                        prism_angles=self.prism_angles,
                        refractive_index_glass=self.refractive_index_glass,
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
                     r1,
                     radial_dist_coeffs):
        camera2 = EfficientCamera(
            principal_point_pixel=principal_point_pixel_cam_1, 
            focal_length_pixels=focal_length_cam_1, 
            r1=r1,
            radial_dist_coeffs=radial_dist_coeffs)
        camera2.update_camera_pose(R, T)
        return camera2


    def forward(self, pixels_virtual_two_cams, pixels_real_two_cams):
        self.prism = Prism(prism_size=self.prism_size, 
                        prism_center=self.prism_center, 
                        prism_angles=self.prism_angles,
                        refractive_index_glass=self.refractive_index_glass,
                        )
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

        #undistorted_virtual_pixels_cam_0 = self.camera1.undistort_pixels(distorted_virtual_pixels_cam_0)
        #undistorted_virtual_pixels_cam_1 = camera2.undistort_pixels(distorted_virtual_pixels_cam_1)
        #undistorted_real_pixels_cam_0 = self.camera1.undistort_pixels(distorted_real_pixels_cam_0)
        #undistorted_real_pixels_cam_1 = camera2.undistort_pixels(distorted_real_pixels_cam_1)
        cam_1_ray_real = self.camera1(undistorted_real_pixels_cam_0)
        cam_2_ray_real = camera2(undistorted_real_pixels_cam_1)
        recon_3D_real, closest_distance_real = closest_point(cam_1_ray_real, cam_2_ray_real)
        #recon_undistorted_real_pixels_cam_0 = self.camera1.reproject(recon_3D_real, R1, T1)
        #recon_undistorted_real_pixels_cam_1 = camera2.reproject(recon_3D_real, R2, T2)
        #recon_distorted_real_pixels_cam_0 = self.camera1.distort_pixels(recon_undistorted_real_pixels_cam_0)
        #recon_distorted_real_pixels_cam_1 = camera2.distort_pixels(recon_undistorted_real_pixels_cam_1)
        """
        recon_distorted_real_pixels_cam_0 = self.camera1.distort_pixels_classical(recon_undistorted_real_pixels_cam_0, 
                                                                                self.radial_dist_coeffs_cam_0)
        recon_distorted_real_pixels_cam_1 = camera2.distort_pixels_classical(recon_undistorted_real_pixels_cam_1,
                                                                            self.radial_dist_coeffs_cam_1)
        """
        cam_1_ray_virtual = self.camera1(undistorted_virtual_pixels_cam_0)
        cam_2_ray_virtual = camera2(undistorted_virtual_pixels_cam_1)
        _, _, cam_1_ray_virtual, intersection_penalty_1 = self.prism(cam_1_ray_virtual)
        _, _, cam_2_ray_virtual, intersection_penalty_2 = self.prism(cam_2_ray_virtual)        
        recon_3D_virtual, closest_distance_virtual = closest_point(cam_1_ray_virtual, cam_2_ray_virtual)        
        recon_3D_1, closest_distance_1 = closest_point(cam_1_ray_virtual, cam_1_ray_real)
        recon_3D_2, closest_distance_2 = closest_point(cam_2_ray_virtual, cam_2_ray_real)
        recon_3D_12, closest_distance_12 = closest_point(cam_1_ray_virtual, cam_2_ray_real)
        recon_3D_21, closest_distance_21 = closest_point(cam_2_ray_virtual, cam_1_ray_real)
        recon_3D = (recon_3D_real + recon_3D_1 + recon_3D_2 + recon_3D_12 + recon_3D_21 + recon_3D_virtual) / 6
        #recon_3D = (recon_3D_real + recon_3D_1 + recon_3D_2) / 3
        num_points = recon_3D.shape[1] // 2
        closest_distance = (closest_distance_real + closest_distance_virtual + closest_distance_1 + closest_distance_2 + closest_distance_12 + closest_distance_21) / 6
        closest_distance = closest_distance[:num_points]
        
        pairwise_distance = euclidean_distance(
            recon_3D[:, :num_points],
            recon_3D[:, num_points:]
        )        
        recon_pixels_1_undistorted = self.camera1.reproject(recon_3D, R1, T1)
        recon_pixels_1 = self.camera1.distort_pixels_classical(recon_pixels_1_undistorted, self.radial_dist_coeffs_cam_0)
        recon_pixels_2_undistorted = camera2.reproject(recon_3D, R2, T2)
        recon_pixels_2 = camera2.distort_pixels_classical(recon_pixels_2_undistorted, self.radial_dist_coeffs_cam_1)
        """
        recon_undistorted_virtual_pixels_cam_0 = self.camera1.reproject(recon_3D_virtual, R1, T1)
        recon_undistorted_virtual_pixels_cam_1 = camera2.reproject(recon_3D_virtual, R2, T2)

        # Distortion loss (constraining the distortion and undistortion function to be the inverse of each other)        
        
        recon_distorted_virtual_pixels_cam_0 = self.camera1.distort_pixels_classical(recon_undistorted_virtual_pixels_cam_0,
                                                                                self.radial_dist_coeffs_cam_0)
        recon_distorted_virtual_pixels_cam_1 = camera2.distort_pixels_classical(recon_undistorted_virtual_pixels_cam_1,
                                                                                self.radial_dist_coeffs_cam_1)
        
        distortion_penalty_cam_0 = self.camera1.calculate_distortion_penalty(distorted_virtual_pixels_cam_0, self.radial_dist_coeffs_cam_0) + self.camera1.calculate_distortion_penalty(distorted_real_pixels_cam_0, self.radial_dist_coeffs_cam_0)
        distortion_penalty_cam_1 = camera2.calculate_distortion_penalty(distorted_virtual_pixels_cam_1, self.radial_dist_coeffs_cam_1) + camera2.calculate_distortion_penalty(distorted_real_pixels_cam_1, self.radial_dist_coeffs_cam_1)        
        """

        distortion_penalty_cam_0 = self.camera1.calculate_distortion_penalty(recon_pixels_1, 
                                                                             self.radial_dist_coeffs_cam_0)
        distortion_penalty_cam_1 = camera2.calculate_distortion_penalty(recon_pixels_2,
                                                                        self.radial_dist_coeffs_cam_1)
        
        return recon_3D, closest_distance, recon_pixels_1, recon_pixels_2, recon_3D_real, recon_3D_virtual, distortion_penalty_cam_0, distortion_penalty_cam_1, intersection_penalty_1, intersection_penalty_2, pairwise_distance


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
        R_stereo_cam = get_rot_mat(
            self.stereo_camera_angles[0],
            self.stereo_camera_angles[1],
            self.stereo_camera_angles[2],
        )
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
                                                             dtype=torch.float64),
                                                requires_grad=False
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
        return recon_3D, closest_distance, recon_distorted_pixels_cam_0, intersection_penalty_1, distortion_penalty_cam_0


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

class Arena_two_real_cameras_quat(nn.Module):
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
        super(Arena_two_real_cameras_quat, self).__init__()

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
        self.stereo_camera_angles = torch.tensor([stereo_alpha, stereo_beta, stereo_gamma]).to(torch.float64)
        self.stereo_camera_quat = nn.Parameter(rotation_matrix_to_quaternion(self.R_stereo_cam),
                                               requires_grad=True)
        

    def get_stereo_camera_angles(self, 
                                 R_stereo_cam):
        axes = torch.eye(3,3).to(torch.float64)
        axes = torch.mm(R_stereo_cam, axes)
        plane = Plane(axes=axes)
        return plane.alpha, plane.beta, plane.gamma
    

    def get_stereo_camera(self, 
                     principal_point_pixel_cam_1,
                     focal_length_cam_1,
                     quat_stereo_cam,
                     T,
                     r1):
        camera2 = EfficientCamera(
            principal_point_pixel=principal_point_pixel_cam_1, 
            focal_length_pixels=focal_length_cam_1, 
            r1=r1)
        camera2.update_camera_pose(quat=quat_stereo_cam, T=T)        
        return camera2


    def forward(self, _, pixels_real_two_cams):
        R1 = torch.eye(3, 3).to(torch.float64)
        T1 = torch.zeros(3, 1).to(torch.float64)
        T2 = self.T_stereo_cam
        
        camera2 = self.get_stereo_camera(self.principal_point_pixel_cam_1,
                                    self.focal_length_cam_1,
                                    self.stereo_camera_quat,
                                    self.T_stereo_cam,
                                    r1=self.stereocam_r1)                
        
        distorted_real_pixels_cam_0 = pixels_real_two_cams[:2,:]
        distorted_real_pixels_cam_1 = pixels_real_two_cams[2:,:]
        undistorted_real_pixels_cam_0 = self.camera1.undistort_pixels(distorted_real_pixels_cam_0)
        #undistorted_real_pixels_cam_0 = distorted_real_pixels_cam_0
        #undistorted_real_pixels_cam_1 = distorted_real_pixels_cam_1
        undistorted_real_pixels_cam_1 = camera2.undistort_pixels(distorted_real_pixels_cam_1)
        cam_1_ray_real = self.camera1(undistorted_real_pixels_cam_0)
        cam_2_ray_real = camera2(undistorted_real_pixels_cam_1)
        recon_3D, closest_distance = closest_point(cam_1_ray_real, cam_2_ray_real)
        recon_undistorted_real_pixels_cam_0 = self.camera1.reproject(recon_3D, T1)
        recon_undistorted_real_pixels_cam_1 = camera2.reproject(recon_3D, T2)
        recon_distorted_real_pixels_cam_0 = self.camera1.distort_pixels(recon_undistorted_real_pixels_cam_0)
        recon_distorted_real_pixels_cam_1 = camera2.distort_pixels(recon_undistorted_real_pixels_cam_1)   
        #recon_distorted_real_pixels_cam_0 = recon_undistorted_real_pixels_cam_0
        #recon_distorted_real_pixels_cam_1 = recon_undistorted_real_pixels_cam_1
        recon_loss_real = (torch.norm(recon_distorted_real_pixels_cam_0 - distorted_real_pixels_cam_0, 
                    p=2, 
                    dim=0) + torch.norm(recon_distorted_real_pixels_cam_1 - distorted_real_pixels_cam_1, 
                    p=2, 
                    dim=0)) / 2
        
        return recon_3D, closest_distance, recon_distorted_real_pixels_cam_0, recon_distorted_real_pixels_cam_1, recon_loss_real
    
    def visualize(self, pixels_virtual_two_cams, color_labels=None):
        num_samples = 10
        undistorted_virtual_pixels_cam_0 = pixels_virtual_two_cams[:2, :].clone()
        undistorted_virtual_pixels_cam_1 = pixels_virtual_two_cams[2:, :].clone()
        test_idx = torch.randperm(undistorted_virtual_pixels_cam_0.shape[1])[:num_samples]

        ray = self.camera1.initialize_ray(undistorted_virtual_pixels_cam_1[:, test_idx])
        ray.t *= 100
        fig, ax = self.camera1.visualize()
        fig, ax = ray.visualize(fig, ax, color_labels)
        
        camera2 = self.get_stereo_camera(self.principal_point_pixel_cam_1,
                                self.focal_length_cam_1,
                                self.stereo_camera_quat,
                                self.T_stereo_cam,
                                self.stereocam_r1)
        ray = camera2.initialize_ray(undistorted_virtual_pixels_cam_1[:, test_idx])

        ray.t *= 100
        fig, ax = camera2.visualize(fig, ax)
        fig, ax = ray.visualize(fig, ax, color_labels)        
        ax.set_aspect('equal', adjustable='datalim')   

# Two cameras
class Arena_two_real_cameras(nn.Module):
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
        super(Arena_two_real_cameras, self).__init__()

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
                                                             dtype=torch.float64),
                                                requires_grad=False
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


    def forward(self, _, pixels_real_two_cams):
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
        
        distorted_real_pixels_cam_0 = pixels_real_two_cams[:2,:]
        distorted_real_pixels_cam_1 = pixels_real_two_cams[2:,:]
        undistorted_real_pixels_cam_0 = self.camera1.undistort_pixels(distorted_real_pixels_cam_0)
        #undistorted_real_pixels_cam_0 = distorted_real_pixels_cam_0
        #undistorted_real_pixels_cam_1 = distorted_real_pixels_cam_1
        undistorted_real_pixels_cam_1 = camera2.undistort_pixels(distorted_real_pixels_cam_1)
        cam_1_ray_real = self.camera1(undistorted_real_pixels_cam_0)
        cam_2_ray_real = camera2(undistorted_real_pixels_cam_1)
        recon_3D, closest_distance = closest_point(cam_1_ray_real, cam_2_ray_real)
        recon_undistorted_real_pixels_cam_0 = self.camera1.reproject(recon_3D, R1, T1)
        recon_undistorted_real_pixels_cam_1 = camera2.reproject(recon_3D, R2, T2)
        recon_distorted_real_pixels_cam_0 = self.camera1.distort_pixels(recon_undistorted_real_pixels_cam_0)
        recon_distorted_real_pixels_cam_1 = camera2.distort_pixels(recon_undistorted_real_pixels_cam_1)   
        #recon_distorted_real_pixels_cam_0 = recon_undistorted_real_pixels_cam_0
        #recon_distorted_real_pixels_cam_1 = recon_undistorted_real_pixels_cam_1
        recon_loss_real = (torch.norm(recon_distorted_real_pixels_cam_0 - distorted_real_pixels_cam_0, 
                    p=2, 
                    dim=0) + torch.norm(recon_distorted_real_pixels_cam_1 - distorted_real_pixels_cam_1, 
                    p=2, 
                    dim=0)) / 2
        
        return recon_3D, closest_distance, recon_distorted_real_pixels_cam_0, recon_distorted_real_pixels_cam_1, recon_loss_real


    def visualize(self, pixels_virtual_two_cams, color_labels=None):
        num_samples = 10
        undistorted_virtual_pixels_cam_0 = pixels_virtual_two_cams[:2, :].clone()
        undistorted_virtual_pixels_cam_1 = pixels_virtual_two_cams[2:, :].clone()
        test_idx = torch.randperm(undistorted_virtual_pixels_cam_0.shape[1])[:num_samples]

        ray = self.camera1.initialize_ray(undistorted_virtual_pixels_cam_1[:, test_idx])
        ray.t *= 100
        fig, ax = self.camera1.visualize()
        fig, ax = ray.visualize(fig, ax, color_labels)
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
        ax.set_aspect('equal', adjustable='datalim')   

#
class Arena_fish_tank(nn.Module):
    def __init__(self, 
    principal_point_pixel_cam_0, 
    principal_point_pixel_cam_1, 
    focal_length_cam_0, 
    focal_length_cam_1, 
    R_stereo_cam, 
    T_stereo_cam, 
    tank_thickness=None,
    tank_angles=None,
    tank_center=None,
    tank_size=None,
    refractive_index_acrylic=None,
    refractive_index_water=None):
        super(Arena_fish_tank, self).__init__()

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
                                                             dtype=torch.float64),
                                                requires_grad=False
                                                )
        self.tank_center = nn.Parameter(tank_center)
        self.tank_angles = nn.Parameter(tank_angles)
        self.tank_thickness = nn.Parameter(tank_thickness)
        self.refractive_index_acrylic = nn.Parameter(refractive_index_acrylic)
        self.refractive_index_water = nn.Parameter(refractive_index_water)
        self.radial_dist_coeffs_cam_0 = nn.Parameter(torch.tensor([0.,0.,0.]).unsqueeze(-1).to(torch.float64),
                                               requires_grad=True)
        self.radial_dist_coeffs_cam_1 = nn.Parameter(torch.tensor([0.,0.,0.]).unsqueeze(-1).to(torch.float64),
                                               requires_grad=True)
        self.tank_size = nn.Parameter(tank_size)
        self.refractive_index_air = torch.ones_like(refractive_index_acrylic)
        

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

    def forward(self, pixels_two_cams):
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
        side_plane1, side_plane2, top_plane = self.get_tank_planes()
        pixels_side_cam = pixels_two_cams[:2, :]
        pixels_top_cam = pixels_two_cams[2:, :]
        pixels_side_cam_undistorted = self.camera1.undistort_pixels_classical(pixels_side_cam, self.radial_dist_coeffs_cam_0)
        pixels_top_cam_undistorted = camera2.undistort_pixels_classical(pixels_top_cam, self.radial_dist_coeffs_cam_1)
        ray = camera2(pixels_side_cam_undistorted)
        ray2, _ = side_plane1(ray)
        ray_side_cam, _ = side_plane2(ray2)
        ray = self.camera1(pixels_top_cam_undistorted)
        ray_top_cam, _ = top_plane(ray)
        recon_3D, closest_distance = closest_point(ray_side_cam, ray_top_cam)
        return recon_3D, closest_distance
    
    def visualize(self, pixels_two_cams, color_labels=None):
        num_samples = 10
        undistorted_pixels_cam_0 = pixels_two_cams[:2, :].clone()
        undistorted_pixels_cam_1 = pixels_two_cams[2:, :].clone()
        side_plane1, side_plane2, top_plane = self.get_tank_planes()

        test_idx = torch.randperm(undistorted_pixels_cam_0.shape[1])[:num_samples]
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
        ray1s = camera2.initialize_ray(undistorted_pixels_cam_0[:, test_idx])
        ray2s, _ = side_plane1(ray1s)
        ray_side, _ = side_plane2(ray2s)
        fig, ax = self.camera1.visualize()
        fig, ax = ray1s.visualize(fig, ax, color_labels)
        fig, ax = ray2s.visualize(fig, ax, color_labels)
        fig, ax = ray_side.visualize(fig, ax, color_labels)
        fig, ax = side_plane1.visualize(fig, ax, color_labels)
        fig, ax = side_plane2.visualize(fig, ax, color_labels)

        
        ray1t = self.camera1.initialize_ray(undistorted_pixels_cam_1[:, test_idx])
        ray_top, _ = top_plane(ray1t)
        fig, ax = camera2.visualize(fig, ax)
        fig, ax = ray1t.visualize(fig, ax, color_labels)  
        fig, ax = ray_top.visualize(fig, ax, color_labels)     
        fig, ax = top_plane.visualize(fig, ax, color_labels)
        ax.set_aspect('equal', adjustable='datalim') 

    def get_tank_planes(self):
        tank_alpha, tank_beta, tank_gamma = self.tank_angles
        rot_mat = get_rot_mat(tank_alpha, tank_beta, tank_gamma)
        axes1 = torch.mm(rot_mat, 
                            torch.tensor([[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]], dtype=torch.float64).t()
                            )        
        
        side_plane1_center = torch.vstack([
            self.tank_center[0],
            self.tank_center[1],
            self.tank_center[2],
        ]
        )
        side_plane1 = RefractingPlane(
            refractive_idx_1=self.refractive_index_air,
            refractive_idx_2=self.refractive_index_acrylic,
            axes=axes1,
            a=self.tank_size[1],
            b=self.tank_size[2],
            center=side_plane1_center,
        )
        side_plane2_center = side_plane1_center - self.tank_thickness * axes1[:,0].unsqueeze(-1)


        side_plane2 = RefractingPlane(
            axes=axes1,
            refractive_idx_1=self.refractive_index_acrylic,
            refractive_idx_2=self.refractive_index_water,
            a=self.tank_size[1],
            b=self.tank_size[2],
            center=side_plane2_center,
        )

        rot_mat_top = get_rot_mat(torch.tensor(0.).to(torch.float64),
                                  torch.tensor(pi/2).to(torch.float64),
                                  torch.tensor(0.).to(torch.float64))
        axes_top = torch.mm(rot_mat_top, 
                            axes1,
                            )
        axes_top = torch.mm(rot_mat_top,
                            axes_top,
                            )
        
        top_plane_center = torch.vstack([
            self.tank_center[0],
            self.tank_center[1],
            self.tank_center[2],
        ]
        ) - self.tank_size[0] / 2 * axes1[:,0].unsqueeze(-1) + self.tank_size[2] / 2 * axes_top[:,0].unsqueeze(-1)
        
        top_plane = RefractingPlane(
            axes=axes_top,
            refractive_idx_1=self.refractive_index_air,
            refractive_idx_2=self.refractive_index_water,
            a=self.tank_size[0],
            b=self.tank_size[1],
            center=top_plane_center
        )
        return side_plane1, side_plane2, top_plane 

class Arena_fish_tank_pairwise_distances(nn.Module):
    def __init__(self, 
    principal_point_pixel_cam_0, 
    principal_point_pixel_cam_1, 
    focal_length_cam_0, 
    focal_length_cam_1, 
    R_stereo_cam, 
    T_stereo_cam, 
    tank_thickness=None,
    tank_angles=None,
    tank_center=None,
    tank_size=None,
    refractive_index_acrylic=None,
    refractive_index_water=None):
        super(Arena_fish_tank_pairwise_distances, self).__init__()

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
                                                             dtype=torch.float64),
                                                )
        self.tank_center = nn.Parameter(tank_center)
        self.tank_angles = nn.Parameter(tank_angles, requires_grad=False)
        self.tank_thickness = nn.Parameter(tank_thickness)
        self.refractive_index_acrylic = nn.Parameter(refractive_index_acrylic)
        self.refractive_index_water = nn.Parameter(refractive_index_water)
        self.radial_dist_coeffs_cam_0 = nn.Parameter(torch.tensor([0.,0.,0.]).unsqueeze(-1).to(torch.float64),
                                               requires_grad=True)
        self.radial_dist_coeffs_cam_1 = nn.Parameter(torch.tensor([0.,0.,0.]).unsqueeze(-1).to(torch.float64),
                                               requires_grad=True)
        self.tank_size = nn.Parameter(tank_size, requires_grad=True)
        self.refractive_index_air = torch.ones_like(refractive_index_acrylic)
        

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

    def forward(self, pixels_two_cams):
        R_stereo_cam = get_rot_mat(
            self.stereo_camera_angles[0],
            self.stereo_camera_angles[1],
            self.stereo_camera_angles[2],
            )
        camera2 = self.get_stereo_camera(self.principal_point_pixel_cam_1,
                                    self.focal_length_cam_1,
                                    R_stereo_cam,
                                    self.T_stereo_cam,
                                    r1=self.stereocam_r1)
        side_plane1, side_plane2, top_plane = self.get_tank_planes()
        pixels_top_cam = torch.hstack((
            pixels_two_cams[:2, :], 
            pixels_two_cams[2:4,:])
        )
        pixels_side_cam = torch.hstack((
            pixels_two_cams[4:6, :],
            pixels_two_cams[6:,:])
        )
        pixels_side_cam_undistorted = camera2.undistort_pixels_classical(pixels_side_cam, self.radial_dist_coeffs_cam_1)
        pixels_top_cam_undistorted = self.camera1.undistort_pixels_classical(pixels_top_cam, self.radial_dist_coeffs_cam_0)
        ray = camera2(pixels_side_cam_undistorted)
        ray2, intersection_penalty_1 = side_plane1(ray)
        ray_side_cam, intersection_penalty_2 = side_plane2(ray2)
        ray = self.camera1(pixels_top_cam_undistorted)
        ray_top_cam, intersection_penalty_3 = top_plane(ray)
        recon_3D, closest_distance = closest_point(ray_side_cam, ray_top_cam)
        num_points = recon_3D.shape[1] // 2
        pairwise_distance = euclidean_distance(
            recon_3D[:, :num_points],
            recon_3D[:, num_points:]
        )
        zero_columns_top = (ray_top_cam.direction == 0).all(dim=0)
        zero_columns_side = (ray_side_cam.direction == 0).all(dim=0)
        num_bad_rays = torch.sum(zero_columns_side) + torch.sum(zero_columns_top)
        return recon_3D, closest_distance, pairwise_distance, intersection_penalty_1 + intersection_penalty_2 + intersection_penalty_3, num_bad_rays


    def visualize(self, pixels_two_cams, rand_sample, color_labels=None):
        num_samples = 10
        undistorted_pixels_cam_top = torch.hstack(
            (pixels_two_cams[:2, :].clone(), pixels_two_cams[2:4, :].clone())
        )
        undistorted_pixels_cam_side = torch.hstack(
            (pixels_two_cams[4:6, :].clone(),
             pixels_two_cams[6:,:])
        )
        side_plane1, side_plane2, top_plane = self.get_tank_planes()

        if rand_sample:
            test_idx = torch.randperm(undistorted_pixels_cam_top.shape[1])[:num_samples]
        else:
            test_idx = torch.arange(0,num_samples)
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
        ray1s = camera2.initialize_ray(undistorted_pixels_cam_side[:, test_idx])
        ray2s, _ = side_plane1(ray1s)
        ray_side, _ = side_plane2(ray2s)
        fig, ax = self.camera1.visualize()
        fig, ax = ray1s.visualize(fig, ax, color_labels)
        fig, ax = ray2s.visualize(fig, ax, color_labels)
        ray_side.t = 500 * ray_side.t
        fig, ax = ray_side.visualize(fig, ax, color_labels)
        fig, ax = side_plane1.visualize(fig, ax, color_labels)
        fig, ax = side_plane2.visualize(fig, ax, color_labels)

        
        ray1t = self.camera1.initialize_ray(undistorted_pixels_cam_top[:, test_idx])
        ray_top, _ = top_plane(ray1t)
        fig, ax = camera2.visualize(fig, ax)
        fig, ax = ray1t.visualize(fig, ax, color_labels)  
        ray_top.t = 500 * ray_top.t
        fig, ax = ray_top.visualize(fig, ax, color_labels)     
        fig, ax = top_plane.visualize(fig, ax, color_labels)
        ax.set_aspect('equal', adjustable='datalim') 
        return camera2, side_plane1, side_plane2, top_plane, pixels_two_cams

    def get_tank_planes(self):
        tank_alpha, tank_beta, tank_gamma = self.tank_angles
        rot_mat = get_rot_mat(tank_alpha, tank_beta, tank_gamma)
        # NOTE: Do not change the axes definition here. Tank angles should be provided as input. THe tank angles can be derived from axes
        axes1 = torch.mm(rot_mat, 
                            torch.tensor([[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]], dtype=torch.float64).t()
                            )        
        
        side_plane1_center = torch.vstack([
            self.tank_center[0],
            self.tank_center[1],
            self.tank_center[2],
        ]
        )
        
        side_plane1 = RefractingPlane(
            refractive_idx_1=self.refractive_index_air,
            refractive_idx_2=self.refractive_index_acrylic,
            axes=axes1,
            a=self.tank_size[1],
            b=self.tank_size[2],
            center=side_plane1_center,
        )
        side_plane2_center = side_plane1_center - self.tank_thickness * axes1[:,0].unsqueeze(-1)


        side_plane2 = RefractingPlane(
            axes=axes1,
            refractive_idx_1=self.refractive_index_acrylic,
            refractive_idx_2=self.refractive_index_water,
            a=self.tank_size[1],
            b=self.tank_size[2],
            center=side_plane2_center,
        )

        rot_mat_top = get_rot_mat(torch.tensor(0.).to(torch.float64),
                                  torch.tensor(-pi/2).to(torch.float64),
                                  torch.tensor(0.).to(torch.float64))
        axes_top = torch.mm(rot_mat_top, 
                            torch.tensor([[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]], dtype=torch.float64).t(),
                            )
        axes_top = torch.mm(rot_mat,
                            axes_top
                            )

        top_plane_center = torch.vstack([
            self.tank_center[0],
            self.tank_center[1],
            self.tank_center[2],
        ]
        ) - self.tank_size[0] / 2 * axes1[:,0].unsqueeze(-1) + self.tank_size[2] / 2 * axes_top[:,0].unsqueeze(-1)
        
        top_plane = RefractingPlane(
            axes=axes_top,
            refractive_idx_1=self.refractive_index_air,
            refractive_idx_2=self.refractive_index_water,
            a=self.tank_size[0],
            b=self.tank_size[1],
            center=top_plane_center
        )
        return side_plane1, side_plane2, top_plane 
    

# Two cameras grid distances
class Arena_two_real_cameras_grid_distance(nn.Module):
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
        super(Arena_two_real_cameras_grid_distance, self).__init__()

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
                                                             dtype=torch.float64),
                                                requires_grad=False
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


    def forward(self, pixels_real_two_cams):
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
        
        distorted_real_pixels_cam_0 = torch.hstack((pixels_real_two_cams[:2,:], 
                                                    pixels_real_two_cams[2:4,:]))
        distorted_real_pixels_cam_1 = torch.hstack((pixels_real_two_cams[4:6,:], 
                                                    pixels_real_two_cams[6:,:]))

        undistorted_real_pixels_cam_0 = self.camera1.undistort_pixels(distorted_real_pixels_cam_0)
        undistorted_real_pixels_cam_1 = camera2.undistort_pixels(distorted_real_pixels_cam_1)
        #undistorted_real_pixels_cam_0 = distorted_real_pixels_cam_0        
        #undistorted_real_pixels_cam_1 = distorted_real_pixels_cam_1
        cam_1_ray_real = self.camera1(undistorted_real_pixels_cam_0)
        cam_2_ray_real = camera2(undistorted_real_pixels_cam_1)
        recon_3D, closest_distance = closest_point(cam_1_ray_real, cam_2_ray_real)
        num_points = recon_3D.shape[1] // 2
        pairwise_distance = euclidean_distance(
            recon_3D[:, :num_points],
            recon_3D[:, num_points:]
        )
        recon_undistorted_real_pixels_cam_0 = self.camera1.reproject(recon_3D, R1, T1)
        recon_undistorted_real_pixels_cam_1 = camera2.reproject(recon_3D, R2, T2)
        recon_distorted_real_pixels_cam_0 = self.camera1.distort_pixels(recon_undistorted_real_pixels_cam_0)
        recon_distorted_real_pixels_cam_1 = camera2.distort_pixels(recon_undistorted_real_pixels_cam_1)   
        #recon_distorted_real_pixels_cam_0 = recon_undistorted_real_pixels_cam_0
        #recon_distorted_real_pixels_cam_1 = recon_undistorted_real_pixels_cam_1
                
        #recon_3D = recon_3D[:, :num_points]
        closest_distance = closest_distance[:num_points]
        #recon_distorted_real_pixels_cam_0 = recon_distorted_real_pixels_cam_0[:, :num_points]
        #recon_distorted_real_pixels_cam_1 = recon_distorted_real_pixels_cam_1[:, :num_points]
        recon_loss_real = (euclidean_distance(
            recon_distorted_real_pixels_cam_0,
            distorted_real_pixels_cam_0) + euclidean_distance(
            recon_distorted_real_pixels_cam_1,
            distorted_real_pixels_cam_1)) / 4
        return recon_3D, closest_distance, recon_distorted_real_pixels_cam_0, recon_distorted_real_pixels_cam_1, recon_loss_real, pairwise_distance


    def visualize(self, pixels_virtual_two_cams, color_labels=None):
        num_samples = 10
        undistorted_virtual_pixels_cam_0 = pixels_virtual_two_cams[:2, :].clone()
        undistorted_virtual_pixels_cam_1 = pixels_virtual_two_cams[4:6, :].clone()
        test_idx = torch.randperm(undistorted_virtual_pixels_cam_0.shape[1])[:num_samples]
        ray = self.camera1.initialize_ray(undistorted_virtual_pixels_cam_0[:, test_idx])
        ray.t *= 100
        fig, ax = self.camera1.visualize()
        fig, ax = ray.visualize(fig, ax, color_labels)
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