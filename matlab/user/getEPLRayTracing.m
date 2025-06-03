%% Function that actually communicates with Python to compute epipolar lines using inverse ray-tracing
% Args: 
% model_path : (string) Path of the trained pth model file
% python_script_path : (string) Path of the Python script that computes
%                       epipolar lines
% dividing_col : (int) Column number used for cropping real and virtual
%               views

function [epipolar_line_projecting_cam] = getEPLRayTracing(model_path, ...
   python_script_path, dividing_col, image_width, cam_label, cam_projecting, user_annotation)


user_annotation = user_annotation - 1; % Accounting for Python's zero-indexing
epipolar_line_projecting_cam = pyrunfile(python_script_path,...
   ["epipolar_line_projecting_cam"], ...
   user_annotation=user_annotation, PATH=model_path, cam_label=cam_label, ...
   cam_projecting=cam_projecting, dividing_col=dividing_col, image_width=image_width, APT_path=APT.Root);
epipolar_line_projecting_cam = double(epipolar_line_projecting_cam) + 1; % +1 to account for Python's zero-indexing