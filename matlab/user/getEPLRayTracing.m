%% Function that actually communicates with Python to compute epipolar lines using inverse ray-tracing
% Args: 
% model_path : (string) Path of the trained pth model file
% python_script_path : (string) Path of the Python script that computes
%                       epipolar lines
% dividing_col : (int) Column number used for cropping real and virtual
%               views

function [epipolar_line_unlabelled, epipolar_line_labelled] = getEPLRayTracing(model_path, ...
    python_script_path, dividing_col, image_width, cam_label, user_annotation)
% pyenv('Version','/groups/branson/bransonlab/aniket/pytorch_remote/bin/python');
user_annotation = user_annotation - 1; % Accounting for Python's zero-indexing
[epipolar_line_unlabelled, epipolar_line_labelled] = pyrunfile(python_script_path,...
    ["epipolar_line_unlabelled", "epipolar_line_labelled"], ...
    user_annotation=user_annotation, PATH=model_path, cam_label=cam_label, ...
    dividing_col=dividing_col, image_width=image_width);
epipolar_line_unlabelled = double(epipolar_line_unlabelled) + 1; % +1 to account for Python's zero-indexing
epipolar_line_labelled = double(epipolar_line_labelled) + 1; % +1 to account for Python's zero-indexing