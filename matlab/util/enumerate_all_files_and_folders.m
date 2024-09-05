function result = enumerate_all_files_and_folders(parent_path)
% Find all files and folders under parent_path, including parent_path
% itself.  Output is an nx1 struct array with same fields as struct returned
% by built-in dir(). Output is ordered such that parent folders always come
% before children.

% Get a list of all files and folders in the parent_path
parent_dis = dir_without_links(parent_path) ;  % ds for "dir struct"  
rest_dis = enumerate_all_files_and_folders_under(parent_path) ;

% Concatenate everything together
result = vertcat(parent_dis, rest_dis) ;
