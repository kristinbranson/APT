function result = enumerate_all_files_and_folders_under_helper(parent_info)
  % Find all files and folders under source_path, not including source_path
  % itself.  Output is an nx1 cell array of char-array-strings.  Output is
  % ordered such that parent folders always come before children.

  % Get a list of all files and folders in the parent_path
  [leaf_name_from_index, is_folder_from_index, file_size_from_index] = simple_dir(parent_info.path) ;
  path_from_index = cellfun(@(name)(fullfile(parent_path, name)), leaf_name_from_index) ;

  % Separate source file, folder names
  path_from_file_index = path_from_index(~is_folder_from_index) ;
  path_from_folder_index = path_from_index(is_folder_from_index) ;  
  is_folder_from_file_index = false(size(path_from_file_index)) ;
  is_folder_from_folder_index = true(size(path_from_file_index)) ;
  size_from_file_index = file_size_from_index(~is_folder_from_index) ;
  size_from_folder_index = file_size_from_index(is_folder_from_index) ;

  % Collect all the stuff for the files
  info_from_file_index = struct('path', path_from_file_index, ...
                                'isdir', num2cell(is_folder_from_file_index), ...
                                'bytes', num2cell(size_from_file_index)) ;  

  % Collect all the stuff for the folders
  info_from_folder_index = struct('path', path_from_folder_index, ...
                                  'isdir', num2cell(is_folder_from_folder_index), ...
                                  'bytes', num2cell(size_from_folder_index)) ;  

  % Create a list of lists, containing all the files/folders under each folder
  % in parent_path
  info_list_from_folder_index = arrayfun(@enumerate_all_files_and_folders_under_helper, info_from_folder_index) ;

  % Concatenate everything together.  Files first, then parent folders, then the
  % rest.
  result = cat(1, info_from_file_index, info_from_folder_index, info_list_from_folder_index{:}) ;
end
