function result = enumerate_all_files_and_folders_under(parent_path)
  % Find all files and folders under parent_path, not including parent_path
  % itself.  Output is an nx1 struct array with same fields as struct returned
  % by built-in dir(). Output is ordered such that parent folders always come
  % before children.

  % Get a list of all files and folders in the parent_path
  dis_from_index = dir_without_links(parent_path) ;
  
  % Compute full path for each entity
  is_folder_from_index = [dis_from_index.isdir]' ;
  is_file_from_index = ~is_folder_from_index ;
  
  % Separate source file, folder names
  dis_from_file_index = dis_from_index(is_file_from_index) ;
  dis_from_folder_index = dis_from_index(is_folder_from_index) ;

  % Create a list of lists, containing all the files/folders under each folder
  % in parent_path
  path_from_folder_index = path_from_dir_struct(dis_from_folder_index) ;
  dis_from_entry_index_array_from_folder_index = ...
    cellfun(@enumerate_all_files_and_folders_under, path_from_folder_index, 'UniformOutput', false) ;  % cell array of ds arrays

  % Concatenate everything together.  Files first, then parent folders, then the
  % rest.
  result = cat(1, dis_from_file_index, dis_from_folder_index, dis_from_entry_index_array_from_folder_index{:}) ;
end
