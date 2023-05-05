function result = linux_path(path_or_paths)
% Convert a platform-native path, which might be a Windows-style path
% (possibly with a leading drive letter), to a linux-style path.  Does not
% handle Windows UNC paths (like '\\server\folder\file') in input.  Also
% works per-element on cellstrings.

if iscell(path_or_paths) ,
  path_from_index = path_or_paths ;
  result = cellfun(@linux_path_core, path_from_index, 'UniformOutput', false) ;
else
  path = path_or_paths ;
  result = linux_path_core(path) ;
end

end



function result = linux_path_core(input_path)
% Convert a platform-native path, which might be a Windows-style path
% (possibly with a leading drive letter), to a linux-style path.  Does not
% handle Windows UNC paths (like '\\server\folder\file') in input.

if ispc() ,
  letter_drive_parent = '/mnt' ;
  if length(input_path)>=2 && isstrprop(input_path(1),'alpha') && isequal(input_path(2),':') ,
    drive_letter = input_path(1) ;
    protoresult_1 = horzcat(letter_drive_parent, '/', lower(drive_letter), '/', input_path(3:end)) ;
  else
    protoresult_1 = input_path ;
  end
  
  protoresult_2 = regexprep(protoresult_1,'\','/');
  
  result = remove_repeated_slashes(protoresult_2) ;
else
  result = input_path ;
end
 
end
