function result = wsl_path_from_native(path_or_paths)
% Convert a native path, which might be a Windows-style path (possibly with a
% leading drive letter), to a WSL path.  Also works per-element on
% cellstrings.  On Windows, converts backslash-separated paths to
% slash-separated paths, and converts paths with a leading drive letter to
% their WSL equivalent (e.g. converts 'C:/foo/bar' to '/mnt/c/foo/bar'). See
% comments at top of DLBackEndClass.m for more info on the distinctions
% between native, WSL, and remote file paths.  Note that on Linux this
% function reduces to the identity function.  Does not handle Windows UNC
% paths (like '\\server\folder\file') in input.

if iscell(path_or_paths) ,
  path_from_index = path_or_paths ;
  result = cellfun(@wsl_path_from_native_core, path_from_index, 'UniformOutput', false) ;
else
  path = path_or_paths ;
  result = wsl_path_from_native_core(path) ;
end

end



function result = wsl_path_from_native_core(input_path)
% Convert a native path, which might be a Windows-style path
% (possibly with a leading drive letter), to a WSL path.  Does not
% handle Windows UNC paths (like '\\server\folder\file') in input.

if ispc() ,
  letter_drive_parent = '/mnt' ;
  if length(input_path)>=2 && isstrprop(input_path(1),'alpha') && isequal(input_path(2),':') ,
    drive_letter = input_path(1) ;
    rest = input_path(3:end) ;
    if isempty(rest) 
      % Don't add trailing slash if nothing follows the drive letter
      protoresult_1 = horzcat(letter_drive_parent, '/', lower(drive_letter)) ;
    else      
      protoresult_1 = horzcat(letter_drive_parent, '/', lower(drive_letter), '/', rest) ;
    end
  else
    protoresult_1 = input_path ;
  end
  
  protoresult_2 = regexprep(protoresult_1,'\','/');
  
  result = remove_repeated_slashes(protoresult_2) ;
else
  result = input_path ;
end
 
end
