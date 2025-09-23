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
  inputPath = apt.Path(input_path, apt.Platform.windows) ;
  outputPath = inputPath.toPosix() ;
  result = outputPath.char() ;
else
  result = input_path ;
end
 
end
