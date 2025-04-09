function result = native_path_from_wsl(path_or_paths)
  % Convert a WSL path to a native path.  Also works per-element on
  % cellstrings.  On Windows, converts slash-separated paths to
  % backslash-separated paths, and converts paths starting with '/mnt' to their native equivalent.
  % (e.g. converts '/mnt/c/foo/bar' to 'C:\foo\bar'). See
  % comments at top of DLBackEndClass.m for more info on the distinctions
  % between native, WSL, and remote file paths.  Note that on Linux this
  % function reduces to the identity function.
  
  if iscell(path_or_paths) ,
    path_from_index = path_or_paths ;
    result = cellfun(@native_path_from_wsl_core, path_from_index, 'UniformOutput', false) ;
  else
    path = path_or_paths ;
    result = native_path_from_wsl_core(path) ;
  end
end  % function



function result = native_path_from_wsl_core(input_path)
  % Convert a single platform-native path to a WSL path.
  if ispc() ,
    path_elements = strsplit(input_path, '/') ;
    element_count = numel(path_elements) ;
    if element_count==0
      error('Internal error: Input path should have at least one element') ;
    elseif element_count<3
      error('Input path ''%s'' cannot be converted to native format', input_path) ;
    end
    % input_path has at least three elements
    if isempty(path_elements{1}) && strcmp(path_elements{2}, 'mnt')      
      drive = path_elements{3} ;
      if numel(drive) ~= 1 || ~all(isletter(drive)) 
        error('To convert to native path, path must start with /mnt/<drive letter>.  Path ''%s'' does not.', input_path) ;
      end
      new_path_rest_elements = path_elements(4:end) ;
      new_path_rest = slash_out(new_path_rest_elements) ;
      new_path = sprintf('%s:/%s', upper(drive), new_path_rest) ;
      result = strrep(new_path, '/', '\') ;
    else
      % Does not start with '/mnt'
      error('To convert to native path, path must start with /mnt/<drive letter>.  Path ''%s'' does not.', input_path) ;
    end
  else
    % On Linux, no translation needed
    result = input_path ;
  end   
end  % function
