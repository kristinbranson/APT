function [parent_path, name] = linux_fileparts2(path)
  % Like filepath(), but returns only two outputs: the parent path and the
  % filename (*with* extension).  And this works on linux-style paths regardless
  % of the current platform.
  char_index_from_slash_index = strfind(path, '/') ;
  if isempty(char_index_from_slash_index)
    % if no slashes at all, parent path is empty.
    % This also covers case where path is empty.
    parent_path = '' ;
    name = path ;
  else
    % Get the index of the last slash in the path
    char_index_of_last_slash = char_index_from_slash_index(end) ;    
    if char_index_of_last_slash==1
      % If the path starts with a slash, want to make the parent path nonempty, to
      % be consistent with fileparts() on Linux
      parent_path = '/' ;
      name = path(2:end) ;      
    else
      % If the last slash is *not* the first character, then we split at the last
      % slash, with the last slash not in either part.
      parent_path = path(1:char_index_of_last_slash-1) ;
      name = path(char_index_of_last_slash+1:end) ;
    end
  end
end
