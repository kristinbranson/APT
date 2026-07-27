function [project_file_path, replace_path] = localize_test_project_path(linux_project_file_path)
  % Map a Linux-style test project path to the local platform's path,
  % and return the corresponding replace_path for projLoad.  On Linux
  % the path passes through unchanged and replace_path is empty.  On
  % Windows, /groups/branson/bransonlab is mapped to Z:.
  if ispc()
    project_file_path = strrep(linux_project_file_path, '/groups/branson/bransonlab', 'Z:') ;
    replace_path = { '/groups/branson/bransonlab', 'Z:' } ;
  else
    project_file_path = linux_project_file_path ;
    replace_path = [] ;
  end
end  % function
