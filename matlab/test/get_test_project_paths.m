function [bransonlab_path, unittest_dir_path, replace_path] = get_test_project_paths()  
  if ispc() ,
    % We assume /groups/branson/bransonlab is mounted on Z:
    bransonlab_path = 'Z:/' ;
    replace_path = { '/groups/branson/bransonlab', bransonlab_path } ;
  else
    bransonlab_path = '/groups/branson/bransonlab' ;
    replace_path = [] ;
  end
  unittest_dir_path = fullfile(bransonlab_path, 'apt/unittest') ;
end
