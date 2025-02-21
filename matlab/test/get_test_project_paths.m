function [bransonlab_path, unittest_dir_path] = get_test_project_paths()  
  if ispc() ,
    % We assume /groups/branson/bransonlab is mounted on Z:
    bransonlab_path = 'Z:/' ;
  else
    bransonlab_path = '/groups/branson/bransonlab' ;
  end
  unittest_dir_path = fullfile(bransonlab_path, 'apt/unittest') ;
end