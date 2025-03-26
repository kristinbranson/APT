function [bransonlab_path, unittest_dir_path, replace_path] = get_test_project_paths()  
  if ispc() ,
    % We assume a copy of /groups/branson/bransonlab/apt/unittest
    % is present locally.
    user_name = get_user_name() ;
    bransonlab_path = sprintf('C:/Users/%s/bransonlab', user_name) ;
    replace_path = { '/groups/branson/bransonlab', bransonlab_path } ;
  else
    bransonlab_path = '/groups/branson/bransonlab' ;
    replace_path = [] ;
  end
  unittest_dir_path = fullfile(bransonlab_path, 'apt/unittest') ;
end
