function build(varargin)
  % Build an APT executable.

  % Parse optional args
  [do_run_test_after, do_compile] = ...
  myparse( varargin ...
         , 'do_run_test_after', false ...
         , 'do_compile', true ...
         ) ;

  % Define things needed for building or running
  main_m_file_name = 'APT_deployed.m' ;
  executable_folder_path = fullfile(APT.Root, 'compiled') ;
  executable_file_base_name = 'APT_deployed' ;

  if do_compile
    % Define the main arguments
    args_0 = ...
      { '-v' ...
      , '-m', main_m_file_name ...
      , '-d', executable_folder_path ...
      , '-o', executable_file_base_name ...
      , '-R', '-logfile,APT_deployed.log' ...
      , '-a', fullfile(APT.Root, 'matlab') ...
      , '-a', fullfile(APT.Root, 'gfx') ...
      , '-a', fullfile(APT.Root, 'deepnet') ...
      , '-a', fullfile(APT.Root, 'java') ...
      , '-a', fullfile(APT.Root, 'docs') ... 
      } ;
        
    % There's a bunch of yaml files needed that describe model parameters,
    % so collect all those up and add them to the args list to be passed to mcc
    param_file_relative_paths = row(struct2cell(APTParameters.paramFileSpecs())) ;
    param_yaml_paths = cellfun(@(relative_path)(fullfile(APT.Root, 'matlab', relative_path)), param_file_relative_paths, 'UniformOutput', false) ;
    nested_param_file_args = cellfun(@(path)({ '-a', path }), param_yaml_paths, 'UniformOutput', false) ;
    param_file_args = flatten_row_cell_array(nested_param_file_args) ;
    args_1 = horzcat(args_0, param_file_args) ;
  
    % Some m-files are called in a way that is not apparent to static analysis
    % (e.g. via feval()).  So we include them using -m arguments so they're
    % available in the exectuable.
    clf_folder_path = fullfile(APT.Root, 'matlab/compute_landmark_features') ;  % clf == compute_landmark_features
    clf_file_names = simple_dir(fullfile(clf_folder_path, '*.m')) ;
    clf_file_paths = cellfun(@(name)(fullfile(clf_folder_path, name)), clf_file_names, 'UniformOutput', false) ;
    nested_m_file_args = cellfun(@(path)({ '-m', path }), clf_file_paths, 'UniformOutput', false) ;
    m_file_args = flatten_row_cell_array(nested_m_file_args) ;
    args = horzcat(args_1, m_file_args) ;
  
    % Finally call mcc()
    tic_id = tic() ;
    mcc(args{:}) ;
    time_to_compile = toc(tic_id) ;
    fprintf(sprintf('-- Time to compile %.2f --\n',time_to_compile));
  end  % if do_compile

  % If called for, run a test of the executable
  if do_run_test_after
    if ispc()
      executable_file_name = sprintf('%s.exe', executable_file_base_name) ;
    else
      executable_file_name = executable_file_base_name ;
    end
    if ispc() 
      export_cmd = '' ;
    else
      if strmp(get_user_name(), 'kabram')
        export_cmd = 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/groups/branson/home/kabram/bransonlab/MCR/v911/runtime/glnxa64:/groups/branson/home/kabram/bransonlab/MCR/v911/bin/glnxa64:/groups/branson/home/kabram/bransonlab/MCR/v911/sys/os/glnxa64:/groups/branson/home/kabram/bransonlab/MCR/v911/extern/bin/glnxa64' ;
      else
        export_cmd = '' ;
      end
    end
    executable_path = fullfile(executable_folder_path, executable_file_name) ;
    test_name = 'AR_GRONe_SA_tracking' ;
    main_cmd = sprintf('%s --test %s', executable_path, test_name) ;
    if ~isempty(export_cmd)
      cmd = sprintf('%s ; %s', export_cmd, main_cmd) ;
    else      
      cmd = main_cmd ;
    end
    system(cmd) ;
  end  % if
end  % function
