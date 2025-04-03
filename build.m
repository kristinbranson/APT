function build()
  % Build a StartAPT executable.
  executable_folder_path = fullfile(APT.Root, 'executable') ;
  args_0 = ...
    { '-v', '-m', 'StartAPT.m', '-d', executable_folder_path, ...
      '-a', fullfile(APT.Root, 'matlab', 'createLabelerMainFigure_assets.mat'), ...
      '-a', fullfile(APT.Root, 'gfx'), ...
      '-a', fullfile(APT.Root, 'matlab', InfoTimeline.TLPROPFILESTR), ...
      '-a', fullfile(APT.Root, 'matlab/+yaml/external/snakeyaml-1.9.jar'), ...
      '-a', fullfile(APT.Root, 'matlab/JavaTableWrapper/+uiextras/+jTable/UIExtrasTable.jar'), ...
      '-a', fullfile(APT.Root, 'matlab', 'trackers', 'dt', 'nets.yaml'), ...
      '-a', fullfile(APT.Root, 'deepnet'), ...
      '-a', fullfile(APT.Root, 'matlab', 'config.default.yaml') } ;

  % There's a bunch of yaml files needed that describe model parameters,
  % so collect all those up and add them to the args list to be passed to mcc
  param_file_relative_paths = row(struct2cell(APTParameters.paramFileSpecs())) ;
  function result = yaml_path_from_relative_path(relative_path)
    result = fullfile(APT.Root, 'matlab', relative_path) ;  
  end
  param_yaml_paths = cellfun(@yaml_path_from_relative_path, param_file_relative_paths, 'UniformOutput', false) ;
  function result = args_from_yaml_path(path)
    result = { '-a', path } ;  
  end
  nested_param_file_args = cellfun(@args_from_yaml_path, param_yaml_paths, 'UniformOutput', false) ;
  param_file_args = flatten_row_cell_array(nested_param_file_args) ;
  args_1 = horzcat(args_0, param_file_args) ;

  % Some m-files are called in a way that is not apparent to static analysis
  % (e.g. via feval()).  So we include them using -m arguments so they're
  % available in the exectuable.
  clf_folder_path = fullfile(APT.Root, 'matlab/compute_landmark_features') ;  % clf == compute_landmark_features
  clf_file_names = simple_dir(fullfile(clf_folder_path, '*.m')) ;
  clf_file_paths = cellfun(@(name)(fullfile(clf_folder_path, name)), clf_file_names, 'UniformOutput', false) ;
  function result = dash_m_args_from_path(path)
    result = { '-m', path } ;  
  end
  nested_m_file_args = cellfun(@dash_m_args_from_path, clf_file_paths, 'UniformOutput', false) ;
  m_file_args = flatten_row_cell_array(nested_m_file_args) ;
  args = horzcat(args_1, m_file_args) ;

  % Finally call mcc()
  mcc(args{:}) ;
end


