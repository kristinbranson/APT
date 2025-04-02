function build()
  % Build a StartAPT executable.
  this_script_path = mfilename('fullpath') ;
  apt_root_folder = fileparts(this_script_path) ;  % Should be the project root
  executable_folder_path = fullfile(apt_root_folder, 'executable') ;
  fixed_args = ...
    { '-v', '-m', 'StartAPT.m', '-d', executable_folder_path, ...
      '-a', fullfile(apt_root_folder, 'matlab', 'createLabelerMainFigure_assets.mat'), ...
      '-a', fullfile(apt_root_folder, 'gfx'), ...
      '-a', fullfile(apt_root_folder, 'matlab', InfoTimeline.TLPROPFILESTR), ...
      '-a', fullfile(apt_root_folder, 'matlab/+yaml/external/snakeyaml-1.9.jar'), ...
      '-a', fullfile(apt_root_folder, 'matlab/JavaTableWrapper/+uiextras/+jTable/UIExtrasTable.jar'), ...
      '-a', fullfile(apt_root_folder, 'matlab', 'trackers', 'dt', 'nets.yaml'), ...
      '-a', fullfile(apt_root_folder, 'matlab', 'config.default.yaml') } ;

  % There's a bunch of yaml files needed that describe models or something, 
  % so collect all those up and add them to the args list to be passed to mcc
  annoying_specs = row(struct2cell(APTParameters.paramFileSpecs())) ;
  function result = yaml_path_from_spec(spec)
    result = fullfile(apt_root_folder, 'matlab', spec) ;  
  end
  annoying_yaml_paths = cellfun(@yaml_path_from_spec, annoying_specs, 'UniformOutput', false) ;
  function result = option_from_yaml_path(path)
    result = { '-a', path } ;  
  end
  annoying_nested_options = cellfun(@option_from_yaml_path, annoying_yaml_paths, 'UniformOutput', false) ;
  annoying_options = flatten_row_cell_array(annoying_nested_options) ;
  args = horzcat(fixed_args, annoying_options) ;

  % Finally call mcc()
  mcc(args{:}) ;
end


