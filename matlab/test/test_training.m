function test_training(project_file_path, varargin)
  % Test tracking in the .lbl file project_file_path.  Optional arguments allow
  % caller to change algorithm name, backedn from those specified in the .lbl
  % file.
  [algo_name, backend_type, backend_params] = ...
    myparse(varargin,...
            'algo_name','',...
            'backend','',...
            'backend_params',struct());

  [labeler, controller] = ...
    StartAPT('projfile', project_file_path) ;
  cleaner = onCleanup(@()(delete(controller))) ;  % this will delete labeler too

  if ~isempty(algo_name) ,
    labeler.trackMakeOldTrackerCurrentByName(algo_name) ;
  end
  if ~isempty(backend_type),
    labeller.set_backend_property('type', backend_type);
    name_from_field_index = fieldnames(backend_params) ;
    for field_index = 1 : numel(name_from_field_index) ,
      name = name_from_field_index{field_index} ;
      value = backend_params.(name) ;
      labeller.set_backend_property(name, value) ;
    end
  end  
  labeler.track() ;

  % block, waiting for tracking to finish
  pause(2) ;
  while labeler.bgTrkIsRunning
    pause(10) ;
  end
  pause(10) ;
end  % function
