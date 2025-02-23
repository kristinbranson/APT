function test_training(project_file_path, varargin)
  % Test tracking in the .lbl file project_file_path.  Optional arguments allow
  % caller to change algorithm name, backedn from those specified in the .lbl
  % file.
  [algo_name, backend_type, backend_params, training_params, niters] = ...
    myparse(varargin,...
            'algo_name','',...
            'backend','',...
            'backend_params',struct(), ...
            'training_params', [], ...
            'niters', 1000) ;

  % Launch the Labeler, LabelerController
  [labeler, controller] = StartAPT() ;
  cleaner = onCleanup(@()(delete(controller))) ;  % this will delete labeler too

  % Set the labeler to silent mode for batch operation
  labeler.silent = true;

  % Load the project
  labeler.projLoadGUI(project_file_path) ;

  if ~isempty(algo_name) ,
    labeler.trackMakeOldTrackerCurrentByName(algo_name) ;
  end
  if ~isempty(backend_type),
    labeler.set_backend_property('type', backend_type);
    name_from_field_index = fieldnames(backend_params) ;
    for field_index = 1 : numel(name_from_field_index) ,
      name = name_from_field_index{field_index} ;
      value = backend_params.(name) ;
      labeller.set_backend_property(name, value) ;
    end
  end  
  if isempty(training_params)
    training_params = struct('dl_steps', {niters}) ;
  else
    training_params.dl_steps = niters ;
  end
  sPrm = labeler.trackGetTrainingParams();
  sPrm = structsetleaf(sPrm,training_params,'verbose',true);
  labeler.trackSetTrainingParams(sPrm);
    
  labeler.train() ;

  % block, waiting for training to finish
  pause(2) ;
  while labeler.bgTrnIsRunning
    pause(10) ;
  end
  pause(10) ;

  % Check that training happened
  if labeler.tracker.trnLastDMC.iterCurr<niters ,
    error('Failed to complete all training iterations') ;
  end
end  % function
