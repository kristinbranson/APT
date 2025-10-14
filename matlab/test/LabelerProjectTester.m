classdef LabelerProjectTester < handle
  % For testing APT in the context of an already-existing .lbl file.

  properties
    labeler  % a Labeler object, or empty
    controller  % a Controller object, or empty
  end
  
  methods    
    function obj = LabelerProjectTester(project_file_path, varargin)
      replace_path = ...
        myparse(varargin,...
                'replace_path',{'',''}) ;
        % replace_path is useful when running on Windows, where paths to movies may
        % differ from what they are on Linux.
      [obj.labeler, obj.controller] = StartAPT() ;
      % Set the labeler to silent mode for batch operation
      obj.labeler.silent = true ;
      % Load the named project
      obj.labeler.projLoadGUI(project_file_path, 'replace_path', replace_path) ;
    end
    
    function delete(obj)
      if ~isempty(obj.controller) && isvalid(obj.controller) ,
        delete(obj.controller) ;
      end
      if ~isempty(obj.labeler) && isvalid(obj.labeler) ,
        delete(obj.labeler) ;
      end      
    end

    function algo_name = test_training(obj, varargin)
      % Test tracking in obj.labeler.  Optional arguments allow caller to change
      % algorithm name, backedn from those specified in the .lbl file.
      [algo_spec, backend_type_as_string, backend_params, training_params, niters] = ...
        myparse(varargin,...
                'algo_spec',{},...
                'backend','',...
                'backend_params',struct(), ...
                'training_params', [], ...
                'niters', 200) ;
    
      labeler = obj.labeler ;
      % controller = obj.controller ;
      
      % Set things up for training
      if ~isempty(algo_spec) ,
        labeler.trackMakeNewTrackerGivenNetTypes(algo_spec) ;
        % if ischar(algo_spec) ,
        %   desired_algo_name = algo_spec ;
        %   labeler.trackMakeNewTrackerGivenAlgoName(desired_algo_name) ;          
        % else
        %   labeler.trackMakeNewTrackerGivenAlgoName(algo_spec{:}) ;
        % end
      end
      algo_name = labeler.tracker.algorithmName ;
      % % HACK START
      % backend_type_as_string = 'conda' ;
      % backend_params = { 'condaEnv', 'apt-2025-04' }  ;
      % % HACK END
      obj.set_backend_params_(backend_type_as_string, backend_params) ;
      if isempty(training_params)
        training_params = struct('dl_steps', {niters}) ;
      else
        training_params.dl_steps = niters ;
      end
      sPrm = labeler.trackGetTrainingParams();
      sPrm = structsetleaf(sPrm,training_params,'verbose',true);
      labeler.trackSetTrainingParams(sPrm);
        
      % Actually train
      labeler.train() ;
    
      % block, waiting for training to finish
      pause(2) ;
      while labeler.bgTrnIsRunning
        pause(10) ;
      end
      pause(10) ;
    
      % Bring any remote artifacts back to frontend
      labeler.downloadProjectCacheIfNeeded() ;

      % Check that training happened.
      % After return, caller can check other aspects of obj.labeler if desired.
      if labeler.lastTrainEndCause ~= EndCause.complete 
        error('Training did not complete successfully') ;
      end      
      if any(isnan(labeler.tracker.trnLastDMC.iterCurr)) || any(labeler.tracker.trnLastDMC.iterCurr < niters) ,
        error('Failed to complete all training iterations') ;
      end
    end  % function
    
    function test_tracking(obj, varargin)
      % Test tracking in the .lbl file specified during LabelerTester construction.
      % Optional arguments allow caller to change algorithm name, backend from those
      % specified in the .lbl file.  Will throw if something goes wrong during
      % tracking.  Will perform some basic checks that tracking succeeded.
      [algo_name, backend_type_as_string, backend_params, ~, ~] = ...
        myparse(varargin,...
                'algo_name',{},...
                'backend','',...
                'backend_params',struct(), ...
                'training_params', [], ...
                'niters', []);
      % Note that training_params and niters are unused, but we want to accept them
      % without warning if they are passed in, to simplify argument-handling in
      % test_training_then_tracking().

      labeler = obj.labeler ;
      % controller = obj.controller ;
    
      % Set things up
      if ~isempty(algo_name) ,
        labeler.trackMakeExistingTrackerCurrentGivenAlgoName(algo_name) ;
      end
      % % HACK START
      % backend_type_as_string = 'conda' ;
      % backend_params = { 'condaEnv', 'apt-2025-04' }  ;
      % % HACK END      
      obj.set_backend_params_(backend_type_as_string, backend_params) ;

      % Track
      labeler.track() ;
    
      % block, waiting for tracking to finish
      pause(2) ;
      while labeler.bgTrkIsRunning
        pause(10) ;
      end
      pause(10) ;

      % Bring any remote artifacts back to frontend
      labeler.downloadProjectCacheIfNeeded() ;
      
      % Perform some tests that tracking worked
      % After return, caller can check other aspects of obj.labeler if desired.
      if labeler.lastTrackEndCause ~= EndCause.complete
        error('Tracking did not complete successfully') ;
      end
      if isempty(labeler.tracker.trkP)
        error('labeler.tracker.trkP is empty---it should be nonempty after tracking') ;
      end
      if ~isa(labeler.tracker.trkP, 'TrkFile') ,
        error('labeler.tracker.trkP is not of class TrkFile after tracking') ;
      end
      if ~iscell(labeler.tracker.trkP.pTrk) 
        error('labeler.tracker.trkP.pTrk is not of class cell after tracking') ;
      end
      if isempty(labeler.tracker.trkP.pTrk)
        error('labeler.tracker.trkP.pTrk is empty---it should be nonempty after tracking') ;
      end      
    end  % function

    function test_training_then_tracking(obj, varargin)
      % Need to take care that we request the full algorithm name when tracking.
      algo_name = obj.test_training(varargin{:}) ;      
      tracking_args = remove_pair_from_key_value_list_if_present(varargin, 'algo_spec') ;
      obj.test_tracking(tracking_args{:}, 'algo_name', algo_name) ;      
    end
    
    function test_gtcompute(obj, varargin)
      [backend_type_as_string, backend_params] = ...
       myparse(varargin, ...
               'backend','', ...
               'backend_params',struct());

      % Clear any preexisting GT performance table
      obj.labeler.gtClearGTPerformanceTable() ;

      % Make sure the GT table has been cleared
      if ~isempty(obj.labeler.gtTblRes) ,
        error('labeler.gtTblRes is nonempty---it should be empty after clearing the GT performance table') ;
      end
      
      % Compute the GT performance
      obj.set_backend_params_(backend_type_as_string,backend_params);
      obj.labeler.gtSetGTMode(true) ;
      obj.labeler.gtComputeGTPerformance() ;

      % Wait for GTing to finish
      pause(2);
      while obj.labeler.bgTrkIsRunning
        pause(10);
      end
      pause(10);

      % % Bring any remote artifacts back to frontend
      % labeler.rehomeProjectCacheIfNeeded() ;      

      % Make sure tracking was successful
      if obj.labeler.lastTrackEndCause ~= EndCause.complete
        error('Tracking for GT did not complete successfully') ;
      end
      
      % Make sure the GT table has been generated
      if isempty(obj.labeler.gtTblRes) ,
        error('labeler.gtTblRes is empty---it should be nonempty after computing GT performance') ;
      end
    end  % function
    
    function set_backend_params_(obj, backend_type_as_string, raw_backend_params)
      % raw_backend_params: structure (or cell array) containing name-value pairs to be set on the backend
      labeler = obj.labeler;
      if ~isempty(backend_type_as_string),
        labeler.set_backend_property('type', backend_type_as_string);
      end  
      if ~isempty(raw_backend_params) ,
        if iscell(raw_backend_params) ,
          backend_params = struct_from_key_value_list(raw_backend_params) ;
        elseif isstruct(raw_backend_params)
          backend_params = raw_backend_params ;
        else
          error('raw_backend_params must be a cell array or a struct') ;
        end
        % Set the backend parameters
        name_from_field_index = fieldnames(backend_params) ;
        for field_index = 1 : numel(name_from_field_index) ,
          name = name_from_field_index{field_index} ;
          value = backend_params.(name) ;
          labeler.set_backend_property(name, value) ;
        end
      end
    end  % function
    
  end  % methods
end  % classdef
