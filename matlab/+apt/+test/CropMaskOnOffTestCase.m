classdef CropMaskOnOffTestCase < matlab.unittest.TestCase
  methods (Static)
    function result = getSetupParams()
      if strcmp(get_user_name(), 'taylora') ,
        jrcAdditionalBsubArgs = '-P scicompsoft' ;
      else
        jrcAdditionalBsubArgs = '' ;
      end
      result = ...
        struct('jrcgpuqueue',{'gpu_a100'}, ...
               'jrcnslots',{4}, ...
               'jrcAdditionalBsubArgs',{jrcAdditionalBsubArgs}) ;
    end  % function
  end  % methods (Static)
  
  methods (Test)
    function cropOffMaskOffTest(obj)
      % ALTTODO: This works, and passes, but note that I don't turn off cropping or masking yet!
      setup_params = apt.test.CropMaskOnOffTestCase.getSetupParams() ;
      [labeler, controller] = ...
        StartAPT('projfile', '/groups/branson/bransonlab/taylora/apt/four-points/four-points-testing-2024-11-19-with-gt-and-rois-added.lbl');
      cleaner = onCleanup(@()(delete(controller))) ;  % this will delete labeler too

      % Set the algo
      labeler.trackSetCurrentTrackerByName('magrone') ;

      % Set the backend
      labeler.trackSetDLBackendType('bsub');

      % Set backend properties
      labeler.set_backend_property('jrcgpuqueue', setup_params.jrcgpuqueue) ;
      labeler.set_backend_property('jrcnslots', setup_params.jrcnslots) ;
      labeler.set_backend_property('jrcAdditionalBsubArgs', setup_params.jrcAdditionalBsubArgs) ;

      % Modify the training parameters
      original_training_params = labeler.trackGetParams();      
      new_training_params = struct('dl_steps', {1000}) ;  % scalar struct
      training_params = structsetleaf(original_training_params, ...
                                      new_training_params, ...
                                      'verbose', true) ;
      labeler.trackSetParams(training_params);

      % Want labeler to do its thing quietly
      labeler.silent = true;

      % Train!
      labeler.train();      

      % block while training        
      pause(2);
      while labeler.bgTrnIsRunning
        pause(10);
      end
      pause(10);
      % blocking done

      % Do verification
      obj.verifyTrue(labeler.tracker.trnLastDMC.iterCurr>=1000, 'Failed to complete all training iterations') ;
    end  % function    
  end  % methods (Test)
end  % classdef
