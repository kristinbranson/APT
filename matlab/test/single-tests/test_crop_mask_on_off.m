function test_crop_mask_on_off()
  for doCrop = 0:1 
    for doMask = 0:1
      test_crop_mask_on_off_helper(doCrop, doMask) ;
    end
  end
end



function test_crop_mask_on_off_helper(doCrop, doMask)
  % Helper method, not itself a test method
  backend = 'docker' ;  % Should work on Linux or Windows
  backend_params = synthesize_backend_params(backend) ;

  [labeler, controller] = ...
    StartAPT('projfile', ...
               '/groups/branson/bransonlab/apt/unittest/four-points-testing-2024-11-19-with-rois-added-and-fewer-smaller-movies.lbl') ;
  cleaner = onCleanup(@()(delete(controller))) ;  % this will delete labeler too

  % Set the algo
  labeler.trackMakeNewTrackerCurrentByName('magrone') ;

  % Set the backend type
  labeler.set_backend_property('type', backend);

  % Set backend properties
  % labeler.set_backend_property('jrcgpuqueue', backend_params.jrcgpuqueue) ;
  % labeler.set_backend_property('jrcnslots', backend_params.jrcnslots) ;
  % labeler.set_backend_property('jrcAdditionalBsubArgs', backend_params.jrcAdditionalBsubArgs) ;
  if ~isempty(backend_params) ,
    backend_params_struct = struct_from_key_value_list(backend_params) ;
    % Set the backend parameters
    name_from_field_index = fieldnames(backend_params_struct) ;
    for field_index = 1 : numel(name_from_field_index) ,
      name = name_from_field_index{field_index} ;
      value = backend_params_struct.(name) ;
      labeler.set_backend_property(name, value) ;
    end
  end

  % Modify the training parameters
  original_training_params = labeler.trackGetTrainingParams();
  iterationCount = 200 ;
  new_training_params = struct('dl_steps', {iterationCount}, 'multi_crop_ims', {doCrop}, 'multi_loss_mask', {doMask}) ;  % scalar struct
  training_params = structsetleaf(original_training_params, ...
                                  new_training_params, ...
                                  'verbose', true) ;
  labeler.trackSetTrainingParams(training_params);

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
  if ~labeler.tracker.trnLastDMC.iterCurr>=iterationCount ,
    error('Failed to complete all training iterations') ;
  end
end  % function    


