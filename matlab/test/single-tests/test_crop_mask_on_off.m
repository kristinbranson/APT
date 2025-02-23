function test_crop_mask_on_off()
  for doCrop = 0:1 
    for doMask = 0:1
      test_crop_mask_on_off_helper(doCrop, doMask) ;
    end
  end
end



function test_crop_mask_on_off_helper(doCrop, doMask)
  % Helper method, not itself a test method
  setup_params = get_setup_parameters() ;
  [labeler, controller] = ...
    StartAPT('projfile', ...
               '/groups/branson/bransonlab/apt/unittest/four-points-testing-2024-11-19-with-rois-added-and-fewer-smaller-movies.lbl') ;
  cleaner = onCleanup(@()(delete(controller))) ;  % this will delete labeler too

  % Set the algo
  labeler.trackMakeNewTrackerCurrentByName('magrone') ;

  % Set the backend type
  labeler.set_backend_property('type', 'bsub');

  % Set backend properties
  labeler.set_backend_property('jrcgpuqueue', setup_params.jrcgpuqueue) ;
  labeler.set_backend_property('jrcnslots', setup_params.jrcnslots) ;
  labeler.set_backend_property('jrcAdditionalBsubArgs', setup_params.jrcAdditionalBsubArgs) ;

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



function result = get_setup_parameters()
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
