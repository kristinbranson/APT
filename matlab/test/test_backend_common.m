function test_backend_common(backend_type)
  % Common function for testing backend configurations using the new MVC architecture
  %
  % Args:
  %   backend_type: String specifying the backend type ('conda', 'docker', 'bsub', 'aws')
  
  % Skip test on Windows for conda backend as it is not supported
  if strcmp(backend_type, 'conda') && ispc()
    warning('conda backend is not supported on Windows, so test always passes on Windows');
    return
  end
  
  % Create labeler and controller
  [labeler, controller] = StartAPT();
  oc = onCleanup(@()(delete(controller))); % This will also delete the labeler
  
  % Define a config so we can create a project
  cfg = simpleMAProjectConfigForTesting();
  
  % Create a new project
  labeler.projNew(cfg) ;

  % Set the labeler to silent mode for batch operation
  labeler.silent = true;
  
  % Set backend to the specified type
  labeler.set_backend_property('type', backend_type);
  
  % Test the backend configuration
  % This will create a BackendTestController and call the appropriate test method
  controller.cbkTrackerBackendTest();
  
  % Test that there's something in the test text
  test_text = labeler.backend.testText() ;
  if numel(test_text) < 5
    error('The test text is too short to be correct') ;
  end

  % If we get here without error, the test passed
  fprintf('%s backend configuration test completed successfully\n', backend_type);
end  % function