function test_conda_backend_test()
  % Test the conda backend configuration using the new MVC architecture
  
  % Skip test on Windows as conda backend is not supported
  if ispc() ,
    warning('conda backend is not supported on Windows, so %s always passes on Windows', mfilename());
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
  
  % Set backend to conda
  labeler.set_backend_property('type', 'conda');
  
  % Test the backend configuration
  % This will create a BackendTestController and call testCondaBackendConfig_()
  controller.cbkTrackerBackendTest();
  
  % Test that there's something in the test text
  test_text = labeler.backend.testText() ;
  if numel(test_text) < 5
    error('The test text is too short to be correct') ;
  end

  % If we get here without error, the test passed
  fprintf('Conda backend configuration test completed successfully\n');
end  % function