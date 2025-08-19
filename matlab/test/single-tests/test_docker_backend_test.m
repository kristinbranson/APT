function test_docker_backend_test()
  % Test the docker backend configuration using the new MVC architecture
  
  % Create labeler and controller
  [labeler, controller] = StartAPT();
  oc = onCleanup(@()(delete(controller))); % This will also delete the labeler
  
  % Define a config so we can create a project
  cfg = simpleMAProjectConfigForTesting();
  
  % Create a new project
  labeler.projNew(cfg) ;

  % Set the labeler to silent mode for batch operation
  labeler.silent = true;
  
  % Set backend to docker
  labeler.set_backend_property('type', 'docker');
  
  % Test the backend configuration
  % This will create a BackendTestController and call testDockerBackendConfig_()
  controller.cbkTrackerBackendTest();
  
  % Test that there's something in the test text
  test_text = labeler.backend.testText() ;
  if numel(test_text) < 5
    error('The test text is too short to be correct') ;
  end

  % If we get here without error, the test passed
  fprintf('Docker backend configuration test completed successfully\n');
end  % function
