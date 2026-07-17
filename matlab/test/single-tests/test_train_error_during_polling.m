function test_train_error_during_polling()
  % Verifies the DeepTracker.forcedPollErrorIndex_ test hook.  Setting it to a
  % positive integer n makes the training bout error out (as if a mid-run error
  % had been detected) upon obtaining its n-th poll result, without any real
  % training failure.  Here we set it to 1, so the bout errors out on the very
  % first poll result, and confirm that the bout ends with EndCause.error and
  % that background training is no longer running.
  %
  % Unlike test_train_spawn_failure_leaves_no_monitor, the job spawn itself
  % succeeds here; the error is injected later, during background polling.

  [~, unittest_dir_path, replace_path] = get_test_project_paths() ;
  project_file_path = fullfile(unittest_dir_path, 'four-points-testing-2025-04-11-with-rois-added-and-fewer-smaller-avi-movies.lbl') ;
  backend = docker_unless_janelia_cluster_then_conda() ;  % Should work on Linux or Windows
  backend_params = synthesize_backend_params(backend) ;
  tester = LabelerProjectTester(project_file_path, 'replace_path', replace_path) ;
  oc = onCleanup(@()(delete(tester))) ;
  labeler = tester.labeler ;

  % Set up the backend and training parameters the way LabelerProjectTester.test_training
  % does, but do not start training yet.
  tester.set_backend_params_(backend, backend_params) ;
  niters = 1000 ;
    % Enough iterations that the job will not finish before the first poll, so the
    % forced error is what ends the bout.
  sPrm = labeler.trackGetTrainingParams() ;
  sPrm = structsetleaf(sPrm, struct('dl_steps', {niters}), 'verbose', true) ;
  labeler.trackSetTrainingParams(sPrm) ;

  % Arrange for the bout to error out on the first poll result.
  labeler.tracker.forcedPollErrorIndex_ = 1 ;

  % Start training.  The spawn should succeed; the forced poll error ends the
  % bout shortly afterward, in the background.
  labeler.train() ;

  % Wait for the (forced) error to end the bout.
  pause(2) ;
  maximumWaitTime = 300 ;  % seconds
  ticId = tic() ;
  while labeler.bgTrnIsRunning && toc(ticId) < maximumWaitTime ,
    pause(5) ;
  end

  % The bout should have ended (background training no longer running) ...
  assert(~labeler.bgTrnIsRunning, ...
         'Background training was still running %g seconds after the forced poll error', maximumWaitTime) ;
  % ... and it should have ended because of an error.
  assert(labeler.lastTrainEndCause == EndCause.error, ...
         'Training bout should have ended with EndCause.error, but ended with %s', char(labeler.lastTrainEndCause)) ;
end  % function
