function test_train_spawn_failure_leaves_no_monitor()
  % Regression test for the bug where a failed job spawn (e.g. the bsub command
  % that submits the training job returns a nonzero exit code -- as happens when
  % a required -P project option is missing) leaves the DeepTracker with a
  % background training monitor and poller already created.  The controller
  % creates the Training Monitor window when the monitor is created, so the
  % window is left on screen showing a stale, green "Initializing..." status,
  % even though the job never actually started.
  %
  % Desired behavior (what this test asserts): when the spawn fails, the
  % DeepTracker should NOT be left holding a background training monitor or
  % poller, so there is no stale monitor window to confuse the user.
  %
  % The spawn failure is induced deterministically, without a real cluster, via
  % the DLBackEndClass.isSpawnForcedToFail_ test hook.

  linux_project_file_path = '/groups/branson/bransonlab/apt/unittest/four-points-testing-2025-04-11-with-rois-added-and-fewer-smaller-avi-movies.lbl' ;
  [project_file_path, replace_path] = localize_test_project_path(linux_project_file_path) ;
  backend = docker_unless_janelia_cluster_then_conda() ;  % Should work on Linux or Windows
  backend_params = synthesize_backend_params(backend) ;
  tester = LabelerProjectTester(project_file_path, 'replace_path', replace_path) ;
  oc = onCleanup(@()(delete(tester))) ;
  labeler = tester.labeler ;

  % Set up the backend and training parameters the way LabelerProjectTester.test_training
  % does, but do not start training yet.
  tester.set_backend_params_(backend, backend_params) ;
  niters = 200 ;
  sPrm = labeler.trackGetTrainingParams() ;
  sPrm = structsetleaf(sPrm, struct('dl_steps', {niters}), 'verbose', true) ;
  labeler.trackSetTrainingParams(sPrm) ;

  % Arrange for the job spawn to fail, mimicking a failed bsub submission.
  labeler.trackDLBackEnd.isSpawnForcedToFail_ = true ;

  % Attempt to train.  The forced spawn failure should make training error out.
  didTrainingThrow = false ;
  try
    labeler.train() ;
  catch
    didTrainingThrow = true ;
  end
  assert(didTrainingThrow, ...
         'Training should have thrown an error due to the forced spawn failure') ;

  % The actual regression checks: no background training monitor or poller should
  % have been left behind by the failed spawn.
  tracker = labeler.tracker ;
  assert(isempty(tracker.bgTrnMonitor), ...
         'A background training monitor was left behind after a failed job spawn') ;
  assert(isempty(tracker.bgTrainPoller), ...
         'A background training poller was left behind after a failed job spawn') ;
end  % function
