function test_track_spawn_failure_leaves_no_monitor()
  % Regression test, analogous to test_train_spawn_failure_leaves_no_monitor but
  % for the tracking path.  A failed job spawn (e.g. the bsub command that
  % submits the tracking job returns a nonzero exit code -- as happens when a
  % required -P project option is missing) used to leave the DeepTracker with a
  % background tracking monitor and poller already created.  The controller
  % creates the Tracking Monitor window when the monitor is created, so the
  % window would be left on screen showing a stale, green "Initializing..."
  % status even though the job never actually started.
  %
  % Desired behavior (what this test asserts): when the spawn fails, the
  % DeepTracker should NOT be left holding a background tracking monitor or
  % poller, so there is no stale monitor window to confuse the user.
  %
  % The spawn failure is induced deterministically, without a real cluster, via
  % the DLBackEndClass.isSpawnForcedToFail_ test hook.
  %
  % This covers the normal tracking path (DeepTracker.trkSpawnCore_).  The
  % ID-linking path (DeepTracker.idlinkSpawn_) is not covered here: it is only
  % reached from a background-monitor callback *after* a detection track bout has
  % already spawned successfully, so it cannot be exercised by the single
  % isSpawnForcedToFail_ hook (which would fail the detection spawn first).

  linux_project_file_path = ...
    ['/groups/branson/bransonlab/apt/unittest/' ...
     'multitarget_bubble_training_20210523_allGT_AR_MAAPT_grone2_UT_resaved_3_lightly_trained.lbl'] ;
  [project_file_path, replace_path] = localize_test_project_path(linux_project_file_path) ;
  backend = docker_unless_janelia_cluster_then_conda() ;  % Should work on Linux or Windows
  backend_params = synthesize_backend_params(backend) ;
  tester = LabelerProjectTester(project_file_path, 'replace_path', replace_path) ;
  oc = onCleanup(@()(delete(tester))) ;
  labeler = tester.labeler ;

  % Set up the backend the way LabelerProjectTester.test_tracking does, but do
  % not start tracking yet.  The project is lightly trained, so it can track
  % without training first.
  tester.set_backend_params_(backend, backend_params) ;

  % Arrange for the job spawn to fail, mimicking a failed bsub submission.
  labeler.trackDLBackEnd.isSpawnForcedToFail_ = true ;

  % Attempt to track.  The forced spawn failure should make tracking error out.
  didTrackingThrow = false ;
  errorMessage = '' ;
  try
    labeler.track() ;
  catch me
    didTrackingThrow = true ;
    errorMessage = me.message ;
  end
  assert(didTrackingThrow, ...
         'Tracking should have thrown an error due to the forced spawn failure') ;
  % Make sure we actually reached (and failed at) the job spawn, rather than
  % throwing earlier for some unrelated reason, which would make the checks
  % below pass vacuously.
  assert(contains(errorMessage, 'Simulated spawn failure'), ...
         'Tracking threw, but not because of the forced spawn failure (message was: %s)', errorMessage) ;

  % The actual regression checks: no background tracking monitor or poller should
  % have been left behind by the failed spawn.
  tracker = labeler.tracker ;
  assert(isempty(tracker.bgTrkMonitor), ...
         'A background tracking monitor was left behind after a failed job spawn') ;
  assert(isempty(tracker.bgTrackPoller), ...
         'A background tracking poller was left behind after a failed job spawn') ;
end  % function
