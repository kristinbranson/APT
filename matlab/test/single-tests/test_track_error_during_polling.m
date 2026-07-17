function test_track_error_during_polling()
  % Verifies the DeepTracker.forcedPollErrorIndex_ test hook for the tracking
  % path.  Setting it to a positive integer n makes the tracking bout error out
  % (as if a mid-run error had been detected) upon obtaining its n-th poll
  % result, without any real tracking failure.  Here we set it to 1, so the bout
  % errors out on the very first poll result, and confirm that the bout ends with
  % EndCause.error and that background tracking is no longer running.
  %
  % This is the tracking counterpart of test_train_error_during_polling.  Unlike
  % test_track_spawn_failure_leaves_no_monitor, the job spawn itself succeeds
  % here; the error is injected later, during background polling.

  [~, unittest_dir_path, replace_path] = get_test_project_paths() ;
  project_file_path = fullfile(unittest_dir_path, 'multitarget_bubble_training_20210523_allGT_AR_MAAPT_grone2_UT_resaved_3_lightly_trained.lbl') ;
  backend = docker_unless_janelia_cluster_then_conda() ;  % Should work on Linux or Windows
  backend_params = synthesize_backend_params(backend) ;
  tester = LabelerProjectTester(project_file_path, 'replace_path', replace_path) ;
  oc = onCleanup(@()(delete(tester))) ;
  labeler = tester.labeler ;

  % Set up the backend the way LabelerProjectTester.test_tracking does, but do
  % not start tracking yet.  The project is lightly trained, so it can track
  % without training first.
  tester.set_backend_params_(backend, backend_params) ;

  % Arrange for the bout to error out on the first poll result.
  labeler.tracker.forcedPollErrorIndex_ = 1 ;

  % Start tracking.  The spawn should succeed; the forced poll error ends the
  % bout shortly afterward, in the background.
  labeler.track() ;

  % Wait for the (forced) error to end the bout.
  pause(2) ;
  maximumWaitTime = 300 ;  % seconds
  ticId = tic() ;
  while labeler.bgTrkIsRunning && toc(ticId) < maximumWaitTime ,
    pause(5) ;
  end

  % The bout should have ended (background tracking no longer running) ...
  assert(~labeler.bgTrkIsRunning, ...
         'Background tracking was still running %g seconds after the forced poll error', maximumWaitTime) ;
  % ... and it should have ended because of an error.
  assert(labeler.lastTrackEndCause == EndCause.error, ...
         'Tracking bout should have ended with EndCause.error, but ended with %s', char(labeler.lastTrackEndCause)) ;
end  % function
