function test_train_error_during_polling()
  % Verifies the DeepTracker.forcedPollErrorIndex_ test hook.  Setting it to a
  % positive integer n makes the training bout error out (as if a mid-run error
  % had been detected) upon obtaining its n-th poll result, without any real
  % training failure.  Here we set it to 3, so the bout errors out on the third
  % poll result -- exercising the case where the monitor has already accumulated
  % several in-progress poll results before the error -- and confirm that the
  % bout ends with EndCause.error and that background training is no longer
  % running.  (Training runs for enough iterations, and its poll interval is
  % short enough, that the third poll comfortably lands mid-training; the
  % tracking bouts in this project complete after only two polls, so the
  % tracking counterpart forces the error on the first poll instead.)
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
    % Enough iterations that the job will not finish before the third poll, so the
    % forced error is what ends the bout.
  sPrm = labeler.trackGetTrainingParams() ;
  sPrm = structsetleaf(sPrm, struct('dl_steps', {niters}), 'verbose', true) ;
  labeler.trackSetTrainingParams(sPrm) ;

  % Arrange for the bout to error out on the third poll result.
  labeler.tracker.forcedPollErrorIndex_ = 3 ;

  % Start training.  The spawn should succeed; the forced poll error ends the
  % bout on the third poll, in the background.
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

  % The training monitor's status line should be red (not a stale green
  % "in progress" message) after the error.  This guards the fix in "Sync
  % monitor status line to Labeler state": the line is resynced from the
  % authoritative lastTrainEndCause on trainEnd, so an errored bout shows a red
  % error message rather than whatever the last poll happened to say.
  controller = tester.controller ;
  trainMonitorViz = controller.trainingMonitorVisualizer_ ;
  assert(~isempty(trainMonitorViz) && isvalid(trainMonitorViz), ...
         'No training monitor visualizer was present after the error') ;
  drawnow() ;  % ensure any pending status-line repaint has been applied
  handles = guidata(trainMonitorViz.hfig) ;
  statusColor = get(handles.text_clusterstatus, 'ForegroundColor') ;
  assert(isequal(statusColor, [1 0 0]), ...
         'Training monitor status line should be red after the error, but its color was [%g %g %g]', ...
         statusColor(1), statusColor(2), statusColor(3)) ;
  statusString = get(handles.text_clusterstatus, 'String') ;
  assert(any(contains(string(statusString), "Error")), ...
         'Training monitor status line should report an error, but reads: %s', char(strjoin(string(statusString), ' '))) ;
end  % function
