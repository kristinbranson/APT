function test_tracker_backup_on_train()
  % algo_spec = 'ma_top_down_bbox_tddobj_tdpobj' ;
  [~, unittest_dir_path, replace_path] = get_test_project_paths() ;
  project_file_path = fullfile(unittest_dir_path, 'four-points-testing-2025-04-11-with-rois-added-and-fewer-smaller-avi-movies-lightly-trained-2-stages.lbl') ;  
  backend = docker_unless_janelia_cluster_then_conda() ;  % Should work on Linux or Windows
  backend_params = synthesize_backend_params(backend) ;
  tester = LabelerProjectTester(project_file_path, 'replace_path', replace_path) ;  
  oc = onCleanup(@()(delete(tester))) ;
  tester.labeler.isInDebugMode = true ;  % enable extra tests of tracking twining
  trackerCountBefore = numel(tester.labeler.trackerHistory) ;
  tester.test_training('backend',backend, ...
                       'backend_params', backend_params) ;
  trackerCountAfter = numel(tester.labeler.trackerHistory) ;
  assert(trackerCountAfter==trackerCountBefore+1, 'Something went wrong with backup tracker generation') ;
  assert(trackerCountAfter>=2, 'There should be at least two trackers after training') ;
  % For the current tracker, the lastTrainEndCause should be complete, but for
  % the backup it should be undefined.
  tracker = tester.labeler.tracker ;
  backup = tester.labeler.trackerHistory{2} ;
  assert(tracker.lastTrainEndCause == EndCause.complete, 'Tracking did not complete') ;
  assert(backup.lastTrainEndCause == EndCause.undefined, 'The lastTrainEndCause for the backup tracker is not EndCause.undefined') ;
  backupTimestamp = datetime(backup.trackerInfo.trainStartTS(1), 'ConvertFrom', 'datenum') ;
  trackerTimestamp = datetime(tracker.trackerInfo.trainStartTS(1), 'ConvertFrom', 'datenum') ;
  assert(trackerTimestamp > backupTimestamp, 'Backup appears to have been trained more recently than current tracker') ;
end  % function
