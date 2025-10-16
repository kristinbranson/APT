function test_DeepTrackerTopDown_twining()
  [~, unittest_dir_path, replace_path] = get_test_project_paths() ;
  project_file_path = fullfile(unittest_dir_path, 'four-points-testing-2025-04-11-with-rois-added-and-fewer-smaller-avi-movies-lightly-trained-2-stages.lbl') ;
  [labeler, controller] = StartAPT() ;
  oc = onCleanup(@()(delete(controller))) ;
  oc2 = onCleanup(@()(delete(labeler))) ;
  % Set the labeler to silent mode for batch operation
  labeler.silent = true ;
  % Load the named project
  labeler.projLoadGUI(project_file_path, 'replace_path', replace_path) ;
  % Make the backup
  labeler.trackMakeBackupOfCurrentTrackerIfHasBeenTrained() ;
  originalTracker = labeler.tracker ;
  backupTracker = labeler.trackerHistory{2} ;
  if ~originalTracker.tfIsTwin(backupTracker)
    error('backupTracker is not a twin of originalTracker') ;
  end
  if ~backupTracker.tfIsTwin(originalTracker)
    error('originalTracker is not a twin of backupTracker') ;
  end
  backupTracker.dryRunOnly = ~(originalTracker.dryRunOnly) ;
  if originalTracker.tfIsTwin(backupTracker)
    error('backupTracker is allegedly a twin of originalTracker, but should not be') ;
  end
  if backupTracker.tfIsTwin(originalTracker)
    error('originalTracker is allegedly a twin of backupTracker, but should not be') ;
  end  
  backupTracker.dryRunOnly = originalTracker.dryRunOnly ;
  backupTracker.trnLastDMC = originalTracker.trnLastDMC ;  % Set these to be identical
  if originalTracker.tfIsTwin(backupTracker)
    error('backupTracker is allegedly a twin of originalTracker, but should not be') ;
  end
  if backupTracker.tfIsTwin(originalTracker)
    error('originalTracker is allegedly a twin of backupTracker, but should not be') ;
  end    
end  % function
