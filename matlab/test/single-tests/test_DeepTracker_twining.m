function test_DeepTracker_twining()
  linux_project_file_path = '/groups/branson/bransonlab/apt/unittest/four-points-testing-2025-04-12-with-rois-added-and-fewer-smaller-avi-movies-lightly-trained.lbl' ;
  [project_file_path, replace_path] = localize_test_project_path(linux_project_file_path) ;
  [labeler, controller] = StartAPT() ;
  oc = onCleanup(@()(delete(controller))) ;
  oc2 = onCleanup(@()(delete(labeler))) ;
  % Put the labeler in batch mode so prompts use defaults
  labeler.isInBatchMode = true ;
  % Load the named project
  labeler.projLoad(project_file_path, 'replace_path', replace_path) ;
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
