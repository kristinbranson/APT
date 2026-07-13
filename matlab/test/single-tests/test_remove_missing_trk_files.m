function test_remove_missing_trk_files()
  % DeepTracker.removeMissingTrkFiles() should remove entries for missing
  % .trk files from *both* the persistent store and the ad-hoc store, even
  % though the persistent trkfile, when present, shadows the ad-hoc ones at
  % read time.  Missing ad-hoc trkfiles should be removed even while
  % shadowed.
  [~, unittest_dir_path, replace_path] = get_test_project_paths() ;
  project_file_path = ...
    fullfile(unittest_dir_path, ...
             'four-points-testing-2025-04-11-with-rois-added-and-fewer-smaller-avi-movies.lbl') ;
  tester = LabelerProjectTester(project_file_path, 'replace_path', replace_path) ;
  oc = onCleanup(@()(delete(tester))) ;
  labeler = tester.labeler ;
  tracker = labeler.tracker ;
  mIdx = MovieIndex(1) ;

  % Make a trkfile that exists on disk, plus paths to ones that don't
  existingTrkFilePath = strcat(tempname(), '.trk') ;
  fid = fopen(existingTrkFilePath, 'w') ;
  fclose(fid) ;
  oc2 = onCleanup(@()(delete(existingTrkFilePath))) ;
  missingPersistentTrkFilePath = strcat(tempname(), '.trk') ;
  missingAdhocTrkFilePath = strcat(tempname(), '.trk') ;

  % Populate both stores: a missing persistent trkfile, plus one existing
  % and one missing ad-hoc trkfile.  (This is a single-view project.)
  tracker.trackResSetPersistentTrkfile(mIdx, {missingPersistentTrkFilePath}) ;
  tracker.trackResAddAdhocTrkfile(mIdx, {existingTrkFilePath}) ;
  tracker.trackResAddAdhocTrkfile(mIdx, {missingAdhocTrkFilePath}) ;

  tracker.removeMissingTrkFiles(mIdx) ;

  % The missing persistent trkfile should have been removed
  if tracker.trackResHasPersistentTrkfile(mIdx)
    error('The persistent trkfile is missing on disk, but removeMissingTrkFiles() did not remove it') ;
  end
  % The missing ad-hoc trkfile should have been removed, and the existing
  % one retained.  (With the persistent trkfile gone, trackResGetTrkfiles
  % returns the ad-hoc rows.)
  trkfiles = tracker.trackResGetTrkfiles(mIdx) ;
  if ~isequal(trkfiles, {existingTrkFilePath})
    error(['After removeMissingTrkFiles(), expected the ad-hoc store to hold exactly the one existing trkfile, ', ...
           'but it holds %d trkfile(s)'], numel(trkfiles)) ;
  end
end  % function
