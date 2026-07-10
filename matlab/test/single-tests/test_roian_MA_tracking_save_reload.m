function test_roian_MA_tracking_save_reload()
  % Track the last movie of a project, save the project, close APT (which
  % deletes the per-session APT cache dir), then relaunch APT and load the
  % saved project.  The tracking results should survive the
  % save+close+reload cycle.
  %
  % As of 2026-07-10 this test fails: the .trk files holding the tracking
  % results live in the per-session cache dir and are not bundled into the
  % .lbl file, so after the cache dir is deleted the reload warns about a
  % missing .trk file and the results are gone.
  if ispc()
    warning('conda backend is not supported on Windows, so %s always passes on Windows', mfilename());
    return
  end
  [~, unittest_dir_path, replace_path] = get_test_project_paths() ;
  project_file_path = ...
    fullfile(unittest_dir_path, ...
             'four-points-testing-2025-04-12-with-rois-added-and-fewer-smaller-avi-movies-lightly-trained-with-short-movie.lbl') ;
  tester = LabelerProjectTester(project_file_path, 'replace_path', replace_path) ;
  oc = onCleanup(@()(delete(tester))) ;
  labeler = tester.labeler ;
  if ~isempty(labeler.tracker.trkP)
    error('labeler.tracker.trkP is nonempty---it should be empty before tracking') ;
  end

  % Switch to the last movie, which should be the short (500-frame) one
  movieCount = labeler.nmovies ;
  labeler.movieSet(movieCount) ;
  lastMoviePath = labeler.movieFilesAllFull{movieCount, 1} ;
  if ~contains(lastMoviePath, 'first500')
    error('Expected the last movie of the project to be the 500-frame movie, but it is %s', lastMoviePath) ;
  end

  % Track the last movie
  backend = 'conda' ;
  backend_params = synthesize_backend_params(backend) ;
  tester.test_tracking('algo_name', 'magrone', ...
                       'backend', backend, ...
                       'backend_params', backend_params) ;

  % The tracking results should have been written to a .trk file inside the
  % per-session APT cache dir
  cacheDirPath = labeler.projTempDir ;
  trkFilePath = labeler.tracker.trkPathFromImovAndViewIndex{movieCount, 1} ;
  if ~startsWith(trkFilePath, cacheDirPath)
    error('Expected the tracking-result path %s to be inside the APT cache dir %s', trkFilePath, cacheDirPath) ;
  end

  % Save the project to a temp file
  temporaryProjectFilePath = strcat(tempname(), '.lbl') ;
  labeler.projSave(temporaryProjectFilePath) ;
  oc2 = onCleanup(@()(delete(temporaryProjectFilePath))) ;

  % Close APT, then wait for the cache dir to be deleted (the Labeler
  % destructor deletes it asynchronously)
  delete(tester) ;
  maximumWaitTime = 60 ;  % seconds
  ticId = tic() ;
  while exist(cacheDirPath, 'dir') && toc(ticId) < maximumWaitTime ,
    pause(0.5) ;
  end
  if exist(cacheDirPath, 'dir')
    error('APT cache dir %s was not deleted within %g seconds of closing APT', cacheDirPath, maximumWaitTime) ;
  end

  % Relaunch APT and load the saved project
  commandWindowText = ...
    evalc(['[labeler2, controller2] = ', ...
           'StartAPT(''projfile'', temporaryProjectFilePath, ''replace_path'', replace_path, ''isInDebugMode'', true, ''isInYodaMode'', true) ;']) ;
  cleaner = onCleanup(@()(delete(controller2))) ;  % this will delete labeler2 too
  cleaner2 = onCleanup(@()(delete(labeler2))) ;  % but just to be sure

  if labeler2.nmovies ~= movieCount
    error('After reload, the project has %d movies---it should have %d', labeler2.nmovies, movieCount) ;
  end
  if labeler2.currMovie ~= movieCount
    error('After reload, the current movie is %d---it should be %d', labeler2.currMovie, movieCount) ;
  end
  % The tracking results should survive the save+close+reload cycle
  if contains(commandWindowText, 'Failed to load trkfile')
    error('Reloading the saved project warned about a missing .trk file---the tracking results should survive a save+close+reload cycle') ;
  end
  if isempty(labeler2.tracker.trkP)
    error('labeler2.tracker.trkP is empty after reload---the tracking results should survive a save+close+reload cycle') ;
  end
end  % function
