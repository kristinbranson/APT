function test_label_movie_export()
  % The label-movie export (LabelerController.labelMakeLabelMovie_) writes
  % one frame per labeled frame of the current movie: it steps the view to
  % each labeled frame, captures the main axes with getframe, and appends
  % the capture to a VideoWriter.  The written movie must open as a video
  % with exactly that many frames, at the requested frame rate.
  %
  % Nothing in the GUI reaches this method today (it is scripting-only), so
  % this is the one place that exercises it -- and, for the Python port,
  % the one place that reaches VideoWriter.

  linux_project_file_path = ...
    '/groups/branson/bransonlab/apt/unittest/four-points-testing-2025-04-12-with-rois-added-and-fewer-smaller-avi-movies-lightly-trained-with-short-movie.lbl' ;
  [project_file_path, replace_path] = localize_test_project_path(linux_project_file_path) ;

  [labeler, controller] = ...
    StartAPT('projfile', project_file_path, ...
             'replace_path', replace_path) ;
  cleaner = onCleanup(@()(delete(controller))) ;
  drawnow() ;

  labeledTargetCounts = labeler.labelPosLabeledFramesStats() ;
  labeledFrameCount = nnz(labeledTargetCounts > 0) ;
  assert(labeledFrameCount > 0, 'The test project''s current movie has no labeled frames') ;

  outputDirPath = tempname() ;
  mkdir(outputDirPath) ;
  outputCleaner = onCleanup(@()(rmdir(outputDirPath, 's'))) ;
  outputMoviePath = fullfile(outputDirPath, 'labeled-frames.avi') ;

  frameRate = 5 ;
  controller.labelMakeLabelMovie_(outputMoviePath, 'frms2inc', 'lbled', 'framerate', frameRate) ;
  drawnow() ;

  assert(exist(outputMoviePath, 'file'), 'The export wrote no movie at %s', outputMoviePath) ;
  reader = VideoReader(outputMoviePath) ;
  assert(reader.NumFrames == labeledFrameCount, ...
         'The exported movie has %d frames; the movie has %d labeled frames', reader.NumFrames, labeledFrameCount) ;
  assert(abs(reader.FrameRate - frameRate) < 1e-6, ...
         'The exported movie plays at %g fps, not the requested %g', reader.FrameRate, frameRate) ;
  firstFrame = read(reader, 1) ;
  assert(ndims(firstFrame) == 3 && size(firstFrame, 3) == 3 && all(size(firstFrame, [1 2]) > 1), ...
         'The first exported frame is not an RGB image: size %s', mat2str(size(firstFrame))) ;
end  % function
