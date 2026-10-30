function test_frame_image_updated_before_overlays()
% Test that the displayed frame image is already updated when the
% frame-change overlays (predictions, frame box, slider) are updated.
%
% Regression test: setCurrentAndPreviousFrameData_() assigned .currFrame
% (whose setter fires updateAfterCurrentFrameSet, which moves the
% overlays) before reading the new frame from disk.  During video
% playback this made the annotations visibly lead the image by a frame.

% Start APT with an SA project without trx
linux_project_file_path = '/groups/branson/bransonlab/apt/unittest/pez7_al_updated_20241015.lbl' ;
[projectFile, replace_path] = localize_test_project_path(linux_project_file_path) ;
[labeler, controller] = StartAPT('projfile', projectFile, ...
                                 'replace_path', replace_path, ...
                                 'isInDebugMode', true, ...
                                 'isInYodaMode', true) ;
cleanupObj = onCleanup(@()(delete(controller))) ;
drawnow() ;

% Pick target frames that are distinct from the current one, and whose
% images differ from one another, so a stale image is detectable.
startFrame = labeler.currFrame ;
targetFrames = startFrame + [ 10 20 ] ;
assert(all(targetFrames <= labeler.nframes), 'Test movie is too short.') ;

% Each time the overlays are updated, record whether the displayed image
% matched the image for the frame being set.
image = controller.images_all(1) ;
expectedIm = [] ;
wasImageCurrentFromCallIndex = false(1, 0) ;
listener = addlistener(labeler, 'updateAfterCurrentFrameSet', ...
                       @(source, event)(recordImageState())) ;  %#ok<NASGU>

% Exercise setFrame()
for frameIndex = targetFrames
  expectedIm = labeler.movieReader(1).readframe(frameIndex, 'docrop', true) ;
  assert(~isequal(expectedIm, image.CData), ...
         'Frame %d image is identical to the displayed image; pick other frames.', frameIndex) ;
  labeler.setFrame(frameIndex) ;
  drawnow() ;
end

% Exercise setFrameAndTarget()
expectedIm = labeler.movieReader(1).readframe(startFrame, 'docrop', true) ;
labeler.setFrameAndTarget(startFrame, labeler.currTarget) ;
drawnow() ;

assert(~isempty(wasImageCurrentFromCallIndex), ...
       'updateAfterCurrentFrameSet never fired.') ;
assert(all(wasImageCurrentFromCallIndex), ...
       ['Frame-change overlays were updated before the frame image ' ...
        '(%d of %d updateAfterCurrentFrameSet events saw a stale image).'], ...
       sum(~wasImageCurrentFromCallIndex), numel(wasImageCurrentFromCallIndex)) ;

% The prev axes should show the previously-visited frame
assert(labeler.prevFrame == targetFrames(end), ...
       'Expected prevFrame to be %d, got %d', targetFrames(end), labeler.prevFrame) ;
expectedPrevIm = labeler.movieReader(1).readframe(targetFrames(end), 'docrop', true) ;
assert(isequal(labeler.prevIm, expectedPrevIm), ...
       'prevIm does not match the image of the previously-visited frame.') ;

  function recordImageState()
    % Listener callback: note whether the displayed image is the expected one.
    if isempty(expectedIm)
      return
    end
    wasImageCurrentFromCallIndex(end+1) = isequal(image.CData, expectedIm) ;
  end  % function

end  % function
