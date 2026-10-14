function test_orthocam_view_extrinsics()
  % Test that OrthoCam's two extrinsics viewers draw the world x-z plane as
  % a 2-D plot: one patch per calibration pattern, the optical axes and
  % camera markers as lines, z on the vertical axis, and no 3-D view.  They
  % used plot3/fill3 under view(0,0), which showed exactly this plane.

  % A 3-by-2 pattern of unit-spaced points and two patterns: one flat at the
  % origin, one tilted and shifted.
  [gridX, gridY] = meshgrid(0:2, 0:1) ;
  worldPoints = [gridX(:) gridY(:)] ;   % [pointCount 2]
  patternRotations = [0 0.3 ; 0 0.2 ; 0 0.1] ;   % [3 patternCount]
  patternTranslations = [0 1 ; 0 2] ;           % [2 patternCount]
  patternCount = size(patternRotations, 2) ;

  singleCameraFigure = OrthoCam.viewExtrinsics1cam(worldPoints, patternRotations, patternTranslations) ;
  cleanupSingle = onCleanup(@()(delete(singleCameraFigure))) ;
  singleCameraAxes = findobj(singleCameraFigure, 'Type', 'axes') ;
  assert(isscalar(singleCameraAxes), 'Expected one axes in the single-camera viewer') ;
  checkPlaneIsXZ_(singleCameraAxes, patternCount) ;
  % The flat pattern at the origin lies along z = 0, x from 0 to 2.
  patches = findobj(singleCameraAxes, 'Type', 'patch') ;
  isFlatPattern = arrayfun(@(p)(all(p.YData == 0) && isequal(sort(p.XData(:))', [0 0 2 2])), patches) ;
  assert(nnz(isFlatPattern) == 1, 'Expected the flat pattern drawn along z = 0 from x = 0 to 2') ;
  % The optical axis runs from the origin to z = -10, drawn dashed.
  opticalAxis = findobj(singleCameraAxes, 'Type', 'line', 'LineStyle', '--') ;
  assert(isscalar(opticalAxis), 'Expected one dashed optical axis') ;
  assert(isequal(opticalAxis.XData, [0 0]) && isequal(opticalAxis.YData, [0 -10]), ...
         'Expected the optical axis from (0, 0) to (0, -10) in x-z, got x %s y %s', ...
         mat2str(opticalAxis.XData), mat2str(opticalAxis.YData)) ;
  cameraLabel = findobj(singleCameraAxes, 'Type', 'text', 'String', 'C') ;
  assert(isscalar(cameraLabel) && isequal(cameraLabel.Position(1:2), [0 -11]), ...
         'Expected the camera label at (0, -11) in x-z') ;

  % The stereo viewer: pattern points as [3 pointCount] with z = 0, pattern
  % extrinsics as rows, and a second camera turned about y.
  patternPointsXYZ = [worldPoints' ; zeros(1, size(worldPoints, 1))] ;
  stereoFigure = OrthoCam.viewExtrinsics(patternPointsXYZ, patternRotations', [patternTranslations' zeros(patternCount, 1)], ...
                                         [0 ; 0 ; 0], [0 ; 0], [0 ; 0.5 ; 0], [1 ; 0]) ;
  cleanupStereo = onCleanup(@()(delete(stereoFigure))) ;
  stereoAxes = findobj(stereoFigure, 'Type', 'axes') ;
  assert(isscalar(stereoAxes), 'Expected one axes in the stereo viewer') ;
  checkPlaneIsXZ_(stereoAxes, patternCount) ;
  cameraMarkers = findobj(stereoAxes, 'Type', 'line', 'Marker', 'x') ;
  assert(numel(cameraMarkers) == 2, 'Expected two camera markers, got %d', numel(cameraMarkers)) ;
  % The title ends in a newline, so its String is two rows.
  titleText = strjoin(cellstr(stereoAxes.Title.String), ' ') ;
  assert(contains(titleText, 'camAngle'), 'Expected the camera angle in the title, got "%s"', titleText) ;
end  % function



function checkPlaneIsXZ_(ax, patternCount)
  % Check that ax is a 2-D plot of the world x-z plane with one patch per
  % pattern.
  assert(isequal(ax.View, [0 90]), 'Expected a 2-D view, got %s', mat2str(ax.View)) ;
  assert(strcmp(ax.XLabel.String, 'x (mm)'), 'Expected x on the horizontal axis') ;
  assert(strcmp(ax.YLabel.String, 'z (mm)'), 'Expected z on the vertical axis, got "%s"', ax.YLabel.String) ;
  assert(isempty(ax.ZLabel.String), 'Expected no z label') ;
  patches = findobj(ax, 'Type', 'patch') ;
  assert(numel(patches) == patternCount, 'Expected one patch per pattern, got %d', numel(patches)) ;
end  % function
