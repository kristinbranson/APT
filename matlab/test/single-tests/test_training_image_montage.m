function test_training_image_montage()
  % The training monitor's "Show sample training images" action builds a
  % DataAugMontage from the training-image .mat files that training writes
  % and shows it (TrainMonitorController.trainImageMontage ->
  % DataAugMontage.show -> Shape.montage), then truesize()s the new figure
  % so the tiled montage image is displayed at one screen pixel per image
  % pixel.  Nothing else in the suite reaches Shape.montage or truesize.
  % This drives DataAugMontage with synthetic data shaped like the real
  % files -- uint8 crops, multi-animal landmark locations, a mask -- with
  % the montage arguments the training monitor passes, and checks that the
  % montage is drawn and that the figure ends up at true size.

  rng(0) ;
  imageCount = 6 ;
  rowCount = 80 ;
  columnCount = 120 ;
  pointCount = 4 ;
  targetCount = 2 ;
  montageRowCount = 2 ;
  montageColumnCount = 3 ;

  trainingImages = struct() ;
  trainingImages.idx = [ (1:imageCount)', (10:10:10*imageCount)', ones(imageCount, 1) ] ;
  trainingImages.ims = uint8(randi([0 255], [imageCount rowCount columnCount 1])) ;
  xs = columnCount * rand([imageCount targetCount pointCount]) ;
  ys = rowCount * rand([imageCount targetCount pointCount]) ;
  trainingImages.locs = cat(4, xs, ys) ;  % [n x ntgt x npts x 2], as the training files hold them
  trainingImages.mask = double(rand([imageCount rowCount columnCount]) > 0.5) ;
  trainingImages.name = 'view 1' ;

  montage = DataAugMontage() ;
  montage.init(trainingImages) ;
  assert(size(montage.locs, 2) == pointCount, 'init() did not put the points on the second axis of locs') ;
  colors = jet(pointCount) ;
  montageArgs = {'nr', montageRowCount, 'nc', montageColumnCount, 'maskalpha', 0.3, ...
                 'framelblscolor', [1 1 0], 'pplotargs', {'Marker', '+', 'MarkerSize', 8}, ...
                 'colors', colors} ;
  figureHandle = montage.show(montageArgs) ;
  cleaner = onCleanup(@()(delete(figureHandle))) ;
  drawnow() ;

  % The montage was drawn: one tiled image of the right size, markers on it,
  % a black figure.
  imageHandles = findobj(figureHandle, 'Type', 'image') ;
  assert(isscalar(imageHandles), 'Expected one montage image in the figure, found %d', numel(imageHandles)) ;
  montageSize = size(imageHandles.CData) ;
  expectedHeight = montageRowCount * rowCount ;
  expectedWidth = montageColumnCount * columnCount ;
  assert(isequal(montageSize(1:2), [expectedHeight expectedWidth]), ...
         'The montage image is %s, not %d x %d', mat2str(montageSize), expectedHeight, expectedWidth) ;
  assert(~isempty(findobj(figureHandle, 'Type', 'line')), 'No landmark markers were plotted on the montage') ;
  assert(isequal(figureHandle.Color, [0 0 0]), 'The montage figure is not black') ;

  % truesize(): one screen pixel per image pixel, so the axes is exactly the
  % image's size in pixels and the figure is at least that big.
  axesHandle = ancestor(imageHandles, 'axes') ;
  axesPixelPosition = getpixelposition(axesHandle) ;
  assert(all(abs(axesPixelPosition(3:4) - [expectedWidth expectedHeight]) < 0.5), ...
         'The montage axes is %s pixels, not %d x %d: the figure was not sized to the image', ...
         mat2str(axesPixelPosition(3:4)), expectedWidth, expectedHeight) ;
  figurePixelPosition = getpixelposition(figureHandle) ;
  assert(all(figurePixelPosition(3:4) >= [expectedWidth expectedHeight]), ...
         'The figure (%s pixels) is smaller than the montage image', mat2str(figurePixelPosition(3:4))) ;
end  % function
