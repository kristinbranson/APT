function test_mp4_seek_read_frame()
% Test that the fast keyframe-based reader (mp4_seek_read_frame, used by
% get_readframe_fcn for .mp4/.mov/.m4v) returns the correct frame for a given
% 1-based index.
%
% The test movie encodes each frame's own 0-based index as a binary barcode: a
% row of black/white cells, cell i lit iff bit i of the index is set.  Because
% the barcode is recoverable from the (lossily) decoded frame, the expected
% content of each frame can be synthesized from its index, so correctness can be
% checked without a slow read(readerobj, f) to serve as ground truth.  The movie
% has 1200 frames with a keyframe every 30 frames and no B-frame reordering,
% matching the structure of the lab's movies.

  % These constants must match how the test movie was generated.
  frameWidth = 192 ;
  cellWidth = 12 ;
  bitCount = 16 ;
  frameCount = 1200 ;

  linuxPath = '/groups/branson/bransonlab/apt/unittest/mp4_seek_test_barcode_1200f.mp4' ;
  moviePath = localize_test_project_path(linuxPath) ;

  index = mp4_read_index(moviePath) ;
  if ~index.isFastSeekable ,
    error('Test movie %s is unexpectedly not fast-seekable', moviePath) ;
  end
  if ~isequal(index.nframes, frameCount) ,
    error('Test movie %s has %d frames, expected %d', moviePath, index.nframes, frameCount) ;
  end

  % Frames to check (1-based): first/last, around keyframe boundaries (keyframes
  % are at 1-based frames 1, 31, 61, ...), interior frames, and a deep frame.
  % The order is deliberately non-monotonic -- with backward jumps and repeats --
  % to exercise seeking in both directions the way a user scrubbing would.  The
  % contiguous ascending runs (25:35 crosses a keyframe, 295:305 does not)
  % exercise the sequential fast path, where each frame is the one right after
  % the previous read.
  framesToCheck = [1, 2, 15, 30, 31, 32, 60, 61, 300, 599, 600, 601, 900, 1199, 1200, ...
                   600, 31, 1200, 1, 1199, 2, 900, 300, ...
                   25:35, 295:305] ;

  reader = VideoReader(moviePath) ;
  for checkIndex = 1 : numel(framesToCheck) ,
    f = framesToCheck(checkIndex) ;
    image = mp4_seek_read_frame(reader, index, f) ;
    decodedIndex = decodeBarcodeIndex_(image, frameWidth, cellWidth, bitCount) ;
    expectedIndex = f - 1 ;  % barcode encodes the 0-based index
    if ~isequal(decodedIndex, expectedIndex) ,
      error('Fast read of frame %d returned barcode for 0-based index %d', f, decodedIndex) ;
    end
  end
end  % function

function value = decodeBarcodeIndex_(image, frameWidth, cellWidth, bitCount)
% Recover the 0-based frame index encoded as a binary barcode in image.  Cell i
% (from the left) is white iff bit i of the index is set; sampling the middle of
% each cell and thresholding at mid-gray recovers each bit.
  gray = double(image(:, :, 1)) ;
  rowCount = size(gray, 1) ;
  rowLo = round(rowCount * 0.25) + 1 ;
  rowHi = round(rowCount * 0.75) ;
  value = 0 ;
  for bitIndex = 0 : bitCount - 1 ,
    centerColumn = bitIndex * cellWidth + floor(cellWidth / 2) + 1 ;  % 1-based
    columnLo = max(1, centerColumn - 2) ;
    columnHi = min(frameWidth, centerColumn + 2) ;
    patch = gray(rowLo : rowHi, columnLo : columnHi) ;
    if median(patch(:)) >= 128 ,
      value = value + 2^bitIndex ;
    end
  end
end  % function
