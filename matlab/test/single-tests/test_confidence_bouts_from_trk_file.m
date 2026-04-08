function test_confidence_bouts_from_trk_file()
% Test confidenceBoutsFromTrkFile() with synthetic triangle-wave confidence.

trackletSpecs(1) = struct( ...
  'frameIndices', 1:6, ...
  'targetIndex', 101, ...
  'period', 8, ...
  'phaseOffset', 0, ...
  'amplitude', 1.0, ...
  'offset', 0.0) ;
trackletSpecs(2) = struct( ...
  'frameIndices', 20:25, ...
  'targetIndex', 202, ...
  'period', 8, ...
  'phaseOffset', 0, ...
  'amplitude', 0.8, ...
  'offset', 0.1) ;

trkFile = TrkFileMock(trackletSpecs) ;

[firstFrameIndex, lastFrameIndex, extremalFrameIndex, trackletIndex, targetIndex, extremalConf, minConf, maxConf] = ...
  confidenceBoutsFromTrkFile(trkFile, 0.35, false, false) ;

assertVectorsEqual(firstFrameIndex, [1; 20], 'low-confidence firstFrameIndex') ;
assertVectorsEqual(lastFrameIndex, [2; 21], 'low-confidence lastFrameIndex') ;
assertVectorsEqual(extremalFrameIndex, [1; 20], 'low-confidence extremalFrameIndex') ;
assertVectorsEqual(trackletIndex, [1; 2], 'low-confidence trackletIndex') ;
assertVectorsEqual(targetIndex, [101; 202], 'low-confidence targetIndex') ;
assertVectorsAlmostEqual(extremalConf, [0; 0.1], 'low-confidence extremalConf') ;
assertScalarsAlmostEqual(minConf, 0, 'low-confidence minConf') ;
assertScalarsAlmostEqual(maxConf, 1, 'low-confidence maxConf') ;

[firstFrameIndex, lastFrameIndex, extremalFrameIndex, trackletIndex, targetIndex, extremalConf, minConf, maxConf] = ...
  confidenceBoutsFromTrkFile(trkFile, 0.75, true, false) ;

assertVectorsEqual(firstFrameIndex, [4; 24], 'high-confidence firstFrameIndex') ;
assertVectorsEqual(lastFrameIndex, [6; 24], 'high-confidence lastFrameIndex') ;
assertVectorsEqual(extremalFrameIndex, [5; 24], 'high-confidence extremalFrameIndex') ;
assertVectorsEqual(trackletIndex, [1; 2], 'high-confidence trackletIndex') ;
assertVectorsEqual(targetIndex, [101; 202], 'high-confidence targetIndex') ;
assertVectorsAlmostEqual(extremalConf, [1; 0.9], 'high-confidence extremalConf') ;
assertScalarsAlmostEqual(minConf, 0, 'high-confidence minConf') ;
assertScalarsAlmostEqual(maxConf, 1, 'high-confidence maxConf') ;
end  % function



function assertVectorsEqual(actual, expected, label)
if ~isequal(actual, expected)
  error('test_confidence_bouts_from_trk_file:Mismatch', ...
        '%s mismatch. Expected %s, got %s.', ...
        label, mat2str(expected), mat2str(actual)) ;
end
end  % function



function assertVectorsAlmostEqual(actual, expected, label)
tolerance = 1e-12 ;
if ~isequal(size(actual), size(expected)) || any(abs(actual - expected) > tolerance)
  error('test_confidence_bouts_from_trk_file:Mismatch', ...
        '%s mismatch. Expected %s, got %s.', ...
        label, mat2str(expected), mat2str(actual)) ;
end
end  % function



function assertScalarsAlmostEqual(actual, expected, label)
tolerance = 1e-12 ;
if abs(actual - expected) > tolerance
  error('test_confidence_bouts_from_trk_file:Mismatch', ...
        '%s mismatch. Expected %g, got %g.', ...
        label, expected, actual) ;
end
end  % function
