function test_confidence_bouts_from_trk_file()
% Test confidenceBoutsFromTrkFile() with synthetic triangle-wave confidence.

% Define two synthetic tracklets with different confidence ranges.
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

% Verify that the default mock returns different confidences by landmark.
trkFile = TrkFileMock(trackletSpecs, [], false) ;
[~, ~, ~, rawConfidence] = trkFile.getPTrkTgt(1, 'auxflds', {'pTrkConf'}) ;
confidenceFromLandmarkIndex = rawConfidence(:, 1, 1, 1) ;
assertVectorsAlmostEqual(confidenceFromLandmarkIndex, [0; 0.25; 0.5], ...
                         'per-landmark confidence at first frame') ;

% Verify that the lack-thereof mode negates the generated confidence.
trkFileLackThereof = TrkFileMock(trackletSpecs, [], true) ;
[~, ~, ~, rawConfidence] = trkFileLackThereof.getPTrkTgt(1, 'auxflds', {'pTrkConf'}) ;
confidenceFromLandmarkIndex = rawConfidence(:, 1, 1, 1) ;
assertVectorsAlmostEqual(confidenceFromLandmarkIndex, [0; -0.25; -0.5], ...
                         'lack-thereof confidence at first frame') ;

% Verify invariance when negating both the confidence and the threshold.
invariantThreshold = 0.35 ;
[firstFrameIndex, lastFrameIndex, extremalFrameIndex] = ...
  confidenceBoutsFromTrkFile(trkFile, invariantThreshold, false, false) ;
[firstFrameIndexFromLackThereof, lastFrameIndexFromLackThereof, extremalFrameIndexFromLackThereof] = ...
  confidenceBoutsFromTrkFile(trkFileLackThereof, -invariantThreshold, true, false) ;
assertVectorsEqual(firstFrameIndexFromLackThereof, firstFrameIndex, ...
                   'negated-threshold invariance firstFrameIndex') ;
assertVectorsEqual(lastFrameIndexFromLackThereof, lastFrameIndex, ...
                   'negated-threshold invariance lastFrameIndex') ;
assertVectorsEqual(extremalFrameIndexFromLackThereof, extremalFrameIndex, ...
                   'negated-threshold invariance extremalFrameIndex') ;

% Record the per-frame minimum confidence values used by quantile tests.
allMinConfidence = [0; 0.25; 0.5; 0.75; 0.5; 0.25; 0.1; 0.3; 0.5; 0.7; 0.5; 0.3] ;
allMaxConfidence = [0.5; 0.75; 1; 1; 1; 0.75; 0.5; 0.7; 0.9; 0.9; 0.9; 0.7] ;

% Verify invariance systematically across threshold style and quantile mode.
caseInfo = struct( ...
  'label', {'absolute low', 'absolute high', 'quantile low', 'quantile high'}, ...
  'thresholdArg', {0.35, 0.65, 0.3, 0.7}, ...
  'isQuantile', {false, false, true, true}) ;
for caseIndex = 1 : numel(caseInfo)
  thisCase = caseInfo(caseIndex) ;
  thresholdArg = thisCase.thresholdArg ;
  if thisCase.isQuantile
    thresholdArgFromLackThereof = thresholdArg ;
  else
    thresholdArgFromLackThereof = -thresholdArg ;
  end

  [firstFrameIndex, lastFrameIndex, extremalFrameIndex] = ...
    confidenceBoutsFromTrkFile(trkFile, thresholdArg, false, thisCase.isQuantile) ;
  [firstFrameIndexFromLackThereof, lastFrameIndexFromLackThereof, extremalFrameIndexFromLackThereof] = ...
    confidenceBoutsFromTrkFile(trkFileLackThereof, ...
                               thresholdArgFromLackThereof, ...
                               true, ...
                               thisCase.isQuantile) ;

  assertVectorsEqual(firstFrameIndexFromLackThereof, firstFrameIndex, ...
                     [thisCase.label ' invariance firstFrameIndex']) ;
  assertVectorsEqual(lastFrameIndexFromLackThereof, lastFrameIndex, ...
                     [thisCase.label ' invariance lastFrameIndex']) ;
  assertVectorsEqual(extremalFrameIndexFromLackThereof, extremalFrameIndex, ...
                     [thisCase.label ' invariance extremalFrameIndex']) ;
end

% Check bout extraction for absolute low-confidence thresholding.
[firstFrameIndex, lastFrameIndex, extremalFrameIndex, trackletIndex, targetIndex, extremalConf, minConf, maxConf] = ...
  confidenceBoutsFromTrkFile(trkFile, 0.35, false, false) ;

assertVectorsEqual(firstFrameIndex, [1; 20; 6; 25], 'low-confidence firstFrameIndex') ;
assertVectorsEqual(lastFrameIndex, [2; 21; 6; 25], 'low-confidence lastFrameIndex') ;
assertVectorsEqual(extremalFrameIndex, [1; 20; 6; 25], 'low-confidence extremalFrameIndex') ;
assertVectorsEqual(trackletIndex, [1; 2; 1; 2], 'low-confidence trackletIndex') ;
assertVectorsEqual(targetIndex, [101; 202; 101; 202], 'low-confidence targetIndex') ;
assertVectorsAlmostEqual(extremalConf, [0; 0.1; 0.25; 0.3], 'low-confidence extremalConf') ;
assertScalarsAlmostEqual(minConf, 0, 'low-confidence minConf') ;
assertScalarsAlmostEqual(maxConf, 1, 'low-confidence maxConf') ;

% Check bout extraction for absolute high-confidence thresholding.
[firstFrameIndex, lastFrameIndex, extremalFrameIndex, trackletIndex, targetIndex, extremalConf, minConf, maxConf] = ...
  confidenceBoutsFromTrkFile(trkFile, 0.75, true, false) ;

assertVectorsEqual(firstFrameIndex, [2; 22], 'high-confidence firstFrameIndex') ;
assertVectorsEqual(lastFrameIndex, [6; 24], 'high-confidence lastFrameIndex') ;
assertVectorsEqual(extremalFrameIndex, [3; 22], 'high-confidence extremalFrameIndex') ;
assertVectorsEqual(trackletIndex, [1; 2], 'high-confidence trackletIndex') ;
assertVectorsEqual(targetIndex, [101; 202], 'high-confidence targetIndex') ;
assertVectorsAlmostEqual(extremalConf, [1; 0.9], 'high-confidence extremalConf') ;
assertScalarsAlmostEqual(minConf, 0, 'high-confidence minConf') ;
assertScalarsAlmostEqual(maxConf, 1, 'high-confidence maxConf') ;

% Check that quantile low-confidence mode matches the equivalent absolute threshold.
quantileThreshold = 0.3 ;
absoluteThreshold = quantile(allMinConfidence, quantileThreshold) ;
[firstFrameIndexFromQuantile, lastFrameIndexFromQuantile, extremalFrameIndexFromQuantile, ...
 trackletIndexFromQuantile, targetIndexFromQuantile, extremalConfFromQuantile, minConfFromQuantile, maxConfFromQuantile] = ...
  confidenceBoutsFromTrkFile(trkFile, quantileThreshold, false, true) ;
[firstFrameIndexFromAbsolute, lastFrameIndexFromAbsolute, extremalFrameIndexFromAbsolute, ...
 trackletIndexFromAbsolute, targetIndexFromAbsolute, extremalConfFromAbsolute, minConfFromAbsolute, maxConfFromAbsolute] = ...
  confidenceBoutsFromTrkFile(trkFile, absoluteThreshold, false, false) ;

assertVectorsEqual(firstFrameIndexFromQuantile, firstFrameIndexFromAbsolute, 'quantile low-confidence firstFrameIndex') ;
assertVectorsEqual(lastFrameIndexFromQuantile, lastFrameIndexFromAbsolute, 'quantile low-confidence lastFrameIndex') ;
assertVectorsEqual(extremalFrameIndexFromQuantile, extremalFrameIndexFromAbsolute, 'quantile low-confidence extremalFrameIndex') ;
assertVectorsEqual(trackletIndexFromQuantile, trackletIndexFromAbsolute, 'quantile low-confidence trackletIndex') ;
assertVectorsEqual(targetIndexFromQuantile, targetIndexFromAbsolute, 'quantile low-confidence targetIndex') ;
assertVectorsAlmostEqual(extremalConfFromQuantile, extremalConfFromAbsolute, 'quantile low-confidence extremalConf') ;
assertScalarsAlmostEqual(minConfFromQuantile, minConfFromAbsolute, 'quantile low-confidence minConf') ;
assertScalarsAlmostEqual(maxConfFromQuantile, maxConfFromAbsolute, 'quantile low-confidence maxConf') ;

% Check that quantile high-confidence mode matches the equivalent absolute threshold.
quantileThreshold = 0.25 ;
absoluteThreshold = quantile(allMaxConfidence, 1 - quantileThreshold) ;
[firstFrameIndexFromQuantile, lastFrameIndexFromQuantile, extremalFrameIndexFromQuantile, ...
 trackletIndexFromQuantile, targetIndexFromQuantile, extremalConfFromQuantile, minConfFromQuantile, maxConfFromQuantile] = ...
  confidenceBoutsFromTrkFile(trkFile, quantileThreshold, true, true) ;
[firstFrameIndexFromAbsolute, lastFrameIndexFromAbsolute, extremalFrameIndexFromAbsolute, ...
 trackletIndexFromAbsolute, targetIndexFromAbsolute, extremalConfFromAbsolute, minConfFromAbsolute, maxConfFromAbsolute] = ...
  confidenceBoutsFromTrkFile(trkFile, absoluteThreshold, true, false) ;

assertVectorsEqual(firstFrameIndexFromQuantile, firstFrameIndexFromAbsolute, 'quantile high-confidence firstFrameIndex') ;
assertVectorsEqual(lastFrameIndexFromQuantile, lastFrameIndexFromAbsolute, 'quantile high-confidence lastFrameIndex') ;
assertVectorsEqual(extremalFrameIndexFromQuantile, extremalFrameIndexFromAbsolute, 'quantile high-confidence extremalFrameIndex') ;
assertVectorsEqual(trackletIndexFromQuantile, trackletIndexFromAbsolute, 'quantile high-confidence trackletIndex') ;
assertVectorsEqual(targetIndexFromQuantile, targetIndexFromAbsolute, 'quantile high-confidence targetIndex') ;
assertVectorsAlmostEqual(extremalConfFromQuantile, extremalConfFromAbsolute, 'quantile high-confidence extremalConf') ;
assertScalarsAlmostEqual(minConfFromQuantile, minConfFromAbsolute, 'quantile high-confidence minConf') ;
assertScalarsAlmostEqual(maxConfFromQuantile, maxConfFromAbsolute, 'quantile high-confidence maxConf') ;
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
