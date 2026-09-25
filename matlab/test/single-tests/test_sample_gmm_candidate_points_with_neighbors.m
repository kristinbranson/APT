function test_sample_gmm_candidate_points_with_neighbors()
  % PostProcess.SampleGMMCandidatePoints() must be able to draw multi-view
  % samples that mix detections from the current frame with detections from
  % the previous and next frames.
  %
  % Regression test: the neighbor-frame branches summed
  % Kpreview_prev + Kperview_curr and Kpreview_next + Kperview_curr,
  % misspellings of Kperview_prev, Kperview_next and Kperview, so any nonzero
  % nsamples_neighbor errored with "Unrecognized function or variable".

  rngState = rng() ;
  rng(0) ;
  cleaner = onCleanup(@()(rng(rngState))) ;

  viewCount = 2 ;
  componentCount = 2 ;
  gmmdata = synthesizeGmmData(viewCount, componentCount) ;
  gmmdataPrev = synthesizeGmmData(viewCount, componentCount) ;
  gmmdataNext = synthesizeGmmData(viewCount, componentCount) ;

  % A stand-in for stereo reconstruction: the 3-D point is the mean of the
  % views' 2-D points with a zero third coordinate, and each view reprojects
  % to itself.
  reconstructfun = @(x, S) (deal([mean(x, 2) ; 0], x)) ;

  sampleCount = 2 ;
  neighborSampleCount = 1 ;
  [Xsample, Wsample, x_re_sample, x_sample] = ...
    PostProcess.SampleGMMCandidatePoints(gmmdata, ...
                                         'nsamples', sampleCount, ...
                                         'nsamples_neighbor', neighborSampleCount, ...
                                         'gmmdata_prev', gmmdataPrev, ...
                                         'gmmdata_next', gmmdataNext, ...
                                         'reconstructfun', reconstructfun) ;

  % nsamples_curr = nsamples + 2*nsamples_neighbor - nsamples_prev - nsamples_next,
  % and one sample comes from each neighbor frame.
  currentSampleCount = sampleCount ;
  perViewSampleCount = currentSampleCount + 2 * neighborSampleCount ;
  assert(isequal(size(x_sample), [2 viewCount perViewSampleCount]), ...
         'Expected %d per-view samples, got size %s', perViewSampleCount, mat2str(size(x_sample))) ;
  assert(isequal(size(x_re_sample), size(x_sample)), ...
         'Reprojected samples do not match the samples in size') ;
  assert(size(Xsample, 1) == 3 && size(Xsample, 2) == numel(Wsample), ...
         '3-D samples and weights disagree in count') ;
  assert(all(isfinite(Xsample(:))) && all(isfinite(Wsample(:))), ...
         'Samples or weights are not finite') ;
end  % function

function gmmdata = synthesizeGmmData(viewCount, componentCount)
  % A viewCount-element struct array of 2-D GMMs with componentCount components
  % each: .mu is 2 x K, .S is 2 x 2 x K, .w is 1 x K and sums to one.
  gmmdata = struct('mu', cell(1, viewCount), 'S', [], 'w', []) ;
  for viewIndex = 1 : viewCount
    gmmdata(viewIndex).mu = 10 * rand(2, componentCount) ;
    gmmdata(viewIndex).S = repmat(eye(2), [1 1 componentCount]) ;
    w = rand(1, componentCount) ;
    gmmdata(viewIndex).w = w / sum(w) ;
  end
end  % function
