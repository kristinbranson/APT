function [startFrames, endFrames, extremeFrames, extremeConfs] = ...
  confidenceBoutsForSingleTracklet(frames, minConf, maxConf, threshold, isConfidenceLackThereof)
% Find bouts of consecutive frames whose confidence passes a threshold.
%
% frames, minConf, maxConf are parallel [N x 1] vectors for a single
% tracklet.  Returns [M x 1] vectors describing the M bouts found (possibly
% zero).

% Determine which frames pass the threshold.
if isConfidenceLackThereof
  isPassing = (maxConf >= threshold) ;
  relevantConf = maxConf ;
else
  isPassing = (minConf <= threshold) ;
  relevantConf = minConf ;
end

% Find bout boundaries using transitions in the passing mask.
edges = diff([false ; isPassing ; false]) ;
boutStartIndices = find(edges == 1) ;  % indices into frames where bouts start
boutEndIndices = find(edges == -1) - 1 ;  % indices into frames where bouts end
boutCount = numel(boutStartIndices) ;

startFrames = frames(boutStartIndices) ;
endFrames = frames(boutEndIndices) ;
extremeFrames = zeros(boutCount, 1) ;
extremeConfs = zeros(boutCount, 1) ;

for iBout = 1 : boutCount
  boutConfs = relevantConf(boutStartIndices(iBout):boutEndIndices(iBout)) ;
  if isConfidenceLackThereof
    [extremeConf, extremeOffset] = max(boutConfs) ;
  else
    [extremeConf, extremeOffset] = min(boutConfs) ;
  end
  extremeConfs(iBout) = extremeConf ;
  extremeFrames(iBout) = frames(boutStartIndices(iBout) + extremeOffset - 1) ;
end

end  % function
