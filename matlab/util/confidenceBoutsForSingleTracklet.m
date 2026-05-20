function [firstFrameIndexFromTrackletBoutIndex, ...
          lastFrameIndexFromTrackletBoutIndex, ...
          extremalConfFrameIndexFromTrackletBoutIndex, ...
          extremalConfFromTrackletBoutIndex] = ...
  confidenceBoutsForSingleTracklet(frameIndexFromTrackletFrameIndex, ...
                                   minConfFromTrackletFrameIndex, ...
                                   maxConfFromTrackletFrameIndex, ...
                                   threshold, ...
                                   isConfidenceLackThereof)
% Find bouts of consecutive frames whose confidence passes a threshold.
%
% frames, minConf, maxConf are parallel [N x 1] vectors for a single
% tracklet.  Returns [M x 1] vectors describing the M bouts found (possibly
% zero).

% Determine which frames pass the threshold.
if isConfidenceLackThereof
  isPassingFromTrackletFrameIndex = (maxConfFromTrackletFrameIndex >= threshold) ;
  extremalConfFromTrackletFrameIndex = maxConfFromTrackletFrameIndex ;
else
  isPassingFromTrackletFrameIndex = (minConfFromTrackletFrameIndex <= threshold) ;
  extremalConfFromTrackletFrameIndex = minConfFromTrackletFrameIndex ;
end

% Find bout boundaries using transitions in the passing mask.
edgeSignFromTrackletStepIndex = diff([false ; isPassingFromTrackletFrameIndex ; false]) ;
  % Idea is that there is a "step" between consecutive frames, also one
  % before the first frame and one after the last frame.  
  % Thus stepCount == frameCount+2-1 == frameCount+1.
firstTrackletFrameIndexFromTrackletBoutIndex = ...
  find(edgeSignFromTrackletStepIndex == 1) ;  % first frame of each bout
lastTrackletFrameIndexFromTrackletBoutIndex = ...
  find(edgeSignFromTrackletStepIndex == -1) - 1 ;  % last frame of each bout
trackletBoutCount = numel(firstTrackletFrameIndexFromTrackletBoutIndex) ;

% Convert the first and last frame index for each bout from the tracklet
% frame index to the actual frame index within the movie.
firstFrameIndexFromTrackletBoutIndex = frameIndexFromTrackletFrameIndex(firstTrackletFrameIndexFromTrackletBoutIndex) ;
lastFrameIndexFromTrackletBoutIndex = frameIndexFromTrackletFrameIndex(lastTrackletFrameIndexFromTrackletBoutIndex) ;

% For each bout, determine the extremal confidence for the bout, and the
% frame index at which it occurs. (The extremal confidence is the
% maxConfidence if isConfidenceLackThereof is false, otherwise the extremal
% confidence is the min confidence.)
extremalConfFromTrackletBoutIndex = zeros(trackletBoutCount, 1) ;
extremalConfFrameIndexFromTrackletBoutIndex = zeros(trackletBoutCount, 1) ;

for trackletBoutIndex = 1 : trackletBoutCount
  % Extract the first and last frames for the bout
  firstTrackletFrameIndex = firstTrackletFrameIndexFromTrackletBoutIndex(trackletBoutIndex) ;
  lastTrackletFrameIndex = lastTrackletFrameIndexFromTrackletBoutIndex(trackletBoutIndex) ;  

  % Extract the confidence for each frame within the bout
  confFromBoutTrackletFrameIndex = extremalConfFromTrackletFrameIndex(firstTrackletFrameIndex:lastTrackletFrameIndex) ;

  % Determine the extremal confidence and where it occurs within the bout
  if isConfidenceLackThereof
    [extremalConf, extremalConfBoutTrackletFrameIndex] = max(confFromBoutTrackletFrameIndex) ;
  else
    [extremalConf, extremalConfBoutTrackletFrameIndex] = min(confFromBoutTrackletFrameIndex) ;
  end

  % For the frame where the extremal confidence occurs, convert the frame
  % index with the bout to the frame index within the whole movie.
  extremalConfTrackletFrameIndex = firstTrackletFrameIndex + extremalConfBoutTrackletFrameIndex - 1 ;
    % the frame index within the tracklet
  extremalConfFrameIndex = frameIndexFromTrackletFrameIndex(extremalConfTrackletFrameIndex) ;
    % the frame index within the movie

  % Store the extremal confidence for this bout
  extremalConfFromTrackletBoutIndex(trackletBoutIndex) = extremalConf ;

  % Store the frame index of the frame where the extremal confidence occurs
  % for this bout.
  extremalConfFrameIndexFromTrackletBoutIndex(trackletBoutIndex) = extremalConfFrameIndex ;
end  % for

end  % function
