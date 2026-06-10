function [firstFrameIndexFromTrackletBoutIndex, ...
          lastFrameIndexFromTrackletBoutIndex, ...
          extremalConfFrameIndexFromTrackletBoutIndex, ...
          extremalConfFromTrackletBoutIndex] = ...
  confidenceBoutsForSingleTracklet(frameIndexFromTrackletFrameIndex, ...
                                   minConfFromTrackletFrameIndex, ...
                                   maxConfFromTrackletFrameIndex, ...
                                   threshold)  %#ok<INUSD>
% Find bouts of consecutive frames whose minimum landmark confidence is at
% or below the threshold (i.e. uncertain bouts).
%
% frames, minConf, maxConf are parallel [N x 1] vectors for a single
% tracklet.  Returns [M x 1] vectors describing the M bouts found (possibly
% zero).  maxConf is accepted for API symmetry but not used.

% Determine which frames are uncertain enough to pass the threshold.
isPassingFromTrackletFrameIndex = (minConfFromTrackletFrameIndex <= threshold) ;
extremalConfFromTrackletFrameIndex = minConfFromTrackletFrameIndex ;

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

% For each bout, find the minimum confidence and the frame at which it
% occurs (the most uncertain frame in the bout).
extremalConfFromTrackletBoutIndex = zeros(trackletBoutCount, 1) ;
extremalConfFrameIndexFromTrackletBoutIndex = zeros(trackletBoutCount, 1) ;

for trackletBoutIndex = 1 : trackletBoutCount
  firstTrackletFrameIndex = firstTrackletFrameIndexFromTrackletBoutIndex(trackletBoutIndex) ;
  lastTrackletFrameIndex = lastTrackletFrameIndexFromTrackletBoutIndex(trackletBoutIndex) ;

  confFromBoutTrackletFrameIndex = extremalConfFromTrackletFrameIndex(firstTrackletFrameIndex:lastTrackletFrameIndex) ;

  [extremalConf, extremalConfBoutTrackletFrameIndex] = min(confFromBoutTrackletFrameIndex) ;

  extremalConfTrackletFrameIndex = firstTrackletFrameIndex + extremalConfBoutTrackletFrameIndex - 1 ;
  extremalConfFrameIndex = frameIndexFromTrackletFrameIndex(extremalConfTrackletFrameIndex) ;

  extremalConfFromTrackletBoutIndex(trackletBoutIndex) = extremalConf ;
  extremalConfFrameIndexFromTrackletBoutIndex(trackletBoutIndex) = extremalConfFrameIndex ;
end  % for

end  % function
