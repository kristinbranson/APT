function [firstFrameIndexFromTrackletBoutIndex, ...
          lastFrameIndexFromTrackletBoutIndex, ...
          extremalInterestFrameIndexFromTrackletBoutIndex, ...
          extremalInterestFromTrackletBoutIndex] = ...
  interestBoutsForSingleTracklet(frameIndexFromTrackletFrameIndex, ...
                                        interestFromTrackletFrameIndex, ...
                                        threshold)
% Find bouts of consecutive frames whose max-landmark interest is
% at or above the threshold (e.g. uncertain bouts).
%
% frameIndex and interest are parallel [N x 1] vectors for a
% single tracklet.  Returns [M x 1] vectors describing the M bouts found
% (possibly zero), with per-bout maximum interest and the frame
% within the movie where that maximum occurs.

% Determine which frames are interesting enough to pass the threshold.
isPassingFromTrackletFrameIndex = ...
  (interestFromTrackletFrameIndex >= threshold) ;

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
firstFrameIndexFromTrackletBoutIndex = ...
  frameIndexFromTrackletFrameIndex(firstTrackletFrameIndexFromTrackletBoutIndex) ;
lastFrameIndexFromTrackletBoutIndex = ...
  frameIndexFromTrackletFrameIndex(lastTrackletFrameIndexFromTrackletBoutIndex) ;

% For each bout, find the maximum interest and the frame at which
% it occurs (the most interesting / least confident frame in the bout).
extremalInterestFromTrackletBoutIndex = zeros(trackletBoutCount, 1) ;
extremalInterestFrameIndexFromTrackletBoutIndex = zeros(trackletBoutCount, 1) ;

for trackletBoutIndex = 1 : trackletBoutCount
  firstTrackletFrameIndex = firstTrackletFrameIndexFromTrackletBoutIndex(trackletBoutIndex) ;
  lastTrackletFrameIndex = lastTrackletFrameIndexFromTrackletBoutIndex(trackletBoutIndex) ;

  interestFromBoutTrackletFrameIndex = ...
    interestFromTrackletFrameIndex(firstTrackletFrameIndex:lastTrackletFrameIndex) ;

  [extremalInterest, extremalInterestBoutTrackletFrameIndex] = ...
    max(interestFromBoutTrackletFrameIndex) ;

  extremalInterestTrackletFrameIndex = ...
    firstTrackletFrameIndex + extremalInterestBoutTrackletFrameIndex - 1 ;
  extremalInterestFrameIndex = ...
    frameIndexFromTrackletFrameIndex(extremalInterestTrackletFrameIndex) ;

  extremalInterestFromTrackletBoutIndex(trackletBoutIndex) = extremalInterest ;
  extremalInterestFrameIndexFromTrackletBoutIndex(trackletBoutIndex) = extremalInterestFrameIndex ;
end  % for

end  % function
