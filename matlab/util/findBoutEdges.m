function result = findBoutEdges(currentFrameIndex, isSelectedFromFrameIndex)
% Find the edges of the bout that currentFrameIndex is in.  
% result is [boutFirstFrameIndex boutLastFrameIndex], or empty if
% isSelectedFromFrameIndex(currentFrameIndex) is false.
% Note that boutFirstFrameIndex and boutLastFrameIndex are *inclusive* at both
% ends.

% If the current frame is not selected, return an empty
if ~isSelectedFromFrameIndex(currentFrameIndex)
  result = [] ;
  return
end

% Find bout start
frameCount = numel(isSelectedFromFrameIndex) ;
for frameIndex = currentFrameIndex : -1 : 1
  isSelected = isSelectedFromFrameIndex(frameIndex) ;
  if ~isSelected ,
    break
  end
end
if isSelected
  % This means we got to the start of the array without finding an unselected
  % frame
  boutFirstFrameIndex = 1 ;
else
  boutFirstFrameIndex = frameIndex+1 ;
end

% Find bout end
for frameIndex = currentFrameIndex : frameCount
  isSelected = isSelectedFromFrameIndex(frameIndex) ;
  if ~isSelected ,
    break
  end
end
if isSelected
  % This means we got to the end of the array without finding an unselected
  % frame
  boutLastFrameIndex = frameCount ;
else
  boutLastFrameIndex = frameIndex-1 ;
end

result = [boutFirstFrameIndex boutLastFrameIndex] ;
end