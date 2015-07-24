function newPos = constrainFigSize(newPos, lastPos, minSize)
%constrainFigSize Constrain the minimum size of a figure 

%   constrainFigSize(newFigPos, lastFigPos, minXYSize) constrains the width
%   and height of a figure by using its last position and new position.
%   newFigPos returned by constrainFigSize contains the new position of the 
%   figure such that only the resizing edge/corner of the figure is changed.
  
%   Copyright 1993-2004 The MathWorks, Inc.  
%   $Revision $  $Date: 2004/08/10 01:50:01 $
  
newSize = newPos(3:4);
tooSmall = (newSize < minSize);
newSize(tooSmall) = minSize(tooSmall);


%  There's a problem which seems to stem from the fact that the
%  resize position is precomputed. That is, the function position
%  returned doesn't always match the actual figure position.

if tooSmall(1) % X needs to be resized
  changeX = newSize(1) - newPos(3);
  
  %determine the resize direction
  leftSideMoved = (lastPos(1) ~= newPos(1));
  
  if (leftSideMoved) % the left side moved
    newPos(1) = newPos(1)-changeX;
    newPos(3) = newSize(1);
  else % the right side moved
    newPos(3) = newPos(3)+changeX;
  end
end
        
if tooSmall(2) % Y needs to be resized
  changeY = newSize(2) - newPos(4);
  
  %determine the resize direction
  bottomMoved = (lastPos(2) ~= newPos(2));
  if (bottomMoved) % the bottom moved
    newPos(2) = newPos(2) - changeY;
    newPos(4) = newSize(2);
  else % the top moved
    newPos(4) = newPos(4)+changeY;
  end
end



