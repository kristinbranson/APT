function [indexInAxesArray,x,y] = findAxesThatTheCursorIsOver(axHandles)
%findAxesThatTheCursorIsOver returns an index of the axes below cursor.
%   [indexInAxesArray,x,y] = findAxesThatTheCursorIsOver(axHandles) determines which axes
%   the cursor is over and returns the location K of this axes in the
%   axHandles array. X and Y represent the cursor location.
  
%   Copyright 1993-2004 The MathWorks, Inc.
%   $Revision: 1.1.6.1 $  $Date: 2005/03/31 16:33:19 $
  
axesCurPt = get(axHandles,{'CurrentPoint'});
indexInAxesArray = [];

% determine which image the cursor is over.
for k = 1:length(axHandles)
    pt = axesCurPt{k};
    x = pt(1,1);
    y = pt(1,2);
    xlim = get(axHandles(k),'Xlim');
    ylim = get(axHandles(k),'Ylim');

    if x >= xlim(1) && x <= xlim(2) && y >= ylim(1) && y <= ylim(2)
       indexInAxesArray = k; 
       break;
    end
end

