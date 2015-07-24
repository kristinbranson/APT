function [x,y] = getCurrentPoint(h)
%getCurrentPoint Return current point.
%   [X,Y] = getCurrentPoint(H) returns the x and y coordinates of the current
%   point. H can be a handle to an axes or a figure.
%
%   This function performs no validation on H.

%   Copyright 2005-2009 The MathWorks, Inc.
%   $Revision: 1.1.6.3 $  $Date: 2009/01/16 11:04:33 $

p = get(h,'CurrentPoint');

isHandleFigure = ishghandle(h,'figure');

if isHandleFigure
  x = p(1);
  y = p(2);
else
  % handle is axes
  x = p(1,1);
  y = p(1,2);
end

