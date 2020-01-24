function vert = getEllipseVertices(h_axes,pos)
% GETELLIPSEVERTICES returns a list of vertices that lie along the perimeter
% of an ellipse given an ellipse described by a MATLAB position rectangle.

%   Copyright 2007 The MathWorks, Inc.
%   $Revision: 1.1.6.2 $  $Date: 2007/06/04 21:11:14 $
    
cx = mean([pos(1),pos(1) + pos(3)]);
cy = mean([pos(2),pos(2) + pos(4)]);

a = cx - pos(1);
b = cy - pos(2);

[single_pixel_width,single_pixel_height] = getAxesScale(h_axes);

% In the case of different x/y scales, delta is the smaller of the
% length/width of 1 pixel expressed in image coordinates.
delta = min(single_pixel_width,single_pixel_height);

[x,y] = ellipseToPolygon(a,b,cx,cy,delta);

vert = [x',y'];
