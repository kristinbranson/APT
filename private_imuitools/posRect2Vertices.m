function vertices = posRect2Vertices(pos)
% posRect2Vertices Convert a position rectangle to a set of vertices.
%   vertices = posRect2Vertices(pos) converts a position rectangle specified
%   by pos and returns a set of vertices. The output vert is always provided
%   in cw order starting from xmin,ymin:
%    
%   vertices = [xmin ymin; xmin ymax; xmax ymax; xmax ymin].  

%   Copyright 2006 The MathWorks, Inc.
%   $Revision: 1.1.6.1 $ $Date: 2006/11/08 17:49:37 $
    
    vertices(1,:) = pos(1:2);
    vertices(2,:) = [pos(1), pos(2) + pos(4)];
    vertices(3,:) = [pos(1) + pos(3), pos(2) + pos(4)];
    vertices(4,:) = [pos(1) + pos(3), pos(2)];
          
 
