function pos = vertices2PosRect(vert)
% vertices2PosRect converts a set of vertices to a position rectangle.
%   pos = vertices2PosRect(vert) converts a set of vertices specified by
%   vert to a position rectangle. vert contains the vertices of a rectangle in the form:    
%   
%   vert = [x1 y1; xn yn]
%    
%   The input vertices are not assumed to be in any sorted order.    

%   Copyright 2006 The MathWorks, Inc.
%   $Revision: 1.1.6.1 $ $Date: 2006/11/08 17:49:38 $
    
    pos(1) = min(vert(:,1));
    pos(2) = min(vert(:,2));
    pos(3) = max(vert(:,1)) - pos(1);
    pos(4) = max(vert(:,2)) - pos(2);             
      
    
