function new_pos = constrainPoint(pos, im_width, im_height)
%constrainPoint Constrain point to image area.
%   new_pos = constrainPoint(pos, im_width, im_height) constrains a
%   position vector, pos, to lie within the rectangle covered by a
%   im_width-by-im_height image.  The position vector has the form [x y].  
%   The output argument NEW_POS is the constrained position vector.
%
%   constrainPoint assumes the image has default spatial coordinates
%   and covers the range [0.5, im_width+0.5] along the x-axis and the range
%   [0.5, im_height+0.5] along the y-axis.
%
%   constrainPoint assumes the input arguments are correctly formed and does 
%   no error checking on them.

%   $Revision $  $Date: 2005/11/15 01:03:31 $
%   Copyright 2005 The MathWorks, Inc.

x_candidate = pos(1);
y_candidate = pos(2);

x_new = min( im_width  + 0.5, max(x_candidate, 0.5) );
y_new = min( im_height + 0.5, max(y_candidate, 0.5) );

new_pos = [x_new y_new];
