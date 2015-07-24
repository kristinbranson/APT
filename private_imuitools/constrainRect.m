function new_pos = constrainRect(pos, im_width, im_height)
%constrainRect Constrain rectangle to image area.
%   new_pos = constrainRect(pos, im_width, im_height) constrains a position
%   vector, pos, to lie within the rectangle covered by a im_width-by-im_height
%   image.  The position vector has the form [x_min, y_min, width, height].  The
%   output argument NEW_POS is the constrained position vector.
%
%   constrainRect assumes the image has default spatial coordinates and covers
%   the range [0.5, im_width+0.5] along the x-axis and the range [0.5,
%   im_height+0.5] along the y-axis.  constrainRect modifies the position vector
%   only by translating it (changing x_min or y_min), so if pos is too wide or
%   high to fit inside the image area then new_pos will also be too wide or
%   high.
%
%   constrainRect assumes the input arguments are correctly formed and does no
%   error checking on them.

%   SLE
%   $Revision: 1.1.6.2 $  $Date: 2005/11/15 01:03:32 $
%   Copyright 2005 The MathWorks, Inc.

x_min = pos(1);
y_min = pos(2);
w     = pos(3);
h     = pos(4);

x_min = min( im_width  + 0.5 - w, max(x_min, 0.5) );
y_min = min( im_height + 0.5 - h, max(y_min, 0.5) );

new_pos = [x_min y_min w h];
