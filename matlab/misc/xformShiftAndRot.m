function [x2,y2] = xformShiftAndRot(x,y,x0,y0,th)
% Affine xformation with shift and rotation
%
% Read coords in a new coord sys that is centered at x0,y0 and rotated at
% th relative to original
%
% x/y: arbitrary arrays of x/y coords (same size, eg outputs of meshgrid)
% x0/y0: center of new coord system
% th: orientation of new coord sys
%
% x2/y2: x and y, read out in new coord sys

szassert(x,size(y));
assert(isscalar(x0));
assert(isscalar(y0));
assert(isscalar(th));

xrel = x-x0;
yrel = y-y0;
% rotate xrel/yrel by -th
costh = cos(-th);
sinth = sin(-th);
x2 = xrel*costh - yrel*sinth;
y2 = xrel*sinth + yrel*costh;