function P2 = readpdf(P0,xg0,yg0,xg1,yg1,x0,y0,th0)
% Translate/rotate an existing PDF to a new location/orientation and return
% the resulting PDF. 
%
% THE RESULT IS NOT (RE)NORMALIZED AND MAY NOT SUM TO 1 OR TO THE ORIGINAL 
% NORMALIZATION WHATEVER THAT MAY BE.
%
% P0: [ny x nx] array
% xg0: [ny x nx] xgrid (original), coords where pdf vals are given
% yg0: [ny x nx] ygrid (original)
% xg1: [nynew x nxnew] xgrid for new/resampled pdf
% yg1: [nynew x nxnew] ygrid for new/resampled pdf
% x0/y0: (0,0) in original pdf (pdfXE/pdfYE etc) maps to (x0,y0) in new pdf
% th0: original pdf should be rotated counterclockwise by th0 in new pdf
%   (counterclockwise with usual/canonical x/y axes layout)

szassert(xg0,size(P0));
szassert(yg0,size(P0));
szassert(xg1,size(yg1));

[xgXFormed,ygXFormed] = xformShiftAndRot(xg1,yg1,x0,y0,th0);
P2 = interp2(xg0,yg0,P0,xgXFormed,ygXFormed,'linear',0);
