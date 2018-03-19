function P2 = readpdf2(P0,xg0,yg0,xg1,yg1,xc,yc,th)
% Like the inverse of readpdf. Readpdf is "I have a PDF, let's place it 
% somewhere in a new coord sys and read off the PDF in the new coord sys." 
% This is "I have a PDF, let's place a new coord sys somewhere and read the
% existing PDF in the new coords."
%
% Maybe can just re-use readpdf.
%
% P0: [ny x nx] array
% xg0: [ny x nx] xgrid (original), coords where pdf vals are given
% yg0: [ny x nx] ygrid (original)
% xg1: [nynew x nxnew] xgrid for new/resampled pdf (in new coords)
% yg1: [nynew x nxnew] ygrid for new/resampled pdf (in new coords)
% xc, yc: center of new coord sys (in original coords)
% th: new coord sys is rotated by this angle relative to original coords
%
% P2: same size as xg1,yg1. Readout of P0 at the locations specified by
% xg1,yg1,xc,yc,th

szassert(xg0,size(P0));
szassert(yg0,size(P0));
szassert(xg1,size(yg1));

costh = cos(th);
sinth = sin(th);
xg2 = xg1*costh - yg1*sinth;
yg2 = xg1*sinth + yg1*costh;
xg2 = xg2 + xc; 
yg2 = yg2 + yc;  

% xg2/yg2 are the grid specified by xg1/yg1, rotated by th and translated 
% to (xc,yc). They represent the locs (in orig coords) where we want to 
% read out P0.

P2 = interp2(xg0,yg0,P0,xg2,yg2,'linear',0);
