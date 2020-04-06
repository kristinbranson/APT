function [err,errL,errR,errB,errreg,yLre,yRre,yBre,errFull] = ...
  objfunBint(x,yL,yR,yB,crig2,lambda,varargin)
% Objective fcn for calibration optim
%
% Bint: bot intrinsic only
%
% x: delta-values for calib params
% x(1:2): delta-cc_B/1000
% x(3:4): delta-fc_B/5000
%
% see objfunLRrot for rest.

assert(isequal(4,numel(x),numel(lambda)));

x = x(:);
dccB = x(1:2)*1e3;
dfcB = x(3:4)*5e3;

crig2Mod = crig2.copy(); % shallow copy but is deep for this obj
crig2Mod.int.B.cc = crig2Mod.int.B.cc+dccB;
crig2Mod.int.B.fc = crig2Mod.int.B.fc+dfcB;
 
% [yLre,yRre,yBre,errL,errR,errB,errFull.L,errFull.R,errFull.B] = ...
%   calibRoundTrip(yL,yR,yB,crig2Mod);
[yLre,yRre,errL,errR,errFull.L,errFull.R] = ...
   calibRoundTrip2(yL,yR,yB,crig2Mod);
yBre = [];
errB = 0;
errFull.B = 0;

errreg = mean(abs(x).*lambda(:)); 
% lambda should be set so that all comps of lambda.*abs(x) have comparable 
% scale

err = errL + errR + errB + errreg;