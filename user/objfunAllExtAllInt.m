function [err,errL,errR,errB,errreg,yLre,yRre,yBre,errFull,crig2Mod] = ...
  objfunAllExtAllInt(x,yL,yR,yB,crig2,lambda,varargin)
% Objective fcn for calibration optim
%
% x: delta-values for calib params
% x(1:3): delta-omBL
% x(4:6): delta-omBR
% x(7:9): delta-TBL
% x(10:12): delta-TBR
% x(13:14): delta-cc_B/1000
% x(15:16): delta-fc_B/5000
% x(17:18): delta-cc_L/1000
% x(19:20): delta-fc_L/5000
% x(21:22): delta-cc_R/1000
% x(23:24): delta-fc_R/5000
%
% see objfunLRrot for rest.

assert(isequal(24,numel(x),numel(lambda)));

x = x(:);
domBL = x(1:3);
domBR = x(4:6);
dTBL = x(7:9);
dTBR = x(10:12);
dccB = x(13:14)*1e3;
dfcB = x(15:16)*5e3;
dccL = x(17:18)*1e3;
dfcL = x(19:20)*5e3;
dccR = x(21:22)*1e3;
dfcR = x(23:24)*5e3;

crig2Mod = crig2.copy(); % shallow copy but is deep for this obj
crig2Mod.omBL = crig2Mod.omBL+domBL;
crig2Mod.omBR = crig2Mod.omBR+domBR; 
crig2Mod.TBL = crig2Mod.TBL+dTBL;
crig2Mod.TBR = crig2Mod.TBR+dTBR; 
crig2Mod.int.B.cc = crig2Mod.int.B.cc+dccB;
crig2Mod.int.B.fc = crig2Mod.int.B.fc+dfcB;
crig2Mod.int.L.cc = crig2Mod.int.L.cc+dccL;
crig2Mod.int.L.fc = crig2Mod.int.L.fc+dfcL;
crig2Mod.int.R.cc = crig2Mod.int.R.cc+dccR;
crig2Mod.int.R.fc = crig2Mod.int.R.fc+dfcR;

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