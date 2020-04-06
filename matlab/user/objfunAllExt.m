function [err,errL,errR,errB,errreg,yLre,yRre,yBre,errFull] = ...
  objfunAllExt(x,yL,yR,yB,crig2,lambda,varargin)
% Objective fcn for calibration optim
%
% allExt: all extrinsic. Kind of wantd to allow just rotations of all 3 
% cams, but that is 9 DOF and it's not clear how to represent the 3-dof
% constraint (12 extrinsic DOFs total, 3 constrained, 9 rotational DOFs).
%
% x: delta-values for calib params
% x(1:3): delta-omBL
% x(4:6): delta-omBR
% x(7:9): delta-TBL
% x(10:12): delta-TBR
%
% see objfunLRrot for rest.

assert(isequal(12,numel(x),numel(lambda)));

x = x(:);
domBL = x(1:3);
domBR = x(4:6);
dTBL = x(7:9);
dTBR = x(10:12);

crig2Mod = crig2.copy(); % shallow copy but is deep for this obj
crig2Mod.omBL = crig2Mod.omBL+domBL;
crig2Mod.omBR = crig2Mod.omBR+domBR; 
crig2Mod.TBL = crig2Mod.TBL+dTBL;
crig2Mod.TBR = crig2Mod.TBR+dTBR; 

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