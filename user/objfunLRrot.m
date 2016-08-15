function [err,errL,errR,errB,errreg,yLre,yRre,yBre,errFull] = ...
  objfunLRrot(x,yL,yR,yB,crig2,lambda,varargin)
% Objective fcn for calibration optim
%
% LRrot: left and right cameras can rotate.
%
% X_B = R_BR*X_R + T_BR
% X_B = R_BL*X_L + T_BL
% X_R or X_L==0 => T_BR, T_BL are locations of right, left cams in B frame.
% Suppose left, right cams are fixed wrt bot => T_BR, T_BL do not change.
% This leaves R_BR and R_BL for 6 rotational DOFs.
%
% x: delta-values for calib params
% x(1:3): delta-omBL
% x(4:6): delta-omBR
%
% yL,yR,yB: [nGTx2] (row,col) cropped coords for manual/GT pts
% crig: calibrated rig containing mean/baseline intrinsic and extrinsic
%   params
% lambda: [numel(x)] vector of regularization factors
%
% err: scalar/summed distance of
% reconstructed-and-reprojected-pts-from-original-gt-pts, eg you take the
% yL and yR and "round trip" them via the calibration into 3D and back and
% look at how far you are from where you started.

assert(isequal(6,numel(x),numel(lambda)));

x = x(:);
domBL = x(1:3);
domBR = x(4:6);

crig2Mod = crig2.copy(); % shallow copy but is deep for this obj
crig2Mod.omBL = crig2Mod.omBL+domBL;
crig2Mod.omBR = crig2Mod.omBR+domBR; 
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