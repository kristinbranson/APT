function [err,errL,errR,errB,errreg,yLre,yRre,yBre] = ...
  objfunIntExtRot(x,yL,yR,yB,crig,lambda,verifyOpts)
% objective fcn for intrinsic + extrinsic/rot optim
% x: delta-values for calib params
% x(1:3): delta-om.LB
% x(4:6): delta-om.RB
% x(7:8): delta-cc_B/1000
% x(9:10): delta-fc_B/5000
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

assert(isequal(10,numel(x),numel(lambda)));

x = x(:);
domLB = x(1:3);
domRB = x(4:6);
dccB = x(7:8)*1e3;
dfcB = x(9:10)*5e3;

crigMod = crig.copy(); % shallow copy but is deep for this obj
try
  crigMod.om.LB = crigMod.om.LB+domLB;
  crigMod.om.BL = crigMod.om.BL-domLB;
  crigMod.om.RB = crigMod.om.RB+domRB;
  crigMod.om.BR = crigMod.om.BR-domRB;
  
  % set R based on om
  fldsRecompute = fieldnames(crigMod.om);
  fldsRm = setdiff(fieldnames(crigMod.R),fldsRecompute);  
  for f=fldsRecompute(:)',f=f{1}; %#ok<FXSET>
    crigMod.R.(f) = rodrigues(crigMod.om.(f));
  end
  crigMod.R = rmfield(crigMod.R,fldsRm);  
  
  crigMod.int.B.cc = crigMod.int.B.cc+dccB;
  crigMod.int.B.fc = crigMod.int.B.fc+dfcB;
  
  crigMod.verify(verifyOpts{:});

  [yLre,yRre,yBre,errL,errR,errB] = calibRoundTrip(yL,yR,yB,crigMod);
catch ME
  ME.rethrow();
end

errreg = mean(abs(x).*lambda(:)); 
% lambda should be set so that all comps of lambda.*abs(x) have comparable 
% scale

err = errL + errR + errB + errreg;