function [err,errL,errR,errB,errreg,yLre,yRre,yBre] = ...
  objfunIntExtRot2(x,yL,yR,yB,crig,lambda,verifyOpts)
% objective fcn for intrinsic + extrinsic/rot optim
% x: delta-values for calib params
% x(1:3): delta-om.LB
% x(4:6): delta-om.RB
% x(7:8): delta-cc_L/1000
% x(9:10): delta-cc_R/1000
% x(11:12): delta-cc_B/1000
% x(13:14): delta-fc_L/5000
% x(15:16): delta-fc_R/5000
% x(17:18): delta-fc_B/5000

assert(isequal(18,numel(x),numel(lambda)));

x = x(:);
domLB = x(1:3);
domRB = x(4:6);
dccL = x(7:8)*1e3;
dccR = x(9:10)*1e3;
dccB = x(11:12)*1e3;
dfcL = x(13:14)*5e3;
dfcR = x(15:16)*5e3;
dfcB = x(17:18)*5e3;

crigMod = crig.copy();

try
  % update om
  crigMod.om.LB = crigMod.om.LB+domLB;
  crigMod.om.BL = crigMod.om.BL-domLB;
  crigMod.om.RB = crigMod.om.RB+domRB;
  crigMod.om.BR = crigMod.om.BR-domRB;
  
  % update R based on om
  fldsRecompute = fieldnames(crigMod.om);
  for f=fldsRecompute(:)',f=f{1}; %#ok<FXSET>
    crigMod.R.(f) = rodrigues(crigMod.om.(f));
  end
  fldsRm = setdiff(fieldnames(crigMod.R),fldsRecompute);
  crigMod.R = rmfield(crigMod.R,fldsRm);
  
  crigMod.int.L.cc = crigMod.int.L.cc+dccL;
  crigMod.int.R.cc = crigMod.int.R.cc+dccR;
  crigMod.int.B.cc = crigMod.int.B.cc+dccB;
  crigMod.int.L.fc = crigMod.int.L.fc+dfcL;  
  crigMod.int.R.fc = crigMod.int.R.fc+dfcR;
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