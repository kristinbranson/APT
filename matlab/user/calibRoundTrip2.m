function [yLre,yRre,errL,errR,errLfull,errRfull] = ...
  calibRoundTrip2(yL,yR,yB,crig)
% Use L+B to reconstruct R
% Use R+B to reconstruct L
%
% yLre: reconstructed L
% yRre: etc

[XLlb,XBlb] = crig.stereoTriangulateLB(yL,yB);
[XBbr,XRbr] = crig.stereoTriangulateBR(yB,yR);

XRlb = crig.camxform(XBlb,'br');
XLbr = crig.camxform(XBbr,'bl');

xpR_re = crig.project(XRlb,'R');
xpL_re = crig.project(XLbr,'L');
yRre = crig.x2y(xpR_re,'R');
yLre = crig.x2y(xpL_re,'L');

errLfull = sqrt(sum((yL-yLre).^2,2));
errRfull = sqrt(sum((yR-yRre).^2,2));
errL = mean(errLfull); % mean L2 distance (in px) across all GT pts
errR = mean(errRfull);
