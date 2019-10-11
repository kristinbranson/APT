function [yLre,yRre,yBre,errL,errR,errB,errLfull,errRfull,errBfull] = ...
  calibRoundTrip(yL,yR,yB,crig)

[XLlb,XBlb] = crig.stereoTriangulateLB(yL,yB);
[XBbr,XRbr] = crig.stereoTriangulateBR(yB,yR);

% AL: sanity check, checks out
% [XLlb_2,XBlb_2] = crig.stereoTriangulateCropped(yL,yB,'L','B');
% [XBbr_2,XRbr_2] = crig.stereoTriangulateCropped(yB,yR,'B','R');

xpL_re = crig.project(XLlb,'L');
xpR_re = crig.project(XRbr,'R');
xpB_re = cat(3,crig.project(XBlb,'B'),crig.project(XBbr,'B'));
xpB_re = mean(xpB_re,3);
yLre = crig.x2y(xpL_re,'L');
yRre = crig.x2y(xpR_re,'R');
yBre = crig.x2y(xpB_re,'B');

errLfull = sqrt(sum((yL-yLre).^2,2));
errRfull = sqrt(sum((yR-yRre).^2,2));
errBfull = sqrt(sum((yB-yBre).^2,2));
errL = mean(errLfull); % mean L2 distance (in px) across all GT pts
errR = mean(errRfull);
errB = mean(errBfull);
