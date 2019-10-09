function [roi,cjitter,rjitter] = cropsmart(imnc,imnr,roinc,roinr,...
  roiCtrCol,roiCtrRow,varargin)
% Crop an roi around a given center point. Do something nondumb when the 
% center is close to an edge of the image.
%
% imnr: positive integer, number of rows in image.
% imnc: " cols ".
% roinr: etc, number of rows in roi.
% roinc: etc.
% roiCtrRow: row coordinate of desired center of roi. Need not be an
%   integer
% roiCtrCol: col ".
%
% roi: [collo colhi rowlo rohi] OR EQUIVALENTLY [xlo xhi ylo yhi]
% cjitter: actual col/x jitter applied
% rjitter: actual row/y jitter applied

[rowjitter,coljitter] = myparse(varargin,...
  'rowjitter',0,... % optional. amount to jitter in row-direction. Special care is taken when the roi is near an image edge.
  'coljitter',0 ... % optional. amount to jitter in col-direction
  );

if roinr>imnr
  error('roinr is greater than imnr.');
end
if roinc>imnc
  error('roinc is greater than imnc.');
end

[rlo,rhi,rjitter] = lclJitter1D(roinr,roiCtrRow,imnr,rowjitter);
[clo,chi,cjitter] = lclJitter1D(roinc,roiCtrCol,imnc,coljitter);
roi = [clo chi rlo rhi];

function [zlo,zhi,zjitter] = lclJitter1D(roisz,roictr,imnz,maxjitter)
roirad = roisz/2;
zloRaw = roictr-roirad; % not rounded, doesn't account for im edges
zhiRaw = roictr+roirad;
zloRoom = zloRaw-0.5; % amt of wiggle room to low-row size
zhiRoom = imnz+0.499-zhiRaw; % etc
if zloRoom<=0
  zlo = 1;
  zhi = zlo+roisz-1;
  zjitter = 0;
elseif zhiRoom<=0
  zlo = imnz-roisz+1;
  zhi = imnz;
  zjitter = 0;
else
  maxJit = min([zloRoom;zhiRoom;maxjitter]); % jitter symmetrically; cap at amount of room available
  maxJit = max(maxJit,0);
  zjitter = 2*(rand-0.5)*maxJit;
  zloJit = round(zloRaw+zjitter);
  zhiJit = round(zhiRaw+zjitter);

  zlo = ceil(zloJit);
  zhi = floor(zhiJit);
  if zlo==zloJit && zhi==zhiJit
    % edge case
    zhi = zhi-1;
  end
end
assert(zlo>=1);
assert(zhi<=imnz);
assert(zhi-zlo+1==roisz);
