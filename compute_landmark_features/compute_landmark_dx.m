function [trx] = compute_landmark_dx(trx)

[npts,D,T,ntargets] = size(trx.pos);
% trx.pos is npts x D x T x ntargets
trx.dx.data = reshape(trx.pos(:,1,2:end,:)-trx.pos(:,1,1:end-1,:),...
  [npts,max(0,T-1),ntargets]);
trx.dx.data(:,end+1,:) = nan;
if trx.realunits && ~isempty(trx.fps) && ~isempty(trx.pxpermm),
  trx.dx.data = trx.dx.data * trx.pxpermm / trx.fps;
  trx.dx.units = parseunits('mm/s');
else
  trx.dx.units = parseunits('px/fr');
end