function [trx] = compute_landmark_y(trx)

[npts,D,T,ntargets] = size(trx.pos);
% trx.pos is npts x D x T x ntargets
trx.y.data = reshape(trx.pos(:,2,:,:),[npts,T,ntargets]);
if trx.realunits && ~isempty(trx.pxpermm),
  trx.y.data = trx.x.data * trx.pxpermm;
  trx.y.units = parseunits('mm');
else
  trx.y.units = parseunits('px');
end