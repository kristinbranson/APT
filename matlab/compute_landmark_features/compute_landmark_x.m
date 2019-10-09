function [trx] = compute_landmark_x(trx)

[npts,D,T,ntargets] = size(trx.pos);
% trx.pos is npts x D x T x ntargets
trx.x.data = reshape(trx.pos(:,1,:,:),[npts,T,ntargets]);
if trx.realunits && ~isempty(trx.pxpermm),
  trx.x.data = trx.x.data * trx.pxpermm;
  trx.x.units = parseunits('mm');
else
  trx.x.units = parseunits('px');
end