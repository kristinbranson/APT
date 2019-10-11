function [trx] = compute_landmark_x_body(trx)

[npts,D,T,ntargets] = size(trx.pos_body);
% trx.pos is npts x D x T x ntargets
trx.x_body.data = reshape(trx.pos_body(:,1,:,:),[npts,T,ntargets]);
if trx.realunits && ~isempty(trx.pxpermm),
  trx.x_body.data = trx.x_body.data * trx.pxpermm;
  trx.x_body.units = parseunits('mm');
else
  trx.x_body.units = parseunits('px');
end