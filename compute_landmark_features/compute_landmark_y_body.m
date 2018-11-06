function [trx] = compute_landmark_y_body(trx)

[npts,D,T,ntargets] = size(trx.pos_body);
% trx.pos is npts x D x T x ntargets
trx.y_body.data = reshape(trx.pos_body(:,2,:,:),[npts,T,ntargets]);
if trx.realunits && ~isempty(trx.pxpermm),
  trx.y_body.data = trx.y_body.data * trx.pxpermm;
  trx.y_body.units = parseunits('mm');
else
  trx.y_body.units = parseunits('px');
end