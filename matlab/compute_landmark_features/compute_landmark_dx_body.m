function [trx] = compute_landmark_dx_body(trx)

[npts,D,T,ntargets] = size(trx.pos_body);
% trx.pos_body is npts x D x T x ntargets
if T == 0,
  trx.dx_body.data = nan([npts,T,ntargets]);
else
  trx.dx_body.data = reshape(trx.pos_body_prev(:,1,2:end,:)-trx.pos_body(:,1,1:end-1,:),[npts,T-1,ntargets]);
  trx.dx_body.data(:,end+1,:) = nan;
end
if trx.realunits && ~isempty(trx.fps) && ~isempty(trx.pxpermm),
  trx.dx_body.data = trx.dx_body.data * trx.pxpermm / trx.fps;
  trx.dx_body.units = parseunits('mm/s');
else
  trx.dx_body.units = parseunits('px/fr');
end