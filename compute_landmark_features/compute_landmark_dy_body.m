function [trx] = compute_landmark_dy_body(trx)

[npts,D,T,ntargets] = size(trx.pos_body);
% trx.pos_body is npts x D x T x ntargets
if T == 0,
  trx.dy_body.data = nan([npts,T,ntargets]);
else
  trx.dy_body.data = reshape(trx.pos_body_prev(:,2,2:end,:)-trx.pos_body(:,2,1:end-1,:),[npts,T-1,ntargets]);
  trx.dy_body.data(:,end+1,:) = nan;
end
if trx.realunits && ~isempty(trx.fps) && ~isempty(trx.pxpermm),
  trx.dy_body.data = trx.dy_body.data * trx.pxpermm / trx.fps;
  trx.dy_body.units = parseunits('mm/s');
else
  trx.dy_body.units = parseunits('px/fr');
end