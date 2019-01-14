function [trx] = compute_landmark_dy(trx)

[npts,D,T,ntargets] = size(trx.pos);
% trx.pos is npts x D x T x ntargets
if T == 0,
  trx.dy.data = nan([npts,T,ntargets]);
else
  trx.dy.data = reshape(trx.pos(:,2,2:end,:)-trx.pos(:,2,1:end-1,:),[npts,T-1,ntargets]);
  trx.dy.data(:,end+1,:) = nan;
end
if trx.realunits && ~isempty(trx.fps) && ~isempty(trx.pxpermm),
  trx.dy.data = trx.dy.data * trx.pxpermm / trx.fps;
  trx.dy.units = parseunits('mm/s');
else
  trx.dy.units = parseunits('px/fr');
end