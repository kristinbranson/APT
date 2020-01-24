function trx = compute_landmark_velmag(trx)

if ~isfield(trx,'dx'),
  trx = compute_landmark_dx(trx);
end
if ~isfield(trx,'dy'),
  trx = compute_landmark_dy(trx);
end
trx.velmag = trx.dx;
trx.velmag.data = sqrt(trx.dx.data.^2 + trx.dy.data.^2);
