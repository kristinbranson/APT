function trx = compute_landmark_absdy(trx)

if ~isfield(trx,'dy'),
  trx = compute_landmark_dy(trx);
end
trx.absdy = trx.dy;
trx.absdy.data = abs(trx.dy.data);
