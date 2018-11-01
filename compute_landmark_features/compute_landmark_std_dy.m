function trx = compute_landmark_std_dy(trx)

if ~isfield(trx,'dy'),
  trx = compute_landmark_dy(trx);
end
trx.std_dy = trx.dy;
trx.std_dy.data = std(trx.dy.data,1);
