function trx = compute_landmark_mean_dy(trx)

if ~isfield(trx,'dy'),
  trx = compute_landmark_dy(trx);
end
trx.mean_dy = trx.dy;
trx.mean_dy.data = mean(trx.dy.data,1);
