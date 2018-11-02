function trx = compute_landmark_mean_dx(trx)

if ~isfield(trx,'dx'),
  trx = compute_landmark_dx(trx);
end
trx.mean_dx = trx.dx;
trx.mean_dx.data = mean(trx.dx.data,1);
