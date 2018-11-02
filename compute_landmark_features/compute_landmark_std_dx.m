function trx = compute_landmark_std_dx(trx)

if ~isfield(trx,'dx'),
  trx = compute_landmark_dx(trx);
end
trx.std_dx = trx.dx;
trx.std_dx.data = std(trx.dx.data,1,1);
