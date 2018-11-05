function trx = compute_landmark_absdx(trx)

if ~isfield(trx,'dx'),
  trx = compute_landmark_dx(trx);
end
trx.absdx = trx.dx;
trx.absdx.data = abs(trx.dx.data);
