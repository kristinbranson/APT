function trx = compute_landmark_mean_velmag(trx)

if ~isfield(trx,'velmag'),
  trx = compute_landmark_velmag(trx);
end
trx.mean_velmag = trx.velmag;
trx.mean_velmag.data = mean(trx.velmag.data,1);
