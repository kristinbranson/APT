function trx = compute_landmark_max_velmag(trx)

if ~isfield(trx,'velmag'),
  trx = compute_landmark_velmag(trx);
end
trx.max_velmag = trx.velmag;
trx.max_velmag.data = max(trx.velmag.data,[],1);
