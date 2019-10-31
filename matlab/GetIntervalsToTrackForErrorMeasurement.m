function intervals = GetIntervalsToTrackForErrorMeasurement(movieidx,flies,ts,trxfiles,varargin)

winrad = myparse(varargin,'winrad',200);

nmovies = max(movieidx);
nflies = nan(1,nmovies);
intervals = nan(0,4);
se = strel(ones(1,2*winrad+1));
for moviei = 1:nmovies,
  if ~any(movieidx==moviei),
    nflies(moviei) = 0;
  else
    nflies(moviei) = max(flies(movieidx==moviei));
  end
  td = load(trxfiles{moviei});
  for fly = 1:nflies(moviei),
    idxcurr = movieidx==moviei&flies==fly;
    if ~any(idxcurr),
      continue;
    end
    nframes = td.trx(fly).nframes;
    islabeled = false(1,nframes);
    islabeled(ts(idxcurr)-td.trx(fly).firstframe+1) = true;
    dotrack = imdilate(islabeled,se);
    [i0s,i1s] = get_interval_ends(dotrack);
    newintervals = nan(numel(i0s),4);
    newintervals(:,1) = moviei;
    newintervals(:,2) = fly;
    newintervals(:,3) = i0s+td.trx(fly).firstframe-1;
    newintervals(:,4) = i1s+td.trx(fly).firstframe-1-1;
    intervals(end+1:end+numel(i0s),:) = newintervals;
  end
end

