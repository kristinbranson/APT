function idxsamples = SubsampleTrainingDataBasedOnMovement(ld,nTr)

nexps = numel(ld.expdirs);
nTr0 = numel(ld.expidx);
fracTr = nTr/nTr0;
idxsamples = [];
npts = size(ld.pts,1);
d = size(ld.pts,2);
for i = 1:nexps,
  
  idxcurr = find(ld.expidx==i);
  nTrcurr = round(fracTr*numel(idxcurr));
  if nTrcurr >= numel(idxcurr),
    idxsample = idxcurr;
  elseif nTrcurr == 0,
    continue;
  else
    
    % furthest-first based clustering
    
    X = reshape(ld.pts(:,:,idxcurr),[npts*d,numel(idxcurr)]);
    [~,~,idxsample] = furthestfirst(X',nTrcurr,'Start','sample');
    idxsample = sort(idxcurr(idxsample));
    
  end
  idxsamples = [idxsamples,idxsample]; %#ok<AGROW>
  
end

