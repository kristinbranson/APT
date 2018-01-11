function hfern = fernhist(prm,fvals,fvalsc)
% Fern bin, using thresholds or however
%
% hfern: [NLEG x 2^D x nFern] fern counts per class

NLEG = 6;
nFern = prm.ftr.nFern;
fernD = prm.ftr.fernD;

nFgPx = size(fvals,3); % fg pxs where ftrs were computed
szassert(fvals,[nFern fernD nFgPx]);
szassert(fvalsc,[nFgPx 1]);

hfern = zeros(NLEG,2^fernD,nFern);
for iPx=1:nFgPx
  if mod(iPx,1e3)==0
    disp(iPx);
  end
    
  fvalsthis = fvals(:,:,iPx);
  fvalsInd = fvalsthis>0;
  %szassert(ftrValsThis,size(ftrValsBigPtlThresh));
  %ftrValsInd = ftrValsThis<ftrValsBigPtlThresh;
  
  fernIdxs = Ferns.indsSimple(fvalsInd);
  szassert(fernIdxs,[nFern 1]);
  leg = fvalsc(iPx);
  for iFern=1:nFern
    hfern(leg,fernIdxs(iFern),iFern) = hfern(leg,fernIdxs(iFern),iFern)+1;
  end
end
