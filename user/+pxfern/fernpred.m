function fvalscpred = fernpred(prm,pfern,fvals)
% given observed features:
% Class = argmax_C P(observed features | C)
%       = argmax_C Prod_{iFern over nFern ferns} P(iFern | C)
%
% pfern: [NLEG x 2^D x nFern]
% fvals: [nFern x fernD x nFgPx] computed feature values
%   (diff-of-px-intensities)
%
% fvalscpred: [nFgPx x 1] predicted class/leg labeling 3rd dim of fvals
% fvalscpredconf: XXX confidence/"probability gap" to 2nd place?

NLEG = 6;
nFern = prm.ftr.nFern;
fernD = prm.ftr.fernD;

nFgPx = size(fvals,3); % fg pxs where ftrs were computed
szassert(fvals,[nFern fernD nFgPx]);

fvalscpred = nan(nFgPx,1);
for iPx=1:nFgPx
  fvalsThis = fvals(:,:,iPx);
  fvalsInd = fvalsThis>0;
  %szassert(ftrValsThis,size(ftrValsBigPtlThresh));
  %ftrValsInd = ftrValsThis<ftrValsBigPtlThresh;  
  fernIdxs = Ferns.indsSimple(fvalsInd);
  szassert(fernIdxs,[nFern 1]);
  
  pleg = ones(NLEG,1); % running probability of getting each leg
  for iFern=1:nFern
    pleg = pleg.*pfern(:,fernIdxs(iFern),iFern);
  end
  [~,fvalscpred(iPx)] = max(pleg);  
  
  if mod(iPx,1e3)==0
    disp(iPx);
  end
end
