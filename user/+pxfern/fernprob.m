function [pfern,hfernN,hfernZ] = fernprob(prm,hfern)
% pfern: [NLEG x 2^D x nFern]. pfern(leg,k,iFern) gives (regularized)
%   probability P(Fern_i==k | C==leg)
% hfernN: [NLEG x 2^D x nFern] normalized raw fern (non-regularized prob)
%   with sum(hfernN,2)==1 for all classes/ferns
% hfernZ: [NLEG x 1] sum of fern counts for each class

NLEG = 6;
nFern = prm.ftr.nFern;
fernD = prm.ftr.fernD;

% normalized fern
hfernZ = sum(hfern,2);
hfernN = bsxfun(@rdivide,hfern,hfernZ);
hfernZ = squeeze(hfernZ);
assert(isequal(hfernZ,repmat(hfernZ(:,1),1,nFern)));
hfernZ = hfernZ(:,1);

%% Create probability estimator
K = 2^fernD;
Nr = prm.ftr.regN;
% pFern(leg,k,iFern) gives regularized prob
%  P(Fern_i=k | C=leg)
pfern = nan(size(hfern)); 
for iFern=1:nFern
  for leg=1:NLEG
    Nk = hfern(leg,:,iFern);
    p = (Nk+Nr)/(sum(Nk)+K*Nr);
    pfern(leg,:,iFern) = p;
  end
end
