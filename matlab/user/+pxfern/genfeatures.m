function fimidxs = genfeatures(prm)
% fimidxs: [numFtrsx2]
% fimidxs(i,:) are linear indices into the ftrIm (square ROI).
% The feature value will be 
% ftr = imforecanon(fimidxs(i,1))-imforecanon(fimidxs(i,2));

% Assume imforecanon, foreground image canonically rotated. Either you are
% generating orientation-aware features and then sampling the raw image, or 
% sampling the raw image first (canonical rot) and using raw ftrs. Same
% diff

ftrRad = prm.ftr.rad;
ftrBoxSz = 2*ftrRad+1;
nptsFtrSqrROI = ftrBoxSz^2;

nFern = prm.ftr.nFern;
fernD = prm.ftr.fernD;
numFtrs = nFern*fernD;
fimidxs = nan(numFtrs,2); 
for i=1:numFtrs
  fimidxs(i,:) = randsample(nptsFtrSqrROI,2);
end
fimidxs = reshape(fimidxs,[nFern fernD 2]);
