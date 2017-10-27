function [fvals,fvalsc] = compfeatures(prm,tGT,fimidxs)
% tGT: MFT table with 
%  * imForeCanonRoi: [roinr roinc] single, bgsubed foreground px
%  * imForeCanonRoiBWLleg: [roinr roinc] uint8, label vector labeling
%    foreground px that are legs 1-6
% fimidxs: [numFtrsx2], numFtrs=nFern*fernD
%
% fvals: [nFern x fernD x nFgPx] computed feature values
%   (diff-of-px-intensities) at all foreground px found/considered
% fvalsc: [nFgPx x 1] class/leg labeling 3rd dim of fvals

%ctrIdx = (nptsFtrSqrROI+1)/2;
%ctrIdx = nan; % we are not using

NLEG = 6;
nFern = prm.ftr.nFern;
fernD = prm.ftr.fernD;
ftrRad = prm.ftr.rad;
ftrBoxSz = 2*ftrRad+1;
nGT = height(tGT);

fvals = nan(nFern,fernD,0); % acc of all feature vals
fvalsc = nan(0,1); % legs labeling 3rd dim of ftrValsBig
for iGT=1:nGT
  if mod(iGT,100)==0
    disp(iGT);
  end
  
  imFC = tGT.imForeCanon{iGT};
  imFCBWLleg = tGT.imForeCanonBWLleg{iGT};
  
  for leg=1:NLEG
    [legPtI,legPtJ] = find(imFCBWLleg==leg);
    nlegPt = numel(legPtI);
    for ilegPt=1:nlegPt
      i = legPtI(ilegPt);
      j = legPtJ(ilegPt);
      [imFtrBox,npadl,npadu] = padgrab(imFC,0,i-ftrRad,i+ftrRad,...
        j-ftrRad,j+ftrRad);
      assert(all(npadl==0));
      assert(all(npadu==0));
      szassert(imFtrBox,[ftrBoxSz ftrBoxSz]);
      
      ftrVals = imFtrBox(fimidxs);
      szassert(ftrVals,[nFern fernD 2]);
      ftrVals = ftrVals(:,:,1)-ftrVals(:,:,2);
      %ftrCtr = imFtrBox(ctrIdx);
      %ftrVals = ftrCtr-ftrVals;      
      
      fvals(:,:,end+1) = ftrVals; %#ok<AGROW>
      fvalsc(end+1,1) = leg; %#ok<AGROW>
    end
  end
end
