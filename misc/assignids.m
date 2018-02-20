function [bwl,bwlpre,splitCC,splitCCnew,pdfTgts] = ...
  assignids(imfore,frm,trx,pdfleg,pdflegXE,pdflegYE,varargin)
% 
% 
% imfore: foreground image
% frm: frame number
% trx: [nTgt] trx array
% pleg: [nrxnc] single-fly pdf for legs. MUST BE canonically rotated
% plegXE: [nc+1] edge xlocs for columns of pleg
% plegYE: [nr+1] edge ylocs for rows of pleg
%
% bwl: [same as imfore] labeled foreground
% splitCC: [nsplit] vector, list of original CCs that were split
% splitCCnew: [nsplit] cell, splitCCnew{i} gives the new CCs that
%   correspond to the original cc splitCC(i)
% pdfTgts: [nsplit] cell, pdfTgts{i} gives PDFs corresponding to each
%   target comprising the ith original CC

[bwthresh,ellSizeFac,pass2alg,...
  scalePdfLeg,scalePdfLegMeanA,scalePdfLegMeanB,...
  verbose] = myparse(varargin,...
  'bwthresh',5,...
  'ellSizeFac',1.1,... % multiply ellipse size (a/b) by this fudge factor. Usually slightly greater than 1 to marginally increase ellipse coverage
  'pass2alg','touch',... % either 'touch' or 'dist'. method for 2nd pass. 
  'scalePdfLeg',false,... % If true, scale/adjust pdfleg based per ellipse size (major/minor axes lengths)
  'scalePdfLegMeanA',nan,... % mean ellipse 'a' value (majoraxis/2), used when scalePdfLeg is true
  'scalePdfLegMeanB',nan,... % etc
  'verbose',false);

assert(all(imfore(:)>=0));
imforeabs = abs(imfore);
bw = imforeabs>bwthresh;
[bwl,bwln] = bwlabel(bw);
[imnr,imnc] = size(imfore);
imsz = imnr*imnc;
%stats = regionprops(forebwl,'Area');

[imgx,imgy] = meshgrid(1:imnc,1:imnr);

pdfEllVal = max(pdfleg(:));
pdflegXctr = (pdflegXE(1:end-1)+pdflegXE(2:end))/2;
pdflegYctr = (pdflegYE(1:end-1)+pdflegYE(2:end))/2;
[pdflegXg,pdflegYg] = meshgrid(pdflegXctr,pdflegYctr);
% x/y gridpoints for pdfleg

[trxtflive,trxxs,trxys,trxths,trxas,trxbs] = ...
  PxAssign.getTrxStuffAtFrm(trx,frm);

ccTrxCtrs = cell(bwln,1); % trx centers in each cc
ntgt = numel(trx);
for itgt=1:ntgt
  if trxtflive(itgt)
    x = round(trxxs(itgt));
    y = round(trxys(itgt));
    ccL = bwl(y,x);
    if ccL>0
      ccTrxCtrs{ccL}(1,end+1) = itgt;
    end
  end
end

% All trx have been assigned to a CC. Some CCs are not assigned to a trx
% (especially eg small noise pixels)

% Find and deal with any CCs that have multiple trx assigned, indicating
% two or more flies nearby+touching
ccMultiTrx = find(cellfun(@numel,ccTrxCtrs)>1);
nCCMultiTrx = numel(ccMultiTrx);

if nCCMultiTrx==0
  bwlpre = bwl;
  splitCC = zeros(0,1);
  splitCCnew = cell(0,1);
  pdfTgts = cell(0,1);
  return;
end

splitCCnew = cell(nCCMultiTrx,1);
pdfTgts = cell(nCCMultiTrx,1);
maxcc = bwln;
for iCC=1:nCCMultiTrx
  cc = ccMultiTrx(iCC);
  iTgtsCC = ccTrxCtrs{cc};
  nTgtsCC = numel(iTgtsCC);
  
  if verbose
    fprintf('CC %d has %d trx: %s\n',cc,nTgtsCC,mat2str(iTgtsCC));
  end
  
  % pdfTgts(:,:,j) is the likelihood of a pixel being in target iTgtsCC(j),
  % based on the ellipse and (probable) legs of trx(iTgtsCC(j))
  pdfTgtsI = nan(imnr,imnc,nTgtsCC);
  for j=1:nTgtsCC % loop over all targets assigned to this CC

    % Get trx info
    iTgt = iTgtsCC(j);
    assert(trxtflive(iTgt));
    trxx = trxxs(iTgt);
    trxy = trxys(iTgt);
    trxth = trxths(iTgt);
    trxa = trxas(iTgt);
    trxb = trxbs(iTgt);

    % optionally, rescale the pleg pdf for this ellipse
    if scalePdfLeg
      % trxa>MeanA => pdfleg is expanded in x-dir.
      % Is the normalization right though? we do not rescale the amplitude
      pdflegXguse = pdflegXg*trxa/scalePdfLegMeanA;
      pdflegYguse = pdflegYg*trxb/scalePdfLegMeanB;      
    else
      pdflegXguse = pdflegXg;
      pdflegYguse = pdflegYg;
    end
    
    % rotate the pleg pdf onto the x/y/th.    
    pdfLegItgt = readpdf(pdfleg,pdflegXguse,pdflegYguse,imgx,imgy,trxx,trxy,trxth);
    szassert(pdfLegItgt,[imnr imnc]);
    % pdfLegI should be normalized ie sum to 1
        
    pdfEllItgt = drawellipseim(trxx,trxy,trxth,...
      2*trxa*ellSizeFac,2*trxb*ellSizeFac,...
      imgx,imgy,pdfEllVal);
    % pdfEllI is not normalized
    
    pdfItgt = pdfLegItgt + pdfEllItgt; % don't normalize
    pdfTgtsI(:,:,j) = pdfItgt;
  end
  pdfTgts{iCC} = pdfTgtsI;
  
  % For each pixel in the CC, take the maximum
  tfcc = bwl==cc;
  pdfTgtsI = reshape(pdfTgtsI,[imnr*imnc nTgtsCC]);
  pdfTgtsFore = pdfTgtsI(tfcc(:),:);
  [~,loc] = max(pdfTgtsFore,[],2);
  % loc is an assignment into 1..nTgtsCC for each pixel in the CC
  
  ccidx = find(tfcc); % linear indices into bwl for current composite cc
  szassert(loc,[numel(ccidx) 1]);
  
  % all loc==1 pixels keep their existing cc. We create new ccs for higher 
  % locs
  newCCs = nan(1,nTgtsCC); % We are splitting cc up into nTgtsCC, one for each trx
  newCCs(1) = cc; % j=1 => first target => keeps cc
  for j=2:nTgtsCC % remaining targets, iTgtsCC(j)<->newCCs(j)
    newcc = maxcc+1;
    bwl(ccidx(loc==j)) = newcc;
    if verbose
      fprintf('Created new cc=%d with %d px\n',newcc,nnz(loc==j));
    end
    maxcc = newcc;
    newCCs(j) = newcc;
  end
  
  bwlpre = bwl; % save before final pass
  
  % 2nd/Final pass. The breaking up of cc into nTgtsCC will be imperfect. 
  % The most obvious class of imperfections is when pixels assigned to the 
  % new CCs are disjoint. We loop over all of the new CCs, and reassign 
  % any disconnected "fragments" per pass2alg.
  for j=1:nTgtsCC
    iTgt = iTgtsCC(j);
    trxx = round(trxxs(iTgt));
    trxy = round(trxys(iTgt));
    trxlinidx = trxy + (trxx-1)*imnr;
    
    newcc = newCCs(j);
    Icc = bwl==newcc;
    s = bwconncomp(Icc,4); % s is a breakdown of all pxs assigned to iTgt
%     if strcmp(pass2alg,'touch')
%       Lcc = labelmatrix(s);
%     end
    
    % find s/cc containing trx center
    iCompTgt = nan;
    for iComp=1:s.NumObjects
      if any(trxlinidx==s.PixelIdxList{iComp})
        iCompTgt = iComp;
        break;
      end
    end
    assert(~isnan(iCompTgt),'trx center must be in a connected component.');
    
    % for every other cc, do something per pass2alg
    for iComp=1:s.NumObjects
      if iComp==iCompTgt
        continue;
      end
      switch pass2alg
        case 'dist'
          % Take first pt in the pixelList; find trxcenter that is closest.
          % Use its CC
          linidx = s.PixelIdxList{iComp}(1);
          xtmp = imgx(linidx);
          ytmp = imgy(linidx);
          d2trx = (xtmp-trxxs).^2 + (ytmp-trxys).^2;
          szassert(d2trx,[ntgt 1]);
          [~,iTgtNearest] = min(d2trx);
          assert(trxtflive(iTgtNearest));
          jNearestTgt = find(iTgtNearest==iTgtsCC);
          assert(isscalar(jNearestTgt),'Nearest trx not found.');
          ccAssign = newCCs(jNearestTgt);          
        case 'touch'
          % Take the first CC encountered that is touching/adjacent to this
          % cc.
          linidxs = s.PixelIdxList{iComp}; % linear indices for this CC
          linidxs = linidxs(:);
          % linear indices of all pixels adjacent to CC
          linidxsadj = [linidxs+1 linidxs-1 linidxs+imnr linidxs-imnr];
          tfIB = linidxsadj>=1 & linidxsadj<=imsz;
          linidxsadj = linidxsadj(tfIB);
          bwladj = bwl(linidxsadj); % bwl values at adj pxs
          bwladj = bwladj(bwladj~=newcc & bwladj~=0); % bwl values at adj pxs, not in this comp
          ccAssign = mode(bwladj); % most frequent adjacent bwlval; (ALT: use first adj, but filter out tiny 1-px ccs etc)
        otherwise
          assert(false);
      end
      bwl(s.PixelIdxList{iComp}) = ccAssign;
    end
  end
  
  splitCCnew{iCC} = newCCs;
end
splitCC = ccMultiTrx;

