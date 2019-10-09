function [s,tBalCats,gPartCatsAdd,prtCat2Split] = stratifiedGroupSplit(...
  nFold,gBal,gPart,varargin)
% Stratified/Partitioned k-fold splitter
%
% Split data for k-fold xval, trying to balance gBal (approximately equal 
% numbers of each gBal value in each split), while fully partitioning
% gPart (each value of gPart occurs in only one split).
%
% nfold: number of splits
% gBal: [n] grouping vector to be balanced among splits as best as possible
% gPart: [n] grouping vector to be partitioned among splits
%
% s: [n] split indicator vector
% tBalCats: [nbalcats x ncol] sortedsummary table for gBal
% tPrtCats: [nprtcats x ncol] etc
% prtCat2Split: [nprtcats x 1] mapping from gPart cat->split

shufflePartCats = myparse(varargin,...
  'shufflePartCats',false... % if true, this randomizes the order of considering/adding gPart categories
                        ... % if false, this adds gPart cats in decreasing order of size
  );

assert(isvector(gBal) && isvector(gPart) && numel(gBal)==numel(gPart));
N = numel(gBal);

gbalC = categorical(gBal);
gprtC = categorical(gPart);
tBalCats = sortedsummary(gbalC);
tPrtCats = sortedsummary(gprtC);
gbalC = reordercats(gbalC,tBalCats.cats);
gprtC = reordercats(gprtC,tPrtCats.cats);
nBalCats = height(tBalCats);
nPrtCats = height(tPrtCats);

fprintf(1,'Crosstab: \n');
[ctbl,~,~,ctbllbls] = crosstab(gbalC,gprtC);
m = num2cell(ctbl);
[nr,nc] = size(m);
m = [ctbllbls(1:nr,1) m];
m = [[{[]} ctbllbls(1:nc,2)']; m];
disp(m);

% find the target/desired proportions of gbal in each split
assert(sum(tBalCats.cnts)==N);
tBalCats.proportion = tBalCats.cnts/N;
fprintf(1,'Desired proportions of gbal:\n');
disp(tBalCats);

% alg is to start with biggest gPart chunks. For each chunk, check its
% gBal. Put that chunk in the split that is currently most underweight that
% gBal value.

gPartCatsAdd = tPrtCats.cats;
if shufflePartCats
  gPartCatsAdd = gPartCatsAdd(randperm(numel(gPartCatsAdd)));  
end

s = nan(N,1); % s(i) gives fold index for row i
splitbalcnts = zeros(nFold,nBalCats); % splitcnts(ifold,ibal) gives running 
% count of how many rows are for split isplit, ibal'th gbal category
prtCat2Split = nan(nPrtCats,1); % prtCat2Split(iprt) gives split index for gPartCatsAdd{iprt}
gbalCats = categories(gbalC);
assert(isequal(gbalCats,tBalCats.cats));
for iPrtCat=1:nPrtCats  
  partVal = gPartCatsAdd{iPrtCat};
  tfThisPrt = partVal==gprtC;
  gBalCthisPart = gbalC(tfThisPrt);
  gBalCthisPartCnts = countcats(gBalCthisPart);
  [~,ibal] = max(gBalCthisPartCnts);
  nbalcats = nnz(gBalCthisPartCnts>0);
  if nbalcats>1
    warningNoTrace('gpart=%s, %d gbal values encountered. Using mode, gbal=%s.',...
      partVal,nbalcats,gbalCats{ibal});
  end
    
  % find ideal/target number of ibal in each split
  splitcnts = sum(splitbalcnts,2);
  splitcntsidealibal = splitcnts * tBalCats.proportion(ibal);
  ibalscore = splitcntsidealibal - splitbalcnts(:,ibal); 
  % [nfold x 1]. score for assigning to each fold based on balancing gBal.
  % bigger is better; units are delta-[rows], eg +5 means that fold is
  % short of its ideal splitbalcnt for ibal by 5 rows.
  
  % Try also to balance splitcnts
  ntot = sum(splitcnts);
  splitcntsideal = ntot/nFold;
  splitcntsscore = splitcntsideal-splitcnts; 
  % [nfold x 1]. score for assigning to each fold based on balancing 
  % splits. bigger is better; units are [rows]

  szassert(ibalscore,[nFold 1]);
  szassert(splitcntsscore,[nFold 1]);
  totscore = ibalscore+splitcntsscore;
  [~,isplit] = max(totscore); % this split is most underweight (or earliest for ties)
  
  nNewRows = nnz(tfThisPrt);
  splitbalcnts(isplit,ibal) = splitbalcnts(isplit,ibal) + nNewRows;
  prtCat2Split(iPrtCat) = isplit;
  s(tfThisPrt) = isplit;
end

assert(~any(isnan(s)));
  