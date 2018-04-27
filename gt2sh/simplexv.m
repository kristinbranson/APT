function simplexv(imgVar,xyLblVar,viewIdx,xvSetVar,prmVar,resID,varargin)
%
% imgVar: varName of image cell array I to use. mat.(imgVar) has size [NxNvw]
% xyLblVar: varName of xyLbl array. [nxnptx2xnVw]. row,ipt,x/y,vw
% viewIdx: scalar integer in 1..Nvw
% xvVar: varName of xv split array to use. mat.(xvVar) is a logical of size
%   [nxNsplit] where sum(...,1) is typically all equalish and sum(...,2) is
%   all 1's. 
% prmVar: varName of parameter struct/variable ("old style")
% resID: base for results filename
% varargin: list of matfiles that will be loaded into mat.*

if ischar(viewIdx)
  viewIdx = str2double(viewIdx);
end
  
matfiles = varargin;

% Load all matfiles
mat = struct();
for i=1:numel(matfiles)  
  tmp = load(matfiles{i});
  fprintf(1,'Loaded mat: %s\n',matfiles{i});
  mat = structmerge(mat,tmp);
end

I = mat.(imgVar);
xyLbl = mat.(xyLblVar);
xvSplits = mat.(xvSetVar);
sPrm = mat.(prmVar);

n = size(xyLbl,1);
szassert(xyLbl,[n 5 2 2]);
szassert(I,[n 2]);
assert(iscell(I));
nsplit = size(xvSplits,2);
szassert(xvSplits,[n nsplit]);
assert(islogical(xvSplits));
assert(isstruct(sPrm));
fprintf(1,'Vars (I xyLbl viewIdx xvSet prm resID): %s %s %d %s %s %s\n',...
  imgVar,xyLblVar,viewIdx,xvSetVar,prmVar,resID);
fprintf(1,'... nSplits=%d.\n',nsplit);
assert(all(sum(xvSplits,2)==1),'Invalid xv split defn.');

% get pLbl
p = reshape(xyLbl(:,:,:,viewIdx),n,10);
I = I(:,viewIdx);
bboxes = makeBBoxes(I);

% Do it
rcs = cell(1,nsplit);
errs = cell(1,nsplit);
for isplit=1:nsplit
  tfTst = xvSplits(:,isplit);
  tfTrn = ~tfTst;
  nTst = nnz(tfTst);
  nTrn = nnz(tfTrn);
  fprintf(1,'... split %d, nTrn=%d, nTst=%d\n',isplit,nTrn,nTst);

  rc = RegressorCascade(sPrm);
  rc.init();
  [~,~,p0,p0info] = rc.trainWithRandInit(I(tfTrn,:),bboxes(tfTrn,:),p(tfTrn,:));
  rcs{isplit} = rc;
  nowstr = datestr(now,'yyyymmddTHHMMSS');
  fprintf(1,'... done training at %s\n',nowstr);

  % Track
  [p_t,pIidx,p0,p0info] = rc.propagateRandInit(I(tfTst,:),bboxes(tfTst,:),sPrm.TestInit);
  RT = sPrm.TestInit.Nrep;
  trkD = rc.prmModel.D;
  Tp1 = rc.nMajor+1;
  pTstT = reshape(p_t,[nTst RT trkD Tp1]);
  nowstr = datestr(now,'yyyymmddTHHMMSS');
  fprintf(1,'... done tracking at %s\n',nowstr);
  
  % Prune
  assert(~strcmp(sPrm.Prune.method,'smoothed trajectory'));
  % Take MD table nec only for smoothed traj pruning method
  tblFake = table(nan(nTst,1),nan(nTst,1),nan(nTst,1),'VariableNames',MFTable.FLDSID);
  [pTstTRed,pruneMD] = CPRLabelTracker.applyPruning(pTstT(:,:,:,end),tblFake,sPrm.Prune);
  szassert(pTstTRed,[nTst trkD]);
  fprintf(1,'Done pruning.\n');
  
  %  Compare
  pGT = p(tfTst,:);
  szassert(pGT,size(pTstTRed));
  dp = pGT-pTstTRed;
  dp = reshape(dp,[nTst,5,2]); % i,ipt,x/y
  errTrk = sqrt(sum(dp.^2,3));
  szassert(errTrk,[nTst 5]);
  errs{isplit} = errTrk;
  
  fprintf(1,'Mean err: %s\n',mat2str(nanmean(errTrk,1)));
  fprintf(1,'Err ptiles: \n');
  PTILES = [50 75 90 95 97.5 99];
  disp(prctile(errTrk,PTILES));  
end

resFile = sprintf('%s__xv__%s__vw%d__%s.mat',resID,imgVar,viewIdx,xvSetVar);
if exist(resFile,'file')>0
  [resFileP,resFileF,resFileE] = myfileparts(resFile);
  resFile = sprintf('%s_%s%s',resFileF,nowstr,resFileE);
  resFile = fullfile(resFileP,resFile);
end
save(resFile,'xyLblVar','imgVar','viewIdx','xvSetVar',...
  'prmVar','resID','varargin','nowstr','rcs','errs');