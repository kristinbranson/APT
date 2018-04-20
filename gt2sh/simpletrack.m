function simpletrack(imgVar,trkSetVar,trkSetCol,viewIdx,prmVar,...
  tblVar,resID,varargin)
%
% imgVar: varName of image cell array I to use. mat.(imgVar) has size [NxNvw]
% trkSetVar: (optional) varName of set array to use. mat.(trkSetVar) is a logical of size [NxNsplit]
% trkSetCol: col index into mat.(trkSetVar) to use
%   NOTE: trkSetVar,trkSetCol can be empty, in which case all imgVars are tracked
% viewIdx: scalar integer in 1..Nvw
% prmVar: varName of parcameter struct/variable ("old style")
% tblVar: (optional) varName of table to use. mat.(tblVar) has size
%   [Nxncols]. Used if GT tracking err is desired
% resID: results ID, base for output/results file
% varargin: list of matfiles that will be loaded into mat.*

if ischar(trkSetCol)
  trkSetCol = str2double(trkSetCol);
end
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

tfTrkSet = ~isempty(trkSetVar);
tfTbl = ~isempty(tblVar);

I = mat.(imgVar);
if tfTrkSet
  trkSet = mat.(trkSetVar);
end
sPrm = mat.(prmVar);
if tfTbl
  tbl = mat.(tblVar);
end

[n,nvw] = size(I);
assert(nvw==2);
if tfTrkSet
  assert(size(trkSet,1)==n && islogical(trkSet));
end
assert(isstruct(sPrm));
if tfTbl
  assert(height(tbl)==n);
end
fprintf(1,'Vars (I trkSet trkSetCol viewIdx prm tbl resID): %s %s %d %d %s %s %s\n',...
  imgVar,trkSetVar,trkSetCol,viewIdx,prmVar,tblVar,resID);

if tfTrkSet
  tfTrk = trkSet(:,trkSetCol);
else
  tfTrk = true(n,1);
end
nTrk = nnz(tfTrk);
fprintf(1,'... nTrk=%d.\n',nTrk);

% I, bbs
I = I(tfTrk,viewIdx);
bboxes = makeBBoxes(I);

% Track
rc = mat.rc;
[p_t,pIidx,p0,p0info] = rc.propagateRandInit(I,bboxes,sPrm.TestInit);
RT = sPrm.TestInit.Nrep;
trkD = rc.prmModel.D;
Tp1 = rc.nMajor+1;
pTstT = reshape(p_t,[nTrk RT trkD Tp1]);
nowstr = datestr(now,'yyyymmddTHHMMSS');
fprintf(1,'Done training at %s\n',nowstr);

% Prune
assert(~strcmp(sPrm.Prune.method,'smoothed trajectory'));
% Take MD table nec only for smoothed traj pruning method
tblFake = table(nan(nTrk,1),nan(nTrk,1),nan(nTrk,1),'VariableNames',MFTable.FLDSID);
[pTstTRed,pruneMD] = CPRLabelTracker.applyPruning(pTstT(:,:,:,end),tblFake,sPrm.Prune);
szassert(pTstTRed,[nTrk trkD]);
fprintf(1,'Done pruning.\n');

% (optional) Compare
if tfTbl
  szassert(tbl.pLbl,[n 20]);
  pGT = tbl.pLbl(tfTrk,[1:5 11:15] + 5*(viewIdx-1));
  
  dp = pGT-pTstTRed;
  dp = reshape(dp,[nTrk,5,2]); % i,ipt,x/y
  errTrk = sqrt(sum(dp.^2,3));
  szassert(errTrk,[nTrk 5]);

  fprintf(1,'Mean err: %s\n',mat2str(nanmean(errTrk,1)));
  fprintf(1,'Err ptiles: \n');
  PTILES = [10 50 90];
  disp(prctile(errTrk,PTILES));
else
  errTrk = [];
end

% results
if tfTrkSet
  resFile = sprintf('%s_vw%d_col%d.mat',resID,viewIdx,trkSetCol);
else
  resFile = sprintf('%s_vw%d_all.mat',resID,viewIdx);
end
if exist(resFile,'file')>0
  [resFileP,resFileF,resFileE] = myfileparts(resFile);
  resFile = sprintf('%s_%s%s',resFileF,nowstr,resFileE);
  resFile = fullfile(resFileP,resFile);
end
save(resFile,'-mat','imgVar','trkSetVar','trkSetCol','viewIdx',...
  'prmVar','tblVar','resID','varargin',...
  'nowstr','p0','p0info','pTstTRed','pruneMD','errTrk');
