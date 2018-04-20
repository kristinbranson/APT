function rc = simpletrain(tblVar,imgVar,viewIdx,trnSetVar,trnSetCol,...
  prmVar,resFile,varargin)
%
% tblVar: varName of table to use. mat.(tblVar) has size [Nxncols]
% imgVar: varName of image cell array I to use. mat.(imgVar) has size [NxNvw]
% viewIdx: scalar integer in 1..Nvw
% trnSetVar: varName of training set array to use. mat.(trnSetVar) is a 
%   logical of size [NxNsplit]
% trnSetCol: col index into mat.(trnSetVar) to use
% prmVar: varName of parcameter struct/variable ("old style")
% resFile: results filename
% varargin: list of matfiles that will be loaded into mat.*
%
% rc: trained RC. this is saved

if ischar(viewIdx)
  viewIdx = str2double(viewIdx);
end
if ischar(trnSetCol)
  trnSetCol = str2double(trnSetCol);
end
  
matfiles = varargin;

% Load all matfiles
mat = struct();
for i=1:numel(matfiles)  
  tmp = load(matfiles{i});
  fprintf(1,'Loaded mat: %s\n',matfiles{i});
  mat = structmerge(mat,tmp);
end

tbl = mat.(tblVar);
I = mat.(imgVar);
trnset = mat.(trnSetVar);
sPrm = mat.(prmVar);
n = height(tbl);
assert(isequal(size(I),[n 2]) && iscell(I));
assert(size(trnset,1)==n && islogical(trnset));
assert(isstruct(sPrm));
fprintf(1,'Vars (tbl I trnset trnsetcol prm): %s %s %s %d %s\n',...
  tblVar,imgVar,trnSetVar,trnSetCol,prmVar);

tfTrn = trnset(:,trnSetCol);
nTrn = nnz(tfTrn);
fprintf(1,'... nTrn=%d.\n',nTrn);

% get pLbl
fprintf(1,'... view %d.\n',viewIdx);
szassert(tbl.pLbl,[n 20]);
p = tbl.pLbl(tfTrn,[1:5 11:15] + 5*(viewIdx-1));
I = I(tfTrn,viewIdx);
bboxes = makeBBoxes(I);

% Do it
rc = RegressorCascade(sPrm);
rc.init();
[~,~,p0,p0info] = rc.trainWithRandInit(I,bboxes,p);

nowstr = datestr(now,'yyyymmddTHHMMSS');
fprintf(1,'Done training at %s\n',nowstr);

if exist(resFile,'file')>0
  [resFileP,resFileF,resFileE] = myfileparts(resFile);
  resFile = sprintf('%s_%s%s',resFileF,nowstr,resFileE);
  resFile = fullfile(resFileP,resFile);
end
save(resFile,'tblVar','imgVar','viewIdx','trnSetVar','trnSetCol',...
  'prmVar','varargin','nowstr','rc','p0','p0info');
