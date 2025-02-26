%%
ROOT = 'f:\Dropbox\MultiviewFlyLegTracking';

ORIGPROJS = { ...
  'multiview labeling\romainJun22NewLabels.lbl'; ...
  'sep1616\sep1616-1531Romain.lbl'; ...
  'sep1316\sep1316-1606Romain.lbl'; ...
  'oct2916\oct2916-1420Romain.lbl'; ...
  'sep1516\sep1516-1537Romain.lbl'};
ORIGPROJSFULL = fullfile(ROOT,ORIGPROJS);
NPROJ = size(ORIGPROJS,1);

TRNDATAROOT = 'f:\romain\20161214MoreData';
TRNDATAFILES = { ...
  'romainJun22NewLabels_trnData_20161221T115407.mat'; ...
  'sep1616-1531Romain_trnData_20161221T121140.mat'; ...
  'sep1316-1606Romain_trnData_20161221T115759.mat'; ...
  'oct2916-1420Romain_trnData_20161221T122734.mat'; ...
  'sep1516-1537Romain_trnData_20170117T064551.mat'};
TRNDATAFILESFULL = fullfile(TRNDATAROOT,TRNDATAFILES);

%% Load projs; localize movieFiles
lbl = cellfun(@(x)load(x,'-mat'),ORIGPROJSFULL);
mfasets = arrayfun(@(x)regexprep(x.movieFilesAll,...
                        '\/localhome\/romain\/Dropbox \(HHMI\)','f:\\Dropbox'),lbl,'uni',0);
tdinfo = cellfun(@load,TRNDATAFILESFULL);

%% Create 2DBottom project
% lObj = Labeler();
IVIEW = 3;
for iprj=1:NPROJ
  % add movie
  lObj.movieAdd(mfasets{iprj}{IVIEW});
  lObj.movieSetGUI(lObj.nmovies);
  nfrm = lObj.nframes;
  
  % set labels
  lposVw = nan(18,2,nfrm);
  tFP = tdinfo(iprj).tFPtrn;
  nrows = size(tFP,1);
  pbig = tFP.p';
  szassert(pbig,[18*3*2 nrows]);
  pbig = reshape(pbig,[18 3 2 nrows]);
  pbigvw = squeeze(pbig(:,IVIEW,:,:));
  lposVw(:,:,tFP.frm) = pbigvw;
  lObj.labelPosBulkImport(lposVw);
end
fprintf('Added movies and set labels for view %d.\n',IVIEW);

%% Create 2D side projects
% lObj = Labeler();
% % Create a 9-pt proj!

IVIEW = 2;

% get list of pt indices for this side
PTMAP = RF.PTMAP;
switch IVIEW
  case 1
    PTMAPROWS_SIDE = RF.PTMAPROWS_LSIDE;
  case 2
    PTMAPROWS_SIDE = RF.PTMAPROWS_RSIDE;
  otherwise
    assert(false);
end
iPtsSide = PTMAP(PTMAPROWS_SIDE,:);
iPtsSide = iPtsSide(:);
assert(numel(iPtsSide)==9);
fprintf(1,'The points for this side are: %s\n',mat2str(iPtsSide(:)'));
pause(5);

for iprj=1:NPROJ
  % add movie
  lObj.movieAdd(mfasets{iprj}{IVIEW});
  lObj.movieSetGUI(lObj.nmovies);
  nfrm = lObj.nframes;
    
  % get/set labels
  tFP = tdinfo(iprj).tFPtrn;
  nrows = size(tFP,1);
  pbig = tFP.p';
  szassert(pbig,[18*3*2 nrows]);
  pbig = reshape(pbig,[18 3 2 nrows]);
  pbigvw = squeeze(pbig(iPtsSide,IVIEW,:,:));
  szassert(pbigvw,[9 2 nrows]);

  lposVw = nan(9,2,nfrm);
  lposVw(:,:,tFP.frm) = pbigvw;
  lObj.labelPosBulkImport(lposVw);
end
fprintf('Added movies and set labels for view %d.\n',IVIEW);


%%

iLegsUse = 1:6;
iPtLegsAllUsed = iPtLegs(iLegsUse,:);

%% get labels for all frames where all legs are labeled at all
lposVw = lbl.labeledpos{1};
lposTag = lbl.labeledpostag{1};
lposB = lposVw(39:end,:,:);
lpostagB = lposTag(39:end,:);
%%
lposB = lposB(iPtLegsAllUsed,:,:);
tf = arrayfun(@(x)nnz(isnan(lposB(:,:,x)))==0,1:29998);
frmsGoodB = find(tf);
%% make a new lbl
lblNew = load('cam2_11_02_28_v001.lbl','-mat');
%%
lblNew.labeledpos{1} = lposB;
lblNew.labeledposTS{1}(:,frmsGoodB) = now;
save('cam2_11_02_28_v001.lbl','-struct','lblNew','-mat','-v7');

%%
%%%%%%%%%%%
%% BEGIN APRIL DATA MUNGING
%%%%%%%%%%%%
%%
lblNew = load('2dtrain_all.lbl','-mat');
%%
LBL = '20160508_allen.lbl';
lbl = load(LBL,'-mat');
iMov = 3;
lbl.movieFilesAll{iMov}
lposVw = lbl.labeledpos{iMov};
lposB = lposVw(1:18,:,:);
nfrm = size(lposB,3);
tf1 = arrayfun(@(x)nnz(isnan(lposB(:,:,x)))==0,1:nfrm);
tf2 = arrayfun(@(x)nnz(isinf(lposB(:,:,x)))==0,1:nfrm);
tf = tf1 & tf2;
fprintf(1,'nlabeled: %d. nlabeled but have inf: %d. nlabeled no inf: %d.\n',nnz(tf1),nnz(tf1&~tf2),nnz(tf));
frmsGoodB = find(tf);
lposBuse = nan(size(lposB));
lposBuse(:,:,frmsGoodB) = lposB(:,:,frmsGoodB);
%%
iMovNew = 2;
assert(strcmp(lblNew.movieFilesAll{iMovNew},lbl.movieFilesAll{iMov}));
szassert(lposBuse,size(lblNew.labeledpos{iMovNew}));
lblNew.labeledpos{iMovNew} = lposBuse;
lblNew.labeledposTS{iMovNew}(:) = nan;
lblNew.labeledposTS{iMovNew}(:,frmsGoodB) = now;
%%
LBL = '20160428T145316_allen.lbl';
lbl = load(LBL,'-mat');
iMov = 1;
lbl.movieFilesAll{iMov}
lposVw = lbl.labeledpos{iMov};
lposB = lposVw(1:18,:,:);
nfrm = size(lposB,3);
tf1 = arrayfun(@(x)nnz(isnan(lposB(:,:,x)))==0,1:nfrm);
tf2 = arrayfun(@(x)nnz(isinf(lposB(:,:,x)))==0,1:nfrm);
tf = tf1 & tf2;
fprintf(1,'nlabeled: %d. nlabeled but have inf: %d. nlabeled no inf: %d.\n',nnz(tf1),nnz(tf1&~tf2),nnz(tf));
frmsGoodB = find(tf);
lposBuse = nan(size(lposB));
lposBuse(:,:,frmsGoodB) = lposB(:,:,frmsGoodB);

%%
iMovNew = 3;
assert(strcmp(lblNew.movieFilesAll{iMovNew},lbl.movieFilesAll{iMov}));
szassert(lposBuse,size(lblNew.labeledpos{iMovNew}));
lblNew.labeledpos{iMovNew} = lposBuse;
lblNew.labeledposTS{iMovNew}(:) = nan;
lblNew.labeledposTS{iMovNew}(:,frmsGoodB) = now;
%%
save('2dtrain_all.lbl','-struct','lblNew','-mat','-v7');