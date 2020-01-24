%%
TDFILE = '/groups/branson/bransonlab/apt/experiments/data/trnDataSH_20180503.mat';
BASEPROJ = '/groups/branson/bransonlab/apt/tmp/sh_empty_base_proj_5pts_2vws.lbl';
FLY2CALIB = '/groups/branson/bransonlab/apt/experiments/data/fly2DLT_lookupTableAL.csv';
td = load(TDFILE);
tMain = td.tMain;
fly2calib = readtable(FLY2CALIB);

%% figure out the unique moviesets
tMain.movFile_read_id = MFTable.formMultiMovieIDArray(tMain.movFile_read);
[~,rowsUn,iMov] = unique(tMain.movFile_read_id);
tMainMovsUn = tMain(rowsUn,{'movFile_read_id' 'movFile_read' 'flyID'});
tMainMovsUn.Properties.VariableNames{3} = 'fly';

numel(unique(tMainMovsUn.movFile_read_id))
numel(unique(tMainMovsUn.fly))
%% figure out calibs for each movie
tMainMovsUnCalib = outerjoin(...
  tMainMovsUn,fly2calib,...
  'Keys','fly','MergeKeys',true,...
  'Type','left');
tMainMovsUnCalib = tblrowsreorder(tMainMovsUnCalib,tMainMovsUn,'movFile_read_id');
tMainMovsUnCalib.calibfile = strtrim(tMainMovsUnCalib.calibfile);

%% figure out crops
roi_crop = td.roi_crop2; 
tMainMovsUnCalib.roi = roi_crop(rowsUn,:,:);

% check all crops for given movie are consistent 
for i=1:height(tMainMovsUn)
  tf = strcmp(tMainMovsUn.movFile_read{i,1},tMain.movFile_read(:,1)) & ...
       strcmp(tMainMovsUn.movFile_read{i,2},tMain.movFile_read(:,2));
  ntf = nnz(tf);
  roicroptmp = reshape(roi_crop(tf,:,:),ntf,8);
  cropUn = unique(roicroptmp,'rows');
  szassert(cropUn,[1 8]);
  assert(isequal(cropUn,reshape(tMainMovsUnCalib.roi(i,:,:),1,8)));
  fprintf(1,'movUn %d: %d rows\n',i,ntf);
end
% must use crop2, crop3 is jittered *per row*, not per movie

%% create tblLbls
assert(isequaln(tMain.pLbl,tMain.pLbl_m));
tblLbls = tMain(:,{'frm' 'pLbl'});
tblLbls.Properties.VariableNames{2} = 'p';
n = height(tblLbls);
tblLbls.iTgt = ones(n,1);
tblLbls.tfocc = false(n,10);
tblLbls.mov = iMov;
   
%%
createProject(BASEPROJ,tMainMovsUnCalib.movFile_read,...
  'tblLbls',tblLbls,...
  'calibFiles',tMainMovsUnCalib.calibfile,...
  'cropRois',tMainMovsUnCalib.roi,...
  'outfile',fullfile(pwd,'sh_trn4523_20180627.lbl'),...
  'diaryfile',fullfile(pwd,'dry_sh_trn4523_20180627.txt'));


% %%
% diary drylong2calib.txt
% n = height(tMainMovsUnCalib);
% assert(n==lObj.nmovies);
% for i=184:n
%   assert(isequal(lObj.movieFilesAll(i,:),tMainMovsUnCalib.movFile_read(i,:)));
%   cfile = tMainMovsUnCalib.calibfile{i};
%   cfile = strtrim(cfile);
%   if isempty(cfile)
%     fprintf(1,'Mov %d fly %d: no crfile.\n',i,tMainMovsUnCalib.fly(i));
%   elseif exist(cfile,'file')==0
%     fprintf(1,'Mov %d fly %d: crfile DNE: %s.\n',i,tMainMovsUnCalib.fly(i),cfile);
%   else
%     warnst = warning('off','MATLAB:load:variableNotFound');
%     crObj = CalRig.loadCreateCalRigObjFromFile(cfile);
%     warning(warnst);
%     lObj.movieSet(i);
%     pause(1);
%     lObj.viewCalSetCurrMovie(crObj);
%     fprintf(1,'Mov %d fly %d: crType: %s. crfile: %s.\n',i,tMainMovsUnCalib.fly(i),class(crObj),cfile);
%   end
% end
% diary off


% %%
% diary drylong2roi2.txt
% n = height(tMainMovsUn);
% assert(n==lObj.nmovies);
% assert(n==size(roicropUn,1));
% for i=1:n
%   assert(isequal(lObj.movieFilesAll(i,:),tMainMovsUn.movFile_read(i,:)));
%   lObj.movieSet(i);
%   pause(0.5);  
%   lObj.cropSetNewRoiCurrMov(1,roicropUn(i,:,1));
%   lObj.cropSetNewRoiCurrMov(2,roicropUn(i,:,2));
%   fprintf(1,'movset %d\n',i);
% end
% diary off;

%% check it
wbObj = WaitBarWithCancel('Table');
tblLbled = lObj.labelGetMFTableLabeled('wbObj',wbObj);
unique(tblLbled.iTgt(:))
unique(tblLbled.tfocc(:))
%%
tblLbled = lObj.mftTableConcretizeMov(tblLbled);
tblLbled = tblLbled(:,{'mov' 'frm' 'p'});
tblCmp = tMain(:,{'movFile_read' 'frm' 'pLbl'});
tblCmp.Properties.VariableNames = {'mov' 'frm' 'p'};
tblLbled = sortrows(tblLbled,{'mov' 'frm'});
tblCmp = sortrows(tblCmp,{'mov' 'frm'});
isequal(tblLbled,tblCmp)


%% Add GT
CROPINFO = '/groups/branson/bransonlab/apt/tmp/cropInfoGT20180611_update20180626.mat';
LBLGT = '/groups/branson/bransonlab/apt/experiments/data/gtsh_main_1150_v1_20180605_SJHcopy_080618_1111.lbl';
% lObj = load <that proj>

ci = load(CROPINFO);

movFilesGT = lObj.movieFilesAllGTFull;
assert(lObj.gtIsGTMode);
tblLbls = lObj.labelGetMFTableLabeled();
fprintf(2,'Removing last two rows of tblLbls, GT movie 100\n');
assert(isequal(tblLbls.mov(end-1:end),[-100;-100]));
tblLbls = tblLbls(1:end-2,:);
tblLbls = tblLbls(:,{'mov' 'frm' 'iTgt' 'tfocc' 'p'});
unique(tblLbls.tfocc(:))
unique(tblLbls.iTgt(:))
tblLbls.mov = abs(tblLbls.mov);

%% figure out the crops for each movFilesGT
movFilesGTdeIntraed = movFilesGT;
[tf,movDeIntraed] = cellfun(@isIntraMovie,movFilesGTdeIntraed,'uni',0);
tf = cell2mat(tf);
isequal(tf(:,1),tf(:,2))
sum(tf,1)
movFilesGTdeIntraed(tf) = movDeIntraed(tf);
movFilesGTdeIntraedID = MFTable.formMultiMovieIDArray(movFilesGTdeIntraed);

ciMov1 = regexprep(ci.tGT.mov1,'Z:','/groups/huston/hustonlab');
ciMov2 = regexprep(ci.tGT.mov2,'Z:','/groups/huston/hustonlab');
ciMovIDs = MFTable.formMultiMovieIDArray([ciMov1 ciMov2]); % movIDs labeling rows of ci.roicrop
[tf,loc] = ismember(movFilesGTdeIntraedID,ciMovIDs);
all(tf)
roisMovFilesGT = ci.roicrop(loc,:,:);

%%
crObjs = lObj.viewCalibrationDataGT;

nowstr = datestr(now,'yyyymmddTHHMMSS');
outfileS = 'sh_trn4523_gt080618';
outfile = [outfileS '.lbl'];
outfile = fullfile('/groups/branson/bransonlab/apt/tmp',outfile);
dryfile = sprintf('dry_%s_%s.txt',outfileS,nowstr);

BASEPROJ = '/groups/branson/bransonlab/apt/tmp/sh_trn4523_20180627.lbl';
lObj2 = createProject(BASEPROJ,movFilesGT,...
  'gt',true,...
  'tblLbls',tblLbls,... 
  'calibObjs',crObjs,... 
  'cropRois',roisMovFilesGT,... 
  'outfile',outfile,...
  'diaryfile',dryfile... 
  );

% Whoops, manually macro-ize movies.
% Also, gtSugg table not set.


%% 20180724 backfill new SH labels

% <open proj with latest SH GT labels>
tGT1 = lObj.labelGetMFTableLabeled();
tGT1.mIdx = tGT1.mov;
tGT1 = lObj.mftTableConcretizeMov(tGT1);

% <open current mega-proj>
tGT0 = lObj.labelGetMFTableLabeled();
tGT0.mIdx = tGT0.mov;
tGT0 = lObj.mftTableConcretizeMov(tGT0);
%%
[tf,loc] = tblismember(tGT0,tGT1,{'mIdx' 'frm'});
all(tf)
FLDS = {'mov' 'frm' 'iTgt' 'p' 'tfocc'};
isequaln(tGT0(:,FLDS),tGT1(loc,FLDS))
%%
tfadd = ~tblismember(tGT1,tGT0,{'mIdx' 'frm'});
tGT1add = tGT1(tfadd,:);
mIdxAdd = unique(tGT1add.mIdx)

%%
assert(all(mIdxAdd<0));
assert(lObj.gtIsGTMode);
for mIdx=mIdxAdd(:)'
  lObj.movieSet(abs(mIdx));
  pause(1); % prob unnec, give UI a little time
  fprintf(1,'working on mIdx=%d\n',int32(mIdx));
  
  tf = tGT1add.mIdx==mIdx;
  tGT1addMidx = tGT1add(tf,{'frm' 'iTgt' 'p' 'tfocc'});
  lObj.labelPosBulkImportTbl(tGT1addMidx);
  fprintf(1,' ... imported %d lbled rows.\n',height(tGT1addMidx));
end
%%
tGT2 = lObj.labelGetMFTableLabeled();
tGT2.mIdx = tGT2.mov;
tGT2 = lObj.mftTableConcretizeMov(tGT2);
FLDS = {'mov' 'mIdx' 'frm' 'iTgt' 'p' 'tfocc'};
isequal(tGT1(:,FLDS),tGT2(:,FLDS))
%%
tGT2tmp = tGT2(:,{'mIdx' 'frm' 'iTgt'});
tGT2tmp.Properties.VariableNames{1} = 'mov';
lObj.gtSetUserSuggestions(tGT2tmp);

size(lObj.gtSuggMFTable)
all(lObj.gtSuggMFTableLbled)


%% check GT stuff by comparing .lbls
LBLGT = '/groups/branson/bransonlab/apt/experiments/data/gtsh_main_1150_v1_20180605_SJHcopy_080618_1111.lbl';
LBL = '/groups/branson/bransonlab/apt/tmp/sh_trn4523_gt080618_macro.lbl';
lbl0 = load(LBLGT,'-mat');
lbl1 = load(LBL,'-mat');

% % This doens't work b/c the nframes is off by one in LBLGT vs LBL.
% FLDS = {'movieFilesAllGT' 'movieInfoAllGT' 'labeledposGT' 'labeledpostagGT'};
% for f=FLDS,f=f{1};
%   isequaln(lbl0.(f),lbl1.(f))
% end

lObj0 = Labeler; % load LBLGT
tbl0 = lObj0.labelGetMFTableLabeled;
tbl0 = tbl0(1:end-2,:);
tbl1 = lObj.labelGetMFTableLabeled;
FLDSCMP = {'mov' 'frm' 'iTgt' 'p' 'tfocc'};
isequaln(tbl0(:,FLDSCMP),tbl1(:,FLDSCMP))

% check, labels are the same

% assume calibObjs are the same.
isequaln(cellfun(@class,lObj0.viewCalibrationDataGT,'uni',0),...
         cellfun(@class,lObj.viewCalibrationDataGT,'uni',0))
       
% check crops
lObj.cropMontage

% looks good


%% XV easy test
XVSPLITS = '../experiments/data/trnSplits_20180509.mat';
splt = load(XVSPLITS);
lObj = Labeler;
% load sh_trn4523.lbl
%%
mfts = MFTSetEnum.AllMovAllLabeled;
tLbled = mfts.getMFTable(lObj);
tMainMFT = tMain(:,{'movFile_read' 'frm'});
tMainMFT.Properties.VariableNames = {'mov' 'frm'};
tMainMFT.iTgt = ones(height(tMainMFT),1);
[~,loc1] = ismember(tMainMFT.mov(:,1),lObj.movieFilesAll(:,1));
[~,loc2] = ismember(tMainMFT.mov(:,2),lObj.movieFilesAll(:,2));
isequal(loc1,loc2)
tMainMFT.mov = MovieIndex(loc1);
isequal(sortrows(tLbled,MFTable.FLDSID),sortrows(tMainMFT,MFTable.FLDSID))

%%
nowstr = datestr(now,'yyyymmddTHHMMSS');
dryfile = sprintf('dry_xvHard_%s.txt',nowstr);
diary(dryfile);
wbObj = WaitBarWithCancel('xv');
lObj.trackCrossValidate(...
  'kfold',3,... % number of folds
  'initData',true,... % if true, call .initData() between folds to minimize mem usage
  'wbObj',wbObj,... % (opt) WaitBarWithCancel
  'tblMFgt',tMainMFT,... % (opt), MFTable of data to consider. Defaults to all labeled rows. tblMFgt should only contain fields .mov, .frm, .iTgt. labels, rois, etc will be assembled from proj
  'partTrn',splt.xvMain3Hard==0,... % (opt) pre-defined training splits. If supplied, partTrn must be a [height(tblMFgt) x kfold] logical. tblMFgt should be supplied.
  'partTst',splt.xvMain3Hard==1 ... % (opt) etc see partTrn
);
xvRes = lObj.xvResults;
save xvResHard1.mat xvRes;
diary off
%% Start of Retrain takes really long. Perf test
% Conclusion: get_Readframe_fcn on 1k movies just takes a while. Add a
% waitbar or something
movsUn = unique(lbl.movieFilesAll(:));
m = containers.Map;
tic;
for i=200:250
  mr = MovieReader;
  mov = movsUn{i};
  mr.open(mov);
 % [q1,q2,q3,q4] = get_readframe_fcn(mov);
  
  m(mov) = mr;
  fprintf('%d: %s\n',i,mov);
end
toc
  
  