%%
close all force;
clear classes;
clc
%%
addpath f:\repo\apt6\gt
addpath f:\repo\apt6\gt2sh\

%% Load GT results and KB orig GT tables
tGT = loadSingleVariableMatfile('cpr_gtres_v00.mat');
sgt = load('SelectedGTFrames_SJH_20180603.mat');
tFrms2Lbl = sgt.frames2label;

%% 
assert(strcmp(tGT.Properties.VariableNames{1},'mov'));
tGT.Properties.VariableNames{1} = 'movFile';
tGT.movFile = cellfun(@FSPath.standardPath,tGT.movFile,'uni',0);

%% KB's intra movies do not have the _i at the end
tfI = strcmp(tFrms2Lbl.type,'intra');
tFrms2Lbl.movFile(tfI,:) = cellfun(@intraizeMovie,tFrms2Lbl.movFile(tfI,:),'uni',0);
tFrms2Lbl.movFile = cellfun(@FSPath.standardPath,tFrms2Lbl.movFile,'uni',0);

%%
tGT.movID = MFTable.formMultiMovieIDArray(tGT.movFile);
tFrms2Lbl.movID = MFTable.formMultiMovieIDArray(tFrms2Lbl.movFile);
%%
[tf,loc] = tblismember(tGT,tFrms2Lbl,{'movID' 'frm'});
fprintf(1,'%d rows in tGT not in tFrms2Lbl.\n',nnz(~tf));

%% NOTE: rows of tGT reordered by innerjoin
tGT0 = tGT;
tGT = innerjoin(tGT,tFrms2Lbl,'keys',{'movID' 'frm'});
tGT.movFile = tGT.movFile_tGT;

[tf,loc] = tblismember(tGT,tGT0,{'movID' 'frm'});
assert(all(tf));
flds = tblflds(tGT0);
isequaln(tGT(:,flds),tGT0(loc,:))

[tf,loc] = tblismember(tGT,tFrms2Lbl,{'movID' 'frm'});
assert(all(tf));
flds = tblflds(tFrms2Lbl);
isequal(tGT(:,flds),tFrms2Lbl(loc,:))

%% Intra rows: find matching row-pairs
clear ijIntra
tfI = strcmp(tGT.type,'intra');
ijIntra(:,2) = find(tfI);
nIntra = size(ijIntra,1);
for iIntra=1:nIntra
  j = ijIntra(iIntra,2);
  movs = tGT.movFile(j,:);
  [tf,movsNonIntra] = cellfun(@isIntraMovie,movs,'uni',0);
  assert(all(cell2mat(tf)));
  movsNonIntra = FSPath.standardPath(movsNonIntra);
  movID = MFTable.formMultiMovieID(movsNonIntra);
  i = find(strcmp(tGT.movID,movID) & tGT.frm==tGT.frm(j));
  switch numel(i)
    case 0
      fprintf(1,' ... no match found for fly %d, frm %d\n',tGT.flyID(j),tGT.frm(j));
    case 1
      ijIntra(iIntra,1) = i;
    otherwise
      assert(false);
  end  
end
%%
tfrm = any(ijIntra==0,2);
ijIntra(tfrm,:) = [];
nIntra = size(ijIntra,1);
fprintf(1,'Removing %d intra rows that had no match. Left with %d intra row-pairs.\n',...
  nnz(tfrm),nIntra);

%% huh? just bad luck?
shgt = load('shgtround2.mat');
shgt.tGT.flyID = cellfun(@parseSHfullmovie,shgt.tGT.mov(:,1));
shgt.tGT.hasLbl = shgt.tfLbl;
for fly = [420 705 739 836]
  tf = fly==shgt.tGT.flyID;
  tTmp = shgt.tGT(tf,{'flyID' 'frm' 'hasLbl'});
  tTmp.origrow = find(tf)
end

%% Err
nGT = height(tGT);
errL2 = tGT.L2err;
errL2 = reshape(errL2,nGT,5,2);
dxy = tGT.pTrk - tGT.pLbl;
dxy = reshape(dxy,nGT,5,2,2); % n,pt,vw,x/y
dxy = permute(dxy,[1 2 4 3]); % n,pt,x/y,vw

% check 
errL22 = squeeze(sqrt(sum(dxy.^2,3)));
tmp = errL2-errL22;
fprintf(1,'Sanity check dxy: max(abs(resid)))=%.3g\n',max(abs(tmp(:))));

dxyIntra = tGT.pLbl(ijIntra(:,1),:) - tGT.pLbl(ijIntra(:,2),:);
dxyIntra = reshape(dxyIntra,nIntra,5,2,2); % n,pt,vw,x/y
dxyIntra = permute(dxyIntra,[1 2 4 3]);
%errL2Intra = sqrt(sum(dxyIntra.^2,4));

% Question: A few FLAT ZEROS in errL2Intra?!

%% Cmp: {train act, trainnoact, testact, testnoact, enriched_act}
% for bulleEyes, circs and ells

TYPES = {'enriched_activation' 'train_activation' 'train_noactivation' 'test_activation' 'test_noactivation'};
nType = numel(TYPES);
assert(isequal(height(tGT),size(dxy,1),size(errL2,1)));
assert(ndims(errL2)==3);
assert(ndims(dxy)==4);
%errL2Cell = cell(nType,1);
dxyCell = cell(nType,1);
for iType=1:nType
  ty = TYPES{iType};
  tf = strcmp(ty,tGT.type);
  %errL2Cell{iType} = errL2(tf,:,:);
  dxyCell{iType} = dxy(tf,:,:,:);
end

% dxyAll = cat(1,dxyCell{:});
% dxyCell{end+1,1} = dxyAll;
% TYPES{end+1} = 'all';

dxyCell{end+1,1} = dxyIntra;
TYPES{end+1} = 'intra';

nType = numel(TYPES);

%%
td = load('f:\aptStephenCPRInvestigate20180327\trnData20180503.mat');

%%
DOSAVE = true;
SAVEDIR = 'figsFull';

PTILES = [50 75 90 95];

hFig = [];

hFig(end+1) = figure(10);
hfig = hFig(end);
set(hfig,'Name','ptileTypes','Position',[2561 401 1920 1124]);
GTPlot.ptileCurves(dxyCell,...
  'ptiles',PTILES,...
  'hFig',hfig,...
  'setNames',TYPES,...
  'axisArgs',{'XTicklabelRotation',30,'FontSize' 10} ...
  );

JETMOD = jet(numel(PTILES)+1);
JETMOD = JETMOD(2:end,:);

hFig(end+1) = figure(20);
hfig = hFig(end);
set(hfig,'Name','BE','Position',[2561 401 1920 1124]);
[~,ax] = GTPlot.bullseyePtiles(dxyCell,...
  td.IMain20180503_crop2(1,:),squeeze(td.xyLblMain20180503_crop2(1,:,:,:)),...
  'hFig',hfig,...
  'setNames',TYPES,...
  'ptiles',PTILES,...
  'ptileCmap',@(x)JETMOD,...
  'lineWidth',1,...
  'xyLblPlotArgs',{'w.' 'markersize' 8}...
  );

hFig(end+1) = figure(25);
hfig = hFig(end);
set(hfig,'Name','BEell','Position',[2561 401 1920 1124]);
[~,ax] = GTPlot.bullseyePtiles(dxyCell,...
  td.IMain20180503_crop2(1,:),squeeze(td.xyLblMain20180503_crop2(1,:,:,:)),...
  'hFig',hfig,...
  'setNames',TYPES,...
  'ptiles',PTILES,...
  'ptileCmap',@(x)JETMOD,...
  'lineWidth',1,...
  'xyLblPlotArgs',{'w.' 'markersize' 8},...
  'contourtype','ellipse'...
  );

if DOSAVE
  for i=1:numel(hFig)
    h = figure(hFig(i));
    fname = h.Name;
    hgsave(h,fullfile(SAVEDIR,[fname '.fig']));
    set(h,'PaperOrientation','landscape','PaperType','arch-d');
    print(h,'-dpdf',fullfile(SAVEDIR,[fname '.pdf']));  
    print(h,'-dpng','-r300',fullfile(SAVEDIR,[fname '.png']));   
    fprintf(1,'Saved %s.\n',fname);
  end
end

%% %%%%%%% APTRT stuff %%%%%%%%%

% Step 1: figure out if any GT flies don't have a reference lblfile 
% selected/specified. The tracking (imported, in labels2) in this reference
% lblfile will be used to generate a reference head posn for that fly

REFTRACKINGLBLS = 'W:\apt\experiments\data\fly2RefPredLbl20180724.mat';
refTrackingLbls = loadSingleVariableMatfile(REFTRACKINGLBLS);
flyGTun = unique(tGT.flyID);
flyGTunNeedRefLbl = setdiff(flyGTun,refTrackingLbls.fly)
nFlyGTunNeedRefLbl = numel(flyGTunNeedRefLbl)

% REFTRACKINGLBLS = 'W:\apt\experiments\data\cprXVerrVsHeadPosn20180529.mat';
% tRefPreds = loadSingleVariableMatfile(REFTRACKINGLBLS);
% 
% flyGTTrn = intersect(tGT.flyID,td.tMain20180503.flyID);
% flyGTNonTrn = setdiff(tGT.flyID,td.tMain20180503.flyID);
% 
% all(ismember(flyGTTrn,tRefPreds.fly))
% any(ismember(flyGTNonTrn,tRefPreds.fly))

%% Step 2. For each flyGTunNeedRefLbl, select a single reference lbl file
% CUT+PASTE from nbGTDataPlan
%
% SH: If you are just trying to get a good 'median head reference point' 
% use the lowest intensity available.  If it is a choice between _wind and 
% _NOwind use "NOwind"
%
% Note, the choices made here are not 100.0% perfect. Eg:
% Fly 544:
%  ... (*) fly544_allIntensities.lbl (updated 20180601)
%  ... fly544_intensity_3.lbl (updated 20180601)
%  ... fly544_intensity_6.lbl (updated 20180601)
%
% However, this is prob not the end of the world. SH can look over 
% REFTRACKINGLBLS to check if desired.

APTPROJFILESDIR = 'Z:\flp-chrimson_experiments\APT_projectFiles\';
dd = dir(fullfile(APTPROJFILESDIR,'*.lbl'));
aptlbls = {dd.name}';

flyGTLbls = cell(nFlyGTunNeedRefLbl,1);
for ifly=1:nFlyGTunNeedRefLbl
  f = flyGTunNeedRefLbl(ifly);
  flystr = sprintf('fly%d',f);
  flypat = sprintf('%s*.lbl',flystr);
  dd = dir(fullfile(APTPROJFILESDIR,flypat));
  
  lblnames = {dd.name}';
  tfNW = contains(lblnames,'NOwind');
  if any(tfNW)
    iLblChoose = find(tfNW);
    assert(isscalar(iLblChoose));
  else
    iLblChoose = 1;
  end
  flyGTLbls{ifly} = fullfile(APTPROJFILESDIR,lblnames{iLblChoose});
  
  fprintf('Fly %d:\n',f);
  for iLbl=1:numel(dd)    
    if iLbl==iLblChoose      
      fprintf(' ... (*) %s (updated %s)\n',dd(iLbl).name,datestr(dd(iLbl).datenum,'yyyymmdd'));
    else
      fprintf(' ... %s (updated %s)\n',dd(iLbl).name,datestr(dd(iLbl).datenum,'yyyymmdd'));
    end
  end
end

%% 2b. update REFTRACKINGLBLS table
tRefPredLblsNew = table(flyGTunNeedRefLbl,flyGTLbls,'VariableNames',...
  {'fly' 'lbl'});

s = load(REFTRACKINGLBLS);
assert(isempty(intersect(s.tFly2RefPredLbl.fly,tRefPredLblsNew.fly)));

s.tFly2RefPredLbl = [s.tFly2RefPredLbl; tRefPredLblsNew];
[~,idx] = sort(s.tFly2RefPredLbl.fly);
s.tFly2RefPredLbl = s.tFly2RefPredLbl(idx,:);

save(REFTRACKINGLBLS,'-mat','-struct','s');

%% Step 3. For each fly without reference head posn info in APTPREDSRESDIR, 
% run APTRT using the lblFile specified in REFTRACKINGLBLS. This uses 
% Mayank's tracking results (in that lblFile) to estimate a reference head
% posn and saves this reference info to APTPREDSRESDIR.

addpath f:\repo\apt6\user\APT2RT
addpath f:\repo\matgeom/matGeom\
setupMatGeom
FLYNUM2BODY = 'f:\repo\apt6\user\flynum2bodyAxis.csv';
FLYNUM2CALIB = 'Z:\flp-chrimson_experiments\fly2DLT_lookupTableStephen_WindowsPaths.csv';
APTPREDSRESDIR = 'f:\aptSH_GT_round2_20180720\aptPredsRes20180724';

%%
tFly2Body = readtable(FLYNUM2BODY);
tFly2Body.Properties.VariableNames{1} = 'fly';
tFly2RefPredLbl = loadSingleVariableMatfile(REFTRACKINGLBLS);
%tFly2Calib = readtable(FLYNUM2CALIB);

diary dry4.txt

flyGTun = unique(tGT.flyID);
nFlyGTun = numel(flyGTun);
fprintf(1,'%d unique GT flies.\n',nFlyGTun);
for ifly=1:nFlyGTun
  fly = flyGTun(ifly);
  
  predResFnameS = sprintf('fly%04d.mat',fly);
  predResFname = fullfile(APTPREDSRESDIR,predResFnameS);
  if exist(predResFname,'file')==0
    if ~any(tFly2Body.fly==fly)
      warningNoTrace('fly %d: no body data. skipping...\n',fly);
      continue;
    end
    
    tf = tFly2RefPredLbl.fly==fly;
    assert(nnz(tf)==1);
    lblFile = tFly2RefPredLbl.lbl{tf};
    
    res = struct();
    res.FLYNUM2BODY = FLYNUM2BODY;
    res.FLYNUM2CALIB = FLYNUM2CALIB;
    res.fly = fly;
    res.lbl = lblFile;
    [res.axAngDegXYZ,res.trans,res.residErr,res.scaleErr,res.quat,res.pivot,res.refHead] = ...
      APT2RT(lblFile,FLYNUM2BODY,FLYNUM2CALIB,1,[],[]);
    
    save(predResFname,'-mat','-struct','res');
    fprintf(1,'Saved %s.\n',predResFname);
    pause(0.1);
  end
end

diary off

%% Step4. For each GT movie in megaproj, run APT2RT on the (gt) Labels 
% using the prediction/reference results generated in Step3. This gives a
% 3D posn for each GT labeled row.

lblBig = fullfile(pwd,'sh_trn4523_gtcomplete_cacheddata.lbl');
% FLYNUM2BODY = 'f:\repo\apt6\user\flynum2bodyAxis.csv';
% FLYNUM2CALIB = 'Z:\flp-chrimson_experiments\fly2DLT_lookupTableStephen_WindowsPaths.csv';
% APTPREDSRESDIR = 'f:\aptSH_GT_round2_20180720\aptPredsRes20180724';
% addpath f:\repo\apt6\user\APTRT
% addpath f:\repo\matgeom/matGeom\
% setupMatGeom

lbl = load(lblBig,'-mat');

%%
nowstr = datestr(now,'yyyymmddTHHMMSS');
diary(sprintf('dry_%s.txt',nowstr));

tblRT = [];
nMovGT = size(lbl.movieFilesAllGT,1);
for iMovGT=1:nMovGT
  fly = parseSHfullmovie(lbl.movieFilesAllGT{iMovGT,1});
  fprintf(1,'Working on gt mov %d, fly %d ... \n',iMovGT,fly);

  predResFname = sprintf('fly%04d.mat',fly);
  predResFname = fullfile(APTPREDSRESDIR,predResFname);  
  fprintf(1,'  Loading reference points from tracking results: %s\n',predResFname);
  predres = load(predResFname);
    
  rt = struct();
  [rt.axisAngleDegXYZ,rt.translations,rt.residualErrors,rt.scaleErrors,...
    rt.quaternion,pivot,refHead] = APT2RT_al(...
      lbl,FLYNUM2BODY,FLYNUM2CALIB,0,predres.pivot,predres.refHead,...
        'flyNum',fly,'iMov0',iMovGT,'iMov1',iMovGT,'iMovViewCal',iMovGT,'gtMovs',true);

  lposGT = lbl.labeledposGT{iMovGT};
  lposGT = SparseLabelArray.full(lposGT);
  lposGT = reshape(lposGT,20,[]);
  frmsGT = find(any(~isnan(lposGT),1));
  nfrmsGT = numel(frmsGT);
  
  FLDS = {'axisAngleDegXYZ' 'translations' 'residualErrors' 'scaleErrors' 'quaternion'};
  for f=FLDS,f=f{1}; %#ok<FXSET>
    nnan = ~isnan(rt.(f));
    assert(isequal(all(nnan,2),any(nnan,2)));
    frms = find(all(nnan,2));
    assert(isequal(frms(:),frmsGT(:)));
  end
  
  fly = repmat(fly,nfrmsGT,1);
  mIdx = MovieIndex(repmat(-iMovGT,nfrmsGT,1));
  frm = frmsGT(:);
  axAngDegXYZ = rt.axisAngleDegXYZ(frms,:);
  trans = rt.translations(frms,:);
  residErr = rt.residualErrors(frms,:);
  scaleErr = rt.scaleErrors(frms,:);
  quat = rt.quaternion(frms,:);
  tblRTnew = table(fly,mIdx,frm,axAngDegXYZ,trans,residErr,scaleErr,quat);
  
  tblRT = cat(1,tblRT,tblRTnew);
  
%   if mod(iMovGT,10)==0
%     fname = sprintf('tblRT_post%d.mat',iMovGT);
%     save(fname,'tblRT');
%     fprintf(1,'Saved %s\n',fname);
%   end
end

diary off
%%
save tblRT_full.mat tblRT;


%% Step5. Join/merge tblRT into tGT.
% NOTE: rows of tGT re-ordered by innerjoin
% tblRT = loadSingleVariableMatfile('tblRT_72movs.mat');

assert(strcmp(tblRT.Properties.VariableNames{1},'fly'));
tblRT.Properties.VariableNames{1} = 'flyID';
tblRT.mIdx = int32(tblRT.mIdx);

tGT0 = tGT;
tGT = innerjoin(tGT,tblRT,'keys',{'flyID' 'mIdx' 'frm'});
height(tGT0)
height(tGT)

[tf,loc] = tblismember(tGT,tGT0,{'flyID' 'mIdx' 'frm'});
assert(all(tf));
flds = tblflds(tGT0);
isequaln(tGT(:,flds),tGT0(loc,:))

[tf,loc] = tblismember(tGT,tblRT,{'flyID' 'mIdx' 'frm'});
assert(all(tf));
flds = tblflds(tblRT);
isequal(tGT(:,flds),tblRT(loc,:))

%% Step6. run SH computeRotations to find rotation diff between tracking 
% and labels for each row of tGT

REMOVE_PT2 = true;

fly2calib = readtable(FLYNUM2CALIB,...
  'Delimiter',',',...
  'ReadVariableNames',false,...
  'HeaderLines',0);
fly2calib.Properties.VariableNames = {'fly' 'calibfile'};
fly2calibMap = containers.Map(fly2calib.fly,fly2calib.calibfile);

nGT = height(tGT);
axisAngRadMag_dLblTrk = nan(nGT,1);
stroErr3dLbl = nan(nGT,5); % stereoReproj err for orthocam/lbls
stroErr3dTrk = nan(nGT,5); % " orthocam/trks
for iGT=1:nGT
  if mod(iGT,10)==0
    fprintf('\n%d\n',iGT);
  end
  
  mIdx = tGT.mIdx(iGT);
  frm = tGT.frm(iGT);
  iMov = abs(mIdx);
  pLbl = tGT.pLbl(iGT,:);
  pTrk = tGT.pTrk(iGT,:);
  movFile1 = tGT.movFile_tGT{iGT,1};
  flyID = tGT.flyID(iGT);
  
  % get/check the calib
  fly = parseSHfullmovie(movFile1);
  assert(fly==flyID);
  calibFile = fly2calibMap(fly);
  crObj0 = CalRig.loadCreateCalRigObjFromFile(calibFile);
  crObj = lbl.viewCalibrationDataGT{iMov};
  tfDLT = isa(crObj0,'CalRigSH');
  dltstuff = [];
  if tfDLT
    assert(isequal(crObj0.kineData,crObj.kineData));
    dltstuff = load(strtrim(calibFile), '-regexp', '^(?!vidObj$).');
    assert(all(isfield(dltstuff,{'DLT_1';'DLT_2'})));
%     dlt_side = DLT_1;
%     dlt_front = DLT_2;
  else
    if ~isequal(crObj0.rvecs,crObj.rvecs)
      warningNoTrace('Fly %d mIdx %d: Orthocam objs don''t match.',fly,int32(mIdx));
    end
  end
  
  % check pLbl
  lpos = lbl.labeledposGT{iMov};
  lpos = SparseLabelArray.full(lpos);
  xyLbl = lpos(:,:,frm); % [10 x 2]
  assert(~any(isnan(xyLbl(:))));
  assert(isequal(xyLbl(:),pLbl(:)));
  uvLbl = cat(3,xyLbl(1:5,:),xyLbl(6:10,:)); % [5x2x2]. pt,(x/y),vw
  xyTrk = reshape(pTrk,10,2);
  uvTrk = cat(3,xyTrk(1:5,:),xyTrk(6:10,:)); % [5x2x2] etc  
  szassert(uvLbl,[5 2 2]);
  szassert(uvTrk,[5 2 2]);
  
  % 3dize
  X3Dlbl = nan(5,3); % pt,(x/y/z)
  X3Dtrk = nan(5,3);
  X3DlblStroErr = nan(1,5);
  X3DtrkStroErr = nan(1,5);
  for ipt=1:5
    if tfDLT
      A = [dltstuff.DLT_1,dltstuff.DLT_2];
      tempxyzblah = reconfu(A,[uvLbl(ipt,:,1) uvLbl(ipt,:,2)]);
      X3Dlbl(ipt,:) = tempxyzblah(1:3);      
      tempxyzblah = reconfu(A,[uvTrk(ipt,:,1) uvTrk(ipt,:,2)]);
      X3Dtrk(ipt,:) = tempxyzblah(1:3);      
    else % orthocam
      [X3Dlbl(ipt,:),X3DlblStroErr(ipt)] = ...
        crObj.stereoTriangulate(uvLbl(ipt,:,1)',uvLbl(ipt,:,2)'); % [1x3]
      [X3Dtrk(ipt,:),X3DtrkStroErr(ipt)] = ...
        crObj.stereoTriangulate(uvTrk(ipt,:,1)',uvTrk(ipt,:,2)'); % [1x3]
    end
  end

  if REMOVE_PT2
    fprintf(1,'removing pt2.');
    X3Dlbl(2,:) = [];
    X3Dtrk(2,:) = [];
  end
  [~,~,~,~,~,axisAngRad] = computeRotations(X3Dlbl,X3Dtrk);
  assert(numel(axisAngRad)==7);
  
  axisAngRadMag_dLblTrk(iGT) = axisAngRad(end);
  stroErr3dLbl(iGT,:) = X3DlblStroErr;
  stroErr3dTrk(iGT,:) = X3DtrkStroErr;
end

if REMOVE_PT2
  tGT.axisAngRadMag_dLblTrk_nopt2 = axisAngRadMag_dLblTrk;
  tGT.stroErr3dLbl_nopt2 = stroErr3dLbl;
  tGT.stroErr3dTrk_nopt2 = stroErr3dTrk;  
else
  tGT.axisAngRadMag_dLblTrk = axisAngRadMag_dLblTrk;
  tGT.stroErr3dLbl = stroErr3dLbl;
  tGT.stroErr3dTrk = stroErr3dTrk;
end

%%
save cpr_gtres_v00_withanls.mat tGT

%% C+P FROM ABOVE. Re-figure out intra rows. tGT row-ordering has changed.
%% Intra rows: find matching row-pairs
clear ijIntra
tfI = strcmp(tGT.type,'intra');
ijIntra(:,2) = find(tfI);
nIntra = size(ijIntra,1);
for iIntra=1:nIntra
  j = ijIntra(iIntra,2);
  movs = tGT.movFile(j,:);
  [tf,movsNonIntra] = cellfun(@isIntraMovie,movs,'uni',0);
  assert(all(cell2mat(tf)));
  movsNonIntra = FSPath.standardPath(movsNonIntra);
  movID = MFTable.formMultiMovieID(movsNonIntra);
  i = find(strcmp(tGT.movID,movID) & tGT.frm==tGT.frm(j));
  switch numel(i)
    case 0
      fprintf(1,' ... no match found for fly %d, frm %d\n',tGT.flyID(j),tGT.frm(j));
    case 1
      ijIntra(iIntra,1) = i;
    otherwise
      assert(false);
  end  
end
%%
tfrm = any(ijIntra==0,2);
ijIntra(tfrm,:) = [];
nIntra = size(ijIntra,1);
fprintf(1,'Removing %d intra rows that had no match. Left with %d intra row-pairs.\n',...
  nnz(tfrm),nIntra);


%% Compute head-angle-err on intra rows

% fly2calib = readtable(FLYNUM2CALIB,...
%   'Delimiter',',',...
%   'ReadVariableNames',false,...
%   'HeaderLines',0);
% fly2calib.Properties.VariableNames = {'fly' 'calibfile'};
% fly2calibMap = containers.Map(fly2calib.fly,fly2calib.calibfile);

intra_axisAngRadMag_dLbl = nan(nIntra,1);
intra_axisAngRadMag_dLbl_nopt2 = nan(nIntra,1);
intra_axAngDegXYZMag = nan(nIntra,1);
intra_daxAngDegXYZMag = nan(nIntra,1);
intra_l2err_dLbl = nan(nIntra,1);
for iIntra=1:nIntra
  iGT = ijIntra(iIntra,1);
  jGT = ijIntra(iIntra,2);
  
  mIdxI = tGT.mIdx(iGT);
  mIdxJ = tGT.mIdx(jGT);
  iMovI = abs(mIdxI);
  iMovJ = abs(mIdxJ);
  crObjI = lbl.viewCalibrationDataGT{iMovI};
  crObjJ = lbl.viewCalibrationDataGT{iMovJ};
  
  assert(tGT.frm(iGT)==tGT.frm(jGT));  
  frm = tGT.frm(iGT);

  pLblI = tGT.pLbl(iGT,:);
  pLblJ = tGT.pLbl(jGT,:);
  
  tfDLT = isa(crObjI,'CalRigSH');
  if tfDLT
    dltstuff = crObjI.kineData.cal.coeff;
    assert(isequal(crObjJ.kineData.cal.coeff,dltstuff));
    assert(all(isfield(dltstuff,{'DLT_1';'DLT_2'})));
  else
    dltstuff = [];
    assert(isequal(crObjI.rvecs,crObjJ.rvecs));
  end
  
  % check pLbl
  xyLblI = reshape(pLblI,10,2);
  xyLblJ = reshape(pLblJ,10,2);
  uvLblI = cat(3,xyLblI(1:5,:),xyLblI(6:10,:)); % [5x2x2]. pt,(x/y),vw
  uvLblJ = cat(3,xyLblJ(1:5,:),xyLblJ(6:10,:)); % [5x2x2]. pt,(x/y),vw  
  szassert(uvLblI,[5 2 2]);
  szassert(uvLblJ,[5 2 2]);
  
  % 3dize
  X3DlblI = nan(5,3); % pt,(x/y/z)
  X3DlblJ = nan(5,3);
%   X3DlblStroErr = nan(1,5);
%   X3DtrkStroErr = nan(1,5);
  for ipt=1:5
    if tfDLT
      A = [dltstuff.DLT_1,dltstuff.DLT_2];
      tempxyzblah = reconfu(A,[uvLblI(ipt,:,1) uvLblI(ipt,:,2)]);
      X3DlblI(ipt,:) = tempxyzblah(1:3);      
      tempxyzblah = reconfu(A,[uvLblJ(ipt,:,1) uvLblJ(ipt,:,2)]);
      X3DlblJ(ipt,:) = tempxyzblah(1:3);      
    else % orthocam
      X3DlblI(ipt,:) = ...
        crObjI.stereoTriangulate(uvLblI(ipt,:,1)',uvLblI(ipt,:,2)'); % [1x3]
      X3DlblJ(ipt,:) = ...
        crObjJ.stereoTriangulate(uvLblJ(ipt,:,1)',uvLblJ(ipt,:,2)'); % [1x3] crObjI/J should be same
    end
  end

  [~,~,~,~,~,axisAngRad] = computeRotations(X3DlblI,X3DlblJ);
  assert(numel(axisAngRad)==7);
  idxno2 = [1 3 4 5];
  [~,~,~,~,~,axisAngRad_nopt2] = ...
    computeRotations(X3DlblI(idxno2,:),X3DlblJ(idxno2,:));  
    
  intra_axisAngRadMag_dLbl(iIntra) = axisAngRad(end);
  intra_axisAngRadMag_dLbl_nopt2(iIntra) = axisAngRad_nopt2(end);
  
  
  axAngDegI = tGT.axAngDegXYZ(iGT,1);
  axAngDegJ = tGT.axAngDegXYZ(jGT,1);
  intra_axAngDegXYZMag(iIntra) = axAngDegI;
  intra_daxAngDegXYZMag(iIntra) = axAngDegJ-axAngDegI;
  pxerr = sqrt(sum((uvLblI-uvLblJ).^2,2)); % [5x1x2]. pt, ., vw
  intra_l2err_dLbl(iIntra) = mean(pxerr(:));
end


%%

DOSAVE = true;
SAVEDIR = 'figsRT';

tiLblArgs = {'fontweight','bold','interpreter','none','fontsize',20};
axLblArgs = {'fontweight','bold','interpreter','none','fontsize',16};

hFig = [];

hFig(end+1) = figure(15);
hfig = hFig(end);
tstr = 'head ang err (trk vs lbl), inc. pt2 vs not. All GT rows';
set(hfig,'Name',tstr,'Position',[2561 401 1920 1124]);
clf;
scatter(tGT.axisAngRadMag_dLblTrk/pi*180,tGT.axisAngRadMag_dLblTrk_nopt2/pi*180);
hold on
xl = xlim(gca);
plot(xl,xl,'-r','linewidth',1);
grid on;
title(tstr,tiLblArgs{:});
xlabel('angle (deg) inc pt2',axLblArgs{:});
ylabel('angle (deg) no pt2',axLblArgs{:});

hFig(end+1) = figure(17);
hfig = hFig(end);
tstr = 'head ang err (inc pt2) vs mean px err, trk vs lbl. All GT rows';
set(hfig,'Name',tstr,'Position',[2561 401 1920 1124]);
clf;
scatter(tGT.meanL2err,tGT.axisAngRadMag_dLblTrk/pi*180);
grid on;
lims = axis;
lims([1 3]) = 0;
axis(lims);
title(tstr,tiLblArgs{:});
xlabel('mean L2 err (px)',axLblArgs{:});
ylabel('angle err (deg) pt2',axLblArgs{:});


tfNotIntra = ~strcmp(tGT.type,'intra');
nNotIntra = nnz(tfNotIntra);
fprintf(1,'%d not-intra rows.\n',nNotIntra);

hFig(end+1) = figure(20);
hfig = hFig(end);
tstr = sprintf('n (not intra)=%d. headAngErr vs headAng',nNotIntra);
set(hfig,'Name',tstr,'Position',[2561 401 1920 1124]);
axs = mycreatesubplots(1,2,.1);
linkaxes(axs);

ax = axs(1);
axes(ax);
tstr = sprintf('n (not intra)=%d. headAngErr vs headAng, inc. pt2',nNotIntra);
scatter(tGT.axAngDegXYZ(tfNotIntra,1),tGT.axisAngRadMag_dLblTrk(tfNotIntra)/pi*180);
grid on;
title(tstr,tiLblArgs{:});
xlabel('head ang mag (deg)',axLblArgs{:});
ylabel('head ang err (deg) inc pt2',axLblArgs{:});

ax = axs(2);
axes(ax);
tstr = sprintf('NO pt2');
scatter(tGT.axAngDegXYZ(tfNotIntra,1),tGT.axisAngRadMag_dLblTrk_nopt2(tfNotIntra)/pi*180);
grid on;
title(tstr,tiLblArgs{:});
xlabel('head ang mag (deg)',axLblArgs{:});
ylabel('head ang err (deg) NO pt2',axLblArgs{:});


hFig(end+1) = figure(25);
hfig = hFig(end);
tstr = sprintf('nIntra=%d. headAngErr (inc pt2) vs mean px err',nIntra);
set(hfig,'Name',tstr,'Position',[2561 401 1920 1124]);
scatter(intra_l2err_dLbl,intra_axisAngRadMag_dLbl/pi*180);
grid on;
title(tstr,tiLblArgs{:});
xlabel('mean err (px)',axLblArgs{:});
ylabel('angle err (deg) inc pt2',axLblArgs{:});
lims = axis;
lims([1 3]) = 0;
axis(lims);


hFig(end+1) = figure(30);
hfig = hFig(end);
tstr = sprintf('nIntra=%d. headAngErr vs headAng',nIntra);
set(hfig,'Name',tstr,'Position',[2561 401 1920 1124]);
axs = mycreatesubplots(1,2,.1);
linkaxes(axs);

ax = axs(1);
axes(ax);
tstr = sprintf('nIntra=%d. headAngErr vs headAng, inc pt2',nIntra);
scatter(intra_axAngDegXYZMag,intra_axisAngRadMag_dLbl/pi*180);
grid on;
title(tstr,tiLblArgs{:});
xlabel('head ang mag (deg)',axLblArgs{:});
ylabel('head ang err (deg) inc pt2',axLblArgs{:});

ax = axs(2);
axes(ax);
tstr = sprintf('NO. pt2');
scatter(intra_axAngDegXYZMag,intra_axisAngRadMag_dLbl_nopt2/pi*180);
grid on;
title(tstr,tiLblArgs{:});
xlabel('head ang mag (deg)',axLblArgs{:});
ylabel('head ang err (deg) NO pt2',axLblArgs{:});

hFig(end+1) = figure(40);
hfig = hFig(end);
tstr = sprintf('nIntra=%d. headAngErr dists',nIntra);
set(hfig,'Name',tstr,'Position',[2561 401 1920 1124]);
clf;
axs = mycreatesubplots(1,2,.1);

ax = axs(1);
axes(ax);
tstr = sprintf('intra-pairs head angle error (nIntra=%d)',nIntra);
hist(intra_axisAngRadMag_dLbl/pi*180);
grid on;
title(tstr,tiLblArgs{:});
xlabel('head ang err (deg)',axLblArgs{:});

ax = axs(2);
axes(ax);
tstr = sprintf('NO pt 2');
hist(intra_axisAngRadMag_dLbl_nopt2/pi*180);
grid on;
title(tstr,tiLblArgs{:});
xlabel('head ang err (deg) NO pt2',axLblArgs{:});

linkaxes(axs);

% get row indices of tGT for various angMag ptiles
ANGMAGPTILES = linspace(0,100,8);
nGT = height(tGT);
angMag = tGT.axAngDegXYZ(:,1);
[~,idx] = sort(angMag);
idxidxs = round(ANGMAGPTILES/100*nGT);
idxidxs = max(1,idxidxs);
idxidxs = min(nGT,idxidxs);
fprintf(1,'idxidxs are %s\n',mat2str(idxidxs));
idxPlot = idx(idxidxs); % plot these rows of tGT
% get cached images from lbl from rows of tGT
ppdataMD = lbl.preProcData.MD(:,{'mov' 'frm'});
ppdataMD.Properties.VariableNames{1} = 'mIdx';
ppdataMD.mIdx = int32(ppdataMD.mIdx);
[tf,loc] = tblismember(tGT,ppdataMD,{'mIdx' 'frm'});
all(tf)
IGTcropped = lbl.preProcData.I(loc,:);

for ivw=1:2
  hFig(end+1) = figure(50+ivw);
  hfig = hFig(end);
  tstr = sprintf('Lbl montage, angMag ptiles=%s. Vw%d',mat2str(ANGMAGPTILES,3),ivw);
  set(hfig,'Name',tstr,'Position',[2561 401 1920 1124]);
  pLblAbsVw = tGT.pLbl(:,[1:5 11:15]+(ivw-1)*5);
  roiVw = tGT.roi(:,(1:4)+(ivw-1)*4);
  pLblRoiVw = Shape.p2pROI(pLblAbsVw,roiVw,5);
  framelbls = arrayfun(@(x)sprintf('angMag=%.3f',x),angMag(idxPlot),'uni',0);
  Shape.montage(IGTcropped(:,ivw),pLblRoiVw,...
    'fig',hfig,...
    'nr',2,'nc',4,...
    'idxs',idxPlot,...
    'framelbls',framelbls,...
    'framelblscolor',[1 1 0],...
    'titlestr',tstr);
end

if DOSAVE
  for i=1:numel(hFig)
    h = figure(hFig(i));
    fname = h.Name;
    hgsave(h,fullfile(SAVEDIR,[fname '.fig']));
    set(h,'PaperOrientation','landscape','PaperType','arch-d');
    print(h,'-dpdf',fullfile(SAVEDIR,[fname '.pdf']));  
    print(h,'-dpng','-r300',fullfile(SAVEDIR,[fname '.png']));   
    fprintf(1,'Saved %s.\n',fname);
  end
end
%%
maxStroErr = nanmax([tGT.stroErr3dLbl tGT.stroErr3dTrk],[],2);
mnStroErr = nanmean([tGT.stroErr3dLbl tGT.stroErr3dTrk],2);

fprintf(1,'%d nnan maxStroErr Els.\n',nnz(~isnan(maxStroErr)));
fprintf(1,'%d nnan mnStroErr Els.\n',nnz(~isnan(mnStroErr)));

hFig(end+1) = figure(20);
hfig = hFig(end);
tstr = 'axisAngRadMag vs maxStroErr';
set(hfig,'Name',tstr,'Position',[2561 401 1920 1124]);
clf;
scatter(maxStroErr,tGT.axisAngRadMag_dLblTrk/pi*180);
grid on;
title(tstr,tiLblArgs{:});
% xlabel('angle (deg) inc pt2',axLblArgs{:});
% ylabel('angle (deg) no pt2',axLblArgs{:});
%%














figure(15);
axs = mycreatesubplots(1,2);
for ivw=1:2
  ax = axs(ivw);
  axes(ax);
  plot(uvLbl(:,1,ivw),uvLbl(:,2,ivw),'.','markersize',20);  
  text(uvLbl(2,1,ivw),uvLbl(2,2,ivw),'pt2');
  hold on;
  plot(uvTrk(:,1,ivw),uvTrk(:,2,ivw),'x','markersize',20);
  text(uvTrk(2,1,ivw),uvTrk(2,2,ivw),'pt2');

  roi = tGT(i,:).roi( (1:4) + (ivw-1)*4 );
  axis(roi);
  grid on;
  axis ij
end
  
figure(16);
ax = axes;
plot3(X3Dlbl(:,1),X3Dlbl(:,2),X3Dlbl(:,3));
hold(ax,'on');
plot3(X3Dtrk(:,1),X3Dtrk(:,2),X3Dtrk(:,3));


%%

% other plot: label montage at various angles

% corr, px err vs angle

% these err plots vs SH's angle

% err vs angMag, scatter
hFig(end+1) = figure(30);
hfig = hFig(end);
set(hfig,'Name','Err vs angMag','Position',[2561 401 1920 1124]);
axs = mycreatesubplots(1,2,.1);
angMag = tGT.axAngDegXYZ(:,1);
for ivw=1:2
  ax = axs(ivw);
  axes(ax);
  hold(ax,'on');
  
  errL2vw = errL2(:,:,ivw);
  errL2vwMn = mean(errL2vw,2);
  %plot(angMag,errL2vw,'.');
  plot(angMag,errL2vwMn,'r.','markersize',20);
  
  tstr = sprintf('nGT=%d. view %d',height(tGT),ivw);
  title(tstr,'fontweight','bold','fontsize',18);  
  xlabel('head angle magnitude','fontweight','bold','fontsize',18);
  ylabel('mean GT err (px)','fontweight','bold','fontsize',18);
  grid on;
end
linkaxes(axs);

% err vs angMag, binned
angMag = tGT.axAngDegXYZ(:,1);
QUARTS = [25 50 75];
angMagQuarts = prctile(angMag,QUARTS);
angMagEdges = [0 angMagQuarts inf];
[angMagCnts,angMagBin] = histc(angMag,angMagEdges);
nbin = max(angMagBin);
dxyErrBins = arrayfun(@(x)dxy(angMagBin==x,:,:,:),(1:nbin)','uni',0);

hFig(end+1) = figure(35);
hfig = hFig(end);
set(hfig,'Name','Err vs angMag bin','Position',[2561 401 1920 1124]);
GTPlot.ptileCurves(dxyErrBins,...
  'ptiles',PTILES,...
  'hFig',hfig,...   
  'setNames',arrayfun(@(x)sprintf('angMag Qtile%d',x),1:4,'uni',0),...
  'axisArgs',{'XTicklabelRotation',30,'FontSize' 12} ...
  );
