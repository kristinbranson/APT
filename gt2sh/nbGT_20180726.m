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

%% 
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

%% Step6. SH angle-computer
save cpr_gtres_v00_withanls.mat tGT


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
