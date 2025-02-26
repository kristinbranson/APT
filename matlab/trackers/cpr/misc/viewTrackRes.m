%%%%%%%%%%%%%%%%%%%%%%%%
%% SETUP USING TD AND RES
%%%%%%%%%%%%%%%%%%%%%%%%
TDFILE = 'td@13@he_pluslblfrms902_2_1_7@0226.mat';
RESFILE = '13@he_pluslblfrms902_2_1_7@for_150902_02_001_07_v2_plus9TrnFrmsForFr1131@iTrn@lotsa1__13@he_pluslblfrms902_2_1_7@@__0226T1442\res.mat';

TDFILE = 'f:\cpr\data\jan\td@13@he@0217.mat';
RESFILE = 'f:\DropBoxNEW\DropBox\Tracking_KAJ\track.results\13@he@for_150902_02_001_07_v2@iTrn@lotsa1__13@he@for_150902_02_001_07_v2@iTstLbl__0225T1021\res.mat';

TDFILE = 'f:\cpr\data\romain\td@@@0301.mat';
RESFILE = 'f:\cpr\data\romain\@@@iTrn@romainR__@@@iTst__0303T1946\res.mat';

% LBLNAME = 'f:\DropBoxNEW\Dropbox\Tracking_KAJ\allen cleaned lbls\150902_02_001_07@20160216.lbl';
% TDFILE = 'f:\DropBoxNEW\Dropbox\Tracking_KAJ\track.results\data\td@150902_02_001_07@@0216.mat';
% RESFILE = 'F:\DropBoxNEW\Dropbox\Tracking_KAJ\track.results\13@he@for_150902_02_001_07_v2@iTrn@lotsa1__13@he@for_150902_02_001_07_v2@iTstLbl__0225T1021\res.mat';
% 
% LBLNAME = 'f:\DropBoxNEW\Dropbox\Tracking_KAJ\allen cleaned lbls\150730_02_006_02@20160216.lbl';
% TDFILE = 'f:\DropBoxNEW\Dropbox\Tracking_KAJ\track.results\data\td@150730_02_006_02@@0216.mat';
% RESFILE = 'F:\DropBoxNEW\Dropbox\Tracking_KAJ\track.results\13@he@for_150730_02_006_02_v2@iTrn@lotsa1__13@he@for_150730_02_006_02_v2@iTstLbl__0225T1018\res.mat';

%TRFILE =
%'11exps@histeq@for_150730_02_006_02@iTrn@lotsa1@781b8@0205T1445.mat';w

%% 
fprintf('Loading TD...\n');
td = load(TDFILE);
td = td.td;

[~,D] = size(td.pGT);
xyGT = reshape(td.pGT,[td.N D/2 2]);

%% 
fprintf('Loading results...\n');
res = load(RESFILE);

%% 
RESULTSTYPE = 'tdI';
RESULTSEXP = '150902_02_001_07@20160216.lbl';
switch RESULTSTYPE
  case 'labeled'
    iRes = 1:size(res.pTstT,1);
    iTst = find(strcmp(td.MD.lblFile,RESULTSEXP) & td.isFullyLabeled);
  case 'tdI'
    iRes = 1:size(res.pTstT,1);
    TDIFILE = 'f:\cpr\data\romain\tdI@halfhalf@@0303.mat';
    iTst = load(TDIFILE);
    iTst = iTst.iTst;
  case 'special'
    iRes = 1; % row indices into res.pTstT, res.pTstTRed
    iTst = 42159; % row indices into td.I 
end
assert(numel(iRes)==numel(iTst));
nTst = numel(iTst);

pTstT = res.pTstT(iRes,:,:,end);
pTstT = permute(pTstT,[1 3 2]);
assert(size(pTstT,2)==D);
assert(size(pTstT,1)==nTst);
RT = size(pTstT,3);
xyTstT = reshape(pTstT,[nTst D/2 2 RT]);

pTstTRed = res.pTstTRed(iRes,:,end);
xyTstTRed = reshape(pTstTRed,[nTst D/2 2]);

lObj = Labeler;
lObj.projNewImStack(td.I,'xyGT',xyGT,...
  'xyTstT',xyTstT,'xyTstTRed',xyTstTRed,'tstITst',iTst);

%%%%%%%%%%%%%%%%%%%%%%%%
%% SETUP USING .LBL and RES
%%%%%%%%%%%%%%%%%%%%%%%%

% movie and GT labels of interest
LBLFILE = 'F:\DropBoxNEW\Dropbox\Tracking_KAJ\allen cleaned lbls\150730_02_002_07@20160216.lbl';

% Results file
%RESFILE = 'F:\cpr\data\jan\13@he_pluslblfrms902_2_1_7@for_150902_02_001_07_v2_plus9TrnFrmsForFr1131@iTrn@lotsa1__13@he_pluslblfrms902_2_1_7@for_150902_02_001_07_v2@iTstLbl__0226T1702\res.mat';
RESFILE = 'F:\DropBoxNEW\Dropbox\Tracking_KAJ\track.results\13@he@for_150730_02_002_07_v2@iTrn@lotsa1__13@he@for_150730_02_002_07_v2@iTstLbl__0225T1028\res.mat';
% ith pt in results corresponds to pt RESIPT2LBLIPT(i) in lbl
RESIPT2LBLIPT = 1:7;
RESXSHIFT = 0;
RESYSHIFT = 0;

% This TDfile should reference the above lblFile in the MD; we need this to
% get iRes
%TDFILE = 'f:\cpr\data\jan\td@13@he_pluslblfrms902_2_1_7@0226';
TDFILE = 'f:\cpr\data\jan\td@13@he@0217';
%TDIFILE = 'f:\cpr\data\jan\tdI@13@for_150902_02_001_07_v2_plus9TrnFrmsForFr1131@0226';
TDIFILE = 'f:\cpr\data\jan\tdI@13@for_150730_02_002_07_v2@0224';
TDIFILEVAR = 'iTstLbl';

%%
td = load(TDFILE);
td = td.td;
res = load(RESFILE);

% iTst are the indices into TD corresponding to res
if isempty(TDIFILE)
  iTst = td.iTst;
else
  tdi = load(TDIFILE);
  iTst = tdi.(TDIFILEVAR);
end
nTst = numel(iTst);
assert(isequal(nTst,size(res.pTstT,1),size(res.pTstTRed,1)));

%%
tstMD = td.MD(iTst,:);
assert(all(strcmp(LBLFILE,tstMD.lblFile)));
fTst = tstMD.frm; % frames corresponding to res

%%
lObj = Labeler;
lObj.projLoadGUI(LBLFILE);
nf = lObj.nframes;
Dlbl = lObj.nLabelPoints*2;

%% pad results, both in frames and pts

nRep = size(res.pTstT,2);
Dtst = size(res.pTstT,3);

pTstTPadFrm = nan(nf,nRep,Dtst);
pTstTRedPadFrm = nan(nf,Dtst);
pTstTPadFrm(fTst,:,:) = res.pTstT(:,:,:,end);
pTstTRedPadFrm(fTst,:,:) = res.pTstTRed(:,:,end);

xyPTstTPad = reshape(permute(pTstTPadFrm,[1 3 2]),[nf Dtst/2 2 nRep]);
xyPTstTRedPad = reshape(pTstTRedPadFrm,[nf Dtst/2 2]);

xyPTstTPadfull = nan(nf,Dlbl/2,2,nRep);
xyPTstTRedPadfull = nan(nf,Dlbl/2,2);
xyPTstTPadfull(:,RESIPT2LBLIPT,:,:) = xyPTstTPad;
xyPTstTRedPadfull(:,RESIPT2LBLIPT,:) = xyPTstTRedPad;

xyPTstTPadfull(:,:,1,:) = xyPTstTPadfull(:,:,1,:) + RESXSHIFT;
xyPTstTPadfull(:,:,2,:) = xyPTstTPadfull(:,:,2,:) + RESYSHIFT;
xyPTstTRedPadfull(:,:,1) = xyPTstTRedPadfull(:,:,1) + RESXSHIFT;
xyPTstTRedPadfull(:,:,2) = xyPTstTRedPadfull(:,:,2) + RESYSHIFT;

%% VIEW USING LABELCORECPRVIEW2
delete(lObj.lblCore);
lc = LabelCoreCPRView2(lObj);
lObj.labelPointsPlotInfo.Colors(RESIPT2LBLIPT,:) = jet(numel(RESIPT2LBLIPT));
lc.init(lObj.nLabelPoints,lObj.labelPointsPlotInfo);
lc.setPs(xyPTstTPadfull,xyPTstTRedPadfull);
lObj.lblCore = lc;
lObj.setFrameGUI(1);

%% Susp
xyLpos = permute(lObj.labeledpos{1},[3 1 2]);
xyLpos(isinf(xyLpos)) = nan; % fully-occluded
assert(isequal(size(xyLpos),size(xyPTstTRedPadfull),[nf Dlbl/2 2]));
pLpos = reshape(xyLpos,[nf Dlbl]);
pTstTRedPadfull = reshape(xyPTstTRedPadfull,[nf Dlbl]);
d = Shape.distP(pLpos,pTstTRedPadfull);
assert(isequal(size(d),[nf Dlbl/2]));
dav = nanmean(d,2);
lObj.setSuspScore({dav});

%% save a lblfile with tracking results
TRACKNOTE = '70best30';
lbl = load(LBLFILE,'-mat');
lbl.labeledpos{1} = permute(xyPTstTRedPadfull,[2 3 1]);
lbl.labeledpostag{1}(:) = [];
[p,f,e] = fileparts(LBLFILE);
lblFileSave = fullfile(p,[f '.track-' TRACKNOTE '.' e]);
assert(exist(lblFileSave,'file')==0,'File %s exists.',lblFileSave);
save(lblFileSave,'-mat','-struct','lbl');

%% plot error by landmark over frames
assert(isequal(size(xyGT),size(xyTstTRedPadded)));
delta = sqrt(sum((xyGT-xyTstTRedPadded).^2,3)); % [nf x npt]
figure;
tfLbled = td.isFullyLabeled;
plot(find(tfLbled),delta(tfLbled,4:7),'-.');
legend({'4' '5' '6' '7'});
grid on;



%% View: time-lapse of GT
figure;
ax = axes;
imshow(td.I{1});
hold(ax,'on');
colors = jet(7);
NSHOW = 20;
PAUSEINT = .1;
hPts = gobjects(NSHOW,4);
for i=1:NSHOW
  for j=4:7
    hPts(i,j) = plot(nan,nan,'o','MarkerFaceColor',colors(j,:));
  end
end
for f=1:5:3661
  i=mod(f,20)+1; 
  for j=4:7
    hPts(i,j).XData = xyGT(f,j);
    hPts(i,j).YData = xyGT(f,j+7);
  end
  title(num2str(f));
  pause(PAUSEINT);
end

%% d (pGT to pred), pred-spread, pred-jump
d = Shape.distP(td.pGT(td.isFullyLabeled,:),pTstTRed);
d = d(:,4:7); % nlfx4

predSD = nan(nlf,4); 
for iTrl = 1:nlf
for iPt = 1:4
  xy = squeeze(xyTstT(iTrl,iPt+3,:,:)); % [2xnrep]
  xymu = mean(xy,2);
  d2 = sum(bsxfun(@minus,xy,xymu).^2,1); % [1xnrep] squared dists from centroid
  predSD(iTrl,iPt) = mean(d2);
end
end

%fprintf(2,'Assuming labeled frames are first n GT frames.\n');
predJMP = nan(nlf,4);
for iTrl = 2:nlf
for iPt = 1:4
  xy1 = squeeze(xyTstTRed(iTrl,iPt+3,:)); % [2x1]
  xy0 = squeeze(xyTstTRed(iTrl-1,iPt+3,:)); % [2x1]
  predJMP(iTrl,iPt) = sum((xy1-xy0).^2);
end
end

figure;
ax = createsubplots(2,4);
ax = reshape(ax,[2 4]);
x = 1+(0:nlf-1)*5;
for iPt = 1:4
  plot(ax(1,iPt),x,5*d(:,iPt),x,predSD(:,iPt),x,predJMP(:,iPt));
  if iPt==1
    legend(ax(1,iPt),'dmu','predSD','predJMP');
    title(ax(1,iPt),num2str(iPt+3));
  end
  axes(ax(2,iPt));
  scatter(5*d(:,iPt),predSD(:,iPt));
end

%% View Ftrs
TRFILE = 'f:\cpr\data\romain\@@@@romainR@unkSHA@0303T1514.mat';
tr = load(TRFILE);
iTrl = 1; % = frame1286, worst susp
muFtrDist = Shape.vizRepsOverTime(td.ITst,res.pTstT,...
  iTrl,tr.regModel.model,'nr',2,'nc',2,'regs',tr.regModel.regs);

%% View channels selected
nReg = numel(regs);
nMini = numel(regs(1).regInfo);
chanmat = nan(nReg,nMini,10);
h1 = figure('windowstyle','docked');
h2 = figure('windowstyle','docked');
ax = createsubplots(1,5);
for iReg = 1:nReg
  f2chan = regs(iReg).ftrPos.xs(:,6);
  for iMini = 1:nMini
    ri = regs(iReg).regInfo{iMini};
    fids = ri.fids(:);
    chans = f2chan(fids);
    chanmat(iReg,iMini,:) = chans;    
    
    for iFern=1:5
      cla(ax(iFern));
      axes(ax(iFern));
      hist(ri.X(:,iFern));
      grid on;
      hold on;
      xlim([-1 1]);
      yl = ylim;
      xt = ri.thrs(iFern);
      plot([xt xt],[yl(1) yl(2)],'r');
      chans = f2chan(ri.fids(:,iFern));
      title(sprintf('Fern %d: chans: %d,%d',iFern,chans(1),chans(2)),...
        'interpreter','none','fontweight','bold');
    end
    
    input(sprintf('iReg %d iMini %d',iReg,iMini));
  end
  
%   tmp = chanmat(iReg,:,:);
%   hist(tmp(:),30);
%   title(sprintf('Channels used for iter=%d',iReg),'fontweight','bold','interpreter','none');
%   grid on;
  
end

%% View Fern stuff
regs = tr.regModel.regs;
thrs1 = nan(0,1);
fernD = nan(0,Dtst/2);
for iReg = 1:numel(regs)
  ri = regs(iReg).regInfo;
  for iRI = 1:numel(ri)
    thrs1(end+1,1) = ri{iRI}.thrs(1);
    ys = reshape(ri{iRI}.ysFern,[32 Dtst/2 2]);
    tmp = sqrt(sum(ys.^2,3)); % [32x7], dist for each pt
    fernD(end+1,:) = mean(tmp,1);
  end
end
    
figure;
x = 1:2500;
plot(x,thrs1);
figure;
plot(x,fernD);