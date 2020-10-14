%%
I = IMain20180503;
I = Igt;

szmain = cellfun(@size,Igt,'uni',0);
szmain = cellfun(@(x,y)cat(2,x,y),szmain(:,1),szmain(:,2),'uni',0);
szmain = cat(1,szmain{:});
tfbigim = ismember(szmain(:,1),[1024 1024 1024 1024]);

%%
t = tMain20180503;
lblCatC = categorical(t.lblCat);
tlblCatC = sortedsummary(lblCatC)

%%
flyC = categorical(t.flyID);
tflyC = sortedsummary(flyC);
movC = categorical(t.movID);
tmovC = sortedsummary(movC);

%%
DOSAVE = true;
SAVEDIR = 'figsMe';
SAVENAME = 'RowsPerFlyMov';

ts = {tflyC tmovC};
tslbl = {'rows per fly' 'rows per mov'};
xlbl = {'fly #' 'mov #'};

hfig = figure(11);
hfig.Position = [2561 401 1920 1124];
cla;
axs = createsubplots(2,2,.1);
axs = reshape(axs,2,2);
for i=1:2
  ax = axs(1,i);
  axes(ax);
  
  tThis = ts{i};
  plot(tThis.cnts,'.','markersize',12);
  grid('on');
  tstr = sprintf('N=%d. %s: ngrps=%d\n',height(t),tslbl{i},height(ts{i}));
  title(tstr,'fontweight','bold','fontsize',16);
  xlabel(xlbl{i},'fontweight','bold','fontsize',14);
  ylabel('nrows','fontweight','bold','fontsize',14);
  
  ax = axs(2,i);
  axes(ax);
  plot(tThis.cnts,'.','markersize',12);
  grid('on');
  set(ax,'Xscale','linear','YScale','log');
end

if DOSAVE
  hgsave(hfig,fullfile(SAVEDIR,[SAVENAME '.fig']));
  set(hfig,'PaperOrientation','landscape','PaperType','arch-d');
  print(hfig,'-dpdf',fullfile(SAVEDIR,[SAVENAME '.pdf']));  
  print(hfig,'-dpng','-r300',fullfile(SAVEDIR,[SAVENAME '.png']));    
end


%%
flyCun = unique(flyC);
for i=1:numel(flyCun)
  tf=flyCun(i)==flyC;
  if numel(unique(t.lblCat(tf)))>1
    disp(i);
  end
end

%%
DOSAVE = true;
SAVEDIR = 'figsMe';
SAVENAME = 'FrmsLbledInMovs';

hfig = figure(12);
hfig.Position = [2561 401 1920 1124];
clf;
ax = axes;
hold(ax,'on');

CMAPMAX = 100;
clrs = jet(CMAPMAX);
colormap(clrs);
for i=1:height(tmovC)
  tf = movC==tmovC.cats{i};
  f = t.frm(tf);
  assert(issorted(f));
  n = numel(f);
  plot(i,f,'.','markersize',20,'color',clrs(n,:));
end
caxis(ax,[0 CMAPMAX]);
colorbar

tstr = sprintf('N=%d. frames labeled in each mov. Movs in decreasing order of numLbls. numLbls given by color.',numel(movC));
title(tstr,'fontweight','bold','fontsize',16);
xlabel('movie #','fontweight','bold','fontsize',14);
ylabel('frame #','fontweight','bold','fontsize',14);

if DOSAVE
  hgsave(hfig,fullfile(SAVEDIR,[SAVENAME '.fig']));
  set(hfig,'PaperOrientation','landscape','PaperType','arch-d');
  print(hfig,'-dpdf',fullfile(SAVEDIR,[SAVENAME '.pdf']));  
  print(hfig,'-dpng','-r300',fullfile(SAVEDIR,[SAVENAME '.png']));    
end

%% 20180508. KB request, colorful plot but ordered by time
tMDC = rowfun(@findMovDateCat,t,...
  'inputvariables',{'movID' 'pLblDate' 'lblCat' 'frm' 'flyID'},...
  'groupingvariables','movID',...
  'outputVariableNames',{'lbldate','lblcat','frms','flyID'},...
  'numoutputs',4);
[~,idx] = sort(tMDC.lbldate);
tMDC = tMDC(idx,:);
tMDC.datenum = datenum(tMDC.lbldate,'yyyymmdd');

%%
DOSAVE = true;
SAVEDIR = 'figsMe';
SAVENAME = 'FrmsLbledInMovsTimeOrdered';

hfig = figure(15);
hfig.Position = [2561 401 1920 1124];
clf;
ax = axes;
hold(ax,'on');

CMAPMAX = 100;
clrs = jet(CMAPMAX);
colormap(clrs);
for i=1:height(tMDC)
  dtnum = tMDC.datenum(i);
  f = tMDC.frms{i};
  assert(issorted(f));
  n = numel(f);
  plot(i,f,'.','markersize',20,'color',clrs(n,:));
end
caxis(ax,[0 CMAPMAX]);
colorbar

% x-axis is nonlinear time. put ticks at years
assert(issorted(tMDC.lbldate));
xticks = [];
xticklbls = {};
for yr=2013:2018
  i = find(strncmp(tMDC.lbldate,num2str(yr),4),1);
  if ~isempty(i)
    xticks(end+1) = i-0.5;
    xticklbls{end+1} = num2str(yr);
  end
end
i = find(strncmp(tMDC.lbldate,'201707',6),1);
xticks(end+1) = i-0.5;
xticklbls{end+1} = '2017Jul';
i = find(strncmp(tMDC.lbldate,'201711',6),1);
xticks(end+1) = i-0.5;
xticklbls{end+1} = '2017Nov';
xticks(end+1) = height(tMDC)+1;
xticklbls{end+1} = '2018';
set(ax,'XTick',xticks,'XTickLabel',xticklbls,'XTickLabelRotation',45);
set(ax,'fontsize',16,'linewidth',1.5);
xlim([0 510]);
ylim([-50 1500]);
grid(ax,'on');
  
tstr = sprintf('N=%d, nMov=%d. Frames labeled in each mov. Movs in temporal order along x-axis. NumLbls given by color.',...
  height(t),height(tMDC));
title(tstr,'fontweight','bold','fontsize',16);
xlabel('movie # (ordered temporally but with even spacing)','fontweight','bold','fontsize',15);
ylabel('frame #','fontweight','bold','fontsize',15);

if DOSAVE
  hgsave(hfig,fullfile(SAVEDIR,[SAVENAME '.fig']));
  set(hfig,'PaperOrientation','landscape','PaperType','arch-d');
  print(hfig,'-dpdf',fullfile(SAVEDIR,[SAVENAME '.pdf']));  
  print(hfig,'-dpng','-r300',fullfile(SAVEDIR,[SAVENAME '.png']));    
end

%%
tfRm = ismember(t.flyID,flyRm);
tfMDCRm = ismember(tMDC.flyID,flyRm);
dstr = cellstr(t.pLblDate);
dnum = str2double(dstr);
tfOrigJan2017 = strncmp(dstr,'2017',4) & 20170000<=dnum & dnum<201707000;
tfRmJan2017 = tfOrigJan2017 & tfRm;
[tMDC.flyIDcmpct,tmp] = grp2idx(tMDC.flyID);

tstr = sprintf('%d/%d flies, %d/%d movs rm-ed. Remaining: %d/%d flies/movs',... %   nnz(tfRm),numel(tfRm),...
  numel(unique(t.flyID(tfRm))),numel(unique(t.flyID)),...
  nnz(tfMDCRm),numel(tfMDCRm),...
  numel(unique(t.flyID(~tfRm))),numel(tfMDCRm)-nnz(tfMDCRm));
tstr2 = sprintf('first half of 2017: %d/%d rows rm-ed leaving %d\n',...
  nnz(tfRmJan2017),nnz(tfOrigJan2017),nnz(tfOrigJan2017)-nnz(tfRmJan2017));

%% remake time-ordered color plot w/suggested removals
DOSAVE = true;
SAVEDIR = 'figsMe';
SAVENAME = 'FrmsLbledInMovsTimeOrdered_suggRm';

hfig = figure(15);
hfig.Position = [2561 401 1920 1124];
clf;
ax = axes;
hold(ax,'on');

CMAPMAX = 100;
clrs = lightjet(CMAPMAX,.35);
GRAY = 0.4*ones(1,3);
colormap(clrs);
for i=1:height(tMDC)
  dtnum = tMDC.datenum(i);
  f = tMDC.frms{i};
  flyIDcmpct = tMDC.flyIDcmpct(i);
  assert(issorted(f));
  n = numel(f);
  if tfMDCRm(i)
    args = {'x','linewidth',2,'markersize',8,'color',GRAY};
  else
    args = {'.','markersize',20,'color',clrs(flyIDcmpct,:)};
  end
  plot(i,f,args{:});
end
caxis(ax,[0 CMAPMAX]);
colorbar

% x-axis is nonlinear time. put ticks at years
assert(issorted(tMDC.lbldate));
xticks = [];
xticklbls = {};
for yr=2013:2018
  i = find(strncmp(tMDC.lbldate,num2str(yr),4),1);
  if ~isempty(i)
    xticks(end+1) = i-0.5;
    xticklbls{end+1} = sprintf('%s',num2str(yr));
  end
end
i = find(strncmp(tMDC.lbldate,'201707',6),1);
xticks(end+1) = i-0.5;
xticklbls{end+1} = sprintf('2017Jul');
i = find(strncmp(tMDC.lbldate,'201711',6),1);
xticks(end+1) = i-0.5;
xticklbls{end+1} = sprintf('2017Nov');
xticks(end+1) = height(tMDC)+1;
xticklbls{end+1} = sprintf('2018');
set(ax,'XTick',xticks,'XTickLabel',xticklbls,'XTickLabelRotation',45);
set(ax,'fontsize',16,'linewidth',1.5);
xlim([0 510]);
ylim([-50 1500]);
grid(ax,'on');
  
% tstr = sprintf('N=%d, nMov=%d. Frames labeled in each mov. Movs in temporal order along x-axis. NumLbls given by color.',...
%   height(t),height(tMDC));

tstr = [tstr '. Color is fly idx (out of 99 training flies)'];
title({tstr},'fontweight','bold','fontsize',16);
xlabel('movie # (ordered temporally but with even spacing)','fontweight','bold','fontsize',15);
ylabel('frame #','fontweight','bold','fontsize',15);

if DOSAVE
  hgsave(hfig,fullfile(SAVEDIR,[SAVENAME '.fig']));
  set(hfig,'PaperOrientation','landscape','PaperType','arch-d');
  print(hfig,'-dpdf',fullfile(SAVEDIR,[SAVENAME '.pdf']));  
  print(hfig,'-dpng','-r300',fullfile(SAVEDIR,[SAVENAME '.png']));    
end

%% stim
[stimOnOff,stimcase] = arrayfun(@flyNum2stimFrames_SJH,t.flyID,'uni',0);
tfLblInStim = false(height(t),1);
for i=1:height(t)
  stimwins = stimOnOff{i};
  tfwin = stimwins(:,1)<=t.frm(i) & t.frm(i)<=stimwins(:,2);
  tfLblInStim(i) = any(tfwin);
end
t.stimcase = cell2mat(stimcase);
t.lblInStim = tfLblInStim;
%%
[stimOnOffMDC,stimcaseMDC] = arrayfun(@flyNum2stimFrames_SJH,tMDC.flyID,'uni',0);
nFrmsInStim = zeros(height(tMDC),1);
for i=1:height(tMDC)
  stimwins = stimOnOffMDC{i};
  tfFrmsInWin = arrayfun(@(x) any(stimwins(:,1)<=x & x<=stimwins(:,2)), tMDC.frms{i});
  nFrmsInStim(i) = sum(tfFrmsInWin);
end
nFrmsNotInStim = cellfun(@numel,tMDC.frms)-nFrmsInStim;
tMDC.stimcase = cell2mat(stimcaseMDC);

%%
tSim = sortedsummary(categorical(t.stimcase));

%% stim  time-ordered color plot w/suggested removals
DOSAVE = true;
SAVEDIR = 'figsMe';
SAVENAME = 'FrmsLbledInMovsTimeOrdered_stim';

hfig = figure(16);
hfig.Position = [2561 401 1920 1124];
clf;
ax = axes;
hold(ax,'on');

STIMCLR = [255 102 0]/255;
STIMPATCHARGS = {'facealpha',0.5,'linestyle','none','facecolor',STIMCLR};
CMAPMAX = 100;
clrs = cool(CMAPMAX); % ,.35);
GRAY = 0.6*ones(1,3);
colormap(clrs);
for i=1:height(tMDC)
  dtnum = tMDC.datenum(i);
  f = tMDC.frms{i};
  flyID = tMDC.flyID(i);
  flyIDcmpct = tMDC.flyIDcmpct(i);
  assert(issorted(f));
  n = numel(f);
  if tfMDCRm(i)
    args = {'x','linewidth',1,'markersize',8,'color',GRAY};
  else
    args = {'.','markersize',20,'color',clrs(flyIDcmpct,:)};
  end
  plot(i,f,args{:});

  stim = flyNum2stimFrames_SJH(flyID);
  xx = [i-0.5 i-0.5 i+0.5 i+0.5];
  nstim = size(stim,1);
  for ist=1:nstim
    yy = stim(ist,[1 2 2 1]);
    zz = ones(1,4);
    patch(xx,yy,zz,'g',STIMPATCHARGS{:});    
  end
end
caxis(ax,[0 CMAPMAX]);
colorbar

% x-axis is nonlinear time. put ticks at years
assert(issorted(tMDC.lbldate));
xticks = [];
xticklbls = {};
for yr=2013:2018
  i = find(strncmp(tMDC.lbldate,num2str(yr),4),1);
  if ~isempty(i)
    xticks(end+1) = i-0.5;
    xticklbls{end+1} = sprintf('%s',num2str(yr));
  end
end
i = find(strncmp(tMDC.lbldate,'201707',6),1);
xticks(end+1) = i-0.5;
xticklbls{end+1} = sprintf('2017Jul');
i = find(strncmp(tMDC.lbldate,'201711',6),1);
xticks(end+1) = i-0.5;
xticklbls{end+1} = sprintf('2017Nov');
xticks(end+1) = height(tMDC)+1;
xticklbls{end+1} = sprintf('2018');
set(ax,'XTick',xticks,'XTickLabel',xticklbls,'XTickLabelRotation',45);
set(ax,'fontsize',16,'linewidth',1.5);
xlim([0 510]);
ylim([-50 1500]);
grid(ax,'on');
  
tstr = sprintf('Color shows fly idx. Stim in orange. ''x'' shows suggested removal');
%tstr = [tstr '. Color is fly idx'];
title({tstr},'fontweight','bold','fontsize',16);
xlabel('movie # (ordered temporally but with even spacing)','fontweight','bold','fontsize',15);
ylabel('frame #','fontweight','bold','fontsize',15);

if DOSAVE
  hgsave(hfig,fullfile(SAVEDIR,[SAVENAME '.fig']));
  set(hfig,'PaperOrientation','landscape','PaperType','arch-d');
  print(hfig,'-dpdf',fullfile(SAVEDIR,[SAVENAME '.pdf']));  
  print(hfig,'-dpng','-r300',fullfile(SAVEDIR,[SAVENAME '.png']));    
end

%% bodyAxis
tBA = readtable('flynum2bodyAxis.csv');
tBA.Properties.VariableNames = {'flyID' 'lbl' 'iMov' 'frm' 'lblCatSortOf'};

tfRm = ismember(t.flyID,flyRm);
tfGoodCalibAndBA = ~tfRm & ismember(t.flyID,tBA.flyID);
fprintf(1,'%d rows, %d flies have both good calib AND bodyAxis\n',...
  nnz(tfGoodCalibAndBA),numel(unique(t.flyID(tfGoodCalibAndBA))))

%% THE FLY LIST
nbFlyLists;
flyTrn = unique(t.flyID);
flyAll = union(flyTrn,flyEnriched);
flyAll = union(flyAll,flyNonTrainingFliesWithBodyAxis);
flyAll = union(flyAll,unique(tBA.flyID));
flyAll = union(flyAll,flyRmCalib);
nfly = numel(flyAll);
%%
tfTrn = ismember(flyAll,flyTrn);
tfBadCalib = ismember(flyAll,flyRmCalib);
tfBodyAxis = ismember(flyAll,unique(tBA.flyID));
tfEnriched = ismember(flyAll,flyEnriched);
nTrn = zeros(nfly,1);
stimCase = zeros(nfly,1);
for i=1:nfly
  thisfly = flyAll(i);
  nTrn(i) = nnz(t.flyID==thisfly);
  assert(nTrn(i)>0==tfTrn(i));
  [~,stimCase(i)] = flyNum2stimFrames_SJH(thisfly);
end
tFly = table(flyAll,tfTrn,nTrn,tfBadCalib,tfBodyAxis,tfEnriched,stimCase,...
  'VariableNames',{'fly' 'isTrn' 'nTrn' 'isBadCalib' 'isBodyAxis' 'isEnriched' 'stimCase'});
writetable(tFly,'shflies.csv');
%% remake time-ordered color plot w/suggested removals
DOSAVE = true;
SAVEDIR = 'figsMe';
SAVENAME = 'AllFlies';

maxfly = max(tFly.fly);
immat = zeros(5,maxfly); 
immat(1,tFly.fly) = max(0,log(tFly.nTrn))/max(log(tFly.nTrn));
immat(2,tFly.fly) = double(tFly.isBadCalib);
immat(3,tFly.fly) = tFly.isBodyAxis;
immat(4,tFly.fly) = tFly.isEnriched;
immat(5,tFly.fly) = tFly.stimCase/max(tFly.stimCase);

hfig = figure(20);
hfig.Position = [2561 782 1920 445];
clf;
ax = axes;
imagesc(immat);
axis ij;
hCB = colorbar;
hCB.Ticks = [0 1];
ax.YTick = 1:5;
ax.FontSize = 18;
set(ax,'YTickLabel',{'nTrnRows' 'badCalib' 'bodyAxis' 'enriched' 'stimCase'});
  
tstr = sprintf('All Known Flies, nfly=%d',maxfly);
title({tstr},'fontweight','bold','fontsize',16);
xlabel('fly # (SH)','fontweight','bold','fontsize',15);
%ylabel('frame #','fontweight','bold','fontsize',15);

if DOSAVE
  hgsave(hfig,fullfile(SAVEDIR,[SAVENAME '.fig']));
  set(hfig,'PaperOrientation','landscape','PaperType','arch-d');
  print(hfig,'-dpdf',fullfile(SAVEDIR,[SAVENAME '.pdf']));  
  print(hfig,'-dpng','-r300',fullfile(SAVEDIR,[SAVENAME '.png']));    
end

%%
tDLT = readtable('y:\apt\experiments\data\fly2DLT_lookupTableAL.csv');
tSHflies = readtable('y:\apt\experiments\data\shflies.csv');

%%
t = outerjoin(tDLT,tSHflies,'Keys','fly','mergekeys',true);
tf = isnan(t.isTrn);
isequal(tf,isnan(t.nTrn),isnan(t.isBadCalib),isnan(t.isBodyAxis),isnan(t.isEnriched),isnan(t.stimCase))
fprintf('%d nan rows isTrn\n',nnz(tf))
t.isTrn(tf) = false;
t.nTrn(tf) = 0;
t.isBodyAxis(tf) = false;
t.isEnriched(tf) = false;
t.isBodyAxis(tf) = false;
[~,stimcase] = arrayfun(@flyNum2stimFrames_SJH,t.fly,'uni',0);
stimcase = cell2mat(stimcase);
isequal(stimcase(~tf),t.stimCase(~tf))
t.stimCase = stimcase;
isbadcalib = ismember(t.calibfile,badcalibs);
isbadcalibaug = ismember(t.calibfile,badcalibswithcoupleextra);
isequal(t.isBadCalib(~tf),isbadcalib(~tf))
isequal(t.isBadCalib(~tf),isbadcalibaug(~tf))
t.isBadCalib = isbadcalibaug;

%%
writetable(t,'y:\apt\experiments\data\shflies20180518.csv');



%% CPR XV err (easy split) vs head position/rotation

% Subtask: need SH's tracking results for each training row=>need SH's
% movie dir for each training row. Recall tblMD.movFile is the moviepath 
% from original training .lbl file, which can be out-of-date. 
% tblMD.movFile_read is a mix between paths into Mayank's copy of data and
% Stephen's data.
%
% To find 

%t = td.tMain20180503

% SH says 20180529:
% -to summerize anything <90 with Sally or LPTC in the fileaname you can 
% use \flp-chrimson_experiments\fly_1_to_88_Chrimson LPTC FLPouts\
% -anything <90 with the tempVideos and 47D05AD_81B12DBD in the path you 
% can use ChrimsonVideos

load trnData20180503.mat
t = tMain20180503;
%%

n = height(t);
mov = t.movFile;

mov = regexprep(mov,'\\','/');
mov = regexprep(mov,'\$dataroot','Z:/');
mov = regexprep(mov,'W:/home/hustons/','Z:/');

% Sally
mov = regexprep(mov,'/Dm11/hustonlab/sally/videos/',... 
  'Z:/sally/videos/');
mov = regexprep(mov,'Z:/videos','Z:/sally/videos');
mov = regexprep(mov,'Z:sallyideos','Z:/sally/videos');
mov = regexprep(mov,'Z:/sally/videos/flies 1-5/',...
  'Z:/flp-chrimson_experiments/fly_1_to_88_Chrimson LPTC FLPouts/flies 1-9_14072014_GMRss00762 CsChrimson FLP1 intensity 4/10 min heat shock at 36 c/');
mov = regexprep(mov,'Z:/sally/videos/flies 24-27_20072014_GMRss00762 CsChrimson FLP1 intensity 5 1s stimulus/fly 26/',...
  'Z:/flp-chrimson_experiments/fly_1_to_88_Chrimson LPTC FLPouts/flies 24-27_20072014_GMRss00762 CsChrimson FLP1 intensity 5 1s stimulus/fly 26/');

% tempVideos
mov = regexprep(mov,...
  'C:/tempVideos/7_7_13_47D05AD_81B12DBD_x_Chrimsonattp18/data',...
  'Z:/ChrimsonVideos_SUBDIR_TBD_20180529'); 
% 20180529 AL: currently cannot access full dir tree under ChrimsonVideos.
% These are all for flies<90 though so there are no trk results anyway. We
% can always get the correct paths for these guys later if nec

mov = regexprep(mov,'fly_453_to_457_27_9_16/',...
  'fly_453_to_457_27_9_16_norpAkirPONMNchr/');
% mov = regexprep(mov,'fly_453_to_457_27_9_16\',...
%   'fly_453_to_457_27_9_16_norpAkirPONMNchr\');
mov = regexprep(mov,'egFlpTrials4analysis/fly178',...
  'fly_178_to_186_21_1_15_NMN_SS02323_chrimsonFLPouts/fly178');
mov = regexprep(mov,'egFlpTrials4analysis/fly212',...
  'fly_210_to_218_28_10_15_SS02323_x_norpACsChrimsonFlp11/fly212');
mov = regexprep(mov,'egFlpTrials4analysis/fly216',...
  'fly_210_to_218_28_10_15_SS02323_x_norpACsChrimsonFlp11/fly216');
mov = regexprep(mov,...
  'Z:/flp-chrimson_experiments/fly_210_to_218_28_10_15_SS02323_x_norpACsChrimsonFlp11/fly212/fly212_trial1/C001H001S0001/C001H001S0001.avi',...
  'Z:/flp-chrimson_experiments/fly_210_to_218_28_10_15_SS02323_x_norpACsChrimsonFlp11/fly212/fly212_trial1/C001H001S0001/C001H001S0001_c.avi');
mov = regexprep(mov,...
  'Z:/flp-chrimson_experiments/fly_210_to_218_28_10_15_SS02323_x_norpACsChrimsonFlp11/fly212/fly212_trial1/C002H001S0001/C002H001S0001.avi',...
  'Z:/flp-chrimson_experiments/fly_210_to_218_28_10_15_SS02323_x_norpACsChrimsonFlp11/fly212/fly212_trial1/C002H001S0001/C002H001S0001_c.avi');
mov = regexprep(mov,...
  'Z:/flp-chrimson_experiments/fly_210_to_218_28_10_15_SS02323_x_norpACsChrimsonFlp11/fly216/fly216_trial1/C001H001S0001/C001H001S0001.avi',...
  'Z:/flp-chrimson_experiments/fly_210_to_218_28_10_15_SS02323_x_norpACsChrimsonFlp11/fly216/fly216_trial1/C001H001S0001/C001H001S0001_c.avi');
mov = regexprep(mov,...
  'Z:/flp-chrimson_experiments/fly_210_to_218_28_10_15_SS02323_x_norpACsChrimsonFlp11/fly216/fly216_trial1/C002H001S0001/C002H001S0001.avi',...
  'Z:/flp-chrimson_experiments/fly_210_to_218_28_10_15_SS02323_x_norpACsChrimsonFlp11/fly216/fly216_trial1/C002H001S0001/C002H001S0001_c.avi');

mov = regexprep(mov,'egFlpTrials4analysis/fly219',...
  'fly_219_to_228_28_10_15_SS00325_x_norpAcsChrimsonFlp11/fly219');
mov = regexprep(mov,'egFlpTrials4analysis/fly229',...
  'fly_229_to_238_1st_to_2nd_12_15_norpAchrimsonFLP_SS002323/fly229');
mov = regexprep(mov,...
  'Z:/flp-chrimson_experiments/fly_229_to_238_1st_to_2nd_12_15_norpAchrimsonFLP_SS002323/fly229/fly229_300ms_stim/',...
  'Z:/flp-chrimson_experiments/fly_229_to_238_1st_to_2nd_12_15_norpAchrimsonFLP_SS002323/fly229_300ms_stim/');

mov = regexprep(mov,'egFlpTrials4analysis/fly230',...
  'fly_229_to_238_1st_to_2nd_12_15_norpAchrimsonFLP_SS002323/fly230');
% mov = regexprep(mov,'egFlpTrials4analysis/fly230',...
%   'fly_229_to_238_1st_to_2nd_12_15_norpAchrimsonFLP_SS002323/fly230');
mov = regexprep(mov,...
  'Z:/flp-chrimson_experiments/fly_229_to_238_1st_to_2nd_12_15_norpAchrimsonFLP_SS002323/fly230/fly230_300msStimuli/',...
  'Z:/flp-chrimson_experiments/fly_229_to_238_1st_to_2nd_12_15_norpAchrimsonFLP_SS002323/fly230_300msStimuli/');

mov = regexprep(mov,'egFlpTrials4analysis/fly234',...
  'fly_229_to_238_1st_to_2nd_12_15_norpAchrimsonFLP_SS002323/fly234');
mov = regexprep(mov,'egFlpTrials4analysis/fly235',...
  'fly_229_to_238_1st_to_2nd_12_15_norpAchrimsonFLP_SS002323/fly235');
mov = regexprep(mov,...
  'Z:/flp-chrimson_experiments/fly_229_to_238_1st_to_2nd_12_15_norpAchrimsonFLP_SS002323/fly235/fly235_300ms_stimuli',...
  'Z:/flp-chrimson_experiments/fly_229_to_238_1st_to_2nd_12_15_norpAchrimsonFLP_SS002323/fly235_300ms_stimuli');
mov = regexprep(mov,'egFlpTrials4analysis/fly241',...
  'fly_239_to_246_1st_to_2nd_12_15_norpAchrimsonFLP_SS000325/fly241');
mov = regexprep(mov,...
  'Z:/flp-chrimson_experiments/fly_239_to_246_1st_to_2nd_12_15_norpAchrimsonFLP_SS000325/fly241/fly241_300ms_stimuli/',...
  'Z:/flp-chrimson_experiments/fly_239_to_246_1st_to_2nd_12_15_norpAchrimsonFLP_SS000325/fly241_300ms_stimuli/');

mov = regexprep(mov,'egFlpTrials4analysis/fly244',...
  'fly_239_to_246_1st_to_2nd_12_15_norpAchrimsonFLP_SS000325/fly244');
mov = regexprep(mov,'egFlpTrials4analysis/fly245',...
  'fly_239_to_246_1st_to_2nd_12_15_norpAchrimsonFLP_SS000325/fly245');
mov = regexprep(mov,'egFlpTrials4analysis/fly251',...
  'fly_247_to_255_2ndDec15_norpAChrimsonFLP_81B12AD47D05DBD/fly251');
mov = regexprep(mov,'egFlpTrials4analysis/fly254',...
  'fly_247_to_255_2ndDec15_norpAChrimsonFLP_81B12AD47D05DBD/fly254');

mov = regexprep(mov,...
  'Z:/flp-chrimson_experiments/fly_247_to_255_2ndDec15_norpAChrimsonFLP_81B12AD47D05DBD/fly254/fly253_300ms_stimuli/',...
  'Z:/flp-chrimson_experiments/fly_247_to_255_2ndDec15_norpAChrimsonFLP_81B12AD47D05DBD/fly254/fly254_300ms_stimuli/');

mov = regexprep(mov,...
  'Z:/flp-chrimson_experiments/fly_450_to_452_26_9_16/fly450/',...
  'Z:/flp-chrimson_experiments/fly_450_to_452_26_9_16norpAkirPONMNchr/fly450/');

%%
exst = nan(n,2);
for i=1:n
  if mod(i,20)==0
    disp(i);
  end
  for j=1:2
    exst(i,j) = exist(mov{i,j},'file');
  end  
end
%%
tfok = exst(:,1)==2;
tfok2 = exst(:,2)==2;
isequal(tfok,tfok2)

idx = find(~tfok);
mov(idx,1)
%%
tftrk = false(n,2);
trkdate = cell(n,2);
for i=1:n
  for j=1:2
    m = mov{i,j};
    [mP,mF,mE] = fileparts(m);
    %dd = dir(fullfile(mP,'*.trk'));
    dd = dir(fullfile(mP,[mF '.trk']));
    assert(numel(dd)<=1);
    tftrk(i,j) = isscalar(dd);
    if tftrk(i,j)
      trkdate{i,j} = datestr(dd.datenum,'yyyymmdd');
    else
      trkdate{i,j} = '';
    end
%     trkstr = arrayfun(@(x)sprintf('%s (%s)',x.name,datestr(x.datenum,'yyyymmdd')),dd,'uni',0);  
%     fprintf(1,'mov %d, %d trks: %s.\n',i,numel(dd),String.cellstr2CommaSepList(trkstr));
  end
end
%%
isequal(tftrk(:,1),tftrk(:,2))
tftrk = tftrk(:,1);
isequal(trkdate(:,1),trkdate(:,2))
trkdate = trkdate(:,1);
%%
fprintf('%d rows no trk, %d rows trk\n',nnz(~tftrk),nnz(tftrk))
fliesNoTrk = unique(t.flyID(~tftrk))
fliesTrk = unique(t.flyID(tftrk))
%%
tfNoTrk = strcmp(trkdate,'');
tf2017Trk = strncmp(trkdate,'2017',4);
tf2018Trk = strncmp(trkdate,'2018',4);

%%
tMain20180503.movFile_20180529_windows = mov;
save trnData20180503.mat -append tMain20180503

%% for each fly with 2018 tracking data, select a single lbl file to 
% run thru APT2RT to generate median/neutral head posn.
% SH: If you are just trying to get a good 'median head reference point' 
% use the lowest intensity available.  If it is a choice between _wind and 
% _NOwind use "NOwind"

t = tMain20180503;
fliesTrk2018 = unique(t.flyID(tf2018Trk));
nfliesTrk2018 = numel(fliesTrk2018);

APTPROJFILESDIR = 'Z:\flp-chrimson_experiments\APT_projectFiles\';
dd = dir(fullfile(APTPROJFILESDIR,'*.lbl'));
aptlbls = {dd.name}';

fliesTrk2018Lbls = cell(nfliesTrk2018,1);
for ifly=1:nfliesTrk2018
  f = fliesTrk2018(ifly);
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
  fliesTrk2018Lbls{ifly} = fullfile(APTPROJFILESDIR,lblnames{iLblChoose});
  
  fprintf('Fly %d:\n',f);
  for iLbl=1:numel(dd)    
    if iLbl==iLblChoose      
      fprintf(' ... (*) %s (updated %s)\n',dd(iLbl).name,datestr(dd(iLbl).datenum,'yyyymmdd'));
    else
      fprintf(' ... %s (updated %s)\n',dd(iLbl).name,datestr(dd(iLbl).datenum,'yyyymmdd'));
    end
  end
end

%%
tFliesTrk2018 = table(fliesTrk2018,fliesTrk2018Lbls,'VariableNames',...
  {'fly' 'lbl'});
save cprXVerrVsHeadPosn20180529.mat tFliesTrk2018;

%%
FLY2DLT = 'Y:\apt\experiments\data\fly2DLT_lookupTableAL.csv';
tFly2DLT = readtable(FLY2DLT);
tFly2DLT.calibfile = regexprep(tFly2DLT.calibfile,'//groups/huston','/groups/huston');

tFly2DLT2 = tFly2DLT;
tFly2DLT2.calibfile = regexprep(tFly2DLT.calibfile,'/groups/huston/hustonlab/','Z:/');
exst = cellfun(@(x)exist(x,'file'),tFly2DLT2.calibfile);
all(exst==2)

FLY2DLTNEW = 'Y:\apt\experiments\data\fly2DLT_lookupTableAL_win.csv';
writetable(tFly2DLT2,FLY2DLTNEW);
%%
for i=1:nfliesTrk2018
  [axAngDegXYZ,trans,residErr,scaleErr,quat,pivot,refHead] = APT2RT(...
    tFliesTrk2018.lbl{i},... 
    'f:\repo\apt4\user\flynum2bodyAxis.csv',...
    'Y:\apt\experiments\data\fly2DLT_lookupTableAL_win.csv',1)
end


  
