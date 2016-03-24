%%
RESFILEROOT = 'f:\DropBoxNEW\Dropbox\Tracking_KAJ\track.results';
RESFILES = {
'13@he@for_150723_02_002_04_v2@iTrn@lotsa1__13@he@for_150723_02_002_04_v2@iTstLbl__0225T1009'
'13@he@for_150730_02_002_01_v2@iTrn@lotsa1__13@he@for_150730_02_002_01_v2@iTstLbl__0225T1021' 
'13@he@for_150730_02_002_07_v2@iTrn@lotsa1__13@he@for_150730_02_002_07_v2@iTstLbl__0225T1028'
'13@he@for_150730_02_006_02_v2@iTrn@lotsa1__13@he@for_150730_02_006_02_v2@iTstLbl__0225T1018'  
'13@he@for_150806_01_000_02_v2@iTrn@lotsa1__13@he@for_150806_01_000_02_v2@iTstLbl__0225T1014'  
'13@he@for_150828_01_002_07_v2@iTrn@lotsa1__13@he@for_150828_01_002_07_v2@iTstLbl__0225T1021'  
'13@he@for_150902_02_001_02_v2@iTrn@lotsa1__13@he@for_150902_02_001_02_v2@iTstLbl__0225T1009'  
'13@he@for_150902_02_001_07_v2@iTrn@lotsa1__13@he@for_150902_02_001_07_v2@iTstLbl__0225T1021'  
'13@he@for_151112_01_002_04_v2@iTrn@lotsa1__13@he@for_151112_01_002_04_v2@iTstLbl__0225T1013'  
'13@he@for_151112_01_002_05_v2@iTrn@lotsa1__13@he@for_151112_01_002_05_v2@iTstLbl__0225T1014'
};
RESFILES = fullfile(RESFILEROOT,RESFILES,'res.mat');

TDIROOT = 'f:\cpr\data\jan';
TDI = {
  'tdI@13@for_150723_02_002_04_v2@0224.mat'
  'tdI@13@for_150730_02_002_01_v2@0224.mat'
  'tdI@13@for_150730_02_002_07_v2@0224.mat'
  'tdI@13@for_150730_02_006_02_v2@0224.mat'
  'tdI@13@for_150806_01_000_02_v2@0224.mat'
  'tdI@13@for_150828_01_002_07_v2@0224.mat'
  'tdI@13@for_150902_02_001_02_v2@0224.mat'
  'tdI@13@for_150902_02_001_07_v2@0224.mat'
  'tdI@13@for_151112_01_002_04_v2@0224.mat'
  'tdI@13@for_151112_01_002_05_v2@0224.mat'
  };
TDI = fullfile(TDIROOT,TDI);

%%
assert(numel(RESFILES)==numel(TDI));
for i = 1:numel(TDI)
  rfile = RESFILES{i};
  tdifile = TDI{i};
  [~,info] = FS.parsename(tdifile);
  assessfile = sprintf('f:\\cpr\\data\\jan\\assess_%s.mat',info.note);
  plotfile = sprintf('f:\\cpr\\data\\jan\\assess_%s',info.note);
  trkppAssess('resFile',rfile,'td',td,'tdiFile',tdifile,...
    'saveAssess',assessfile,'savePlots',plotfile);
  
  pause(2);
  close all;
end

%% mega plots
dd = dir('assess_for*.mat');
dd = {dd.name}';

FLDS = {'dTrk47Av' 'dTrnMin' 'dTrnMinTrk' 'xyRepMat47Av'};
s = struct();
for f = FLDS,f=f{1}; %#ok<FXSET>
  s.(f) = nan(0,1);
end
s.dminDistToTrnDShape = nan(0,1);
s.dTrk47AvSubsetForDShape = nan(0,1);
s.lbl = cell(0,1);
s.lbldmin = cell(0,1);

for i = 1:numel(dd)
  assess = load(dd{i});  
  n = numel(assess.dTrk47Av); % this field must be present

  fprintf(1,'%s: n=%d\n',dd{i},n);
  
  assert(all(isfield(assess,FLDS)));
  for f = FLDS,f=f{1}; %#ok<FXSET>
    s.(f) = [s.(f); assess.(f)];
  end
  if isfield(assess,'dminDistToTrnDShape')
    s.dminDistToTrnDShape = [s.dminDistToTrnDShape; assess.dminDistToTrnDShape];
    s.dTrk47AvSubsetForDShape = [s.dTrk47AvSubsetForDShape; assess.dTrk47AvSubsetForDShape];
    s.lbldmin = [s.lbldmin; repmat(dd(i),numel(assess.dminDistToTrnDShape),1)];
  end
  s.lbl = [s.lbl; repmat(dd(i),n,1)];
end

hFig = gobjects(0,1);
hFig(end+1,1) = figure('windowstyle','docked');
clear ax;
ax(1) = subplot(1,2,1);
gscatter(s.dTrnMin,s.dTrk47Av,s.lbl);
[r,p] = corrcoef(s.dTrnMin,s.dTrk47Av);
tstr = sprintf('tracking err vs dist-GT-from-training: r = %.3g,p = %.3g',r(1,2),p(1,2));
title(tstr,'fontweight','bold','interpreter','none');
grid on;
xlabel('dTrnMin','fontweight','bold');
ylabel('dTrk47Av','fontweight','bold');
grid on;
ax(2) = subplot(1,2,2);
gscatter(s.dTrnMinTrk,s.dTrk47Av,s.lbl);
[r,p] = corrcoef(s.dTrnMinTrk,s.dTrk47Av);
tstr = sprintf('tracking err vs dist-Trk-from-training: r = %.3g,p = %.3g',r(1,2),p(1,2));
title(tstr,'fontweight','bold','interpreter','none');
grid on;
xlabel('dTrnMinTrk','fontweight','bold');
ylabel('dTrk47Av','fontweight','bold');
hLeg = findall(hFig,'type','legend');
set(hLeg,'Interpreter','none');
delete(hLeg(1));

hFig(end+1,1) = figure('windowstyle','docked');
gscatter(s.xyRepMat47Av,s.dTrk47Av,s.lbl);
[r,p] = corrcoef(s.xyRepMat47Av,s.dTrk47Av);
tstr = sprintf('tracking err vs repMad: r = %.3g,p = %.3g',r(1,2),p(1,2));
title(tstr,'fontweight','bold','interpreter','none');
grid on;
xlabel('xyRepMad47Av','fontweight','bold');
ylabel('dTrk47Av','fontweight','bold');
delete(findall(hFig(end),'type','legend'));

hFig(end+1,1) = figure('windowstyle','docked');
gscatter(s.dminDistToTrnDShape,s.dTrk47AvSubsetForDShape,s.lbldmin);
[r,p] = corrcoef(s.dminDistToTrnDShape,s.dTrk47AvSubsetForDShape);
tstr = sprintf('tracking err vs dist-from-training dShape: r = %.3g,p = %.3g',...
  r(1,2),p(1,2));
title(tstr,'fontweight','bold','interpreter','none');
grid on;
xlabel('dminDP','fontweight','bold');
ylabel('dTrk47Av','fontweight','bold');
