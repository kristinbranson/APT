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
  'inputvariables',{'movID' 'pLblDate' 'lblCat' 'frm'},...
  'groupingvariables','movID',...
  'outputVariableNames',{'lbldate','lblcat','frms'},...
  'numoutputs',3);
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
