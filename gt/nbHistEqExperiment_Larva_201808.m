%LBL = 'Larva94A04_CM_al.lbl';
%lbl = load(LBL,'-mat');


%%
IMOV = 2;
mfahe = lObj.movieFilesAllHistEqLUT;
imI = mfahe{IMOV}.ISamp;
imJ = mfahe{IMOV}.JSamp;
%% what is imhist doing
[cntI,bins] = imhist(imI,256);
dbins = unique(diff(bins));
binsE = [bins-128; 2^16];
cntI2 = histcounts(imI(:),binsE);
cntI2 = cntI2(:);
isequal(cntI,cntI2)
%%
[cntJ,binsJ] = imhist(imJ,256);
isequal(bins,binsJ)
%%
lut0 = mfahe{IMOV}.lut;
hgram = mfahe{IMOV}.hgram;
ISamp = mfahe{IMOV}.ISamp;
[...
  lut,lutAL,...
  Ibin,binC,binE,intens2bin,...
  J,Jal,...
  Jbin,JbinAL,...
  hI,hJ,hJal,cI,cJ,cJal,...
  Tbin,TbinAL,Tbininv,TbininvAL] = ...
  HistEq.histMatch(ISamp,hgram,'docheck',true);
assert(isequal(lut,lut0));
%%
[J2,hgram2] = imhistmatch(ISamp,mfahe{1}.ISamp,256);
%% Compute Tbininv defined by

% check Tbininv

%% cstar(T(k))
axs = [];

figure;
axs(end+1,1) = axes;
k = 1:256;
plot(k,cI,k,cgram(Tbin),k,cgram(TbinAL),'linewidth',3)
grid on;
legend('cI','cgram(Tbin)','cgram(TbinAL)');

%plot(k,log(hI),k,log(hJ),'linewidth',3);
figure
axs(end+1,1) = axes;
plot(k,cJ,k,cgram,k,cJal,'linewidth',3);
legend('cJ','cgram','cJal');
grid on;

linkaxes(axs);

figure;
axs(end+1,1) = axes;
plot(k,Tbin,k,Tbininv,'linewidth',3)
grid on;
legend('Tbin','Tbininv');


figure
axs = mycreatesubplots(3,1);
axes(axs(1));
imagesc(ISamp);
axes(axs(2));
imagesc(J);
axes(axs(3));
imagesc(Jal);
linkaxes(axs);
linkprop(axs,'CLim');

%%
lbl0 = load('Larva94A04_CM_al_preProc_toyTrained.lbl','-mat');
lbl1 = load('Larva94A04_CM_al_preProcHE_toyTrained.lbl','-mat');
lbl2 = load('Larva94A04_CM_al_preProcCLAHE.lbl','-mat');
lbl3 = load('Larva94A04_CM_al_preProc_MLdefHE.lbl','-mat');
d0 = lbl0.preProcData;
d1 = lbl1.preProcData;
d2 = lbl2.preProcData;
d3 = lbl3.preProcData;

%%
FRMSPERMOV = 3;
nmov = max(d0.MD.mov);
assert(isequal(d0.MD,d1.MD));
if exist('hFig','var')
  deleteValidHandles(hFig);
end
hFig = gobjects(nmov,1);
for imov=1:nmov
  tstr = sprintf('Mov %d\n',imov);
  hFig(imov) = figure('Name',tstr,'Position',[2561 401 1920 1124]);
  axs = mycreatesubplots(4,FRMSPERMOV);
  idxMov = find(d0.MD.mov==imov);
  idxMovSamp = randsample(idxMov,FRMSPERMOV);
  
  for iF=1:FRMSPERMOV
    idx = idxMovSamp(iF);
    assert(d0.MD.mov(idx)==imov);
    f = d0.MD.frm(idx);
    im0 = d0.I{idx};
    im1 = d1.I{idx};
    im2 = d2.I{idx};
    im3 = d3.I{idx};
    
    ax = axs(1,iF);
    axes(ax);
    imagesc(im0);
    colormap gray
    title(sprintf('frm%d',f),'fontweight','bold');
    set(ax,'XTick',[],'YTick',[]);
    
    ax = axs(2,iF);
    axes(ax);
    imagesc(im1);
    colormap gray
    set(ax,'XTick',[],'YTick',[]);
    
    ax = axs(3,iF);
    axes(ax);
    imagesc(im2);
    colormap gray
    set(ax,'XTick',[],'YTick',[]);    
    
    ax = axs(4,iF);
    axes(ax);
    imagesc(im3);
    colormap gray
    set(ax,'XTick',[],'YTick',[]);   
  end
  
  clims = cat(1,axs.CLim);
  climmax = max(clims(:,2));
  arrayfun(@(x)set(x,'CLim',[0 climmax]),axs);
  
  set(hFig(imov),'UserData',axs);
end

%%
lbl0 = load('Larva94A04_CM_al_preProc_toyTrained.lbl','-mat');
d0 = lbl0.preProcData;
for i=1:numel(d0.I)
  d0.I{i} = adapthisteq(d0.I{i});
end
save('Larva94A04_CM_al_preProcCLAHE.lbl','-mat','-struct','lbl0');

%%
lbl0 = load('Larva94A04_CM_al_preProc_toyTrained.lbl','-mat');
d0 = lbl0.preProcData;
for i=1:numel(d0.I)
  d0.I{i} = histeq(d0.I{i}); % Nbin=64 apparently
end
save('Larva94A04_CM_al_preProc_MLdefHE.lbl','-mat','-struct','lbl0');

%%
xv0 = load('xvres_raw_20180822.mat');
xv1 = load('xvres_he_20180822.mat');
xv2 = load('xvres_clahe_20180822.mat');

%%
DOSAVE = true;
SAVEDIR = 'figs';

SETNAMES = {'raw' 'smartHE' 'clahe'};
[n,npts] = size(xv0.xvres.dGTTrk);
nvwfake = 4;
nptsfake = npts/nvwfake;
xverrbig = [];
xverrbig(:,:,:,:,1) = reshape(xv0.xvres.dGTTrk,[n nptsfake 1 nvwfake]);
xverrbig(:,:,:,:,2) = reshape(xv1.xvres.dGTTrk,[n nptsfake 1 nvwfake]);
xverrbig(:,:,:,:,3) = reshape(xv2.xvres.dGTTrk,[n nptsfake 1 nvwfake]);

PERM = [3 1 2];
SETNAMES = SETNAMES(PERM);
xverrbig = xverrbig(:,:,:,:,PERM);

hFig = [];

CREATESUBPLOTBRDRS = [.06 0;.12 .03];

PTILES = [60 90];
hFig(end+1,1) = figure(11);
hfig = hFig(end);
set(hfig,'Name','HistEq flavors','Position',[2561 401 1920 1124]);
[~,ax] = GTPlot.ptileCurves(xverrbig,...
  'hFig',hfig,...
  'setNames',SETNAMES,...
  'ptiles',PTILES,...
  'createsubplotsborders',CREATESUBPLOTBRDRS,...
  'titleArgs',{}...
  );
%  'axisArgs',{'XTicklabelRotation',90,'FontSize' 8});
ax(4,1).XTickLabel = ax(1,1).XTickLabel;
ax(1,1).XTickLabel = [];

PTILES = [94 97];
hFig(end+1,1) = figure(13);
hfig = hFig(end);
set(hfig,'Name','HistEq flavors high ptiles','Position',[2561 401 1920 1124]);
[~,ax] = GTPlot.ptileCurves(xverrbig,...
  'hFig',hfig,...
  'setNames',SETNAMES,...
  'ptiles',PTILES,...
  'createsubplotsborders',CREATESUBPLOTBRDRS,...
  'titleArgs',{}...
  );
%  'axisArgs',{'XTicklabelRotation',90,'FontSize' 8});
ax(4,1).XTickLabel = ax(1,1).XTickLabel;
ax(1,1).XTickLabel = [];

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
