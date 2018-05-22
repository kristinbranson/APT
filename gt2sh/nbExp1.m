%%
load trnDataSH_Apr18.mat
load trk__vw2_all.mat
load trk_Igt_crop_vw2_all
%%
nGT = height(tGT);
pGT = reshape(xyLbl_GT_crop(:,:,:,2),nGT,10);
ids = tGT.id2_nonunique;
idxs = randsample(nGT,20);
Shape.montage(Igt_crop(:,2),pGT,'nr',4,'nc',5,'p2',pTstTRed,'idxs',idxs,...
  'framelbls',ids(idxs));

%%
PAT = 'trk__res_IFR_crop_vw%d_trnSetscol%d__Igt_crop__all.mat';
vws = 1:2;
cols = 1:6;
pts = 1:5;
nvw = numel(vws);
ncol = numel(cols);
npts = numel(pts);
trkmats = cell(nvw,ncol);
trkerr = nan(488,5,nvw,ncol);
for ivw=vws
  for c=cols
    trkmats{ivw,c} = load(fullfile('matresults',sprintf(PAT,ivw,c)));
    trkerr(:,:,ivw,c) = trkmats{ivw,c}.errTrk;
  end
end
PAT2 = 'trk__rc_IFR_crop_vw%d_trnSet50col1__Igt_crop__all.mat';
for ivw=vws
  trkmats2{ivw,1} = load(fullfile('matresults',sprintf(PAT2,ivw)));
  trkerr2(:,:,ivw) = trkmats2{ivw,1}.errTrk;
end
trkmats = cell2mat(trkmats);
trkmats2 = cell2mat(trkmats2);
trkmats = [trkmats2 trkmats];
trkerr = cat(4,trkerr2,trkerr);
ntrns = [50 100 200 400 800 1600 3200];

%% xv res
xvmats = cell(nvw,1);
xvmats{1} = load(fullfile('matresults','xv3__xv__IFR_crop__vw2__xvFRsplit3.mat'));
xvmats{2} = load(fullfile('matresults','xv3vw1__xv__IFR_crop__vw1__xvFRsplit3.mat'));
xverr = nan(4961,5,2);
xverr(:,:,1) = cat(1,xvmats{1}.errs{:});
xverr(:,:,2) = cat(1,xvmats{2}.errs{:});

%%
save exp1results.mat trkmats trkerr ntrns xvmats xverr;

%% GT err vs Ntrn: boxplots
hFig = figure(11);
clf
set(hFig,'Color',[1 1 1],'Position',[2561 401 1920 1124]);
axs = createsubplots(nvw,npts+1,.05);
axs = reshape(axs,nvw,npts+1);
for ivw=vws
  for ipt=[pts inf]
    if ~isinf(ipt)
      % normal branch
      errs = squeeze(trkerr(:,ipt,ivw,:)); % nxntrn
      ax = axs(ivw,ipt);
      tstr = sprintf('vw%d pt%d',ivw,ipt);
    else      
      errs = squeeze(sum(trkerr(:,:,ivw,:),2)/npts); % [nxntrn]
      ax = axs(ivw,npts+1);
      tstr = sprintf('vw%d, mean allpts',ivw);
    end
    axes(ax);
    tfPlot1 = ivw==1 && ipt==1;
    
    if tfPlot1
      tstr = ['GT err vs Ntrn: ' tstr];
      args = {'labels' numarr2trimcellstr(ntrns')};
    else
      args = {};
    end
    boxplot(errs,'plotstyle','traditional','outliersize',2,args{:});
    title(tstr,'fontweight','bold');
    ax.YGrid = 'on';
    if tfPlot1
      ylabel('raw err (px)','fontweight','bold');
    end
    if ipt==1
      % none
    else
      set(ax,'XTickLabel',[],'YTickLabel',[]);
    end
  end
  
end
linkaxes(axs(1,:),'y');
linkaxes(axs(2,:),'y');
ylim(axs(1,1),[0 30]);
ylim(axs(2,1),[0 20]);

%% GT err vs Ntrn: prctiles

DOSAVE = true;
SAVEDIR = fullfile(pwd,'figsMe');
PTILES = [50 75 90 95 97.5 99];
XVNTRN = 3307;
nGT = size(errs,1);

hFig = figure(12);
clf
set(hFig,'Color',[1 1 1],'Position',[2561 401 1920 1124]);
axs = createsubplots(nvw,npts+1,[.05 0;.12 .12]);
axs = reshape(axs,nvw,npts+1);
for ivw=vws
  for ipt=[pts inf]
    if ~isinf(ipt)
      % normal branch
      errs = squeeze(trkerr(:,ipt,ivw,:)); % nxntrn
      y = prctile(errs,PTILES); % [nptlsxntrn]

      ax = axs(ivw,ipt);
      tstr = sprintf('vw%d pt%d',ivw,ipt);   
      
      yxv = prctile(xverr(:,ipt,ivw),PTILES); % [nptls]
    else      
      errs = squeeze(sum(trkerr(:,:,ivw,:),2)/npts); % [nxntrn]
      y = prctile(errs,PTILES); % [nptlsxntrn]
      
      ax = axs(ivw,npts+1);
      tstr = sprintf('vw%d, mean allpts',ivw);
      
      xverrmean = squeeze(sum(xverr(:,:,ivw),2)/npts); % [nx1]
      yxv = prctile(xverrmean,PTILES); % [nptls]
    end
    axes(ax);
    tfPlot1 = ivw==1 && ipt==1;
    if tfPlot1
      tstr = ['GT err vs Ntrn: ' tstr];
    end    
    
    args = {'YGrid' 'on' 'XGrid' 'on' 'XScale' 'log' 'XLim' [0 4e3] 'XTick' ntrns 'XTicklabelRotation',90};    
    x = ntrns;
    h = plot(x,y','.-');
    set(ax,args{:});
    hold(ax,'on');
    ax.ColorOrderIndex = 1;
    yxvplot = [yxv(:)' nan]; % add phantom nan to plot by cols    
    hxv = plot(XVNTRN,yxvplot,'*');
    
    title(tstr,'fontweight','bold','fontsize',16);
    if tfPlot1
      legH = [h;hxv(1)];
      legstrs = [...
        strcat(numarr2trimcellstr(PTILES'),'%');...
        {'xv'}];
      legend(legH,legstrs);
      xlabel('Ntrain','fontweight','normal','fontsize',14);

      ystr = sprintf('raw err (px)');
      ylabel(ystr,'fontweight','normal','fontsize',14);
    else
      set(ax,'XTickLabel',[]);
    end
    if ipt==1
    else
      set(ax,'YTickLabel',[]);
    end
  end
end
linkaxes(axs(:),'y');
ylim(axs(1,1),[0 50]);
%linkaxes(axs(2,:),'y');
% ylim(axs(2,1),[0 20]);

if DOSAVE
%  set(hFig,'InvertHardCopy','off');
  hgsave(hFig,fullfile('figsMe','Exp1_GTErrVsNTrain.fig'));
  set(hFig,'PaperOrientation','landscape','PaperType','arch-c');
  print(hFig,'-dpdf',fullfile('figsMe','Exp1_GTErrVsNTrain.pdf'));
  %SaveFigLotsOfWays(hFig,'GTErrVsNTrain',{'fig' 'pdf'});
end

%% KB percentiles

DOSAVE = true;
PLOTFULL = true;

nlandmarks = npts;
nviews = 2;
nptiles = numel(PTILES);
normerr_prctiles = nan(nptiles,nlandmarks,nviews,numel(ntrns)+1); % final 4th D: xv results
for l = 1:nlandmarks
  for v = 1:nviews
    for k = 1:numel(ntrns)
      normerr_prctiles(:,l,v,k) = prctile(trkerr(:,l,v,k),PTILES);
    end
    normerr_prctiles(:,l,v,k+1) = prctile(xverr(:,l,v),PTILES);
  end
end

hfig = 13;
figure(hfig);
clf
set(hfig,'Color',[1 1 1],'Position',[2561 401 1920 1124]);

trns2show = [100 400 3200];
[~,itrns2show] = ismember(trns2show,ntrns);
itrns2show(end+1) = numel(ntrns)+1; % xv results
trns2show(end+1) = XVNTRN;
ntrnsshow = numel(itrns2show);
colors = jet(nptiles);
hax = createsubplots(nviews,ntrnsshow,[.01 .01;.05 .01]);
hax = reshape(hax,[nviews,ntrnsshow]);

h = nan(1,nptiles);
for viewi = 1:nviews
  if PLOTFULL
    im = Igt{1,viewi};
    xyLbl = pLbl2xyvSH(tGT.pLbl);
    xyLbl = squeeze(xyLbl(1,:,:,viewi)); % nptx2
  else
    im = Igt_crop{1,viewi};
    xyLbl = squeeze(xyLbl_GT_crop(1,:,:,viewi)); % [5x2]
  end
  
  for k = 1:ntrnsshow
    ax = hax(viewi,k);
    imagesc(im,'Parent',ax);
    colormap gray
    axis(ax,'image','off');
    hold(ax,'on');
    plot(ax,xyLbl(:,1),xyLbl(:,2),'m+');
    if viewi==1
      if k==1        
        tstr = sprintf('Ntrn=%d, NTst=488 GT',trns2show(k));
      elseif k<ntrnsshow
        tstr = sprintf('Ntrn=%d',trns2show(k));
      else
        tstr = sprintf('Ntrn=%d (xv)',trns2show(k));
      end
      title(ax,tstr,'fontweight','bold','fontsize',16);
    end

    for p = 1:nptiles
      for l = 1:nlandmarks
        rad = normerr_prctiles(p,l,viewi,k);
        h(p) = drawellipse(xyLbl(l,1),xyLbl(l,2),0,rad,rad,...
          'Color',colors(p,:),'Parent',ax,'linewidth',2);
      end
    end
  end
end

set(hfig,'Position',[2561 401 1920 1124]);

legends = cell(1,nptiles);
for p = 1:nptiles
  legends{p} = sprintf('%sth %%ile',num2str(PTILES(p)));
end
hl = legend(h,legends);
set(hl,'Color','k','TextColor','w','EdgeColor','w');
truesize(hfig);

if DOSAVE
  hgsave(hfig,fullfile('figsMe','GTTrkErrPtiles.fig'));
  set(hfig,'PaperOrientation','landscape','PaperType','arch-d');
  print(hfig,'-dpdf',fullfile('figsMe','GTTrkErrPtiles.pdf'));  
end

%% KB: per-landmark frac leq curves

DOSAVEFIGS = 1;
minerr = inf;

predcolors = lines(numel(ntrns));

% trkerr: [nx5x2xnntrns]
fracleqerr = cell(nlandmarks,nviews,numel(ntrns));
for l = 1:nlandmarks
  for v = 1:nviews
    for p = 1:numel(ntrns)
      sortederr = sort(trkerr(:,l,v,p));
      [sortederr,nleqerr] = unique(sortederr);
      fracleqerr{l,v,p} = cat(2,nleqerr./size(trkerr,1),sortederr);
      %minerr = min(minerr,fracleqerr{l,v,p}(find(fracleqerr{l,v,p}(:,1)>=minfracplot,1),2));
    end
  end
end

hfig = 14;
figure(hfig);
set(hfig,'Color',[1 1 1],'Position',[2561 401 1920 1124]);
clf;

hax = createsubplots(nviews,nlandmarks,[.05 0;.1 .1]);
hax = reshape(hax,[nviews,nlandmarks]);

% minmaxerr = inf;
% for p = 1:npredfns,
%   minmaxerr = min(minmaxerr,prctile(vectorize(normerr(:,:,p,:)),99.9));
% end

trns2show = [100 400 3200];
[~,itrns2show] = ismember(trns2show,ntrns);
clrs = lines(numel(trns2show));
clrs = clrs(end:-1:1,:);
clear h;
for l = 1:nlandmarks
  for v = 1:nviews
    ax = hax(v,l);
    hold(ax,'on');
    grid(ax,'on');

    tfPlot1 = v==1 && l==1;    

    for ip=1:numel(itrns2show)
      p = itrns2show(ip);
      h(ip) = plot(ax,fracleqerr{l,v,p}(:,2),fracleqerr{l,v,p}(:,1),'-',...
        'linewidth',1.5,'color',clrs(ip,:));
      tstr = sprintf('vw%d pt%d',v,l);
      if tfPlot1
        tstr = ['ErrCDF vs NTrn: ' tstr];       
      end
      title(ax,tstr,'fontweight','bold','fontsize',16);
    end
%     if l == 1 && v == 1,
%       legend(h,prednames,'Location','southeast');
%     end
%     title(hax(v,l),sprintf('%s, %s',lbld.cfg.LabelPointNames{l},lbld.cfg.ViewNames{v}));

    set(ax,'XTick',[1 2 4 8 16 32],'XScale','log');
    if tfPlot1
      title(ax,tstr,'fontweight','bold');
      xlabel(ax,'Error (raw,  px)','fontsize',14);
      ylabel(ax,'Frac. smaller','fontsize',14);
      
      legstr = strcat('nTrn=',numarr2trimcellstr(trns2show'));
      legend(h,legstr,'location','southeast');
    else
      set(ax,'XTickLabel',[],'YTickLabel',[]);
    end
  end
end

linkaxes(hax(:),'x');
xlim(hax(1),[1 32]);
% set(hax,'XLim',[minerr,minmaxerr],'YLim',[minfracplot,1],'XScale','log');%,'YScale','log');%

% if nlandmarks > 1,
%   xticks = [.01,.025,.05,.10:.10:minmaxerr];
%   xticks(xticks<minerr | xticks > minmaxerr) = [];
%   set(hax,'XTick',xticks);
% end
% yticks = [.01:.01:.05,.1:.1:1];
% yticks(yticks<minfracplot) = [];
% set(hax,'YTick',yticks);
set(hfig,'Units','pixels','Position',[2561 401 1920 1124]);

if DOSAVEFIGS  
  hgsave(hfig,fullfile('figsMe','Exp1_GTErrCDFvsNTrn.fig'));
  set(hfig,'PaperOrientation','landscape','PaperType','arch-c');
  print(hfig,'-dpdf',fullfile('figsMe','Exp1_GTErrCDFvsNTrn.pdf'));  
end

