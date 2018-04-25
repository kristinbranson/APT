%% Gen splits, round1 track
trkSet50Tst200 = any(parts50(:,end-3:end),2);
nnz(trkSet50Tst200)
trkSet50Comp = any(parts50(:,2:end-4),2);
nnz(trkSet50Comp)

unique(trnSet50+trkSet50Tst200+trkSet50Comp)

save trnSplits_20180418T173507.mat -append trkSet50Tst200 trkSet50Comp

%% Browse results, round* track
load trnDataSH_Apr18.mat
load trnSplits_20180418T173507.mat
%% VIEW1
IVIEW = 1;
%load exp1trk1v1__rc_IFR_crop_vw1_trnSet50col1__IFR_crop__trkSet50Compcol1.mat
%load exp1trk2bestvw1__exp1rc2bestVw1_IFR_crop_vw1_trnSet100Bestcol1__IFR_crop__trkSet100BestCompcol1
%load exp1trk2wrstvw1__exp1rc2wrstVw1_IFR_crop_vw1_trnSet100Wrstcol1__IFR_crop__trkSet100WrstCompcol1
%load exp1trk3bestvw1__exp1rc3bestVw1_IFR_crop_vw1_trnSet150Bestcol1__IFR_crop__trkSet150BestCompcol1.mat
load exp1trk3wrstvw1__exp1rc3wrstVw1_IFR_crop_vw1_trnSet150Wrstcol1__IFR_crop__trkSet150WrstCompcol1.mat

%% VIEW2
IVIEW = 2;
%load exp1trk1v2__rc_IFR_crop_vw2_trnSet50col1__IFR_crop__trkSet50Compcol1.mat
%load exp1trk2bestvw2__exp1rc2bestVw2_IFR_crop_vw2_trnSet100Bestcol2__IFR_crop__trkSet100BestCompcol2
%load exp1trk2wrstvw2__exp1rc2wrstVw2_IFR_crop_vw2_trnSet100Wrstcol2__IFR_crop__trkSet100WrstCompcol2
%load exp1trk3bestvw2__exp1rc3bestVw2_IFR_crop_vw2_trnSet150Bestcol2__IFR_crop__trkSet150BestCompcol2.mat
load exp1trk3wrstvw2__exp1rc3wrstVw2_IFR_crop_vw2_trnSet150Wrstcol2__IFR_crop__trkSet150WrstCompcol2.mat

%% Browse p0s 
figure
ax = axes;
hold(ax,'on');
clrs = lines(5);
for ipt=1:5
  plot(p0(:,ipt),p0(:,ipt+5),'.','markersize',8,'color',clrs(ipt,:));
end
  
%% VTracking errs
errS = sum(errTrk,2);
figure
hist(errS,50);

%% Best/Worst 50

%row = find(trkSet50Comp); % SET ME !! rows into IFR 
%row = find(trkSet100BestComp(:,IVIEW)); % SET ME rows into IFR
ROWSTRACKED = find(trkSet150WrstComp(:,IVIEW)); 

tErr = table(ROWSTRACKED,errS,'VariableNames',{'row' 'errS'});
[~,i] = sort(errS);
tErrBest50 = tErr(i(1:50),:)
tErrWrst50 = tErr(i(end-49:end),:)

nFR = size(IFR_crop,1);
pTrk = nan(nFR,10);
pTrk(ROWSTRACKED,:) = pTstTRed; % all tracked results


%% Browse best 50
ids = strcat(numarr2trimcellstr(tErrBest50.row),'#',numarr2trimcellstr(tErrBest50.errS));

Shape.montage(IFR_crop(:,IVIEW),reshape(xyLbl_FR_crop(:,:,:,IVIEW),nFR,10),...
  'nr',5,'nc',5,'idxs',tErrBest50.row(1:25),'framelbls',ids(1:25),'p2',pTrk);

Shape.montage(IFR_crop(:,IVIEW),reshape(xyLbl_FR_crop(:,:,:,IVIEW),nFR,10),...
  'nr',5,'nc',5,'idxs',tErrBest50.row(26:end),'framelbls',ids(26:end),'p2',pTrk);
%%
%trnSet100Best = repmat(trnSet50,1,2); % cols: vw1/2
%szassert(trnSet100Best,[nFR 2]);
%trnSet100Best(tErrBest50.row,IVIEW) = true;
%sum(trnSet100Best,1)
%bestC = categorical(sum(trnSet100Best,2));
%summary(bestC)

% trnSet150Best = trnSet100Best; % cols: vw1/vw2
% szassert(trnSet150Best,[nFR 2]);

trnSet200Best = trnSet150Best; % cols: vw1/vw2
szassert(trnSet200Best,[nFR 2]);

%%
trnSet200Best(tErrBest50.row,IVIEW) = true;
sum(trnSet200Best,1)
bestC = categorical(sum(trnSet200Best,2));
summary(bestC)
%%
%save trnSplits_20180418T173507.mat -append trnSet100Best 
% save trnSplits_20180418T173507.mat -append trnSet150Best
save trnSplits_20180418T173507.mat -append trnSet200Best



%% Browse worst 50
ids = strcat(numarr2trimcellstr(tErrWrst50.row),'#',numarr2trimcellstr(tErrWrst50.errS));

Shape.montage(IFR_crop(:,IVIEW),reshape(xyLbl_FR_crop(:,:,:,IVIEW),nFR,10),...
  'nr',5,'nc',5,'idxs',tErrWrst50.row(1:25),'framelbls',ids(1:25),'p2',pTrk);

Shape.montage(IFR_crop(:,IVIEW),reshape(xyLbl_FR_crop(:,:,:,IVIEW),nFR,10),...
  'nr',5,'nc',5,'idxs',tErrWrst50.row(26:end),'framelbls',ids(26:end),'p2',pTrk);

% !!! MISLABEL: row 3832 view2. row 2027 view2. !!!

%% 
% trnSet100Wrst = repmat(trnSet50,1,2); % cols: vw1/2
% szassert(trnSet100Wrst,[nFR 2]);
% trnSet100Wrst(tErrWrst50.row,IVIEW) = true;
% sum(trnSet100Wrst,1)
% wrstC = categorical(sum(trnSet100Wrst,2));
% summary(wrstC)

% trnSet150Wrst = trnSet100Wrst; % cols: vw1/vw2
% trnSet150Wrst(tErrWrst50.row,IVIEW) = true;
% sum(trnSet150Wrst,1)
% wrstC = categorical(sum(trnSet150Wrst,2));
% summary(wrstC)

trnSet200Wrst = trnSet150Wrst; % cols: vw1/vw2
trnSet200Wrst(tErrWrst50.row,IVIEW) = true;
sum(trnSet200Wrst,1)
wrstC = categorical(sum(trnSet200Wrst,2));
summary(wrstC)

%%
%save trnSplits_20180418T173507.mat -append trnSet100Best trnSet100Wrst
% save trnSplits_20180418T173507.mat -append trnSet150Best trnSet150Wrst
save trnSplits_20180418T173507.mat -append trnSet200Best trnSet200Wrst

%% Gen splits, roundN trk
load trnSplits_20180418T173507.mat
%%
% trkSet100BestComp = ~trnSet100Best & ~trkSet50Tst200; % cols: vw1/vw2
% trkSet100WrstComp = ~trnSet100Wrst & ~trkSet50Tst200;
% sum(trkSet100BestComp,1)
% sum(trkSet100WrstComp,1)
% 
% unique(trnSet100Best+trkSet100BestComp+trkSet50Tst200)
% unique(trnSet100Wrst+trkSet100WrstComp+trkSet50Tst200)

trkSet200BestComp = ~trnSet200Best & ~trkSet50Tst200; % cols: vw1/vw2
trkSet200WrstComp = ~trnSet200Wrst & ~trkSet50Tst200;
sum(trkSet200BestComp,1)
sum(trkSet200WrstComp,1)

unique(trnSet200Best+trkSet200BestComp+trkSet50Tst200)
unique(trnSet200Wrst+trkSet200WrstComp+trkSet50Tst200)

%%
%save trnSplits_20180418T173507.mat -append trkSet100BestComp trkSet100WrstComp
save trnSplits_20180418T173507.mat -append trkSet200BestComp trkSet200WrstComp



%% RoundN trk results
dd = dir('exp1trk4*.mat');
n = {dd.name}'
res = cellfun(@load,n);
%%
figure;
errS = arrayfun(@(x)sum(x.errTrk,2),res,'uni',0);
errS = cat(2,errS{:});
boxplot(errS)
ylim([0 100]);
ax = gca;
ax.YGrid = 'on'
%% EMP: wrst seems better 
prctile(errS,[50 75 90 95 97.5 99])
%%
ranksum(errS(:,1),errS(:,3))
ranksum(errS(:,2),errS(:,4))



%% analysis
bestwrst = {'best' 'wrst'};
vws = 1:2;
gtset = {'GT' 'Tst200'};
res = cell(2,2,2); % view,bestworst,gt
resnames = cell(2,2,2);
err = cell(2,2,2);
for iBest=1:2
for iVw=vws
for iGt=1:2
  bw = bestwrst{iBest};
  bwCap = bw;
  bwCap(1) = upper(bwCap(1));
  gt = gtset{iGt};
  switch gt
    case 'GT'
      PAT = 'exp1rc4%sVw%d%s__exp1rc4%sVw%d_IFR_crop_vw%d_trnSet200%scol%d__Igt_crop__all.mat';
    case 'Tst200'
      PAT = 'exp1rc4%sVw%d%s__exp1rc4%sVw%d_IFR_crop_vw%d_trnSet200%scol%d__IFR_crop__trkSet50Tst200col1.mat';
  end
  resname = sprintf(PAT,bw,iVw,gt,bwCap,iVw,iVw,bwCap,iVw);
  fprintf('loading %s\n',resname);
  resname = fullfile('matresults',resname);
  res{iVw,iBest,iGt} = load(resname);
  resnames{iVw,iBest,iGt} = resname;
  err{iVw,iBest,iGt} = res{iVw,iBest,iGt}.errTrk;
end
end
end
res = cell2mat(res);

%%
exp1res = load('exp1results.mat');
%%
set(groot,'defaultAxesColorOrder','factory');
co = lines(7);
co = co([1 2 4:end],:); % skip the yellow it's a bit gaint
set(groot,'defaultAxesColorOrder',co); 

%% GT err vs Ntrn: prctiles

DOSAVE = 1;
SAVEDIR = fullfile(pwd,'figsMe');
PTILES = [90 97.5 99];
nTrn = [50 200];
nvw = 2;
npts = 5;
nptile = numel(PTILES);

[~,exp1trncols] = ismember(nTrn,exp1res.ntrns);
exp1err = exp1res.trkerr(:,:,:,exp1trncols); % [nGTx5xnvwxnTrn] GT error from original exps

hFig = figure(21);
clf
set(hFig,'Color',[1 1 1],'Position',[2561 401 1920 1124]);
axs = createsubplots(nvw,npts+1,[.05 0;.12 .12]);
axs = reshape(axs,nvw,npts+1);
for ivw=vws
  for ipt=[pts inf]
    if ~isinf(ipt)
      % normal branch.      
      yexpOrig = prctile(squeeze(exp1err(:,ipt,ivw,:)),PTILES); % [nptlxnTrn] 
      ygtbest = prctile(err{ivw,1,1}(:,ipt),PTILES); % [nptl]
      ygtwrst = prctile(err{ivw,2,1}(:,ipt),PTILES); % [nptl]

      ax = axs(ivw,ipt);
      tstr = sprintf('vw%d pt%d',ivw,ipt);
    else 
      yexpOrig = prctile(squeeze(sum(exp1err(:,:,ivw,:),2)/npts),PTILES); % [nptlxnTrn]
      ygtbest = prctile(sum(err{ivw,1,1},2)/npts,PTILES); % [nptl]
      ygtwrst = prctile(sum(err{ivw,2,1},2)/npts,PTILES); % [nptl]      
      
      ax = axs(ivw,npts+1);
      tstr = sprintf('vw%d, mean allpts',ivw);      
    end
    szassert(yexpOrig,[nptile numel(nTrn)]);
    
    axes(ax);
    tfPlot1 = ivw==1 && ipt==1;
    if tfPlot1
      tstr = ['GT err vs Ntrn: ' tstr];
    end    
    
    x = nTrn;
    hOrig = plot(x(:),yexpOrig');
    hold(ax,'on');
    ax.ColorOrderIndex = 1;
    plot(x(:),[yexpOrig(:,1)';ygtwrst(:)'],'-','linewidth',2);
    ax.ColorOrderIndex = 1;
    plot(x(:),[yexpOrig(:,1)';ygtbest(:)'],'--');
    hBlackLines(1) = plot(nan,nan,'k-');
    hBlackLines(2) = plot(nan,nan,'k-','linewidth',2);
    hBlackLines(3) = plot(nan,nan,'k--');

    args = {'YGrid' 'on' 'XGrid' 'on' 'XScale' 'log' 'XLim' [0 350] 'XTick' nTrn 'XTicklabelRotation',90};    
    set(ax,args{:});    
    title(tstr,'fontweight','bold','fontsize',16);
    
    if tfPlot1
      legH = [hOrig;hBlackLines(:)];
      legstrs = [...
        strcat(numarr2trimcellstr(PTILES'),'%');...
        {'orig';'addWrstTrkd';'addBestTrkd'}];
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
ylim(axs(1,1),[0 60]);
%linkaxes(axs(2,:),'y');
% ylim(axs(2,1),[0 20]);

if DOSAVE
%  set(hFig,'InvertHardCopy','off');
  hgsave(hFig,fullfile(SAVEDIR,'Exp2_GTErrAddBestvsWrstTrkErr.fig'));
  set(hFig,'PaperOrientation','landscape','PaperType','arch-c');
  print(hFig,'-dpdf',fullfile(SAVEDIR,'Exp2_GTErrAddBestvsWrstTrkErr.pdf'));
  %SaveFigLotsOfWays(hFig,'GTErrVsNTrain',{'fig' 'pdf'});
end

