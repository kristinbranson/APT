%% input args

ppbasefile = 'ppbase.mat';
outdir = 'subsetout_20181029T192332';

%%
ppbase = load(ppbasefile);
lblfile = ppbase.lblfile;
ld = load(lblfile,'-mat');
hmdirs = ppbase.lblfilehmdirs;

%% load results
ddres = dir(fullfile(outdir,'*.mat'));
nres = numel(ddres);
PAT = 'imov(?<iMov>[0-9]{2,2})_itgt(?<iTgt>[0-9]{2,2})_sfrm(?<frm0>[0-9]{6,6})_.+.mat';
res = cell(nres,1);
resMD = cell(nres,1);
for i=1:numel(ddres)
  resnameS = ddres(i).name;
  sMD = regexp(resnameS,PAT,'names');
  fns = fieldnames(sMD);
  for f=fns(:)',f=f{1};
    sMD.(f) = str2double(sMD.(f));
  end
  
  resname = fullfile(outdir,resnameS);
  res{i} = load(resname);
  fprintf('Loaded %s\n',resname);
  
  assert(res{i}.targets==sMD.iTgt);
  assert(strcmp(res{i}.hmdir,hmdirs{sMD.iMov}));
  assert(isscalar(res{i}.allppobj));
  
  % whoops GV
  pp = res{i}.allppobj{1};
  for ipt=pp.pts2run
    xyHM = squeeze(pp.postdata.viterbi_grid.x(:,ipt,:));
    xy = PostProcess.UntransformByTrx(xyHM,pp.trx,pp.heatmap_origin); 
    pp.postdata.viterbi_grid.x(:,ipt,:) = xy;
  end
  
  res{i} = pp;
  resMD{i} = sMD;
end

res = cat(1,res{:});
resMD = struct2table(cell2mat(resMD));
  
%% aggregate postdatas
algs = fieldnames(res(1).postdata);
nalg = numel(algs);
pdall = struct();

for ialg=1:nalg
  alg = algs{ialg};  
  pdalg = struct();  
  
  for ires=1:nres
    pdalgI = res(ires).postdata.(alg);
    if isfield(pdalgI,'score') && strncmp(alg,'viterbi',7)
      pdalgI.score = reshape(pdalgI.score,[1 size(pdalgI.score)]);
    end
    
    if ires==1
      pdalg = pdalgI;
    else
      fns = fieldnames(pdalgI);
      for f=fns(:)',f=f{1};
        pdalg.(f) = cat(1,pdalg.(f),pdalgI.(f));
      end
    end
    
    assert(height(res(ires).tblMFT)==size(pdalgI.x,1));    
  end
  
  pdall.(alg) = pdalg;  
end


tMFTall = [];
for ires=1:nres
  t = res(ires).tblMFT;
  t.iMov = repmat(resMD.iMov(ires),height(t),1);
  t.iTgt = repmat(resMD.iTgt(ires),height(t),1);
  assert(t.frm(1)==resMD.frm0(ires));
  
  tMFTall = [tMFTall;t];
end
tMFTall = tMFTall(:,{'iMov' 'frm' 'iTgt'});
tMFTall.Properties.VariableNames{1} = 'mov';
pts2run = res(1).pts2run;



%% GT
tGT = Labeler.lblFileGetLabels(ld);
assert(strcmp(tGT.Properties.VariableNames{1},'mov'));
lbl = ld;
[tf,loc] = tblismember(tGT,tMFTall,MFTable.FLDSID);
%frmsGT = tGT.frm(tf);
isampGT = loc(tf);
nGT = nnz(tf);
npts = lbl.cfg.NumLabelPoints;
lposGT = reshape(tGT.p(tf,:),[nGT npts 2]);
fprintf(1,'%d frmsGT.\n',nGT);

algsAll = fieldnames(pdall);
dxyAll = cell(0,1);
for alg=algsAll(:)',alg=alg{1}; %#ok<FXSET>
  %[n x npts x (x/y) x nviews x nsets]
  tpos = pdall.(alg).x(isampGT,:,:); % nGT x npts x d   
  dxyAll{end+1,1} = tpos-lposGT;
end
dxyAll = cat(5,dxyAll{:});

dxyAllPtsRun = dxyAll(:,pts2run,:,:,:);

[hFig,hAxs] = GTPlot.ptileCurves(dxyAllPtsRun,...
  'ptiles',[50 75 90],...
  'setNames',algsAll,...
  'ptNames',arrayfun(@(x)sprintf('pt%02d',x),pts2run,'uni',0),...
  'createsubplotsborders',[.05 0;.15 .15]...
  );


%% Jumpiness Hist -- ALL LUMPED
algs = fieldnames(pdall);
nAlgs = numel(algs);
npts2run = numel(pts2run)
dzall = []; % [n x npts2run x nalg]. magnitude of jump

hFig = figure(11);
clf;
axs = mycreatesubplots(nAlgs,npts2run,[.1 .05]);
clrs = lines(npts2run);

ALGMARKS = {'.' 'x' '^'};
for ialg=1:nAlgs
  alg = algs{ialg};
  x = pdall.(alg).x; % n x npt x d
  a = diff(x,2,1);
  amag = sqrt(sum(a.^2,3)); % (n-2) x npt
  
  for iipt=1:npts2run
    ipt = pts2run(iipt);
    ax = axs(ialg,iipt);
    axes(ax);
    histogram(amag(:,ipt),0:20);
  end
  
  ylabel(axs(ialg,1),alg,'fontweight','bold','interpreter','none');
end

for iipt=1:npts2run
  linkaxes(axs(:,iipt));
end

%% Jumpiness ptileplot
isampAll = (1:height(tMFTall))';
tMFTallUnFT = unique(tMFTall(:,{'mov' 'iTgt'}));
nUnFT = height(tMFTallUnFT);
fprintf('%d unique (mov,tgt) pairs.\n',nUnFT);
amag = []; % nsamp x npts2run x nalg
for iUnFT=1:nUnFT
  mov = tMFTallUnFT.mov(iUnFT);
  iTgt = tMFTallUnFT.iTgt(iUnFT);
  
  tf = tMFTall.mov==mov & tMFTall.iTgt==iTgt;
  tMFTallThisFT = tMFTall(tf,:);
  isampAllThisFT = isampAll(tf);
  
  maxfrm = max(tMFTallThisFT.frm);
  tffrm = false(maxfrm,1);
  tffrm(tMFTallThisFT.frm) = true;
  [sp,ep] = get_interval_ends(tffrm);
  ep = ep-1;  
  nints = numel(sp);
  
  for iint=1:nints
    idxsp = find(tMFTallThisFT.frm==sp(iint));
    idxep = find(tMFTallThisFT.frm==ep(iint));    
    isamp = isampAllThisFT(idxsp:idxep);
    fprintf('(mov,tgt)=(%d,%d). interval %d, frm%d->%d\n',...
      mov,iTgt,iint,sp(iint),ep(iint));
    
    amagagg = [];
    for ialg=1:nAlgs
      alg = algs{ialg};
      xtmp = pdall.(alg).x(isamp,pts2run,:); % nsamp x npts2run x d
      atmp = diff(xtmp,2,1);
      amagtmp = sqrt(sum(atmp.^2,3)); % (nsamp-2) x npts2run
      
      amagagg = cat(3,amagagg,amagtmp);
    end
    % amagagg is [nsamp-2 x npts2run x nAlgs]
    
    amag = cat(1,amag,amagagg);
  end
end    

amagplot = reshape(amag,[size(amag,1) npts2run 1 1 size(amag,3)]);
[hFig,hAxs] = GTPlot.ptileCurves(amagplot,...
  'ptiles',[50 75 90],...
  'setNames',algsAll,...
  'ptNames',arrayfun(@(x)sprintf('pt%02d',x),pts2run,'uni',0),...
  'createsubplotsborders',[.05 0;.15 .15]...
  );

%% AC cost vs Motion cost

x = pdall.maxdensity_indep.x;

x_a2 = 0.5*(x(1:end-1,:,:)+x(2:end,:,:)); % 2pt m.a.
v = diff(x,1,1); % pd.v(i,ipt,:) gives (dx,dy) that takes you from maxdens_i to maxdens_(i+1)
v(end+1,:,:) = nan;
vmag = sqrt(sum(v.^2,3));
v_a2 = diff(x_a2,1,1); % etc
v_a2(end+1,:,:) = nan;
vmag_a2 = sqrt(sum(v_a2.^2,3));

damp = pp.viterbi_dampen 

% at i; assume motion that took you from i-1->i continues to i+1
x_pred = nan(size(x));
x_pred(3:end,:,:) = x(2:end-1,:,:) + damp*v(1:end-2,:,:);
x_pred_a2 = nan(size(x_a2));
x_pred_a2(3:end,:,:) = x_a2(2:end-1,:,:) + damp*v_a2(1:end-2,:,:);

dxmag_pred = sqrt(sum((x_pred-x).^2,3)); % [n x npt]
dxmag_pred_a2 = sqrt(sum((x_pred_a2-x_a2).^2,3)); % [n x npt]

% figure out typical scale of heatmap/ac
IRES = 1;
pp = res(IRES);
nfrm1 = pp.N;
fprintf(1,'Looking at heatmaps using pp/res idx=%d with N=%d.\n',IRES,nfrm1);
hm_hwhm = nan(nfrm1,npts2run); % half-width-half-max (radius of heatmap dist at half-max)
hm_atmax = nan(nfrm1,npts2run);
hmgrid = pp.heatmapdata.grid{1}; % rows are linear indices, cols are [x y]
for f=1:nfrm1
  if mod(f,10)==0, disp(f); end
  for iipt=1:npts2run
    ipt = pp.pts2run(iipt);
    hm = pp.ReadHeatmapScore(ipt,1,f); % double in [0,1]
    [hmmax,hmmaxidx] = max(hm(:));
    hmmax_xy = hmgrid(hmmaxidx,:);
    hmnzidx = find(hm(:)>0);
    hmnz = hm(hmnzidx);
    hmnz_xy = hmgrid(hmnzidx,:); 
    hmnz_dxy = hmnz_xy-hmmax_xy;
    hmnz_r = sqrt(sum(hmnz_dxy.^2,2)); % dist from nz hm point to hm peak/max
    hmnz_r = round(hmnz_r);
    assert(isequal(size(hmnz),size(hmnz_r)));
    
    hm_atmax(f,ipt) = hmmax;
    
    % for each hmnz_r, find the average hmnz
    hmnz_rgt0 = hmnz(hmnz_r>0);
    hmnz_r_rgt0 = hmnz_r(hmnz_r>0);
    hmnz_r_meanmag = accumarray(hmnz_r_rgt0,hmnz_rgt0,[],@mean);
    
    rhwhm = find(hmnz_r_meanmag<hmmax/2,1);
    hm_hwhm(f,ipt) = rhwhm;
  end
end
    
%% MC distribs
npts2run = numel(pp.pts2run);
hFig = figure(12);
clf;
axs = mycreatesubplots(1,4,[.1 .05;.1 .05]);

ylblargs = {'fontweight' 'bold' 'interpreter' 'none'};
DXFLDS = {dxmag_pred dxmag_pred_a2};
DXFLDNAMES = {'dxmag_pred' 'dxmag_pred_a2'};
for iDx=1:numel(DXFLDS)
  fld = DXFLDNAMES{iDx};
  dxmag = DXFLDS{iDx};
%   %DXdxmag = pdmi.(fld); % n x npt
  dxmag2 = dxmag.^2;
  
  ax = axs(iDx);
  axes(ax)
  boxplot(dxmag(:,pp.pts2run),'labels',pp.pts2run);
  grid on;
  title(fld,'fontweight','bold','interpreter','none');
  
  ax = axs(iDx+2);
  axes(ax)
  boxplot(dxmag2(:,pp.pts2run),'labels',pp.pts2run);
  grid on;
  title(sprintf('%s^2',fld),'fontweight','bold','interpreter','none');
  
%   for iipt=1:npts2run
%     ipt = pp.pts2run(iipt);
%     ax = axs(iDx,iipt);
%     axes(ax);
%     histogram(dxmag(:,ipt));    
%     if iipt==1
%       ylabel(fld,ylblargs{:});
%     end
%     
%     ax = axs(iDx+2,iipt);
%     axes(ax);
%     histogram(dxmag2(:,ipt),100);        
%     if iipt==1
%       ylabel(sprintf('%s^2',fld),ylblargs{:});
%     end
%   end
end
linkaxes(axs(1:2));
linkaxes(axs(3:4));
%% AC distribs (silly)

hFig = figure(14);
clf;
axs = mycreatesubplots(3,npts2run,[.1 .05;.1 .05]);

for iipt=1:npts2run
  ipt = pp.pts2run(iipt);
  
%   zmean = mean(pp.sampledata.z(:,ipt));
%   zmedn = median(pp.sampledata.z(:,ipt));
%   fprintf(2,'Your mean zfac is: %.3f\n',zmean);
%   fprintf(2,'Your median zfac is: %.3f\n',zmedn);

  ax = axs(1,iipt);
  axes(ax);
  histogram(hm_hwhm(:,ipt));
  if iipt==1
    ylabel('hmap_hwhm',ylblargs{:});
  end
  
  ax = axs(2,iipt);
  axes(ax);
  histogram(hm_atmax(:,ipt));
  if ipt==1
    ylabel('max hm',ylblargs{:});
  end
  
  ax = axs(3,iipt);
  axes(ax);
  % diff between AC cost at half-max and at max
  histogram( -log(hm_atmax(:,ipt)/2) + log(hm_atmax(:,ipt)) ,15 );
  % duh this is just log(2)
  if iipt==1
    ylabel('dAC from peak to half',ylblargs{:});
  end
end

% for iipt=1:npts2run
%   linkaxes(axs(1:2,iipt));
%   linkaxes(axs(3:4,iipt));
% end

%%
hFig = figure(17);
clf
axs = mycreatesubplots(2,npts2run,[.1 .05;.1 .05]);
for iipt=1:npts2run
  ipt = pp.pts2run(iipt);
  ax = axs(1,iipt);
  
  vmagI = vmag(1:end-2,ipt); % vmag(1,:) gives vel from t=1->t=2, which is used to predict x @ t=3
  dxmag_pred_tmp = dxmag_pred(3:end,ipt);
  binctrs = 0:15;
  tfcell = arrayfun(@(x)round(vmagI)==x,binctrs,'uni',0);
  dxmag_pred_binmean = cellfun(@(x)median(dxmag_pred_tmp(x)),tfcell);
  
  scatter(ax,vmagI,dxmag_pred_tmp);
  hold(ax,'on');
  plot(ax,binctrs,dxmag_pred_binmean,'r','linewidth',2);
  ylabel(ax,sprintf('pt%d',ipt));
  
  ax = axs(2,iipt);
  vmagI = vmag_a2(1:end-2,ipt);  
  dxmag_pred_tmp = dxmag_pred_a2(3:end,ipt);  
  tfcell = arrayfun(@(x)round(vmagI)==x,binctrs,'uni',0);
  dxmag_pred_binmean = cellfun(@(x)median(dxmag_pred_tmp(x)),tfcell);
  scatter(ax,vmagI,dxmag_pred_tmp);
  hold(ax,'on');
  plot(ax,binctrs,dxmag_pred_binmean,'r','linewidth',2);
end
linkaxes(axs(:));
axis([0 15 0 15]);
