%% input args

DATASET = 'sh';

switch DATASET
  case 'bub'
    ppbasefile = 'ppbase.mat';
    outdir = 'mftsbub_out_20181101T090301';

    ppbase = load(ppbasefile);
    lblfile = ppbase.lblfile;
    ld = load(lblfile,'-mat');
    hmdirs = ppbase.lblfilehmdirs;

  case 'sh'
    outdirs = {
      'out_1_90_20181108T083201'
      'out_91_719_20181109T194358'
    };
    
    lblfile = '/groups/branson/bransonlab/apt/experiments/data/sh_trn4879_gtcomplete_cacheddata.lbl';
    ld = load(lblfile,'-mat');
end

%% load results

tic

switch DATASET
  case 'bub'
    ddres = dir(fullfile(outdir,'*.mat'));
    nres = numel(ddres);
    PAT = 'imov(?<iMov>[0-9]{2,2})_itgt(?<iTgt>[0-9]{2,2})_sfrm(?<frm0>[0-9]{6,6})_nfrm(?<nfrm>[0-9]{6,6}).mat';
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
      pp = res{i}.allppobj{1};
      if pp.N~=sMD.nfrm
        warningNoTrace('res %s, pp.N=%d. probably truncated by trx start/endframe. resetting sMD.nfrm to %d.\n',...
          resnameS,pp.N,pp.N);
        sMD.nfrm = pp.N;
      end

      % whoops GV
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
    
  case 'sh'
    resAgg = [];
    hmdirAgg = [];
    for i=1:numel(outdirs)
      outdir = outdirs{i};
      ddres = dir(fullfile(outdir,'*.mat'));
      nres = numel(ddres);
      res = cell(nres,1);
      hmdir = cell(nres,1);
      for i=1:nres
        resnameS = ddres(i).name;
        resname = fullfile(outdir,resnameS);
        res{i} = load(resname);
        fprintf('Loaded %s\n',resname);

        assert(isscalar(res{i}.allppobj));
        hmdir{i} = res{i}.hmdir;
        res{i} = res{i}.allppobj{1};
      end

      res = cat(1,res{:});

      resAgg = cat(1,resAgg,res);
      hmdirAgg = cat(1,hmdirAgg,hmdir);
    end
    
    res = resAgg;
    hmdir = hmdirAgg;
    nres = numel(res);
    
end

ttoc = toc;
fprintf(1,'Results loaded in %d s\n',round(ttoc));

%% make tMFTall
switch DATASET
  case 'bub'
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
    
    tGT = Labeler.lblFileGetLabels(ld);
    assert(strcmp(tGT.Properties.VariableNames{1},'mov'));
    lbl = ld;
    npts = lbl.cfg.NumLabelPoints;

  case 'sh'
    
    tGT = Labeler.lblFileGetLabels(ld);
    assert(strcmp(tGT.Properties.VariableNames{1},'mov'));
    tGT = tGT(:,1:4);
    tGT.p = tGT.p(:,[6:10 16:20]);
    lbl = ld;
    npts = lbl.cfg.NumLabelPoints;
    
    MOVFILE = '/groups/branson/bransonlab/apt/tmp/postproc/sh/shtrn719_vw2_movs.txt';
    movs = readtxtfile(MOVFILE);
    szassert(movs,size(lbl.movieFilesAll(:,2)));
    movdirs = cellfun(@fileparts,movs,'uni',0);
%     
%     tGT.iMov = tGT.mov;
%     tGT.mov = movs(tGT.mov);
    
    hmdirimov = nan(nres,1);
    tMFTall = [];
    for ires=1:nres
      hmd = hmdir{ires};
      idx = strfind(hmd,'/hmap/');
      idx = idx+numel('/hmap/')-1;
      hmd = hmd(idx:end);
      hmdMovDir = fileparts(hmd);
      
      imov = find(strcmp(hmdMovDir,movdirs));
      assert(isscalar(imov));
      hmdirimov(ires) = imov;

      n = res(ires).N;
      t = table(repmat(imov,n,1),(1:n)',ones(n,1),...
        'VariableNames',{'mov' 'frm' 'iTgt'});

      tMFTall = [tMFTall;t];
    end
    pts2run = res(1).pts2run;
end

%% adjust for crops if nec
tfcrop = false;
switch DATASET
  case 'sh'
    tfcrop = true;
    CROPFILE = '/groups/branson/bransonlab/apt/tmp/postproc/sh/shtrn719_vw2_crops.txt';
    crops = dlmread(CROPFILE);
    szassert(crops,[numel(movs) 4]);
end
  
%% aggregate postdatas
algs = fieldnames(res(1).postdata);
nalg = numel(algs);
pdall = struct();

DAMPS = 0:.05:1;
nDamp = numel(DAMPS);    

Nall = sum([res.N]);
fprintf('Nall=%d\n',Nall);

for ialg=1:nalg
  alg = algs{ialg};  
  pdalg = struct();
  fprintf(1,'### Alg is %s\n',alg);
  
  for ires=1:nres
    if mod(ires,10)==0
      disp(ires);
    end
    
    pp = res(ires);
    pdalgI = res(ires).postdata.(alg);
    if isfield(pdalgI,'score') && strncmp(alg,'viterbi',7) && size(pdalgI.score,1)~=1
      pdalgI.score = reshape(pdalgI.score,[1 size(pdalgI.score)]);
    end
    
    % compute derived stats
    x = pdalgI.x; % [n x npt x 2]
    
    if tfcrop
      imov = hmdirimov(ires);
      roi = crops(imov,:);
      x(:,:,1) = x(:,:,1)+roi(1)-1;
      x(:,:,2) = x(:,:,2)+roi(3)-1;
    end
    
    v = diff(x,1,1); % v(i,ipt,:) gives (dx,dy) that takes you from t=i to t=i+1
    v(end+1,:,:) = nan; % so v has same size as x, [n x npt x 2]
    vmag = sqrt(sum(v.^2,3)); 
    a = diff(x,2,1); 
    a = cat(1,nan(1,pp.npts,2),a,nan(1,pp.npts,2));
    % a(i,ipt,:) gives finite-diff accel, centered at t=i (using
    % t=i-1,i,i+1)
    amag = sqrt(sum(a.^2,3)); % (n-2) x npt

    assert(isequal(size(x),size(v),size(a)));
    assert(isequal(size(vmag),size(amag)));
    
    % compute x_pred using actual damping used
    damp = pp.viterbi_dampen; 
    % at i; assume motion that took you from i-1->i continues to i+1
    x_pred = nan(size(x));
    x_pred(3:end,:,:) = x(2:end-1,:,:) + damp*v(1:end-2,:,:);
    dx_pred = x-x_pred;
    dx_pred_mag = sqrt(sum(dx_pred.^2,3)); % [n x npt]

    assert(isequal(size(x),size(x_pred),size(dx_pred)));
    szassert(dx_pred_mag,size(vmag));
    
    % compute x_pred using arbitrary damps
    [n,npt,d] = size(x);
    x_preddamps = nan(n,npt,d,nDamp);
    x_preddamps(3:end,:,:,:) = x(2:end-1,:,:) + reshape(DAMPS,[1 1 1 nDamp]).*v(1:end-2,:,:); % at i; assume motion that took you from i-1->i continues to i+1 
    dx_preddamps = x-x_preddamps;
    dx_preddamps_mag = sum(dx_preddamps.^2,3); % [n x npt x 1 x nDamp]
    dx_preddamps_mag = reshape(dx_preddamps_mag,[n npt nDamp]);
    szassert(dx_preddamps,size(x_preddamps));
    
    pdalgI.x = x;
    pdalgI.v = v;
    pdalgI.vmag = vmag;
    pdalgI.a = a;
    pdalgI.amag = amag;
    pdalgI.x_pred = x_pred;
    pdalgI.dx_pred = dx_pred;
    pdalgI.dx_pred_mag = dx_pred_mag;
    pdalgI.x_preddamps = x_preddamps;
    pdalgI.dx_preddamps = dx_preddamps;
    pdalgI.dx_preddamps_mag = dx_preddamps_mag;

    res(ires).postdata.(alg) = pdalgI; % we only added stuff to pdalgI
    
    assert(isequal(n,pp.N,size(x,1),size(v,1),size(vmag,1),size(a,1),size(amag,1),size(x_pred,1),size(dx_pred,1),size(dx_pred_mag,1),...
      size(x_preddamps,1),size(dx_preddamps,1),size(dx_preddamps_mag,1)));

    if ires==1
      pdalg = pdalgI;
      
      % set this for all ires for this current alg
      fldsExpand = fieldnames(pdalg);
      if ismember('score',fldsExpand) && size(pdalg.score,1)==1
        fldsExpand = setdiff(fldsExpand,'score');
      end
      
      fprintf('Expanding fields: %s\n',String.cellstr2CommaSepList(fldsExpand));
      for f=fldsExpand(:)';f=f{1};
        if isnumeric(pdalg.(f))          
          pdalg.(f)(Nall,:) = nan;
        elseif islogical(pdalg.(f))
          pdalg.(f)(Nall,:) = false;
        else
          assert(false);
        end
      end
      
      pdalg.ires = zeros(Nall,1);
      pdalg.ires(1:n) = 1;
      
      pdalgExpandCurrRow = n; % pdalg.(expandfld) is filled thru this row      
    else
      fns = fieldnames(pdalgI);
      for f=fns(:)',f=f{1};
        if ismember(f,fldsExpand)
          pdalg.(f)(pdalgExpandCurrRow+1:pdalgExpandCurrRow+n,:) = reshape(pdalgI.(f),n,[]);
        else
          pdalg.(f) = cat(1,pdalg.(f),pdalgI.(f));
        end
      end
      
      pdalg.ires(pdalgExpandCurrRow+1:pdalgExpandCurrRow+n) = ires;
      
      pdalgExpandCurrRow = pdalgExpandCurrRow+n;
    end
    
    switch DATASET
      case 'bub'
        assert(height(res(ires).tblMFT)==size(pdalgI.x,1));    
    end
  end
  
  assert(pdalgExpandCurrRow==Nall);
  
  pdall.(alg) = pdalg;  
end



%% GT
[tf,loc] = tblismember(tGT,tMFTall,MFTable.FLDSID);
%frmsGT = tGT.frm(tf);
isampGT = loc(tf);
nGT = nnz(tf);
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

im = zeros(60,40);
I = {im};
xyLbl = [25 45;25 30;25 15;15 15;15 30;15 45];

% SEE Also GT+Damp below
[hFig,hAxs] = GTPlot.bullseyePtiles(dxyAllPtsRun,I,xyLbl,...
  'ptiles',[50 75 90],...
  'setNames',algsAll,...
  'lineWidth',2,...
  'contourtype','ellipse');

%% Jumpiness Hist -- ALL LUMPED
% algs = fieldnames(pdall);
% nAlgs = numel(algs);
% npts2run = numel(pts2run)
% dzall = []; % [n x npts2run x nalg]. magnitude of jump
% 
% hFig = figure(11);
% clf;
% axs = mycreatesubplots(nAlgs,npts2run,[.1 .05]);
% clrs = lines(npts2run);
% 
% ALGMARKS = {'.' 'x' '^'};
% for ialg=1:nAlgs
%   alg = algs{ialg};
%   x = pdall.(alg).x; % n x npt x d
%   a = diff(x,2,1);
%   amag = sqrt(sum(a.^2,3)); % (n-2) x npt
%   
%   for iipt=1:npts2run
%     ipt = pts2run(iipt);
%     ax = axs(ialg,iipt);
%     axes(ax);
%     histogram(amag(:,ipt),0:20);
%   end
%   
%   ylabel(axs(ialg,1),alg,'fontweight','bold','interpreter','none');
% end
% 
% for iipt=1:npts2run
%   linkaxes(axs(:,iipt));
% end

%% Jumpiness ptileplot
amag = []; % nsamp x npts2run x nalg
for ialg=1:nalg
  alg = algs{ialg};
  amagtmp = pdall.(alg).amag(:,pts2run,:); 
  amag = cat(3,amag,amagtmp);
end
% amagagg is [Nall x npts2run x nAlgs] with some nans at ends of segments
% etc

npts2run = numel(pts2run);
hFig = figure(15);
amagplot = reshape(amag,[Nall npts2run 1 1 nalg]);
[hFig,hAxs] = GTPlot.ptileCurves(amagplot,...
  'ptiles',[50 75 90],...
  'hFig',hFig,...
  'setNames',algsAll,...
  'ptNames',arrayfun(@(x)sprintf('pt%02d',x),pts2run,'uni',0),...
  'createsubplotsborders',[.05 0;.15 .15]...
  );

%% viterbi Basin

% tGT = Labeler.lblFileGetLabels(ld);
% assert(strcmp(tGT.Properties.VariableNames{1},'mov'));
% lbl = ld;
%%
% if no thresh, then super-peaked distros in rescaled space and the central 
% y-val is indistinguishable from y=1. but using only larger vmags (above
% thresh etc) looks better.
VMAG_MINOKTHRESH = 3;

[tf,loc] = tblismember(tGT,tMFTall,MFTable.FLDSID);
%frmsGT = tGT.frm(tf);
isampGT = loc(tf);
nGT = nnz(tf);
npts = lbl.cfg.NumLabelPoints;
lposGT = reshape(tGT.p(tf,:),[nGT npts 2]);
fprintf(1,'%d frmsGT.\n',nGT);

pdmi = pdall.maxdensity_indep;
assert(max(pdmi.ires)==numel(res));

lposGTuse = cell(npts2run,1); 
% lposGTuse{iipt} is nan(0,2); % [nuse x 2] labels
tposGTuse = cell(npts2run,1);
% tposGTuse{iipt} is nan(0,2,3); % [nuse x 2 x 3] tracked locs (maxdensity). 4th dim: [t-2,t-1,t]

for iipt=1:npts2run
  ipt = pts2run(iipt);
  lposGTuse{iipt} = nan(0,2);
  tposGTuse{iipt} = nan(0,2,3);
  for iisampGT=1:nGT
    isamp = isampGT(iisampGT);
    tMFTwin = tMFTall(isamp-2:isamp,:);

    vmagtmp = pdmi.vmag(isamp-2:isamp-1,ipt);
    if all(pdmi.ires(isamp-2:isamp)==pdmi.ires(isamp)) && ...
        all(tMFTwin.mov==tMFTwin.mov(1)) && ...
        all(tMFTwin.iTgt==tMFTwin.iTgt(1)) && ...
        isequal(tMFTwin.frm,(tMFTwin.frm(1):tMFTwin.frm(3))') && ...
        all(vmagtmp(:)>VMAG_MINOKTHRESH)
      lposGTuse{iipt}(end+1,:) = lposGT(iisampGT,ipt,:);
      tmp = pdmi.x(isamp-2:isamp,ipt,:);
      tmp = reshape(tmp,3,2)';
      tposGTuse{iipt}(end+1,:,:) = tmp;
    else
      warningNoTrace('iissampGT=%d, skipping, not enough precursors or vmag too small.',iisampGT);
    end
  end

  fprintf('iipt=%d,ipt=%d\n',iipt,ipt);  
  size(lposGTuse{iipt})
  size(tposGTuse{iipt})
  errGT = lposGTuse{iipt}-tposGTuse{iipt}(:,:,3);
  errGT = sqrt(sum(errGT.^2,2));
  mean(errGT)
  
  pause(5);
end

%%
npts2run = numel(pts2run);
tformpts = cell(npts2run,1);
% tformpts{iipt} = nan(nuse,2,4); % tformed pts. 4th dim: t-2,t-1,t,lbl
for iipt=1:npts2run
  ipt = pts2run(iipt);
  
  nuse = size(lposGTuse{iipt},1);
  tformpts{iipt} = nan(nuse,2,4);

  for iuse=1:nuse
    if mod(iuse,10)==1
      disp(iuse);
    end
    
    lpos = lposGTuse{iipt}(iuse,:);
    tpos = squeeze(tposGTuse{iipt}(iuse,:,:));
    tpos = tpos';
    szassert(lpos,[1 2]);
    szassert(tpos,[3 2]);
    
    if ~isequal(tpos(1,:),tpos(2,:))
      tform = fitgeotrans(tpos(1:2,:),[0 0;0 1],'nonreflectivesimilarity');
      xypts = [tpos; lpos];
      szassert(xypts,[4 2]);
      
      xytform = tform.transformPointsForward(xypts);
      tformpts{iipt}(iuse,:,:) = xytform';
    else
      warningNoTrace('iuse %d, ipt %d. tpos1 and tpos2 are identical (v=0). cannot reproject',iuse,ipt);
    end
    
  end
end
%%
hFig = figure(11);
clf;
axs = mycreatesubplots(2,npts2run,[.1 .05;.1 .05]);
for iipt=1:npts2run
  ipt = pts2run(iipt);
  
  nuse = size(tformpts{iipt},1);
  fprintf('\niipt=%d, ipt=%d, nuse=%d\n',iipt,ipt,nuse);
  
  xy = squeeze(tformpts{iipt}(:,:,3)); % trk at t  
  ifo = GTPlot.gaussianFit(xy);  
  fprintf('ipt=%d. gaussian fit trk@t. mean: %s\n',pts2run(iipt),mat2str(ifo.mean,3));
  fprintf('ipt=%d. trk@t. y median: %s\n',pts2run(iipt),num2str(median(xy(:,2)),3));
  %[v,d] = eig(ifo.cov)

  xy = squeeze(tformpts{iipt}(:,:,4)); % lbl at t  
  ifolbl = GTPlot.gaussianFit(xy);
  fprintf('ipt=%d. gaussian fit lbl@t. mean: %s\n',pts2run(iipt),mat2str(ifolbl.mean,3));
  fprintf('ipt=%d. lbl@t. y median: %s\n',pts2run(iipt),num2str(median(xy(:,2)),3));
  %[v,d] = eig(ifolbl.cov)
  
  ax = axs(1,iipt);
  axes(ax)
  %plot(tformpts(:,iipt,1,3),tformpts(:,iipt,2,3),'r.');
  s = scatterhistogram(tformpts{iipt}(:,1,3),tformpts{iipt}(:,2,3));
  s.XLimits = [-25 25];
  s.YLimits = [-25 25];
  %grid on;
  
  ax = axs(2,iipt);
  axes(ax)
  %plot(tformpts(:,iipt,1,4),tformpts(:,iipt,2,4),'r.');
  s = scatterhistogram(tformpts{iipt}(:,1,4),tformpts{iipt}(:,2,4));
  s.XLimits = [-25 25];
  s.YLimits = [-25 25];
  %grid on;  
end
%linkaxes(axs(:));
%axis equal square


%% AC cost vs Motion cost

%x = pdall.maxdensity_indep.x;

% figure out typical scale of heatmap/ac
[nmax,IRES] = max([res.N]);
%IRES = 1;
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
axs = mycreatesubplots(1,2,[.1 .05;.1 .05]);

ylblargs = {'fontweight' 'bold' 'interpreter' 'none'};
%DXFLDS = {dxmag_pred dxmag_pred_a2};
DXFLDNAMES = {'dx_pred_mag'};
for iDx=1:numel(DXFLDNAMES)
  fld = DXFLDNAMES{iDx};
  dxmag = pdall.maxdensity_indep.(fld);
%  dxmag = DXFLDS{iDx};
%   %DXdxmag = pdmi.(fld); % n x npt
  dxmag2 = dxmag.^2;
  
  ax = axs(iDx);
  axes(ax)
  boxplot(dxmag(:,pp.pts2run),'labels',pp.pts2run);
  grid on;
  title(fld,'fontweight','bold','interpreter','none');
  
  ax = axs(iDx+1);
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
%linkaxes(axs(1:2));
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

%% dxmag_pred_mag vs vmag
hFig = figure(17);
clf
axs = mycreatesubplots(2,npts2run,[.1 .05;.1 .05]);
for iipt=1:npts2run
  ipt = pp.pts2run(iipt);
  ax = axs(1,iipt);
  
  pdmi = pdall.maxdensity_indep;
  vmagI = pdmi.vmag(1:end-2,ipt); % vmag(1,:) gives vel from t=1->t=2, which is used to predict x @ t=3
  dxmag_pred_tmp = pdmi.dx_pred_mag(3:end,ipt);
  szassert(vmagI,size(dxmag_pred_tmp));
  binctrs = 0:15;
  tfcell = arrayfun(@(x)round(vmagI)==x,binctrs,'uni',0);
  dxmag_pred_binmean = cellfun(@(x)nanmedian(dxmag_pred_tmp(x)),tfcell);
  
  scatter(ax,vmagI,dxmag_pred_tmp);
  hold(ax,'on');
  plot(ax,binctrs,dxmag_pred_binmean,'r','linewidth',2);
  ylabel(ax,sprintf('pt%d',ipt));
  
  ax = axs(2,iipt);
  vmagI = pdall.median.vmag(1:end-2,ipt);  
  dxmag_pred_tmp = pdall.median.dx_pred_mag(3:end,ipt);  
  szassert(vmagI,size(dxmag_pred_tmp));
  tfcell = arrayfun(@(x)round(vmagI)==x,binctrs,'uni',0);
  dxmag_pred_binmean = cellfun(@(x)nanmedian(dxmag_pred_tmp(x)),tfcell);
  scatter(ax,vmagI,dxmag_pred_tmp);
  hold(ax,'on');
  plot(ax,binctrs,dxmag_pred_binmean,'r','linewidth',2);
end
linkaxes(axs(:));
axis([0 15 0 15]);

%% dxmag_pred_mag vs vmag 2. Compare prederr vs second vel
hFig = figure(18);
clf
axs = mycreatesubplots(2,npts2run,[.1 .05;.1 .05]);
for iipt=1:npts2run
  ipt = pp.pts2run(iipt);
  ax = axs(1,iipt);
  
  pdmi = pdall.maxdensity_indep;
  vmagI = pdmi.vmag(2:end-1,ipt); % vmag(2,:) gives vel from t=2->3; compare this to prediction error at t=3
  dxmag_pred_tmp = pdmi.dx_pred_mag(3:end,ipt);
  szassert(vmagI,size(dxmag_pred_tmp));
  binctrs = 0:15;
  tfcell = arrayfun(@(x)round(vmagI)==x,binctrs,'uni',0);
  dxmag_pred_binmean = cellfun(@(x)nanmedian(dxmag_pred_tmp(x)),tfcell);
  
  scatter(ax,vmagI,dxmag_pred_tmp);
  hold(ax,'on');
  plot(ax,binctrs,dxmag_pred_binmean,'r','linewidth',2);
  ylabel(ax,sprintf('pt%d',ipt));
  
  ax = axs(2,iipt);
  vmagI = pdall.median.vmag(2:end-1,ipt);  
  dxmag_pred_tmp = pdall.median.dx_pred_mag(3:end,ipt);  
  szassert(vmagI,size(dxmag_pred_tmp));
  tfcell = arrayfun(@(x)round(vmagI)==x,binctrs,'uni',0);
  dxmag_pred_binmean = cellfun(@(x)nanmedian(dxmag_pred_tmp(x)),tfcell);
  scatter(ax,vmagI,dxmag_pred_tmp);
  hold(ax,'on');
  plot(ax,binctrs,dxmag_pred_binmean,'r','linewidth',2);
end
linkaxes(axs(:));
axis([0 15 0 15]);

%% Damping
% THIS IS SCREWY SUGGESTS MAYBE the motion model should be that the vel 
% predictor only works when there is motion above a certain thresh. Or do a
% soft ramp etc.

xprederrmag = pdall.maxdensity_indep.dx_preddamps_mag(:,pp.pts2run,:);

szassert(xprederrmag,[Nall npts2run nDamp]);
xprederrmagAccGTplot = reshape(xprederrmag,[Nall npts2run 1 1 nDamp]);

hFig = figure(25);
clf;
[hFig,hAxs] = GTPlot.ptileCurves(xprederrmagAccGTplot,...
 'hFig',hFig,...
 'setNames',numarr2trimcellstr(DAMPS),...
 'ptnames',numarr2trimcellstr(pp.pts2run),...
 'titleArgs',{'fontweight','bold'});

hFig = figure(26);
clf;
[hFig,hAxs] = GTPlot.ptileCurves(xprederrmagAccGTplot,...
 'hFig',hFig,...
 'setNames',numarr2trimcellstr(DAMPS),...
 'ptnames',numarr2trimcellstr(pp.pts2run),...
 'titleArgs',{'fontweight','bold'},...
 'ptiles',[15 30 45 60 75] ...
);

hFig = figure(27);
clf;
xprederrsqAccMu = nanmean(xprederrmag,1);
xprederrsqAccMu = reshape(xprederrsqAccMu,npts2run,nDamp)';
xprederrsqAccMdn = nanmedian(xprederrmag,1);
xprederrsqAccMdn = reshape(xprederrsqAccMdn,npts2run,nDamp)';

h = plot(DAMPS(:),xprederrsqAccMu);
hold on;
plot(DAMPS(:),5*xprederrsqAccMdn,'.-');
title('prederreq vs damping','fontweight','bold');
grid on;
yl = ylim;
yl(1) = 0;
ylim(yl);

%% Damping, looking only at bigger velocities
% SEEMS TO CONCUR

pdmi = pdall.maxdensity_indep;
VMAGMINOKTHRESH = 1;

for iipt=1:npts2run
  ipt = pts2run(iipt);

  xprederrmag = squeeze(pdmi.dx_preddamps_mag(:,ipt,:)); % Nall x nDamp
  vmag = pdmi.vmag(:,ipt);
  % vmag(1) and vmag(2) are used to generate xpred(3)
  tfpredok = false(size(vmag));
  tfpredok(3:end) = vmag(1:end-2)>VMAGMINOKTHRESH & vmag(2:end-1)>VMAGMINOKTHRESH;

  xprederrplot = xprederrmag(tfpredok,:);
  xprederrplot = reshape(xprederrplot,[nnz(tfpredok) 1 1 1 nDamp]);
  fprintf(1,'iipt=%d,ipt=%d. %d/%d rows meet velocity requirement.\n',...
    iipt,ipt,nnz(tfpredok),numel(tfpredok));

  hFig = figure(29+iipt);
  clf;
  [hFig,hAxs] = GTPlot.ptileCurves(xprederrplot,...
   'hFig',hFig,...
   'setNames',numarr2trimcellstr(DAMPS),...
   'ptnames',numarr2trimcellstr(ipt),...
   'titleArgs',{'fontweight','bold'},...
   'ptiles',[15 30 45 60 75],...
    'createsubplotsborders',[.12 0;.12 .12]...
   );
end
%% Damping + GT


% tGT = Labeler.lblFileGetLabels(ld);
% assert(strcmp(tGT.Properties.VariableNames{1},'mov'));
lbl = ld;
[tf,loc] = tblismember(tGT,tMFTall,MFTable.FLDSID);
%frmsGT = tGT.frm(tf);
isampGT = loc(tf);
nGT = nnz(tf);
npts = lbl.cfg.NumLabelPoints;
lposGT = reshape(tGT.p(tf,:),[nGT npts 2]);
fprintf(1,'%d frmsGT.\n',nGT);

tpos = pdmi.x(isampGT,:,:);
tpospred = pdmi.x_preddamps(isampGT,:,:,:);
szassert(tpos,[nGT npts 2]);
szassert(tpospred,[nGT npts 2 nDamp]);

dxyGT = lposGT-tpos;
dxyGTmag = sqrt(sum(dxyGT.^2,3)); % [nGT npts]
dxyGTpred = lposGT-tpospred;
dxyGTpredmag = sqrt(sum(dxyGTpred.^2,3)); % [nGT npts 1 nDamp]

errmagplot = cat(5,...
  reshape(dxyGTpredmag(:,pts2run,:,:),[nGT npts2run 1 1 nDamp]),...
  dxyGTmag(:,pts2run) );
setNames = [arrayfun(@(x)sprintf('dmp%02d'),1:nDamp,'uni',0) {'noMM'}];

[hFig,hAxs] = GTPlot.ptileCurves(errmagplot,...
  'ptiles',[50 75 90],...
  'setNames',setNames,...
  'ptNames',arrayfun(@(x)sprintf('pt%02d',x),pts2run,'uni',0),...
  'createsubplotsborders',[.05 0;.15 .15]...
  );


errxyplot = dxyGTpred(:,pts2run,:,:);
errxyplot = reshape(errxyplot,[nGT npts2run 2 1 nDamp]);
DAMPS2RUN = 1:4:nDamp;
errxyplot = errxyplot(:,:,:,:,DAMPS2RUN);
setNames = arrayfun(@(x)sprintf('dmp%02d',x),DAMPS2RUN,'uni',0);
im = zeros(60,40);
I = {im};
xyLbl = [25 45;25 30;25 15;15 15;15 30;15 45];
% SEE Also GT+Damp below
[hFig,hAxs] = GTPlot.bullseyePtiles(errxyplot,I,xyLbl,...
  'ptiles',[50 75 90],...
  'setNames',setNames,...
  'lineWidth',2,...
  'contourtype','ellipse');


%% autocorr

MAXLAG = 20;
nLag = MAXLAG+1;

pdmi = pdall.maxdensity_indep;
assert(max(pdmi.ires)==numel(res));

xc = []; % [nlag x npts2run x {x,y} x nres]
vlag1 = []; % [n x npts2run x {x,y} x {t=i,t=i+1}]
for ires=1:numel(res)
  tfres = pdmi.ires==ires;
  fprintf(1,'ires=%d, %d rows\n',ires,nnz(tfres));
  
  v = pdmi.v(tfres,pts2run,:);
  vgood = v(1:end-1,:,:); % clip last nan
  cxkeep = nan(nLag,npts2run);
  cykeep = nan(nLag,npts2run);
%   cxkeeprestricted = nan(nLag,npts2run);
%   cykeeprestricted = nan(nLag,npts2run);
  for iipt=1:npts2run
    vx = vgood(:,iipt,1);
    vy = vgood(:,iipt,2);
    cx = xcorr(vx,vx);    
    cy = xcorr(vy,vy);

    n = numel(vx);
    cxkeep(:,iipt) = cx(n:n+MAXLAG,:);
    cykeep(:,iipt) = cy(n:n+MAXLAG,:);
  end
  
  xc = cat(4,xc,cat(3,cxkeep,cykeep));
  
  vlag1 = cat(1,vlag1,cat(4,vgood(1:end-1,:,:),vgood(2:end,:,:)));
end

%%

xcsum = sum(xc,4); % sum over all intervals. each sample equally weighted
nlag = size(xcsum,1);

figure(31);
clf;
axs = mycreatesubplots(npts2run,1,[0.12 0.05;0.12 0.05]);
x = 0:nlag-1;
for iipt=1:npts2run
  ax = axs(iipt,1);
  axes(ax);
  
  plot(x,xcsum(:,iipt,1)/xcsum(1,iipt,1),'bo-','linewidth',2);
  hold on;
  plot(x,xcsum(:,iipt,2)/xcsum(1,iipt,2),'rx-','linewidth',2);
  grid on;
  plot(x,zeros(size(x)),'k-');
end

    
%% Vlag1 scatter

V_1D_MINOKTHRESHS = 0:2:6;
JITTERSZ = 0.4;
nThresh = numel(V_1D_MINOKTHRESHS);


hFig = figure(32);
set(hFig,'name','vlag1 X');
clf
axsX = mycreatesubplots(nThresh,npts2run,[0.12 0.05;0.12 0.05]);

hFig = figure(33);
set(hFig,'name','vlag1 Y');
clf
axsY = mycreatesubplots(nThresh,npts2run,[0.12 0.05;0.12 0.05]);

axs = cat(3,axsX,axsY); % [nThresh x npts2run x {x,y}]

for iThresh=1:nThresh
  thresh = V_1D_MINOKTHRESHS(iThresh);
  tfok = all(abs(vlag1)>=thresh,4);
  
  for iipt=1:npts2run
    ipt = pts2run(iipt);
    
    assert(size(vlag1,2)==npts2run);
    vlag1IPTokx = vlag1(tfok(:,iipt,1),iipt,1,:);
    vlag1IPToky = vlag1(tfok(:,iipt,2),iipt,2,:);
    vlag1OKs = {vlag1IPTokx vlag1IPToky};
    
    nokx = size(vlag1IPTokx,1);
    noky = size(vlag1IPToky,1);
    fprintf(1,'thresh=%f, ipt=%d, nokx/noky=%d/%d.\n',thresh,ipt,nokx,noky);
    nOKs = {nokx noky};
    % for xy=1:2
    %   ax = axs(xy);
    %   axes(ax);
    %   xscat = vlag1(:,IIPT,xy,1);
    %   yscat = vlag1(:,IIPT,xy,2);
    %   p1 = polyfit(xscat,yscat,1);
    %   p3 = polyfit(xscat,yscat,3);
    %   [r p] = corrcoef(xscat,yscat);
    %   xbins = -30:30;
    %   ybins = arrayfun(@(x)median(yscat(round(xscat)==x)),xbins);
    %   xscat = xscat+JITTERSZ*2*(rand(size(xscat))-0.5);
    %   yscat = yscat+JITTERSZ*2*(rand(size(yscat))-0.5);
    %   plot(xscat,yscat,'.');
    %   tstr = sprintf('linear: slope=%.3g r=%.3g p=%.3g',p1(1),r(1,2),p(1,2));
    %   hold on;
    %   x = -30:1:30;
    %   y1 = p1(1)*x+p(2);
    %   y3 = p3(1)*x.^3 + p3(2)*x.^2 + p3(3)*x.^1 + p3(4);
    %   plot(x,y1,'r-',x,y3,'r-','linewidth',2);
    %   plot(xbins,ybins,'rs','markerfacecolor',[1 0 0]);
    %   title(tstr);
    %   grid on
    %   axis([-30 30 -30 30]);
    % end
    % linkaxes(axs);

    for xy=1:2
      ax = axs(iThresh,iipt,xy);
      axes(ax);
      
      vlagmat = vlag1OKs{xy};
      nok = nOKs{xy};
      
      xscat = vlagmat(:,1,1,1);
      yscat = vlagmat(:,1,1,2);
      p1 = polyfit(xscat,yscat,1);
      p3 = polyfit(xscat,yscat,3);
      [r p] = corrcoef(xscat,yscat);
      xbins = -30:30;
      ybins = arrayfun(@(x)median(yscat(round(xscat)==x)),xbins);
      xscat = xscat+JITTERSZ*2*(rand(size(xscat))-0.5);
      yscat = yscat+JITTERSZ*2*(rand(size(yscat))-0.5);
      plot(xscat,yscat,'.');
      tstr = sprintf('nOK=%d. slope=%.2g r=%.2g p=%.2g',nok,p1(1),r(1,2),p(1,2));
      hold on;
      x = -30:1:30;
      y1 = p1(1)*x+p(2);
      y3 = p3(1)*x.^3 + p3(2)*x.^2 + p3(3)*x.^1 + p3(4);
      plot(x,y1,'r-',x,y3,'r-','linewidth',2);
      plot(xbins,ybins,'rs','markerfacecolor',[1 0 0]);
      title(tstr);
      grid on
      axis([-30 30 -30 30]);
    end
  end
end
%linkaxes(axs);

%% Viterbigrid xAC
acmid = pp.viterbi_grid_acradius+1;

hFig = figure(36);
axs = mycreatesubplots(npts2run,2);
for iipt=1:npts2run
  ipt = pts2run(iipt);
  for xy=1:2
    z = pdall.viterbi_grid.xAC(:,ipt,xy);
    ax = axs(iipt,xy);
    axes(ax);
    
    histogram(z);
    
    nmidfar = nnz(z<acmid-1 | z>acmid+1);
    nmidpm1 = nnz(z==acmid-1 | z==acmid+1);
    nmid = nnz(z==acmid);
    fprintf(1,'ipt %d, xy=%d. %.1f %% far from mid, %.1f mid +/-1, %.1f mid.\n',...
      ipt,xy,nmidfar/numel(z)*100,nmidpm1/numel(z)*100,nmid/numel(z)*100);
  end
end
    
    
  