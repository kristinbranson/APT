%%
LBL = 'f:\aptMultiTargetAssignPx20171016\multitarget_bubble.lbl';
lbl = load(LBL,'-mat');
mfa = lbl.movieFilesAll;
lbl.projMacros.rootdatadir = 'f:\pathmacros20170731\localdata';
mfaf = FSPath.fullyLocalizeStandardize(mfa,lbl.projMacros);
tfaf = Labeler.trxFilesLocalize(lbl.trxFilesAll,mfaf);

BGDATAFILE = 'registrationdata.mat';

prm.ftr.rad = 5; % px. feature radius 
prm.ftr.nFern = 20;
prm.ftr.fernD = 6;
prm.ftr.regN = 1; % regularization count

NLEG = 6;
IPT2LEG = [0 0 0 0 0 0 0 0 0 0 0 1 2 3 4 5 6];
TITLEARGS = {'fontweight','bold','interpreter','none'};

%% Smoothing experiment
SZS = [0 3 5 7];
figure;
axs = createsubplots(1,numel(SZS));
for iSz=1:numel(SZS)
  sz = SZS(iSz);
  sig = sz/2/3;
  
  if sz==0
    imSmth = imFore;
  else
    h = fspecial('gaussian',sz,sig)
    imSmth = imfilter(imFore,h,'replicate');
  end
  
  ax = axs(iSz);
  axes(ax);
  imagesc(imSmth);
  colormap gray;
  axis xy equal image;
  ax.XTickLabel = [];
  ax.YTickLabel = [];
  title(sprintf('sz/sig=%d/%d',sz,sig),'fontweight','bold');
end

linkprop(axs,'CLim');
  

%% Training data and PP
% - overall bg
% - MFT, im (or imroi), trx
% - bgsub to get imfore(roi)
% - thresh to get imforebw(roi)
% - canonrot/crop to get training images imforecanonroi
% - assign fg px to a leg, either ii) perp dist, and/or mask ell and use cc
% relative to label

% end result, MFT with
%  * mov, frm, iTgt, pTrx, trxa, trxb, trxth
%  * imForeCanon: [canonnr canonnc] single, bgsubed foreground px, target
%  canonically oriented. Roi has rad prm.ppRad.
%   * imForeCanonBWLleg: [canonnr canonnc] uint8, label vector labeling
%   legs 1-6

prm = struct();
prm.pp.rad = 100; % px. roi radius in preprocessed images
prm.pp.fltrOn = true;
prm.pp.fltrSz = 5;
prm.pp.fltrSig = (prm.pp.fltrSz-1)/2/2;
prm.pp.fgBwThresh = 9; 
prm.pp.ellSzFacBWLleg = 1.35; % expand ell by this much

lpos = lbl.labeledpos;
nMov = numel(mfaf);

if prm.pp.fltrOn
  hFltr = fspecial('gaussian',prm.pp.fltrSz,prm.pp.fltrSig);
else
  hFltr = [];
end

s = struct(...
  'mov',cell(0,1),'frm',[],'iTgt',[],...
  'p',[],'pTrx',[],'trxa',[],'trxb',[],'trxth',[],...
  'pcanon',[],'idxLposLegCanon',[],...
  'imForeCanon',[],'imForeCanonBWMasked',[],'imForeCanonBWLleg',[]...
  );
for iMov=1:nMov
  mov = mfaf{iMov};
  mr = MovieReader;
  mr.open(mfaf{iMov});
  mr.forceGrayscale = true;
  fprintf('Opened movie %s...\n',mov);
  imnc = mr.info.nc;
  imnr = mr.info.nr;

  bgpath = fileparts(mov);
  bgfile = fullfile(bgpath,BGDATAFILE);
  bg = load(bgfile,'-mat'); 
  bg = bg.bkgdImage'; % xpose
  fprintf('Loaded bg file %s...\n',bgfile);

  trx = load(tfaf{iMov});
  trx = trx.trx;
  fprintf('Loaded trx %s...\n',tfaf{iMov});
  
  lposI = lpos{iMov};
  
  [~,~,nfrm,ntgt] = size(lposI);
  assert(ntgt==numel(trx));
  assert(mr.nframes==nfrm);
  for f=1:nfrm
    if mod(f,1000)==0
      disp(f);
    end
    for itgt=1:ntgt
      lposIFT = lposI(:,:,f,itgt);
      if any(~isnan(lposIFT))
        imF = mr.readframe(f);

        s(end+1,1).mov = iMov;
        s(end).frm = f;
        s(end).iTgt = itgt;
        s(end).p = lposIFT(:)';
        
        trxI = trx(itgt);
        idx = trxI.off+f;
        s(end).pTrx = [trxI.x(idx) trxI.y(idx)];
        s(end).trxa = trxI.a(idx);
        s(end).trxb  = trxI.b(idx);
        s(end).trxth = trxI.theta(idx);
        
        trxx = s(end).pTrx(1);
        trxy = s(end).pTrx(2);
        
        % create an affine xform that puts/aligns the target at the origin 
        th = -s(end).trxth;
        cost = cos(th);
        sint = sin(th);
        T = [cost -sint -cost*trxx+sint*trxy; ...
             sint cost -sint*trxx-cost*trxy; ...
             0 0 1]';
        at = affine2d(T); % rotates by -trxtheta, and puts (trxx,trxy) at origin
                
        assert(isa(bg,'double'));
        imFore = single(bg)-single(imF);

        if prm.pp.fltrOn
          imFore = imfilter(imFore,hFltr,'replicate');
        end
        
        [xgcanon,ygcanon] = meshgrid(-prm.pp.rad:prm.pp.rad,-prm.pp.rad:prm.pp.rad);
        [xg,yg] = at.transformPointsInverse(xgcanon,ygcanon);
        imForeCanon = interp2(1:imnc,1:imnr,imFore,xg,yg,'linear',0);

        xylpos = Shape.vec2xy(s(end).p);
        [xyLposCanon(:,1),xyLposCanon(:,2)] = ...
          at.transformPointsForward(xylpos(:,1),xylpos(:,2));
        colLposCanon = round(xyLposCanon(:,1))+prm.pp.rad+1; % -rad -> 1; rad -> 2*rad+1
        rowLposCanon = round(xyLposCanon(:,2))+prm.pp.rad+1; % etc
        [nrcanon,nccanon] = size(xgcanon);
        
        assert(all(1<=colLposCanon & colLposCanon<=nccanon));
        assert(all(1<=rowLposCanon & rowLposCanon<=nrcanon));
        idxLposCanon = rowLposCanon + (colLposCanon-1)*nrcanon;
        idxLposLegCanon = idxLposCanon(12:17,:);
        szassert(idxLposLegCanon,[NLEG 1]);
        
        % Compute imForeCanonBWLleg. Threshold, mask ellipse, run cc and
        % look check label pts.
        imForeCanonBW = abs(imForeCanon)>=prm.pp.fgBwThresh;
        imEll = drawellipseim(0,0,0,2*s(end).trxa*prm.pp.ellSzFacBWLleg,...
          2*s(end).trxb*prm.pp.ellSzFacBWLleg,xgcanon,ygcanon,1);
        imForeCanonBWMasked = imForeCanonBW;
        imForeCanonBWMasked(imEll>0) = 0;
                
        % identify legs
        bwlLegs = zeros(size(imForeCanonBWMasked));
        cc = bwconncomp(imForeCanonBWMasked);
        for iCC=1:cc.NumObjects
          pxIdxCC = cc.PixelIdxList{iCC};
          tf = ismember(idxLposLegCanon,pxIdxCC);
          legsInCC = find(tf);
          if isscalar(legsInCC) 
            bwlLegs(pxIdxCC) = legsInCC;
          elseif isequal([3;4],sort(legsInCC))
            bwlLegs(pxIdxCC) = 3;
          end
        end
        
        bwlLegs(bwlLegs==4) = 3; % currently lump legs 3/4 together
        
        pcanon = Shape.xy2vec(xyLposCanon);
        s(end).pcanon = pcanon;
        s(end).idxLposLegCanon = idxLposLegCanon;
        s(end).imForeCanon = imForeCanon;
        s(end).imForeCanonBWMasked = imForeCanonBWMasked;
        s(end).imForeCanonBWLleg = uint8(bwlLegs);
      end
    end
  end
end
t = struct2table(s);

%% Histogram leg numpx sizes
nGT = height(t);
npxPerLeg = nan(nGT,NLEG);
for iGT=1:nGT
  imfcBWLleg = t.imForeCanonBWLleg{iGT};
  for iLeg=1:NLEG
    npxPerLeg(iGT,iLeg) = nnz(imfcBWLleg==iLeg);
  end
end

figure;
axs = createsubplots(2,3);
axs = reshape(axs,2,3);
LEGS = [1 6 2 5 3 4];
for i=1:6
  iLeg = LEGS(i);
  ax = axs(iLeg);
  axes(ax);
  histogram(npxPerLeg(:,iLeg));
  grid on;
  title(num2str(iLeg),'fontweight','bold','interpreter','none');
end
linkaxes(axs(:,1));
linkaxes(axs(:,3));

npxPerLegPtiles = prctile(npxPerLeg,[90 95:99 99.9])'

%% keep only under Xth percentile. UPDATES TABLE
KEEPUPTOPTILE = 95;

nGT = height(t);
iGTdisc = zeros(0,2); % col1: iGT to discard. col2: leg that caused it
for iLeg=1:NLEG
  npxILegX = prctile(npxPerLeg(:,iLeg),KEEPUPTOPTILE);
  for iGT=1:nGT
    imfcBWLleg = t.imForeCanonBWLleg{iGT};
    tfleg = imfcBWLleg==iLeg;
    npx = nnz(tfleg);
    if npx>npxILegX
      t.imForeCanonBWLleg{iGT}(tfleg) = 0;      
      iGTdisc(end+1,:) = [iGT iLeg];
    end
  end
end

%% Save training data
fname = sprintf('trndata.%s.mat',datestr(now,'yyyymmddTHHMMSS'));
prmPP = prm.pp;
save(fname,'t','npxPerLeg','KEEPUPTOPTILE','iGTdisc','prmPP');
fprintf('Saved %s.\n',fname);

%% Browse training data
ellfac = prm.pp.ellSzFacBWLleg;
figure;
axs = createsubplots(2,2);
axs = reshape(axs,2,2);
i = 200;
roictr = prm.pp.rad+1;

ax = axs(1);
axes(ax);
imagesc(s(i).imForeCanon);
hold on;
hEll = drawellipse(roictr,roictr,0,2*s(i).trxa*ellfac,2*s(i).trxb*ellfac);
set(hEll,'LineWidth',2,'Color',[1 0 0]);
plot(s(i).pcanon(1:17)+roictr,s(i).pcanon(18:34)+roictr,'.w','markersize',12);
axis xy square equal
hCB = colorbar('east');
hCB.Color = [1 1 1];

axes(axs(1,2));
imagesc(s(i).imForeCanonBWMasked);
hold on;
hEll = drawellipse(roictr,roictr,0,2*s(i).trxa*ellfac,2*s(i).trxb*ellfac);
set(hEll,'LineWidth',2,'Color',[1 0 0]);
plot(s(i).pcanon(1:17)+roictr,s(i).pcanon(18:34)+roictr,'.r','markersize',12);
axis xy square equal

axes(axs(2,1));
imagesc(s(i).imForeCanonBWLleg);
axis xy square equal
hCB = colorbar('east');
hCB.Color = [1 1 1];

linkaxes(axs);
axis(ax(1),[0 2*roictr 0 2*roictr]);

%%
fimidxs = pxfern.genfeatures(prm);

%% Viz Ftr idxs/locs
nFern = prm.ftr.nFern;
fernD = prm.ftr.fernD;
ftrRad = prm.ftr.rad;
ftrBoxSz = 2*ftrRad+1;
nptsFtrSqrROI = ftrBoxSz^2;
for iFern=1:nFern
  figure;
  axs = createsubplots(1,fernD);  
  for iFtr=1:fernD    
    axes(axs(iFtr));  
    imftr = zeros(ftrBoxSz,ftrBoxSz);
    imftr(fimidxs(iFern,iFtr,:)) = 1;
    imagesc(imftr);
    axis xy square equal
    axis([0 ftrBoxSz+1 0 ftrBoxSz+1]);
  end  
end

%%
[fvals,fvalsc] = pxfern.compfeatures(prm,t,fimidxs);

%% Viz ftrsBig
PTILE = 40;
fprintf('Feature hist has size: %s\n',mat2str(size(fvals)));
fprintf('Our ptile is %d\n',PTILE);

figure;
axs = createsubplots(nFern,fernD);
axs = reshape(axs,nFern,fernD);
fvalsPtlThresh = nan(nFern,fernD);
for iFern=1:nFern
  for iD = 1:fernD
    x = fvals(iFern,iD,:);
    x = x(:);
    
    ax = axs(iFern,iD);
    axes(ax);
    histogram(x,100);
    grid on;
    
    ptile = prctile(x,PTILE);
    fvalsPtlThresh(iFern,iD) = ptile;
    hold on;
    yl = ax.YLim;
    plot([ptile ptile],yl,'r-');    
  end
end

% We are interested in the "low end", where the foreground value at the
% ftrlocation is comparable or larger than at the center. Eg negative
% values of ftrValsBig will be more rare.
fprintf('%dth percentile of feature hists:\n',PTILE);
fvalsPtlThresh

%%
hfern = pxfern.fernhist(prm,fvals,fvalsc);
[pfern,hfernN,hfernZ] = pxfern.fernprob(prm,hfern);

%% Viz normalized hists (max likelihood PDFs, no regularization) AND
% pFern
figure('Name','Raw PDFs');
axs = createsubplots(nFern/2,2);
for iFern=1:nFern
  ax = axs(iFern);
  axes(ax);
  x = hfernN(:,:,iFern)'; % plot ferncountdist for legs 1-6 (cols)
  plot(x,'LineWidth',2);
  grid on;
  if iFern==1
    legend(arrayfun(@num2str,1:NLEG,'uni',0));
  else
    ax.XTickLabel = [];
    ax.YTickLabel = [];
  end
end
linkaxes(axs,'y');

figure('Name','Regularized PDFs');
axs = createsubplots(nFern/2,2);
for iFern=1:nFern
  ax = axs(iFern);
  axes(ax);
  x = pfern(:,:,iFern)'; % plot ferncountdist for legs 1-6 (cols)
  plot(x,'LineWidth',2);
  grid on;
  if iFern==1
    legend(arrayfun(@num2str,1:NLEG,'uni',0));
  else
    ax.XTickLabel = [];
    ax.YTickLabel = [];
  end
end
linkaxes(axs,'y');

%%
fvalscpred = pxfern.fernpred(prm,pfern,fvals);

%% Assess
c = confusionmat(fvalsc,fvalscpred,'order',1:6)
cZ = sum(c,2);
cnorm = c./cZ;
corrpcts = diag(cnorm);

%%
load trndata.20171030T095840.mat;
%%
CVNAME = 'ppFltr3Test';
prm = struct();
prm.ftr.rad = 5; % px. feature radius 
prm.ftr.nFern = 40;
prm.ftr.fernD = 6;
prm.ftr.regN = 1;

[cvpart,C,corrPct,xvres] = pxfern.xv(prm,t,5);
corrPctBig = cat(2,corrPct{:});
round(corrPctBig*100)

xvfname = sprintf('cv.%s.%s.mat',CVNAME,datestr(now,'yyyymmddTHHMMSS'));
save(xvfname,'prm','cvpart','C','corrPct');
fprintf('Saved %s.\n',xvfname);

%%



% figure;
% axs = createsubplots(NLEG,1);
% for iLeg=1:NLEG
%   ax = axs(iLeg);
%   axes(axs(iLeg));
%   plot(cnorm(iLeg,:));
%   grid on;
%   ystr = sprintf('corrPct=%.2f,n=%d',cnorm(iLeg,iLeg),cZ(iLeg));
%   ylabel(ystr,'fontwcveight','bold');
%   if iLeg~=1
%     ax.YTickLabel = [];
%   end
%   if iLeg~=6
%     ax.XTickLabel = [];
%   end    
% end
  
figure;
imagesc(cnorm);
colorbar
title('Fern predictions confusion mat','fontweight','bold');
ylabel('actual','fontweight','bold');
xlabel('pred','fontweight','bold');

fprintf('corr prediction pcts:\n');
disp(corrpcts)

%% 
load trndata.20171030T073229.mat;


%% Assess XV res by frame
n = height(t);
bwlLegPredAcc = cell(n,1);
npxAcc = nan(n,1);
npxCorrAcc = nan(n,1);
for i=1:n  
  trow = t(i,:);
  tfXVmft = xvres.mov==trow.mov & xvres.frm==trow.frm & xvres.iTgt==trow.iTgt;
  xvresMft = xvres(tfXVmft,:);
  
  bwlLeg = trow.imForeCanonBWLleg{1};
  bwlLegPred = zeros(size(bwlLeg));
  for ixv=1:height(xvresMft)
    xvresrow = xvresMft(ixv,:);
    bwlLegPred(xvresrow.i,xvresrow.j) = xvresrow.legpred;
    assert(bwlLeg(xvresrow.i,xvresrow.j)==xvresrow.leg);
  end
  
  npxAcc(i) = height(xvresMft);
  npxCorrAcc(i) = nnz(xvresMft.legpred==xvresMft.leg);
  bwlLegPredAcc{i} = bwlLegPred;
  
  if mod(i,100)==0
    disp(i);
  end
end

%%
t = [t table(npxAcc,npxCorrAcc,npxCorrAcc./npxAcc,bwlLegPredAcc,...
  'VariableNames',{'npxPred' 'npxPredCorr' 'pctPredCorr' 'imForeCanonBWLlegpred'})];
[~,idx] = sort(t.pctPredCorr,'descend');
t = t(idx,:);
  


%% Assess XV res with training data
ellfac = prm.pp.ellSzFacBWLleg;
roictr = prm.pp.rad+1;
  
figure;
axs = createsubplots(2,3);
axs = reshape(axs,2,3);

for i=200:200:height(t)
  ax = axs(1,1);
  axes(ax);
  cla
  imagesc(t.imForeCanon{i});
  hold on;
  hEll = drawellipse(roictr,roictr,0,2*t.trxa(i)*ellfac,2*t.trxb(i)*ellfac);
  set(hEll,'LineWidth',2,'Color',[1 0 0]);
  plot(t.pcanon(i,1:17)+roictr,t.pcanon(i,18:34)+roictr,'.w','markersize',12);
  axis xy square equal
  hCB = colorbar('east');
  hCB.Color = [1 1 1];

  axes(axs(2,1));
  cla
  imagesc(t.imForeCanonBWMasked{i});
  hold on;
  hEll = drawellipse(roictr,roictr,0,2*t.trxa(i)*ellfac,2*t.trxb(i)*ellfac);
  set(hEll,'LineWidth',2,'Color',[1 0 0]);
  plot(t.pcanon(i,1:17)+roictr,t.pcanon(i,18:34)+roictr,'.r','markersize',12);
  axis xy square equal

  axes(axs(1,2));
  cla
  imagesc(t.imForeCanonBWLleg{i});
  axis xy square equal
  hCB = colorbar('east');
  hCB.Color = [1 1 1];

  axes(axs(2,2));
  cla
  imagesc(t.imForeCanonBWLlegpred{i});
  axis xy square equal
  hCB = colorbar('east');
  hCB.Color = [1 1 1];
  tstr = sprintf('pred: %d/%d=%d%%',t.npxPredCorr(i),t.npxPred(i),...
    round(t.pctPredCorr(i)*100));
  title(tstr,'interpreter','none','fontweight','bold');
  linkprop(axs(:,2),'CLim');

  linkaxes(axs);
  axis(ax(1),[0 2*roictr 0 2*roictr]);
  
  input(num2str(i));
end