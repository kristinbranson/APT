%%
LBL = 'f:\aptMultiTargetAssignPx20171016\multitarget_bubble.lbl';
lbl = load(LBL,'-mat');
mfa = lbl.movieFilesAll;
lbl.projMacros.rootdatadir = 'f:\pathmacros20170731\localdata';
mfaf = FSPath.fullyLocalizeStandardize(mfa,lbl.projMacros);
tfaf = Labeler.trxFilesLocalize(lbl.trxFilesAll,mfaf);

BGDATAFILE = 'registrationdata.mat';
ROIRAD = 150;

TITLEARGS = {'fontweight','bold','interpreter','none'};

%%
lpos = lbl.labeledpos;
nMov = numel(mfaf);
s = struct(...
  'mov',cell(0,1),'frm',[],'iTgt',[],...
  'p',[],'pTrx',[],...
  'thetaTrx',[],'aTrx',[],'bTrx',[],...
  'roi',[],'imroi',[],'imbgroi',[],'imfore',[]...
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
        s(end).thetaTrx = trxI.theta(idx);
        s(end).aTrx = trxI.a(idx);
        s(end).bTrx = trxI.b(idx);
        
        trxx = s(end).pTrx(1);
        trxy = s(end).pTrx(2);
        roi = round([trxx-ROIRAD trxx+ROIRAD trxy-ROIRAD trxy+ROIRAD]);
        s(end).roi = roi;
        s(end).imroi = padgrab(imF,intmax(class(imF)),...
          roi(3),roi(4),roi(1),roi(2));
        assert(isa(bg,'double'));
        s(end).imbgroi = single(padgrab(bg,1,...
          roi(3),roi(4),roi(1),roi(2)));
        s(end).imfore = single(s(end).imbgroi)-single(s(end).imroi);
      end
    end    
  end
end
t = struct2table(s);

%% Probe aTrx and bTrx
figure
axs = createsubplots(1,2);

ax = axs(1);
axes(ax);
hist(t.aTrx,50);
xlabel(ax,'a','fontweight','bold');

ax = axs(2);
axes(ax);
hist(t.bTrx,50);
xlabel(ax,'b','fontweight','bold');

amu = mean(t.aTrx);
bmu = mean(t.bTrx);
fprintf(1,'Mean a and b (n=%d) are: %.2f, %.2f\n',height(t),amu,bmu);

%% Align and center labeled shapes
pLbl = t.p;
pTrx = t.pTrx;
pLblTrx = [pLbl(:,1:17) pTrx(:,1) pLbl(:,18:34) pTrx(:,2)]; % 18th pt is pTrx
[pLblTrxA,thAlgn] = Shape.alignOrientationsOrigin(pLblTrx,1,7); % aligned based on body pts, now with arbitrary offset b/c was rotated about origin
n = height(t);

% shapes are now aligned but have arbitrary/random offsets. center them

pAcom = nan(size(pLblTrxA));
pAcombody = nan(size(pLblTrxA));
pAtrx = nan(size(pLblTrxA));
for i=1:n
  xy = Shape.vec2xy(pLblTrxA(i,:));
  xyCOM = bsxfun(@minus,xy,mean(xy));
  xyCOMbody = bsxfun(@minus,xy,mean(xy(1:7,:)));
  xyTrx = bsxfun(@minus,xy,xy(18,:));
  assert(size(xy,1)==18);
  pAcom(i,:) = Shape.xy2vec(xyCOM);
  pAcombody(i,:) = Shape.xy2vec(xyCOMbody);
  pAtrx(i,:) = Shape.xy2vec(xyTrx);  
end

%% Standardize pAtrx based on size

% pAtrx is aligned along x-axis, origin is pTrx.

trxa = t.aTrx;
trxb = t.bTrx;
n = size(pAtrx,1);
szassert(trxa,[n 1]);
szassert(trxb,[n 1]);

xfac = trxa/amu; % scalefactor along x. xfac>1 => fly was longer than av
yfac = trxb/bmu; % scalefactor along y

pAtrxnorm = nan(size(pAtrx));
for i=1:n
  xy = Shape.vec2xy(pAtrx(i,:));
  xy(:,1) = xy(:,1)/xfac(i);
  xy(:,2) = xy(:,2)/yfac(i);
  pAtrxnorm(i,:) = Shape.xy2vec(xy);
end

%% Aligned image superpose

% for each imfore
% i) canonically align it
% ii) optionally, scale x/y per trxa/b
% iii) add up a big hist

FGTHRESH = 6;

xroictr = -ROIRAD:ROIRAD;
yroictr = -ROIRAD:ROIRAD;
[xg,yg] = meshgrid(xroictr,yroictr);
imforeAlgn = zeros(size(xg));
imforeBWAlgn = zeros(size(xg));
imforeAlgnNorm = zeros(size(xg));
imforeBWAlgnNorm = zeros(size(xg));
for ishape=1:n  
  if mod(ishape,500)==0
    disp(ishape);
  end
  imfore = t.imfore{ishape};
  imfore = max(imfore,0);
  imforebw = double(imfore>=FGTHRESH);
  th = t.thetaTrx(ishape);
  imforecanon = readpdf(imfore,xg,yg,xg,yg,0,0,-th);
  imforebwcanon = readpdf(imforebw,xg,yg,xg,yg,0,0,-th);
  a = t.aTrx(ishape);
  b = t.bTrx(ishape);
  xfac = a/amu;
  yfac = b/bmu;
  imforecanonscale = interp2(xg,yg,imforecanon,xg*xfac,yg*yfac,'linear',0);
  imforebwcanonscale = interp2(xg,yg,imforebwcanon,xg*xfac,yg*yfac,'linear',0);
  
  imforeAlgn = imforeAlgn + imforecanon/sum(imforecanon(:));
  imforeAlgnNorm = imforeAlgnNorm + imforecanonscale/sum(imforecanonscale(:));
  imforeBWAlgn = imforeBWAlgn + imforebwcanon/sum(imforebwcanon(:));
  imforeBWAlgnNorm = imforeBWAlgnNorm + imforebwcanonscale/sum(imforebwcanonscale(:));
end

imforeAlgn = imforeAlgn/sum(imforeAlgn(:));
imforeAlgnNorm = imforeAlgnNorm/sum(imforeAlgnNorm(:));
imforeBWAlgn = imforeBWAlgn/sum(imforeBWAlgn(:));
imforeBWAlgnNorm = imforeBWAlgnNorm/sum(imforeBWAlgnNorm(:));

figure;
axs = createsubplots(2,2);
axs = reshape(axs,2,2);
axes(axs(1,1));
imagesc(imforeAlgn);
hold on
hEll = ellipsedraw(2*amu,2*bmu,ROIRAD+1,ROIRAD+1,0,'r-');
title('Superposed flies',TITLEARGS{:});
axis image
axes(axs(1,2));
imagesc(imforeAlgnNorm);
hold on;
hEll = ellipsedraw(2*amu,2*bmu,ROIRAD+1,ROIRAD+1,0,'r-');
title('Superposed flies, normalized',TITLEARGS{:});
axis image
axes(axs(2,1));
imagesc(imforeBWAlgn);
hold on;
hEll = ellipsedraw(2*amu,2*bmu,ROIRAD+1,ROIRAD+1,0,'r-');
title('Superposed bw flies',TITLEARGS{:});
axis image
axes(axs(2,2));
imagesc(imforeBWAlgnNorm);
hold on;
hEll = ellipsedraw(2*amu,2*bmu,ROIRAD+1,ROIRAD+1,0,'r-');
title('Superposed bw flies, normalized',TITLEARGS{:});
axis image

linkaxes(axs);
linkprop(axs,'CLim');

colorbar

assert(isequal(unique(diff(xroictr)),unique(diff(yroictr)),1));
xroiedge = [xroictr-0.5 xroictr(end)+0.5];
yroiedge = [yroictr-0.5 yroictr(end)+0.5];
save pdfImAlgn20180218.mat FGTHRESH xg yg xroictr yroictr xroiedge yroiedge...
  imforeAlgn imforeBWAlgn imforeAlgnNorm imforeBWAlgnNorm amu bmu n;


%% aligned landmark dist histogram
histcCell = cell(17,1);
histcCellSmth = cell(17,1);
histcCellnorm = cell(17,1);
histcCellnormSmth = cell(17,1);
LMHISTRAD = 60;
xe = -LMHISTRAD:LMHISTRAD;
ye = -LMHISTRAD:LMHISTRAD;
for ipt=1:17
  x = pAtrx(:,ipt);
  y = pAtrx(:,ipt+18);
  histcCell{ipt} = histcounts2(x,y,xe,ye);
  histcCellSmth{ipt} = imgaussfilt(histcCell{ipt},2);

  xnorm = pAtrxnorm(:,ipt);
  ynorm = pAtrxnorm(:,ipt+18);
  histcCellnorm{ipt} = histcounts2(xnorm,ynorm,xe,ye);
  histcCellnormSmth{ipt} = imgaussfilt(histcCellnorm{ipt},2);
end

assert(size(pAtrx,2)==36);
assert(size(pAtrxnorm,2)==36);
x = pAtrx(:,1:17);
y = pAtrx(:,19:35);
histcallpts = histcounts2(x(:),y(:),xe,ye);
histcallpts = imgaussfilt(histcallpts,2);
x = pAtrxnorm(:,1:17);
y = pAtrxnorm(:,19:35);
histcallptsnorm = histcounts2(x(:),y(:),xe,ye);
histcallptsnorm = imgaussfilt(histcallptsnorm,2);

figure;
axs = createsubplots(1,2);

ax = axs(1);
axes(ax);
imagesc(histcallpts');
hold on;
hEll = ellipsedraw(2*amu,2*bmu,LMHISTRAD+1,LMHISTRAD+1,0,'r-');
axis square xy equal;
title('aligned, trx-centered',TITLEARGS{:});
ax = axs(2);
axes(ax);
imagesc(histcallptsnorm');
hold on;
hEll = ellipsedraw(2*amu,2*bmu,LMHISTRAD+1,LMHISTRAD+1,0,'r-');
axis square xy equal;
title('aligned, trx-centered, size-normed',TITLEARGS{:});

linkaxes(axs);
axis(axs(1),[0 120 0 120]);

for ipt=1:17
  figure;
  axs = createsubplots(1,2);
  axes(axs(1));
  imagesc(histcCellSmth{ipt}');
  axis square xy equal;
  title(num2str(ipt),TITLEARGS{:});

  axes(axs(2));
  imagesc(histcCellnormSmth{ipt}');
  axis square xy equal;
  
  linkaxes(axs);
  axis(axs(1),[0 120 0 120]);
end
%% aligned landmark dist histogram, connect legs only

assert(size(pAtrx,2)==36);
assert(size(pAtrxnorm,2)==36);
pAtrx17 = pAtrx(:,[1:17 19:35]);
pAtrxnorm17 = pAtrxnorm(:,[1:17 19:35]);

SEGS = [8 9;10 11;2 12;3 17;9 13;11 16;6 14;6 15];
NPTSSEG = [7; 7; 16; 16; 16; 16; 29; 29]; % each segment gets this many pts
nptPerShape = sum(NPTSSEG);

nseg = size(SEGS,1);
assert(nseg==numel(NPTSSEG));
n = size(pAtrx17,1);

seglen = nan(n,nseg);
seglennorm = nan(n,nseg);
xyLegPts = nan(n,nptPerShape,2);
xyLegPtsnorm = nan(n,nptPerShape,2);
for iShape=1:n
  x = zeros(1,0);
  y = zeros(1,0);
  xnorm = zeros(1,0);
  ynorm = zeros(1,0);
  xyShape = reshape(pAtrx17(iShape,:),17,2);
  xyShapeNorm = reshape(pAtrxnorm17(iShape,:),17,2);
  for iSeg = 1:nseg
    segPt1 = SEGS(iSeg,1);
    segPt2 = SEGS(iSeg,2);
    nptsegI = NPTSSEG(iSeg);
    
    xseg1 = xyShape(segPt1,1);
    xseg2 = xyShape(segPt2,1);
    yseg1 = xyShape(segPt1,2);
    yseg2 = xyShape(segPt2,2);
    xnew = linspace(xseg1,xseg2,nptsegI);
    ynew = linspace(yseg1,yseg2,nptsegI);
    x = [x xnew]; %#ok<AGROW>
    y = [y ynew]; %#ok<AGROW>

    len = sqrt((xseg1-xseg2)^2+(yseg1-yseg2)^2);
    seglen(iShape,iSeg) = len;

    xseg1 = xyShapeNorm(segPt1,1);
    xseg2 = xyShapeNorm(segPt2,1);
    yseg1 = xyShapeNorm(segPt1,2);
    yseg2 = xyShapeNorm(segPt2,2);
    xnew = linspace(xseg1,xseg2,nptsegI);
    ynew = linspace(yseg1,yseg2,nptsegI);
    xnorm = [xnorm xnew]; %#ok<AGROW>
    ynorm = [ynorm ynew]; %#ok<AGROW>

    len = sqrt((xseg1-xseg2)^2+(yseg1-yseg2)^2);
    seglennorm(iShape,iSeg) = len;
  end
  xyLegPts(iShape,:,1) = x;
  xyLegPts(iShape,:,2) = y;
  xyLegPtsnorm(iShape,:,1) = xnorm;
  xyLegPtsnorm(iShape,:,2) = ynorm;
end

xe = -LMHISTRAD:LMHISTRAD;
ye = -LMHISTRAD:LMHISTRAD;
x = xyLegPts(:,:,1);
y = xyLegPts(:,:,2);
histcallpts = histcounts2(x,y,xe,ye);
histclegS = imgaussfilt(histcallpts,2);
x = xyLegPtsnorm(:,:,1);
y = xyLegPtsnorm(:,:,2);
histcallptsnorm = histcounts2(x,y,xe,ye);
histclegnormS = imgaussfilt(histcallptsnorm,2);

figure;
axs = createsubplots(1,2);

ax = axs(1);
axes(ax);
imagesc(histclegS');
hold on;
hEll = ellipsedraw(2*amu,2*bmu,LMHISTRAD+1,LMHISTRAD+1,0,'r-');
axis square xy equal;
title('Regular',TITLEARGS{:});
ax = axs(2);
axes(ax);
imagesc(histclegnormS');
hold on;
hEll = ellipsedraw(2*amu,2*bmu,LMHISTRAD+1,LMHISTRAD+1,0,'r-');
axis square xy equal;
title('Normalized',TITLEARGS{:});

linkaxes(axs);
axis(axs(1),[0 120 0 120]);


fprintf(1,'Mean segment lengths across %d trials: %s\n',n,...
  mat2str(mean(seglen,1),3));
fprintf(1,'Mean segment lengths across %d trials, normalized: %s\n',n,...
  mat2str(mean(seglennorm,1),3));
fprintf(1,'You used these nptsegs: %s\n',mat2str(NPTSSEG));

pleg = histcallpts'/sum(histcallpts(:));
plegS = histclegS'/sum(histclegS(:));
plegnorm = histcallptsnorm'/sum(histcallptsnorm(:));
plegnormS = histclegnormS'/sum(histclegnormS(:));
save pdfLeg20171024.mat pleg plegS plegnorm plegnormS xe ye amu bmu;

%% downsampled LUT ftrs
xroictr = -ROIRAD:ROIRAD;
yroictr = -ROIRAD:ROIRAD;
[xg,yg] = meshgrid(xroictr,yroictr);
lutacc = cell(numel(xg),1); % lut{i} contains the aggregated LUT for the given linear pixel index

xDS3 = -ROIRAD/3:ROIRAD/3;
yDS3 = -ROIRAD/3:ROIRAD/3;
roi2colidxDS3 = round(xg/3)+ROIRAD/3+1; % xg2idxDS3(i,j) gives the column idx into xgDS3 for the corresponding DS3 pt
roi2rowidxDS3 = round(yg/3)+ROIRAD/3+1; % etc
for ishape=1:n  
  if mod(ishape,500)==0
    disp(ishape);
  end
  
  imfore = t.imfore{ishape};
  th = t.thetaTrx(ishape);
  imforecanon = readpdf(imfore,xg,yg,xg,yg,0,0,-th);
  a = t.aTrx(ishape);
  b = t.bTrx(ishape);
  xfac = a/amu;
  yfac = b/bmu;
  imforecanonscale = interp2(xg,yg,imforecanon,xg*xfac,yg*yfac,'linear',0);

  imforecanonscalebw = abs(imforecanonscale)>7;
  imforecanonscaleDS3 = imresize(imforecanonscale,1/3);
  
  % for each point in bw, find the corresponding 3x3 in the downsamp. 
  szassert(imforecanonscale,size(xg));
  assert(xg(1,151)==0);
  assert(yg(151,1)==0);
  szassert(imforecanonscaleDS3,[101 101]);
  idxfore = find(imforecanonscalebw); % linear indices into xg/yg for foreground pxs
  nfore = numel(idxfore);
  for i=1:nfore
    idx = idxfore(i);
    colidxDS3 = roi2colidxDS3(idx);
    rowidxDS3 = roi2rowidxDS3(idx);
    if rowidxDS3<25 || rowidxDS3>75 || colidxDS3<25 || colidxDS3>75
      % FG px at edge due to neighboring fly etc. we are interested only in 
      % central fly
      continue;
    end
    lut = imforecanonscaleDS3(rowidxDS3-1:rowidxDS3+1,colidxDS3-1:colidxDS3+1);
    if isempty(lutacc{idx})
      lutacc{idx} = zeros(3);
    end
    lutacc{idx} = lutacc{idx}+lut;
  end
end


%% Find ANY frames where trx are close together
s = struct('iMov',cell(0,1),'frm',[],'iTgt',[],'jTgt',[],'dTrx',[]);
nmov = lObj.nmovies;
DIST_THRESH_LO = 30;
DIST_THRESH_HI = 60; % px
for iMov=1
  lObj.movieSetGUI(iMov);
  frm2trx = lObj.frm2trx;
  [nfrm,ntgt] = size(frm2trx);
  trx = lObj.trx;
  for f=1:nfrm
    if mod(f,1e3)==0
      fprintf('mov %d, frm %d\n',iMov,f);
    end
    tgtsLive = find(frm2trx(f,:));
    nTgtsLive = numel(tgtsLive);
    for iitgt=1:nTgtsLive
    for jjtgt=iitgt+1:nTgtsLive
      iTgt = tgtsLive(iitgt);
      jTgt = tgtsLive(jjtgt);
      
      trxI = trx(iTgt);
      trxJ = trx(jTgt);
      idxI = f+trxI.off;
      idxJ = f+trxJ.off;
      pTrxI = [trxI.x(idxI) trxI.y(idxI)];
      pTrxJ = [trxJ.x(idxJ) trxJ.y(idxJ)];
      d = sqrt(sum((pTrxI-pTrxJ).^2,2));      
      if DIST_THRESH_LO<d && d<DIST_THRESH_HI
        s(end+1,1).iMov = iMov;
        s(end).frm = f;
        s(end).iTgt = iTgt;
        s(end).jTgt = jTgt;
        s(end).dTrx = d;
      end
    end
    end
  end
end
tDistAll = struct2table(s);
%tDistAll.frm = cell2mat(tDistAll.frm);
tDistAll = sortrows(tDistAll,{'dTrx' 'iMov' 'frm'});

%% Pick a test set:
% 5 examples each from 30-40, 40-50, 50-60
RANGES = [30 40;40 50;50 60];
tKeep = [];
for irange=1:3
  rng = RANGES(irange,:);
  tf = rng(1)<=tDistAll.dTrx & tDistAll.dTrx<=rng(2);
  tTmp = tDistAll(tf,:);
  p = randperm(height(tTmp));
  tTmp = tTmp(p,:);

  numAccepted = 0;
  for i=1:height(tTmp)
    row = tTmp(i,:);
    lObj.setFrameAndTargetGUI(row.frm,row.iTgt);
    str = sprintf('dTrx=%.2f',row.dTrx);
    resp = input(str);
    if resp==1
      tKeep = [tKeep;row]; %#ok<AGROW>
      numAccepted = numAccepted+1;
      if numAccepted>=5
        break;
      end
    end
  end
end

%% ORIG EXAMPLE frm=5828, iTgt/jTgt=3/4, dist=36
im = lObj.currIm;
im = im{1};
figure
imagesc(im);
axis xy square 
colormap gray

%% NEW EXAMPLES
s = struct('im',cell(0,1),'trxx',[],'trxy',[],'trxid',[]);
for i=1:height(tEx)
  row = tEx(i,:);
  lObj.setFrameAndTargetGUI(row.frm,row.iTgt);
  im = lObj.currIm{1};
  trx = lObj.currTrx;
  trxidx = trx.off+row.frm;
  trxx = trx.x(trxidx);
  trxy = trx.y(trxidx);
  
  s(end+1,1).im = im;
  s(end).trxx = trxx;
  s(end).trxy = trxy;
  s(end).trxid = trx.id;
end
tExIm = struct2table(s);
tEx = [tEx tExIm];

%% Load prereqs: bgimage, pdfLeg, trx
load examples15_dTrx_30_60.mat
pdfLeg = load('pdfLeg20171023.mat');
%%
assert(all(tEx.iMov==1));
trx = load(tfaf{1});
trx = trx.trx;

bgpath = fileparts(mfaf{1});
bgfile = fullfile(bgpath,BGDATAFILE);
bg = load(bgfile,'-mat'); 
bg = bg.bkgdImage'; % xpose

%% Full browse
figure;
axs = createsubplots(2,4);
axs = reshape(axs,2,4);
linkaxes(axs);
FORETHRESH = 7;
for i=1:height(tEx)
  trow = tEx(i,:);
  
  ax = axs(1);
  axes(ax);
  cla;
  im = trow.im{1};
  imagesc(im);
  axis xy square equal;
  colormap(ax,'gray');
  title('raw',TITLEARGS{:});
  set(gca,'XTick',[],'YTick',[]);
  
  ax = axs(2);
  axes(ax);
  cla;
  imagesc(bg);
  axis xy square equal;
  colormap(ax,'gray');
  title('bg',TITLEARGS{:});
  set(gca,'XTick',[],'YTick',[]);
  
  im2 = bg-double(im);
  imd = abs(im2);
  ax = axs(3);
  axes(ax);
  cla;
  imagesc(imd);
  axis xy square equal;
  colormap(ax,'jet');
  hCB = colorbar('East');
  hCB.Color = [1 1 1];
  title('foreabs',TITLEARGS{:});
  set(gca,'XTick',[],'YTick',[]);  
  
  forebw = imd>FORETHRESH;
  axes(axs(4));
  cla;
  imagesc(forebw);
  axis xy square equal
  title('forethresh',TITLEARGS{:});
  set(gca,'XTick',[],'YTick',[]);    
  
  forebwl = bwlabel(forebw);
  ax = axs(5);
  axes(ax);
  cla;
  imagesc(forebwl);
  axis xy square equal
  title('forebwl',TITLEARGS{:});
  hCB = colorbar('East');
  hCB.Color = [1 1 1];
  set(gca,'XTick',[],'YTick',[]);  
    
  [bwlnew,bwlnewpre,splitCC,splitCCnew] = assignids(im2,trow.frm,trx,pdfLeg.plegS,...
    pdfLeg.xe,pdfLeg.ye,'bwthresh',FORETHRESH,'verbose',true);

  ax = axs(6);
  axes(ax);
  cla;
  imagesc(imd/255*100);
  axis xy square equal
  hold on;
  imagesc(bwlnew,'AlphaData',0.5);
  title('new bwl',TITLEARGS{:});
  set(gca,'XTick',[],'YTick',[]);    
  
  ax = axs(7);
  axes(ax);
  cla;
  immasked = im;
  ccnew = splitCCnew{1}(1);
  immasked(bwlnew~=ccnew) = 0;
  imagesc(immasked);
  axis xy square equal
  hold on;
  colormap(ax,'gray');
  title('New CC 1',TITLEARGS{:});
  set(gca,'XTick',[],'YTick',[]);    
  
  ax = axs(8);
  axes(ax);
  cla;
  immasked = im;
  ccnew = splitCCnew{1}(2);
  immasked(bwlnew~=ccnew) = 0;
  imagesc(immasked);
  axis xy square equal
  hold on;
  colormap(ax,'gray');
  title('New CC 2',TITLEARGS{:});
  set(gca,'XTick',[],'YTick',[]);    
  
  axis(ax,[1 1024 1 1024]);
  
  str = sprintf('Example %d, dTrx=%.2f',i,trow.dTrx);
  input(str);
end

%% Brief browse
hFig = figure;
hTG = uitabgroup('Parent',hFig);
for i=1:height(tEx)
  
  hT = uitab(hTG,'Title',sprintf('Ex%d',i));
  
  axs = mycreatesubplots(2,4,.02,hT);
  axs = reshape(axs,2,4);

  trow = tEx(i,:);

  ax = axs(1,1);
  axes(ax);
  cla;
  im = trow.im{1};
  imagesc(im);
  axis xy square equal;
  colormap(ax,'gray');
  title('raw',TITLEARGS{:});
  set(gca,'XTick',[],'YTick',[]);
    
  im2 = bg-double(im);
  imd = abs(im2);  
  forebw = imd>FORETHRESH;  
  forebwl = bwlabel(forebw);
    
  [cc,ccpre,splitCC,splitCCnew,pdfTgts] = assignids(im2,...
    trow.frm,trx,...
    pdfLeg.plegnormS,pdfLeg.xe,pdfLeg.ye,...
    'scalePdfLeg',true,...
    'scalePdfLegMeanA',amu,...
    'scalePdfLegMeanB',bmu,...
    'bwthresh',FORETHRESH,...
    'verbose',true);

  ax = axs(2,1);
  axes(ax);
  cla;
%  imagesc(imd/255*100);
  %imagesc(bwlnewpre,'AlphaData',0.5);
  imagesc(bwlnewpre);
  axis xy square equal
  title('new bwl pre',TITLEARGS{:});
  set(gca,'XTick',[],'YTick',[]);
  
  ax = axs(2,2);
  axes(ax);
  cla;
%   imagesc(imd/255*100);  
%   hold on;
%  imagesc(bwlnew,'AlphaData',0.5);
  imagesc(bwlnew);
  axis xy square equal
  title('new bwl',TITLEARGS{:});
  set(gca,'XTick',[],'YTick',[]);
%   if numel(splitCCnew)>0
%     climmin = min(splitCCnew{1});
%     climmax = max(splitCCnew{1});
%     ax.CLim = [climmin climmax];
%   end
  
  if numel(splitCCnew)>0
    ax = axs(1,2);
    axes(ax);
    cla;
    immasked = im;
    ccnew = splitCCnew{1}(1);
    tfmask = bwlnew~=ccnew;
    immasked(tfmask) = bg(tfmask);
    imagesc(immasked);
    axis xy square equal
    hold on;
    colormap(ax,'gray');
    title('New CC 1',TITLEARGS{:});
    set(gca,'XTick',[],'YTick',[]);    

    ax = axs(1,3);
    axes(ax);
    cla;
    immasked = im;
    ccnew = splitCCnew{1}(2);
    tfmask = bwlnew~=ccnew;
    immasked(tfmask) = bg(tfmask);
    imagesc(immasked);
    axis xy square equal
    hold on;
    colormap(ax,'gray');
    title('New CC 2',TITLEARGS{:});
    set(gca,'XTick',[],'YTick',[]);
    
    pdfTgts = pdfTgts{1};
    npdfs = size(pdfTgts,3);
    assert(npdfs==2);
    for j=1:2
      ax = axs(2,j+2);
      axes(ax);
      cla;
      imagesc(pdfTgts(:,:,j));
      axis xy square equal
      hold on;
      hCB = colorbar('East');
      hCB.Color = [1 1 1];
      if i==1
        clim = caxis;
        clim(2) = clim(2)/4;
        caxis(ax,clim);
      end
      title(sprintf('PDF %d',j),TITLEARGS{:});
      set(ax,'XTick',[],'YTick',[]);
    end
  end

  linkaxes(axs);
  
  iTgt = trow.iTgt;
  jTgt = trow.jTgt;
  trxI = trx(iTgt);
  trxJ = trx(jTgt);
  idxI = trxI.off+trow.frm;
  idxJ = trxJ.off+trow.frm;
  xmid = (trxI.x(idxI)+trxJ.x(idxJ))/2;
  ymid = (trxI.y(idxI)+trxJ.y(idxJ))/2;
  ROITWOFLYRAD = 70;
  roi = [xmid-ROITWOFLYRAD xmid+ROITWOFLYRAD ymid-ROITWOFLYRAD ymid+ROITWOFLYRAD];
  
  axis(axs(1),roi);
  str = sprintf('Example %d, dTrx=%.2f',i,trow.dTrx);
  fprintf(str);
end