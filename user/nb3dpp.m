%%
trkdir = 'F:\Dropbox\romain20181203_and2019PostProc\pp3d\mdn0303_60ktrn_3kfrm\';
trk1 = 'bias_video_cam_0_date_2018_11_06_time_14_40_07_v001_trn20190219T160303_iter60000_20190222T145210.trk';
trk2 = 'bias_video_cam_1_date_2018_11_06_time_14_40_11_v001_trn20190219T160303_iter60000_20190222T145210.trk';
trk1 = load(fullfile(trkdir,trk1),'-mat');
trk2 = load(fullfile(trkdir,trk2),'-mat');
%%
ptrk1 = trk1.pTrk;
ptrk2 = trk2.pTrk;
ptrk1 = permute(ptrk1,[3 1 2]);
ptrk2 = permute(ptrk2,[3 1 2]);
[nfrm,npt,d] = size(ptrk1)
szassert(ptrk2,size(ptrk1));
%%
tic;
lblvcd = 'F:\Dropbox\romain20181203_and2019PostProc\romainTrackNov18_al_portable_mp4s.lbl';
lblvcd = load(lblvcd,'-mat');
cr = lblvcd.viewCalibrationData

%% trk RP err
for ipt=1%1:6
  ptrk1tips = ptrk1(:,ipt,:);
  ptrk2tips = ptrk2(:,ipt,:);

  ptrk1tips = reshape(ptrk1tips,nfrm,d);
  ptrk2tips = reshape(ptrk2tips,nfrm,d);
  
  frms = 1:10e3;
  pt1tipsdec = ptrk1tips(frms,:);
  pt2tipsdec = ptrk2tips(frms,:);

  fprintf(1,'### pt %d ###\n',ipt);
  tic
  [X1,xp1rp,xp2rp,rperr1,rperr2] = cr.stereoTriangulate(pt1tipsdec.',pt2tipsdec.');
  toc
  rpe = [rperr1(:) rperr2(:)];
  prctile(rpe,[50 75 90 95 99])
  rpemn = mean(rpe,2);
end
  

%% find frms pt1 > 8px RP err
RPE_THRESH = 15;
flgrpe = find(rpemn>RPE_THRESH)
%find(~tfisunet)

%%
tic;
hmb0 = load('mdn0303_60ktrn_3kfrm\mdn_60kiter_frm7_10k_mat_vw0.mat');
hmb1 = load('mdn0303_60ktrn_3kfrm\mdn_60kiter_frm7_10k_mat_vw1.mat');
toc;
%%
IPT = 1;
hmb = struct();
ftmp = 1:1e3;
hmb.frmagg = hmb0.frmagg(ftmp);
hmb0pt = hmb0.hmagg(ftmp,:,:,IPT);
%clear hmb0;
hmb1pt = hmb1.hmagg(ftmp,:,:,IPT);
%clear hmb1;
hmb.hmagg = cat(4,hmb0pt,hmb1pt);
%%
clear hmb0 hmb1

%%
rc1 = load('mdn0303_60ktrn_3kfrm\mdn_60kiter_3kfrm_mat_vw0_comp2.mat');
rc2 = load('mdn0303_60ktrn_3kfrm\mdn_60kiter_3kfrm_mat_vw1_comp2.mat');

%% with hmap
tic;
[X3d,x2d,trpeconf,isspecial,prefview,...
  x2dcmp,... 
  hmapscore,hmapscorecmp,hmaproi] = runal3dpp(trk1,trk2,cr,...
  'hmbig',hmb,...
  'ipts',1,'frms',7001:8000,'dxyz',0.005,'unethmsarereconned',false);
toc
%% wout hmap
tic;
[X3d_nohm,x2d_nohm,trpeconf_nohm,isspecial_nohm,prefview_nohm,...
  x2dcmp_nohm] = runal3dpp(trk1,trk2,cr,...
  'roisEPline',[1 720 1 540; 1 720 1 540],...
  'ipts',1,'frms',7001:8000,'dxyz',0.005,'unethmsarereconned',false);
toc

%%
trpe = trpeconf{1};
%X3d0 = trpe.X1strotri;
d3d = X3d-X3d_nohm;
err3d = sqrt(sum(d3d.^2,2));
[~,idx] = sort(err3d,'descend');
trpe.err3d = err3d;
trpe.frm = (7001:8000)';
trpe.isal = isal;
trpe.alpreferview = alpreferview;
trpeS = trpe(idx,:);
%%
trpeS(1:50,{'frm' 'err3d' 'isal' 'alpreferview'})
%%
for i=1:300
  disp(trpeS(i,{'frm' 'err3d' 'isal' 'alpreferview'}));
  f = trpeS.frm(i)-7e3;
  disp(round(squeeze(alscorecmpxy(f,:,:,:))));
  
  input(num2str(i));
end

%%
[nfrm,hmnr,hmnc,nvw] = size(hmb.hmagg);
xgv = 1:hmnc;
ygv = 1:hmnr;
[xg,yg] = meshgrid(xgv,ygv);
xyg = [xg(:) yg(:)];

hmbre = struct();
hmbre.frmagg = hmb.frmagg;
hmbre.hmagg = nan(size(hmb.hmagg));
tic;
for f=ftmp(:)'
  hmrc1 = recon(xyg,rc1.mu(:,:,IPT,f)',rc1.sig(2:-1:1,2:-1:1,:,IPT,f),rc1.A(:,IPT,f),hmnr,hmnc,[]);
  hmrc2 = recon(xyg,rc2.mu(:,:,IPT,f)',rc2.sig(2:-1:1,2:-1:1,:,IPT,f),rc2.A(:,IPT,f),hmnr,hmnc,[]);
  hmbre.hmagg(f-1e3,:,:,1) = hmrc1-1;
  hmbre.hmagg(f-1e3,:,:,2) = hmrc2-1;
  if mod(f,100)==0
    disp(f);
  end
end
toc;
%%
tic;
[X3d_re,x2d_re,trpeconf_re,isal_re,alpreferview_re,alscore_re,alscorecmp_re,...
  alscorecmpxy_re,alroi_re] = ...
            runal3dpp(trk1,trk2,hmbre,cr,...
            'ipts',2,'frms',ftmp,'dxyz',0.005,'unethmsarereconned',true);
toc
%%
assert(isequal(trpeconf,trpeconf_re));
assert(isequal(isal,isal_re));
assert(isequal(alpreferview,alpreferview_re));


fisal = find(isal);
dx2d = x2d-x2d_re;
ex2d = squeeze(sqrt(sum(dx2d.^2,2))); % nfrm x nvw
ex2d = ex2d(fisal,:);
ex2dmu = mean(ex2d,2);
[ex2dmnuS,idx] = sort(ex2dmu,'descend')

%%
f = fisal(idx(3))
hm0_vw = squeeze(hmb.hmagg(f,:,:,:))+1;
hm1_vw = squeeze(hmbre.hmagg(f,:,:,:))+1;

figure(1);
axs = mycreatesubplots(2,4);
  
for ivw=1:2
  axes(axs(1,1+(ivw-1)*2));
  imagesc(hm0_vw(:,:,ivw));
  colorbar
  clreg = caxis;

  axes(axs(2,1+(ivw-1)*2));
  imagesc(log(hm0_vw(:,:,ivw)));
  colorbar
  cllog = caxis;

  axes(axs(1,2+(ivw-1)*2));
  imagesc(hm1_vw(:,:,ivw));
  colorbar
  caxis(clreg);

  axes(axs(2,2+(ivw-1)*2));
  imagesc(log(hm1_vw(:,:,ivw)));
  colorbar
  caxis(cllog);
end

linkaxes(axs(1:4));
linkaxes(axs(5:8));

trpeconf{1}(f,1:7)
fprintf(1,'alpreferview is %d\n',alpreferview(f));
fprintf(1,'orig\n');
round(squeeze(alscorecmpxy(f,:,:,:)))
fprintf(1,'new\n');
round(squeeze(alscorecmpxy_re(f,1,:,:)))'















%% 3d diffs
X3d0 = trpeconf{1}.X1strotri;

d3d = X3d-X3d0;
err3d = sqrt(sum(d3d.^2,2));
plot(err3d,'-x');

%% trajsmooth
v3d = diff(X3d,1,1); % v(i,:) gives (dx,dy,dz) that takes you from t=i to t=i+1
v3d(end+1,:,:) = nan; % so v has same size as x, [n x npt x 2]
v3dmag = sqrt(sum(v3d.^2,2)); 

a3d = diff(X3d,2,1); 
a3d = cat(1,nan(1,3),a3d,nan(1,3));
a3dmag = sqrt(sum(a3d.^2,2)); % (n-2) x npt

v3d0 = diff(X3d0,1,1); % v(i,:) gives (dx,dy,dz) that takes you from t=i to t=i+1
v3d0(end+1,:,:) = nan; % so v has same size as x, [n x npt x 2]
v3d0mag = sqrt(sum(v3d0.^2,2)); 

a3d0 = diff(X3d0,2,1); 
a3d0 = cat(1,nan(1,3),a3d0,nan(1,3));
a3d0mag = sqrt(sum(a3d0.^2,2)); % (n-2) x npt

%assert(isequal(size(v3dmag),size(a3dmag)));

prctile([a3dmag a3d0mag],[10 50 75 90 95 99])

%% GT diffs
IPT = 1;

tlbl = load('tlbl446.mat');
tlbl = tlbl.tlbl;
tlbl = tlbl(tlbl.mov==1 & 7001<=tlbl.frm&tlbl.frm<=8000,:);

frmsPP = 7001:8000;
[tf,ilbl] = ismember(tlbl.frm,frmsPP);
assert(all(tf));

nlbl = height(tlbl);
plbl = reshape(tlbl.p,nlbl,19,2,2); % i, ipt, vw, x/y
plbl = squeeze(plbl(:,IPT,:,:)); % i, vw, x/y
plbl = permute(plbl,[1 3 2]); % i, x/y, vw

ppp = x2d(ilbl,:,:);
ptrk = cat(3,trpeconf{1}.ptrk1,trpeconf{1}.ptrk2);
ptrk = ptrk(ilbl,:,:);
prp = cat(3,trpeconf{1}.xp1rp,trpeconf{1}.xp2rp);
prp = prp(ilbl,:,:);

dpp = plbl-ppp;
errpp = squeeze(sum(dpp.^2,2));
dtrk = plbl-ptrk;
errtrk = squeeze(sum(dtrk.^2,2));
drp = plbl-prp;
errrp = squeeze(sum(drp.^2,2));














%%
hmb0 = load('mdn0303_60ktrn_3kfrm\mdn_60kiter_frm7_10k_mat_vw0.mat');
hmb1 = load('mdn0303_60ktrn_3kfrm\mdn_60kiter_frm7_10k_mat_vw1.mat');
%% 
frms = 1:50:3e3;
nfrm = numel(frms);
hmb0agg = reshape(hmb0.hmagg(frms,:,:,:),[nfrm 540 720 1 2]);
hmb1agg = reshape(hmb1.hmagg(frms,:,:,:),[nfrm 540 720 1 2]);
hmb = cat(4,hmb0agg,hmb1agg)+1; % nfrm,nr,nc,nvw,npt
min(hmb0agg(:))
max(hmb0agg(:))
min(hmb1agg(:))
max(hmb1agg(:))
min(hmb(:))
max(hmb(:))
%%
[nfrm,hmnr,hmnc,nvw,npt] = size(hmb)
xgv = 1:hmnc;
ygv = 1:hmnr;
[xg,yg] = meshgrid(xgv,ygv);
xyg = [xg(:) yg(:)];

%%
d = 2;
samps = 2:5;
nsamps = numel(samps);

mu = cell(nsamps,nvw,npt);
S = cell(nsamps,nvw,npt);
w = cell(nsamps,nvw,npt);
maxdevnorm = nan(nsamps,nvw,npt,nfrm);

for isamp=1:nsamps
for ivw=1:nvw 
for ipt=1:npt
  k = samps(isamp);
  mucurr = nan(k,d,nfrm);
  Scurr = nan(d,d,k,nfrm);
  priorcurr = nan(k,nfrm);

  for iF=1:10:nfrm    
    f = 7e3+frms(iF);
    
    fprintf(1,'k=%d,ivw=%d,ipt=%d,iF=%d,f=%d.',k,ivw,ipt,iF,f);

    if ivw==1
      pTrk = trk1.pTrk;
      pTrku = trk1.pTrklocs_unet+1;
    else
      pTrk = trk2.pTrk;
      pTrku = trk2.pTrklocs_unet+1;
    end
    
    isunet = isequaln(pTrk(ipt,:,f),pTrku(ipt,:,f));
      
%     if mod(iF,50)==0
%       disp(f);
%     end

    hm = squeeze(hmb(iF,:,:,ivw,ipt));
    idx = find(hm>0);
    xyhmnz = [xg(idx) yg(idx)];
    hmnz = hm(idx);

    n = numel(idx);
    x = reshape(xyhmnz,[1 n 1 1 2]);
    [mu,w,S] = ...
      PostProcess.GMMFitSamples(x,k,'weights',hmnz(:)','jointpoints',false);
    if isamp==1
      [mu0,w0,S0] = PostProcess.GMMFitHeatmapData(hm,'xgrid',xg,'ygrid',yg);
      [mu1,w1,S1] = PostProcess.GMMFitHeatmapData(hm,'singlecomp',true,'xgrid',xg,'ygrid',yg);
    end
    
    % repro
    [hmre,madev,madevnorm] = recon(xyg,mu,S,w,hmnr,hmnc,hm);
    if isamp==1
      [hmre0,madev0,madevnorm0] = recon(xyg,mu0,S0,w0,hmnr,hmnc,hm);
      [hmre1,madev1,madevnorm1] = recon(xyg,mu1,S1,w1,hmnr,hmnc,hm);
    end
        
    fprintf(1,' isunet=%d, hmresum=%.3f, maxhm=%.3f, madevnorm=%.3f\n',...
      isunet,hmresum,max(hm(:)),madevnorm);
    if isamp==1
      fprintf(1,' ... madevnorm0=%.3f, madevnorm1=%.3f\n',madevnorm0,madevnorm1);
    end
    maxdevnorm(isamp,ivw,ipt,iF) = madevnorm;
    
    input('hk');
  end
end
end
end






%% Recon: GMM test
%load hm

[hmnr,hmnc] = size(hm)
xgv = 1:hmnc;
ygv = 1:hmnr;
[xg,yg] = meshgrid(xgv,ygv);
xyg = [xg(:) yg(:)];

hm(hm<.015) = 0;
idx = find(hm>0);
xyhmnz = [xg(idx) yg(idx)];
hmnz = hm(idx);

n = numel(idx);
x = reshape(xyhmnz,[1 n 1 1 2]);
mu = cell(10,1);
w = cell(10,1);
S = cell(10,1);
hmre = cell(10,1);
for k=2:10
  tic;
  [mu{k},w{k},S{k}] = ...
    PostProcess.GMMFitSamples(x,k,'weights',hmnz(:)','jointpoints',false);
  toc
  [hmre{k},madev,madevnorm,kldev] = recon(xyg,mu{k},S{k},w{k},hmnr,hmnc,hm);
  
  fprintf(1,' k=%d, madev=%.3f, madevnorm=%.3f, kldev=%.3f\n',k,madev,madevnorm,kldev);
end

[mu0,w0,S0] = PostProcess.GMMFitHeatmapData(hm,'singlecomp',true,'xgrid',xg,'ygrid',yg);
[~,madev,madevnorm] = recon(xyg,mu0,S0,w0,hmnr,hmnc,hm);
fprintf(1,' k=1, madev=%.3f, madevnorm=%.3f\n',madev,madevnorm);

mu2 = cell(10,1);
w2 = cell(10,1);
S2 = cell(10,1);
hmre2 = cell(10,1);
for k=2:2:10
  [mu2{k},w2{k},S2{k}] = PostProcess.GMMFitHeatmapData(hm,'xgrid',xg,'ygrid',yg,'nmixpermax',k);
  [hmre2{k},madev,madevnorm,kldev] = recon(xyg,mu2{k},S2{k},w2{k},hmnr,hmnc,hm);
  fprintf(1,' k=%d, madev=%.3f, madevnorm=%.3f, kldev=%.3f\n',k,madev,madevnorm,kldev);
end

%% Recon

% trk1 = load...
% trk2 = load...

IPT = 1;
IVW = 1;
ISUNET = true;

% hmagg = load...
%recon = load('mdn0303_60ktrn_3kfrm\mdn_60kiter_3kfrm_mat_vw0_comp.mat');
rcifo = load('mdn0303_60ktrn_3kfrm\mdn_60kiter_3kfrm_mat_vw0_comp2.mat');

%%
ISINTERACTIVE = false;

[~,hmnr,hmnc,~] = size(hmagg);
xgv = 1:hmnc;
ygv = 1:hmnr;
[xg,yg] = meshgrid(xgv,ygv);
xyg = [xg(:) yg(:)];

if IVW==1
  isunet = trk1.pTrk(IPT,1,:)==trk1.pTrklocs_unet(IPT,1,:)+1;
  trkloc = squeeze(trk1.pTrk(IPT,:,:));
else
  isunet = trk2.pTrk(IPT,1,:)==trk2.pTrklocs_unet(IPT,1,:)+1;
  trkloc = squeeze(trk2.pTrk(IPT,:,:));
end

hfig = figure(11);
axs = mycreatesubplots(2,3);

if ISUNET
  frms = find(isunet);
else
  frms = find(~isunet);
end
fprintf(1,'ISUNET=%d, %d frames.\n',ISUNET,numel(frms));

%for f=frms(:)'
%frmslook = frms3(idx(5:end));
mde2agg = nan(numel(frms),1);
numnz = nan(numel(frms),1);
for iF=1:numel(frms)
  if mod(iF,50)==0
    disp(iF);
  end
  
  f = frms(iF);  
  
  hm = squeeze(hmagg(f,:,:,IPT));
  hm = hm + 1;
  hm(hm<0.015) = 0;
  
  numnz(iF) = nnz(hm);
  continue;
  
  cc = bwconncomp(hm);
  ncomp = cc.NumObjects;
  
  rc1 = mvnpdf(xyg,reconinfo.mu(2:-1:1,IPT,f,1)',reconinfo.sig(2:-1:1,2:-1:1,IPT,f,1));
  rc1 = rc1/sum(rc1(:))*reconinfo.A(IPT,f,1);
  rc1 = reshape(rc1,[hmnr hmnc]);

  [rc2,~,~,~,ngauss] = recon(xyg,rcifo.mu(:,:,IPT,f)',rcifo.sig(2:-1:1,2:-1:1,:,IPT,f),rcifo.A(:,IPT,f),hmnr,hmnc,hm);
%   rc2 = mvnpdf(xyg,recon.mu(2:-1:1,IPT,f,2)',recon.sig(2:-1:1,2:-1:1,IPT,f,2));
%   rc2 = rc2/sum(rc2(:))*recon.A(IPT,f,2);
%   rc2 = reshape(rc2,[hmnr hmnc]);
  
  loc = trkloc(:,f);
  
  d1 = hm-rc1;
  d2 = hm-rc2;
  rce1 = sum(abs(d1(:)))/hmnr/hmnc;
  rce2 = sum(abs(d2(:)))/hmnr/hmnc;
  mde1 = max(abs(d1(:)));
  mde2 = max(abs(d2(:)));
  
  mde2agg(iF) = mde2;
  
  if ISINTERACTIVE
    axes(axs(1,1));
    cla;
    imagesc(hm);
    colorbar
    cl = caxis;
    hold on
    plot(loc(1),loc(2),'rx','linewidth',5,'markersize',20);
    
    axes(axs(1,2));
    cla;
    imagesc(rc1);
    colorbar
    caxis(cl);
    
    axes(axs(1,3));
    cla;
    imagesc(rc2);
    colorbar
    caxis(cl);
    
    axes(axs(2,2));
    cla;
    imagesc(hm-rc1);
    colorbar
    cl = caxis;
    
    axes(axs(2,3));
    cla;
    imagesc(hm-rc2);
    colorbar
    caxis(cl);
    
    linkaxes(axs);
    if ISUNET
      xlim(axs(1),loc(1)+[-25 25]);
      ylim(axs(1),loc(2)+[-25 25]);
    end
    
    str = sprintf('frame %d. rce1/rce2=%.4g/%.4g, mde1/mde2=%.3g/%.3g, ncomp/ngauss %d/%d\n',...
      f,rce1,rce2,mde1,mde2,ncomp,ngauss);
    input(str);
  end
end

    
    
    
    
    
    
    
    
    
%%
hfig = figure(11);
axs = mycreatesubplots(2,2);
Nlook = 50;

maxdevnorm = nan(Nlook,1);
isInteract = true;

for f=40:Nlook
  if mod(f,50)==0
    disp(f);
  end
  
  mu = mucurr(:,:,f);
  S = Scurr(:,:,:,f);
  w = priorcurr(:,f);
  
  hmre = zeros(hmnr*hmnc,1);
  for i=1:k
    if ~any(isnan(mu(i,:)))
      y = mvnpdf(xyg,mu(i,:),S(:,:,i));
      hmre = hmre + w(i)*y;
    end
  end
  hmre = reshape(hmre,[hmnr hmnc]);
  
  hm = hmb(:,:,f);
  
  hmre = hmre*sum(hm(:));
  
  d = hmre-hm;
  maxdevnorm(f) = max(abs(d(:)));
  
  if isInteract
    fprintf(1,'Sum hm is %.3f, sum hmre is %.3f\n',sum(hm(:)),sum(hmre(:)));
    axes(axs(1,1));
    cla
    imagesc(hm);
    %caxis([0 1]);
    hold on;
    colorbar
    %plot(ptrk1(f,1,1),ptrk1(f,1,2),'x','markersize',20);
    axes(axs(1,2));
    cla
    imagesc(hmre);
    %caxis([0 1]);
    colorbar
    
    axes(axs(2,1));
    cla
    imagesc(dev);
    colorbar;
    axes(axs(2,2));
    cla
    hS = scatter(mu(:,1),mu(:,2),500*w,'filled');
    axis ij

    linkaxes(axs);

    xlo = find(any(hm,1),1,'first');
    xhi = find(any(hm,1),1,'last');
    ylo = find(any(hm,2),1,'first');
    yhi = find(any(hm,2),1,'last');
    axis([xlo xhi ylo yhi]);
  
    fprintf('Max deviation is %.3f\n',max(abs(d(:))));  
  
    input(num2str(f));  
  end
end

%%
hmb1 = load('mdn0303_60k\mdn_60k_mat_vw0.mat');
hmb2 = load('mdn0303_60k\mdn_60k_mat_vw1.mat');
trk1 = load('mdn0303_60k\bias_video_cam_0_date_2018_11_06_time_14_40_07_v001_trn20190219T160303_iter60000_20190222T112830.trk','-mat');
trk2 = load('mdn0303_60k\bias_video_cam_1_date_2018_11_06_time_14_40_11_v001_trn20190219T160303_iter60000_20190222T112830.trk','-mat');
%%
IPT = 1;
tf1 = trk1.pTrklocs_mdn==trk1.pTrklocs_unet;
tf2 = trk2.pTrklocs_mdn==trk2.pTrklocs_unet;
tf1 = squeeze(tf1(IPT,:,:));
tf2 = squeeze(tf2(IPT,:,:));
tfisunet1 = all(tf1,1);
tfisunet2 = all(tf2,1);
tfisunet = tfisunet1 & tfisunet2;
isunet1 = find(tfisunet1);
isunet2 = find(tfisunet2);
isunet = intersect(isunet1,isunet2);
%%

isunet = false(10e3,2);
ismdn = false(10e3,2);
dist = squeeze(sqrt(sum((trk2.pTrklocs_mdn-trk2.pTrklocs_unet).^2,2)));
dist = dist(1:2,:)';
unethmmax1 = nan(0,2);
unethmmax2 = nan(0,2);
for ipt=1:2
  for f=1:10e3
    isunet(f,ipt) = isequal(trk2.pTrk(ipt,:,f),trk2.pTrklocs_unet(ipt,:,f)+1);
    ismdn(f,ipt) = isequal(trk2.pTrk(ipt,:,f),trk2.pTrklocs_mdn(ipt,:,f)+1);
    if isunet(f,ipt)
      assert(dist(f,ipt)==0);
      hm1_vw2 = hmb2.hmbig(:,:,f,ipt);
      unethmmax1(end+1) = max(hm1_vw2(:));
%       if unethmmax1(end)==0
%         error
%       end
    else
      hm1_vw2 = hmb2.hmbig(:,:,f,ipt);
      unethmmax2(end+1) = max(hm1_vw2(:));
    end
  end
end


%%
IPT = 2;
FRM = 1785;
THRESH = .015;
for f=1805:1881
FRM = f;
idx1 = find(FRM==hmb1.frmagg);
idx2 = find(FRM==hmb2.frmagg);
assert(idx1==idx2);
hm1raw = squeeze(hmb1.hmagg(idx1,:,:,IPT));
hm2raw = squeeze(hmb2.hmagg(idx2,:,:,IPT));
hm1raw = double(hm1raw+1);
hm2raw = double(hm2raw+1);
hm1_vw2 = hm1raw;
hm2 = hm2raw;
hm1_vw2(hm1_vw2<THRESH) = 0;
hm2(hm2<THRESH) = 0;
hm1_vw2 = hm1_vw2/sum(hm1_vw2(:));
hm2 = hm2/sum(hm2(:));
[sbest,Xbest,xy1best,xy2best,xlo1,xhi1,ylo1,yhi1,xlo2,xhi2,ylo2,yhi2] = ...
  al3dpp(hm1_vw2,hm2,cr,'dxyz',.01);

idx = find(FRM==trk1.pTrkFrm);
pTrk1 = trk1.pTrk(IPT,:,idx)';
pTrk2 = trk2.pTrk(IPT,:,idx)';
[~,pTrk1rp,pTrk2rp,rperr1,rperr2] = cr.stereoTriangulate(pTrk1,pTrk2);
fprintf(1,'Rperr vw1 vw2: %.2f %.2f\n',rperr1,rperr2);


xyctr1 = [xlo1+xhi1 ylo1+yhi1]/2;
xyctr2 = [xlo2+xhi2 ylo2+yhi2]/2;
xyrad1 = [xhi1-xlo1 yhi1-ylo1]/2;
xyrad2 = [xhi2-xlo2 yhi2-ylo2]/2;
xyraduse = max(xyrad1,xyrad2);

hfig = figure(12);
clf
axs = mycreatesubplots(1,2);
axes(axs(1));
imagesc(hm1_vw2);
hold on;
plot(xy1best(1),xy1best(2),'rx','markerfacecolor',[1 0 0],'markersize',20,'linewidth',3);
plot(trk1.pTrk(IPT,1,idx),trk1.pTrk(IPT,2,idx),'ro','linewidth',3);
plot(pTrk1rp(1),pTrk1rp(2),'r^','linewidth',3);
axis([xyctr1(1)-xyraduse(1) xyctr1(1)+xyraduse(1) xyctr1(2)-xyraduse(2) xyctr1(2)+xyraduse(2)]);
colorbar
axes(axs(2));
imagesc(hm2);
hold on;
plot(xy2best(1),xy2best(2),'rx','markerfacecolor',[1 0 0],'markersize',20,'linewidth',3);
plot(trk2.pTrk(IPT,1,idx),trk2.pTrk(IPT,2,idx),'ro','linewidth',3);
plot(pTrk2rp(1),pTrk2rp(2),'r^','linewidth',3);
axis([xyctr2(1)-xyraduse(1) xyctr2(1)+xyraduse(1) xyctr2(2)-xyraduse(2) xyctr2(2)+xyraduse(2)]);
%caxis(caxis(axs(1)))
colorbar

xall1 = [xy1best(1) trk1.pTrk(IPT,1,idx) pTrk1rp(1)];
yall1 = [xy1best(2) trk1.pTrk(IPT,2,idx) pTrk1rp(2)];
xall2 = [xy2best(1) trk2.pTrk(IPT,1,idx) pTrk2rp(1)];
yall2 = [xy2best(2) trk2.pTrk(IPT,2,idx) pTrk2rp(2)];
interp2(log(hm1_vw2),xall1,yall1) + interp2(log(hm2),xall2,yall2)

input(num2str(FRM));
end

%%
IPART = 1;
hfig = figure(3);
for ifrm=1:numel(hmb1.frmagg)
  f = hmb1.frmagg(ifrm);
  hm = hmb1.hmagg(ifrm,:,:,IPART);
  hm = squeeze(hm);
  
  fstr = num2str(f);
  args = {'fontweight','bold','interpreter','none'};
    
  imagesc(hm);
  %caxis([100 255]);
  colorbar
  cx = caxis;
  tstr = sprintf('%s: %s',fstr,mat2str(cx));
  title(tstr,args{:});
  
  input(num2str(ifrm));
end