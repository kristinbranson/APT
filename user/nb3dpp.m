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
hmb0 = load('mdn0303_60ktrn_3kfrm\mdn_60kiter_3kfrm_mat_vw0.mat');
hmb0 = load('mdn0303_60ktrn_3kfrm\mdn_60kiter_frm7_10k_mat_vw0.mat');
%%
IPT = 1;
hmb = struct();
hmb.frmagg = hmb0.frmagg;
hmb0pt = hmb0.hmagg(:,:,:,IPT);
%clear hmb0;
hmb1pt = hmb1.hmagg(:,:,:,IPT);
%clear hmb1;
hmb.hmagg = cat(4,hmb0pt,hmb1pt);

%%
tic;
[X3d,x2d,trpeconf,isal,alpreferview,alscore,alscorecmp,alscorecmpxy,alroi] = ...
            runal3dpp(trk1,trk2,hmb,cr,...
            'ipts',1,'frms',7001:8000,'dxyz',0.005);
toc

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
load(fullfile(trkdir,'mdn_vw0.mat'));

%%
N = 50;
VIEW = 1;
hmb = double(hmbig(:,:,1:N,VIEW))/65;
[hmnr,hmnc,n] = size(hmb)
min(hmb(:))
max(hmb(:))
%%
xgv = 1:hmnc;
ygv = 1:hmnr;
[xg,yg] = meshgrid(xgv,ygv);

%%
nsamples_perview = 5;
d_in = 2;
ncurr = N;
mucurr = nan([nsamples_perview,d_in,ncurr]);
Scurr = nan([d_in,d_in,nsamples_perview,ncurr]);
priorcurr = nan([nsamples_perview,ncurr]);

for f=40:N
  if mod(f,50)==0
    disp(f);
  end

  hm = hmb(:,:,f);
  idx = find(hm>0);
  xyhmnz = [xg(idx) yg(idx)];
  hmnz = hm(idx);
  
  n = numel(idx);
  x = reshape(xyhmnz,[1 n 1 1 2]);
  [mucurr(:,:,f),priorcurr(:,f),Scurr(:,:,:,f)] = ...
    PostProcess.GMMFitSamples(x,nsamples_perview,'weights',hmnz(:)','jointpoints',false);  
end

%%
hfig = figure(11);
axs = mycreatesubplots(2,2);
Nlook = 50;

maxdev = nan(Nlook,1);
isInteract = true;

xyg = [xg(:) yg(:)];
for f=40:Nlook
  if mod(f,50)==0
    disp(f);
  end
  
  mu = mucurr(:,:,f);
  S = Scurr(:,:,:,f);
  w = priorcurr(:,f);
  
  hmre = zeros(hmnr*hmnc,1);
  for i=1:nsamples_perview
    if ~any(isnan(mu(i,:)))
      y = mvnpdf(xyg,mu(i,:),S(:,:,i));
      hmre = hmre + w(i)*y;
    end
  end
  hmre = reshape(hmre,[hmnr hmnc]);
  
  hm = hmb(:,:,f);
  
  hmre = hmre*sum(hm(:));
  
  d = hmre-hm;
  maxdev(f) = max(abs(d(:)));
  
  if isInteract
    fprintf(1,'Sum hm is %.3f, sum hmre is %.3f\n',sum(hm(:)),sum(hmre(:)));
    axes(axs(1,1));
    imagesc(hm);
    %caxis([0 1]);
    hold on;
    plot(ptrk1(f,1,1),ptrk1(f,1,2),'x','markersize',20);
    axes(axs(1,2));
    imagesc(hmre);
    %caxis([0 1]);
    
    axes(axs(2,1));
    imagesc(d);
    colorbar;
    axes(axs(2,2));
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
      hm1 = hmb2.hmbig(:,:,f,ipt);
      unethmmax1(end+1) = max(hm1(:));
%       if unethmmax1(end)==0
%         error
%       end
    else
      hm1 = hmb2.hmbig(:,:,f,ipt);
      unethmmax2(end+1) = max(hm1(:));
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
hm1 = hm1raw;
hm2 = hm2raw;
hm1(hm1<THRESH) = 0;
hm2(hm2<THRESH) = 0;
hm1 = hm1/sum(hm1(:));
hm2 = hm2/sum(hm2(:));
[sbest,Xbest,xy1best,xy2best,xlo1,xhi1,ylo1,yhi1,xlo2,xhi2,ylo2,yhi2] = ...
  al3dpp(hm1,hm2,cr,'dxyz',.01);

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
imagesc(hm1);
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
interp2(log(hm1),xall1,yall1) + interp2(log(hm2),xall2,yall2)

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