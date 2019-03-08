%%
%ld = load('f:\Dropbox\romainNov2018Tracking\romainTrackNov18_al_portable.lbl','-mat');
%ld = load('f:\Dropbox\romain20181203\romainTrackNov18_al_portable_mp4s.lbl','-mat');
ld = load('/groups/branson/home/leea30/apt/romain20181129/romainTrackNov18_al_portable_mp4s.lbl','-mat');
t = Labeler.lblFileGetLabels(ld);

%%
vcd = ld.viewCalibrationData;
cr = vcd;

nphyspts = 19;
nviews = 2;
%%
ngridpts = height(t);
pLbl = reshape(t.p,[ngridpts nphyspts nviews 2]); % [i,ipt,ivw,x/y]

tic;
pcam1 = permute(pLbl(:,:,1,:),[4 1 2 3]); % 2 x n x nphyspts
pcam2 = permute(pLbl(:,:,2,:),[4 1 2 3]);
pcam1 = reshape(pcam1,2,ngridpts*nphyspts);
pcam2 = reshape(pcam2,2,ngridpts*nphyspts);
rperr1 = nan(ngridpts*nphyspts,1);
rperr2 = nan(ngridpts*nphyspts,1);
wbObj = WaitBarWithCancelCmdline('strotri');
wbObj.startPeriod('strotri','shownumden',true,'denominator',ngridpts*nphyspts);

d1erragg = [];
d2erragg = [];
for i=1:ngridpts*nphyspts
  wbObj.updateFracWithNumDen(i);
  [X1tmp,~,~,rperr1(i),rperr2(i)] = cr.stereoTriangulate(pcam1(:,i),pcam2(:,i));
  
  
  sp = cr.stroParams;
  R = sp.RotationOfCamera2;
  T = sp.TranslationOfCamera2;
  xyRP2 = worldToImage(sp.CameraParameters2,R,T,X1tmp','applyDistortion',true);
  xyRP2nd = worldToImage(sp.CameraParameters2,R,T,X1tmp','applyDistortion',false);
  d2 = (xyRP2-xyRP2nd);
  R = eye(3);
  T = [0 0 0];
  xyRP1 = worldToImage(sp.CameraParameters1,R,T,X1tmp','applyDistortion',true);
  xyRP1nd = worldToImage(sp.CameraParameters1,R,T,X1tmp','applyDistortion',false);
  d1 = (xyRP1-xyRP1nd);
  %fprintf('d1: %s. d2: %s\n',mat2str(d1),mat2str(d2));  
  d1erragg = [d1erragg; d1];
  d2erragg = [d2erragg; d2];
end

toc

rperr1 = reshape(rperr1,ngridpts,nphyspts);
rperr2 = reshape(rperr2,ngridpts,nphyspts);

CalRigMLStro.rperrPlot(rperr1,rperr2);

tstr = sprintf('RP err (nLblRows=%d)',ngridpts);
title(tstr,'fontweight','bold','fontsize',18);
%% Browse large RP err
LARGE_RPERR_THRESH = 3;
[i1,iphyspt1] = find(rperr1>LARGE_RPERR_THRESH);
[i2,iphyspt2] = find(rperr1>LARGE_RPERR_THRESH);
i = [i1;i2];
iphyspt = [iphyspt1;iphyspt2];
[t(i,{'mov' 'frm'}) table(iphyspt)]

%%
IMNR = 540;
IMNC = 720;
GRIDDX = 180;
%Z1RANGE = 85:105;
%nz = numel(Z1RANGE);

roi = [1 IMNC 1 IMNR];

xgv = 0:GRIDDX:IMNC;
ygv = 0:GRIDDX:IMNR;
[x,y] = meshgrid(xgv,ygv);
xy = [x(:) y(:)]';
ngridpts = size(xy,2);

fprintf('%d grid pts\n',ngridpts);

xyn = nan(2,ngridpts,2); % x/y, i, cam. normalized coords
X = cell(2,1); 
% X{icam} is nan(3,nzicam,ngridpts); % x/y/z, i, iz. 3D coords, in each cams' coord sys
for icam=1:2
  z1range = cr.eplineZrange{icam};
  for i=1:ngridpts
    if mod(i,10)==0
      disp(i);
    end
    [~,~,Xc1,Xc1OOB] = cr.computeEpiPolarLine(icam,xy(:,i),mod(icam,2)+1,roi);
    if icam==1
      X{icam}(:,:,i) = Xc1;
    else
      X{icam}(:,:,i) = cr.camxform(Xc1,[2 1]);
    end
  end  
end

%%
gridptclrs = jet(ngridpts);
LINESTYLES = {'-','-.'};

hFig = figure(11);
clf
set(hFig,'Name','EPlines');

for icam=1:2
  mrkr = LINESTYLES{icam};
  for igp=1:ngridpts
    clr = gridptclrs(igp,:);
    plot3(X{icam}(1,:,igp),X{icam}(2,:,igp),X{icam}(3,:,igp),'color',clr,'linestyle',mrkr);
    hold on;
  end
end

grid on;
axis equal;
xlabel('x','fontweight','bold','fontsize',20);
ylabel('y','fontweight','bold','fontsize',20);
zlabel('z','fontweight','bold','fontsize',20);

x = X{1}(:,2,20)-X{1}(:,1,20);
y = X{2}(:,2,20)-X{2}(:,1,20);
z = cross(x,y);
r = sqrt(sum(z(1:2).^2));
th = atan2(r,z(3));
phi = atan2(z(2),z(1));
az = (phi+pi/2)/pi*180;
el = (pi/2-th)/pi*180;
view(az,el);

%% Figure out DXYZ
dxyz = 0.01;

Z0 = 92.5; % eyeballed
PTOPE_XRANGE = [-10 10];
PTOPE_YRANGE = [-7 7];
PTOPE_ZRANGE = [Z0 Z0];
xgv = PTOPE_XRANGE(1):dxyz:PTOPE_XRANGE(2);
ygv = PTOPE_YRANGE(1):dxyz:PTOPE_YRANGE(2);
zgv = PTOPE_ZRANGE(1):dxyz:PTOPE_ZRANGE(2);
[x,y,z] = meshgrid(xgv,ygv,zgv);
Xg1 = [x(:) y(:) z(:)]; % [3xng]. Xgrid, cam1 coord sys
ng = size(Xg1,1);

sp = cr.stroParams;
R = sp.RotationOfCamera2;
T = sp.TranslationOfCamera2;
xy2 = worldToImage(sp.CameraParameters2,R,T,Xg1,'applyDistortion',true);
R = eye(3);
T = [0 0 0];
xy1 = worldToImage(sp.CameraParameters1,R,T,Xg1,'applyDistortion',true);

%%
PTOPE_XRANGE = [-10 10];
PTOPE_YRANGE = [-7 7];
PTOPE_ZRANGE = [80 110];
DXYZ = 0.01;
xgv = PTOPE_XRANGE(1):DXYZ:PTOPE_XRANGE(2);
ygv = PTOPE_YRANGE(1):DXYZ:PTOPE_YRANGE(2);
zgv = PTOPE_ZRANGE(1):DXYZ:PTOPE_ZRANGE(2);

for iz=1:numel(zgv)

  %[x,y,z] = meshgrid(xgv,ygv,zgv);
  [x,y] = meshgrid(xgv,ygv);
  Xg1 = [x(:) y(:)];
  Xg1(:,3) = zgv(iz);
  ng = size(Xg1,1);

  sp = cr.stroParams;
  R = sp.RotationOfCamera2;
  T = sp.TranslationOfCamera2;
  xy2 = worldToImage(sp.CameraParameters2,R,T,Xg1,'applyDistortion',true);
  xy2nd = worldToImage(sp.CameraParameters2,R,T,Xg1,'applyDistortion',false);
  R = eye(3);
  T = [0 0 0];
  xy1 = worldToImage(sp.CameraParameters1,R,T,Xg1,'applyDistortion',true);
  xy1nd = worldToImage(sp.CameraParameters1,R,T,Xg1,'applyDistortion',false);
  %%
  tfIB1 = roi(1)<=xy1(:,1) & xy1(:,1)<=roi(2) & ...
          roi(3)<=xy1(:,2) & xy1(:,2)<=roi(4);
  tfIB2 = roi(1)<=xy2(:,1) & xy2(:,1)<=roi(2) & ...
          roi(3)<=xy2(:,2) & xy2(:,2)<=roi(4);
  tfIB = tfIB1 & tfIB2;
  fprintf(1,'%d candidates, %d IB1, %d IB2, %d IB\n',numel(tfIB),...
    nnz(tfIB1),nnz(tfIB2),nnz(tfIB));
end

%%

XgFov = Xg1(tfIB,:);
figure(22);
pcshow(XgFov,'MarkerSize',80);
xlabel('x','fontweight','bold','fontsize',20);
ylabel('y','fontweight','bold','fontsize',20);
zlabel('z','fontweight','bold','fontsize',20);

%%
idxIB = find(tfIB);
save pp3dIfo xgv ygv zgv Xg1 xy1 xy2 xy1nd xy2nd idxIB
%%
figure;
scatter(pp3dIfo.xy1(idxIB,1),pp3dIfo.xy1(idxIB,2));

%% XV err
err = reshape(err,[446 19 1 2 1]);
[hFig,hAxs] = GTPlot.ptileCurves(err,...
  'ptiles',[50 75 90 95 97.5],...
  'axisArgs',{'XTicklabelRotation',45,'FontSize' 16},...
  'createsubplotsborders',[.05 0;.12 .12],...
  'titleArgs',{'fontweight','bold'}...
  );