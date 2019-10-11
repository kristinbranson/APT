%%
CRIGFILES = {
  'f:\stephen/crig_noHandles_20160911.mat'
  'f:\stephen/EPline.debug.20160912/ForAllen/crSH_20160912.mat'};
crigs = cellfun(@load,CRIGFILES,'uni',0);
nrig = numel(crigs);
%%
gam1s = nan(2,nrig);
p01s = nan(2,nrig);
R1s = nan(3,3,nrig);
t1s = nan(3,nrig);
Rdet1s = zeros(1,nrig);
Rnorm1s = zeros(1,nrig);
gam2s = nan(2,nrig);
p02s = nan(2,nrig);
R2s = nan(3,3,nrig);
t2s = nan(3,nrig);
Rdet2s = zeros(1,nrig);
Rnorm2s = zeros(1,nrig);
for iRig=1:nrig
  cr = crigs{iRig}.cr;
  kdfile = cr.kineDataFile;
  fprintf('%s\n',kdfile);
  
  dlt1 = cr.kineData.cal.coeff.DLT_1;
  dlt2 = cr.kineData.cal.coeff.DLT_2;
  
  [gam1s(:,iRig),~,p01s(:,iRig),~,R1s(:,:,iRig),~,t1s(:,iRig)] = ...
    DLT.dlt2cam(dlt1);
  Rdet1s(iRig) = det(R1s(:,:,iRig));
  Rnorm1s(iRig) = norm(R1s(:,:,iRig)*R1s(:,:,iRig)'-eye(3));
  
  [gam2s(:,iRig),~,p02s(:,iRig),~,R2s(:,:,iRig),~,t2s(:,iRig)] = ...
    DLT.dlt2cam(dlt2);
  Rdet2s(iRig) = det(R2s(:,:,iRig));
  Rnorm2s(iRig) = norm(R2s(:,:,iRig)*R2s(:,:,iRig)'-eye(3));
end
%%
[gam2,s,p0,K,R,om,t] = DLT.dlt2cam(dlt2);
det(R)
L = DLT.cam2dlt(gam2,0,p0,om,t);
[gam2,s2,p02,K2,R2,om2,t2] = DLT.dlt2cam(L);
L2 = DLT.cam2dlt(gam2,0,p02,om2,t2);
  

%% calib data
X = e.threedGridPointsList.allVertices;
x1 = e.twodGridPointsList.allVertices.view1;
x2 = e.twodGridPointsList.allVertices.view2;
idxIg1 = e.twodGridPointsList.view1.ignore;
idxIg2 = e.twodGridPointsList.view2.ignore;

N = size(X,1);
szassert(X,[N 3]);
szassert(x1,[N 2]);
szassert(x2,[N 2]);

X1cut = X;
x1cut = x1;
X1cut(idxIg1,:) = [];
x1cut(idxIg1,:) = [];
X2cut = X;
x2cut = x2;
X2cut(idxIg2,:) = [];
x2cut(idxIg2,:) = [];

[L1,res1] = DLTFU(X1cut,x1cut);
[LL1,res11] = DLTFU(X,x1,idxIg1);
[L2,res2] = DLTFU(X2cut,x2cut);
[LL2,res22] = DLTFU(X2cut,x2cut);

%% manual relabel 30 pts
v1 = e.twodGridPointsList.allVertices.view1;
idxIg1 = e.twodGridPointsList.view1.ignore;
v2 = e.twodGridPointsList.allVertices.view2;
idxIg2 = e.twodGridPointsList.view2.ignore;
n1 = size(v1,1);
n2 = size(v2,1);

mr(1) = MovieReader; mr(1).open('C001H001S0001_c.avi');
mr(2) = MovieReader; mr(2).open('C002H001S0001_c.avi');
im1 = mr(1).readframe(1);
im1 = im1+25;
im2 = mr(2).readframe(1);
im2 = im2+25;
hFig = figure;
ax = createsubplots(1,2);
imagesc(ax(1),im1);
hold(ax(1),'on');
axis(ax(1),'ij','image');
imagesc(ax(2),im2);
hold(ax(2),'on');
axis(ax(2),'ij','image');

GAMMA = .4;
mgray = gray(256);
mgray2 = imadjust(mgray,[],[],GAMMA);
arrayfun(@(x)colormap(x,mgray2),ax);

hP1 = plot(ax(1),nan,nan,'.','Color',[1 1 0],'MarkerSize',16);
v1Keep = v1;
v1KeepLbl = arrayfun(@num2str,(1:n1)','uni',0);
v1Keep(idxIg1,:) = [];
v1KeepLbl(idxIg1,:) = [];
set(hP1,'XData',v1Keep(:,1),'YData',v1Keep(:,2));

hP2 = plot(ax(2),nan,nan,'.','Color',[1 1 0],'MarkerSize',16);
v2Keep = v2;
v2KeepLbl = arrayfun(@num2str,(1:n2)','uni',0);
v2Keep(idxIg2,:) = [];
v2KeepLbl(idxIg2,:) = [];
set(hP2,'XData',v2Keep(:,1),'YData',v2Keep(:,2));

for iFile=1:250
  xy = v1Keep(iFile,:);
  %set(hP,'XData',xy1(1),'YData',xy1(2));
  %  set(hT,'Position',[xy1(1) xy1(2) 1],'String',num2str(i));
%   hP.XData = plot(ax(1),xy1(1),xy1(2),'.','Color',[1 0 0],'MarkerSize',16);
  text(ax(1),xy(1),xy(2),v1KeepLbl{iFile},'Color',[1 0 0]);
%   input(sprintf('%d\n',i));
%   xy2 = fv2(i,:);
%   text(ax(2),xy2(1),xy2(2),num2str(i),'Color',[1 0 0]);
%  end
end
for iFile=1:size(v2Keep,1)
  xy = v2Keep(iFile,:);
  %set(hP,'XData',xy1(1),'YData',xy1(2));
  %  set(hT,'Position',[xy1(1) xy1(2) 1],'String',num2str(i));
%   hP.XData = plot(ax(1),xy1(1),xy1(2),'.','Color',[1 0 0],'MarkerSize',16);
  text(ax(2),xy(1),xy(2),v2KeepLbl{iFile},'Color',[1 0 0]);
%   input(sprintf('%d\n',i));
%   xy2 = fv2(i,:);
%   text(ax(2),xy2(1),xy2(2),num2str(i),'Color',[1 0 0]);
%  end
end

%% View1: Get uvMan and X

% set up maps
IPT70_TO_GLOBAL = [883:903 924:21:1218 1217:-1:1199 1198:-21:904];
LBL = 'f:\stephen\DLTcalib20170125\view1_mainface70.lbl';
lbl = load(LBL,'-mat');
lpos = lbl.labeledpos{1};
uvMan = lpos(:,:,1);
szassert(uvMan,[70 2]);

IPT22_TO_GLOBAL = [1:21:190 86:97]; % Duped global points for 1:21:190: 442:461
LBL = 'f:\stephen\DLTcalib20170125\view1_face2.lbl';
lbl = load(LBL,'-mat');
lpos = lbl.labeledpos{1};
uvMan = [uvMan; lpos(1:22,:,1)];
szassert(uvMan,[70+22 2]);

iptGlobal = [IPT70_TO_GLOBAL(:); IPT22_TO_GLOBAL(:)];
szassert(iptGlobal,[70+22 1]);

% get 3d coords
dltstuff = load('exampleDLTstuff_359_373.mat');
X = dltstuff.threedGridPointsList.allVertices(iptGlobal,:);
szassert(X,[70+22 3]);

e = load('DLTs_17_18_16_fly_359_to_373.mat');
X = e.threedGridPointsList.allVertices;
uvMan = e.twodGridPointsList.allVertices.view2;
idxIgnore = e.twodGridPointsList.view2.ignore;
X(idxIgnore,:) = [];
uvMan(idxIgnore,:) = [];


% DLTFU for p0
[L0,L0res] = dltfu(X,uvMan)

%%
[gam1,~,u0v0_1,TIO_0,om1,x0y0z0] = DLT.dlt2cam(L0);
round(gam1)
round(u0v0_1)
det(TIO_0)
norm(TIO_0*TIO_0'-eye(3))
round(x0y0z0)

%% Optim
p0 = [gam1;u0v0_1;om1;x0y0z0];
oFcn = @(p)DLT.objFcnNoSkew(p,X,uvMan);
[~,d2sum0,~] = oFcn(p0)
pOpt = p0;

%% Run
opts = optimset;
opts.MaxFunEvals = 1e6;
opts.MaxIter = 5e5;
opts.Display = 'iter';
opts.TolFun = 1e-7;
[pOpt,resnorm,res] = lsqnonlin(oFcn,pOpt,[],[],opts);

%% Viz
%e1 = load('DLTs_17_18_16_fly_359_to_373.mat')
e = load('exampleDLTstuff_359_373.mat');
tgpl = e.threedGridPointsList;
faces3d = {tgpl.face1 tgpl.face2 tgpl.face3};
COLORS = {[1 0 0] [0 0 1] [0 1 0]};

%%
uvMan1 = e0.twodGridPointsList.allVertices.view1;
ignore = e0.twodGridPointsList.view1.ignore;
uvMan1(ignore,:) = [];
LSHOW = L1_rigidOpt;

mr(1) = MovieReader; mr(1).open('C002H001S0001_c.avi');
im1 = mr(1).readframe(1);
im1 = im1+25;
hFig = figure;
ax = axes;
imagesc(ax,im1);
hold(ax(1),'on');
axis(ax(1),'ij','image');

GAMMA = .4;
mgray = gray(256);
mgray2 = imadjust(mgray,[],[],GAMMA);
arrayfun(@(x)colormap(x,mgray2),ax);

for iFace=[1 3]
  X = faces3d{iFace}; 
  [uFace,vFace] = dlt_3D_to_2D(LSHOW,X(:,1),X(:,2),X(:,3));
  h(iFace) = plot(ax,uFace,vFace,'x','Color',COLORS{iFace},'MarkerSize',10);
  
  if iFace==1
    plot(ax,uvMan1(:,1),uvMan1(:,2),'.','Color',[1 1 0],'MarkerSize',10);
  end
end

%% residuals
e0 = load('DLTs_17_18_16_fly_359_to_373.mat');
e1 = load('exampleDLTstuff_359_373.mat');
X = e1.threedGridPointsList.allVertices;
N = size(X,1);

uv_v1_orig = e0.twodGridPointsList.allVertices.view2;
ig_v1_orig = e0.twodGridPointsList.view2.ignore;
tfcmp_orig = true(N,1);
tfcmp_orig(ig_v1_orig) = false;
%uv_v1_mine = uvMan;
%tfcmp_mine = false(N,1);
%tfcmp_mine(iptGlobal,:) = true;

szassert(X,[N 3]);
szassert(uv_v1_orig,[N 2]);
%szassert(uv_v1_mine,[N 2]);

L0 = e1.DLT_2;
Lopt = load('pOpt_0127_view2.mat','Lopt');
Lopt = Lopt.Lopt;
% Lopt2 = load('pOpt_0127_A.mat','Lopt');
% Lopt2 = Lopt2.Lopt;

Xcmp_orig = X(tfcmp_orig,:);
uv_v1_orig_cmp = uv_v1_orig(tfcmp_orig,:);
[uv_v1_orig_re_opt(:,1),uv_v1_orig_re_opt(:,2)] = ...
  dlt_3D_to_2D(Lopt,Xcmp_orig(:,1),Xcmp_orig(:,2),Xcmp_orig(:,3));
d_v1_orig_opt = mean(sqrt(sum((uv_v1_orig_cmp-uv_v1_orig_re_opt).^2,2)),1);
% [uv_v1_orig_re_opt2(:,1),uv_v1_orig_re_opt2(:,2)] = ...
%   dlt_3D_to_2D(Lopt2,Xcmp_orig(:,1),Xcmp_orig(:,2),Xcmp_orig(:,3));
% % d_v1_orig_opt2 = mean(sqrt(sum((uv_v1_orig_cmp-uv_v1_orig_re_opt2).^2,2)),1);
[uv_v1_orig_re(:,1),uv_v1_orig_re(:,2)] = ...
  dlt_3D_to_2D(L0,Xcmp_orig(:,1),Xcmp_orig(:,2),Xcmp_orig(:,3));
d_v1_orig = mean(sqrt(sum((uv_v1_orig_cmp-uv_v1_orig_re).^2,2)),1);

Xcmp_mine = X(iptGlobal,:);
uv_v1_mine_cmp = uvMan;
[uv_v1_mine_re_opt(:,1),uv_v1_mine_re_opt(:,2)] = ...
  dlt_3D_to_2D(Lopt,Xcmp_mine(:,1),Xcmp_mine(:,2),Xcmp_mine(:,3));
d_v1_mine_opt = mean(sqrt(sum((uv_v1_mine_cmp-uv_v1_mine_re_opt).^2,2)),1);
[uv_v1_mine_re_opt2(:,1),uv_v1_mine_re_opt2(:,2)] = ...
  dlt_3D_to_2D(Lopt2,Xcmp_mine(:,1),Xcmp_mine(:,2),Xcmp_mine(:,3));
% d_v1_mine_opt2 = mean(sqrt(sum((uv_v1_mine_cmp-uv_v1_mine_re_opt2).^2,2)),1);
% [uv_v1_mine_re(:,1),uv_v1_mine_re(:,2)] = ...
%   dlt_3D_to_2D(L0,Xcmp_mine(:,1),Xcmp_mine(:,2),Xcmp_mine(:,3));
% d_v1_mine = mean(sqrt(sum((uv_v1_mine_cmp-uv_v1_mine_re).^2,2)),1);
% 

%% PREPARE OPTIM DATA

% cal
e0 = load('DLTs_17_18_16_fly_359_to_373.mat');
X = e0.threedGridPointsList.allVertices;
N = size(X,1);

uvCal1 = e0.twodGridPointsList.allVertices.view1;
ignoreVw1 = e0.twodGridPointsList.view1.ignore;
uvCal2 = e0.twodGridPointsList.allVertices.view2;
ignoreVw2 = e0.twodGridPointsList.view2.ignore;

tfuse1 = true(N,1);
tfuse1(ignoreVw1) = false;
tfuse2 = true(N,1);
tfuse2(ignoreVw2) = false;

szassert(X,[N 3]);
szassert(uvCal1,[N 2]);
szassert(uvCal2,[N 2]);

Xcal1 = X(tfuse1,:);
uvCal1 = uvCal1(tfuse1,:);
Xcal2 = X(tfuse2,:);
uvCal2 = uvCal2(tfuse2,:);

fprintf('Cal data: %d rows view1, %d rows view2.\n',nnz(tfuse1),nnz(tfuse2));

% load stroData
tbl = load('stroData20170130.mat');
tbl = tbl.tbl;
uvStro1 = tbl.xxyy(:,[1 3]);
uvStro2 = tbl.xxyy(:,[2 4]);

fprintf('Stro data: using %d rows.\n',size(tbl,1));

%oFcn
oFcn = @(zzP)DLT.objFcnStro(zzP,Xcal1,Xcal2,uvCal1,uvCal2,uvStro1,uvStro2);
oFcnL = @(zzP)DLT.objFcnStroL(zzP,Xcal1,Xcal2,uvCal1,uvCal2,uvStro1,uvStro2);
oFcnCore = @(zzL1,zzL2)DLT.objFcnStroCore(zzL1,zzL2,Xcal1,Xcal2,uvCal1,uvCal2,uvStro1,uvStro2);

save optDataManPlusStro20170131.mat Xcal1 Xcal2 uvCal1 uvCal2 uvStro1 uvStro2 tbl oFcn oFcnL oFcnCore;

%% PREPARE OPTIM DATA 2
allenlbls = load('pOpt_0127.mat');
Xcal1 = allenlbls.X;
uvCal1 = allenlbls.uvMan;
iptGlobal = allenlbls.iptGlobal;

%oFcn
oFcn = @(zzP)DLT.objFcnStro(zzP,Xcal1,Xcal2,uvCal1,uvCal2,uvStro1,uvStro2);
oFcnL = @(zzP)DLT.objFcnStroL(zzP,Xcal1,Xcal2,uvCal1,uvCal2,uvStro1,uvStro2);
oFcnCore = @(zzL1,zzL2)DLT.objFcnStroCore(zzL1,zzL2,Xcal1,Xcal2,uvCal1,uvCal2,uvStro1,uvStro2);

save optDataAllenVw1ManVw2PlusStro20170131.mat iptGlobal Xcal1 Xcal2 uvCal1 uvCal2 uvStro1 uvStro2 tbl oFcn oFcnL oFcnCore;

%% OPTIM STARTING PTS 
% manual cal
e0 = load('DLTs_17_18_16_fly_359_to_373.mat');
L1 = e0.view1_dlt_coeff;
L2 = e0.view2_dlt_coeff;
[gam1,~,u0v0_1,~,om1,x0y0z0_1] = DLT.dlt2cam(L1);
[gam2,~,u0v0_2,~,om2,x0y0z0_2] = DLT.dlt2cam(L2);
p20 = [gam1(:);u0v0_1(:);om1(:);x0y0z0_1(:);gam2(:);u0v0_2(:);om2(:);x0y0z0_2(:)];
p22 = [L1(:); L2(:)];
save optP0_man_20170131.mat L1 L2 p20 p22;

%% rigid
e1 = load('exampleDLTstuff_359_373.mat');

L1 = e1.DLT_1;
L2 = e1.DLT_2;
p22 = [L1(:); L2(:)];
save optP0_rigid_20170131.mat L1 L2 p22;

%% allen opt
tmp1 = load('pOpt_0127.mat');
tmp2 = load('pOpt_0127_view2.mat');
L1 = tmp1.Lopt;
L2 = tmp2.Lopt;
[gam1,~,u0v0_1,~,om1,x0y0z0_1] = DLT.dlt2cam(L1);
[gam2,~,u0v0_2,~,om2,x0y0z0_2] = DLT.dlt2cam(L2);
p20 = [gam1(:);u0v0_1(:);om1(:);x0y0z0_1(:);gam2(:);u0v0_2(:);om2(:);x0y0z0_2(:)];
p22 = [L1(:); L2(:)];
save optP0_allenOpt_20170131.mat L1 L2 p20 p22;

%% Run -- Data man+stro. P20. p0 manual cal.
p0 = load('optP0_man_20170131.mat');
oFcnInfo = load('optDataManPlusStro20170131.mat');
p0 = p0.p20;
oFcn = oFcnInfo.oFcn;

[res0.d,res0.d2Proj1,res0.d2Proj2,res0.d2ReProj1,res0.d2ReProj2] = oFcn(p0);

opts = optimoptions('lsqnonlin','MaxFunctionEvaluations',5e4,...
  'Display','iter','TolFun',1e-7,'StepTolerance',1e-7,'MaxIterations',5e3);

pOpt = p0;
pOpt = lsqnonlin(oFcn,pOpt,[],[],opts);

gamOpt1 = pOpt(1:2);
u0v0opt1 = pOpt(3:4);
omOpt1 = pOpt(5:7);
x0y0z0opt1 = pOpt(8:10);
gamOpt2 = pOpt(11:12);
u0v0opt2 = pOpt(13:14);
omOpt2 = pOpt(15:17);
x0y0z0opt2 = pOpt(18:20);
Lopt1 = DLT.cam2dlt(gamOpt1,0,u0v0opt1,omOpt1,x0y0z0opt1);
Lopt2 = DLT.cam2dlt(gamOpt2,0,u0v0opt2,omOpt2,x0y0z0opt2);

[res1.d,res1.d2Proj1,res1.d2Proj2,res1.d2ReProj1,res1.d2ReProj2] = oFcn(pOpt);
save optRes_dataManPlusStro_p0Man_20170131.mat oFcn p0 pOpt Lopt1 Lopt2 res0 res1;

%% Run -- Data man+stro. P22. p0 manual cal.
p0 = load('optP0_man_20170131.mat');
oFcnInfo = load('optDataManPlusStro20170131.mat');
p0 = p0.p22;
oFcn = oFcnInfo.oFcnL;

[res0.d,res0.d2Proj1,res0.d2Proj2,res0.d2ReProj1,res0.d2ReProj2] = oFcn(p0);

opts = optimoptions('lsqnonlin','MaxFunctionEvaluations',5e4,...
  'Display','iter','TolFun',1e-7,'StepTolerance',1e-7,'MaxIterations',5e3);

pOpt = p0;
pOpt = lsqnonlin(oFcn,pOpt,[],[],opts);

[res1.d,res1.d2Proj1,res1.d2Proj2,res1.d2ReProj1,res1.d2ReProj2] = oFcn(pOpt);

Lopt1 = pOpt(1:11);
Lopt2 = pOpt(12:end);
[detval1,normval1] = DLT.checkPhysicality(Lopt1)
[detval2,normval2] = DLT.checkPhysicality(Lopt2)
save optRes_dataManPlusStro22_p0Man_20170131.mat oFcn p0 pOpt Lopt1 Lopt2 res0 res1 detval1 normval1 detval2 normval2;

%% Run -- Data man+stro. P20. p0 allenOpt.
p0 = load('optP0_allenOpt_20170131.mat');
oFcnInfo = load('optDataManPlusStro20170131.mat');
p0 = p0.p20;
oFcn = oFcnInfo.oFcn;

[res0.d,res0.d2Proj1,res0.d2Proj2,res0.d2ReProj1,res0.d2ReProj2] = oFcn(p0);

opts = optimoptions('lsqnonlin','MaxFunctionEvaluations',5e4,...
  'Display','iter','TolFun',1e-7,'StepTolerance',1e-7,'MaxIterations',5e3);

pOpt = p0;
pOpt = lsqnonlin(oFcn,pOpt,[],[],opts);

gamOpt1 = pOpt(1:2);
u0v0opt1 = pOpt(3:4);
omOpt1 = pOpt(5:7);
x0y0z0opt1 = pOpt(8:10);
gamOpt2 = pOpt(11:12);
u0v0opt2 = pOpt(13:14);
omOpt2 = pOpt(15:17);
x0y0z0opt2 = pOpt(18:20);
Lopt1 = DLT.cam2dlt(gamOpt1,0,u0v0opt1,omOpt1,x0y0z0opt1);
Lopt2 = DLT.cam2dlt(gamOpt2,0,u0v0opt2,omOpt2,x0y0z0opt2);

[res1.d,res1.d2Proj1,res1.d2Proj2,res1.d2ReProj1,res1.d2ReProj2] = oFcn(pOpt);
save optRes_dataManPlusStro_p0allenOpt_20170131.mat oFcn p0 pOpt Lopt1 Lopt2 res0 res1;

%% Run -- Data man+stro. P22. p0AllenOpt.
p0 = load('optP0_allenOpt_20170131.mat');
oFcnInfo = load('optDataManPlusStro20170131.mat');
p0 = p0.p22;
oFcn = oFcnInfo.oFcnL;

[res0.d,res0.d2Proj1,res0.d2Proj2,res0.d2ReProj1,res0.d2ReProj2] = oFcn(p0);

opts = optimoptions('lsqnonlin','MaxFunctionEvaluations',5e4,...
  'Display','iter','TolFun',1e-7,'StepTolerance',1e-7,'MaxIterations',5e3);

pOpt = p0;
pOpt = lsqnonlin(oFcn,pOpt,[],[],opts);

[res1.d,res1.d2Proj1,res1.d2Proj2,res1.d2ReProj1,res1.d2ReProj2] = oFcn(pOpt);

Lopt1 = pOpt(1:11);
Lopt2 = pOpt(12:end);
[detval1,normval1] = DLT.checkPhysicality(Lopt1)
[detval2,normval2] = DLT.checkPhysicality(Lopt2)
save optRes_dataManPlusStro22_p0allenOpt_20170131.mat oFcn p0 pOpt Lopt1 Lopt2 res0 res1 detval1 normval1 detval2 normval2;

%% Run -- Data man+stro. P22. p0 rigid.
p0 = load('optP0_rigid_20170131.mat');
oFcnInfo = load('optDataManPlusStro20170131.mat');
p0 = p0.p22;
oFcn = oFcnInfo.oFcnL;

[res0.d,res0.d2Proj1,res0.d2Proj2,res0.d2ReProj1,res0.d2ReProj2] = oFcn(p0);

opts = optimoptions('lsqnonlin','MaxFunctionEvaluations',5e4,...
  'Display','iter','TolFun',1e-7,'StepTolerance',1e-7,'MaxIterations',5e3);

pOpt = p0;
pOpt = lsqnonlin(oFcn,pOpt,[],[],opts);

Lopt1 = pOpt(1:11);
Lopt2 = pOpt(12:end);
[detval1,normval1] = DLT.checkPhysicality(Lopt1)
[detval2,normval2] = DLT.checkPhysicality(Lopt2)

[res1.d,res1.d2Proj1,res1.d2Proj2,res1.d2ReProj1,res1.d2ReProj2] = oFcn(pOpt);
save optRes_dataManPlusStro_p22_p0rigid_20170131.mat oFcn p0 pOpt Lopt1 Lopt2 res0 res1 detval1 normval1 detval2 normval2;




%% Run -- Data allen+man+stro. P20. p0 manual cal.
p0 = load('optP0_man_20170131.mat');
oFcnInfo = load('optDataAllenVw1ManVw2PlusStro20170131.mat');
p0 = p0.p20;
oFcn = oFcnInfo.oFcn;

[res0.d,res0.d2Proj1,res0.d2Proj2,res0.d2ReProj1,res0.d2ReProj2] = oFcn(p0);

opts = optimoptions('lsqnonlin','MaxFunctionEvaluations',5e4,...
  'Display','iter','TolFun',1e-7,'StepTolerance',1e-7,'MaxIterations',5e3);

pOpt = p0;
pOpt = lsqnonlin(oFcn,pOpt,[],[],opts);

gamOpt1 = pOpt(1:2);
u0v0opt1 = pOpt(3:4);
omOpt1 = pOpt(5:7);
x0y0z0opt1 = pOpt(8:10);
gamOpt2 = pOpt(11:12);
u0v0opt2 = pOpt(13:14);
omOpt2 = pOpt(15:17);
x0y0z0opt2 = pOpt(18:20);
Lopt1 = DLT.cam2dlt(gamOpt1,0,u0v0opt1,omOpt1,x0y0z0opt1);
Lopt2 = DLT.cam2dlt(gamOpt2,0,u0v0opt2,omOpt2,x0y0z0opt2);

[res1.d,res1.d2Proj1,res1.d2Proj2,res1.d2ReProj1,res1.d2ReProj2] = oFcn(pOpt);
save optRes_dataAllenPlusManPlusStro_p0Man_20170131.mat oFcn p0 pOpt Lopt1 Lopt2 res0 res1;

%% Run -- Data allen+man+stro. P22. p0 manual cal.
p0 = load('optP0_man_20170131.mat');
oFcnInfo = load('optDataAllenVw1ManVw2PlusStro20170131.mat');
p0 = p0.p22;
oFcn = oFcnInfo.oFcnL;

[res0.d,res0.d2Proj1,res0.d2Proj2,res0.d2ReProj1,res0.d2ReProj2] = oFcn(p0);

opts = optimoptions('lsqnonlin','MaxFunctionEvaluations',5e4,...
  'Display','iter','TolFun',1e-7,'StepTolerance',1e-7,'MaxIterations',5e3);

pOpt = p0;
pOpt = lsqnonlin(oFcn,pOpt,[],[],opts);

[res1.d,res1.d2Proj1,res1.d2Proj2,res1.d2ReProj1,res1.d2ReProj2] = oFcn(pOpt);

Lopt1 = pOpt(1:11);
Lopt2 = pOpt(12:end);
[detval1,normval1] = DLT.checkPhysicality(Lopt1)
[detval2,normval2] = DLT.checkPhysicality(Lopt2)
save optRes_dataAllenPlusManPlusStro_p22_p0Man_20170131.mat oFcn p0 pOpt Lopt1 Lopt2 res0 res1 detval1 normval1 detval2 normval2;


%% Run -- Data allen+man+stro. P22. p0rigid
p0 = load('optP0_rigid_20170131.mat');
oFcnInfo = load('optDataAllenVw1ManVw2PlusStro20170131.mat');
p0 = p0.p22;
oFcn = oFcnInfo.oFcnL;

[res0.d,res0.d2Proj1,res0.d2Proj2,res0.d2ReProj1,res0.d2ReProj2] = oFcn(p0);

opts = optimoptions('lsqnonlin','MaxFunctionEvaluations',5e4,...
  'Display','iter','TolFun',1e-7,'StepTolerance',1e-7,'MaxIterations',5e3);

pOpt = p0;
pOpt = lsqnonlin(oFcn,pOpt,[],[],opts);

[res1.d,res1.d2Proj1,res1.d2Proj2,res1.d2ReProj1,res1.d2ReProj2] = oFcn(pOpt);

Lopt1 = pOpt(1:11);
Lopt2 = pOpt(12:end);
[detval1,normval1] = DLT.checkPhysicality(Lopt1)
[detval2,normval2] = DLT.checkPhysicality(Lopt2)

save optRes_dataAllenPlusManPlusStro_p22_p0rigid_20170131.mat oFcn p0 pOpt Lopt1 Lopt2 res0 res1 detval1 normval1 detval2 normval2;

%% Run -- Data allen+man+stro. P20. p0 allenOpt.
p0 = load('optP0_allenOpt_20170131.mat');
oFcnInfo = load('optDataAllenVw1ManVw2PlusStro20170131.mat');
p0 = p0.p20;
oFcn = oFcnInfo.oFcn;

[res0.d,res0.d2Proj1,res0.d2Proj2,res0.d2ReProj1,res0.d2ReProj2] = oFcn(p0);

opts = optimoptions('lsqnonlin','MaxFunctionEvaluations',5e4,...
  'Display','iter','TolFun',1e-7,'StepTolerance',1e-7,'MaxIterations',5e3);

pOpt = p0;
pOpt = lsqnonlin(oFcn,pOpt,[],[],opts);

gamOpt1 = pOpt(1:2);
u0v0opt1 = pOpt(3:4);
omOpt1 = pOpt(5:7);
x0y0z0opt1 = pOpt(8:10);
gamOpt2 = pOpt(11:12);
u0v0opt2 = pOpt(13:14);
omOpt2 = pOpt(15:17);
x0y0z0opt2 = pOpt(18:20);
Lopt1 = DLT.cam2dlt(gamOpt1,0,u0v0opt1,omOpt1,x0y0z0opt1);
Lopt2 = DLT.cam2dlt(gamOpt2,0,u0v0opt2,omOpt2,x0y0z0opt2);

[res1.d,res1.d2Proj1,res1.d2Proj2,res1.d2ReProj1,res1.d2ReProj2] = oFcn(pOpt);
save optRes_dataAllenPlusManPlusStro_p20_p0AllenOpt_20170131.mat oFcn p0 pOpt Lopt1 Lopt2 res0 res1;

%% Run -- Data allen+man+stro. P22. p0allenOpt
p0 = load('optP0_allenOpt_20170131.mat');
oFcnInfo = load('optDataAllenVw1ManVw2PlusStro20170131.mat');
p0 = p0.p22;
oFcn = oFcnInfo.oFcnL;

[res0.d,res0.d2Proj1,res0.d2Proj2,res0.d2ReProj1,res0.d2ReProj2] = oFcn(p0);

opts = optimoptions('lsqnonlin','MaxFunctionEvaluations',5e4,...
  'Display','iter','TolFun',1e-7,'StepTolerance',1e-7,'MaxIterations',5e3);

pOpt = p0;
pOpt = lsqnonlin(oFcn,pOpt,[],[],opts);

[res1.d,res1.d2Proj1,res1.d2Proj2,res1.d2ReProj1,res1.d2ReProj2] = oFcn(pOpt);

Lopt1 = pOpt(1:11);
Lopt2 = pOpt(12:end);
[detval1,normval1] = DLT.checkPhysicality(Lopt1)
[detval2,normval2] = DLT.checkPhysicality(Lopt2)

save optRes_dataAllenPlusManPlusStro_p22_p0allenOpt_20170131.mat oFcn p0 pOpt Lopt1 Lopt2 res0 res1 detval1 normval1 detval2 normval2;

%% Run -- Data allen Cam 1. Skew P22. p0 allenOpt.

oData = load('optDataAllenVw1ManVw2PlusStro20170131.mat');
oFcn = @(zzP)DLT.objFcnSkew(zzP,oData.Xcal1,oData.uvCal1);
oFcn2 = @(zzP)DLT.objFcnSkew2(zzP,oData.Xcal1,oData.uvCal1);
p0 = load('optP0_allenOpt_20170131.mat');
% p0.p0 is in terms of x0y0z0 etc
[gam,s,u0v0,~,om,~,t] = DLT.dlt2cam(p0.L1);
p0 = [gam;s;u0v0;om;t]; 
      
%[res0.d,res0.d2sum] = oFcn(p0);

opts1 = optimoptions('lsqnonlin','MaxFunctionEvaluations',2e5,...
  'Display','iter','TolFun',1e-7,'StepTolerance',1e-7,'MaxIterations',2e4);

pOpt = p0;
pOpt = lsqnonlin(oFcn,pOpt,[],[],opts1);

gamOpt = pOpt(1:2);
sOpt = pOpt(3);
u0v0opt = pOpt(4:5);
omOpt = pOpt(6:8);
tOpt = pOpt(9:11);

[res1.d,res1.d2sum] = oFcn(pOpt);
save optRes_cam1_dataAllen_skew22_p0AllenOpt_20170202.mat oFcn oFcn2 p0 pOpt;

% viz
[~,~,uvP,uvRef] = oFcn(pOpt);
DLT.vizPorRP(uvP,uvRef);

[~,~,uvP,uvRef] = oFcn(p0);
DLT.vizPorRP(uvP,uvRef);

%% Run -- Data man Cam 2. Skew P22. p0 allenOpt.

oData = load('optDataAllenVw1ManVw2PlusStro20170131.mat');
oFcn = @(zzP)DLT.objFcnSkew(zzP,oData.Xcal2,oData.uvCal2);
oFcn2 = @(zzP)DLT.objFcnSkew2(zzP,oData.Xcal2,oData.uvCal2);
p0 = load('optP0_allenOpt_20170131.mat');
% p0.p0 is in terms of x0y0z0 etc
[gam,s,u0v0,R,om,~,t] = DLT.dlt2cam(p0.L2);
p0 = [gam;s;u0v0;om;t]; 
      
%[res0.d,res0.d2sum] = oFcn(p0);

opts = optimoptions('lsqnonlin','MaxFunctionEvaluations',2e5,...
  'Display','iter','TolFun',1e-7,'StepTolerance',1e-7,'MaxIterations',2e4);

pOpt = p0;
pOpt = lsqnonlin(oFcn,pOpt,[],[],opts);

gamOpt = pOpt(1:2);
sOpt = pOpt(3);
u0v0opt = pOpt(4:5);
omOpt = pOpt(6:8);
tOpt = pOpt(9:11);

[res1.d,res1.d2sum] = oFcn(pOpt);
save optRes_cam2_dataAllen_skew22_p0AllenOpt_20170202.mat oFcn oFcn2 p0 pOpt;

[~,~,uvP,uvRef] = oFcn(pOpt);
DLT.vizPorRP(2,uvP,uvRef);

[~,~,uvP,uvRef] = oFcn(p0);
DLT.vizPorRP(2,uvP,uvRef);

%% Run -- Data allen Cam 1. NoPP (p8). p0 allenOpt.

u0v0_force = [768/2 512/2]; % cam1 images are 512rows x 768cols

oData = load('optDataAllenVw1ManVw2PlusStro20170131.mat');
oFcn = @(zzP)DLT.objFcnNoPP(zzP,u0v0_force,oData.Xcal1,oData.uvCal1);
%p0 = load('optP0_allenOpt_20170131.mat');
p0 = load('optP0_man_20170131.mat');
% p0.p0 is in terms of x0y0z0 etc
[gam,~,~,~,om,~,t] = DLT.dlt2cam(p0.L1);
p0 = [gam;om;t]; 
      
%[res0.d,res0.d2sum] = oFcn(p0);

opts1 = optimoptions('lsqnonlin','MaxFunctionEvaluations',2e5,...
  'Display','iter','TolFun',1e-7,'StepTolerance',1e-7,'MaxIterations',2e4);

pOpt = p0;
pOpt = lsqnonlin(oFcn,pOpt,[],[],opts1);

gamOpt = pOpt(1:2);
sOpt = pOpt(3);
u0v0opt = pOpt(4:5);
omOpt = pOpt(6:8);
tOpt = pOpt(9:11);

[res1.d,res1.d2sum] = oFcn(pOpt);
save optRes_cam1_dataAllen_skew22_p0AllenOpt_20170202.mat oFcn oFcn2 p0 pOpt;

% viz
[~,~,uvP,uvRef] = oFcn(pOpt);
DLT.vizPorRP(1,uvP,uvRef);

[~,~,uvP,uvRef] = oFcn(p0);
DLT.vizPorRP(1,uvP,uvRef);

%% Run -- Data allen Cam 1. SkewKC4 (P15). p0 allenOpt.

oData = load('optDataAllenVw1ManVw2PlusStro20170131.mat');
oFcn = @(zzP)DLT.objFcnSkewKC4(zzP,oData.Xcal1,oData.uvCal1);
oFcn0 = @(zzP)DLT.objFcnSkew(zzP,oData.Xcal1,oData.uvCal1);
p0 = load('optP0_allenOpt_20170131.mat');
% p0.p0 is in terms of x0y0z0 etc
[gam,s,u0v0,~,om,~,t] = DLT.dlt2cam(p0.L1);
kc4 = zeros(4,1);
p0 = [gam;s;u0v0;om;t;kc4]; 
      
%[res0.d,res0.d2sum] = oFcn(p0);

opts1 = optimoptions('lsqnonlin','MaxFunctionEvaluations',2e5,...
  'Display','iter','TolFun',1e-7,'StepTolerance',1e-7,'MaxIterations',2e4);

pOpt = p0;
pOpt = lsqnonlin(oFcn,pOpt,[],[],opts1);

gamOpt = pOpt(1:2);
sOpt = pOpt(3);
u0v0opt = pOpt(4:5);
omOpt = pOpt(6:8);
tOpt = pOpt(9:11);

[res1.d,res1.d2sum] = oFcn(pOpt);
save optRes_cam1_dataAllen_skewKC4_p0AllenOpt_20170202.mat oFcn p0 pOpt;

% viz
[~,~,uvP,uvRef] = oFcn(pOpt);
DLT.vizPorRP(1,uvP,uvRef);

[~,~,uvP,uvRef] = oFcn(p0);
DLT.vizPorRP(1,uvP,uvRef);

[~,~,uvP,uvRef] = oFcn0(q.p0);
DLT.vizPorRP(1,uvP,uvRef);

%% Run -- Data man Cam 1. SkewKC4 (P15). p0 allenOpt.

oData = load('optDataManPlusStro20170131.mat');
oFcn = @(zzP)DLT.objFcnSkewKC4(zzP,oData.Xcal1,oData.uvCal1);
p0 = load('optP0_allenOpt_20170131.mat');
% p0.p0 is in terms of x0y0z0 etc
[gam,s,u0v0,R,om,~,t] = DLT.dlt2cam(p0.L1);
kc4 = zeros(4,1);
p0 = [gam;s;u0v0;om;t;kc4]; 
      
%[res0.d,res0.d2sum] = oFcn(p0);

opts1 = optimoptions('lsqnonlin','MaxFunctionEvaluations',2e5,...
  'Display','iter','TolFun',1e-7,'StepTolerance',1e-7,'MaxIterations',2e4);

pOpt = p0;
pOpt = lsqnonlin(oFcn,pOpt,[],[],opts1);

% [res1.d,res1.d2sum] = oFcn(pOpt);
% save optRes_cam1_dataAllen_skewKC4_p0AllenOpt_20170202.mat oFcn p0 pOpt;

% viz
[~,~,uvP,uvRef] = oFcn(pOpt);
DLT.vizPorRP(1,uvP,uvRef);

[~,~,uvP,uvRef] = oFcn(p0);
DLT.vizPorRP(1,uvP,uvRef);

[~,~,uvP,uvRef] = oFcn0(q.p0);
DLT.vizPorRP(1,uvP,uvRef);


%% Run -- Data allen Cam 2. SkewKC4 (P15). p0 allenOpt.

oData = load('optDataAllenVw1ManVw2PlusStro20170131.mat');
oFcn = @(zzP)DLT.objFcnSkewKC4(zzP,oData.Xcal2,oData.uvCal2);
p0 = load('optP0_allenOpt_20170131.mat');
% p0.p0 is in terms of x0y0z0 etc
[gam,s,u0v0,~,om,~,t] = DLT.dlt2cam(p0.L2);
kc4 = zeros(4,1);
p0 = [gam;s;u0v0;om;t;kc4]; 
      
%[res0.d,res0.d2sum] = oFcn(p0);

opts1 = optimoptions('lsqnonlin','MaxFunctionEvaluations',2e4,...
  'Display','iter','TolFun',1e-7,'StepTolerance',1e-10,'MaxIterations',5e3);

pOpt = p0;
pOpt = lsqnonlin(oFcn,pOpt,[],[],opts1);

gamOpt = pOpt(1:2);
sOpt = pOpt(3);
u0v0opt = pOpt(4:5);
omOpt = pOpt(6:8);
tOpt = pOpt(9:11);

[res1.d,res1.d2sum] = oFcn(pOpt);
save optRes_cam1_dataAllen_skewKC4_p0AllenOpt_20170202.mat oFcn p0 pOpt;

% viz
[~,~,uvP,uvRef] = oFcn(pOpt);
DLT.vizPorRP(1,uvP,uvRef);

[~,~,uvP,uvRef] = oFcn(p0);
DLT.vizPorRP(1,uvP,uvRef);

[~,~,uvP,uvRef] = oFcn0(q.p0);
DLT.vizPorRP(1,uvP,uvRef);



%% Run -- Data allenv1+manv2+stro (inc new stro data). skew + 4th order radial. p0 allen cam1/cam2 optim


% prep data: include new stro data
oFcnInfo = load('optDataAllenVw1ManVw2PlusStro20170131.mat');
tbl = load('stroData_4lbls_20170202.mat');
tbl = tbl.tbl;
uvStro1 = tbl.xxyy(:,[1 3]);
uvStro2 = tbl.xxyy(:,[2 4]);
fprintf('Stro data: using %d rows.\n',size(uvStro1,1));

oFcn = @(zzP)DLT.objFcnStroSkewKC4(zzP,...
  oFcnInfo.Xcal1,oFcnInfo.Xcal2,...
  oFcnInfo.uvCal1,oFcnInfo.uvCal2,uvStro1,uvStro2);

% p0
p0 = load('optRes_dataAllenPlusManPlusStro_p20_p0allenOpt_20170131.mat');
p01 = p0.pOpt(1:10);
p02 = p0.pOpt(11:20);

gam1 = p01(1:2);
s1 = 0;
u0v0_1 = p01(3:4);
om1 = p01(5:7);
x0y0z0_1 = p01(8:10);
R1 = rodrigues(om1);
t1 = -R1*x0y0z0_1;
kc4_1 = zeros(4,1);

gam2 = p02(1:2);
s2 = 0;
u0v0_2 = p02(3:4);
om2 = p02(5:7);
x0y0z0_2 = p02(8:10);
R2 = rodrigues(om2);
t2 = -R2*x0y0z0_2;
kc4_2 = zeros(4,1);

p0 = [gam1;s1;u0v0_1;om1;t1;kc4_1; ...
      gam2;s2;u0v0_2;om2;t2;kc4_2];

    
RIGHT NOW THIS IS WRONG B/C STEREO_TRI IS NOT ACCOUNTING FOR DISTORTIONS
%   gam1 = p(1:2);
%   s1 = p(3);
%   u0v0_1 = p(4:5);
%   om1 = p(6:8);
%   t1 = p(9:11);
%   kc1 = [p(12:15); 0];
%   gam2 = p(16:17);
%   s2 = p(18);
%   u0v0_2 = p(19:20);
%   om2 = p(21:23);
%   t2 = p(24:26);
%   kc2 = [p(27:30); 0];


opts = optimoptions('lsqnonlin','MaxFunctionEvaluations',1e4,...
  'Display','iter','TolFun',1e-7,'StepTolerance',1e-11,'MaxIterations',170);

pOpt = p0;
pOpt = lsqnonlin(oFcn,pOpt,[],[],opts);

gamOpt1 = pOpt(1:2);
u0v0opt1 = pOpt(3:4);
omOpt1 = pOpt(5:7);
x0y0z0opt1 = pOpt(8:10);
gamOpt2 = pOpt(11:12);
u0v0opt2 = pOpt(13:14);
omOpt2 = pOpt(15:17);
x0y0z0opt2 = pOpt(18:20);
Lopt1 = DLT.cam2dlt(gamOpt1,0,u0v0opt1,omOpt1,x0y0z0opt1);
Lopt2 = DLT.cam2dlt(gamOpt2,0,u0v0opt2,omOpt2,x0y0z0opt2);

[res1.d,res1.d2Proj1,res1.d2Proj2,res1.d2ReProj1,res1.d2ReProj2] = oFcn(pOpt);
save optRes_dataAllenPlusManPlusStro_p20_p0AllenOpt_20170131.mat oFcn p0 pOpt Lopt1 Lopt2 res0 res1;






%% Run -- skew + 4th order radial no principal pt, allenv1+manv2+stro (inc new stro data), p0 allen cam1/cam2 optim


%% Summarize

% find best: {dataMan, dataAllen} x {p20 (constrained/physical), p22} 

rmsfcn = @(x)sqrt(sum(x)/numel(x));

% dataMan + p20
OPTRESFILES = {
  'dMan_p20_p0al' 'optRes_dataManPlusStro_p20_p0allenOpt_20170131.mat'
  'dMan_p20_p0mn' 'optRes_dataManPlusStro_p20_p0mancal_20170131.mat'
  'dMan_p22_p0al' 'optRes_dataManPlusStro_p22_p0allenOpt_20170131.mat'
  'dMan_p22_p0mn' 'optRes_dataManPlusStro_p22_p0mancal_20170131.mat'
  'dMan_p22_p0rd' 'optRes_dataManPlusStro_p22_p0rigid_20170131.mat'};
s = struct();
for iFile=1:size(OPTRESFILES,1)
  q = load(OPTRESFILES{iFile,2});
  fld0 = [OPTRESFILES{iFile,1} '_res0'];
  fld1 = [OPTRESFILES{iFile,1} '_res1'];  
  s.(fld0) = [ rmsfcn(q.res0.d2Proj1); rmsfcn(q.res0.d2Proj2); rmsfcn(q.res0.d2ReProj1); rmsfcn(q.res0.d2ReProj2) ];
  s.(fld1) = [ rmsfcn(q.res1.d2Proj1); rmsfcn(q.res1.d2Proj2); rmsfcn(q.res1.d2ReProj1); rmsfcn(q.res1.d2ReProj2) ];
end
tbl_dMan = struct2table(s);

% dataAllen
OPTRESFILES = {
  'dAl_p20_p0al' 'optRes_dataAllenPlusManPlusStro_p20_p0allenOpt_20170131.mat'
  'dAl_p20_p0mn' 'optRes_dataAllenPlusManPlusStro_p20_p0mancal_20170131.mat'
  'dAl_p22_p0al' 'optRes_dataAllenPlusManPlusStro_p22_p0allenOpt_20170131.mat'
  'dAl_p22_p0mn' 'optRes_dataAllenPlusManPlusStro_p22_p0mancal_20170131.mat'
  'dAl_p22_p0rd' 'optRes_dataAllenPlusManPlusStro_p22_p0rigid_20170131.mat'};

s = struct();
for iFile=1:size(OPTRESFILES,1)
  q = load(OPTRESFILES{iFile,2});
  fld0 = [OPTRESFILES{iFile,1} '_r0'];
  fld1 = [OPTRESFILES{iFile,1} '_r1'];
  s.(fld0) = [ rmsfcn(q.res0.d2Proj1); rmsfcn(q.res0.d2Proj2); rmsfcn(q.res0.d2ReProj1); rmsfcn(q.res0.d2ReProj2) ];
  s.(fld1) = [ rmsfcn(q.res1.d2Proj1); rmsfcn(q.res1.d2Proj2); rmsfcn(q.res1.d2ReProj1); rmsfcn(q.res1.d2ReProj2) ];
end
tbl_dAl = struct2table(s);


%% Viz
mr(1) = MovieReader; mr(1).open('C001H001S0001_c.avi');
mr(2) = MovieReader; mr(2).open('C002H001S0001_c.avi');
im1 = mr(1).readframe(1);
im1 = im1+25;
im2 = mr(2).readframe(1);
im2 = im2+25;
ims = {im1 im2};

hFig = figure('units','normalized','outerposition',[0 0 1 1]);
axs = createsubplots(2,2);
axs = reshape(axs,[2 2]);
GAMMA = .4;
mgray = gray(256);
mgray2 = imadjust(mgray,[],[],GAMMA);

hProj = gobjects(2,2); % projected/reprojected points
hRef = gobjects(2,2); % reference/comparison points

for i=1:2
for j=1:2
  ax = axs(i,j);
  imagesc(ax,ims{j});
  hold(ax,'on');
  axis(ax,'ij','image');
  colormap(ax,mgray2);

  hProj(i,j) = plot(ax,nan,nan,'.','Color',[1 1 0],'MarkerSize',14);
  hRef(i,j) = plot(ax,nan,nan,'+','Color',[1 .5 .33],'MarkerSize',7,'linewidth',2);
  
%   if j==2
    ax.YTickLabel = [];
%   end
%   if iFile==2
    ax.XTickLabel = [];
%   end
end
end

OPTRESFILES = {
  'dMan_p20_p0al' 'optRes_dataManPlusStro_p20_p0allenOpt_20170131.mat'
  'dMan_p20_p0mn' 'optRes_dataManPlusStro_p20_p0mancal_20170131.mat'
  'dMan_p22_p0al' 'optRes_dataManPlusStro_p22_p0allenOpt_20170131.mat'
  'dMan_p22_p0mn' 'optRes_dataManPlusStro_p22_p0mancal_20170131.mat'
  'dMan_p22_p0rd' 'optRes_dataManPlusStro_p22_p0rigid_20170131.mat'
  'dAl_p20_p0al' 'optRes_dataAllenPlusManPlusStro_p20_p0allenOpt_20170131.mat'
  'dAl_p20_p0mn' 'optRes_dataAllenPlusManPlusStro_p20_p0mancal_20170131.mat'
  'dAl_p22_p0al' 'optRes_dataAllenPlusManPlusStro_p22_p0allenOpt_20170131.mat'
  'dAl_p22_p0mn' 'optRes_dataAllenPlusManPlusStro_p22_p0mancal_20170131.mat'
  'dAl_p22_p0rd' 'optRes_dataAllenPlusManPlusStro_p22_p0rigid_20170131.mat'};
STRODECIMATION = .25;
for iFile=1:size(OPTRESFILES,1)
  id = OPTRESFILES{iFile,1};
  ores = load(OPTRESFILES{iFile,2});
  
  oFcn = ores.oFcn;  
  [~,~,~,~,~,...
    uvCal1,uvCalProj1,...
    uvCal2,uvCalProj2,...
    uvStro1,uvStro2,uvReProj1,uvReProj2] = oFcn(ores.p0);
  dataProj = {uvCalProj1 uvCalProj2;uvReProj1 uvReProj2};  
  dataRef = {uvCal1 uvCal2;uvStro1 uvStro2};
  for i=1:2
  for j=1:2
    dproj = dataProj{i,j};
    dref = dataRef{i,j};
    n = size(dref,1);
    
    if i==2
      %idx = randsample(n,round(n*STRODECIMATION));
      [~,~,idx] = furthestfirst(dref,round(n*STRODECIMATION),'start',[]);
      
      set(hProj(i,j),'XData',dproj(idx,1),'YData',dproj(idx,2));
      set(hRef(i,j),'XData',dref(idx,1),'YData',dref(idx,2));
    else
      set(hProj(i,j),'XData',dproj(:,1),'YData',dproj(:,2));
      set(hRef(i,j),'XData',dref(:,1),'YData',dref(:,2));
    end
    
    rmsval = sqrt(sum(sum((dref-dproj).^2,2))/n);
    tstr = sprintf('rms: %.3f',rmsval);
    title(axs(i,j),tstr,'fontweight','bold','interpreter','none');    
  end
  end  
  
  fname = sprintf('%s_cmp_r0.png',id);
  print(hFig,'-dpng',fname);
  fprintf('Printed %s...\n',fname);
end
  
%% Viz Stro RP
MOVS = {
  'f:\stephen\data\flp-chrimson_experiments\fly_359_to_373_17_18_16_SS00325norpAmalesFlpChrimson\fly371\C001H001S0020\C001H001S0020_c.avi'
  'f:\stephen\data\flp-chrimson_experiments\fly_359_to_373_17_18_16_SS00325norpAmalesFlpChrimson\fly371\C002H001S0020\C002H001S0020_c.avi'
  };
LBLTOUSE = 'fly371.lbl';
tblUse = tbl(strcmp(tbl.lbl,LBLTOUSE),:);
%tblUse = tbl(1:end-6,:);
mr(1) = MovieReader; mr(1).open(MOVS{1});
mr(2) = MovieReader; mr(2).open(MOVS{2});
im1 = mr(1).readframe(1);
im1 = im1+25;
im2 = mr(2).readframe(1);
im2 = im2+25;
ims = {im1 im2};

hFig = figure('units','normalized','outerposition',[0 0 1 1]);
axs = createsubplots(1,2);
axs = reshape(axs,[1 2]);
GAMMA = .4;
mgray = gray(256);
mgray2 = imadjust(mgray,[],[],GAMMA);

hProj = gobjects(1,2); % projected/reprojected points
hRef = gobjects(1,2); % reference/comparison points

for j=1:2
  ax = axs(1,j);
  imagesc(ax,ims{j});
  hold(ax,'on');
  axis(ax,'ij','image');
  colormap(ax,mgray2);

  hProj(1,j) = plot(ax,nan,nan,'.','Color',[0 1 1],'MarkerSize',14);
  hRef(1,j) = plot(ax,nan,nan,'+','Color',[1 .5 .33],'MarkerSize',7,'linewidth',2);
  
  ax.YTickLabel = [];
  ax.XTickLabel = [];
end

crig = lbl.viewCalibrationData{1};
L1 = crig.kineData.cal.coeff.DLT_1;
L2 = crig.kineData.cal.coeff.DLT_2;
nrow = size(tblUse,1);
uv1 = tblUse.xxyy(:,[1 3]);
uv2 = tblUse.xxyy(:,[2 4]);
uv1RP = nan(nrow,2);
uv2RP = nan(nrow,2);
for i=1:nrow
  xyz = DLT.stereoTriangulate2(uv1(i,:)',L1,uv2(i,:)',L2);
  [uv1RP(i,1),uv1RP(i,2)] = dlt_3D_to_2D(L1,xyz(1),xyz(2),xyz(3));
  [uv2RP(i,1),uv2RP(i,2)] = dlt_3D_to_2D(L2,xyz(1),xyz(2),xyz(3));
end
  
drms1 = sqrt(sum(sum((uv1-uv1RP).^2,2))/nrow);
drms2 = sqrt(sum(sum((uv2-uv2RP).^2,2))/nrow);

set(hProj(1,1),'XData',uv1RP(:,1),'YData',uv1RP(:,2));
set(hProj(1,2),'XData',uv2RP(:,1),'YData',uv2RP(:,2));
tstr = sprintf('rms: %.3f',drms1);
title(axs(1),tstr,'fontweight','bold','interpreter','none');    
set(hRef(1,1),'XData',uv1(:,1),'YData',uv1(:,2));
set(hRef(1,2),'XData',uv2(:,1),'YData',uv2(:,2));
tstr = sprintf('rms: %.3f',drms2);
title(axs(2),tstr,'fontweight','bold','interpreter','none');    

  
fname = sprintf('fly371_stereoReProjErr.png');
print(hFig,'-dpng',fname);
% fprintf('Printed %s...\n',fname);


%% 
% - 2 ways to optimize: {10 DOF per, 11}.
% - data: {manual labels + stro, my labels1+manual labels 2 + stro}.
% - sets: 
%  - manual L1 L2.
%  - manual L1 L2, 10DOF optim.
%  - manual L1 L2, 11DOF optim.
%  - rigid L1 L2.
%  - rigid L1 L2, 11DOF stro optimized.
%  - rigid L1 L2, 10DOF stro optimized.
%  - p_v1v2optimized
%  - p_v1v2optimized, 10DOF optimized.
%  - p_v1v2optimized, 11DOF optimized.
% visualize a) Xcal proj, and b) proj vs gt/reference.

% STRO REPROJ ERROR FOR CAL PTS

% SKEW
