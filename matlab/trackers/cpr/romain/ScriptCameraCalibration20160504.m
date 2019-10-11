%%
DROPBOXROOT = 'f:/DropBox/MultiViewFlyLegTracking';
CALIB_DIR = fullfile(DROPBOXROOT,'CamerasCalibration');
CALIB_RESULTS_LB = fullfile(CALIB_DIR,'CalibrationsApr2016','Calib_Results_stereoLeftBottom.mat');
CALIB_RESULTS_BR = fullfile(CALIB_DIR,'CalibrationsApr2016','Calib_Results_stereoBottomRight.mat');
%TRACK_DIR = fullfile(DROPBOXROOT,'trackingApril28-14-53');
TRACK_DIR = fullfile(DROPBOXROOT,'trackingApril28-15-23');
ROI_DIR = fullfile(DROPBOXROOT,'trackingApril28-14-53');
LBL_FILE = fullfile(TRACK_DIR,'20160508_allen.lbl');
%LBL_FILE = fullfile(TRACK_DIR,'20160428T145316_allen.lbl');

%%
crig = CalibratedRig(CALIB_RESULTS_LB,CALIB_RESULTS_BR);
crig.setROIs(ROI_DIR);

%%
lbl = load(LBL_FILE,'-mat');
%%
mfa = lbl.movieFilesAll;
tfBot = cellfun(@(x)~isempty(regexp(x,'_cam_2_','once')),mfa);
tfL   = cellfun(@(x)~isempty(regexp(x,'_cam_0_','once')),mfa);
tfR   = cellfun(@(x)~isempty(regexp(x,'_cam_1_','once')),mfa);
assert(nnz(tfBot)==1);
assert(nnz(tfL)==1);
assert(nnz(tfR)==1);
lpos = struct();
lpos.b = lbl.labeledpos{tfBot};
lpos.l = lbl.labeledpos{tfL};
lpos.r = lbl.labeledpos{tfR};
nptsLbl = struct();
frmLbl = struct();
for f = {'b' 'l' 'r'},f=f{1}; %#ok<FXSET>
  nptsLbl.(f) = Labeler.labelPosNPtsLbled(lpos.(f));
  frmLbl.(f) = find(nptsLbl.(f));
  nptsLblUn = unique(nptsLbl.(f)(frmLbl.(f)));
  fprintf(1,'%s nptsLblUn:\n',f);
  disp(nptsLblUn);
end

%% find frames to consider
fLR = intersect(frmLbl.l,frmLbl.r);
[tf,loc] = ismember( (fLR-1)/2, frmLbl.b );
fLR = fLR(tf);
fB = frmLbl.b(loc);
frmmat = [fLR fB];
nfrm = size(frmmat,1);
fprintf(1,'%d labeled frames\n',nfrm);
npts = size(lpos.l,1);
dXB = nan(npts,3,nfrm);
muXB = nan(npts,3,nfrm);
for iF = 1:nfrm
  fLR = frmmat(iF,1);
  fB = frmmat(iF,2);
  lposL = lpos.l(:,:,fLR);
  lposR = lpos.r(:,:,fLR);
  lposB = lpos.b(:,:,fB);
  
  yL = [lposL(:,2) lposL(:,1)]; % [row col]
  yR = [lposR(:,2) lposR(:,1)];
  yB = [lposB(:,2) lposB(:,1)]; 
  [XL,XB] = crig.stereoTriangulateLB(yL,yB);
  [XB2,XR] = crig.stereoTriangulateBR(yB,yR);
  
  dXB(:,:,iF) = XB'-XB2';
  muXB(:,:,iF) = nanmean(cat(3,XB',XB2'),3);
  
  fprintf(1,'Done with frm %d/%d\n',iF,nfrm);
end

%% compare overall deviations for each point
d2XB = sqrt(squeeze(sum(dXB.^2,2)));
d2XBmd = nanmedian(d2XB,2);
d2XBmu = nanmean(d2XB,2);
[d2XBmd d2XBmu]

%% compare deviations by dim
for i = 1:3
  x = dXB(:,i,:);
  dXBdimMd(i) = nanmedian(x(:));
  dXBdimMu(i) = nanmean(x(:));
  x = muXB(:,i,:);
  XBdimMd(i) = nanmedian(x(:));
  XBdimMu(i) = nanmean(x(:));
end

%% 
RECONSTRUCT_FILENAME = 'reconstruct3d.mat';
save(fullfile(TRACK_DIR,RECONSTRUCT_FILENAME),'frmmat','muXB','dXB');

%% Plot dXB (discrepancy between views) for various pts/coords
hFig = figure('windowstyle','docked');
ax = createsubplots(4,3);
ax = reshape(ax,4,3);
for igrp = 1:4
for col = 1:3
  axx = ax(igrp,col);
  
  ipt = (igrp-1)*5+(1:5);
  x = squeeze(dXB(ipt,col,:));
  plot(axx,x','.-');
  legend(axx,arrayfun(@num2str,ipt,'uni',0));
  grid(axx,'on');
  
  if igrp~=1
    set(axx,'YTickLabel',[]);
  end
  if igrp~=4
    set(axx,'XTickLabel',[]);
  end
end
end
for col=1:3
  linkaxes(ax(:,col));
end

%% Look for corrs between dXB and muXB
hFig = figure('windowstyle','docked');
ax = createsubplots(10,6);
ax = reshape(ax,20,3);
for ipt = 1:20
for col = 1:3
  axx = ax(ipt,col);
  
  x = squeeze(dXB(ipt,col,:));
  y = squeeze(muXB(ipt,col,:));
  scatter(axx,x,y)
  grid(axx,'on');
  
  if ipt~=1
    set(axx,'YTickLabel',[]);
  end
  if ipt~=20
    set(axx,'XTickLabel',[]);
  end
end
end
for col=1:3
  linkaxes(ax(:,col));
end






