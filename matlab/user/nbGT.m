%%
GTLBL = 'rf_gt.lbl';
lblGT = load(GTLBL,'-mat');

%%
lpos = lblGT.labeledpos{1};
lpostag = lblGT.labeledpostag{1};

% set "wrong-side" pts to nan
[~,nphyspt,nview,nfrm] = RF.lposDim(lpos);
assert(nphyspt==6);
lpos4d = reshape(lpos,[nphyspt,nview,2,nfrm]);
LEFTPTS = 1:3;
RGHTPTS = 4:6;
lpos4d(RGHTPTS,1,:,:) = nan;
lpos4d(LEFTPTS,2,:,:) = nan;
lpos = reshape(lpos4d,size(lpos));

% make table
tFPgt = RF.FPtable(lpos,lpostag);
% subsequent only operates on tFPgt

%% check tfVws2VwLbl
nrow = size(tFPgt,1);
for irow=1:nrow
  tf = tFPgt.tfVws2VwLbl{irow};
  assert(isequal(tf,[1 1 1 0 0 0;0 0 0 1 1 1;1 1 1 1 1 1]));
end

%%
CRIG = 'crig_jun22_20170123.mat';
crig = load(CRIG);
crig = crig.crig;

%% Reconstruct
tFPgtAug = RF.recon3D(tFPgt,crig);

%% GT 2dProjs. 2d proj err.
p2dGTre = nan(nrow,6,3,2); % pt, vw, x/y
for irow=1:nrow
  XLrow = tFPgtAug.XL{irow};
  for iVw=1:3
    Xvw = crig.viewXformCPR(XLrow,1,iVw); % iViewBase==1
    [r,c] = crig.projectCPR(Xvw,iVw);
    assert(numel(r)==6);
    assert(numel(c)==6);
    p2dGTre(irow,:,iVw,1) = c;
    p2dGTre(irow,:,iVw,2) = r;
  end
end

%% Results to compare
RES2D = {
  'rf2dLtype1_pp.trk' 'rf2dRtype1_pp.trk' 'rf2dBtype1_pp.trk';
  'rf2dL_5exps.trk' 'rf2dR_5exps.trk' 'rf2dB_5exps.trk';
  'rf2dL_5exps_initUseFF.trk' 'rf2dR_5exps_initUseFF.trk' 'rf2dB_5exps_initUseFF.trk';};
RES2DLBL = {'2d_ty1';'2d_5exp';'2d_5expff'};
RES3D = {
  'trackResPruned_20170110.mat';
  'trackResPruned_jitterForeMore_20170116.mat';
  'trackResPruned_goBig_20170117.mat'};
RES3DLBL = {'3d_jan10';'3d_jtr';'3d_big'};
RES3DF2D = {
  'trackRes3Dfrom2D_type1pp_20170123.mat';
  'trackRes3Dfrom2D_5exps_20170123.mat';
  'trackRes3Dfrom2D_5expsInitUseFF_20170123.mat';};
RES3DF2DLBL = {'3df_ty1';'3df_5exp';'3df_5expff'};

NRES2D = size(RES2D,1);
NRES3D = size(RES3D,1);
NRES3DF2D = size(RES3DF2D,1);

frms = tFPgtAug.frm;
nfrm = numel(frms);
pRes2d = nan(NRES2D,nrow,6,3,2); % res, pt, vw, x/y. only left pts in L view, only right pts in R view.
pRes3d = nan(NRES3D,nrow,6,3,2);
pRes3dF2d = nan(NRES3D,nrow,6,3,2);

% res2d
for iRes=1:NRES2D
  trkall = cellfun(@(x)load(x,'-mat'),RES2D(iRes,:));
  for iVw=1:3
    trk = trkall(iVw);
    switch iVw
      case 1
        tmp = trk.pTrk(1:3,:,frms);
        tmp = permute(tmp,[3 1 2]);
        pRes2d(iRes,:,1:3,iVw,:) = tmp;
      case 2
        tmp = trk.pTrk(1:3,:,frms);
        tmp = permute(tmp,[3 1 2]);
        pRes2d(iRes,:,4:6,iVw,:) = tmp;
      case 3
        tmp = trk.pTrk(1:6,:,frms);
        tmp = permute(tmp,[3 1 2]);
        pRes2d(iRes,:,:,iVw,:) = tmp;
    end
  end
end

% res3d
%pRes3d = nan(NRES3D,nrow,6,3,2);
for iRes=1:NRES3D
  tr = load(RES3D{iRes},'-mat');
  pTstTRed = tr.pTstTRedSingle(:,:,end);
  szassert(pTstTRed,[2000 54]);
  pTstTRedFrms = pTstTRed(frms,:);
  XL = reshape(pTstTRedFrms,[nfrm 18 3]);
  XL = permute(XL,[3 2 1]); % [3 18 nfrm]  
  for iFrm=1:nfrm
    for iVw=1:3
      switch iVw
        case 1
          iptOutOf18 = 1:3;
          iptOutOf6 = 1:3;
        case 2
          iptOutOf18 = 4:6;
          iptOutOf6 = 4:6;
        case 3
          iptOutOf18 = 1:6;
          iptOutOf6 = 1:6;
      end
      XLfrmvw = XL(:,iptOutOf18,iFrm);
      Xvw = crig.viewXformCPR(XLfrmvw,1,iVw);
      [r,c] = crig.projectCPR(Xvw,iVw);
      pRes3d(iRes,iFrm,iptOutOf6,iVw,1) = c;
      pRes3d(iRes,iFrm,iptOutOf6,iVw,2) = r;
    end
  end
end
  
% res3dF2d
%pRes3dF2d = nan(NRES3DF2D,nrow,6,3,2);
for iRes=1:NRES3DF2D
  tr = load(RES3DF2D{iRes},'-mat');
  szassert(tr.XLre,[3 18 2000])
  XL = tr.XLre(:,:,frms); % [3 18 frm]
  for iFrm=1:nfrm
    for iVw=1:3
      switch iVw
        case 1
          iptOutOf18 = 1:3;
          iptOutOf6 = 1:3;
        case 2
          iptOutOf18 = 4:6;
          iptOutOf6 = 4:6;
        case 3
          iptOutOf18 = 1:6;
          iptOutOf6 = 1:6;
      end
      XLfrmvw = XL(:,iptOutOf18,iFrm);
      Xvw = crig.viewXformCPR(XLfrmvw,1,iVw);
      [r,c] = crig.projectCPR(Xvw,iVw);
      pRes3dF2d(iRes,iFrm,iptOutOf6,iVw,1) = c;
      pRes3dF2d(iRes,iFrm,iptOutOf6,iVw,2) = r;
    end
  end
end

%%
pResBig = cat(1,pRes2d,pRes3d,pRes3dF2d);
pResBigLbl = cat(1,RES2DLBL,RES3DLBL,RES3DF2DLBL);

%%
nRes = size(pResBig,1);
pDiffRes = nan(nRes,nfrm,6,3);
pDiffPtRes = nan(nRes,6);
for iRes=1:nRes
  pResThis = squeeze(pResBig(iRes,:,:,:,:)); % [nfrm 6 3 2]
  assert(nnz(~isnan(pResThis(:,4:6,1,:)))==0);
  assert(nnz(~isnan(pResThis(:,1:3,2,:)))==0);
  
  pDiff = sqrt(sum((pResThis-p2dGTre).^2,4));
  szassert(pDiff,[nfrm 6 3]);
  pDiffRes(iRes,:,:,:) = pDiff;
  
  for ipt=1:6
    tmp = pDiff(:,ipt,:);
    pDiffPtRes(iRes,ipt) = nanmean(tmp(:));
  end
  % p2dGTre = nan(nrow,6,3,2); % pt, vw, x/y
end
%%
assert(exist('lObj','var')>0);
axAll = lObj.gdata.axes_all;
if exist('hLine','var')>0
  deleteValidGraphicsHandles(hLine);
end
NPTS = 6;
NRES = size(pResBig,1);
MARKERS = {'*' 'x' '+' 's' 'p' 'h' 'v' '^' '<'};
hLine = gobjects(3,NPTS,NRES);
for iAx = 1:3
  ax = axAll(iAx);
  hold(ax,'on');
  for iPt = 1:NPTS
    for iRes = 1:NRES
      hLine(iAx,iPt,iRes) = plot(ax,nan,nan,'.',...
        'markersize',6,...
        'marker',MARKERS{iRes},...
        'Color',RF.COLORS{iPt});
    end
  end
end

for iF = 1:nfrm
  lObj.setFrameGUI(frms(iF));
  for iVw=1:3
    for iPt=1:NPTS
      for iRes=1:NRES
        xy = squeeze(pResBig(iRes,iF,iPt,iVw,:));
        hLine(iVw,iPt,iRes).XData = xy(1);
        hLine(iVw,iPt,iRes).YData = xy(2);
      end
    end
  end
  
  input(sprintf('frame=%d',frms(iF)));
end


%%

%lposCurr = squeeze(lpos(4,:,:,11952)); % 3x2
