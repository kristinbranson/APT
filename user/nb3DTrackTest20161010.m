%%
ROOTDIR = 'f:\Dropbox\MultiViewFlyLegTracking\multiview labeling';
%assert(strcmp(pwd,ROOTDIR));

LBL = 'romainJun22NewLabels.lbl';
LBL = fullfile(ROOTDIR,LBL);

CRIG = 'crig2Optimized_calibjun2916_roiTrackingJun22_20160810_2.mat';
CRIG = fullfile(ROOTDIR,CRIG);

NVIEW = 3;
NREALPT = 57/3;

%%
lbl = load(LBL,'-mat');
crig2 = load(CRIG,'-mat');
crig2 = crig2.crig2Mod;

%%
lpos = lbl.labeledpos{1};
lpostag = lbl.labeledpostag{1};
nfrm = size(lpos,3);
lpos = reshape(lpos,[NREALPT NVIEW 2 nfrm]);
lpostag = reshape(lpostag,[NREALPT NVIEW nfrm]);

%% Generate MD table
% 
% Fields
% frm
% npts2VwLblNO. scalar, number of pts that have >= 2 views labeled (nonOcc)
% ipts2VwLblNO. vector with npts2VwLblNO els. pt indices.
% iVwsLblNO. cell vector with npts2VwLblNO els. view indices for each pt in
%   ipts2VwLblNO.
frm = nan(0,1);
npts2VwLblNO = nan(0,1);
ipts2VwLblNO = cell(0,1);
iVwsLblNO = cell(0,1);
iVwsLblNOCode = cell(0,1);
% yL = nan(0,2); % GT pts: (row,col) cropped coords in L view
% yR = nan(0,2);
% yB = nan(0,2);
VIEWCODES = 'lrb';
for f=1:nfrm
  
  iVwLbledNonOccPt = cell(NREALPT,1);
  tf2VwLbledNonOcc = false(NREALPT,1);
  for ipt = 1:NREALPT
    lposptfrm = squeeze(lpos(ipt,:,:,f));
    ltagptfrm = squeeze(lpostag(ipt,:,f));
    ltagptfrm = ltagptfrm(:);
    assert(isequal(size(lposptfrm),[NVIEW 2]));
    assert(isequal(size(ltagptfrm),[NVIEW 1]));
    tfVwLbled = ~any(isnan(lposptfrm),2);
    tfVwNotOcc = cellfun(@isempty,ltagptfrm);    
    iVwLbledNonOccPt{ipt} = find(tfVwLbled & tfVwNotOcc);      
    tf2VwLbledNonOcc(ipt) = numel(iVwLbledNonOccPt{ipt})>=2;
  end

  if any(tf2VwLbledNonOcc)
    frm(end+1,1) = f;
    npts2VwLblNO(end+1,1) = nnz(tf2VwLbledNonOcc);
    ipts2VwLblNO{end+1,1} = find(tf2VwLbledNonOcc);
    iVwsLblNO{end+1,1} = iVwLbledNonOccPt(tf2VwLbledNonOcc);
    iVwsLblNOCode{end+1,1} = cellfun(@(x)VIEWCODES(x),...
      iVwLbledNonOccPt(tf2VwLbledNonOcc),'uni',0);
  end
  
%   tf2VwLbledNonOcc(end+1,1) = nnz(tfVwLbled & tfVwNotOcc)>=2;
%   yL(end+1,:) = lpos(ipt,1,[2 1],f);
%   yR(end+1,:) = lpos(ipt,2,[2 1],f);
%   yB(end+1,:) = lpos(ipt,3,[2 1],f);  
end

nPtsLRCode = cellfun(@(x)nnz(strcmp('lr',x)),iVwsLblNOCode);
tFrmPts = table(frm,npts2VwLblNO,ipts2VwLblNO,iVwsLblNO,iVwsLblNOCode,nPtsLRCode);
% nGood = numel(iPtGood);
% fprintf('Found %d labeled pts.\n',nGood);

%% 
for i=1:size(tFrmPts,1)
  codes = tFrmPts.iVwsLblNOCode{i};
  for iV = 1:numel(iVws)
    if strcmp(codes{iV},'lr')
      fprintf(1,'%d: %d\n',tFrmPts.frm(i),tFrmPts.ipts2VwLblNO{i}(iV));
    end
  end
end


%% Add reconstruct/err stats
%
% For points labeled in all three views ('lrb'):
%  * Use each viewpair to recon/project in 3rd view and compute error.
%  * Recon 3D pt.
% For all points labeled in only two views:
%  * Recon 3D pt.
%
% New fields:
%  * X [3x19]. 3D recon pt in certain frame, say 'l'.
%  * errReconL. [npts2VwlblNOx1]. For points with 'lrb'. L2 err in L view (recon vs gt)
%  * errReconR. etc
%  * errReconB.

t19 = tFrmPts(tFrmPts.npts2VwLblNO==19,:);
nRows = size(t19,1);
XL = cell(nRows,1);
XLlr = cell(nRows,1);
errReconL = nan(nRows,19);
errReconR = nan(nRows,19);
errReconB = nan(nRows,19);
errReconL_lr = nan(nRows,19);
errReconR_lr = nan(nRows,19);
errReconB_lr = nan(nRows,19);
for iRow = 1:nRows
  frm = t19.frm(iRow);
  XLrow = nan(3,19);
  XLlrrow = nan(3,19);
  for iPt = 1:19
    code = t19.iVwsLblNOCode{iRow}{iPt};
    
    lposPt = squeeze(lpos(iPt,:,:,frm));
    assert(isequal(size(lposPt),[3 2]));
    yL = lposPt(1,[2 1]);
    yR = lposPt(2,[2 1]);
    yB = lposPt(3,[2 1]);
    switch code
      case 'lr'
        %assert(all(isnan(yB(:))));
        XLrow(:,iPt) = crig2.stereoTriangulateCropped(yL,yR,'L','R');        
      case 'lb'
        %assert(all(isnan(yR(:))));
        XLrow(:,iPt) = crig2.stereoTriangulateLB(yL,yB);        
      case 'rb'
        %assert(all(isnan(yL(:))));
        XBbr = crig2.stereoTriangulateBR(yB,yR);
        XLrow(:,iPt) = crig2.camxform(XBbr,'bl');
      case 'lrb'
        [~,~,~,...
          errReconL(iRow,iPt),errReconR(iRow,iPt),errReconB(iRow,iPt),...
          ~,~,~,...
          XLrow(:,iPt),~,~,...
          errReconL_lr(iRow,iPt),errReconR_lr(iRow,iPt),errReconB_lr(iRow,iPt),...
          XLlrrow(:,iPt)] = ...
            crig2.calibRoundTripFull(yL,yR,yB);
    end
  end
  XL{iRow} = XLrow;
  XLlr{iRow} = XLlrrow;
end

tAug = table(XL,errReconL,errReconR,errReconB,...
  errReconL_lr,errReconR_lr,errReconB_lr,XLlr);
t19aug = [t19 tAug];

%% Make an expanded frm/pt err browsing table
s19expanded = struct(...
  'frm',cell(0,1),...
  'ipt',[],...
  'code',[],...
  'XL',[],...
  'errReconL',[],...
  'errReconR',[],...
  'errReconB',[],...
  'errReconL_lr',[],...
  'errReconR_lr',[],...
  'errReconB_lr',[],...
  'XLlr',[]);
%%
nRows = size(t19aug,1);
ERRFLDS = {'errReconL' 'errReconR' 'errReconB' 'errReconL_lr' 'errReconR_lr' 'errReconB_lr'};
for iRow=1:nRows
  assert(t19aug.npts2VwLblNO(iRow)==19);
  assert(isequal(t19aug.ipts2VwLblNO{iRow},(1:19)'));
  frm = t19aug.frm(iRow);
  for iPt = 1:19
    s19expanded(end+1,1).frm = frm;
    s19expanded(end,1).ipt = iPt;
    s19expanded(end,1).code = t19aug.iVwsLblNOCode{iRow}{iPt};
    s19expanded(end,1).XL = t19aug.XL{iRow}(:,iPt)';
    s19expanded(end,1).XLlr = t19aug.XLlr{iRow}(:,iPt)';
    for f=ERRFLDS,f=f{1}; %#ok<FXSET>
      s19expanded(end,1).(f) = t19aug.(f)(iRow,iPt);
    end
  end
end
t19expanded = struct2table(s19expanded);
t19expandedLRB = t19expanded(strcmp(t19expanded.code,'lrb'),:);
%% 
hFig = figure('windowstyle','docked');
for f = ERRFLDS,f=f{1};
  clf(hFig);
  z = t19expandedLRB.(f);
  hist(z,50);
  iptsbig = unique(t19expandedLRB.ipt(z>20));
  fprintf('\n%s:\n',f);
  disp(iptsbig);
  input('hk');
end

%%
t19expandedLRB = t19expandedLRB(t19expandedLRB.ipt~=19,:);
%% Browse original/recon labels for given frame/pt
% CONC: first one is mislabel in side view. second one is mislabel in
% bottom view. 
% IDEA: just recon from two pts that lead to most consistency?
%tRow = [1711 2032]; bad errReconR
tRow = [120 1711 1889 2289];
TROWIDX = 3;
frame = t19expandedLRB.frm(tRow);
lObj.setFrame(frame(TROWIDX));
%lposCurr = squeeze(lpos(4,:,:,11952)); % 3x2
axAll = lObj.gdata.axes_all;
X = cell(1,3);
Xlr = cell(1,3);
X{1} = t19expandedLRB.XL(tRow(TROWIDX),:)';
X{2} = crig2.camxform(X{1},'lr');
X{3} = crig2.camxform(X{1},'lb');
Xlr{1} = t19expandedLRB.XLlr(tRow(TROWIDX),:)';
Xlr{2} = crig2.camxform(Xlr{1},'lr');
Xlr{3} = crig2.camxform(Xlr{1},'lb');
for iAx = 1:3
  ax = axAll(iAx);
  hold(ax,'on');
  [r,c] = crig2.projectCPR(Xlr{iAx},iAx);
  hLine(iAx) = plot(ax,c,r,'wx','markersize',8);
end

%%
hold(ax,'on');
clrs = parula(19);
for iF=1:numel(frms)
  lposF = squeeze(lpos(:,3,:,frms(iF)));
  for iPt=1:19
    plot(ax,lposF(iPt,1),lposF(iPt,2),'o','Color',clrs(iPt,:));
  end
  input(num2str(iF));
end



  

%%
%
% 
% [~,~,~,~,~,~,errfull(:,1),errfull(:,2),errfull(:,3)] = ...
%   calibRoundTrip(yL,yR,yB,crig2);

% 
clear errfull;
[~,~,~,~,errfull(:,1),errfull(:,2)] = calibRoundTrip2(yL,yR,yB,crig2);

%%
IDXRM = 21; % strong outlier
yL(IDXRM,:) = [];
yR(IDXRM,:) = [];
yB(IDXRM,:) = [];
frmGood(IDXRM) = [];
iPtGood(IDXRM) = [];
clear errfull;

%% run optimization

% Before: compute current error
% [~,~,~,errL,errR,errB] = calibRoundTrip(yL,yR,yB,crig2);
% err = errL + errR + errB;
% fprintf('## Before optim: [errL errR errB] err: [%.2f %.2f %.2f] %.2f\n',...
%   errL,errR,errB,err);
[~,~,errL,errR] = calibRoundTrip2(yL,yR,yB,crig2);
fprintf('## Before optim: [errL errR] err: [%.2f %.2f] %.2f\n',errL,errR,errL+errR);
  
%%
% OBJFUNS = ... % objFun, and NX
%   {'objfunIntExtRot' 10; ...
%    'objfunIntExtRot2' 18; ...
%    'objfunIntExtRot3' 16};
OBJFUNS = ...
  {'objfunLRrot' 6; ...
   'objfunLRrotBint' 10; ...
   'objfunAllExt' 12; ...
   'objfunAllExtBint' 16; ...
   'objfunBint' 4; ...
   'objfunAllExtAllInt' 24};
NOBJFUNS = size(OBJFUNS,1);
results = struct();

for iObjFun = 1:NOBJFUNS
  
  objFun = OBJFUNS{iObjFun,1};
  nx = OBJFUNS{iObjFun,2};
  fprintf('### ObjFun: %s\n',objFun);  
  
  % Set lambda (regularization)
  % Let's say, if any param changes by ~10% that's a big change, equivalent
  % to a 2 px error in round-tripped projected pts
  % If dParams are all ~10%*Params
  LAMBDAS = {
    2*10*ones(nx,1);
    zeros(nx,1)
    };
  X0S = {
    zeros(nx,1);
    .02*rand(nx,1)-0.01;
    };
  NLAMBDA = numel(LAMBDAS);
  NX0S = numel(X0S);

  opts = optimset('Display','off');

  x1s = cell(0,1);
  fval = cell(0,1);
  errs = cell(0,1);
  for iLambda = 1:NLAMBDA
  for iX0 = 1:NX0S
    lambda = LAMBDAS{iLambda};
    x0 = X0S{iX0};

    [x1,fv] = ...
      fminsearch(@(x) feval(objFun,x,yL,yR,yB,crig2,lambda,{'silent',true}),x0,opts);
    [err,errL,errR,errB,errreg,~,~,~,errFull] = feval(objFun,x1,yL,yR,yB,crig2,lambda,{});
    fprintf('## After optim: [errL errR errB] err(noreg) errReg med[errL errR errB] mad[errL errR errB]: [%.2f %.2f %.2f] %.2f %.2f [%.2f %.2f %.2f] [%.2f %.2f %.2f]\n',...
      errL,errR,errB,errL+errR+errB,errreg,...
	  median(errFull.L),median(errFull.R),median(errFull.B),...
	  mad(errFull.L,1),mad(errFull.R,1),mad(errFull.B,1));

    x1s{end+1,1} = x1;
    fval{end+1,1} = fv;
    errs{end+1,1} = [errL errR errB errreg err];
  end
  end
  
  results.(objFun).x1s = x1s;
  results.(objFun).fval = fval;
  results.(objFun).errs = errs;
end

%% assess
% domBL(1:3) domBR(1:3) dTBL(1:3) dTBR(1:3) dccB(1:2) dfcB(1:2) dccL(1:2)
% dfcL(1:2) dccR(1:2) dfcR(1:2)
results.objfunLRrot.idxStandard = 1:6;
results.objfunLRrotBint.idxStandard = [1:6 13:16];
results.objfunAllExt.idxStandard = 1:12;
results.objfunAllExtBint.idxStandard = 1:16;
results.objfunBint.idxStandard = 13:16;
results.objfunAllExtAllInt.idxStandard = 1:24;
flds = fieldnames(results);
for f=flds(:)',f=f{1}; %#ok<FXSET>
  results.(f).x1s_std = cell(size(results.(f).x1s));
  for i=1:numel(results.(f).x1s);
    xtmp = nan(1,24);
    xtmp(results.(f).idxStandard) = results.(f).x1s{i};    
    results.(f).x1s_std{i} = xtmp;
  end
end
%%
x = 1:24;
figure;
plot(x,results.objfunLRrot.x1s_std{1},'o-',...
	 x,results.objfunLRrot.x1s_std{3},'o-');
hold on;
plot(x,results.objfunLRrotBint.x1s_std{1},'x-',...
     x,results.objfunLRrotBint.x1s_std{3},'x-');
plot(x,results.objfunAllExt.x1s_std{1},'+-',...
     x,results.objfunAllExt.x1s_std{3},'+-');
plot(x,results.objfunAllExtBint.x1s_std{1},'v-',...
     x,results.objfunAllExtBint.x1s_std{3},'v-');
plot(x,results.objfunBint.x1s_std{1},'^-',...	
	 x,results.objfunBint.x1s_std{3},'^-');
plot(x,results.objfunAllExtAllInt.x1s_std{1},'.-',...
     x,results.objfunAllExtAllInt.x1s_std{3},'.-');
grid on;
legend(...
  'LRrot (reg)','LRrot',...
  'LRrotBint (reg)','LRrotBint',...
  'allext (reg)','allext',...
  'allextBint(reg)','allextBint',...
  'Bint(reg)','Bint',...
  'allExtAllInt(reg)','allExtAllInt');

%%
x = 1:24;
figure;
plot(x,results.objfunAllExtBint.x1s_std{1},'v-',...
     x,results.objfunAllExtBint.x1s_std{3},'v-');
hold on;
plot(x,results.objfunAllExtAllInt.x1s_std{1},'.-',...
     x,results.objfunAllExtAllInt.x1s_std{3},'.-');
grid on;
legend(...
  'allextBint(reg)','allextBint',...
  'allExtAllInt(reg)','allExtAllInt');


%%
% Found 33 labeled pts.
% ## Before optim: [errL errR errB] err: [4.30 1.21 2.30] 7.81
% ### ObjFun: objfunLRrot
% ## After optim: [errL errR errB] err(noreg) errReg med[errL errR errB] mad[errL errR errB]: [0.88 1.03 0.97] 2.89 0.03 [0.65 0.83 0.68] [0.45 0.64 0.47]
% ## After optim: [errL errR errB] err(noreg) errReg med[errL errR errB] mad[errL errR errB]: [0.94 1.05 0.99] 2.98 0.05 [0.67 0.95 0.68] [0.40 0.68 0.45]
% ## After optim: [errL errR errB] err(noreg) errReg med[errL errR errB] mad[errL errR errB]: [0.87 1.02 0.97] 2.87 0.00 [0.59 0.76 0.65] [0.44 0.59 0.44]
% ## After optim: [errL errR errB] err(noreg) errReg med[errL errR errB] mad[errL errR errB]: [0.94 1.05 0.99] 2.98 0.00 [0.66 0.95 0.68] [0.39 0.68 0.44]
% ### ObjFun: objfunLRrotBint
% ## After optim: [errL errR errB] err(noreg) errReg med[errL errR errB] mad[errL errR errB]: [0.85 1.02 0.96] 2.83 0.05 [0.55 0.85 0.60] [0.40 0.57 0.31]
% ## After optim: [errL errR errB] err(noreg) errReg med[errL errR errB] mad[errL errR errB]: [0.95 0.99 0.97] 2.91 0.17 [0.69 0.67 0.61] [0.43 0.55 0.39]
% ## After optim: [errL errR errB] err(noreg) errReg med[errL errR errB] mad[errL errR errB]: [0.88 1.03 0.96] 2.87 0.00 [0.66 0.92 0.64] [0.36 0.67 0.41]
% ## After optim: [errL errR errB] err(noreg) errReg med[errL errR errB] mad[errL errR errB]: [0.96 0.99 0.95] 2.91 0.00 [0.63 0.63 0.54] [0.42 0.45 0.29]
% ### ObjFun: objfunAllExt
% ## After optim: [errL errR errB] err(noreg) errReg med[errL errR errB] mad[errL errR errB]: [0.90 1.02 0.98] 2.90 0.04 [0.65 0.85 0.61] [0.42 0.61 0.42]
% ## After optim: [errL errR errB] err(noreg) errReg med[errL errR errB] mad[errL errR errB]: [1.01 1.05 1.06] 3.12 0.13 [0.69 0.85 0.76] [0.54 0.66 0.44]
% ## After optim: [errL errR errB] err(noreg) errReg med[errL errR errB] mad[errL errR errB]: [0.90 1.04 0.97] 2.91 0.00 [0.66 0.82 0.55] [0.43 0.60 0.39]
% ## After optim: [errL errR errB] err(noreg) errReg med[errL errR errB] mad[errL errR errB]: [0.99 1.08 1.06] 3.13 0.00 [0.73 0.96 0.75] [0.52 0.67 0.48]
% ### ObjFun: objfunAllExtBint
% ## After optim: [errL errR errB] err(noreg) errReg med[errL errR errB] mad[errL errR errB]: [0.85 1.01 0.94] 2.80 0.07 [0.55 0.76 0.49] [0.28 0.47 0.29]
% ## After optim: [errL errR errB] err(noreg) errReg med[errL errR errB] mad[errL errR errB]: [0.81 1.00 0.92] 2.73 0.10 [0.50 0.69 0.53] [0.26 0.59 0.29]
% ## After optim: [errL errR errB] err(noreg) errReg med[errL errR errB] mad[errL errR errB]: [0.81 1.00 0.90] 2.71 0.00 [0.56 0.61 0.49] [0.39 0.42 0.29]
% ## After optim: [errL errR errB] err(noreg) errReg med[errL errR errB] mad[errL errR errB]: [0.82 1.00 0.92] 2.74 0.00 [0.70 0.71 0.63] [0.47 0.60 0.49]
% ### ObjFun: objfunBint
% ## After optim: [errL errR errB] err(noreg) errReg med[errL errR errB] mad[errL errR errB]: [1.12 1.24 0.97] 3.33 0.52 [0.78 0.93 0.67] [0.53 0.68 0.41]
% ## After optim: [errL errR errB] err(noreg) errReg med[errL errR errB] mad[errL errR errB]: [1.12 1.24 0.97] 3.33 0.51 [0.80 0.94 0.67] [0.52 0.68 0.39]
% ## After optim: [errL errR errB] err(noreg) errReg med[errL errR errB] mad[errL errR errB]: [1.14 1.18 0.91] 3.23 0.00 [0.85 0.86 0.55] [0.50 0.44 0.40]
% ## After optim: [errL errR errB] err(noreg) errReg med[errL errR errB] mad[errL errR errB]: [1.79 1.90 1.36] 5.05 0.00 [1.45 1.58 0.92] [0.67 0.81 0.65]

% AL20160810: choose allExtBint, with regularization, start at 0, for good
% improvement and most typical pattern of x1. As a class/objfun allExtBint 
% seems to show best improvement overall
x1use = results.objfunAllExtBint.x1s{1};
[~,~,~,~,~,~,~,~,~,crig2Mod] = objfunAllExtBint(x1use,yL,yR,yB,crig2,zeros(size(x1use)),{});

%% Oops use calibRoundTrip2
%
% ## Before optim: [errL errR] err: [16.85 13.54] 30.39
% ### ObjFun: objfunLRrot
% ## After optim: [errL errR errB] err(noreg) errReg med[errL errR errB] mad[errL errR errB]: [9.53 9.47 0.00] 19.00 0.08 [1.91 2.03 0.00] [0.90 0.79 0.00]
% ## After optim: [errL errR errB] err(noreg) errReg med[errL errR errB] mad[errL errR errB]: [9.53 9.47 0.00] 18.99 0.07 [1.90 2.00 0.00] [0.93 0.81 0.00]
% ## After optim: [errL errR errB] err(noreg) errReg med[errL errR errB] mad[errL errR errB]: [9.55 9.51 0.00] 19.06 0.00 [2.04 2.21 0.00] [0.89 0.86 0.00]
% ## After optim: [errL errR errB] err(noreg) errReg med[errL errR errB] mad[errL errR errB]: [9.64 9.60 0.00] 19.23 0.00 [2.32 2.35 0.00] [1.11 1.14 0.00]
% ### ObjFun: objfunLRrotBint
% ## After optim: [errL errR errB] err(noreg) errReg med[errL errR errB] mad[errL errR errB]: [9.39 9.42 0.00] 18.81 0.10 [1.68 1.76 0.00] [0.84 0.88 0.00]
% ## After optim: [errL errR errB] err(noreg) errReg med[errL errR errB] mad[errL errR errB]: [9.55 9.59 0.00] 19.15 0.14 [2.15 2.23 0.00] [1.16 0.94 0.00]
% ## After optim: [errL errR errB] err(noreg) errReg med[errL errR errB] mad[errL errR errB]: [9.53 9.49 0.00] 19.02 0.00 [1.86 2.07 0.00] [0.90 0.78 0.00]
% ## After optim: [errL errR errB] err(noreg) errReg med[errL errR errB] mad[errL errR errB]: [9.55 9.59 0.00] 19.14 0.00 [2.20 2.30 0.00] [1.09 1.05 0.00]
% ### ObjFun: objfunAllExt
% ## After optim: [errL errR errB] err(noreg) errReg med[errL errR errB] mad[errL errR errB]: [9.52 9.49 0.00] 19.01 0.04 [1.89 2.06 0.00] [0.88 0.81 0.00]
% ## After optim: [errL errR errB] err(noreg) errReg med[errL errR errB] mad[errL errR errB]: [9.83 9.82 0.00] 19.65 0.14 [2.34 2.38 0.00] [1.26 1.18 0.00]
% ## After optim: [errL errR errB] err(noreg) errReg med[errL errR errB] mad[errL errR errB]: [9.52 9.49 0.00] 19.01 0.00 [1.86 1.96 0.00] [0.90 0.87 0.00]
% ## After optim: [errL errR errB] err(noreg) errReg med[errL errR errB] mad[errL errR errB]: [9.48 9.62 0.00] 19.10 0.00 [1.92 2.18 0.00] [0.93 0.73 0.00]
% ### ObjFun: objfunAllExtBint
% ## After optim: [errL errR errB] err(noreg) errReg med[errL errR errB] mad[errL errR errB]: [9.36 9.37 0.00] 18.73 0.12 [1.61 1.74 0.00] [0.74 0.85 0.00]
% ## After optim: [errL errR errB] err(noreg) errReg med[errL errR errB] mad[errL errR errB]: [9.44 9.43 0.00] 18.87 0.13 [1.66 1.85 0.00] [0.91 0.75 0.00]
% ## After optim: [errL errR errB] err(noreg) errReg med[errL errR errB] mad[errL errR errB]: [9.39 9.37 0.00] 18.76 0.00 [1.82 1.77 0.00] [0.91 1.05 0.00]
% ## After optim: [errL errR errB] err(noreg) errReg med[errL errR errB] mad[errL errR errB]: [9.36 9.35 0.00] 18.72 0.00 [1.65 1.66 0.00] [0.72 1.01 0.00]
% ### ObjFun: objfunBint
% ## After optim: [errL errR errB] err(noreg) errReg med[errL errR errB] mad[errL errR errB]: [13.75 13.69 0.00] 27.44 0.40 [6.66 6.48 0.00] [1.21 1.36 0.00]
% ## After optim: [errL errR errB] err(noreg) errReg med[errL errR errB] mad[errL errR errB]: [13.80 13.57 0.00] 27.36 0.30 [6.59 6.38 0.00] [1.15 1.29 0.00]
% ## After optim: [errL errR errB] err(noreg) errReg med[errL errR errB] mad[errL errR errB]: [13.87 13.48 0.00] 27.35 0.00 [6.62 6.33 0.00] [1.24 1.19 0.00]
% ## After optim: [errL errR errB] err(noreg) errReg med[errL errR errB] mad[errL errR errB]: [13.84 13.51 0.00] 27.35 0.00 [6.62 6.38 0.00] [1.22 1.25 0.00]
% ### ObjFun: objfunAllExtAllInt
% ## After optim: [errL errR errB] err(noreg) errReg med[errL errR errB] mad[errL errR errB]: [9.26 9.39 0.00] 18.65 0.07 [1.85 1.69 0.00] [1.12 0.83 0.00]
% ## After optim: [errL errR errB] err(noreg) errReg med[errL errR errB] mad[errL errR errB]: [9.34 9.45 0.00] 18.78 0.13 [1.54 1.90 0.00] [0.91 0.88 0.00]
% ## After optim: [errL errR errB] err(noreg) errReg med[errL errR errB] mad[errL errR errB]: [9.31 9.37 0.00] 18.67 0.00 [1.75 1.59 0.00] [1.08 0.75 0.00]
% ## After optim: [errL errR errB] err(noreg) errReg med[errL errR errB] mad[errL errR errB]: [9.26 9.50 0.00] 18.76 0.00 [1.59 2.02 0.00] [1.00 1.14 0.00]

x1use = results.objfunAllExtBint.x1s{1};
[~,~,~,~,~,~,~,~,~,crig2Mod] = objfunAllExtBint(x1use,yL,yR,yB,crig2,zeros(size(x1use)),{});

%%
x1use = results.objfunAllExtAllInt.x1s{3};
[~,~,~,~,~,~,~,~,~,crig2AllExtAllInt] = objfunAllExtAllInt(x1use,yL,yR,yB,crig2,zeros(size(x1use)),{});

%   omcurr = om + dparams(1:3);
%   Tcurr = T + dparams(4:6);
%   [XL,XR] = stereo_triangulation(xL,xR,omcurr,Tcurr,fc_left,cc_left,kc_left,alpha_c_left,fc_right,cc_right,kc_right,alpha_c_right);
%   [xL_re] = project_points2(XL,zeros(size(om)),zeros(size(T)),fc_left,cc_left,kc_left,alpha_c_left);
%   [xR_re] = project_points2(XR,zeros(size(om)),zeros(size(T)),fc_right,cc_right,kc_right,alpha_c_right);
%   d1 = median(xL_re-xL,2);
%   d2 = median(xR_re-xR,2);
%   tweakmederr1(:,mousei) = d1;
%   tweakmederr2(:,mousei) = d2;
%   s1 = median(abs(bsxfun(@minus,d1,xL_re-xL)),2);
%   s2 = median(abs(bsxfun(@minus,d1,xR_re-xR)),2);
%   tweakmaderr1(:,mousei) = s1;
%   tweakmaderr2(:,mousei) = s2;
%   tweakT(:,mousei) = Tcurr;
%   tweakom(:,mousei) = omcurr;
% 
%   fprintf('Mouse %s: med err1 = %s, err2 = %s, mad err1 = %s, mad err2 = %s\n',...
%     mice{mousei},mat2str(d1,3),mat2str(d2,3),mat2str(s1,3),mat2str(s2,3));
%   fprintf('dT = %s, dom*180/pi = %s\n',mat2str(dparams(4:6),3),mat2str(dparams(1:3)*180/pi,3));

