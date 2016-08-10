%%
ROOTDIR = 'f:\Dropbox\MultiViewFlyLegTracking\multiview labeling';
assert(strcmp(pwd,ROOTDIR));

LBL = fullfile(ROOTDIR,'romainMV2_safeLabels.lbl');
NVIEW = 3;
NREALPT = 57/3;
CRIG = fullfile(ROOTDIR,'crig2_calibjun2916_roiTrackingJun22_20160809.mat');

lbl = load(LBL,'-mat');
crig2 = load(CRIG,'-mat');
crig2 = crig2.crig2;
%%
lpos = lbl.labeledpos{1};
lpostag = lbl.labeledpostag{1};
nfrm = size(lpos,3);
lpos = reshape(lpos,[NREALPT NVIEW 2 nfrm]);
lpostag = reshape(lpostag,[NREALPT NVIEW nfrm]);

%%
iPtGood = nan(0,1);
frmGood = nan(0,1);
yL = nan(0,2); % GT pts: (row,col) cropped coords in L view
yR = nan(0,2);
yB = nan(0,2);
for f=1:nfrm
for ipt = 1:NREALPT
  tflbled = nnz(isnan(lpos(ipt,:,:,f)))==0;
  tfnotocc = all(cellfun(@isempty,lpostag(ipt,:,f)));
  tf = tflbled & tfnotocc;  
  if tf
    iPtGood(end+1,1) = ipt;
    frmGood(end+1,1) = f;
    
    yL(end+1,:) = lpos(ipt,1,[2 1],f);
    yR(end+1,:) = lpos(ipt,2,[2 1],f);
    yB(end+1,:) = lpos(ipt,3,[2 1],f);
  end
end
end
nGood = numel(iPtGood);
fprintf('Found %d labeled pts.\n',nGood);

%% Data cleaning
[~,~,~,~,~,~,errfull(:,1),errfull(:,2),errfull(:,3)] = ...
  calibRoundTrip(yL,yR,yB,crig2);
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
[~,~,~,errL,errR,errB] = calibRoundTrip(yL,yR,yB,crig2);
err = errL + errR + errB;
fprintf('## Before optim: [errL errR errB] err: [%.2f %.2f %.2f] %.2f\n',...
  errL,errR,errB,err);
  
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
   'objfunBint' 4};
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
% domBL(1:3) domBR(1:3) dTBL(1:3) dTBR(1:3) dccB(1:2) dfcB(1:2)
results.objfunLRrot.idxStandard = 1:6;
results.objfunLRrotBint.idxStandard = [1:6 13:16];
results.objfunAllExt.idxStandard = 1:12;
results.objfunAllExtBint.idxStandard = 1:16;
results.objfunBint.idxStandard = 13:16;
flds = fieldnames(results);
for f=flds(:)',f=f{1}; %#ok<FXSET>
  results.(f).x1s_std = cell(size(results.(f).x1s));
  for i=1:numel(results.(f).x1s);
    xtmp = nan(1,16);
    xtmp(results.(f).idxStandard) = results.(f).x1s{i};    
    results.(f).x1s_std{i} = xtmp;
  end
end
%%
x = 1:16;
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
grid on;
legend(...
  'LRrot (reg)','LRrot',...
  'LRrotBint (reg)','LRrotBint',...
  'allext (reg)','allext',...
  'allextBint(reg)','allextBint',...
  'Bint(reg)','Bint');

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

