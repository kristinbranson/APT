%% Load safe labels and starting calibration

% SET ME: project/lbl file containing safe/sure labels. Currently, the
% project is expected to contain only a single movie.
LBL = 'f:\Dropbox\MultiViewFlyLegTracking\multiview labeling\romainMV2_safeLabels.lbl';

% SET ME: base/starting CalibratedRig2 object (eg calibration object used
% to generate labels in LBL)
CRIG = 'f:\Dropbox\MultiViewFlyLegTracking\multiview labeling\crig2_calibjun2916_roiTrackingJun22_20160809.mat';
%CRIG = 'f:\Dropbox\MultiViewFlyLegTracking\multiview labeling\crig2Optimized_calibjun2916_roiTrackingJun22_20160810_AllExtAllInt.mat';

lbl = load(LBL,'-mat');
lbl = Labeler.lblModernize(lbl);
if size(lbl.movieFilesAll,1)>1
  warning('Project has more than one movie. Only using labels for first movie.');
end  

crig2 = load(CRIG,'-mat');
flds = fieldnames(crig2);
assert(isstruct(crig2) && isscalar(flds),...
  'Unexpected contents in calibration rig file: %s',CRIG);
crig2 = crig2.(flds{1});
if ~isa(crig2,'CalibratedRig2')
  warning('Expected a CalibratedRig2 object.');
end

%% Read table of labeled data
nView = lbl.cfg.NumViews;
assert(nView==3);
nRealPt = lbl.cfg.NumLabelPoints;
assert(nRealPt==57/3);

lpos = lbl.labeledpos{1};
lpostag = lbl.labeledpostag{1};
nfrm = size(lpos,3);
lpos = reshape(lpos,[nRealPt nView 2 nfrm]);
lpostag = reshape(lpostag,[nRealPt nView nfrm]);

iPtGood = nan(0,1);
frmGood = nan(0,1);
yL = nan(0,2); % GT pts: (row,col) cropped coords in L view
yR = nan(0,2);
yB = nan(0,2);
for f=1:nfrm
for ipt = 1:nRealPt
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

tFP = table(frmGood,iPtGood,yL,yR,yB);
tFP.Properties.VariableNames{'frmGood'} = 'frm';
tFP.Properties.VariableNames{'iPtGood'} = 'ipt';
nGood = size(tFP,1);
fprintf('Found %d labeled (non-occluded) pts.\n',nGood);

clear iPtGood frmGood yL yR yB;

%% Inspect data. Browse the table tFPerr0. The errL and errR columns 
% represent reconstruction error (in pixels) in the left/right views, based 
% on the current calibration object. Unusual outliers could indicate 
% corrupt or mislabeled points.
clear errfull;
[~,~,~,~,errfull(:,1),errfull(:,2)] = calibRoundTrip2(tFP.yL,tFP.yR,tFP.yB,crig2);
tFPerr0 = table(tFP.frm,tFP.ipt,errfull(:,1),errfull(:,2),...
  'VariableNames',{'frm','ipt','errL','errR'})

%% (Optional) Remove any outlier rows from data (skip this cell if no 
% outlier rows)

% SET ME: Optionally specify the row numbers as an array here. These rows 
% will be removed. 
ROWRM = 1;

tFP(ROWRM,:) = [];
tFPerr0(ROWRM,:) = [];
fprintf('Removed %d rows, leaving %d rows of data remaining.\n',numel(ROWRM),size(tFP,1));

%% Run optimization: summary of "before" error
[~,~,errL,errR] = calibRoundTrip2(tFP.yL,tFP.yR,tFP.yB,crig2);
fprintf('## Before optim: [errL errR] err: [%.2f %.2f] %.2f\n',errL,errR,errL+errR);
  
%% Run optimization
%
% This cell tries a few different objective functions that optimize over
% different sets of parameters. Eg:
%
% objfunLRRot: left and right cameras can rotate
% objfunLRrotBint: left and right cams can rotate, plus bottom intrinsic params
% objfunAllExt: all extrinsic parameters
% ...etc
%
% For each objective function, we currently run the optimization in four 
% ways ("types"):
% 1. With regularization, initialize at 0
% 2. W/out reguarization, initialize at 0
% 3. With regularization, initialize with a little randomization
% 4. W/out regularization, initialize with a little randomization
%
% Regularization adds a "damping" parameter that introduces a cost in the 
% optimization for varying the parameters. The idea is that the 
% original/starting parameters are probably pretty reasonable, so we
% discourage huge changes during the optimization. 
%
% Starting the optimization at 0 means that the calibration starts off
% precisely equal to the original/baseline calibration. This is intuitively
% natural. Randomizing the starting point by a small amount might 
% check/confirm that the optimization is robust. If the results are very 
% different for the randomized starting point it may indicate overly
% an sensitive parameter landscape.
%
% The most important number to focus on in the output is the err(noreg) 
% value (the first number not-in-a-bracket). This is the residual 
% projection error after each optimization, in pixels. This number can be 
% compared with the "before" error (again, number not-in-brackets).

% Some different flavors of objective functions 
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
      fminsearch(@(x) feval(objFun,x,tFP.yL,tFP.yR,tFP.yB,crig2,lambda,{'silent',true}),x0,opts);
    [err,errL,errR,errB,errreg,~,~,~,errFull] = feval(objFun,x1,tFP.yL,tFP.yR,tFP.yB,crig2,lambda,{});
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

%% (Skippable) Try to visualize results 
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
  for i=1:numel(results.(f).x1s)
    xtmp = nan(1,24);
    xtmp(results.(f).idxStandard) = results.(f).x1s{i};    
    results.(f).x1s_std{i} = xtmp;
  end
end
%% (Skippable) Try to visualize results 
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

%% (Skippable) Try to visualize results 
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


%% Results 20160810
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

%% Generate an optimized calibration file
%
% After looking at optimization results, select a row (objective function,
% plus "type" 1-4) that you want to use/save. In August, we went with
% allExtBint and allExtAllInt as looking pretty good.
%
% AL20160810: allExtBint, with regularization, start at 0, shows good
% improvement and most typical pattern of x1.

type = 1; % With regularization, start at 0
x1use = results.objfunAllExtBint.x1s{type};
[~,~,~,~,~,~,~,~,~,crig2Mod] = objfunAllExtBint(x1use,tFP.yL,tFP.yR,tFP.yB,crig2,zeros(size(x1use)),{});

% Uncomment this to save calibration object
% save crig2Optimized_calibjun2916_roiTrackingJun22_20160810_2.mat crig2Mod

% Summary of "after" error
[~,~,errL,errR] = calibRoundTrip2(tFP.yL,tFP.yR,tFP.yB,crig2Mod);
fprintf('## After optim: [errL errR] err: [%.2f %.2f] %.2f\n',errL,errR,errL+errR);

%% AL20160810: allExtAllInt also looks good. No regularization, start at 0

type = 3; % No reguarization, start at 0
x1use = results.objfunAllExtAllInt.x1s{type};
[~,~,~,~,~,~,~,~,~,crig2AllExtAllInt] = objfunAllExtAllInt(x1use,tFP.yL,tFP.yR,tFP.yB,crig2,zeros(size(x1use)),{});

% save crig2Optimized_calibjun2916_roiTrackingJun22_20160810_AllExtAllInt.mat crig2AllExtAllInt