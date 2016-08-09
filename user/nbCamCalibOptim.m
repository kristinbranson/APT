%%
ROOTDIR = 'f:\Dropbox\MultiViewFlyLegTracking\multiview labeling';
assert(strcmp(pwd,ROOTDIR));

LBL = fullfile(ROOTDIR,'romainMV2_safeLabels.lbl');
NVIEW = 3;
NREALPT = 57/3;
CRIG = fullfile(ROOTDIR,'calibratedrig_calibjun2916_roiTrackingJun22_20160707.mat');

lbl = load(LBL,'-mat');
crig = load(CRIG,'-mat');
crig = crig.crig;
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
  calibRoundTrip(yL,yR,yB,crig);
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
[~,~,~,errL,errR,errB] = calibRoundTrip(yL,yR,yB,crig);
err = errL + errR + errB;
fprintf('## Before optim: [errL errR errB] err: [%.2f %.2f %.2f] %.2f\n',...
  errL,errR,errB,err);
  
%%
NX = 16;
% Set lambda (regularization)
% Let's say, if any param changes by ~10% that's a big change, equivalent
% to a 2 px error in round-tripped projected pts
% If dParams are all ~10%*Params

LAMBDAS = {
  2*10*ones(NX,1);
  zeros(NX,1)
  };
X0S = {
  zeros(NX,1);
  .1*rand(NX,1)-0.05;
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
    fminsearch(@(x) objfunIntExtRot3(x,yL,yR,yB,crig,lambda,{'silent',true}),x0,opts);
  [err,errL,errR,errB,errreg] = objfunIntExtRot3(x1,yL,yR,yB,crig,lambda,{});
  fprintf('## After optim: [errL errR errB] err(noreg) errReg: [%.2f %.2f %.2f] %.2f %.2f\n',...
    errL,errR,errB,errL+errR+errB,errreg);
  
  x1s{end+1,1} = x1;
  fval{end+1,1} = fv;
  errs{end+1,1} = [errL errR errB errreg err];
end
end


%%
dp0 = dparams;

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

