% Train function
%       + phisTr: training labels.
%       + bboxesTr: bounding boxes.
%       + IsTr: training images
%       + cpr_type: 1 for Cao et al 2013, 2 for Burgos-Artizzu et al 2013
%       (without occlusion) and 3 for Burgos-Artizzu et al 2013
%       (occlusion).
%       + model_type: 'larva' (Marta's larvae with two muscles and two
%       landmarks for muscle), 'mouse_paw' (Adam's mice with one landmarks in one
%       view), 'mouse_paw2' (Adam's mice with two landmarks, one in each
%       view), 'mouse_paw3D' (Adam's mice, one landmarks in the 3D
%       reconstruction), fly_RF2 (Romain's flies, six landmarks)
%       + feature type: for 1-4 see FULL_demoRCPR.m, 5 for points in an
%       elipse with focus in any pair of landmarks, and 6 for points in a
%       circunference around each landmark.
%       + radius: dimensions of the area where features are computed, for
%       feature_type=5 (recomended 1.5) is the semi-major axis, for
%       feature_type=6 is the radius of the circumference (recomended 25). 
%       + Prm3D: parameters for 3D rexontruction (empty if 2D).
%       + pStar: initial position of the the labels (optional)
%       + regModel: regression model
%       + regPrm: regression parameters (using the paramters recomended in
%       Burgos-Artizzu et al 2013).
%       + prunePrm: prune parameters using the paramters recomended in
%       Burgos-Artizzu et al 2013).

function [regModel,regPrm,prunePrm,phisPr,err]=train(phisTr,bboxesTr,IsTr,varargin)

% TODO: nothing is done with prunePrm other than returning it & crossval
prunePrm=struct('prune',1,'maxIter',2,'th',.5,'tIni',10,'numInit',5);
occlPrm=struct('nrows',3,'ncols',3,'nzones',1,'Stot',1,'th',.5);

% set default parameters
[cpr_type,model_type,model_nfids,model_d,model_nviews,...
  ftr_type,ftr_gen_radius,...
  Prm3D,pStar,...
  expidx,...
  cascade_depth,nferns,...
  naugment,augment_pad,...
  prunePrm,occlPrm,...
  nftrs_test_perfern,...
  nChn,...
  nferns_choose,fern_depth,fern_thresh,fern_regularization,...
  ncrossvalsets,...
  nsample_std,nsample_cor,cvidx,cvi] = ...
  myparse(varargin,...
  'cpr_type','noocclusion',...
  'model_type','mouse_paw3D',...
  'model_nfids',[],...
  'model_d',[],...
  'model_nviews',[],...
  'ftr_type',6,...
  'ftr_gen_radius',[],...
  'calibrationdata',[],...
  'pStar',[],...
  'expidx',[],...
  'cascade_depth',100,...
  'nferns',50,...
  'naugment',20,...
  'augment_pad',10,...
  'prunePrm',prunePrm,...
  'occlPrm',occlPrm,...
  'nftrs_test_perfern',400,...
  'nChn',1,...
  'nferns_choose',0,...
  'fern_depth',5,...
  'fern_thresh',.2,...
  'fern_regularization',.01,...
  'ncrossvalsets',0,...
  'nsample_std',1000,...
  'nsample_cor',5000,...
  'cvidx',[],'cvseti',[]);

if ischar(phisTr) && ischar(bboxesTr) && ischar(IsTr),
  
  paramfile1 = phisTr;
  paramfile2 = bboxesTr;
  savefile = IsTr;

  load(paramfile1);
  load(paramfile2);
  
else
  
  savefile = '';

end

% by default, minimum image size / 2
if isempty(ftr_gen_radius),
  sz = min(cellfun(@(x) min(size(x,1),size(x,2)),IsTr));
  ftr_gen_radius = sz / 4;
end

% for hold-out set splitting
if isempty(expidx),
  expidx = 1:size(phisTr,1);
end

%Create model
model = shapeGt('createModel',model_type,model_d,model_nfids,model_nviews);

if model.d == 3 && isempty(Prm3D),
  error('Calibration data must be input for 3d models');
end


switch cpr_type,
  case 'noocclusion',
    if occlPrm.Stot > 1,
      occlPrm.Stot = 1;
    end
  case 'occlusion',
    if occlPrm.Stot <= 1,
      occlPrm.Stot = 3;
    end
end

%RCPR(features+restarts) PARAMETERS
ftrPrm = struct('type',ftr_type,'F',nftrs_test_perfern,...
  'nChn',nChn,'radius',ftr_gen_radius,'nsample_std',nsample_std,...
  'nsample_cor',nsample_cor);

prm=struct('thrr',[-1 1]*fern_thresh,'reg',fern_regularization);
regPrm = struct('type',1,'K',nferns,'occlPrm',occlPrm,...
  'loss','L2','R',nferns_choose,'M',fern_depth,'model',model,'prm',prm,'ftrPrm',ftrPrm,...
  'Prm3D',Prm3D);

% TRAIN

%Train model
if ncrossvalsets > 1,

  if isempty(cvidx) && ~isempty(cvi),
    error('Cannot set cvi and not cvidx');
  end
  
  if isempty(cvidx),  
    cvidx = CVSet(expidx,ncrossvalsets);
  end

  if isempty(cvi),

    regModel = cell(1,ncrossvalsets);
    phisPr = nan(size(phisTr));
    
    for cvi = 1:ncrossvalsets,
      
      idxtrain = cvidx ~= cvi;
      idxtest = cvidx == cvi;
      
      fprintf('Training for cross-validation set %d / %d\n',cvi,ncrossvalsets);
      
      regModel{cvi} = train1(phisTr(idxtrain,:),bboxesTr(idxtrain,:),IsTr(idxtrain),...
        pStar,naugment,augment_pad,model,regPrm,ftrPrm,cascade_depth);
      
      if nargout >= 4 || ~isempty(savefile),
        phisPr(idxtest,:) = test_rcpr([],bboxesTr(idxtest,:),IsTr(idxtest),regModel{cvi},regPrm,prunePrm);
        errcurr = mean( sqrt(sum( (phisPr(idxtest,:)-phisTr(idxtest,:)).^2, 2)) );
        fprintf('Error for validation set %d = %f\n',cvi,errcurr);
      end
    
    end
    err = mean( sqrt(sum( (phisPr-phisTr).^2, 2)) );
    
  else
    
    idxtrain = cvidx ~= cvi;
    idxtest = cvidx == cvi;
    
    fprintf('Training for cross-validation set %d / %d\n',cvi,ncrossvalsets);
    
    regModel = train1(phisTr(idxtrain,:),bboxesTr(idxtrain,:),IsTr(idxtrain),...
      pStar,naugment,augment_pad,model,regPrm,ftrPrm,cascade_depth);

    phisPr = test_rcpr([],bboxesTr(idxtest,:),IsTr(idxtest),regModel,regPrm,prunePrm);
    errcurr = mean( sqrt(sum( (phisPr-phisTr(idxtest,:)).^2, 2)) );
    fprintf('Error for validation set %d = %f\n',cvi,errcurr);
    
  end
  
  
else
  
  cvidx = true(size(expidx)); %#ok<NASGU>
  
  regModel = train1(phisTr,bboxesTr,IsTr,...
    pStar,naugment,augment_pad,model,regPrm,ftrPrm,cascade_depth);
  
  if nargout >= 4 || ~isempty(savefile),
    phisPr = test_rcpr([],bboxesTr,IsTr,regModel,regPrm,prunePrm);
    err = mean( sqrt(sum( (phisPr-phisTr).^2, 2)) );
  end
    
end

if ~isempty(savefile),
  
  save(savefile,'regModel','regPrm','prunePrm','phisPr','err','paramfile1','paramfile2','cvidx');
  
end

function regModel = train1(phisTr,bboxesTr,IsTr,pStar,naugment,augment_pad,...
    model,regPrm,ftrPrm,cascade_depth)
  
% augment data
[pCur,pGt,pGtN,pStar,imgIds,N,N1]=shapeGt('initTr',...
  IsTr,phisTr,model,pStar,bboxesTr,naugment,augment_pad);
initData=struct('pCur',pCur,'pGt',pGt,'pGtN',pGtN,'pStar',pStar,...
  'imgIds',imgIds,'N',N,'N1',N1);

%Create training structure
trPrm=struct('model',model,'pStar',[],'posInit',bboxesTr,...
  'T',cascade_depth,'L',naugment,'regPrm',regPrm,'ftrPrm',ftrPrm,...
  'pad',augment_pad,'verbose',1,'initData',initData);
if model.d == 3,
  trPrm.Prm3D=Prm3D;
end
  
[regModel,~] = rcprTrain(IsTr,phisTr,trPrm);
  
    
function cvidx = CVSet(expidx,ncrossvalsets)
      
% choose cross-validation set memberships
[uniqueidx,~,expidx1] = unique(expidx);
nuniqueidx = numel(uniqueidx);
order = randperm(nuniqueidx);
expidx1 = order(expidx1);
counts = hist(expidx1,1:nuniqueidx);
N = numel(expidx);
s = [0,cumsum(counts)];
j = 1;
cvidx = nan(1,N);

for i = 1:ncrossvalsets,
  
  nperset = round((N-s(j))/(ncrossvalsets-i+1));
  [~,k] = min(abs(s(j+1:end)-s(j) - nperset));
  k = j + k;
  cvidx(ismember(expidx1,j:k-1)) = i;
  j = k;
  
end

