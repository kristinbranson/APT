%% 3D training and prediction.

%% Files and variables

calib_file = '/home/mayank/Dropbox/AdamVideos/multiPoint/CameraCalibrationParams20150217.mat';
data_file = '../../data/mouse/Data/multiPoint_All_local.mat';
data_outfile = '../../data/mouse/Data/multiPoint3D_All.mat';
train_outfile = '../../data/mouse/Data/multiPoint3D_All_train.mat';
test_outfile = '../../data/mouse/Data/multiPoint3D_All_test.mat';
model_file = '../../data/mouse/Data/multiPoint3D_All_model.mat';
tracking_result_file = '../../data/mouse/Data/multiPoint3D_tracks.mat';
mouse_model = 'mouse_paw3D';
H0_n = 400;
RT1 = 5; % number of initializations for test

%% Find the 3-d coordinates

calibrationdata = load(calib_file);
J = load(data_file);
J_out = J;
movie_file = fullfile(J.expdirs{1},'movie_comb.avi');
readframe = get_readframe_fcn(movie_file);
ex_frame = readframe(10);
sz = size(ex_frame,2);
for expi = 1:numel(J.expdirs)
  xL = permute(J.labeledpos_perexp{expi}(1,:,:),[2 3 1]);
  xR = permute(J.labeledpos_perexp{expi}(2,:,:),[2 3 1]);
  xR(1,:) = xR(1,:) - sz/2; % compensate for joint frame.
  [~,~,~,mouseid] = regexp(J.expdirs{expi},'M\d\d\d_');
  mouseid = mouseid{1}(1:end-1);
  mousendx = find(strcmp(calibrationdata.mice,mouseid));
  omcurr = calibrationdata.ompermouse(:,mousendx);
  Tcurr = calibrationdata.Tpermouse(:,mousendx);
  [X3d,X3d_right]  = stereo_triangulation(xL,xR,omcurr,Tcurr,calibrationdata.fc_left,...
         calibrationdata.cc_left,calibrationdata.kc_left,calibrationdata.alpha_c_left,...
         calibrationdata.fc_right,calibrationdata.cc_right,calibrationdata.kc_right,...
         calibrationdata.alpha_c_right);
%  X3d = bsxfun(@minus,X3d,calibrationdata.origin(:,mousendx));       
  J_out.labeledpos_perexp{expi} = permute(X3d,[3 1 2]);
end
save(data_outfile,'-struct','J_out');

%% split into training and testing.
% same days should be together.

uid_strs = {};
uids = nan(1,numel(J.expdirs));
for expi = 1:numel(J.expdirs)
  [~,~,~,curuidstr] = regexp(J.expdirs{expi},'M\d\d\d_\d{8}');
  curuidstr = curuidstr{1}(1:end);
  curuid = find(strcmp(uid_strs,curuidstr));
  if isempty(curuid),
    uid_strs{end+1} = curuidstr;
    curuid = numel(uid_strs); 
  end
  uids(expi) = curuid;
end

cvp = cvpartition(numel(uid_strs),'k',5);

k_num = 1; % partition number of cvp.

allndx = 1:numel(uid_strs);
curTrain = ismember(uids,allndx(cvp.training(k_num)));
curTest = ismember(uids,allndx(cvp.test(k_num)));

prepareTrainingFiles(data_outfile,train_outfile,find(curTrain),500);
prepareTrainingFiles(data_outfile,test_outfile,find(curTest),500);

%% Load the data

% labels
load(train_outfile);
[dirn,filen,ext] = fileparts(train_outfile);
imgfile = fullfile(dirn,[filen '_Is' ext]);
% images
load(imgfile);

%% Do the bbox and H0 calculation
%bboxes
bboxes0=[min(phisTr)-10 max(phisTr)-min(phisTr)+20];
bboxesTr=repmat(bboxes0,numel(IsTr),1);

npts = size(phisTr,1);
h0_samples = randsample(npts,H0_n);
% Compute H0
H = nan(256,H0_n);
count = 1;
for i=h0_samples(:)'
  H(:,count)=imhist(IsTr{i});
  count = count + 1;
end
H0=median(H,2);

for ndx = 1:numel(IsTr)
  IsTr{ndx} = histeq(IsTr{ndx},H0);
end

%% Train and save

[regModel,regPrm,prunePrm]=train(phisTr,bboxesTr,IsTr,2,'mouse_paw3D',7,25,calibrationdata);
regPrm.Prm3D.bboxes0=bboxes0;

save(model_file,'regModel','regPrm','prunePrm','H0');

%% Test

T = load(test_outfile);
[dirn,filen,ext] = fileparts(test_outfile);
imgfile = fullfile(dirn,[filen '_Is' ext]);
% images
I = load(imgfile);
nTest = numel(I.IsTr);

bboxesTest = repmat(bboxes0,numel(I.IsTr),1);

%Initialize randomly using RT1 shapes drawn from training
piT=shapeGt('initTest',I.IsTr,bboxesTest,regModel.model,regModel.pStar,regModel.pGtN,RT1);

%Create test struct
testPrmT = struct('RT1',RT1,'pInit',bboxesTest,...
    'regPrm',regPrm,'initData',piT,'prunePrm',prunePrm,...
    'verbose',0);
%Test
t=clock;
[pT3D_temp,pRTT3D] = rcprTest(I.IsTr,regModel,testPrmT);
t=etime(clock,t);
nfids = size(pT3D_temp,2)/3;
pT3D = [];
pT3D(1,:) = pT3D_temp(:,1:nfids);
pT3D(2,:) = pT3D_temp(:,nfids+1:2*nfids);
pT3D(3,:) = pT3D_temp(:,2*nfids+1:3*nfids);

modeltest = shapeGt('createModel','mouse_paw3D');
loss=shapeGt('dist',modeltest,pT3D',T.phisTr);

%% Measure accuracy -- ignore. kept it for code.
if false
  pT3D_right = rodrigues(calibrationdata.om0)*pT3D + repmat(calibrationdata.T0,[1 size(pT3D,2)]);
  pixel_left = project_points(pT3D,zeros(3,1),zeros(3,1),calibrationdata.fc_left,...
    calibrationdata.cc_left,calibrationdata.kc_left);
  pixel_right = project_points(pT3D_right,zeros(3,1),zeros(3,1),calibrationdata.fc_right,...
    calibrationdata.cc_right,calibrationdata.kc_right);
  
  phisPred = [];
  phisPred(:,1:nfids) = pixel_left(1,:);
  phisPred(:,nfids+1:nfids*2) = pixel_right(1,:);
  phisPred(:,nfids*2+1:nfids*3) = pixel_left(2,:);
  phisPred(:,nfids*3+1:4*nfids) = pixel_right(2,:);
  
  modeltest = shapeGt('createModel','mouse_paw2');
  loss=shapeGt('dist',modeltest,phisPred,Q.phisTr);
end
%% predict on a movie and store the resutls in 2D

moviefiles = {};
p_all = {};
parfor ndx = 1:numel(T.expdirs_all)
  moviefile = fullfile(T.expdirs_all{ndx},'movie_comb.avi');
  pT3D = track_video('moviefile',moviefile,'model',model_file,'H0_file',model_file,'bboxes',regPrm.Prm3D.bboxes0);
  
  pT3D = pT3D';
  pT3D_right = rodrigues(calibrationdata.om0)*pT3D + repmat(calibrationdata.T0,[1 size(pT3D,2)]);
  pixel_left = project_points(pT3D,zeros(3,1),zeros(3,1),calibrationdata.fc_left,...
    calibrationdata.cc_left,calibrationdata.kc_left);
  pixel_right = project_points(pT3D_right,zeros(3,1),zeros(3,1),calibrationdata.fc_right,...
    calibrationdata.cc_right,calibrationdata.kc_right);
  phisPred = [];
  phisPred(:,1:nfids) = pixel_left(1,:);
  phisPred(:,nfids+1:nfids*2) = pixel_right(1,:)+sz/2;
  phisPred(:,nfids*2+1:nfids*3) = pixel_left(2,:);
  phisPred(:,nfids*3+1:4*nfids) = pixel_right(2,:);
  moviefiles{ndx} = moviefile;
  p_all{ndx} = phisPred;
end

save(tracking_result_file,'p_all','moviefiles');

%%
Labeler_fix(p_all,moviefiles);

