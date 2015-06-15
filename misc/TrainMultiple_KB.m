% Train the three models needed for Adam's mice.
%   - Train (if the first loaded file contains phisTr and bboxesTr)
%       + The second file must contain the frames for training (IsTr)
%       + doeq and loadH0 determine if the frames are equalized and if the
%       base histogram is loaded or computed.
%       + cpr_type: 1 for Cao et al 2013, 2 for Burgos-Artizzu et al 2013
%       (without occlusion) and 2 for Burgos-Artizzu et al 2013
%       (occlusion).
%       + regModel,regPr and prunePrm (and H0 if equalizing) are saved in
%       separate files for each trained model.
%clear all
doeq = true;
loadH0 = false;
cpr_type = 2;
docacheims = true;
maxNTr = 5000;
radius = 100;

addpath ..;
addpath ../video_tracking;
addpath /groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect/misc;
addpath /groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect/filehandling;
addpath(genpath('/groups/branson/home/bransonk/tracking/code/piotr_toolbox_V3.02'));

defaultfolder = '/groups/branson/home/bransonk/tracking/code/rcpr/data';
defaultfile = 'M134labeleddata.mat';

[file,folder]=uigetfile('.mat',sprintf('Select training file containg clicked points (e.g. %s)',defaultfile),...
  defaultfolder);
if isnumeric(file),
  return;
end

ld = load(fullfile(folder,file));

% where to save the models
defaultFolderSave = folder;
[~,n] = fileparts(file);
defaultFileSave = fullfile(defaultfolder,[n,'_TrackerModel.mat']);
[file1,folder1]=uiputfile('*.mat','Save model to file',defaultFileSave);
file1=fullfile(folder1,file1);
file2=[file1(1:end-4),'_pt1.mat'];
file3=[file1(1:end-4),'_pt2.mat'];

%load(fullfile(folder,file),'phisTr','bboxesTr');

%% read in the training images

[~,n,e] = fileparts(file);
cachedimfile = fullfile(folder,[n,'_cachedims',e]);
if docacheims && exist(cachedimfile,'file'),
  load(cachedimfile,'IsTr');
  imsz = size(IsTr{1});
else
  
  IsTr = cell(1,numel(ld.expidx));
  off = 0;
  for expi = 1:numel(ld.expdirs),
    
    idxcurr = find(ld.expidx == expi);
    fprintf('Reading %d frames from experiment %d / %d: %s\n',numel(idxcurr),expi,numel(ld.expdirs),ld.expdirs{expi});
    
    if isempty(idxcurr),
      continue;
    end
    [readframe,nframes,fid,headerinfo] = get_readframe_fcn(fullfile(ld.expdirs{expi},ld.moviefilestr));
    im = readframe(1);
    imsz = size(im);
    ncolors = size(im,3);
    for i = 1:numel(idxcurr),
      
      f = ld.ts(idxcurr(i));
      im = readframe(f);
      if ncolors > 1,
        im = rgb2gray(im);
      end
      IsTr{i+off} = im;
      
    end
    off = off + numel(idxcurr);
    if fid > 0,
      fclose(fid);
    end
    
  end
  IsTr = IsTr(:);
  if docacheims,
    save('-v7.3',cachedimfile,'IsTr');
  end
end

%% subsample training data
% defaultFolderIs = folder;
% [~,n] = fileparts(file);
% defaultFileIs = fullfile(defaultFolderIs,[n,'_Is.mat']);
% if ~exist(defaultFileIs,'file'),
%   defaultFileIs = defaultFolderIs;
% end
% 
% [fileIs,folderIs]=uigetfile('.mat',sprintf('Select file containing images corresponding to points in %s',file),defaultFileIs);
% load(fullfile(folderIs,fileIs));
nTr=min(numel(IsTr),maxNTr);
if nTr < numel(IsTr),
  idx = SubsampleTrainingDataBasedOnMovement(ld,maxNTr);
  idx = idx(randperm(numel(idx)));
  nTr = numel(idx);
  %idx=randsample(numel(IsTr),nTr);
else
  idx = randperm(nTr);
end

%% histogram equalization

if doeq
  if loadH0
    [fileH0,folderH0]=uigetfile('.mat');
    load(fullfile(folderH0,fileH0));
  else
    H=nan(256,1500);
    for i=1:1500,
      H(:,i)=imhist(IsTr{idx(i)});
    end
    H0=median(H,2);
  end
  model1.H0=H0;
  % normalize one video at a time
  for expi = 1:numel(ld.expdirs),
    idxcurr = idx(ld.expidx(idx)==expi);
    bigim = cat(1,IsTr{idxcurr});
    bigimnorm = histeq(bigim,H0);
    IsTr(idxcurr) = mat2cell(bigimnorm,repmat(imsz(1),[1,numel(idxcurr)]),imsz(2));
  end
    
%   for i=1:nTr,
%     IsTr2{idx(i)}=histeq(IsTr{idx(i)},H0);
%   end
end

%% train

phisTr = reshape(permute(ld.pts,[3,1,2]),[size(ld.pts,3),size(ld.pts,1)*size(ld.pts,2)]);
bboxesTr = repmat([1,1,imsz([2,1])],[numel(idx),1]);

% Train first model
[testmodel2.regModel,testmodel2.regPrm,testmodel2.prunePrm,pred2,err2] = ...
  train(phisTr(idx,:),bboxesTr,IsTr(idx),'cpr_type','noocclusion','model_type',...
  'mouse_paw2','ftr_type',6,'ftr_gen_radius',radius,'expidx',ld.expidx(idx),'ncrossvalsets',5,...
  'naugment',5);

[model1.regModel,model1.regPrm,model1.prunePrm]=train(phisTr(idx,:),bboxesTr,IsTr(idx),cpr_type,'mouse_paw2',6,radius);
model1.regPrm.Prm3D=[];
model1.datafile=fullfile(folder,file);
save(file1,'-struct','model1')

% Train model for point 1
phisTr2 = phisTr(:,[1 3]);
bboxesTr2 = [phisTr2-40, 80*ones(size(phisTr2))];
[model2.regModel,model2.regPrm,model2.prunePrm]=train(phisTr2(idx,:),bboxesTr2(idx,:),IsTr(idx),cpr_type,'mouse_paw',6,radius);
model2.regPrm.Prm3D=[];
model2.datafile=fullfile(folder,file);
save(file2,'-struct','model2')

% Train model for point 2
phisTr3 = phisTr(:,[2 4]);
bboxesTr3 = [phisTr3-40, 80*ones(size(phisTr3))];
[model3.regModel,model3.regPrm,model3.prunePrm]=train(phisTr3(idx,:),bboxesTr3(idx,:),IsTr(idx),cpr_type,'mouse_paw',6,radius);
model3.regPrm.Prm3D=[];
model3.datafile=fullfile(folder,file);
save(file3,'-struct','model3')

%% test 

partSize = 100;

expi = nan;
expdir = '/tier2/hantman/Jay/videos/M147VGATXChrR2_anno/20150303L/CTR/M147_20150303_v017';
[readframe,nframes,fid,headerinfo] = get_readframe_fcn(fullfile(expdir,ld.moviefilestr));
firstframe = 1;
endframe = nframes;

im = readframe(1);
imsz = size(im);
ncolors = size(im,3);

bboxes=repmat([1 1 imsz(2) imsz(1)],endframe-firstframe+1,1);

% 1. First tracking
IsTe=cell(partSize,1);
p=nan(nframes,4);
for t_i=firstframe:partSize:endframe;
  off_i = t_i - firstframe + 1;
  t_f=min(t_i+partSize-1,endframe);
  off_f = t_f - firstframe + 1;
  nframescurr = t_f-t_i+1;
  fprintf('\n1st tracking: frames %i-%i\n',t_i,t_f);
  for k=1:nframescurr,
    t=t_i+k-1;
    im = readframe(t);
    if ncolors > 1,
      im = rgb2gray(im);
    end
    IsTe{k} = im;
  end
  bigim = cat(1,IsTe{1:nframescurr});
  bigimnorm = histeq(bigim,H0);
  IsTe(1:nframescurr) = mat2cell(bigimnorm,repmat(imsz(1),[1,nframescurr]),imsz(2));
  p(off_i:off_f,:)=test_rcpr([],bboxes(off_i:off_f,:),IsTe(1:nframescurr),model1.regModel,model1.regPrm,model1.prunePrm);
end

fprintf('Initial tracking results\n');

figure(1);
clf;
him = imagesc(im,[0,255]);
axis image;
colormap gray;
hold on;
htrx = nan(1,2);
htrx(1) = plot(nan,nan,'m-');
htrx(2) = plot(nan,nan,'c-');
hcurr = nan(1,2);
hcurr(1) = plot(nan,nan,'mo','LineWidth',3);
hcurr(2) = plot(nan,nan,'co','LineWidth',3);
htext = text(imsz(2)/2,5,'d. train = ??','FontSize',24,'HorizontalAlignment','center','VerticalAlignment','top','Color','r');

idxcurr = idx(ld.expidx(idx)==expi);
tstraincurr = ld.ts(idxcurr);

for t = firstframe:endframe,
  
  i = t - firstframe + 1;
  im = readframe(t);
  set(him,'CData',im);
  set(htrx(1),'XData',p(max(1,t-500):t,1),'YData',p(max(1,t-500):t,3));
  set(htrx(2),'XData',p(max(1,t-500):t,2),'YData',p(max(1,t-500):t,4));
  set(hcurr(1),'XData',p(t,1),'YData',p(t,3));
  set(hcurr(2),'XData',p(t,2),'YData',p(t,4));
  mindcurr = min(abs(t-tstraincurr));
  set(htext,'String',num2str(mindcurr));
  
  drawnow;
  
end

% 2. Smooth results and create bounding boxes
p_med=medfilt1(p,10);
p_med(1,:) = p(1,:);
bboxes_med1=[p_med(:,1)-40 p_med(:,3)-40 80*ones(size(p,1),2)];
bboxes_med2=[p_med(:,2)-40 p_med(:,4)-40 80*ones(size(p,1),2)];
    
% 3. Second tracking using the median of the previus tracking to
% create small bboxes.
IsTe=cell(partSize,1);
p_all=nan(nframes,4);
for t_i=firstframe:partSize:endframe;
  off_i = t_i - firstframe + 1;
  t_f=min(t_i+partSize-1,endframe);
  off_f = t_f - firstframe + 1;
  nframescurr = t_f-t_i+1;
  fprintf('\n2nd tracking: frames %i-%i\n',t_i,t_f)
  for k=1:nframescurr,
    t=t_i+k-1;
    im = readframe(t);
    if ncolors > 1,
      im = rgb2gray(im);
    end
    IsTe{k} = im;
  end
  bigim = cat(1,IsTe{1:nframescurr});
  bigimnorm = histeq(bigim,H0);
  IsTe(1:nframescurr) = mat2cell(bigimnorm,repmat(imsz(1),[1,nframescurr]),imsz(2));

  p_all(off_i:off_f,[1 3])=test_rcpr([],bboxes_med1(off_i:off_f,:),IsTe(1:nframescurr),model2.regModel,model2.regPrm,model2.prunePrm);
  p_all(off_i:off_f,[2 4])=test_rcpr([],bboxes_med2(off_i:off_f,:),IsTe(1:nframescurr),model3.regModel,model3.regPrm,model3.prunePrm);
  
  d1=sqrt(sum(diff(p_all(:,[1 3])).^2,2));
  d2=sqrt(sum(diff(p_all(:,[2 4])).^2,2));
  fprintf('\n%i jumps for point 1\n',sum(d1>20));
  fprintf('\n%i jumps for point 2\n',sum(d2>20));
        
  if fid>0
    fclose(fid);
  end
end


fprintf('Second tracking results\n');

clf;
him = imagesc(im,[0,255]);
axis image;
colormap gray;
hold on;
htrxold = nan(1,2);
htrxold(1) = plot(nan,nan,'g-');
htrxold(2) = plot(nan,nan,'r-');
htrx = nan(1,2);
htrx(1) = plot(nan,nan,'m-');
htrx(2) = plot(nan,nan,'c-');
hcurrold = nan(1,2);
hcurrold(1) = plot(nan,nan,'go','LineWidth',3);
hcurrold(2) = plot(nan,nan,'ro','LineWidth',3);
hcurr = nan(1,2);
hcurr(1) = plot(nan,nan,'mo','LineWidth',3);
hcurr(2) = plot(nan,nan,'co','LineWidth',3);
%htext = text(imsz(2)/2,5,'d. train = ??','FontSize',24,'HorizontalAlignment','center','VerticalAlignment','top','Color','r');

% idxcurr = idx(ld.expidx(idx)==expi);
% tstraincurr = ld.ts(idxcurr);

for t = firstframe:endframe,
  
  i = t - firstframe + 1;
  im = readframe(t);
  set(him,'CData',im);
  set(htrxold(1),'XData',p_all(max(1,t-500):t,1),'YData',p_all(max(1,t-500):t,3));
  set(htrxold(2),'XData',p_all(max(1,t-500):t,2),'YData',p_all(max(1,t-500):t,4));
  set(hcurrold(1),'XData',p_all(t,1),'YData',p_all(t,3));
  set(hcurrold(2),'XData',p_all(t,2),'YData',p_all(t,4));
  set(htrx(1),'XData',p(max(1,t-500):t,1),'YData',p(max(1,t-500):t,3));
  set(htrx(2),'XData',p(max(1,t-500):t,2),'YData',p(max(1,t-500):t,4));
  set(hcurr(1),'XData',p(t,1),'YData',p(t,3));
  set(hcurr(2),'XData',p(t,2),'YData',p(t,4));
%   mindcurr = min(abs(t-tstraincurr));
%   set(htext,'String',num2str(mindcurr));
  
  drawnow;
  
end

%% now try training 3d model

addpath /groups/branson/home/bransonk/codepacks/TOOLBOX_calib;
calib_file = '/groups/branson/home/bransonk/behavioranalysis/code/adamTracking/analysis/CameraCalibrationParams20150217.mat';
calibrationdata = load(calib_file);

% triangulate training data
ld3 = ld;
movie_file = fullfile(ld.expdirs{1},ld.moviefilestr);
readframe = get_readframe_fcn(movie_file);
ex_frame = readframe(1);
sz = size(ex_frame,2);
nfids = size(ld.pts,1)/2;
ld3.pts = nan([nfids,3,size(ld.pts,3)]);

errL = nan(numel(ld.expdirs),3);
errR = nan(numel(ld.expdirs),3);
for expi = 1:numel(ld.expdirs)
  idxcurr = ld.expidx == expi;
  % xL is npts/2 x 2 x N
  % xL(1,i,j) is the x-coord for left view, landmark i, example j
  xL = permute(ld.pts(1:2:end-1,:,idxcurr),[2,1,3]);
  xR = permute(ld.pts(2:2:end,:,idxcurr),[2,1,3]);
  xL = reshape(xL,[2,nfids*nnz(idxcurr)]);
  xR = reshape(xR,[2,nfids*nnz(idxcurr)]);
  xR(1,:) = xR(1,:) - sz/2; % compensate for joint frame.
  [~,~,~,mouseid] = regexp(ld.expdirs{expi},'M\d\d\d_');
  mouseid = mouseid{1}(1:end-1);
  mousendx = find(strcmp(calibrationdata.mice,mouseid));
%   omcurr = calibrationdata.ompermouse(:,mousendx);
%   Tcurr = calibrationdata.Tpermouse(:,mousendx);
  omcurr = calibrationdata.om0;
  Tcurr = calibrationdata.T0;
  [X3d,X3d_right]  = stereo_triangulation(xL,xR,omcurr,Tcurr,calibrationdata.fc_left,...
    calibrationdata.cc_left,calibrationdata.kc_left,calibrationdata.alpha_c_left,...
    calibrationdata.fc_right,calibrationdata.cc_right,calibrationdata.kc_right,...
    calibrationdata.alpha_c_right);
  [xL_re] = project_points2(X3d,zeros(size(omcurr)),zeros(size(Tcurr)),calibrationdata.fc_left,calibrationdata.cc_left,calibrationdata.kc_left,calibrationdata.alpha_c_left);
  [xR_re] = project_points2(X3d_right,zeros(size(omcurr)),zeros(size(Tcurr)),calibrationdata.fc_right,calibrationdata.cc_right,calibrationdata.kc_right,calibrationdata.alpha_c_right);
  d = sqrt(sum((xL_re-xL).^2,1));
  errL(expi,:) = [mean(d),min(d),max(d)];
  d = sqrt(sum((xR_re-xR).^2,1));
  errR(expi,:) = [mean(d),min(d),max(d)];

  ld3.pts(:,:,idxcurr) = permute(reshape(X3d,[3,nfids,nnz(idxcurr)]),[2,1,3]);
end

phisTr3 = reshape(permute(ld3.pts,[3,1,2]),[size(ld3.pts,3),size(ld3.pts,1)*size(ld3.pts,2)]);
minv = min(phisTr3,[],1);
maxv = max(phisTr3,[],1);
fracpad = .25;
radius3d = 100;
meandv = mean(maxv-minv);
bboxes0 = [minv-meandv*fracpad,maxv-minv+meandv*2*fracpad];

bboxesTr=repmat(bboxes0,numel(idx),1);

[model3d.regModel,model3d.regPrm,model3d.prunePrm]=train(phisTr3(idx,:),bboxesTr,IsTr(idx),cpr_type,'mouse_paw3D',7,radius3d,calibrationdata);

model3d.regPrm.Prm3D.bboxes0=bboxes0;

boxw3d = .15*min(maxv-minv);
bboxesTr2 = [phisTr3-boxw3d, 2*boxw3d*ones(size(phisTr3))];
[model3d_2.regModel,model3d_2.regPrm,model3d_2.prunePrm]=train(phisTr3(idx,:),bboxesTr2(idx,:),IsTr(idx),cpr_type,'mouse_paw3D',7,radius3d,calibrationdata);
model3d_2.regPrm.Prm3D.bboxes0=bboxesTr2(idx,:);

%% test 3d

partSize = 100;

expi = nan;
expdir = '/tier2/hantman/Jay/videos/M147VGATXChrR2_anno/20150303L/CTR/M147_20150303_v017';
[readframe,nframes,fid,headerinfo] = get_readframe_fcn(fullfile(expdir,ld.moviefilestr));
firstframe = 1;
endframe = nframes;

im = readframe(1);
imsz = size(im);
ncolors = size(im,3);

bboxes=repmat(bboxes0,[endframe-firstframe+1,1]);

% 1. First tracking
IsTe=cell(partSize,1);
p3d=nan(nframes,3);
for t_i=firstframe:partSize:endframe;
  off_i = t_i - firstframe + 1;
  t_f=min(t_i+partSize-1,endframe);
  off_f = t_f - firstframe + 1;
  nframescurr = t_f-t_i+1;
  fprintf('\n1st tracking: frames %i-%i\n',t_i,t_f);
  for k=1:nframescurr,
    t=t_i+k-1;
    im = readframe(t);
    if ncolors > 1,
      im = rgb2gray(im);
    end
    IsTe{k} = im;
  end
  bigim = cat(1,IsTe{1:nframescurr});
  bigimnorm = histeq(bigim,H0);
  IsTe(1:nframescurr) = mat2cell(bigimnorm,repmat(imsz(1),[1,nframescurr]),imsz(2));
  p3d(off_i:off_f,:)=test_rcpr([],bboxes(off_i:off_f,:),IsTe(1:nframescurr),model3d.regModel,model3d.regPrm,model3d.prunePrm);
end

fprintf('Initial tracking results\n');

figure(1);
clf;
him = imagesc(im,[0,255]);
axis image;
colormap gray;
hold on;
htrxold = nan(1,2);
htrxold(1) = plot(nan,nan,'g-');
htrxold(2) = plot(nan,nan,'r-');
htrx = nan(1,2);
htrx(1) = plot(nan,nan,'m-');
htrx(2) = plot(nan,nan,'c-');
hcurrold = nan(1,2);
hcurrold(1) = plot(nan,nan,'go','LineWidth',3);
hcurrold(2) = plot(nan,nan,'ro','LineWidth',3);
hcurr = nan(1,2);
hcurr(1) = plot(nan,nan,'mo','LineWidth',3);
hcurr(2) = plot(nan,nan,'co','LineWidth',3);
%htext = text(imsz(2)/2,5,'d. train = ??','FontSize',24,'HorizontalAlignment','center','VerticalAlignment','top','Color','r');

% idxcurr = idx(ld.expidx(idx)==expi);
% tstraincurr = ld.ts(idxcurr);

xL_re = project_points2(permute(p3d,[2,1]),zeros(size(omcurr)),zeros(size(Tcurr)),calibrationdata.fc_left,calibrationdata.cc_left,calibrationdata.kc_left,calibrationdata.alpha_c_left);
xR_re = project_points2(permute(p3d,[2,1]),calibrationdata.om0,calibrationdata.T0,calibrationdata.fc_right,calibrationdata.cc_right,calibrationdata.kc_right,calibrationdata.alpha_c_right);
xR_re(1,:) = xR_re(1,:) + sz/2;

for t = firstframe:endframe,
  
  i = t - firstframe + 1;
  im = readframe(t);
  set(him,'CData',im);
  set(htrxold(1),'XData',p_all(max(1,t-500):t,1),'YData',p_all(max(1,t-500):t,3));
  set(htrxold(2),'XData',p_all(max(1,t-500):t,2),'YData',p_all(max(1,t-500):t,4));
  set(hcurrold(1),'XData',p_all(t,1),'YData',p_all(t,3));
  set(hcurrold(2),'XData',p_all(t,2),'YData',p_all(t,4));
  
  set(htrx(1),'XData',xL_re(1,max(1,t-500):t),'YData',xL_re(2,max(1,t-500):t));
  set(htrx(2),'XData',xR_re(1,max(1,t-500):t),'YData',xR_re(2,max(1,t-500):t));
  set(hcurr(1),'XData',xL_re(1,t),'YData',xL_re(2,t));
  set(hcurr(2),'XData',xR_re(1,t),'YData',xR_re(2,t));
%   mindcurr = min(abs(t-tstraincurr));
%   set(htext,'String',num2str(mindcurr));
  
  drawnow;
  
end

% 2. Smooth results and create bounding boxes
p_med3=medfilt1(p3d,10);
p_med3(1,:) = p3d(1,:);
bboxes_med3d=[p_med3-boxw3d 2*boxw3d*ones(size(p3d))];
    
% 3. Second tracking using the median of the previus tracking to
% create small bboxes.
IsTe=cell(partSize,1);
p3d_2=nan(nframes,3);
for t_i=firstframe:partSize:endframe;
  off_i = t_i - firstframe + 1;
  t_f=min(t_i+partSize-1,endframe);
  off_f = t_f - firstframe + 1;
  nframescurr = t_f-t_i+1;
  fprintf('\n2nd tracking: frames %i-%i\n',t_i,t_f)
  for k=1:nframescurr,
    t=t_i+k-1;
    im = readframe(t);
    if ncolors > 1,
      im = rgb2gray(im);
    end
    IsTe{k} = im;
  end
  bigim = cat(1,IsTe{1:nframescurr});
  bigimnorm = histeq(bigim,H0);
  IsTe(1:nframescurr) = mat2cell(bigimnorm,repmat(imsz(1),[1,nframescurr]),imsz(2));

  p3d_2(off_i:off_f,:)=test_rcpr([],bboxes_med3d(off_i:off_f,:),IsTe(1:nframescurr),model3d_2.regModel,model3d_2.regPrm,model3d_2.prunePrm);
          
  if fid>0
    fclose(fid);
  end
end

figure(1);
clf;
hax = axes('Position',[0,0,1,1]);
him = imagesc(im,[0,255]);
axis image;
truesize;
colormap gray;
hold on;
htrxold = nan(1,2);
htrxold(1) = plot(nan,nan,'-','Color',[0,.6,0],'LineWidth',2);
htrxold(2) = plot(nan,nan,'-','LineWidth',2,'Color',[.6,0,0]);
htrx = nan(1,2);
htrx(1) = plot(nan,nan,'-','LineWidth',2,'Color',[.6,0,.6]);
htrx(2) = plot(nan,nan,'c-','LineWidth',2,'Color',[0,.6,.6]);
hcurrold = nan(1,2);
hcurrold(1) = plot(nan,nan,'go','LineWidth',3,'MarkerSize',12);
hcurrold(2) = plot(nan,nan,'ro','LineWidth',3,'MarkerSize',12);
hcurr = nan(1,2);
hcurr(1) = plot(nan,nan,'mo','LineWidth',3,'MarkerSize',12);
hcurr(2) = plot(nan,nan,'co','LineWidth',3,'MarkerSize',12);
%htext = text(imsz(2)/2,5,'d. train = ??','FontSize',24,'HorizontalAlignment','center','VerticalAlignment','top','Color','r');

% idxcurr = idx(ld.expidx(idx)==expi);
% tstraincurr = ld.ts(idxcurr);

xL_re2 = project_points2(permute(p3d_2,[2,1]),zeros(size(omcurr)),zeros(size(Tcurr)),calibrationdata.fc_left,calibrationdata.cc_left,calibrationdata.kc_left,calibrationdata.alpha_c_left);
xR_re2 = project_points2(permute(p3d_2,[2,1]),calibrationdata.om0,calibrationdata.T0,calibrationdata.fc_right,calibrationdata.cc_right,calibrationdata.kc_right,calibrationdata.alpha_c_right);
xR_re2(1,:) = xR_re2(1,:) + sz/2;

savevideo = true;

if savevideo,
  vidobj = VideoWriter('test3dtracking.avi');
  open(vidobj);
end

for t = firstframe:endframe,
  
  i = t - firstframe + 1;
  im = readframe(t);
  set(him,'CData',im);
  
  set(htrxold(1),'XData',p_all(max(1,t-50):t,1),'YData',p_all(max(1,t-50):t,3));
  set(htrxold(2),'XData',p_all(max(1,t-50):t,2),'YData',p_all(max(1,t-50):t,4));
  set(hcurrold(1),'XData',p_all(t,1),'YData',p_all(t,3));
  set(hcurrold(2),'XData',p_all(t,2),'YData',p_all(t,4));
  
  set(htrx(1),'XData',xL_re2(1,max(1,t-50):t),'YData',xL_re2(2,max(1,t-50):t));
  set(htrx(2),'XData',xR_re2(1,max(1,t-50):t),'YData',xR_re2(2,max(1,t-50):t));
  set(hcurr(1),'XData',xL_re2(1,t),'YData',xL_re2(2,t));
  set(hcurr(2),'XData',xR_re2(1,t),'YData',xR_re2(2,t));
%   mindcurr = min(abs(t-tstraincurr));
%   set(htext,'String',num2str(mindcurr));
  
  drawnow;
  if savevideo,
    fr = getframe(1);
    writeVideo(vidobj,fr);
  end
  
end

close(vidobj);