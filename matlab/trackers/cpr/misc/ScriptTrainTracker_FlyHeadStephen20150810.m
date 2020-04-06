% try a variety of parameters

%clear all
doeq = false;
doscale = true;
loadH0 = false;
cpr_type = 2;
docacheims = false;
maxNTr = 10000;
winrad = 475;
docurate = false;

addpath ..;
addpath ../video_tracking;
addpath(genpath('/groups/branson/home/bransonk/tracking/code/piotr_toolbox_V3.02'));
addpath /groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect/misc;
addpath /groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect/filehandling;
addpath /groups/branson/bransonlab/projects/flyHeadTracking/code;

defaultfolder = '/groups/branson/home/bransonk/tracking/code/rcpr/data';
%defaultfile = 'M134_M174_20150423.mat';
defaultfile = 'FlyHeadStephenTestData_20150813.mat';

[file,folder]=uigetfile('.mat',sprintf('Select training file containg clicked points (e.g. %s)',defaultfile),...
  fullfile(defaultfolder,defaultfile));
if isnumeric(file),
  return;
end

[~,savestr] = fileparts(file);

ld = load(fullfile(folder,file));

%% opening the videos is slow, so read in the frames

ims = cell(2,nTr0);
for expi = 1:numel(ld.vid1files),
  idxcurr = find(ld.expidx==expi);
  if isempty(idxcurr),
    continue;
  end
  fprintf('Exp %d / %d\n',expi,numel(ld.vid1files));
  fprintf('Opening %s...\n',ld.vid1files{expi});
  vidobj = VideoReader(ld.vid1files{expi});
  for i = idxcurr(:)',
    t = ld.ts(i);
    ims{1,i} = read(vidobj,t);
  end
  fprintf('Opening %s...\n',ld.vid2files{expi});
  vidobj = VideoReader(ld.vid2files{expi});
  for i = idxcurr(:)',
    t = ld.ts(i);
    ims{2,i} = read(vidobj,t);
  end
  
end

% offset 
for i = 1:numel(ld.expidx),
  ld.pts(1,2,:,i) = ld.pts(1,2,:,i) + size(ims{1,i},2);
end

%% reshape the data

[d0,nviews0,npts0,nTr0] = size(ld.pts);
% (x,y) x (p1_v1,p1_v2,p2_v1,p2_v2,...) x N
ld.pts = reshape(ld.pts,[d0,nviews0*npts0,nTr0]);
ld.pts = permute(ld.pts,[2,1,3]);

%% curate training data

if docurate,
  
  hfig = 1;
  figure(hfig);
  clf;
  hax = gca;
  
  npts = size(ld.pts,1);
  colors = jet(npts);
  
  isgooddata = nan(1,numel(ld.expidx));
  for i = 1:numel(ld.expidx),

    im = cat(2,ims{1,i},ims{2,i});
    hold(hax,'off');
    imagesc(im,'Parent',hax,[0,255]);
    axis(hax,'image','off');
    hold(hax,'on');
    colormap gray;
    for j = 1:npts,
      plot(hax,ld.pts(j,1,i),ld.pts(j,2,i),'wo','MarkerFaceColor',colors(j,:));
    end
    
    [~,n] = fileparts(ld.matfiles{ld.expidx(i)});
    text(5,5,sprintf('%s, %d',n,t),'HorizontalAlignment','left','VerticalAlignment','top','Parent',hax,'Color','w','Interpreter','none');
    
    while true,
      res = input('Correct = 1, incorrect = 0: ');
      if ismember(res,[0,1]),
        break;
      end
    end
    isgooddata(i) = res;
    
  end
  
  ld0 = ld;
  ld.pts = ld.pts(:,:,isgooddata~=0);
  ld.ts = ld.ts(isgooddata~=0);
  ld.expidx = ld.expidx(isgooddata~=0);
  
end


%% subsample training data

nTr=min(numel(ld.expidx),maxNTr);
if nTr < numel(ld.expidx),
  idx = SubsampleTrainingDataBasedOnMovement(ld,maxNTr);
  idx = idx(randperm(numel(idx)));
  nTr = numel(idx);
  %idx=randsample(numel(IsTr),nTr);
else
  idx = randperm(nTr);
end

%% plot a subset of the training data

hfig = 2;
figure(hfig);
clf;
nr = 8;
nc = 6;
nplot = nr*nc;
idxsample = sort(idx(unique(round(linspace(1,numel(idx),nplot)))));
nplot = numel(idxsample);
hax1 = createsubplots(nr,nc,0);

npts = size(ld.pts,1);
colors = jet(npts);

for ii = 1:nplot,
  i = idxsample(ii);
  t = ld.ts(i);
  hax = hax1(ii);
  im = cat(2,ims{1,i},ims{2,i});
  hold(hax,'off');
  imagesc(im,'Parent',hax,[0,255]);
  axis(hax,'image','off');
  hold(hax,'on');
  colormap gray;
  for j = 1:npts,
    plot(hax,ld.pts(j,1,i),ld.pts(j,2,i),'.','Color',colors(j,:));
  end
  
  [~,n] = fileparts(ld.matfiles{ld.expidx(i)});
  text(5,5,sprintf('%s, %d',n,t),'HorizontalAlignment','left','VerticalAlignment','top','Parent',hax,'Color','w','Interpreter','none');
  
end

[~,n] = fileparts(file);
savefig(fullfile(savedir,sprintf('%s_trainingdata.pdf',n)),hfig,'pdf');

%% read in the training images

[~,n,e] = fileparts(file);
cachedimfile = fullfile(folder,[n,'_cachedims',e]);

if docacheims && exist(cachedimfile,'file'),
  load(cachedimfile,'IsTr');
else
  
  IsTr = cell(1,numel(ld.expidx));
  for i = 1:numel(ld.expidx),
    IsTr{i} = rgb2gray(cat(2,ims{1,i},ims{2,i}));
  end
  IsTr = IsTr(:);
  if docacheims,
    save('-v7.3',cachedimfile,'IsTr');
  end
end

%% histogram equalization

if doeq
  if loadH0
    [fileH0,folderH0]=uigetfile('.mat');
    load(fullfile(folderH0,fileH0));
  else
    H=nan(256,nTr);
    mu = nan(1,nTr);
    for i=1:nTr,
      H(:,i)=imhist(IsTr{idx(i)});
      mu(i) = mean(IsTr{idx(i)}(:));
    end
    H0=median(H,2);
    H0 = H0/sum(H0)*numel(IsTr{1});
  end
  model1.H0=H0;
  % normalize one video at a time
  for expi = 1:numel(ld.expdirs),
    idxcurr = idx(ld.expidx(idx)==expi);
    if isempty(idxcurr),
      continue;
    end
    bigim = cat(1,IsTr{idxcurr});
    bigimnorm = histeq(bigim,H0);
    IsTr(idxcurr) = mat2cell(bigimnorm,repmat(imsz(1),[1,numel(idxcurr)]),imsz(2));
  end
    
%   for i=1:nTr,
%     IsTr2{idx(i)}=histeq(IsTr{idx(i)},H0);
%   end
elseif doscale,
  for expi = 1:numel(ld.matfiles),
    idxcurr = idx(ld.expidx(idx)==expi);
    if isempty(idxcurr),
      continue;
    end
    maxv = 0;
    for i = idxcurr,
      maxv = max(maxv,max(IsTr{i}(:)));
    end
    maxv = single(maxv);
    for i = idxcurr,
      IsTr{i} = uint8(single(IsTr{i})/maxv*255);
    end
  end
end

hfig = 3;
figure(hfig);
clf;
hax1 = createsubplots(nr,nc,0);

npts = size(ld.pts,1);
colors = jet(npts);

for ii = 1:nplot,
  i = idxsample(ii);
  t = ld.ts(i);
  hax = hax1(ii);
  im = IsTr{i};
  hold(hax,'off');
  imagesc(im,'Parent',hax,[0,255]);
  axis(hax,'image','off');
  hold(hax,'on');
  colormap gray;
  for j = 1:npts,
    plot(hax,ld.pts(j,1,i),ld.pts(j,2,i),'.','Color',colors(j,:));
  end
  
  [~,n] = fileparts(ld.matfiles{ld.expidx(i)});
  text(5,5,sprintf('%s, %d',n,t),'HorizontalAlignment','left','VerticalAlignment','top','Parent',hax,'Color','w','Interpreter','none');
  
end


%% save training data

% ld.pts is npts x (x,y) x nTr
% allPhisTr is nexamples x (npts*2)
allPhisTr = reshape(permute(ld.pts,[3,1,2]),[size(ld.pts,3),size(ld.pts,1)*size(ld.pts,2)]);
tmp2 = struct;
tmp2.phisTr = allPhisTr(idx,:);
imsz = cellfun(@(x) [size(x,2),size(x,1)],IsTr(idx),'Uni',0);
tmp2.bboxesTr = [ones(numel(idx),2),cat(1,imsz{:})];
tmp2.IsTr = IsTr(idx);
save(sprintf('TrainData_%s.mat',savestr),'-struct','tmp2');

%% choose neighbors

nneighbors = npts0-1;
ds = zeros(1,npts*(npts-1)/2);
for i = 1:numel(idx),
  x = reshape(allPhisTr(idx(i),:),[npts,2]);
  ds = ds + pdist(x);
end
ds = ds / numel(idx);
D = squareform(ds);
%D = bsxfun(@rdivide,D,sum(D,1));
D(eye(npts)==1) = nan;
ftr_neighbors = cell(1,npts);
for i = 1:npts,
  [~,order] = sort(D(i,:));
  ftr_neighbors{i} = order(1:nneighbors);  
end

%% train tracker

params = struct;
params.cpr_type = 'noocclusion';
params.model_type = 'FlyHeadStephen';
params.model_nfids = npts;
params.model_d = 2;
params.model_nviews = 1;
% params.ftr_type = 5;
% params.ftr_gen_radius = winrad;

params.ftr_type = 11;
params.cascade_depth = 100;
%params.ftr_type = 10;
%params.ftr_gen_radius = 25;
params.ftr_gen_radius = 25;%fliplr(round(logspace(log10(1),log10(10),params.cascade_depth)));
params.ftr_neighbors = ftr_neighbors;

params.expidx = ld.expidx(idx);

params.ncrossvalsets = 1;
params.naugment = 100;
params.nsample_std = 1000;
params.nsample_cor = 5000;
params.nferns = 50;
%params.fern_thresh = .2*2^(16-1);
params.fern_thresh = .2;
params.docomperr = false;
params.augment_dorotate = false;

params.prunePrm = struct;
params.prunePrm.prune = 0;
params.prunePrm.maxIter = 2;
params.prunePrm.th = 0.5000;
params.prunePrm.tIni = 10;
params.prunePrm.numInit = 50;
params.prunePrm.usemaxdensity = 1;
params.prunePrm.maxdensity_sigma = 5; 

paramsfile1 = fullfile('/groups/branson/home/bransonk/tracking/code/rcpr/rcpr_v1_stable/misc',...
  sprintf('TrainData_%s.mat',savestr));
paramsfile2 = sprintf('TrainParams_%s.mat',savestr);
trainresfile = sprintf('TrainedModel_%s.mat',savestr);

save(paramsfile2,'-struct','params');

[regModel,regPrm,prunePrm,phisPr,err] = train(paramsfile1,paramsfile2,trainresfile);

[idxinit,initlocs] = mykmeans(tmp2.phisTr,params.prunePrm.numInit,'Start','furthestfirst','Maxiter',100);
tmp = load(trainresfile);
tmp.prunePrm.initlocs = initlocs';
if doeq,
  tmp.H0 = H0;
end
tmp.doscale = doscale;
tmp.prunePrm.motion_2dto3D = false;
tmp.prunePrm.motionparams = {'poslambda',.5};
save(trainresfile,'-append','-struct','tmp');

%% test tracker on one labeled fly

expdir = {'/groups/branson/bransonlab/projects/flyHeadTracking/7_7_13_47D05AD_81B12DBD_x_Chrimsonattp18/data/fly_0037/fly_0037_trial_005/C001H001S0001/C001H001S0001.avi'
  '/groups/branson/bransonlab/projects/flyHeadTracking/7_7_13_47D05AD_81B12DBD_x_Chrimsonattp18/data/fly_0037/fly_0037_trial_005/C002H001S0001/C002H001S0001.avi'};
%expdir = {ld.vid1files{end-10},ld.vid2files{end-10}};
firstframe = 1;
endframe = 1000;
testresfile = '';
clear phisPr phisPrAll;
[phisPr,phisPrAll]=test({expdir},trainresfile,testresfile,'moviefilestr','',...
  'firstframe',firstframe,'endframe',endframe);%,'readframe',readframe,'nframes',nframes);

nfids = size(phisPr{1},2)/2;

[readframe,nframes,fid] = get_readframe_fcn(expdir);
endframe = min(endframe,nframes);
im = readframe(1);

figure(1);
clf;
hax = axes('Position',[0,0,1,1]);
him = imagesc(im,[0,255]);
axis image;
colormap gray;
hold on;
htrx = nan(1,nfids);
hcurr = nan(1,nfids);
colors = jet(nfids);
alphas = linspace(0,1,102)';
for i = 1:nfids,
  htrx(i) = patch(nan(102,1),nan(102,1),colors(i,:),...
    'EdgeAlpha','interp','EdgeColor',colors(i,:)*.7,'FaceColor','none',...
    'FaceVertexAlphaData',alphas,'LineWidth',2);
%  htrx(i) = plot(nan,nan,'-','Color',colors(i,:)*.7);
  hcurr(i) = plot(nan,nan,'.','Color',colors(i,:),'MarkerSize',24);
end
htext = text(5.5,5.5,'0 s','HorizontalAlignment','left','VerticalAlignment','top','Color',[.8,0,.8],'FontSize',36);
set(hax,'FontSize',24,'Color','k');
truesize;
axis off;
%ax = [0,0,0,0];
%border = 20;

box off;

dosave = false;

if dosave,
  n = expdir{1}(numel(rootdatadir)+2:end);
  n = regexp(n,'^([^/]*)/.*(fly_\d+_trial_\d+)/','tokens','once');
  n = [n{1},'_',n{2}];
  %[~,n] = fileparts(expdir{1});
  vidobj = VideoWriter(sprintf('TrackingResults_%s_20150811.avi',n));
  open(vidobj);
  gfdata = [];
end

for t = 1:endframe-firstframe+1,
  
  [im,timestamp] = readframe(firstframe+t-1);
  imsz = size(im);
  set(him,'CData',im);
  for i = 1:nfids,
    set(htrx(i),'XData',[phisPr{1}(max(1,t-100):t,i);nan],...
      'YData',[phisPr{1}(max(1,t-100):t,i+nfids);nan],...
      'FaceVertexAlphaData',alphas(max(1,102-t):end));
    set(hcurr(i),'XData',phisPr{1}(t,i),'YData',phisPr{1}(t,nfids+i));
  end

  set(htext,'String',sprintf('%.3f',timestamp));
  
  drawnow;
  if dosave,
    if isempty(gfdata),
      gfdata = getframe_initialize(hax);
      fr = getframe_invisible(hax);
      gfdata.sz = size(fr);
    end
    fr = getframe_invisible_nocheck(gfdata,gfdata.sz);
    writeVideo(vidobj,fr);
  end
  
end

if dosave,
close(vidobj);
end

%fclose(fid);

%% test tracker on more videos

rootdatadir = '/groups/branson/bransonlab/projects/flyHeadTracking';
expnames = {
  '7_7_13_47D05AD_81B12DBD_x_Chrimsonattp18'
  'SallyDigitizedVideos'
  };

testvid1files = {};
testvid2files = {};

framewidthrange = [1024,1024];
frameheightrange = [1024,1024];

usedvidfiles = ld.vid1files(unique(ld.expidx(idx)));

for expi = 1:numel(expnames),
  expdir = fullfile(rootdatadir,expnames{expi},'data');
  flydirs = mydir(expdir,'isdir',true,'name','^fly_\d+$');
    
  for flyi = 1:numel(flydirs),
    
    flydir = flydirs{flyi};
    trialdirs = mydir(flydir,'isdir',true,'name','_trial_\d+$');

    isgood = zeros(1,numel(trialdirs));
    for triali = 1:numel(trialdirs),
      trialdir = trialdirs{triali};
      
      [~,n] = fileparts(trialdir);
      m = regexp(n,'^fly_(\d+)_trial_(\d+)$','once','tokens');
      assert(~isempty(m));
      fly = str2double(m{1});
      trial = str2double(m{2});
      fprintf('Experiment: %s, fly %d, trial %d\n',expnames{expi},fly,trial);
      
      viddir1 = mydir(trialdir,'isdir',true,'name','^C001');
      assert(numel(viddir1)==1);
      viddir1 = viddir1{1};
      vidfile1 = mydir(viddir1,'name','^C001.*\.avi');
      assert(numel(vidfile1)==1);
      vidfile1 = vidfile1{1};
      vidinfo = aviinfo(vidfile1); %#ok<FREMO>
      if vidinfo.Width < framewidthrange(1) || vidinfo.Width > framewidthrange(2) || ...
          vidinfo.Height < frameheightrange(1) || vidinfo.Height > frameheightrange(2),
        continue;
      end      
      viddir2 = mydir(trialdir,'isdir',true,'name','^C002');
      assert(numel(viddir2)==1);
      viddir2 = viddir2{1};
      vidfile2 = mydir(viddir2,'name','^C002.*\.avi');
      assert(numel(vidfile2)==1);
      vidfile2 = vidfile2{1};
      vidinfo = aviinfo(vidfile2); %#ok<FREMO>
      if vidinfo.Width < framewidthrange(1) || vidinfo.Width > framewidthrange(2) || ...
          vidinfo.Height < frameheightrange(1) || vidinfo.Height > frameheightrange(2),
        continue;
      end
      if ismember(vidfile1,usedvidfiles),
        isgood(triali) = 1;
      else
        isgood(triali) = 2;
      end
    end
    
    if any(isgood>0),
      trialidx = find(isgood==max(isgood));
      triali = trialidx(randsample(numel(trialidx),1));
      testvid1files{end+1} = vidfile1;
      testvid2files{end+1} = vidfile2;
    end
    
  end
    
end

testresfiles = cell(1,numel(testvid1files));
parfor expi = 1:numel(testvid1files),
  expdir = {testvid1files{expi},testvid2files{expi}};
  testresdir = fileparts(fileparts(expdir{1}));
  testresfiles{expi} = fullfile(testresdir,sprintf('TrackingResults_%s.mat',savestr));

  test({expdir},trainresfile,testresfiles{expi},'moviefilestr','');
end

%% compute cross-validation error

params = struct;
params.cpr_type = 'noocclusion';
params.model_type = 'FlyHeadStephen';
params.model_nfids = npts;
params.model_d = 2;
params.model_nviews = 1;
% params.ftr_type = 5;
% params.ftr_gen_radius = winrad;

params.ftr_type = 11;
params.cascade_depth = 100;
%params.ftr_type = 10;
%params.ftr_gen_radius = 25;
params.ftr_gen_radius = fliplr(round(logspace(log10(1),log10(10),params.cascade_depth)));
params.ftr_neighbors = ftr_neighbors;

params.expidx = ld.expidx(idx);

ld.expflies = cell(1,numel(ld.vid1files));
for expi = 1:numel(ld.vid1files),
  n = ld.vid1files{expi}(numel(rootdatadir)+2:end);
  n = regexp(n,'^([^/]*)/.*(fly_\d+)_','tokens','once');
  n = [n{1},'_',n{2}];
  ld.expflies{expi} = n;
end
[ld.expflies,~,ld.exp2flyidx] = unique(ld.expflies);
ld.flyidx = ld.exp2flyidx(ld.expidx);
params.cvidx = ld.flyidx(idx);
params.ncrossvalsets = numel(ld.expflies);

params.naugment = 100;
params.nsample_std = 1000;
params.nsample_cor = 5000;
params.nferns = 50;
%params.fern_thresh = .2*2^(16-1);
params.fern_thresh = .2;
params.docomperr = false;
params.augment_dorotate = false;

params.prunePrm = struct;
params.prunePrm.prune = 0;
params.prunePrm.maxIter = 2;
params.prunePrm.th = 0.5000;
params.prunePrm.tIni = 10;
params.prunePrm.numInit = 50;
params.prunePrm.usemaxdensity = 1;
params.prunePrm.maxdensity_sigma = 5; 

paramsfile1 = fullfile('/groups/branson/home/bransonk/tracking/code/rcpr/rcpr_v1_stable/misc',...
  sprintf('TrainData_%s.mat',savestr0));
paramsfile2 = sprintf('TrainParams_%s.mat',savestr);
trainresfile = sprintf('TrainedModel_%s.mat',savestr);

save(paramsfile2,'-struct','params');

[regModel,regPrm,prunePrm,phisPr,err] = train(paramsfile1,paramsfile2,trainresfile);

[idxinit,initlocs] = mykmeans(tmp2.phisTr,params.prunePrm.numInit,'Start','furthestfirst','Maxiter',100);
tmp = load(trainresfile);
tmp.prunePrm.initlocs = initlocs';
if doeq,
  tmp.H0 = H0;
end
tmp.doscale = doscale;
tmp.prunePrm.motion_2dto3D = false;
tmp.prunePrm.motionparams = {'poslambda',.5};
save(trainresfile,'-append','-struct','tmp');


%% try ftr_gen_radius = 25

savestr = 'FlyHeadStephenTestData_ftrrad3_nferns50_20150813';

NCORESPERJOB = 1;
curdir = pwd;
TMP_ROOT_DIR = '/scratch/bransonk';
MCR_CACHE_ROOT = fullfile(TMP_ROOT_DIR,'mcr_cache_root');
MCR = '/groups/branson/bransonlab/share/MCR/v717';
SCRIPT = '/groups/branson/home/bransonk/tracking/code/rcpr/rcpr_v1_stable/misc/train/distrib/run_train.sh';

params = struct;
params.cpr_type = 'noocclusion';
params.model_type = 'FlyHeadStephen';
params.model_nfids = npts;
params.model_d = 2;
params.model_nviews = 1;
% params.ftr_type = 5;
% params.ftr_gen_radius = winrad;

params.ftr_type = 11;
params.cascade_depth = 20;
%params.ftr_type = 10;
params.ftr_gen_radius = 3;
%params.ftr_gen_radius = fliplr(round(logspace(log10(1),log10(10),params.cascade_depth)));
params.ftr_neighbors = ftr_neighbors;

params.expidx = ld.expidx(idx);

ld.expflies = cell(1,numel(ld.vid1files));
for expi = 1:numel(ld.vid1files),
  n = ld.vid1files{expi}(numel(rootdatadir)+2:end);
  n = regexp(n,'^([^/]*)/.*(fly_\d+)_','tokens','once');
  n = [n{1},'_',n{2}];
  ld.expflies{expi} = n;
end
[ld.expflies,~,ld.exp2flyidx] = unique(ld.expflies);
ld.flyidx = ld.exp2flyidx(ld.expidx);
params.cvidx = ld.flyidx(idx);
params.ncrossvalsets = numel(ld.expflies);

params.naugment = 100;
params.nsample_std = 1000;
params.nsample_cor = 5000;
params.nferns = 25;
%params.fern_thresh = .2*2^(16-1);
params.fern_thresh = .2;
params.docomperr = false;
params.augment_dorotate = false;

params.prunePrm = struct;
params.prunePrm.prune = 0;
params.prunePrm.maxIter = 2;
params.prunePrm.th = 0.5000;
params.prunePrm.tIni = 10;
params.prunePrm.numInit = 50;
params.prunePrm.usemaxdensity = 1;
params.prunePrm.maxdensity_sigma = 5; 

paramsfile1 = fullfile('/groups/branson/home/bransonk/tracking/code/rcpr/rcpr_v1_stable/misc',...
  sprintf('TrainData_%s.mat',savestr0));

paramsfile2 = sprintf('TrainParams_%s.mat',savestr);
trainresfile = sprintf('TrainedModel_%s.mat',savestr);

save(paramsfile2,'-struct','params');

trainresfiles_cv = {};
scriptfiles_cv = {};
outfiles_cv = {};

trainsavedir = fullfile('/nobackup/branson',savestr);
if ~exist(trainsavedir,'dir'),
  mkdir(trainsavedir);
end

for cvi = 1:params.ncrossvalsets,

  jobid = sprintf('%s_%d',savestr,cvi);
  trainresfiles_cv{cvi} = fullfile(trainsavedir,[jobid,'.mat']);
  scriptfiles_cv{cvi} = fullfile(trainsavedir,[jobid,'.sh']);
  outfiles_cv{cvi} = fullfile(trainsavedir,[jobid,'.log']);

  fid = fopen(scriptfiles_cv{cvi},'w');
  fprintf(fid,'if [ -d %s ]\n',TMP_ROOT_DIR);
  fprintf(fid,'  then export MCR_CACHE_ROOT=%s.%s\n',MCR_CACHE_ROOT,jobid);
  fprintf(fid,'fi\n');
  fprintf(fid,'%s %s %s %s %s cvi %d\n',...
    SCRIPT,MCR,paramsfile1,paramsfile2,trainresfiles_cv{cvi},cvi);
  fclose(fid);
  unix(sprintf('chmod u+x %s',scriptfiles_cv{cvi}));
  
  cmd = sprintf('ssh login1 ''source /etc/profile; cd %s; qsub -pe batch %d -N %s -j y -b y -o ''%s'' -cwd ''\"%s\"''''',...
    curdir,NCORESPERJOB,jobid,outfiles_cv{cvi},scriptfiles_cv{cvi});

  unix(cmd);  
  
end
  
% collect results
tmp0 = load(trainresfiles_cv{1});
tmp0.regModel = cell(1,params.ncrossvalsets);
D = size(tmp0.phisPr,2);
tmp0.phisPr = nan(nTr,D);
for cvi = 1:params.ncrossvalsets,
  tmp = load(trainresfiles_cv{cvi});
  tmp0.regModel{cvi} = tmp.regModel;
  tmp0.phisPr(tmp.cvidx==cvi,:) = tmp.phisPr;
end
tmp0.prunePrm.initlocs = initlocs';
if doeq,
  tmp0.H0 = H0;
end
tmp0.doscale = doscale;
tmp0.prunePrm.motion_2dto3D = false;
tmp0.prunePrm.motionparams = {'poslambda',.5};

trainresfile = sprintf('TrainedModel_%s.mat',savestr);
save(trainresfile,'-struct','tmp0');

res = struct;
[res.phisPr_cv,res.phisPrAll_cv,res.err_cv,res.errPerIter_cv] = cvtest(paramsfile1,paramsfile2,trainresfile);

% 50 ferns:
% err after 100 iters with ftr_gen_radius ranging from 10 to 1 is 7.8692
% err after 30 iters with ftr_gen_radius = 25 is 8.7237
% err after 30 iters with ftr_gen_radius = 10 is 7.9982
% err after 30 iters with ftr_gen_radius = 5 is 7.8850
% err after 30 iters with ftr_gen_radius = 3 is 7.2938
% err after 30 iters with ftr_gen_radius = 1 is 7.7130

% 25 ferns
% err after 30 iters with ftr_gen_radius = 3 is 7.5944

%% look at effect of training set size

savestr = 'FlyHeadStephenTestData_ftrrad3_nferns50_20150813';

fracstrain = [1,.99,.95,.9:-.1:.1];

NCORESPERJOB = 1;
curdir = pwd;
TMP_ROOT_DIR = '/scratch/bransonk';
MCR_CACHE_ROOT = fullfile(TMP_ROOT_DIR,'mcr_cache_root');
MCR = '/groups/branson/bransonlab/share/MCR/v717';
SCRIPT = '/groups/branson/home/bransonk/tracking/code/rcpr/rcpr_v1_stable/misc/train/distrib/run_train.sh';

params = struct;
params.cpr_type = 'noocclusion';
params.model_type = 'FlyHeadStephen';
params.model_nfids = npts;
params.model_d = 2;
params.model_nviews = 1;
% params.ftr_type = 5;
% params.ftr_gen_radius = winrad;

params.ftr_type = 11;
params.cascade_depth = 30;
%params.ftr_type = 10;
params.ftr_gen_radius = 3;
%params.ftr_gen_radius = fliplr(round(logspace(log10(1),log10(10),params.cascade_depth)));
params.ftr_neighbors = ftr_neighbors;

params.expidx = ld.expidx(idx);

ld.expflies = cell(1,numel(ld.vid1files));
for expi = 1:numel(ld.vid1files),
  n = ld.vid1files{expi}(numel(rootdatadir)+2:end);
  n = regexp(n,'^([^/]*)/.*(fly_\d+)_','tokens','once');
  n = [n{1},'_',n{2}];
  ld.expflies{expi} = n;
end
[ld.expflies,~,ld.exp2flyidx] = unique(ld.expflies);
ld.flyidx = ld.exp2flyidx(ld.expidx);
params.cvidx = ld.flyidx(idx);
params.ncrossvalsets = numel(ld.expflies);

params.naugment = 100;
params.nsample_std = 1000;
params.nsample_cor = 5000;
params.nferns = 50;
%params.fern_thresh = .2*2^(16-1);
params.fern_thresh = .2;
params.docomperr = false;
params.augment_dorotate = false;

params.prunePrm = struct;
params.prunePrm.prune = 0;
params.prunePrm.maxIter = 2;
params.prunePrm.th = 0.5000;
params.prunePrm.tIni = 10;
params.prunePrm.numInit = 50;
params.prunePrm.usemaxdensity = 1;
params.prunePrm.maxdensity_sigma = 5; 

paramsfile1 = fullfile('/groups/branson/home/bransonk/tracking/code/rcpr/rcpr_v1_stable/misc',...
  sprintf('TrainData_%s.mat',savestr0));

paramsfile2 = sprintf('TrainParams_%s.mat',savestr);
trainresfile = sprintf('TrainedModel_%s.mat',savestr);

save(paramsfile2,'-struct','params');

trainresfiles_cv = {};
scriptfiles_cv = {};
outfiles_cv = {};

trainsavedir = fullfile('/nobackup/branson',savestr);
if ~exist(trainsavedir,'dir'),
  mkdir(trainsavedir);
end

for fraci = 1:numel(fracstrain),
  fractrain = fracstrain(fraci);
  
  for cvi = 1:params.ncrossvalsets,
    
    jobid = sprintf('%s_f%d_%d',savestr,fractrain*100,cvi);
    trainresfiles_cv{cvi,fraci} = fullfile(trainsavedir,[jobid,'.mat']);
    scriptfiles_cv{cvi,fraci} = fullfile(trainsavedir,[jobid,'.sh']);
    outfiles_cv{cvi,fraci} = fullfile(trainsavedir,[jobid,'.log']);

    fid = fopen(scriptfiles_cv{cvi,fraci},'w');
    fprintf(fid,'if [ -d %s ]\n',TMP_ROOT_DIR);
    fprintf(fid,'  then export MCR_CACHE_ROOT=%s.%s\n',MCR_CACHE_ROOT,jobid);
    fprintf(fid,'fi\n');
    fprintf(fid,'%s %s %s %s %s cvi %d fractrain %f\n',...
      SCRIPT,MCR,paramsfile1,paramsfile2,trainresfiles_cv{cvi,fraci},cvi,fractrain);
    fclose(fid);
    unix(sprintf('chmod u+x %s',scriptfiles_cv{cvi,fraci}));
    
    cmd = sprintf('ssh login1 ''source /etc/profile; cd %s; qsub -pe batch %d -N %s -j y -b y -o ''%s'' -cwd ''\"%s\"''''',...
      curdir,NCORESPERJOB,jobid,outfiles_cv{cvi,fraci},scriptfiles_cv{cvi,fraci});

    unix(cmd);
  end
  
end
  
% collect results
clear res;
for fraci = 1:numel(fracstrain),
  tmp0 = load(trainresfiles_cv{1,fraci});
  tmp0.regModel = cell(1,params.ncrossvalsets);
  D = size(tmp0.phisPr,2);
  tmp0.phisPr = nan(nTr,D);
  for cvi = 1:params.ncrossvalsets,
    tmp = load(trainresfiles_cv{cvi,fraci});
    tmp0.regModel{cvi} = tmp.regModel;
    tmp0.phisPr(tmp.cvidx==cvi,:) = tmp.phisPr;
  end
  tmp0.prunePrm.initlocs = initlocs';
  if doeq,
    tmp0.H0 = H0;
  end
  tmp0.doscale = doscale;
  tmp0.prunePrm.motion_2dto3D = false;
  tmp0.prunePrm.motionparams = {'poslambda',.5};
  
  trainresfile = sprintf('TrainedModel_%s_f%d.mat',savestr,100*fracstrain(fraci));
  save(trainresfile,'-struct','tmp0');

  rescurr = struct;
  [rescurr.phisPr_cv,rescurr.phisPrAll_cv,rescurr.err_cv,rescurr.errPerIter_cv] = cvtest(paramsfile1,paramsfile2,trainresfile);

  res(fraci) = rescurr;
end

%% look at effect of number of flies trained on

savestr = 'FlyHeadStephenTestData_ftrrad3_nferns50_20150813_nflies';

NCORESPERJOB = 1;
curdir = pwd;
TMP_ROOT_DIR = '/scratch/bransonk';
MCR_CACHE_ROOT = fullfile(TMP_ROOT_DIR,'mcr_cache_root');
MCR = '/groups/branson/bransonlab/share/MCR/v717';
SCRIPT = '/groups/branson/home/bransonk/tracking/code/rcpr/rcpr_v1_stable/misc/train/distrib/run_train.sh';

params = struct;
params.cpr_type = 'noocclusion';
params.model_type = 'FlyHeadStephen';
params.model_nfids = npts;
params.model_d = 2;
params.model_nviews = 1;
% params.ftr_type = 5;
% params.ftr_gen_radius = winrad;

params.ftr_type = 11;
params.cascade_depth = 30;
%params.ftr_type = 10;
params.ftr_gen_radius = 3;
%params.ftr_gen_radius = fliplr(round(logspace(log10(1),log10(10),params.cascade_depth)));
params.ftr_neighbors = ftr_neighbors;

params.expidx = ld.expidx(idx);

ld.expflies = cell(1,numel(ld.vid1files));
for expi = 1:numel(ld.vid1files),
  n = ld.vid1files{expi}(numel(rootdatadir)+2:end);
  n = regexp(n,'^([^/]*)/.*(fly_\d+)_','tokens','once');
  n = [n{1},'_',n{2}];
  ld.expflies{expi} = n;
end
[ld.expflies,~,ld.exp2flyidx] = unique(ld.expflies);
ld.flyidx = ld.exp2flyidx(ld.expidx);
params.cvidx = ld.flyidx(idx);
params.ncrossvalsets = numel(ld.expflies);

params.naugment = 100;
params.nsample_std = 1000;
params.nsample_cor = 5000;
params.nferns = 50;
%params.fern_thresh = .2*2^(16-1);
params.fern_thresh = .2;
params.docomperr = false;
params.augment_dorotate = false;

params.prunePrm = struct;
params.prunePrm.prune = 0;
params.prunePrm.maxIter = 2;
params.prunePrm.th = 0.5000;
params.prunePrm.tIni = 10;
params.prunePrm.numInit = 50;
params.prunePrm.usemaxdensity = 1;
params.prunePrm.maxdensity_sigma = 5; 

paramsfile1 = fullfile('/groups/branson/home/bransonk/tracking/code/rcpr/rcpr_v1_stable/misc',...
  sprintf('TrainData_%s.mat',savestr0));

paramsfile2 = sprintf('TrainParams_%s.mat',savestr);

save(paramsfile2,'-struct','params');

trainresfiles_cv = {};
scriptfiles_cv = {};
outfiles_cv = {};

trainsavedir = fullfile('/nobackup/branson',savestr);
if ~exist(trainsavedir,'dir'),
  mkdir(trainsavedir);
end

nsets_trains = fliplr(unique([params.ncrossvalsets-1,params.ncrossvalsets-2,params.ncrossvalsets-3,...
  params.ncrossvalsets-5,round(params.ncrossvalsets/2),round(params.ncrossvalsets/4),1]));
niters = 5;

for seti = 1:numel(nsets_trains),

  nsets_train = nsets_trains(seti);
  
  if nsets_train == params.ncrossvalsets-1,
    niterscurr = 1;
  else
    niterscurr = niters;
  end
  
  for iter = 1:niterscurr,
  
    for cvi = 1:params.ncrossvalsets,
    
      jobid = sprintf('%s_n%d_i%d_%d',savestr,nsets_train,iter,cvi);
      trainresfiles_cv{cvi,iter,seti} = fullfile(trainsavedir,[jobid,'.mat']);
      scriptfiles_cv{cvi,iter,seti} = fullfile(trainsavedir,[jobid,'.sh']);
      outfiles_cv{cvi,iter,seti} = fullfile(trainsavedir,[jobid,'.log']);

      fid = fopen(scriptfiles_cv{cvi,iter,seti},'w');
      fprintf(fid,'if [ -d %s ]\n',TMP_ROOT_DIR);
      fprintf(fid,'  then export MCR_CACHE_ROOT=%s.%s\n',MCR_CACHE_ROOT,jobid);
      fprintf(fid,'fi\n');
      fprintf(fid,'%s %s %s %s %s cvi %d nsets_train %d\n',...
        SCRIPT,MCR,paramsfile1,paramsfile2,trainresfiles_cv{cvi,iter,seti},cvi,nsets_train);
      fclose(fid);
      unix(sprintf('chmod u+x %s',scriptfiles_cv{cvi,iter,seti}));
      
      cmd = sprintf('ssh login1 ''source /etc/profile; cd %s; qsub -pe batch %d -N %s -j y -b y -o ''%s'' -cwd ''\"%s\"''''',...
        curdir,NCORESPERJOB,jobid,outfiles_cv{cvi,iter,seti},scriptfiles_cv{cvi,iter,seti});

      unix(cmd);
    end
  end
end
  
% collect results
clear res;
for seti = 1:numel(nsets_trains),
  nsets_train = nsets_trains(seti);
  
  if nsets_train == params.ncrossvalsets-1,
    niterscurr = 3;
  else
    niterscurr = niters;
  end
  
  for iter = 1:niterscurr,

%     tmp0 = load(trainresfiles_cv{1,iter,seti});
%     D = tmp0.regModel.model.D;
%     tmp0.regModel = cell(1,params.ncrossvalsets);
%     for cvi = 1:params.ncrossvalsets,
%       tmp = load(trainresfiles_cv{cvi,iter,seti});
%       tmp0.regModel{cvi} = tmp.regModel;
%     end
%     tmp0.prunePrm.initlocs = initlocs';
%     if doeq,
%       tmp0.H0 = H0;
%     end
%     tmp0.doscale = doscale;
%     tmp0.prunePrm.motion_2dto3D = false;
%     tmp0.prunePrm.motionparams = {'poslambda',.5};
  
    trainresfile = sprintf('TrainedModel_%s_n%d_i%d.mat',savestr,nsets_train,iter);
%     save(trainresfile,'-struct','tmp0');

    rescurr = struct;
    [rescurr.phisPr_cv,rescurr.phisPrAll_cv,rescurr.err_cv,rescurr.errPerIter_cv] = cvtest(paramsfile1,paramsfile2,trainresfile);

    res(seti,iter) = rescurr;
  end
end

% average results

meanerr = nan(1,numel(nsets_trains));
sigerr = nan(1,numel(nsets_trains));
for seti = 1:min(numel(nsets_trains),size(res,1));
  for niterscurr = 1:niters,
    if isempty(res(seti,niterscurr).err_cv),
      niterscurr = niterscurr-1;
      break;
    end
  end
  if niterscurr == 0,
    continue;
  end
  
  meanerr(seti) = mean([res(seti,1:niterscurr).err_cv]);
  sigerr(seti) = std([res(seti,1:niterscurr).err_cv],1)/sqrt(niterscurr);
end

figure(123);
clf;
errorbar(nsets_trains,meanerr,sigerr,'ko-','MarkerFaceColor','k');
xlabel('N. flies train');
ylabel('Error (px)');

%% try scaling intensity of each image
savestr = 'FlyHeadStephenTestData_ftrrad3_nferns50_scale_20150813';

NCORESPERJOB = 1;
curdir = pwd;
TMP_ROOT_DIR = '/scratch/bransonk';
MCR_CACHE_ROOT = fullfile(TMP_ROOT_DIR,'mcr_cache_root');
MCR = '/groups/branson/bransonlab/share/MCR/v717';
SCRIPT = '/groups/branson/home/bransonk/tracking/code/rcpr/rcpr_v1_stable/misc/train/distrib/run_train.sh';

params = struct;
params.cpr_type = 'noocclusion';
params.model_type = 'FlyHeadStephen';
params.model_nfids = npts;
params.model_d = 2;
params.model_nviews = 1;
% params.ftr_type = 5;
% params.ftr_gen_radius = winrad;

params.ftr_type = 11;
params.cascade_depth = 30;
%params.ftr_type = 10;
params.ftr_gen_radius = 3;
%params.ftr_gen_radius = fliplr(round(logspace(log10(1),log10(10),params.cascade_depth)));
params.ftr_neighbors = ftr_neighbors;

params.expidx = ld.expidx(idx);

ld.expflies = cell(1,numel(ld.vid1files));
for expi = 1:numel(ld.vid1files),
  n = ld.vid1files{expi}(numel(rootdatadir)+2:end);
  n = regexp(n,'^([^/]*)/.*(fly_\d+)_','tokens','once');
  n = [n{1},'_',n{2}];
  ld.expflies{expi} = n;
end
[ld.expflies,~,ld.exp2flyidx] = unique(ld.expflies);
ld.flyidx = ld.exp2flyidx(ld.expidx);
params.cvidx = ld.flyidx(idx);
params.ncrossvalsets = numel(ld.expflies);

params.naugment = 100;
params.nsample_std = 1000;
params.nsample_cor = 5000;
params.nferns = 50;
%params.fern_thresh = .2*2^(16-1);
params.fern_thresh = .2;
params.docomperr = false;
params.augment_dorotate = false;

params.prunePrm = struct;
params.prunePrm.prune = 0;
params.prunePrm.maxIter = 2;
params.prunePrm.th = 0.5000;
params.prunePrm.tIni = 10;
params.prunePrm.numInit = 50;
params.prunePrm.usemaxdensity = 1;
params.prunePrm.maxdensity_sigma = 5; 

paramsfile1 = fullfile('/groups/branson/home/bransonk/tracking/code/rcpr/rcpr_v1_stable/misc',...
  sprintf('TrainData_%s.mat',savestr0));

tmp = load(paramsfile1);
minmaxint = 100;
for i = 1:numel(tmp.IsTr),
  tmp2 = double(tmp.IsTr{i});
  tmp3 = tmp2(:,1:size(tmp2,2)/2);
  tmp4 = tmp2(:,size(tmp2,2)/2+1:end);
  tmp.IsTr{i} = uint8(255*cat(2,tmp3/max(minmaxint,max(tmp3(:))),tmp4/max(minmaxint,max(tmp4(:)))));
end

savestr1 = 'FlyHeadStephenTestData_20150813_scaled';
paramsfile1 = fullfile('/groups/branson/home/bransonk/tracking/code/rcpr/rcpr_v1_stable/misc',...
  sprintf('TrainData_%s.mat',savestr1));
save(paramsfile1,'-struct','tmp');


paramsfile2 = sprintf('TrainParams_%s.mat',savestr);
trainresfile = sprintf('TrainedModel_%s.mat',savestr);

save(paramsfile2,'-struct','params');

trainresfiles_cv = {};
scriptfiles_cv = {};
outfiles_cv = {};

trainsavedir = fullfile('/nobackup/branson',savestr);
if ~exist(trainsavedir,'dir'),
  mkdir(trainsavedir);
end
for cvi = 1:params.ncrossvalsets,

  jobid = sprintf('%s_%d',savestr,cvi);
  trainresfiles_cv{cvi} = fullfile(trainsavedir,[jobid,'.mat']);
  scriptfiles_cv{cvi} = fullfile(trainsavedir,[jobid,'.sh']);
  outfiles_cv{cvi} = fullfile(trainsavedir,[jobid,'.log']);

  fid = fopen(scriptfiles_cv{cvi},'w');
  fprintf(fid,'if [ -d %s ]\n',TMP_ROOT_DIR);
  fprintf(fid,'  then export MCR_CACHE_ROOT=%s.%s\n',MCR_CACHE_ROOT,jobid);
  fprintf(fid,'fi\n');
  fprintf(fid,'%s %s %s %s %s cvi %d\n',...
    SCRIPT,MCR,paramsfile1,paramsfile2,trainresfiles_cv{cvi},cvi);
  fclose(fid);
  unix(sprintf('chmod u+x %s',scriptfiles_cv{cvi}));
  
  cmd = sprintf('ssh login1 ''source /etc/profile; cd %s; qsub -pe batch %d -N %s -j y -b y -o ''%s'' -cwd ''\"%s\"''''',...
    curdir,NCORESPERJOB,jobid,outfiles_cv{cvi},scriptfiles_cv{cvi});

  unix(cmd);  
  
end
  
% collect results
tmp0 = load(trainresfiles_cv{1});
tmp0.regModel = cell(1,params.ncrossvalsets);
D = size(tmp0.phisPr,2);
tmp0.phisPr = nan(nTr,D);
for cvi = 1:params.ncrossvalsets,
  tmp = load(trainresfiles_cv{cvi});
  tmp0.regModel{cvi} = tmp.regModel;
  tmp0.phisPr(tmp.cvidx==cvi,:) = tmp.phisPr;
end
tmp0.prunePrm.initlocs = initlocs';
if doeq,
  tmp0.H0 = H0;
end
tmp0.doscale = doscale;
tmp0.prunePrm.motion_2dto3D = false;
tmp0.prunePrm.motionparams = {'poslambda',.5};

trainresfile = sprintf('TrainedModel_%s.mat',savestr);
save(trainresfile,'-struct','tmp0');

res = struct;
[res.phisPr_cv,res.phisPrAll_cv,res.err_cv,res.errPerIter_cv] = cvtest(paramsfile1,paramsfile2,trainresfile);

%% compute the error by testing on full training movies

NCORESPERJOB = 1;
curdir = pwd;
TMP_ROOT_DIR = '/scratch/bransonk';
MCR_CACHE_ROOT = fullfile(TMP_ROOT_DIR,'mcr_cache_root');
MCR = '/groups/branson/bransonlab/share/MCR/v717';
SCRIPT = '/groups/branson/home/bransonk/tracking/code/rcpr/rcpr_v1_stable/misc/test/distrib/run_test.sh';

testresfiles_cv = {};
scriptfiles_cv = {};
outfiles_cv = {};

for expi = 1:numel(ld.expdirs),

  expdir = ld.expdirs{expi};
  [~,n] = fileparts(expdir);

  if ~any(ld.expidx==expi),
    continue;
  end
  
  idxcurr = find(params.expidx==expi);
  if isempty(idxcurr),
    fprintf('No training examples from %s\n',expdir);    
    trainfilecurr = trainresfile_motion_combine;
  else
  
    cvi = unique(params.cvidx(idxcurr));
    assert(numel(cvi)==1);
    trainfilecurr = trainresfiles_motion_combine{cvi};
  end
  
       
  jobid = sprintf('track_cv_%s_%s',n,datestr(now,'yyyymmddTHHMMSSFFF'));
  testresfiles_cv{expi} = fullfile(trainsavedir,[jobid,'.mat']);
  scriptfiles_cv{expi} = fullfile(trainsavedir,[jobid,'.sh']);
  outfiles_cv{expi} = fullfile(trainsavedir,[jobid,'.log']);

  fid = fopen(scriptfiles_cv{expi},'w');
  fprintf(fid,'if [ -d %s ]\n',TMP_ROOT_DIR);
  fprintf(fid,'  then export MCR_CACHE_ROOT=%s.%s\n',MCR_CACHE_ROOT,jobid);
  fprintf(fid,'fi\n');
  fprintf(fid,'%s %s %s %s %s %s\n',...
    SCRIPT,MCR,expdir,trainfilecurr,testresfiles_cv{expi},ld.moviefilestr);
  fclose(fid);
  unix(sprintf('chmod u+x %s',scriptfiles_cv{expi}));
  
  cmd = sprintf('ssh login1 ''source /etc/profile; cd %s; qsub -pe batch %d -N %s -j y -b y -o ''%s'' -cwd ''\"%s\"''''',...
    curdir,NCORESPERJOB,jobid,outfiles_cv{expi},scriptfiles_cv{expi});

  unix(cmd);  
  
end

% collect results
trx = struct;
for expi = 1:max(params.expidx),

  idxcurr = find(ld.expidx==expi);
  if isempty(idxcurr),
    continue;
  end
  idxcurr1 = find(ismember(idx,idxcurr));
  
  expdir = ld.expdirs{expi};
  
  cvi = unique(params.cvidx(idxcurr1));
  assert(numel(cvi)==1);
  
  if ~exist(testresfiles_cv{expi},'file'),
    fprintf('File %s does not exist, skipping\n',testresfiles_cv{expi});
    continue;
  end
  
  tdcurr = load(testresfiles_cv{expi});
  
  if expi == 1,
    nphi = numel(tdcurr.phisPr);
    trx.phisPr = cell(nphi,1);
    trx.phisPrAll = cell(nphi,1);
    for i = 1:nphi,
      trx.phisPr{i} = nan(numel(params.cvidx),size(tdcurr.phisPr{i},2));
      trx.phisPrAll{i} = nan([numel(params.cvidx),size(tdcurr.phisPrAll{i},2),size(tdcurr.phisPrAll{i},3)]);
    end
  end
  
  ts = ld.ts(idxcurr);
  
  assert(all(ismember([repmat(expi,[numel(ts),1]),ts(:)],[ld.expidx(:),ld.ts(:)],'rows')));
  
  for i = 1:nphi,
    trx.phisPr{i}(idxcurr,:) = tdcurr.phisPr{i}(ts,:);
    trx.phisPrAll{i}(idxcurr,:,:) = tdcurr.phisPrAll{i}(ts,:,:);
  end  
end

save(fullfile(trainsavedir,sprintf('CVPredictions_%s.mat',savestr)),'-struct','trx');

%% plot cv error

regi = [2,3];

mouseidxcurr = mouseidx(params.expidx);
if numel(regi) == 1,
  p1 = trx.phisPr{regi}(:,[1,3]);
  p2 = trx.phisPr{regi}(:,[2,4]);
else
  p1 = trx.phisPr{regi(1)};
  p2 = trx.phisPr{regi(2)};
end
  
err = sqrt(sum((p1-allPhisTr(:,[1,3])).^2,2) + sum((p2-allPhisTr(:,[2,4])).^2,2));
nbins = 30;
[edges,ctrs] = SelectHistEdges(nbins,[0,50],'linear');
fracpermouse = nan(nmice,nbins);
for mousei = 1:nmice,
  
  idxcurr = mouseidxcurr == mousei;
  counts = hist(err(idxcurr),ctrs);
  fracpermouse(mousei,:) = counts / sum(counts);
  
end

fractotal = hist(err,ctrs);
fractotal = fractotal / sum(fractotal);

hfig = 1;

figure(hfig);
clf;
hpermouse = plot(ctrs,fracpermouse,'-');
hold on;
h = plot(ctrs,fractotal,'k-','LineWidth',3);

box off;
xlabel('Mean squared error, both views (px)');
ylabel('Fraction of data');

ylim = get(gca,'YLim');
ylim(1) = 0;
set(gca,'YLim',ylim);

prctiles = [50,75,95,99];%,100];
humanerr = [5.736611,9.073831,11.347343,13.273021];%,19.030200];
autoerr = prctile(err,prctiles);

disp(autoerr);

dy = ylim(2) / (numel(prctiles)+1);
colors = jet(256);
colors = colors(RescaleToIndex(1:numel(prctiles)+1,256),:);
colors = flipud(colors);

for j = 1:numel(prctiles),
  plot([humanerr(j),autoerr(j)],dy*j+[0,0],'-','Color',colors(j+1,:)*.7,'LineWidth',2);
  hhuman = plot(humanerr(j),dy*j,'o','Color',colors(j+1,:)*.7,'LineWidth',2,'MarkerSize',12);  
  hauto = plot(autoerr(j),dy*j,'x','Color',colors(j+1,:)*.7,'LineWidth',2,'MarkerSize',12);
  text(max([humanerr(j),autoerr(j)]),dy*j,sprintf('   %dth %%ile',prctiles(j)),'HorizontalAlignment','left',...
    'Color',.7*colors(j+1,:),'FontWeight','bold');
end

legend([hhuman,hauto],'Human','Tracker');

SaveFigLotsOfWays(hfig,fullfile(trainsavedir,'CVErr'));

%% put this error in context

hfig = 5;
figure(hfig);

i = 4509;

im = IsTr{idx(i)};
clf;
imagesc(im,[0,255]);
axis image;
hold on;
colormap gray;
plot(phisTr(i,1),phisTr(i,3),'+','Color',colors(1,:),'MarkerSize',12,'LineWidth',3);
plot(phisTr(i,2),phisTr(i,4),'+','Color',colors(1,:),'MarkerSize',12,'LineWidth',3);
for j = 1:numel(prctiles),
  errcurr = autoerr(j)/sqrt(2);
  theta = linspace(-pi,pi,100);
  plot(phisTr(i,1)+errcurr*cos(theta),phisTr(i,3)+errcurr*sin(theta),'-','LineWidth',2,'Color',colors(j+1,:)*.75+.25);
  plot(phisTr(i,2)+errcurr*cos(theta),phisTr(i,4)+errcurr*sin(theta),'-','LineWidth',2,'Color',colors(j+1,:)*.75+.25);

%   errcurr = humanerr(j)/sqrt(2);
%   plot(phisTr(i,1)+errcurr*cos(theta),phisTr(i,3)+errcurr*sin(theta),'--','LineWidth',2,'Color',colors(j+1,:));
%   plot(phisTr(i,2)+errcurr*cos(theta),phisTr(i,4)+errcurr*sin(theta),'--','LineWidth',2,'Color',colors(j+1,:));

end
  
SaveFigLotsOfWays(hfig,fullfile(trainsavedir,'CVErrOnImage'));
