%% set up paths

addpath(genpath('..'));
addpath /groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect/misc;
addpath /groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect/filehandling;
addpath(genpath('/groups/branson/home/bransonk/tracking/code/piotr_toolbox_V3.02'));

datatype = 'stephen';
switch datatype,
  case 'stephen'
    paramfile = 'param.stephen20170411.yaml';
    lbldatafile = 'PlotGTAccuracyData_Stephen20170410.mat';
    minanimalid = 90;
    savedir = 'TrackingResults_Stephen20170411';
    trackermatfile = 'TrainedCPRTrackers_Stephen20170411.mat';    
    ncores = 12;
    nframesplot = 1000;
  case 'jan'
    paramfile = 'param.jan20170415.yaml';
    lbldatafile = 'GTNNData_Jan20170414.mat';
    minanimalid = 0;
    savedir = 'TrackingResults_Jan20170415';
    traindatafile = 'TrainingData_Jan20170415.mat';
    trainresfilestr = 'TrackingResults_Jan20170415';
    trackermatfile = 'TrainedCPRTrackers_Jan20170415.mat';
    ncores = 32;
    nframesplot = 1000;
  case 'roian'
    paramfile = 'param.roian20170416.yaml';
    lbldatafile = 'GTNNData_Roian20170416.mat';
    minanimalid = 0;
    savedir = 'TrackingResults_Roian20170416';
    traindatafile = 'TrainingData_Roian20170416.mat';
    trainresfilestr = 'TrackingResults_Roian20170416';
    ncores = 12;
    trackermatfile = 'TrainedCPRTrackers_Roian20170416.mat';
    nframesplot = 1000;
  case 'jay'
    paramfile = 'param.jay20170416.yaml';
    lbldatafile = 'GTNNData_Jay20170416.mat';
    minanimalid = 0;
    savedir = 'TrackingResults_Jay20170416';
    traindatafile = 'TrainingData_Jay20170416.mat';
    trainresfilestr = 'TrackingResults_Jay20170416';
    trackermatfile = 'TrainedCPRTrackers_Jay20170416.mat';
    ncores = 32;
    nframesplot = 1000;
end

%% load in data

istrained = exist(trackermatfile,'file');

if istrained,
  load(trackermatfile);
else
  load(lbldatafile,'cvd','lbld');
end

nviews = size(lbld.movieFilesAll,2);
nlandmarks = size(lbld.labeledpos{1},1)/nviews;
nlblmovies = size(lbld.movieFilesAll,1);
ncvsplits = numel(cvd.split);

landmarkcolors = lines(nlandmarks);

sPrm = ReadYaml(paramfile);

%% crop images around all possible label locs
% 
% xlimsperview = nan(nviews,2);
% ylimsperview = nan(nviews,2);
% for v = 1:nviews,
%   off = nlandmarks*(v-1);
%   for i = 1:nlblmovies,
%     xlimsperview(v,1) = min(xlimsperview(1),min(min(lbld.labeledpos{i}(off+1:off+nlandmarks,1,:))));
%     xlimsperview(v,2) = max(xlimsperview(1),max(max(lbld.labeledpos{i}(off+1:off+nlandmarks,1,:))));
%     ylimsperview(v,1) = min(ylimsperview(1),min(min(lbld.labeledpos{i}(off+1:off+nlandmarks,2,:))));
%     ylimsperview(v,2) = max(ylimsperview(1),max(max(lbld.labeledpos{i}(off+1:off+nlandmarks,2,:))));
%   end
% end

%% read in all images

if exist(traindatafile,'file'),

else
  
IsAll = cell(0,nviews);
expidxAll = zeros(0,1);
fAll = zeros(0,1);
bboxesAll = zeros(0,4,nviews);
ptsAll = zeros([0,nlandmarks*2,nviews]);

NtrainAll = 0;

for i = 1:nlblmovies,

  fprintf('Reading in training data from movie %d...\n',i);
  islabeled = IsLabeled(lbld.labeledpos{i});
  if ~any(islabeled),
    continue;
  end
  fslabeled = find(islabeled);
  ncurr = numel(fslabeled);
  expidxAll(NtrainAll+1:NtrainAll+ncurr) = i;
  fAll(NtrainAll+1:NtrainAll+ncurr) = fslabeled;
  
  for v = 1:nviews,
    readframe = get_readframe_fcn(lbld.movieFilesAll{i,v});        
    off = nlandmarks*(v-1);
    
    % nlandmarks*2 x nframes
    p = reshape(lbld.labeledpos{i}(off+1:off+nlandmarks,:,fslabeled),[2*nlandmarks,ncurr]);
    ptsAll(NtrainAll+1:NtrainAll+ncurr,:,v) = p';

    for fi = 1:ncurr,
      f = fslabeled(fi);
      im = readframe(f);
      [nr,nc,ncolors] = size(im);
      if ncolors > 1,
        im = rgb2gray(im);
      end
      IsAll{NtrainAll+fi,v} = im;
      bboxesAll(NtrainAll+fi,:,v) = [1,1,nc,nr];
    end
    
  end  
  
  NtrainAll = NtrainAll + ncurr;
  fprintf('NtrainAll = %d\n',NtrainAll);
  
end

%% save

save(traindatafile,'IsAll','NtrainAll','bboxesAll','cvd','expidxAll','fAll','lbld','ptsAll','paramfile','sPrm','ncores');

end

%% initialize for training

if ~istrained,

allrcs = cell(ncvsplits,nviews);

%% train

scriptdir = '/nrs/branson/TrainRCPR';
TMP_ROOT_DIR = '/scratch/bransonk';
MCR_CACHE_ROOT = fullfile(TMP_ROOT_DIR,'mcr_cache_root');
MCR = '/groups/branson/bransonlab/share/MCR/v91';
SCRIPT = '/groups/branson/home/bransonk/tracking/code/APT/gt/ClusterTrainRCPR_cv/for_redistribution_files_only/run_ClusterTrainRCPR_cv.sh';
curdir = pwd;
if ~exist(scriptdir,'dir'),
  mkdir(scriptdir);
end
cvi = 1;

for cvi = 1:ncvsplits,  
  for v = 1:nviews,
    resfile = sprintf('%s_cv%d_view%d.mat',trainresfilestr,cvi,v);
    jobid = sprintf('cpr%d_%d_%s',cvi,v,datatype);
    scriptfile = fullfile(scriptdir,sprintf('train%s.sh',jobid));
    outfile = fullfile(scriptdir,sprintf('train%s.log',jobid));
    
    fid = fopen(scriptfile,'w');
    fprintf(fid,'if [ -d %s ]\n',TMP_ROOT_DIR);
    fprintf(fid,'  then export MCR_CACHE_ROOT=%s.%s\n',MCR_CACHE_ROOT,jobid);
    fprintf(fid,'fi\n');
    fprintf(fid,'%s %s %s %d %d %s\n',...
      SCRIPT,MCR,traindatafile,cvi,v,resfile);
    fclose(fid);
    unix(sprintf('chmod u+x %s',scriptfile));
    cmd = sprintf('ssh login1 ''source /etc/profile; cd %s; qsub -pe batch %d -N %s -j y -b y -o ''%s'' -cwd ''\"%s\"''''',...
      curdir,ncores,jobid,outfile,scriptfile);
    unix(cmd);
  end
end

%% load results

for cvi = 1:ncvsplits,
  
  for v = 1:nviews,
    resfile = sprintf('%s_cv%d_view%d.mat',trainresfilestr,cvi,v);
    r = load(resfile);
    allrcs{cvi,v} = r.rc;
  end
end

%% run on some training data

cvi = 1;
v = 1;
rc = allrcs{cvi,v};
istrain = true(1,NtrainAll);
istrain(ismember(expidxAll,cvd.split{cvi})) = false;

ntest = 7;
idxtest = randsample(find(istrain),ntest);
pTest = cell(1,nviews);
pTestSelected = cell(1,nviews);
for v = 1:nviews,
  [pTest{v}] = allrcs{cvi,v}.propagateRandInit(IsAll(idxtest,v),bboxesAll(idxtest,:,v),sPrm.TestInit);
  % ntest x number of outputs x nlandmarks x (x,y) x niters+1
  pTest{v} = reshape(pTest{v},ntest,sPrm.TestInit.Nrep,nlandmarks*2,allrcs{cvi,v}.nMajor+1);
  pTestSelected{v} = nan([ntest,nlandmarks*2,allrcs{cvi,v}.nMajor+1]);
  for ii = 1:allrcs{cvi,v}.nMajor+1,
    % [NxDxR]
    pTmp = permute(pTest{v}(:,:,:,ii),[1,3,2]);
    pTestSelected{v}(:,:,ii) = rcprTestSelectOutput(pTmp,allrcs{cvi,v}.prmModel,sPrm.Prune);
  end
end

hfig = 1;
figure(hfig);
clf;
hax = createsubplots(nviews,ntest,.01);
hax = reshape(hax,[nviews,ntest]);

for ii = 1:ntest,
  i = idxtest(ii);
  for v = 1:nviews,
    image(repmat(IsAll{i,v},[1,1,3]),'Parent',hax(v,ii));
    axis(hax(v,ii),'image','off');
    hold(hax(v,ii),'on');
    for l = 1:nlandmarks,
      plot(hax(v,ii),squeeze(pTest{v}(ii,:,l,end)),squeeze(pTest{v}(ii,:,l+nlandmarks,end)),'.','Color',landmarkcolors(l,:)*.5);
      plot(hax(v,ii),pTestSelected{v}(ii,l,end),pTestSelected{v}(ii,l+nlandmarks,end),'o','Color',landmarkcolors(l,:));
      plot(hax(v,ii),ptsAll(i,l,v),ptsAll(i,nlandmarks+l,v),'+','Color',.5*landmarkcolors(l,:)+.5);
    end
  end
end

truesize(hfig);

%% run on some test data

ntest = 7;
idxtest = randsample(find(~istrain),ntest);
pTest = cell(1,nviews);
pTestSelected = cell(1,nviews);
for v = 1:nviews,
  [pTest{v}] = allrcs{cvi,v}.propagateRandInit(IsAll(idxtest,v),bboxesAll(idxtest,:,v),sPrm.TestInit);
  % ntest x number of outputs x nlandmarks x (x,y) x niters+1
  pTest{v} = reshape(pTest{v},ntest,sPrm.TestInit.Nrep,nlandmarks*2,rc.nMajor+1);
  pTestSelected{v} = nan([ntest,nlandmarks*2,rc.nMajor+1]);
  for ii = 1:rc.nMajor+1,
    % [NxDxR]
    pTmp = permute(pTest{v}(:,:,:,ii),[1,3,2]);
    pTestSelected{v}(:,:,ii) = rcprTestSelectOutput(pTmp,rc.prmModel,sPrm.Prune);
  end
end

hfig = 2;
figure(hfig);
clf;
hax = createsubplots(nviews,ntest,.01);
hax = reshape(hax,[nviews,ntest]);

for ii = 1:ntest,
  i = idxtest(ii);
  for v = 1:nviews,
    image(repmat(IsAll{i,v},[1,1,3]),'Parent',hax(v,ii));
    axis(hax(v,ii),'image','off');
    hold(hax(v,ii),'on');
    for l = 1:nlandmarks,
      plot(hax(v,ii),squeeze(pTest{v}(ii,:,l,end)),squeeze(pTest{v}(ii,:,l+nlandmarks,end)),'.','Color',landmarkcolors(l,:)*.5);
      plot(hax(v,ii),pTestSelected{v}(ii,l,end),pTestSelected{v}(ii,l+nlandmarks,end),'o','Color',landmarkcolors(l,:));
      plot(hax(v,ii),ptsAll(i,l,v),ptsAll(i,nlandmarks+l,v),'+','Color',.5*landmarkcolors(l,:)+.5);
    end
  end
end
truesize(hfig);

%% run on all test data

for cvi = 1:ncvsplits,
  
  istrain = true(1,NtrainAll);
  istrain(ismember(expidxAll,cvd.split{cvi})) = false;
  idxtest = find(~istrain);
  ntest = numel(idxtest);
  
  pSelected = nan([nlandmarks*nviews,2,ntest]);
  for v = 1:nviews,
    pTmp = allrcs{cvi,v}.propagateRandInit(IsAll(idxtest,v),bboxesAll(idxtest,:,v),sPrm.TestInit);
    pTmp = reshape(pTmp,ntest,sPrm.TestInit.Nrep,nlandmarks*2,allrcs{cvi,v}.nMajor+1);
    
    pTmp = permute(pTmp(:,:,:,end),[1,3,2]);
    % n x 2*nlandmarks
    pSelectedTmp = rcprTestSelectOutput(pTmp,allrcs{cvi,v}.prmModel,sPrm.Prune);
    pSelectedTmp = reshape(pSelectedTmp,[ntest,nlandmarks,2]);
    pSelectedTmp = permute(pSelectedTmp,[2,3,1]);
    
    off = nlandmarks*(v-1);
    pSelected(off+1:off+nlandmarks,:,:) = pSelectedTmp;
  end
  
  expidxtest = unique(expidxAll(idxtest));
  for i = expidxtest(:)',
    idxcurr = find(expidxAll(idxtest)==i);
    fscurr = fAll(idxtest(idxcurr));
    lbld.cpr_2d_locs{i} = nan(size(lbld.labeledpos{i}));
    lbld.cpr_2d_locs{i}(:,:,fscurr) = pSelected(:,:,idxcurr);
  end
end

% make sure everything is predicted
for i = 1:nlblmovies,
  islabeled = IsLabeled(lbld.labeledpos{i});
  if ~any(islabeled),
    continue;
  end
  ispred = IsLabeled(lbld.cpr_2d_locs{i});
  assert(all(ispred(islabeled)));
end

%% eyeball errors

errs_cpr = zeros(nlandmarks*nviews,0);
errs_pd = zeros(nlandmarks*nviews,0);
errs_final = zeros(nlandmarks*nviews,0);
errs_mrf = zeros(nlandmarks*nviews,0);

for i = 1:nlblmovies,
  islabeled = IsLabeled(lbld.labeledpos{i});
  ncurr = nnz(islabeled);
  if ncurr == 0,
    continue;
  end
  fslabeled = find(islabeled);
  errs_cpr(:,end+1:end+ncurr) = sqrt(sum((lbld.cpr_2d_locs{i}(:,:,fslabeled)-lbld.labeledpos{i}(:,:,fslabeled)).^2,2));
  if isempty(lbld.pd_locs{i}),
    continue;
  end
  errs_pd(:,end+1:end+ncurr) = sqrt(sum((lbld.pd_locs{i}(:,:,fslabeled)-lbld.labeledpos{i}(:,:,fslabeled)).^2,2));
  errs_mrf(:,end+1:end+ncurr) = sqrt(sum((lbld.mrf_locs{i}(:,:,fslabeled)-lbld.labeledpos{i}(:,:,fslabeled)).^2,2));
end

prctiles_compute = [50,75,90,95,97.5,99,99.5];
cat(3,prctile(errs_cpr,prctiles_compute,2),prctile(errs_mrf,prctiles_compute,2))

%% save results

save(trackermatfile,'allrcs','paramfile','lbld','cvd','sPrm');

end

%% track videos

ncorestrack = 1;
chunksize = 500;
scriptdir = '/nrs/branson/TrackRCPR';
TMP_ROOT_DIR = '/scratch/bransonk';
MCR_CACHE_ROOT = fullfile(TMP_ROOT_DIR,'mcr_cache_root');
MCR = '/groups/branson/bransonlab/share/MCR/v91';
SCRIPT = '/groups/branson/home/bransonk/tracking/code/APT/gt/ClusterTrackRCPR/for_redistribution_files_only/run_ClusterTrackRCPR.sh';

if ~exist(savedir,'dir'),
  mkdir(savedir);
end
if ~exist(scriptdir,'dir'),
  mkdir(scriptdir);
end

tmp = lbld.animalids;
tmp(lbld.animalids < minanimalid) = -1;
[animalids,~,animalidx] = unique(tmp);
animalids = animalids(2:end);
animalidx = animalidx-1;
isnntracked = ~cellfun(@isempty,lbld.pd_locs);
movieidxtrack = nan(1,numel(animalids));
trackingresfiles = cell(numel(animalids),nviews);
resfiles = cell(numel(animalids),nviews);
for ii = 1:numel(animalids),
  
  id = animalids(ii);
  
  fprintf('Tracking a movie for animal %d (%d)\n',id,ii);
  
  idxcurr = find(animalidx==ii & isnntracked);
  if isempty(idxcurr),
    idxcurr = find(animalidx==ii);
  end
    
  i = randsample(idxcurr,1);
  movieidxtrack(ii) = i;
  
  for v = 1:nviews,
    
    [~,nframes] = get_readframe_fcn(lbld.movieFilesAll{i,v});
    nchunks = ceil(nframes/chunksize);
    resfiles{ii,v} = cell(1,nchunks);
    for j = 1:nchunks,
      f0 = (j-1)*chunksize+1;
      f1 = min(j*chunksize,nframes);

      resfile = fullfile(savedir,sprintf('%s_%d_%dto%d.mat',VideoPath2Identifier(lbld.movieFilesAll{i,1},datatype),v,f0,f1));
      resfiles{ii,v}{j} = resfile;
      jobid = sprintf('cprtk%d_%d_%d_%s',i,v,f0,datatype);
      scriptfile = fullfile(scriptdir,sprintf('%s.sh',jobid));
      outfile = fullfile(scriptdir,sprintf('%s.log',jobid));
    
      fid = fopen(scriptfile,'w');
      fprintf(fid,'if [ -d %s ]\n',TMP_ROOT_DIR);
      fprintf(fid,'  then export MCR_CACHE_ROOT=%s.%s\n',MCR_CACHE_ROOT,jobid);
      fprintf(fid,'fi\n');
      fprintf(fid,'%s %s %s %d %d %d %d %s %s %d',SCRIPT,MCR,...
        lbld.movieFilesAll{i,v},cvi,v,f0,f1,...
        trackermatfile,resfile,ncorestrack);
      fclose(fid);
      unix(sprintf('chmod u+x %s',scriptfile));
      cmd = sprintf('ssh login1 ''source /etc/profile; cd %s; qsub -pe batch %d -N %s -j y -b y -o ''%s'' -cwd ''\"%s\"''''',...
        curdir,ncores,jobid,outfile,scriptfile);
      unix(cmd);
    end
    
  end
  
end
%% load tracking results

for ii = 1:numel(animalids),
  
  i = movieidxtrack(ii);

  R = cell(1,nviews);
  for v = 1:nviews,
    
    nchunks = numel(resfiles{ii,v});
    
    for j = 1:nchunks,
      
      resfile = resfiles{ii,v}{j};
      Rcurr = load(resfile);
      f0 = (j-1)*chunksize+1;
      nframeschunk = size(Rcurr.cpr_2d_locs,1);
      f1 = f0+nframeschunk-1;

      if j == 1,
        R{v} = Rcurr;
      else
        R{v}.cpr_2d_locs(f0:f1,:,:) = Rcurr.cpr_2d_locs;
        R{v}.cpr_all2d_locs(f0:f1,:,:,:) = Rcurr.cpr_all2d_locs;
      end      
      
      R{v}.labels = permute(lbld.labeledpos{i}(off+1:off+nlandmarks,:,f0:f1),[3,1,2]);
    end
  end
  
  savefile = fullfile(savedir,[VideoPath2Identifier(lbld.movieFilesAll{i,1},datatype),'.mat']);
  save(savefile,'R');
  
end

%% make videos

switch datatype,
  case 'stephen',
    figpos = [10,10,1536,516];
  case 'jan'
    figpos = [10,10,512,512];
  case 'roian',
    figpos = [10,10,644,644];
  case 'jay',
    figpos = [10,10,704,260];
  otherwise
    error('not implemented');
end

order = randperm(numel(movieidxtrack));
for ii = 1:numel(movieidxtrack),
  i = movieidxtrack(order(ii));
  trxfile = fullfile(savedir,[VideoPath2Identifier(lbld.movieFilesAll{i,1},datatype),'.mat']);
  [p,n,e] = fileparts(trxfile);
  resvideo = fullfile(p,[n,'_HistVideo.avi']);
  if exist(resvideo,'file'),
    continue;
  end
  fprintf('ii = %d, i = %d, trxfile = %s\n',ii,i,trxfile);
  
  %tmp = load(trxfile);
  %nlandmarks = size(tmp.R{1}.cpr_2d_locs,2);
  if nlandmarks == 1,
    lkcolors = zeros(0,3);
  else
    lkcolors = lines(nlandmarks);
  end
  td = load(trxfile);
  nframes = size(td.R{1}.cpr_2d_locs,1);
  if nframes <= nframesplot,
    firstframe = 1;
    endframe = nframes;
  else
    if strcmp(datatype,'jay'),
      firstframe = 251;
      endframe = min(nframes,nframesplot+firstframe-1);
    else
      fmid = round((nframes+1)/2);
      firstframe = fmid-ceil(nframesplot/2);
      endframe = firstframe+nframesplot-1;
    end
  end
  if strcmp(datatype,'jay'),
    info = mmfileinfo(lbld.movieFilesAll{i,1});
    w = info.Video.Width;
    h = info.Video.Height;
    cropframelims = [1,w/2,1,h;w/2+1,w,1,h];
  else
    cropframelims = zeros(0,4);
  end
  MakeTrackingResultsHistogramVideo({},trxfile,'moviefilestr','','lkcolors',lkcolors,'PlotTrxLen',30,...
    'TextColor',[.99,.99,.99],'TrxColor','k','figpos',figpos,'smoothsig',sPrm.Prune.maxdensity_sigma,...
    'firstframe',firstframe,'endframe',endframe,'cropframelims',cropframelims,'resvideo',resvideo);
  
end

save movieidxtrack.mat movieidxtrack order ii;

for ii = 1:numel(movieidxtrack),
  i = movieidxtrack(order(ii));
  trxfile = fullfile(savedir,[VideoPath2Identifier(lbld.movieFilesAll{i,1},datatype),'.mat']);
  [p,n,e] = fileparts(trxfile);
  resvideo = fullfile(p,[n,'_HistVideo.avi']);
  if ~exist(resvideo,'file'),
    continue;
  end
  resmp4 = fullfile(p,[n,'_HistVideo.mp4']);
  if exist(resmp4,'file'),
    continue;
  end
  info = mmfileinfo(resvideo);
  cmd = sprintf('avconv -i %s -c:v mpeg4 -b:v 600k -s:v %dx%d -mbd rd -flags +mv4+aic -trellis 2 -cmp 2 -subcmp 2 -g 300 %s',...
    resvideo,info.Video.Width,info.Video.Height,resmp4);
  disp(cmd); %#ok<DSPS>
  unix(cmd);
end

%% make videos with trk only

movieidentifiers = cell(1,nlblmovies);
for i = 1:nlblmovies,
  movieidentifiers{i} = VideoPath2Identifier(lbld.movieFilesAll{i,1},datatype);
end

for ii = 1:numel(movieidxtrack),
  i = movieidxtrack(order(ii));
  trxfile = fullfile(savedir,[VideoPath2Identifier(lbld.movieFilesAll{i,1},datatype),'.mat']);
  [p,n,e] = fileparts(trxfile);
  resvideo = fullfile(p,[n,'_TrkVideo.avi']);
  if exist(resvideo,'file'),
    continue;
  end
  fprintf('ii = %d, i = %d, trxfile = %s\n',ii,i,trxfile);
  
  %tmp = load(trxfile);
  %nlandmarks = size(tmp.R{1}.cpr_2d_locs,2);
  lkcolors = lines(nlandmarks);
  
  info = mmfileinfo(lbld.movieFilesAll{i,1});
  %figpos = [10,10,nviews*info.Video.Width*2,info.Video.Height*2];

  nframes = size(td.R{1}.cpr_2d_locs,1);
  if nframes <= nframesplot,
    firstframe = 1;
    endframe = nframes;
  else
    if strcmp(datatype,'jay'),
      firstframe = 251;
      endframe = min(nframes,nframesplot+firstframe-1);
    else
      fmid = round((nframes+1)/2);
      firstframe = fmid-ceil(nframesplot/2);
      endframe = firstframe+nframesplot-1;
    end
  end
  if strcmp(datatype,'jay'),
    info = mmfileinfo(lbld.movieFilesAll{i,1});
    w = info.Video.Width;
    h = info.Video.Height;
    cropframelims = [1,w/2,1,h;w/2+1,w,1,h];
  else
    cropframelims = zeros(0,4);
  end
  
  MakeTrackingResultsHistogramVideo({},trxfile,'moviefilestr','','lkcolors',lkcolors,'PlotTrxLen',0,...
    'TextColor',[.99,.99,.99],'TrxColor','k','figpos',figpos,'smoothsig',sPrm.Prune.maxdensity_sigma,...
    'resvideo',resvideo,'plotdensity',false,'firstframe',firstframe,'endframe',endframe,'cropframelims',cropframelims);
  
end
