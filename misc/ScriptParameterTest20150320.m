% try a variety of parameters

%clear all
doeq = true;
loadH0 = false;
cpr_type = 2;
docacheims = true;
maxNTr = 10000;
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

% combine fixed
if isfield(ld,'fixedlabels'),

  % remove bad labels
  maxx1 = 352;
  badidx = ld.fixedlabels.pts(1,1,:) > maxx1 | ld.fixedlabels.pts(2,1,:) < maxx1;
  ld.fixedlabels.pts(:,:,badidx) = [];
  ld.fixedlabels.ts(badidx) = [];
  ld.fixedlabels.expidx(badidx) = [];
  ld.fixedlabels.err(badidx) = [];
  
  % choose the worst errors
  if numel(ld.fixedlabels.expidx) > numel(ld.expidx),
    [sortederr,order] = sort(ld.fixedlabels.err,2,'descend');
    idxadd = order(1:numel(ld.expidx));
    
    % curate errors
    iscorrected = nan(1,numel(idxadd));
    
    batchsize = numel(idxadd);
    nbatches = ceil(numel(idxadd)/batchsize);
    maxt = max(ld.fixedlabels.ts)+1;
    for batchi = 1:nbatches,
      ii0 = (batchi-1)*batchsize+1;
      ii1 = min(numel(idxadd),ii0+batchsize-1);
      
      [~,order1] = sort(ld.fixedlabels.expidx(idxadd(ii0:ii1))*maxt+ld.fixedlabels.ts(idxadd(ii0:ii1)));
      order1 = order1 + ii0 - 1;
      
      expidxcurr = ld.fixedlabels.expidx(idxadd(order1));
      tscurr = ld.fixedlabels.ts(idxadd(order1));
      idxnewexp = find([true,diff(expidxcurr,1,2)~=0 | diff(tscurr,1,2)~=1,true]);
      
      for ii = 1:numel(idxnewexp)-1,
        
        ii0 = order1(idxnewexp(ii));
        ii1 = order1(idxnewexp(ii+1)-1);
        iis = order1(idxnewexp(ii):idxnewexp(ii+1)-1);
        is = idxadd(iis);
        i0 = is(1);
        i1 = is(end);
        t0 = ld.fixedlabels.ts(idxadd(ii0));
        t1 = ld.fixedlabels.ts(idxadd(ii1));
        assert(t1-t0+1==numel(is));
        assert(all( (t0:t1) == ld.fixedlabels.ts(is) ));
        assert(ld.fixedlabels.expidx(i0)==ld.fixedlabels.expidx(i1));
        expdir = ld.fixedlabels.expdirs{ld.fixedlabels.expidx(i0)};
        moviefile = fullfile(expdir,ld.moviefilestr);
        [readframe,~,fid] = get_readframe_fcn(moviefile);
 
        clf;
        i = i0;
        im = readframe(ld.fixedlabels.ts(i));
        him = image(im); axis image; hold on;
        plot(squeeze(ld.fixedlabels.auto_pts(:,1,is))',squeeze(ld.fixedlabels.auto_pts(:,2,is))','-','Color',[.5,0,0]);
        plot(squeeze(ld.fixedlabels.pts(:,1,is))',squeeze(ld.fixedlabels.pts(:,2,is))','-','Color',[0,.5,0]);
        hauto(1) = plot(nan,nan,'ro','MarkerSize',10,'LineWidth',3);
        hauto(2) = plot(nan,nan,'ro','MarkerSize',10,'LineWidth',3);
        hfix(1) = plot(nan,nan,'gx','MarkerSize',10,'LineWidth',3);
        hfix(2) = plot(nan,nan,'gx','MarkerSize',10,'LineWidth',3);        
        while true,
          for i = is,
            im = readframe(ld.fixedlabels.ts(i));
            set(him,'CData',im);
            set(hauto(1),'XData',ld.fixedlabels.auto_pts(1,1,i),...
              'YData',ld.fixedlabels.auto_pts(1,2,i));
            set(hauto(2),'XData',ld.fixedlabels.auto_pts(2,1,i),...
              'YData',ld.fixedlabels.auto_pts(2,2,i));
            set(hfix(1),'XData',ld.fixedlabels.pts(1,1,i),...
              'YData',ld.fixedlabels.pts(1,2,i));
            set(hfix(2),'XData',ld.fixedlabels.pts(2,1,i),...
              'YData',ld.fixedlabels.pts(2,2,i));
            pause(.1);
          end

          iscorrectedcurr = input(sprintf('Err = %f, expdir = %s, t = %d:%d: ',ld.fixedlabels.err(is(1)),expdir,t0,t1));
          if iscorrectedcurr == 0 || iscorrectedcurr == 1,
            break;
          end
        end

        
        if fid > 0,
          fclose(fid);
        end

        iscorrected(iis) = iscorrectedcurr;
        
      end
    end

    
  else
    idxadd = 1:numel(ld.fixedlabels.expidx);
  end

  
  [expidxadd,~,newexpidx0] = unique(ld.fixedlabels.expidx(idxadd));
  expdirsadd = ld.fixedlabels.expdirs(expidxadd);
  
  % some of these fixed experiments are old
  [ism,oldexpidx] = ismember(expdirsadd,ld.expdirs);
  % update expidx
  newexpidx = nan(size(newexpidx0));
  for i = find(ism),
    newexpidx(newexpidx0==i) = oldexpidx(i);
  end
  idxnew = find(~ism);
  for i = 1:numel(idxnew),
    newexpidx(newexpidx0==idxnew(i)) = numel(ld.expdirs) + i;
  end
  
  ld.isfixed = [false(1,size(ld.pts,3)),true(1,numel(idxadd))];
  ld.expdirs = cat(2,ld.expdirs,expdirsadd);
  ld.pts = cat(3,ld.pts,ld.fixedlabels.pts(:,:,idxadd));
  ld.ts = cat(2,ld.ts,ld.fixedlabels.ts(idxadd));
  ld.expidx = cat(2,ld.expidx,newexpidx);
  
end

nTr=min(numel(ld.expidx),maxNTr);
if nTr < numel(ld.expidx),
  idx = SubsampleTrainingDataBasedOnMovement(ld,maxNTr);
  idx = idx(randperm(numel(idx)));
  nTr = numel(idx);
  %idx=randsample(numel(IsTr),nTr);
else
  idx = randperm(nTr);
end

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

%% save training data
allPhisTr = reshape(permute(ld.pts,[3,1,2]),[size(ld.pts,3),size(ld.pts,1)*size(ld.pts,2)]);
tmp2 = struct;
tmp2.phisTr = allPhisTr(idx,:);
tmp2.bboxesTr = repmat([1,1,imsz([2,1])],[numel(idx),1]);
tmp2.IsTr = IsTr(idx);
save('TrainData_M135_2D_20150320.mat','-struct','tmp2');

%% see if nsamples matters, 2D tracking

NCORESPERJOB = 2;
curdir = pwd;
TMP_ROOT_DIR = '/scratch/bransonk';
MCR_CACHE_ROOT = fullfile(TMP_ROOT_DIR,'mcr_cache_root');
MCR = '/groups/branson/bransonlab/share/MCR/v717';
SCRIPT = '/groups/branson/home/bransonk/tracking/code/rcpr/rcpr_v1_stable/misc/train/distrib/run_train.sh';
scriptdir = '/nobackup/branson/pawtracking_paramsweep20150320';

if ~exist(scriptdir,'dir'),
  mkdir(scriptdir);
end

params = struct;
params.cpr_type = 'noocclusion';
params.model_type = 'mouse_paw2';
params.ftr_type = 6;
params.ftr_gen_radius = 100;
params.expidx = ld.expidx(idx);
params.ncrossvalsets = 10;
params.naugment = 20;
params.nsample_std = 1000;

nsample_cor_try = [500,1000,2500,5000,10000,numel(idx)*params.naugment];

for i = 1:numel(nsample_cor_try),

  params.nsample_cor = nsample_cor_try(i);
  paramsfile1 = '/groups/branson/home/bransonk/tracking/code/rcpr/rcpr_v1_stable/misc/TrainData_2D_20150320.mat';
  paramsfile2 = fullfile(scriptdir,sprintf('TrainParams_2D_20150320_ncor%d.mat',params.nsample_cor));
  save(paramsfile2,'-struct','params');
  savefile = fullfile(scriptdir,sprintf('TrainResults_2D_20150320_ncor%d.mat',params.nsample_cor));

  scriptfile = fullfile(scriptdir,sprintf('TrainParamSweep_2D_20150320_ncor%d.sh',params.nsample_cor));
  outfile = fullfile(scriptdir,sprintf('log_TrainParamSweep_2D_20150320_ncor%d.txt',params.nsample_cor));
  
  jobid = sprintf('ncor%d',params.nsample_cor);
  fid = fopen(scriptfile,'w');
  fprintf(fid,'if [ -d %s ]\n',TMP_ROOT_DIR);
  fprintf(fid,'  then export MCR_CACHE_ROOT=%s.%s\n',MCR_CACHE_ROOT,jobid);
  fprintf(fid,'fi\n');
  fprintf(fid,'%s %s %s %s %s\n',...
    SCRIPT,MCR,paramsfile1,paramsfile2,savefile);
  fclose(fid);
  unix(sprintf('chmod u+x %s',scriptfile));
  
  cmd = sprintf('ssh login1 ''source /etc/profile; cd %s; qsub -pe batch %d -N %s -j y -b y -o ''%s'' -cwd ''\"%s\"''''',...
    curdir,NCORESPERJOB,jobid,outfile,scriptfile);

  unix(cmd);
  
end

while true,
  
  isdone = true;
  for i = 1:numel(nsample_cor_try),
    savefile = fullfile(scriptdir,sprintf('TrainResults_2D_20150320_ncor%d.mat',nsample_cor_try(i)));
    if ~exist(savefile,'file'),
      isdone = false;
      break;
    end
  end
  
  if isdone,
    break;
  end
  
  pause(10);
  
end

errs_ncor = nan(1,numel(nsample_cor_try));
for i = 1:numel(nsample_cor_try),
  savefile = fullfile(scriptdir,sprintf('TrainResults_2D_20150320_ncor%d.mat',nsample_cor_try(i)));
  if exist(savefile,'file'),
    tmp = load(savefile,'err');
    errs_ncor(i) = tmp.err;
  end
end

params.nsamples_cor = 5000;

%% see if naugment matters, 2D tracking

params = struct;
params.cpr_type = 'noocclusion';
params.model_type = 'mouse_paw2';
params.ftr_type = 6;
params.ftr_gen_radius = 100;
params.expidx = ld.expidx(idx);
params.ncrossvalsets = 10;
params.naugment = 20;
params.nsample_std = 1000;
params.nsample_cor = 5000;

naugment_try = [5,10,20,50];

for i = 1:numel(naugment_try),

  params.naugment = naugment_try(i);
  paramsfile1 = '/groups/branson/home/bransonk/tracking/code/rcpr/rcpr_v1_stable/misc/TrainData_2D_20150320.mat';
  paramsfile2 = fullfile(scriptdir,sprintf('TrainParams_2D_20150320_naugment%d.mat',params.naugment));
  save(paramsfile2,'-struct','params');
  savefile = fullfile(scriptdir,sprintf('TrainResults_2D_20150320_naugment%d.mat',params.naugment));

  scriptfile = fullfile(scriptdir,sprintf('TrainParamSweep_2D_20150320_naugment%d.sh',params.naugment));
  outfile = fullfile(scriptdir,sprintf('log_TrainParamSweep_2D_20150320_naugment%d.txt',params.naugment));
  
  jobid = sprintf('naugment%d',params.naugment);
  fid = fopen(scriptfile,'w');
  fprintf(fid,'if [ -d %s ]\n',TMP_ROOT_DIR);
  fprintf(fid,'  then export MCR_CACHE_ROOT=%s.%s\n',MCR_CACHE_ROOT,jobid);
  fprintf(fid,'fi\n');
  fprintf(fid,'%s %s %s %s %s\n',...
    SCRIPT,MCR,paramsfile1,paramsfile2,savefile);
  fclose(fid);
  unix(sprintf('chmod u+x %s',scriptfile));
  
  cmd = sprintf('ssh login1 ''source /etc/profile; cd %s; qsub -pe batch %d -N %s -j y -b y -o ''%s'' -cwd ''\"%s\"''''',...
    curdir,NCORESPERJOB,jobid,outfile,scriptfile);

  unix(cmd);
  
end

while true,
  
  isdone = true;
  for i = 1:numel(naugment_try),
    savefile = fullfile(scriptdir,sprintf('TrainResults_2D_20150320_naugment%d.mat',naugment_try(i)));
    if ~exist(savefile,'file'),
      isdone = false;
      break;
    end
  end
  
  if isdone,
    break;
  end
  
  pause(10);
  
end

errs_naugment = nan(1,numel(naugment_try));
for i = 1:numel(naugment_try),
  savefile = fullfile(scriptdir,sprintf('TrainResults_2D_20150320_naugment%d.mat',naugment_try(i)));
  if exist(savefile,'file'),
    tmp = load(savefile,'err');
    errs_naugment(i) = tmp.err;
  end
end

%% train first tracker

params = struct;
params.cpr_type = 'noocclusion';
params.model_type = 'mouse_paw2';
params.ftr_type = 6;
params.ftr_gen_radius = 100;
params.expidx = ld.expidx(idx);
params.ncrossvalsets = 1;
params.naugment = 50;
params.nsample_std = 1000;
params.nsample_cor = 5000;

params.prunePrm = struct;
params.prunePrm.prune = 0;
params.prunePrm.maxIter = 2;
params.prunePrm.th = 0.5000;
params.prunePrm.tIni = 10;
params.prunePrm.numInit = 50;
params.prunePrm.usemaxdensity = 1;
params.prunePrm.maxdensity_sigma = 5; %#ok<STRNU>

paramsfile1 = '/groups/branson/home/bransonk/tracking/code/rcpr/rcpr_v1_stable/misc/TrainData_M135_2D_20150320.mat';
paramsfile2 = 'TrainParams_M135_2D_20150324.mat';
trainresfile = 'TrainedModel_2D_20150324.mat';

save(paramsfile2,'-struct','params');

[regModel,regPrm,prunePrm,phisPr,err] = train(paramsfile1,paramsfile2,trainresfile);

npts = size(ld.pts,1);
X = reshape(ld.pts,[size(ld.pts,1)*size(ld.pts,1),size(ld.pts,3)]);
[idxinit,initlocs] = mykmeans(X',ninit,'Start','furthestfirst','Maxiter',100);
tmp = load(trainresfile);
tmp.prunePrm.initlocs = initlocs';
tmp.H0 = H0;
save(trainresfile,'-struct','tmp');

%% Train model for point 1

params = struct;
params.cpr_type = 'noocclusion';
params.model_type = 'mouse_paw';
params.ftr_type = 6;
params.ftr_gen_radius = 25;
params.expidx = ld.expidx(idx);
params.ncrossvalsets = 1;
params.naugment = 50;
params.nsample_std = 1000;
params.nsample_cor = 5000;

params.prunePrm = struct;
params.prunePrm.prune = 0;
params.prunePrm.maxIter = 2;
params.prunePrm.th = 0.5000;
params.prunePrm.tIni = 10;
params.prunePrm.numInit = 50;
params.prunePrm.usemaxdensity = 1;
params.prunePrm.maxdensity_sigma = 5;
params.prunePrm.windowradius = 40;

medfilwidth = 10;

params.prunePrm.initfcn = @(p) InitializeSecondRoundTracking(p,1,2,medfilwidth,params.prunePrm.windowradius);

paramsfile1_pt1 = '/groups/branson/home/bransonk/tracking/code/rcpr/rcpr_v1_stable/misc/TrainData_M135_2D_pt1_20150320.mat';
paramsfile2_pt1 = 'TrainParams_M135_2D_pt1_20150324.mat';
trainresfile_pt1 = 'TrainedModel_M135_2D_pt1_20150324.mat';

save(paramsfile2_pt1,'-struct','params');

allPhisTr2 = allPhisTr(:,[1 3]);
phisTr2 = allPhisTr2(idx,:);
bboxesTr2 = [phisTr2-params.prunePrm.windowradius, 2*params.prunePrm.windowradius*ones(size(phisTr2))];
tmp3 = struct;
tmp3.phisTr = phisTr2;
tmp3.bboxesTr = bboxesTr2;
tmp3.IsTr = IsTr(idx);
save(paramsfile1_pt1,'-struct','tmp3');

[regModel_pt1,regPrm_pt1,prunePrm_pt1,phisPr_pt1,err_pt1] = train(paramsfile1_pt1,paramsfile2_pt1,trainresfile_pt1);

tmp = load(trainresfile_pt1);
tmp.H0 = H0;
save(trainresfile_pt1,'-struct','tmp');

%% Train model for point 2

params = struct;
params.cpr_type = 'noocclusion';
params.model_type = 'mouse_paw';
params.ftr_type = 6;
params.ftr_gen_radius = 25;
params.expidx = ld.expidx(idx);
params.ncrossvalsets = 1;
params.naugment = 50;
params.nsample_std = 1000;
params.nsample_cor = 5000;

params.prunePrm = struct;
params.prunePrm.prune = 0;
params.prunePrm.maxIter = 2;
params.prunePrm.th = 0.5000;
params.prunePrm.tIni = 10;
params.prunePrm.numInit = 50;
params.prunePrm.usemaxdensity = 1;
params.prunePrm.maxdensity_sigma = 5;
params.prunePrm.windowradius = 40;

medfilwidth = 10;

params.prunePrm.initfcn = @(p) InitializeSecondRoundTracking(p,2,2,medfilwidth,params.prunePrm.windowradius);

paramsfile1_pt2 = '/groups/branson/home/bransonk/tracking/code/rcpr/rcpr_v1_stable/misc/TrainData_M135_2D_pt2_20150320.mat';
paramsfile2_pt2 = 'TrainParams_M135_2D_pt2_20150324.mat';
trainresfile_pt2 = 'TrainedModel_M135_2D_pt2_20150324.mat';

save(paramsfile2_pt2,'-struct','params');

allPhisTr3 = allPhisTr(:,[2 4]);
phisTr3 = allPhisTr3(idx,:);
bboxesTr3 = [phisTr3-params.prunePrm.windowradius, 2*params.prunePrm.windowradius*ones(size(phisTr3))];
tmp4 = struct;
tmp4.phisTr = phisTr3;
tmp4.bboxesTr = bboxesTr3;
tmp4.IsTr = IsTr(idx);
save(paramsfile1_pt2,'-struct','tmp4');

[regModel_pt2,regPrm_pt2,prunePrm_pt2,phisPr_pt2,err_pt2] = train(paramsfile1_pt2,paramsfile2_pt2,trainresfile_pt2);

tmp = load(trainresfile_pt2);
tmp.H0 = H0;
save(trainresfile_pt2,'-struct','tmp');

allregressors = struct;
allregressors.regModel = cell(1,3);
allregressors.regPrm = cell(1,3);
allregressors.prunePrm = cell(1,3);
allregressors.H0 = H0;
allregressors.traindeps = [0,1,1];

tmp = load(trainresfile);
allregressors.regModel{1} = tmp.regModel;
allregressors.regPrm{1} = tmp.regPrm;
allregressors.prunePrm{1} = tmp.prunePrm;
tmp = load(trainresfile_pt1);
allregressors.regModel{2} = tmp.regModel;
allregressors.regPrm{2} = tmp.regPrm;
allregressors.prunePrm{2} = tmp.prunePrm;
tmp = load(trainresfile_pt2);
allregressors.regModel{3} = tmp.regModel;
allregressors.regPrm{3} = tmp.regPrm;
allregressors.prunePrm{3} = tmp.prunePrm;

trainresfile_combine = 'TrainedModel_M135_2D_combined_20150324.mat';
save(trainresfile_combine,'-struct','allregressors');

%% version 2 with motion-based prediction


allregressors = struct;
allregressors.regModel = cell(1,3);
allregressors.regPrm = cell(1,3);
allregressors.prunePrm = cell(1,3);
allregressors.H0 = H0;
allregressors.traindeps = [0,1,1];

tmp = load(trainresfile);
allregressors.regModel{1} = tmp.regModel;
allregressors.regPrm{1} = tmp.regPrm;
allregressors.prunePrm{1} = tmp.prunePrm;
allregressors.prunePrm{1}.motion_2dto3D = true;
allregressors.prunePrm{1}.calibrationdata = calibrationdata;
allregressors.prunePrm{1}.motionparams = {'poslambda',.5};

tmp = load(trainresfile_pt1);
allregressors.regModel{2} = tmp.regModel;
allregressors.regPrm{2} = tmp.regPrm;
allregressors.prunePrm{2} = tmp.prunePrm;
allregressors.prunePrm{2}.motion_2dto3D = false;
allregressors.prunePrm{2}.motionparams = {'poslambda',.75};
allregressors.prunePrm{2}.initfcn = @(p) InitializeSecondRoundTracking(p,1,2,0,tmp.prunePrm.windowradius);

tmp = load(trainresfile_pt2);
allregressors.regModel{3} = tmp.regModel;
allregressors.regPrm{3} = tmp.regPrm;
allregressors.prunePrm{3} = tmp.prunePrm;
allregressors.prunePrm{3}.motion_2dto3D = false;
allregressors.prunePrm{3}.motionparams = {'poslambda',.75};
allregressors.prunePrm{3}.initfcn = @(p) InitializeSecondRoundTracking(p,2,2,0,tmp.prunePrm.windowradius);

trainresfile_motion_combine = 'TrainedModel_2D_motion_combined_20150324.mat';
save(trainresfile_motion_combine,'-struct','allregressors');

%% track movie

firstframe = 51;
endframe = 250;
expdir = '/tier2/hantman/Jay/videos/M147VGATXChrR2_anno/20150303L/CTR/M147_20150303_v017';
testresfile = '';
[phisPr,phisPrAll]=test(expdir,trainresfile_motion_combine,testresfile,ld.moviefilestr,[],firstframe,endframe);

[readframe,nframes,fid] = get_readframe_fcn(fullfile(expdir,ld.moviefilestr));
im = readframe(1);
imsz = size(im);

fprintf('Initial tracking results\n');

figure(1);
clf;
him = imagesc(im,[0,255]);
axis image;
colormap gray;
hold on;
hother = nan(1,2);
hother(1) = plot(nan,nan,'b.');
hother(2) = plot(nan,nan,'b.');
htrx = nan(1,2);
htrx(1) = plot(nan,nan,'m.-');
htrx(2) = plot(nan,nan,'c.-');
hcurr = nan(1,2);
hcurr(1) = plot(nan,nan,'mo','LineWidth',3);
hcurr(2) = plot(nan,nan,'co','LineWidth',3);
%htext = text(imsz(2)/2,5,'d. train = ??','FontSize',24,'HorizontalAlignment','center','VerticalAlignment','top','Color','r');

idxcurr = idx(ld.expidx(idx)==expi);
tstraincurr = ld.ts(idxcurr);

%p = phisPr{1};
p1 = phisPr{2};
p2 = phisPr{3};

for t = 1:endframe-firstframe+1,
  
  im = readframe(firstframe+t-1);
  set(him,'CData',im);
%   set(hother(1),'XData',squeeze(pall(t,1,:)),'YData',squeeze(pall(t,3,:)));
%   set(hother(2),'XData',squeeze(pall(t,2,:)),'YData',squeeze(pall(t,4,:)));
  set(htrx(1),'XData',p1(max(1,t-500):t,1),'YData',p1(max(1,t-500):t,2));
  set(htrx(2),'XData',p2(max(1,t-500):t,1),'YData',p2(max(1,t-500):t,2));
  set(hcurr(1),'XData',p1(t,1),'YData',p1(t,2));
  set(hcurr(2),'XData',p2(t,1),'YData',p2(t,2));
  pause(.25);
  
end

%% track more movies

NCORESPERJOB = 1;
curdir = pwd;
TMP_ROOT_DIR = '/scratch/bransonk';
MCR_CACHE_ROOT = fullfile(TMP_ROOT_DIR,'mcr_cache_root');
MCR = '/groups/branson/bransonlab/share/MCR/v717';
SCRIPT = '/groups/branson/home/bransonk/tracking/code/rcpr/rcpr_v1_stable/misc/test/distrib/run_test.sh';

rootdirs = {%'/tier2/hantman/Jay/videos/M118_CNO_G6'
  %'/tier2/hantman/Jay/videos/M119_CNO_G6'
  %'/tier2/hantman/Jay/videos/M122_CNO_M1BPN_G6_anno'
  %'/tier2/hantman/Jay/videos/M127D22_hm4D_Sl1_BPN'
  %'/tier2/hantman/Jay/videos/M130_hm4DBPN_KordM1'
  '/tier2/hantman/Jay/videos/M134C3VGATXChR2'
  '/tier2/hantman/Jay/videos/M147VGATXChrR2_anno'};

expdirs = cell(1,numel(rootdirs));
exptypes = cell(1,numel(rootdirs));

for i = 1:numel(rootdirs),
  
  [~,exptypes{i}] = fileparts(rootdirs{i});
  expdirs{i} = recursiveDir(rootdirs{i},'movie_comb.avi');
  
  expdirs{i} = cellfun(@(x) fileparts(x),expdirs{i},'Uni',0);
  
end

saverootdir = '/groups/branson/home/bransonk/tracking/code/rcpr/rcpr_v1_stable/misc/TrackingResults20150329';
if ~exist(saverootdir,'dir'),
  mkdir(saverootdir);
end

scriptfiles = {};
outfiles = {};
testresfiles = {};

for typei = numel(rootdirs):-1:1,
  
  savedircurr = fullfile(saverootdir,exptypes{typei});
  if ~exist(savedircurr,'dir'),
    mkdir(savedircurr);
  end
    
  for i = 1:numel(expdirs{typei}),
    
    expdir = expdirs{typei}{i};
    [~,n] = fileparts(expdir);
    
    
    if numel(testresfiles)>=typei && numel(testresfiles{typei}) >= i && ...
        ~isempty(testresfiles{typei}{i}) && exist(testresfiles{typei}{i},'file'),
      continue;
    end
    
    if numel(scriptfiles) >= typei && numel(scriptfiles{typei}) >= i && ~isempty(scriptfiles{typei}{i}),
      [~,jobid] = fileparts(scriptfiles{typei}{i});
    else
      jobid = sprintf('track_%s_%s',n,datestr(now,'yyyymmddTHHMMSSFFF'));
      testresfiles{typei}{i} = fullfile(savedircurr,[jobid,'.mat']);
      scriptfiles{typei}{i} = fullfile(savedircurr,[jobid,'.sh']);
      outfiles{typei}{i} = fullfile(savedircurr,[jobid,'.log']);
    end

    fid = fopen(scriptfiles{typei}{i},'w');
    fprintf(fid,'if [ -d %s ]\n',TMP_ROOT_DIR);
    fprintf(fid,'  then export MCR_CACHE_ROOT=%s.%s\n',MCR_CACHE_ROOT,jobid);
    fprintf(fid,'fi\n');
    fprintf(fid,'%s %s %s %s %s %s\n',...
      SCRIPT,MCR,expdir,trainresfile_motion_combine,testresfiles{typei}{i},ld.moviefilestr);
    fclose(fid);
    unix(sprintf('chmod u+x %s',scriptfiles{typei}{i}));
  
    cmd = sprintf('ssh login1 ''source /etc/profile; cd %s; qsub -pe batch %d -N %s -j y -b y -o ''%s'' -cwd ''\"%s\"''''',...
      curdir,NCORESPERJOB,jobid,outfiles{typei}{i},scriptfiles{typei}{i});

    unix(cmd);
  end
  
end


% load and combine files


for typei = numel(rootdirs):-1:1,
  
  parentdirs = regexp(expdirs{typei},['^',rootdirs{typei},'/([^/]*)/'],'tokens','once');
  parentdirs = [parentdirs{:}];
  [uniqueparentdirs,~,parentidx] = unique(parentdirs);
  
  for pdi = 1:numel(uniqueparentdirs),
    
    savefile = fullfile(saverootdir,sprintf('TrackingResults_%s_%s_%s.mat',exptypes{typei},uniqueparentdirs{pdi},datestr(now,'yyyymmdd')));
    if exist(savefile,'file'),
      continue;
    end
    savestuff = struct;
    savestuff.curr_vid = 1;
    savestuff.moviefiles_all = {};
    savestuff.p_all = {};
    savestuff.hyp_all = {};
  
    idxcurr = find(parentidx==pdi);
    for ii = 1:numel(idxcurr),
      
      i = idxcurr(ii);
      expdir = expdirs{typei}{i};
      if ~exist(testresfiles{typei}{i},'file'),
        error('%s does not exist.\n',testresfiles{typei}{i});
      end
      savestuff.moviefiles_all{1,ii} = fullfile(expdir,ld.moviefilestr);
      tmp = load(testresfiles{typei}{i});
      savestuff.p_all{ii,1}(:,[1,3]) = tmp.phisPr{2};
      savestuff.p_all{ii,1}(:,[2,4]) = tmp.phisPr{3};
      savestuff.hyp_all{ii,1}(:,[1,3],:) = tmp.phisPrAll{2};
      savestuff.hyp_all{ii,1}(:,[2,4],:) = tmp.phisPrAll{3};
    end
    
    fprintf('Saving tracking results for %d videos to %s\n',numel(idxcurr),savefile);
    save(savefile,'-struct','savestuff');
  end
    
end

%% compare fixed and original trx

correcteddir = fullfile(saverootdir,'corrected');
correctedfiles = mydir(fullfile(correcteddir,'*.mat'));
err = {};
for i = 1:numel(correctedfiles),
  
  [~,n] = myfileparts(correctedfiles{i});
  origfile = fullfile(saverootdir,n);
  assert(exist(origfile,'file')>0);
  
  td0 = load(origfile);
  tdfix = load(correctedfiles{i});
  
  winmoviefiles = strrep(td0.moviefiles_all,'/tier2/hantman','Y:');
  winmoviefiles = strrep(winmoviefiles,'/','\');
  assert(all(strcmp(winmoviefiles,tdfix.moviefiles_all)));
  
  for j = 1:numel(tdfix.p_all),
    N = size(td0.p_all{j},1);
    err{end+1} = sqrt(sum((tdfix.p_all{j}(1:N,:)-td0.p_all{j}).^2,2));
  end
  
end



%% convert to 3D

addpath /groups/branson/home/bransonk/codepacks/TOOLBOX_calib;
calib_file = '/groups/branson/home/bransonk/behavioranalysis/code/adamTracking/analysis/CameraCalibrationParams20150217.mat';
calibrationdata = load(calib_file);

omcurr = calibrationdata.om0;
Tcurr = calibrationdata.T0;

sz = imsz(2);

typei = numel(rootdirs);
expi = 1;
expdir = expdirs{typei}{expi};

td = load(testresfiles{typei}{expi});

xL = td.phisPrAll{1}(:,[1,3],:);
xR = td.phisPrAll{1}(:,[2,4],:);
[T,D,K] = size(xL);

% these don't necessarily match up -- fix
xL = permute(xL,[2,1,3]);
xR = permute(xR,[2,1,3]);

[X3d,X3d_right] = stereo_triangulation(xL,xR,calibrationdata.om0,calibrationdata.T0,calibrationdata.fc_left,...
  calibrationdata.cc_left,calibrationdata.kc_left,calibrationdata.alpha_c_left,...
  calibrationdata.fc_right,calibrationdata.cc_right,calibrationdata.kc_right,...
  calibrationdata.alpha_c_right);
[xL_re] = project_points2(X3d,zeros(size(calibrationdata.om0)),zeros(size(calibrationdata.T0)),calibrationdata.fc_left,calibrationdata.cc_left,calibrationdata.kc_left,calibrationdata.alpha_c_left);
[xR_re] = project_points2(X3d_right,zeros(size(calibrationdata.om0)),zeros(size(calibrationdata.T0)),calibrationdata.fc_right,calibrationdata.cc_right,calibrationdata.kc_right,calibrationdata.alpha_c_right);
d = sqrt(sum((xL_re-xL(:,:)).^2,1));
errL = [mean(d),min(d),max(d)];
d = sqrt(sum((xR_re-xR(:,:)).^2,1));
errR = [mean(d),min(d),max(d)];

X3d = reshape(X3d,[3,T,K]);

% compute the votes for each returned point

appearancecost = nan(T,K);
for t = 1:T,
  dL = pdist(reshape(xL(:,t,:),[2,K])').^2;
  dR = pdist(reshape(xR(:,t,:),[2,K])').^2;
  wL = sum(squareform(exp( -dL/prunePrm.maxdensity_sigma^2/2 )),1);
  wL = wL / sum(wL);
  wR = sum(squareform(exp( -dR/prunePrm.maxdensity_sigma^2/2 )),1);
  wR = wR / sum(wR);
  appearancecost(t,:) = -log(wL) - log(wR);
end

[Xbest,mincost] = ChooseBestTrajectory(X3d,appearancecost,'poslambda',.5);
[xL_re] = project_points2(Xbest,zeros(size(calibrationdata.om0)),zeros(size(calibrationdata.T0)),calibrationdata.fc_left,calibrationdata.cc_left,calibrationdata.kc_left,calibrationdata.alpha_c_left);
[xR_re] = project_points2(Xbest,calibrationdata.om0,calibrationdata.T0,calibrationdata.fc_right,calibrationdata.cc_right,calibrationdata.kc_right,calibrationdata.alpha_c_right);

[readframe,nframes,fid] = get_readframe_fcn(fullfile(expdir,ld.moviefilestr));

figure(1);
clf;
him = imagesc(readframe(1),[0,255]);
axis image;
colormap gray;
hold on;
hother = nan(1,2);
hother(1) = plot(nan,nan,'b.');
hother(2) = plot(nan,nan,'b.');
htrx = nan(1,2);
htrx(1) = plot(nan,nan,'m-');
htrx(2) = plot(nan,nan,'c-');
hcurr = nan(1,2);
hcurr(1) = plot(nan,nan,'mo','LineWidth',3);
hcurr(2) = plot(nan,nan,'co','LineWidth',3);
%htext = text(imsz(2)/2,5,'d. train = ??','FontSize',24,'HorizontalAlignment','center','VerticalAlignment','top','Color','r');

idxcurr = idx(ld.expidx(idx)==expi);
tstraincurr = ld.ts(idxcurr);

%p = phisPr{1};
p1 = xL_re';
p2 = xR_re';

for t = 90:120,%1:nframes;
  
  im = readframe(t);
  set(him,'CData',im);
%   set(hother(1),'XData',squeeze(pall(t,1,:)),'YData',squeeze(pall(t,3,:)));
%   set(hother(2),'XData',squeeze(pall(t,2,:)),'YData',squeeze(pall(t,4,:)));
  set(htrx(1),'XData',p1(max(1,t-500):t,1),'YData',p1(max(1,t-500):t,2));
  set(htrx(2),'XData',p2(max(1,t-500):t,1),'YData',p2(max(1,t-500):t,2));
  set(hcurr(1),'XData',p1(t,1),'YData',p1(t,2));
  set(hcurr(2),'XData',p2(t,1),'YData',p2(t,2));
  drawnow;
  input('');
  
end

%% test initialization

% furthest-first+k-means based clustering
ninit = 50;

tmp = load(savefile);
prunePrm = tmp.prunePrm;
prunePrm.numInit = ninit;
prunePrm.prune = 0;
prunePrm.usemaxdensity = true;
prunePrm.maxdensity_sigma = 5;

[phisPr,err]=cvtest(paramsfile1,paramsfile2,savefile,prunePrm,initlocs','');



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