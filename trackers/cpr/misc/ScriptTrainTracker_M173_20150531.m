% try a variety of parameters

%clear all
doeq = true;
loadH0 = false;
cpr_type = 2;
docacheims = true;
maxNTr = 15000;
radius = 100;

addpath ..;
addpath ../video_tracking;
addpath /groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect/misc;
addpath /groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect/filehandling;
addpath(genpath('/groups/branson/home/bransonk/tracking/code/piotr_toolbox_V3.02'));

defaultfolder = '/groups/branson/home/bransonk/tracking/code/rcpr/data';
%defaultfile = 'M134_M174_20150423.mat';
defaultfile = 'M118_M119_M122_M127_M130_M173_M174_20150531.mat';

[file,folder]=uigetfile('.mat',sprintf('Select training file containg clicked points (e.g. %s)',defaultfile),...
  fullfile(defaultfolder,defaultfile));
if isnumeric(file),
  return;
end

[~,savestr] = fileparts(file);

ld = load(fullfile(folder,file));
% 
% % where to save the models
% defaultFolderSave = folder;
% [~,n] = fileparts(file);
% defaultFileSave = fullfile(defaultfolder,[n,'_TrackerModel.mat']);
% [file1,folder1]=uiputfile('*.mat','Save model to file',defaultFileSave);
% file1=fullfile(folder1,file1);
% file2=[file1(1:end-4),'_pt1.mat'];
% file3=[file1(1:end-4),'_pt2.mat'];

%load(fullfile(folder,file),'phisTr','bboxesTr');

%% fix expdir name changes

olds = {
  'M1_4s'
  'M1_2s'
  'M1_1s'
  'CueButNoLaser'
  'M134C3VGATXChR2/20150302L/M134_20150302_v001'
  'M134C3VGATXChR2/20150302L/M134_20150302_v002'
  'M134C3VGATXChR2/20150302L/M134_20150302_v003'
  'M134C3VGATXChR2/20150302L/M134_20150302_v004'
  'M134C3VGATXChR2/20150302L/M134_20150302_v005'
  'M134C3VGATXChR2/20150302L/M134_20150302_v006'
  'M134C3VGATXChR2/20150302L/M134_20150302_v007'
  'M134C3VGATXChR2/20150302L/M134_20150302_v008'
  'M134C3VGATXChR2/20150302L/M134_20150302_v009'
  'M134C3VGATXChR2/20150302L/M134_20150302_v010'
  'M134C3VGATXChR2/20150302L/M134_20150302_v011'
  'M134C3VGATXChR2/20150302L/M134_20150302_v012'
  'M134C3VGATXChR2/20150302L/M134_20150302_v013'
  'M134C3VGATXChR2/20150302L/M134_20150302_v014'
  'Laser0.5secNoCue_cM1'
  'M173VGATXChR2/20150427L/M173_20150427_v001'
  'M173VGATXChR2/20150427L/M173_20150427_v002'
  'M173VGATXChR2/20150428L/M173_20150428_v001'
  'M173VGATXChR2/20150428L/M173_20150428_v002'
  'M173VGATXChR2/20150429L/M173_20150429_v001'
  'M173VGATXChR2/20150429L/M173_20150429_v002'
};
news = {
  'Laser4secNoCue_cM1'
  'Laser2secNoCue_cM1'
  'Laser1secNoCue_cM1'
  'CTR'
  'M134C3VGATXChR2/20150302L/Laser0_5secNoCue_cM1/M134_20150302_v001'
  'M134C3VGATXChR2/20150302L/Laser0_5secNoCue_cM1/M134_20150302_v002'
  'M134C3VGATXChR2/20150302L/Laser0_5secNoCue_cM1/M134_20150302_v003'
  'M134C3VGATXChR2/20150302L/Laser0_5secNoCue_cM1/M134_20150302_v004'
  'M134C3VGATXChR2/20150302L/Laser0_5secNoCue_cM1/M134_20150302_v005'
  'M134C3VGATXChR2/20150302L/Laser0_5secNoCue_cM1/M134_20150302_v006'
  'M134C3VGATXChR2/20150302L/Laser0_5secNoCue_cM1/M134_20150302_v007'
  'M134C3VGATXChR2/20150302L/Laser0_5secNoCue_cM1/M134_20150302_v008'
  'M134C3VGATXChR2/20150302L/Laser0_5secNoCue_cM1/M134_20150302_v009'
  'M134C3VGATXChR2/20150302L/Laser0_5secNoCue_cM1/M134_20150302_v010'
  'M134C3VGATXChR2/20150302L/Laser2secNoCue_cM1/M134_20150302_v011'
  'M134C3VGATXChR2/20150302L/Laser2secNoCue_cM1/M134_20150302_v012'
  'M134C3VGATXChR2/20150302L/Laser2secNoCue_cM1/M134_20150302_v013'
  'M134C3VGATXChR2/20150302L/Laser2secNoCue_cM1/M134_20150302_v014'
  'Laser0_5secNoCue_cM1'
  'M173VGATXChR2/20150427L/CTR/M173_20150427_v001'
  'M173VGATXChR2/20150427L/CTR/M173_20150427_v002'
  'M173VGATXChR2/20150428L/CTR/M173_20150428_v001'
  'M173VGATXChR2/20150428L/CTR/M173_20150428_v002'
  'M173VGATXChR2/20150429L/CTR/M173_20150429_v001'
  'M173VGATXChR2/20150429L/CTR/M173_20150429_v002'
  };

for tmpj = 1:numel(olds),
  newexpdirs = strrep(ld.expdirs,olds{tmpj},news{tmpj});
  ndiff = nnz(~strcmp(newexpdirs,ld.expdirs));
  fprintf('%s -> %s: changed %d experiment names\n',olds{tmpj},news{tmpj},ndiff);
  ld.expdirs = newexpdirs;
  if isfield(ld,'fixedlabels'),
    ld.fixedlabels.expdirs = strrep(ld.fixedlabels.expdirs,olds{tmpj},news{tmpj});
  end
end

for i = 1:numel(ld.expdirs),
  if ~exist(ld.expdirs{i},'dir'),
    fprintf('Missing %s\n',ld.expdirs{i});
  end
  assert(exist(ld.expdirs{i},'dir')>0);
end
if isfield(ld,'fixedlabels'),
  for i = 1:numel(ld.fixedlabels.expdirs),
    assert(exist(ld.fixedlabels.expdirs{i},'dir')>0);
  end
end

oldld = ld;

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

nframes = nan(1,numel(ld.expdirs));
for i = 1:numel(ld.expdirs),
  [~,nframes(i),fid] = get_readframe_fcn(fullfile(ld.expdirs{i},ld.moviefilestr));
  if fid > 0,
    fclose(fid);
  end
end  
badidx = nframes(ld.expidx) < ld.ts;
assert(~any(badidx));

% combine fixed
docurate = false;
if isfield(ld,'fixedlabels'),

  % remove bad labels
  maxx1 = 352;
  badidx = ld.fixedlabels.pts(1,1,:) > maxx1 | ld.fixedlabels.pts(2,1,:) < maxx1;
  
  
  ld.fixedlabels.nframes = nan(1,numel(ld.fixedlabels.expdirs));
  for i = 1:numel(ld.fixedlabels.expdirs),
    [~,nframes,fid] = get_readframe_fcn(fullfile(ld.fixedlabels.expdirs{i},ld.fixedlabels.moviefilestr));
    if fid > 0,
      fclose(fid);
    end
    ld.fixedlabels.nframes(i) = nframes;
  end
  
  badidx2 = ld.fixedlabels.nframes(ld.fixedlabels.expidx) < ld.fixedlabels.ts;
  badidx = badidx(:) | badidx2(:);
  
  ld.fixedlabels.pts(:,:,badidx) = [];
  ld.fixedlabels.ts(badidx) = [];
  ld.fixedlabels.expidx(badidx) = [];
  ld.fixedlabels.err(badidx) = [];
  
  % choose the worst errors
  if numel(ld.fixedlabels.expidx) > numel(ld.expidx),
    err = ld.fixedlabels.err;
    idxadd = nan(1,numel(ld.expidx));
    nadd = 0;
    for i = 1:numel(ld.expidx),
      
      [minerr,j] = min(err);
      if isnan(minerr),
        break;
      end
      idxadd(i) = j;
      nadd = nadd + 1;
      err(max(1,j-9):min(numel(err),j+9)) = nan;
      
    end

    idxadd = idxadd(1:nadd);
    
    %[sortederr,order] = sort(ld.fixedlabels.err,2,'descend');
    %idxadd = order(1:numel(ld.expidx));
    
    if docurate,
      
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
      
      iscorrected = true(1,numel(idxadd));
      
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
  ld.expdirs = cat(2,ld.expdirs,expdirsadd(~ism));
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

% sanity check

for i = idx(:)',
  
  expdir = ld.expdirs{ld.expidx(i)};
  t = ld.ts(i);
  if ~any((strcmp(oldld.expdirs(oldld.expidx),expdir) & oldld.ts == t)),
    if ~isfield(oldld,'fixedlabels') || ...
        ~any(strcmp(oldld.fixedlabels.expdirs(oldld.fixedlabels.expidx),expdir) & oldld.fixedlabels.ts == t),
    error('!');
    end
  end
  
end


hfig = 2;
figure(hfig);
clf;
nr = 5;
nc = 5;
nplot = nr*nc;
hax = createsubplots(nr,nc,.01);


mouse = regexp(ld.expdirs,'videos/([^/]*)/','tokens','once');
mouse = [mouse{:}];
[mice,~,mouseidx] = unique(mouse);
nmice = numel(mice);

i0 = round(linspace(1,nplot+1,nmice+1));

idxsample = [];
for mousei = 1:nmice,
  
  idxcurr = idx(mouseidx(ld.expidx(idx))==mousei);
  ncurr = i0(mousei+1)-i0(mousei);
  idxsample(i0(mousei):i0(mousei+1)-1) = idxcurr(randsample(numel(idxcurr),ncurr));  
  
end

for ii = 1:nplot,
  
  i = idxsample(ii);
  expdir = ld.expdirs{ld.expidx(i)};
  [readframe] = get_readframe_fcn(fullfile(expdir,ld.moviefilestr));
  im = readframe(ld.ts(i));
  image(im,'Parent',hax(ii));
  axis(hax(ii),'image','off');
  hold(hax(ii),'on');
  plot(hax(ii),ld.pts(1,1,i),ld.pts(1,2,i),'wo','MarkerFaceColor','r');
  plot(hax(ii),ld.pts(2,1,i),ld.pts(2,2,i),'wo','MarkerFaceColor','g');
  
end


%% read in the training images

[~,n,e] = fileparts(file);
cachedimfile = fullfile(folder,[n,'_cachedims',e]);
if docacheims && exist(cachedimfile,'file'),
  load(cachedimfile,'IsTr');
  imsz = size(IsTr{1});
else
  
  IsTr = cell(1,numel(ld.expidx));
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
      IsTr{idxcurr(i)} = im;
      
    end
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
    H=nan(256,nTr);
    mu = nan(1,nTr);
    for i=1:nTr,
      H(:,i)=imhist(IsTr{idx(i)});
      mu(i) = mean(IsTr{idx(i)}(:));
    end
    % normalize to brighter movies, not to dimmer movies
    idxuse = mu >= prctile(mu,75);
    H0=median(H(:,idxuse),2);
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
end


hfig = 3;
figure(hfig);
clf;
nr = 5;
nc = 5;
nplot = nr*nc;
hax = createsubplots(nr,nc,.01);

for ii = 1:nplot,
  
  i = idxsample(ii);
  im = IsTr{i};
  imagesc(im,'Parent',hax(ii),[0,255]);
  axis(hax(ii),'image','off');
  hold(hax(ii),'on');
  plot(hax(ii),ld.pts(1,1,i),ld.pts(1,2,i),'wo','MarkerFaceColor','r');
  plot(hax(ii),ld.pts(2,1,i),ld.pts(2,2,i),'wo','MarkerFaceColor','g');
  
end


%% save training data
allPhisTr = reshape(permute(ld.pts,[3,1,2]),[size(ld.pts,3),size(ld.pts,1)*size(ld.pts,2)]);
tmp2 = struct;
tmp2.phisTr = allPhisTr(idx,:);
tmp2.bboxesTr = repmat([1,1,imsz([2,1])],[numel(idx),1]);
tmp2.IsTr = IsTr(idx);
save('-v7.3',sprintf('TrainData_%s.mat',savestr),'-struct','tmp2');

%% plot some training examples

hfig = 1;
figure(hfig);
clf;
nr = 5;
nc = 5;
nplot = nr*nc;
hax = createsubplots(nr,nc,0);

i0 = round(linspace(1,nplot+1,nmice+1));

idxsample2 = [];
for mousei = 1:nmice,
  
  idxcurr = find(mouseidx(ld.expidx(idx))==mousei);
  ncurr = i0(mousei+1)-i0(mousei);
  idxsample2(i0(mousei):i0(mousei+1)-1) = idxcurr(randsample(numel(idxcurr),ncurr));  
  
end

for ii = 1:nplot,
  
  i = idxsample2(ii);
  imagesc(tmp2.IsTr{i},'Parent',hax(ii),[0,255]);
  axis(hax(ii),'image','off');
  hold(hax(ii),'on');
  plot(hax(ii),tmp2.phisTr(i,1),tmp2.phisTr(i,3),'wo','MarkerFaceColor','r');
  plot(hax(ii),tmp2.phisTr(i,2),tmp2.phisTr(i,4),'wo','MarkerFaceColor','g');
  
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
params.prunePrm.maxdensity_sigma = 5; 

paramsfile1 = fullfile('/groups/branson/home/bransonk/tracking/code/rcpr/rcpr_v1_stable/misc',...
  sprintf('TrainData_%s.mat',savestr));
paramsfile2 = sprintf('TrainParams_%s_2D_20150531.mat',savestr);
trainresfile = sprintf('TrainedModel_%s.mat',savestr);

save(paramsfile2,'-struct','params');

[regModel,regPrm,prunePrm,phisPr,err] = train(paramsfile1,paramsfile2,trainresfile);

npts = size(ld.pts,1);
X = reshape(ld.pts,[size(ld.pts,1)*size(ld.pts,1),size(ld.pts,3)]);
[idxinit,initlocs] = mykmeans(X',params.prunePrm.numInit,'Start','furthestfirst','Maxiter',100);
tmp = load(trainresfile);
tmp.prunePrm.initlocs = initlocs';
tmp.H0 = H0;
tmp.prunePrm.motion_2dto3D = false;
tmp.prunePrm.motionparams = {'poslambda',.5};
save(trainresfile,'-append','-struct','tmp');

%% test first tracker

firstframe = 101;
endframe = 300;
expdir = '/tier2/hantman/Jay/videos/M173VGATXChR2/20150420L/CTR/M173_20150420_v028';
testresfile = '';
[phisPr,phisPrAll]=test(expdir,trainresfile,testresfile,...
  'moviefilestr',ld.moviefilestr,'firstframe',firstframe,'endframe',endframe);

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

%p = phisPr{1};
p1 = phisPr{1}(:,[1,3]);
p2 = phisPr{1}(:,[2,4]);

for t = 1:endframe-firstframe+1,
  
  im = readframe(firstframe+t-1);
  set(him,'CData',im);
%   set(hother(1),'XData',squeeze(pall(t,1,:)),'YData',squeeze(pall(t,3,:)));
%   set(hother(2),'XData',squeeze(pall(t,2,:)),'YData',squeeze(pall(t,4,:)));
  set(htrx(1),'XData',p1(max(1,t-500):t,1),'YData',p1(max(1,t-500):t,2));
  set(htrx(2),'XData',p2(max(1,t-500):t,1),'YData',p2(max(1,t-500):t,2));
  set(hcurr(1),'XData',p1(t,1),'YData',p1(t,2));
  set(hcurr(2),'XData',p2(t,1),'YData',p2(t,2));
  %pause(.25);
  drawnow;
  
end
% 
% %% Train model for point 1
% 
% params = struct;
% params.cpr_type = 'noocclusion';
% params.model_type = 'mouse_paw';
% params.ftr_type = 6;
% params.ftr_gen_radius = 25;
% params.expidx = ld.expidx(idx);
% params.ncrossvalsets = 1;
% params.naugment = 50;
% params.nsample_std = 1000;
% params.nsample_cor = 5000;
% 
% params.prunePrm = struct;
% params.prunePrm.prune = 0;
% params.prunePrm.maxIter = 2;
% params.prunePrm.th = 0.5000;
% params.prunePrm.tIni = 10;
% params.prunePrm.numInit = 50;
% params.prunePrm.usemaxdensity = 1;
% params.prunePrm.maxdensity_sigma = 5;
% params.prunePrm.windowradius = 40;
% 
% medfilwidth = 10;
% 
% params.prunePrm.initfcn = @(p) InitializeSecondRoundTracking(p,1,2,medfilwidth,params.prunePrm.windowradius);
% 
% paramsfile1_pt1 = fullfile('/groups/branson/home/bransonk/tracking/code/rcpr/rcpr_v1_stable/misc',...
%   sprintf('TrainData_%s_2D_pt1.mat',savestr));
% paramsfile2_pt1 = sprintf('TrainParams_%s_2D_pt1.mat',savestr);
% trainresfile_pt1 = sprintf('TrainModel_%s_2D_pt1.mat',savestr);
% 
% save(paramsfile2_pt1,'-struct','params');
% 
% allPhisTr2 = allPhisTr(:,[1 3]);
% phisTr2 = allPhisTr2(idx,:);
% bboxesTr2 = [phisTr2-params.prunePrm.windowradius, 2*params.prunePrm.windowradius*ones(size(phisTr2))];
% copyfile(paramsfile1,paramsfile1_pt1);
% tmp3 = struct;
% tmp3.phisTr = phisTr2;
% tmp3.bboxesTr = bboxesTr2;
% %tmp3.IsTr = IsTr(idx);
% save(paramsfile1_pt1,'-append','-struct','tmp3');
% 
% [regModel_pt1,regPrm_pt1,prunePrm_pt1,phisPr_pt1,err_pt1] = train(paramsfile1_pt1,paramsfile2_pt1,trainresfile_pt1);
% 
% tmp = load(trainresfile_pt1);
% tmp.H0 = H0;
% save(trainresfile_pt1,'-struct','tmp');
% 
% %% Train model for point 2
% 
% params = struct;
% params.cpr_type = 'noocclusion';
% params.model_type = 'mouse_paw';
% params.ftr_type = 6;
% params.ftr_gen_radius = 25;
% params.expidx = ld.expidx(idx);
% params.ncrossvalsets = 1;
% params.naugment = 50;
% params.nsample_std = 1000;
% params.nsample_cor = 5000;
% 
% params.prunePrm = struct;
% params.prunePrm.prune = 0;
% params.prunePrm.maxIter = 2;
% params.prunePrm.th = 0.5000;
% params.prunePrm.tIni = 10;
% params.prunePrm.numInit = 50;
% params.prunePrm.usemaxdensity = 1;
% params.prunePrm.maxdensity_sigma = 5;
% params.prunePrm.windowradius = 40;
% 
% medfilwidth = 10;
% 
% params.prunePrm.initfcn = @(p) InitializeSecondRoundTracking(p,2,2,medfilwidth,params.prunePrm.windowradius);
% 
% paramsfile1_pt2 = fullfile('/groups/branson/home/bransonk/tracking/code/rcpr/rcpr_v1_stable/misc',...
%   sprintf('TrainData_%s_2D_pt2.mat',savestr));
% paramsfile2_pt2 = sprintf('TrainParams_%s_2D_pt2.mat',savestr);
% trainresfile_pt2 = sprintf('TrainModel_%s_2D_pt2.mat',savestr);
% 
% save(paramsfile2_pt2,'-struct','params');
% 
% allPhisTr3 = allPhisTr(:,[2 4]);
% phisTr3 = allPhisTr3(idx,:);
% bboxesTr3 = [phisTr3-params.prunePrm.windowradius, 2*params.prunePrm.windowradius*ones(size(phisTr3))];
% 
% copyfile(paramsfile1,paramsfile1_pt2);
% tmp4 = struct;
% tmp4.phisTr = phisTr3;
% tmp4.bboxesTr = bboxesTr3;
% %tmp4.IsTr = IsTr(idx);
% save(paramsfile1_pt2,'-append','-struct','tmp4');
% %save(paramsfile1_pt2,'-struct','tmp4');
% 
% [regModel_pt2,regPrm_pt2,prunePrm_pt2,phisPr_pt2,err_pt2] = train(paramsfile1_pt2,paramsfile2_pt2,trainresfile_pt2);
% 
% save(trainresfile_pt2,'-append','H0');
% 
% allregressors = struct;
% allregressors.regModel = cell(1,3);
% allregressors.regPrm = cell(1,3);
% allregressors.prunePrm = cell(1,3);
% allregressors.H0 = H0;
% allregressors.traindeps = [0,1,1];
% 
% tmp = load(trainresfile);
% allregressors.regModel{1} = tmp.regModel;
% allregressors.regPrm{1} = tmp.regPrm;
% allregressors.prunePrm{1} = tmp.prunePrm;
% tmp = load(trainresfile_pt1);
% allregressors.regModel{2} = tmp.regModel;
% allregressors.regPrm{2} = tmp.regPrm;
% allregressors.prunePrm{2} = tmp.prunePrm;
% tmp = load(trainresfile_pt2);
% allregressors.regModel{3} = tmp.regModel;
% allregressors.regPrm{3} = tmp.regPrm;
% allregressors.prunePrm{3} = tmp.prunePrm;
% 
% trainresfile_combine = sprintf('TrainedModel_%s_2D_combined.mat',savestr);
% save(trainresfile_combine,'-struct','allregressors');
% 
% %% version 2 with motion-based prediction
% 
% 
% allregressors = struct;
% allregressors.regModel = cell(1,3);
% allregressors.regPrm = cell(1,3);
% allregressors.prunePrm = cell(1,3);
% allregressors.H0 = H0;
% allregressors.traindeps = [0,1,1];
% 
% tmp = load(trainresfile);
% allregressors.regModel{1} = tmp.regModel;
% allregressors.regPrm{1} = tmp.regPrm;
% allregressors.prunePrm{1} = tmp.prunePrm;
% %allregressors.prunePrm{1}.motion_2dto3D = true;
% %allregressors.prunePrm{1}.calibrationdata = calibrationdata;
% % turned this off 20150427 because points in each view are different
% allregressors.prunePrm{1}.motion_2dto3D = false;
% allregressors.prunePrm{1}.motionparams = {'poslambda',.5};
% 
% tmp = load(trainresfile_pt1);
% allregressors.regModel{2} = tmp.regModel;
% allregressors.regPrm{2} = tmp.regPrm;
% allregressors.prunePrm{2} = tmp.prunePrm;
% allregressors.prunePrm{2}.motion_2dto3D = false;
% allregressors.prunePrm{2}.motionparams = {'poslambda',.75};
% allregressors.prunePrm{2}.initfcn = @(p) InitializeSecondRoundTracking(p,1,2,0,tmp.prunePrm.windowradius);
% 
% tmp = load(trainresfile_pt2);
% allregressors.regModel{3} = tmp.regModel;
% allregressors.regPrm{3} = tmp.regPrm;
% allregressors.prunePrm{3} = tmp.prunePrm;
% allregressors.prunePrm{3}.motion_2dto3D = false;
% allregressors.prunePrm{3}.motionparams = {'poslambda',.75};
% allregressors.prunePrm{3}.initfcn = @(p) InitializeSecondRoundTracking(p,2,2,0,tmp.prunePrm.windowradius);
% 
% trainresfile_motion_combine = sprintf('TrainedModel_%s_2D_motion_combined.mat',savestr);
% save(trainresfile_motion_combine,'-struct','allregressors');
% 
% %% track movie
% 
% firstframe = 51;
% endframe = 250;
% expdir = '/tier2/hantman/Jay/videos/M174VGATXChR2/20150416L/L2secOn3Grab/M174_20150416_v007';
% testresfile = '';
% [phisPr,phisPrAll]=test(expdir,trainresfile_motion_combine,testresfile,...
%   'moviefilestr',ld.moviefilestr,'firstframe',firstframe,'endframe',endframe);
% 
% [readframe,nframes,fid] = get_readframe_fcn(fullfile(expdir,ld.moviefilestr));
% im = readframe(1);
% imsz = size(im);
% 
% fprintf('Initial tracking results\n');
% 
% figure(1);
% clf;
% him = imagesc(im,[0,255]);
% axis image;
% colormap gray;
% hold on;
% hother = nan(1,2);
% hother(1) = plot(nan,nan,'b.');
% hother(2) = plot(nan,nan,'b.');
% htrx = nan(1,2);
% htrx(1) = plot(nan,nan,'m.-');
% htrx(2) = plot(nan,nan,'c.-');
% hcurr = nan(1,2);
% hcurr(1) = plot(nan,nan,'mo','LineWidth',3);
% hcurr(2) = plot(nan,nan,'co','LineWidth',3);
% %htext = text(imsz(2)/2,5,'d. train = ??','FontSize',24,'HorizontalAlignment','center','VerticalAlignment','top','Color','r');
% 
% idxcurr = idx(ld.expidx(idx)==expi);
% tstraincurr = ld.ts(idxcurr);
% 
% %p = phisPr{1};
% p1 = phisPr{2};
% p2 = phisPr{3};
% 
% for t = 1:endframe-firstframe+1,
%   
%   im = readframe(firstframe+t-1);
%   set(him,'CData',im);
% %   set(hother(1),'XData',squeeze(pall(t,1,:)),'YData',squeeze(pall(t,3,:)));
% %   set(hother(2),'XData',squeeze(pall(t,2,:)),'YData',squeeze(pall(t,4,:)));
%   set(htrx(1),'XData',p1(max(1,t-500):t,1),'YData',p1(max(1,t-500):t,2));
%   set(htrx(2),'XData',p2(max(1,t-500):t,1),'YData',p2(max(1,t-500):t,2));
%   set(hcurr(1),'XData',p1(t,1),'YData',p1(t,2));
%   set(hcurr(2),'XData',p2(t,1),'YData',p2(t,2));
%   %pause(.25);
%   drawnow;
%   
% end

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
  %'/tier2/hantman/Jay/videos/M134C3VGATXChR2'
  %'/tier2/hantman/Jay/videos/M174VGATXChR2'
  '/tier2/hantman/Jay/videos/M173VGATXChR2'
  %'/tier2/hantman/Jay/videos/M147VGATXChrR2_anno'
  };

expdirs = cell(1,numel(rootdirs));
exptypes = cell(1,numel(rootdirs));

for i = 1:numel(rootdirs),
  
  [~,exptypes{i}] = fileparts(rootdirs{i});
  expdirs{i} = recursiveDir(rootdirs{i},'movie_comb.avi');
  
  expdirs{i} = cellfun(@(x) fileparts(x),expdirs{i},'Uni',0);

  fprintf('Tracking %d experiments in %s\n',numel(expdirs{i}),rootdirs{i});

end

saverootdir = '/groups/branson/home/bransonk/tracking/code/rcpr/data/TrackingResults20150531';
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
    fprintf(fid,'%s %s %s %s %s moviefilestr %s\n',...
      SCRIPT,MCR,expdir,trainresfile,testresfiles{typei}{i},ld.moviefilestr);
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
      savestuff.p_all{ii,1} = tmp.phisPr{1};
      savestuff.hyp_all{ii,1} = tmp.phisPrAll{1};
    end
    
    fprintf('Saving tracking results for %d videos to %s\n',numel(idxcurr),savefile);
    save(savefile,'-struct','savestuff');
  end
    
end

%% plot tracking results!

typei = 1;
order = randsample(numel(testresfiles{typei}),20);

vidobj = VideoWriter('TrackingResults_RandomVideos_20150531.avi');
open(vidobj);
gfdata = [];

for i = order(1:end)',
  
  while true,
  
    expdir = expdirs{typei}{i};
    trxfile = testresfiles{typei}{i};
    
    load(trxfile,'phisPr'),
    d0 = sqrt(sum(bsxfun(@minus,phisPr{1}(:,[1,3]),phisPr{1}(1,[1,3])).^2,2)+...
      sum(bsxfun(@minus,phisPr{1}(:,[2,4]),phisPr{1}(1,[2,4])).^2,2));
    d1 = sqrt(sum(bsxfun(@minus,phisPr{1}(:,[1,3]),phisPr{1}(end,[1,3])).^2,2)+...
      sum(bsxfun(@minus,phisPr{1}(:,[2,4]),phisPr{1}(end,[2,4])).^2,2));
    firstframe = max(1,find(d0 > 20,1)-25);
    endframe = min(size(phisPr{end},1),find(d1 > 20,1,'last')+25);
    if ~isempty(firstframe),
      break;
    end
    
    newi = randsample(setdiff(1:numel(testresfiles{typei}),order),1);
    order(order==i) = newi;
    i = newi;
    
  end
    
  [vidobj] = MakeTrackingResultsHistogramVideo(expdir,trxfile,'firstframe',firstframe,'endframe',endframe,...
    'vidobj',vidobj);
  
end

close(vidobj);

%% track videos that were missing from M174

rootdirs_missing = {%'/tier2/hantman/Jay/videos/M118_CNO_G6'
  %'/tier2/hantman/Jay/videos/M119_CNO_G6'
  %'/tier2/hantman/Jay/videos/M122_CNO_M1BPN_G6_anno'
  %'/tier2/hantman/Jay/videos/M127D22_hm4D_Sl1_BPN'
  %'/tier2/hantman/Jay/videos/M130_hm4DBPN_KordM1'
  %'/tier2/hantman/Jay/videos/M134C3VGATXChR2'
  '/tier2/hantman/Jay/videos/M174VGATXChR2'
  %'/tier2/hantman/Jay/videos/M173VGATXChR2'
  %'/tier2/hantman/Jay/videos/M147VGATXChrR2_anno'
  };

expdirs_missing = cell(1,numel(rootdirs_missing));
exptypes_missing = cell(1,numel(rootdirs_missing));

for i = 1:numel(rootdirs_missing),
  
  [~,exptypes_missing{i}] = fileparts(rootdirs_missing{i});
  expdirs_missing{i} = recursiveDir(rootdirs_missing{i},'movie_comb.avi');
  
  expdirs_missing{i} = cellfun(@(x) fileparts(x),expdirs_missing{i},'Uni',0);

  fprintf('Tracking %d experiments in %s\n',numel(expdirs_missing{i}),rootdirs_missing{i});

end

oldresdir = '/groups/branson/home/bransonk/tracking/code/rcpr/data/TrackingResults20150430';
oldresfiles = mydir(fullfile(oldresdir,'TrackingResults_*.mat'));
oldexpdirs = {};
for i = 1:numel(oldresfiles),
  
  tmp = load(oldresfiles{i});
  oldexpdirs = [oldexpdirs,cellfun(@fileparts,tmp.moviefiles_all,'uni',0)];
  
end

for i = 1:numel(rootdirs_missing),
  expdirs_missing{i} = setdiff(expdirs_missing{i},oldexpdirs);
end

saverootdir = '/groups/branson/home/bransonk/tracking/code/rcpr/data/TrackingResults_M174New_20150531';
if ~exist(saverootdir,'dir'),
  mkdir(saverootdir);
end

scriptfiles = {};
outfiles = {};
testresfiles = {};

for typei = numel(rootdirs_missing):-1:1,
  
  savedircurr = fullfile(saverootdir,exptypes_missing{typei});
  if ~exist(savedircurr,'dir'),
    mkdir(savedircurr);
  end
    
  for i = 1:numel(expdirs_missing{typei}),
    
    expdir = expdirs_missing{typei}{i};
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
    fprintf(fid,'%s %s %s %s %s moviefilestr %s\n',...
      SCRIPT,MCR,expdir,trainresfile,testresfiles{typei}{i},ld.moviefilestr);
    fclose(fid);
    unix(sprintf('chmod u+x %s',scriptfiles{typei}{i}));
  
    cmd = sprintf('ssh login1 ''source /etc/profile; cd %s; qsub -pe batch %d -N %s -j y -b y -o ''%s'' -cwd ''\"%s\"''''',...
      curdir,NCORESPERJOB,jobid,outfiles{typei}{i},scriptfiles{typei}{i});

    unix(cmd);
  end
  
end



% load and combine files

for typei = numel(rootdirs_missing):-1:1,
  
  parentdirs = regexp(expdirs_missing{typei},['^',rootdirs_missing{typei},'/([^/]*)/'],'tokens','once');
  parentdirs = [parentdirs{:}];
  [uniqueparentdirs,~,parentidx] = unique(parentdirs);
  
  for pdi = 1:numel(uniqueparentdirs),
    
    savefile = fullfile(saverootdir,sprintf('TrackingResults_%s_%s_%s.mat',exptypes_missing{typei},uniqueparentdirs{pdi},datestr(now,'yyyymmdd')));
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
      expdir = expdirs_missing{typei}{i};
      if ~exist(testresfiles{typei}{i},'file'),
        error('%s does not exist.\n',testresfiles{typei}{i});
      end
      savestuff.moviefiles_all{1,ii} = fullfile(expdir,ld.moviefilestr);
      tmp = load(testresfiles{typei}{i});
      savestuff.p_all{ii,1} = tmp.phisPr{1};
      savestuff.hyp_all{ii,1} = tmp.phisPrAll{1};
    end
    
    fprintf('Saving tracking results for %d videos to %s\n',numel(idxcurr),savefile);
    save(savefile,'-struct','savestuff');
  end
    
end

%% train first tracker with cross-validation

params = struct;
params.cpr_type = 'noocclusion';
params.model_type = 'mouse_paw2';
params.ftr_type = 6;
params.ftr_gen_radius = 100;
params.expidx = ld.expidx(idx);
params.ncrossvalsets = 50;
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

paramsfile1 = fullfile('/groups/branson/home/bransonk/tracking/code/rcpr/rcpr_v1_stable/misc',...
  sprintf('TrainData_%s.mat',savestr));
paramsfile2 = fullfile('/groups/branson/home/bransonk/tracking/code/rcpr/rcpr_v1_stable/misc',...
  sprintf('TrainParams_%s_2D_20150427_CV.mat',savestr));
trainresfile = sprintf('TrainedModel_%s_CV.mat',savestr);

params.expidx = ld.expidx(idx);
params.cvidx = CVSet(params.expidx,params.ncrossvalsets);

save(paramsfile2,'-struct','params');

trainsavedir = '/groups/branson/home/bransonk/tracking/code/rcpr/data/CrossValidationResults20150430';
if ~exist(trainsavedir,'dir'),
  mkdir(trainsavedir);
end

NCORESPERJOB = 4;
curdir = pwd;
TMP_ROOT_DIR = '/scratch/bransonk';
MCR_CACHE_ROOT = fullfile(TMP_ROOT_DIR,'mcr_cache_root');
MCR = '/groups/branson/bransonlab/share/MCR/v717';
SCRIPT = '/groups/branson/home/bransonk/tracking/code/rcpr/rcpr_v1_stable/misc/train/distrib/run_train.sh';

trainresfiles = cell(1,params.ncrossvalsets);
trainscriptfiles = cell(1,params.ncrossvalsets);
trainoutfiles = cell(1,params.ncrossvalsets);
for cvi = 1:params.ncrossvalsets,
  
  jobid = sprintf('train_%02d_%s',cvi,datestr(now,'yyyymmddTHHMMSSFFF'));
  trainresfiles{cvi} = fullfile(trainsavedir,[jobid,'.mat']);
  trainscriptfiles{cvi} = fullfile(trainsavedir,[jobid,'.sh']);
  trainoutfiles{cvi} = fullfile(trainsavedir,[jobid,'.log']);

  fid = fopen(trainscriptfiles{cvi},'w');
  fprintf(fid,'if [ -d %s ]\n',TMP_ROOT_DIR);
  fprintf(fid,'  then export MCR_CACHE_ROOT=%s.%s\n',MCR_CACHE_ROOT,jobid);
  fprintf(fid,'fi\n');
  fprintf(fid,'%s %s %s %s %s cvi %d nthreads %d',SCRIPT,MCR,paramsfile1,paramsfile2,trainresfiles{cvi},cvi,NCORESPERJOB);
  fclose(fid);
  unix(sprintf('chmod u+x %s',trainscriptfiles{cvi}));
  
  cmd = sprintf('ssh login1 ''source /etc/profile; cd %s; qsub -pe batch %d -N %s -j y -b y -o ''%s'' -cwd ''\"%s\"''''',...
    curdir,NCORESPERJOB,jobid,trainoutfiles{cvi},trainscriptfiles{cvi});

  unix(cmd);
end

% collect the results
td = struct('regModel',{{}},'regPrm',[],'prunePrm',[],...
  'phisPr',nan(numel(params.cvidx),4),'paramfile1','','err',nan(1,params.ncrossvalsets),...
  'paramfile2','','cvidx',cvidx);
for cvi = 1:params.ncrossvalsets,
 
  if ~exist(trainresfiles{cvi},'file'),
    fprintf('File %s does not exist, skipping\n',trainresfiles{cvi});
    continue;
  end
  tdcurr = load(trainresfiles{cvi});
  td.regModel{cvi} = tdcurr.regModel;
  if cvi == 1,
    td.regPrm = tdcurr.regPrm;
    td.prunePrm = tdcurr.prunePrm;
    td.paramfile1 = tdcurr.paramfile1;
    td.paramfile2 = tdcurr.paramfile2;
    td.cvidx = tdcurr.cvidx;
  end
  if isfield(td,'phisPr'),
    td.phisPr(cvidx==cvi,:) = tdcurr.phisPr;
  end
  if isfield(td,'err'),
    td.err(cvi) = tdcurr.err;
  end
end

td.H0 = H0;
td.prunePrm.motion_2dto3D = false;
td.prunePrm.motionparams = {'poslambda',.5};
save(trainresfile,'-struct','td');

% add motion parameters to individual tracking results
for cvi = 1:params.ncrossvalsets,
  if ~exist(trainresfiles{cvi},'file'),
    fprintf('File %s does not exist, skipping\n',trainresfiles{cvi});
    continue;
  end
  tdcurr = load(trainresfiles{cvi});
  tdcurr.H0 = H0;
  tdcurr.prunePrm.motion_2dto3D = false;
  tdcurr.prunePrm.motionparams = {'poslambda',.5};
  save(trainresfiles{cvi},'-struct','tdcurr');
end

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

for expi = 1:max(params.expidx),

  idxcurr = find(params.expidx==expi);
  if isempty(idxcurr),
    continue;
  end
  expdir = ld.expdirs{expi};
  
  cvi = unique(params.cvidx(idxcurr));
  assert(numel(cvi)==1);
  
  [~,n] = fileparts(expdir);
       
  jobid = sprintf('track_cv_%s_%s',n,datestr(now,'yyyymmddTHHMMSSFFF'));
  testresfiles_cv{expi} = fullfile(trainsavedir,[jobid,'.mat']);
  scriptfiles_cv{expi} = fullfile(trainsavedir,[jobid,'.sh']);
  outfiles_cv{expi} = fullfile(trainsavedir,[jobid,'.log']);

  fid = fopen(scriptfiles_cv{expi},'w');
  fprintf(fid,'if [ -d %s ]\n',TMP_ROOT_DIR);
  fprintf(fid,'  then export MCR_CACHE_ROOT=%s.%s\n',MCR_CACHE_ROOT,jobid);
  fprintf(fid,'fi\n');
  fprintf(fid,'%s %s %s %s %s moviefilestr %s\n',...
    SCRIPT,MCR,expdir,trainresfiles{cvi},testresfiles_cv{expi},ld.moviefilestr);
  fclose(fid);
  unix(sprintf('chmod u+x %s',scriptfiles_cv{expi}));
  
  cmd = sprintf('ssh login1 ''source /etc/profile; cd %s; qsub -pe batch %d -N %s -j y -b y -o ''%s'' -cwd ''\"%s\"''''',...
    curdir,NCORESPERJOB,jobid,outfiles_cv{expi},scriptfiles_cv{expi});

  unix(cmd);  
  
end

% collect results
trx = struct;
for expi = 1:max(params.expidx),

  idxcurr = find(params.expidx==expi);
  if isempty(idxcurr),
    continue;
  end
  expdir = ld.expdirs{expi};
  
  cvi = unique(params.cvidx(idxcurr));
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
  
  ts = ld.ts(idx(idxcurr));
  
  assert(all(ismember([repmat(expi,[numel(ts),1]),ts(:)],[ld.expidx(:),ld.ts(:)],'rows')));
  
  for i = 1:nphi,
    trx.phisPr{i}(idxcurr,:) = tdcurr.phisPr{i}(ts,:);
    trx.phisPrAll{i}(idxcurr,:,:) = tdcurr.phisPrAll{i}(ts,:,:);
  end  
end

save(fullfile(trainsavedir,sprintf('CVPredictions_%s.mat',savestr)),'-struct','trx');

%[regModeltmp,regPrmtmp,prunePrmtmp,phisPrtmp,errtmp] = train(paramsfile1,paramsfile2,trainresfile,'cvi',1);

%% train point 1 tracker w cross-validation

params = struct;
params.cpr_type = 'noocclusion';
params.model_type = 'mouse_paw';
params.ftr_type = 6;
params.ftr_gen_radius = 25;
params.expidx = ld.expidx(idx);
params.ncrossvalsets = 50;
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
%params.cascade_depth = 1;

medfilwidth = 10;

params.prunePrm.initfcn = @(p) InitializeSecondRoundTracking(p,1,2,medfilwidth,params.prunePrm.windowradius);

params.expidx = ld.expidx(idx);
load(paramsfile2,'cvidx');
params.cvidx = cvidx;

paramsfile1_pt1 = fullfile('/groups/branson/home/bransonk/tracking/code/rcpr/rcpr_v1_stable/misc',...
  sprintf('TrainData_%s_2D_pt1.mat',savestr));
paramsfile2_pt1 = sprintf('TrainParams_%s_2D_pt1_CV.mat',savestr);
trainresfile_pt1 = sprintf('TrainModel_%s_2D_pt1_CV.mat',savestr);

save(paramsfile2_pt1,'-struct','params');

trainresfiles_pt1 = cell(1,params.ncrossvalsets);
trainscriptfiles_pt1 = cell(1,params.ncrossvalsets);
trainoutfiles_pt1 = cell(1,params.ncrossvalsets);
for cvi = 1:params.ncrossvalsets,
  
  jobid = sprintf('train_%02d_pt1_%s',cvi,datestr(now,'yyyymmddTHHMMSSFFF'));
  trainresfiles_pt1{cvi} = fullfile(trainsavedir,[jobid,'.mat']);
  trainscriptfiles_pt1{cvi} = fullfile(trainsavedir,[jobid,'.sh']);
  trainoutfiles_pt1{cvi} = fullfile(trainsavedir,[jobid,'.log']);

  fid = fopen(trainscriptfiles_pt1{cvi},'w');
  fprintf(fid,'if [ -d %s ]\n',TMP_ROOT_DIR);
  fprintf(fid,'  then export MCR_CACHE_ROOT=%s.%s\n',MCR_CACHE_ROOT,jobid);
  fprintf(fid,'fi\n');
  fprintf(fid,'%s %s %s %s %s cvi %d nthreads %d',SCRIPT,MCR,paramsfile1_pt1,paramsfile2_pt1,trainresfiles_pt1{cvi},cvi,NCORESPERJOB);
  fclose(fid);
  unix(sprintf('chmod u+x %s',trainscriptfiles_pt1{cvi}));
  
  cmd = sprintf('ssh login1 ''source /etc/profile; cd %s; qsub -pe batch %d -N %s -j y -b y -o ''%s'' -cwd ''\"%s\"''''',...
    curdir,NCORESPERJOB,jobid,trainoutfiles_pt1{cvi},trainscriptfiles_pt1{cvi});

  unix(cmd);
end


%[regModeltmp,regPrmtmp,prunePrmtmp,phisPrtmp,errtmp] = train(paramsfile1_pt1,paramsfile2_pt1,trainresfile_pt1,'cvi',1);

%% train point 2 tracker w cross-validation

params = struct;
params.cpr_type = 'noocclusion';
params.model_type = 'mouse_paw';
params.ftr_type = 6;
params.ftr_gen_radius = 25;
params.expidx = ld.expidx(idx);
params.ncrossvalsets = 50;
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
%params.cascade_depth = 1;

medfilwidth = 10;

params.prunePrm.initfcn = @(p) InitializeSecondRoundTracking(p,1,2,medfilwidth,params.prunePrm.windowradius);

params.expidx = ld.expidx(idx);
load(paramsfile2,'cvidx');
params.cvidx = cvidx;

paramsfile1_pt2 = fullfile('/groups/branson/home/bransonk/tracking/code/rcpr/rcpr_v1_stable/misc',...
  sprintf('TrainData_%s_2D_pt2.mat',savestr));
paramsfile2_pt2 = sprintf('TrainParams_%s_2D_pt2_CV.mat',savestr);
trainresfile_pt2 = sprintf('TrainModel_%s_2D_pt2_CV.mat',savestr);

save(paramsfile2_pt2,'-struct','params');

trainresfiles_pt2 = cell(1,params.ncrossvalsets);
trainscriptfiles_pt2 = cell(1,params.ncrossvalsets);
trainoutfiles_pt2 = cell(1,params.ncrossvalsets);
for cvi = 1:params.ncrossvalsets,
  
  jobid = sprintf('train_%02d_pt2_%s',cvi,datestr(now,'yyyymmddTHHMMSSFFF'));
  trainresfiles_pt2{cvi} = fullfile(trainsavedir,[jobid,'.mat']);
  trainscriptfiles_pt2{cvi} = fullfile(trainsavedir,[jobid,'.sh']);
  trainoutfiles_pt2{cvi} = fullfile(trainsavedir,[jobid,'.log']);

  fid = fopen(trainscriptfiles_pt2{cvi},'w');
  fprintf(fid,'if [ -d %s ]\n',TMP_ROOT_DIR);
  fprintf(fid,'  then export MCR_CACHE_ROOT=%s.%s\n',MCR_CACHE_ROOT,jobid);
  fprintf(fid,'fi\n');
  fprintf(fid,'%s %s %s %s %s cvi %d nthreads %d',SCRIPT,MCR,paramsfile1_pt2,paramsfile2_pt2,trainresfiles_pt2{cvi},cvi,NCORESPERJOB);
  fclose(fid);
  unix(sprintf('chmod u+x %s',trainscriptfiles_pt2{cvi}));
  
  cmd = sprintf('ssh login1 ''source /etc/profile; cd %s; qsub -pe batch %d -N %s -j y -b y -o ''%s'' -cwd ''\"%s\"''''',...
    curdir,NCORESPERJOB,jobid,trainoutfiles_pt2{cvi},trainscriptfiles_pt2{cvi});

  unix(cmd);
end


%[regModeltmp,regPrmtmp,prunePrmtmp,phisPrtmp,errtmp] = train(paramsfile1_pt2,paramsfile2_pt2,trainresfile_pt2,'cvi',1);

%% collect all 3 tracking results

allregressors_cv = struct;
allregressors_cv.regModel = cell(1,3);
allregressors_cv.regPrm = cell(1,3);
allregressors_cv.prunePrm = cell(1,3);
allregressors_cv.H0 = H0;
allregressors_cv.traindeps = [0,1,1];

trainresfiles_motion_combine = cell(1,params.ncrossvalsets);

for cvi = 1:params.ncrossvalsets,
  trainresfiles_motion_combine{cvi} = fullfile(trainsavedir,sprintf('TrainedModel_%s_cv%02d_2D_motion_combined.mat',savestr,cvi));
  
  fprintf('%s...\n',trainresfiles_motion_combine{cvi});

  if ~exist(trainresfiles{cvi},'file'),
    fprintf('File %s does not exist, skipping\n',trainresfiles{cvi});
  else
    tmp = load(trainresfiles{cvi});
    allregressors_cv.regModel{1} = tmp.regModel;
    allregressors_cv.regPrm{1} = tmp.regPrm;
    allregressors_cv.prunePrm{1} = tmp.prunePrm;
    allregressors_cv.prunePrm{1}.motion_2dto3D = false;
    allregressors_cv.prunePrm{1}.motionparams = {'poslambda',.5};
  end

  if ~exist(trainresfiles_pt1{cvi},'file'),
    fprintf('File %s does not exist, skipping\n',trainresfiles_pt1{cvi});
  else
    tmp = load(trainresfiles_pt1{cvi});
    allregressors_cv.regModel{2} = tmp.regModel;
    allregressors_cv.regPrm{2} = tmp.regPrm;
    allregressors_cv.prunePrm{2} = tmp.prunePrm;
    allregressors_cv.prunePrm{2}.motion_2dto3D = false;
    allregressors_cv.prunePrm{2}.motionparams = {'poslambda',.75};
    allregressors_cv.prunePrm{2}.initfcn = @(p) InitializeSecondRoundTracking(p,1,2,0,tmp.prunePrm.windowradius);
  end

  if ~exist(trainresfiles_pt2{cvi},'file'),
    fprintf('File %s does not exist, skipping\n',trainresfiles_pt2{cvi});
  else
    tmp = load(trainresfiles_pt2{cvi});
    allregressors_cv.regModel{3} = tmp.regModel;
    allregressors_cv.regPrm{3} = tmp.regPrm;
    allregressors_cv.prunePrm{3} = tmp.prunePrm;
    allregressors_cv.prunePrm{3}.motion_2dto3D = false;
    allregressors_cv.prunePrm{3}.motionparams = {'poslambda',.75};
    allregressors_cv.prunePrm{3}.initfcn = @(p) InitializeSecondRoundTracking(p,2,2,0,tmp.prunePrm.windowradius);
  end
  save(trainresfiles_motion_combine{cvi},'-struct','allregressors_cv');
end

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
  fprintf(fid,'%s %s %s %s %s moviefilestr %s\n',...
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
nbins = 1000;
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
% hpermouse = plot(ctrs,cumsum(fracpermouse,2),'-');
% hold on;
h = plot(ctrs,cumsum(fractotal),'k-','LineWidth',3);
hold on;
box off;
xlabel('Mean squared error, both views (px)','FontSize',24);
ylabel('Cumulative frac. data','FontSize',24);

ylim = [0,1.05];
%ylim = get(gca,'YLim');
ylim(1) = 0;
set(gca,'YLim',ylim);

prctiles = [50,75,95,99];%,100];
humanerr = [5.736611,9.073831,11.347343,13.273021];%,19.030200];
autoerr = prctile(err,prctiles);

colors = jet(256);
colors = colors(RescaleToIndex(1:numel(prctiles)+1,256),:);
colors = flipud(colors)*.7;

for j = 1:numel(prctiles),
  plot(humanerr(j),prctiles(j)/100,'o','Color',colors(j,:),'MarkerFaceColor',colors(j,:),'LineWidth',2,'MarkerSize',12);
  plot([humanerr(j),autoerr(j)],prctiles(j)/100+[0,0],'--','Color',colors(j,:),'MarkerFaceColor',colors(j,:),'LineWidth',4);
  plot(autoerr(j),prctiles(j)/100,'x','Color',colors(j,:),'MarkerFaceColor',colors(j,:),'LineWidth',2,'MarkerSize',12);
  text(humanerr(j),prctiles(j)/100,sprintf('%d%% ',prctiles(j)),'HorizontalAlignment','right','VerticalAlignment','middle',...
    'Color',colors(j,:),'FontSize',24);
end
set(gca,'FontSize',24);

disp(autoerr);

%dy = ylim(2) / (numel(prctiles)+1);
% colors = jet(256);
% colors = colors(RescaleToIndex(1:numel(prctiles)+1,256),:);
% colors = flipud(colors);
% 
% for j = 1:numel(prctiles),
%   plot([humanerr(j),autoerr(j)],dy*j+[0,0],'-','Color',colors(j+1,:)*.7,'LineWidth',2);
%   hhuman = plot(humanerr(j),dy*j,'o','Color',colors(j+1,:)*.7,'LineWidth',2,'MarkerSize',12);  
%   hauto = plot(autoerr(j),dy*j,'x','Color',colors(j+1,:)*.7,'LineWidth',2,'MarkerSize',12);
%   text(max([humanerr(j),autoerr(j)]),dy*j,sprintf('   %dth %%ile',prctiles(j)),'HorizontalAlignment','left',...
%     'Color',.7*colors(j+1,:),'FontWeight','bold');
% end
% 
% legend([hhuman,hauto],'Human','Tracker');
% 5.6771    9.4123   18.1824   33.2849
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
