%% paths

addpath /groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect/misc;
addpath /groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect/filehandling;

datatype = 'romain';

switch datatype,
  case 'stephen'
    rootdatadir = '/groups/branson/bransonlab/mayank/stephenCV/results';
    lblfile = '/groups/branson/bransonlab/mayank/PoseTF/headTracking/FlyHeadStephenRound1_Janelia_fixedmovies.lbl';
    cvsplitfile = '/groups/branson/bransonlab/mayank/stephenCV/cvSplit.mat';
    savefile = 'GTNNData_Stephen20170410.mat';
    ddrootdir = '/groups/branson/bransonlab/mayank/stephenCV/';
    lblrootdir = '/groups/branson/bransonlab/mayank/PoseEstimationData/Stephen/';
    figsavestr = '_Stephen20170410';
    lkskeep = [];
  case 'jan'
    rootdatadir = '/nrs/branson/mayank/janResults';
    lblfile = '/groups/branson/bransonlab/mayank/PoseTF/janLegTracking/160819_Dhyey_2_al_fixed.lbl';
    cvsplitfile = '/groups/branson/bransonlab/mayank/PoseTF/janLegTracking/valSplits.mat';
    savefile = 'GTNNData_Jan20170414.mat';
    figsavestr = '_Jan20170414';
    lkskeep = 4:7;
  case 'roian'
    rootdatadir = '/nrs/branson/mayank/roianResults';
    lblfile = '/groups/branson/bransonlab/mayank/PoseTF/data/roian/head_tail_20170411.lbl';
    cvsplitfile = '/groups/branson/bransonlab/mayank/PoseTF/data/roian/valSplits.mat';
    savefile = 'GTNNData_Roian20170416.mat';
    figsavestr = '_Roian20170416';
    lkskeep = [];
  case 'jay'
    rootdatadir = '/nrs/branson/mayank/jay/results';
    lblfile = '/groups/branson/bransonlab/mayank/PoseTF/data/jayMouse/miceLabels_20170412.lbl';
    cvsplitfile = '/groups/branson/bransonlab/mayank/PoseTF/data/jayMouse/valSplits.mat';
    savefile = 'GTNNData_Jay20170416.mat';
    figsavestr = '_Jay20170416';
    lkskeep = [];
  case 'romain'
    rootdatadir = '/nrs/branson/mayank/romain/results';
    lblfile = '/groups/branson/bransonlab/mayank/PoseTF/RomainLeg/RomainCombined_fixed_fixedbabloo_20170410.lbl';
    cvsplitfile = '/groups/branson/bransonlab/mayank/PoseTF/RomainLeg/valSplits.mat';
    savefile = 'GTNNData_Romain20170419.mat';
    figsavestr = '_Romain20170419';
    lkskeep = [];
end


gtmarker = '+';
predmarkers = 'osd';
gtcolor = [.75,0,.75];
predcolors = [
  .25,.25,1
  0,.75,.75
  0,.75,0
];
if strcmp(datatype,'roian'),
  predfns = {'pd_locs','mrf_locs'};
  prednames = {'Part detector','+ 2D pose'};
else
  predfns = {'pd_locs','mrf_locs','final_locs'};
  prednames = {'Part detector','+ 2D pose','+ 3D pose + time'};
end
npredfns = numel(predfns);

dosavefigs = true;

%% collect and load data

lbld = load(lblfile,'-mat');
lbld.animalids = ParseAnimalIdFromFileNames(lbld.movieFilesAll(:,1),datatype);

cvd = load(cvsplitfile);

% if split is a matrix
if ~iscell(cvd.split),
  ncvs = size(cvd.split,1);
  split = cell(1,ncvs);
  for i = 1:ncvs,
    split{i} = double(cvd.split(i,:));
  end
  cvd.split = split;
end

% 0-indexed, fix
for i = 1:numel(cvd.split),
  cvd.split{i} = cvd.split{i} + 1;
end

% make sure animal ids are separate per split
cvd.animalids = cell(size(cvd.split));
for i = 1:numel(cvd.split),
  cvd.animalids{i} = unique(lbld.animalids(cvd.split{i}));
end
if ~ismember(datatype,{'jay'}),
  for i = 1:numel(cvd.split),
    for j = i+1:numel(cvd.split),
      assert(isempty(intersect(cvd.animalids{i},cvd.animalids{j})));
    end
  end
end

lbld0 = lbld;
cvd0 = cvd;

if strcmp(datatype,'romain'),
  hmapmatfiles = {};
  detectmatfiles = mydir(fullfile(rootdatadir,'*.mat'));
else
  hmapmatfiles = mydir(fullfile(rootdatadir,'*hmap.mat'));
  detectmatfiles = strrep(hmapmatfiles,'_hmap','');
end

nmovies = numel(detectmatfiles);

assert(all(cellfun(@exist,detectmatfiles)>0));

% remove duplicates in lbld
nlblmovies = size(lbld.movieFilesAll,1);
doremove = false(1,nlblmovies);
ischecked = false(1,nlblmovies);
replaced = 1:nlblmovies;


mstrs = cell(1,nlblmovies);
for i = 1:nlblmovies,
  moviename = lbld.movieFilesAll{i,1};
  ss = strsplit_jaaba(moviename,'/');
  switch datatype,
    case 'stephen',
      mstr = fullfile(ss{end-2:end});
    case {'jan','roian','romain'}
      mstr = ss{end};
    case 'jay'
      mstr = regexp(moviename,'M\d{3}_\d{8}_[vV]\d+[^\d]','once','match');
      mstr = mstr(1:end-1);
      assert(~isempty(mstr));
    otherwise
      error('not implemented');
  end
  mstrs{i} = mstr;
end
for i = 1:nlblmovies,
  if ischecked(i),
    continue;
  end
  
  moviename = lbld.movieFilesAll{i,1};
 
  mstr = mstrs{i};
  idx = find(strcmp(mstrs,mstr));
  %idx = find(cellfun(@(x) endsWith(x,mstr),lbld.movieFilesAll(:,1)));

  if numel(idx)==1,
    assert(i==idx);
    ischecked(i) = true;
    continue;
  end
  
  islabeled = false(1,numel(idx));
  for jj = 1:numel(idx),
    j = idx(jj);
    islabeled(jj) = any(IsLabeled(lbld.labeledpos{j}));
    % reshape(all(all(~isnan(lbld.labeledpos{j}),1),2),[1,size(lbld.labeledpos{j},3)]));
  end
  
  if ~any(islabeled),
    jkeep = idx(1);
    %fprintf('%d: none labeled\n',i);
  elseif nnz(islabeled) == 1,
    jkeep = idx(islabeled);
    %fprintf('%d: one labeled\n',i);
  else
    labeledidx = idx(islabeled);
    issame = true(numel(labeledidx));
    nlabeled = nan(1,numel(labeledidx));
    for jj1 = 1:numel(labeledidx),
      j1 = labeledidx(jj1);
      islabeled1 = reshape(all(all(~isnan(lbld.labeledpos{j1}),1),2),[1,size(lbld.labeledpos{j1},3)]);
      nlabeled(jj1) = nnz(islabeled1);
      for jj2 = jj1+1:numel(labeledidx),
        j2 = labeledidx(jj2);
        % check if these are the same
        islabeled2 = reshape(all(all(~isnan(lbld.labeledpos{j2}),1),2),[1,size(lbld.labeledpos{j2},3)]);
        issame(jj1,jj2) = all(islabeled1==islabeled2);
        if issame,
          issame(jj1,jj2) = all(all(lbld.labeledpos{j1}(:,:,islabeled1)==lbld.labeledpos{j2}(:,:,islabeled1)));
        end
      end
    end
    
    if ~all(issame(:)),
      fprintf('Found duplicate of %s with mismatching labels:\n',mstr);
      for jj = 1:numel(labeledidx),
        j = labeledidx(jj);
        fprintf('%s, %d frames labeled\n',lbld.movieFilesAll{j,1},nlabeled(jj));
      end
    end
    jkeep = labeledidx(argmax(nlabeled));
  end
  idxremove = setdiff(idx,jkeep);
  
  nlabeledkeep = nnz(all(all(~isnan(lbld.labeledpos{jkeep}),1),2));
  for j = idxremove(:)',
    nlabeledremove = nnz(all(all(~isnan(lbld.labeledpos{j}),1),2));
    fprintf('%s: Removing %d (%s, %d labels), keeping %d (%s, %d labels)\n',mstr,j,lbld.movieFilesAll{j,1},nlabeledremove,jkeep,lbld.movieFilesAll{jkeep,1},nlabeledkeep);
  end
  replaced(idxremove) = jkeep;
  doremove(idxremove) = true;
  ischecked(idx) = true;
end

lbld.movieFilesAll(doremove,:) = [];
lbld.labeledpos(doremove) = [];
lbld.animalids(doremove) = [];

new2oldidx = 1:nlblmovies;
new2oldidx(doremove) = [];
[~,old2newidx] = ismember(1:nlblmovies,new2oldidx);

cvd.animalids = cell(size(cvd.split));
for i = 1:numel(cvd.split),
  idxremoved = cvd.split{i}(old2newidx(cvd.split{i})==0);
  idxreplaced = replaced(idxremoved);
  assert(all(ismember(idxreplaced,cvd.split{i})));
  cvd.split{i} = old2newidx(cvd.split{i}(old2newidx(cvd.split{i})~=0));
  cvd.animalids{i} = unique(lbld.animalids(cvd.split{i}));
  if ~ismember(datatype,{'jay'}),
    assert(isempty(setxor(cvd.animalids{i},cvd0.animalids{i})));
  end
end

nlblmovies = size(lbld.movieFilesAll,1);
nviews = size(lbld.movieFilesAll,2);
nlandmarks = size(lbld.labeledpos{1},1)/nviews;

% if we don't have detectors for all landmarks
if ~isempty(lkskeep),
  
  lkidx = bsxfun(@plus,(0:nviews-1)*nlandmarks,lkskeep(:));
  
  for j = 1:nlblmovies,
    lbld.labeledpos{j} = lbld.labeledpos{j}(lkidx(:),:,:);
  end
  nlandmarks = maxnlandmarks;
  
end

% add predicted data to lbld
for i = 1:npredfns,
  predfn = predfns{i};
  lbld.(predfn) = cell(size(lbld.labeledpos));
end

% index of movie in dd in lbld
dd2lblidx = nan(1,nmovies);

% loop through all movies with predictions
for i = 1:nmovies,
  
  % load in predicted data
  dd = load(detectmatfiles{i});
  dd = dd.R;
  ddmoviefiles = cellfun(@(x) x.movie, dd,'Uni',0);

  % find match
  switch datatype,
    case 'stephen'
      j = GetLabelMovieIdx(ddmoviefiles,lbld,ddrootdir);
    case {'jan','roian','jay'}
      j = find(strcmp(lbld.movieFilesAll(:,1),ddmoviefiles{1}));
    otherwise
      error('not implemented');
  end
  assert(numel(j)==1);
  dd2lblidx(i) = j;
  
  % loop through prediction types
  for k = 1:npredfns,
    predfn = predfns{k};
    lbld.(predfn){j} = nan(size(lbld.labeledpos{j}));
    for viewi = 1:nviews,
      % copy over
      off = (viewi-1)*nlandmarks;
      lbld.(predfn){j}(off+1:off+nlandmarks,:,:) = permute(dd{viewi}.(predfn),[2,3,1]);
    end
  end
  
end

% check for labeled movies that have no predictions
fprintf('Labeled data with no predictions:\n');
for j = 1:numel(lbld.labeledpos),
  islabeled = IsLabeled(lbld.labeledpos{j});
  %islabeled = reshape(all(all(~isnan(lbld.labeledpos{j}),1),2),[1,size(lbld.labeledpos{j},3)]);
  if ~any(islabeled),
    continue;
  end
  ispred = ~isempty(lbld.pd_locs{j});
  ispred2 = any(dd2lblidx==j);
  assert(ispred==ispred2);
  if ~ispred,
    fprintf('%d: %s, %d frames labeled\n',j,lbld.movieFilesAll{j,1},nnz(islabeled));
  end
end

% add the hmap file to the lbldata struct
lbld.hmapmatfiles = cell(1,nlblmovies);
lbld.detectmatfiles = cell(1,nlblmovies);
for i = 1:numel(detectmatfiles),
  j = dd2lblidx(i);
  lbld.detectmatfiles{j} = detectmatfiles{i};
  lbld.hmapmatfiles{j} = hmapmatfiles{i};
end

%% save

save(savefile,'lbld','cvd','detectmatfiles','hmapmatfiles');

%% plot detection vs label for some examples

nsamplesperid = 3;
naxperfig = 5;
hfigbase = 100;
lkcolors = lines(nlandmarks);

labeledframes = nan(1,0);
movieidx = nan(1,0);

off = 0;
allpredsfound = nan(1,nlblmovies);
expislabeled = false(1,nlblmovies);
for j = 1:nlblmovies,

  % which frames are labeled
  islabeled = IsLabeled(lbld.labeledpos{j});
  if ~any(islabeled),
    continue;
  end
  expislabeled(j) = true;
    
  % whether they are predicted for all predfns
  ispred = false(1,npredfns);
  for p = 1:npredfns,
    if isempty(lbld.(predfns{p}){j}),
      ispred(p) = false;
    else
      ispredcurr = IsLabeled(lbld.(predfns{p}){j});
      ispred(p) = all(ispredcurr(islabeled));
    end
  end
  if ~all(ispred),
    fprintf('No predictions for movie %d for the following predictors:\n',j);
    fprintf('%s ',predfns{~ispred});
    fprintf('\n');
    allpredsfound(j) = 0;
    continue;
  end
  fprintf('All predictions available for movie %d\n',j);
  allpredsfound(j) = 1;
  
  nlabeledcurr = nnz(islabeled);

  movieidx(off+1:off+nlabeledcurr) = j;
  labeledframes(off+1:off+nlabeledcurr) = find(islabeled);
  
  off = off + nlabeledcurr;
  
end

animalids = unique(lbld.animalids(expislabeled));

idxsample = [];
erridx2animalid = lbld.animalids(movieidx);
for i = 1:numel(animalids),
  id = animalids(i);
  idxcurr = find(erridx2animalid==id);

  if isempty(idxcurr),
    continue;
  end
  if numel(idxcurr) > nsamplesperid,
    idxcurr = idxcurr(round(linspace(1,numel(idxcurr),nsamplesperid)));
  end
  idxsample = [idxsample;idxcurr]; %#ok<AGROW>
end


hfigs = PlotExampleLabelsAndPreds(lbld,predfns,movieidx(idxsample),labeledframes(idxsample),...
  'hfigbase',hfigbase,'naxperfig',naxperfig,...
  'landmarkcolors',lkcolors,'figpos',[10,10,2100,1500],...
  'prednames',prednames);

if dosavefigs,
  for figi = 1:numel(hfigs),
    hfig = hfigs(figi);
    set(hfig,'Color','w','InvertHardCopy','off');
    SaveFigLotsOfWays(hfig,sprintf('TrackingExamples_%02d%s',figi,figsavestr),{'pdf','fig','png'});
  end
end

