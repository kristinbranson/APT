% add missing data to lbld

%% paths

addpath /groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect/misc;
addpath /groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect/filehandling;

datatype = 'roian';

switch datatype,
  case 'stephen'
    nnrootdatadir = '/groups/branson/bransonlab/mayank/stephenCV/results';
    nnlblfile = '/groups/branson/bransonlab/mayank/PoseTF/headTracking/FlyHeadStephenRound1_Janelia_fixedmovies.lbl';
    nncvsplitfile = '/groups/branson/bransonlab/mayank/stephenCV/cvSplit.mat';
    nnsavefile = 'GTNNData_Stephen20170410.mat';
    ddrootdir = '/groups/branson/bransonlab/mayank/stephenCV/';
    lkskeep = [];
    
    cprresfile = '/groups/branson/home/bransonk/tracking/code/APT/gt/TrainedCPRTrackers_Stephen20170411.mat';
    
    finerootdatadir = '/groups/branson/bransonlab/mayank/stephenCV/fineResults';
    
    addsavefile = '/groups/branson/home/bransonk/tracking/code/APT/gt/PredLabelsAndCPR_Stephen20170411.mat';
    
  case 'jan'
    nnrootdatadir = '/nrs/branson/mayank/janResults';
    nnlblfile = '/groups/branson/bransonlab/mayank/PoseTF/janLegTracking/160819_Dhyey_2_al_fixed.lbl';
    nncvsplitfile = '/groups/branson/bransonlab/mayank/PoseTF/janLegTracking/valSplits.mat';
    nnsavefile = 'GTNNData_Jan20170414.mat';
    figsavestr = '_Jan20170414';
    lkskeep = 4:7;
    
    cprresfile = '/groups/branson/home/bransonk/tracking/code/APT/gt/TrainedCPRTrackers_Jan20170415.mat';
    
    finerootdatadir = '/nrs/branson/mayank/janFineResults';
    
    addsavefile = '/groups/branson/home/bransonk/tracking/code/APT/gt/PredLabelsAndCPR_Jan20170415.mat';
    
  case 'roian'
    nnrootdatadir = '/nrs/branson/mayank/roianResults';
    nnlblfile = '/groups/branson/bransonlab/mayank/PoseTF/data/roian/head_tail_20170411.lbl';
    nncvsplitfile = '/groups/branson/bransonlab/mayank/PoseTF/data/roian/valSplits.mat';
    nnsavefile = 'GTNNData_Roian20170416.mat';
    lkskeep = [];
    
    cprresfile = '/groups/branson/home/bransonk/tracking/code/APT/gt/TrainedCPRTrackers_Roian20170416.mat';
    finerootdatadir = '';
    addsavefile = '/groups/branson/home/bransonk/tracking/code/APT/gt/PredLabelsAndCPR_Roian20170416.mat';
  case 'jay'
    nnrootdatadir = '/nrs/branson/mayank/jay/results';
    nnlblfile = '/groups/branson/bransonlab/mayank/PoseTF/data/jayMouse/miceLabels_20170412.lbl';
    nncvsplitfile = '/groups/branson/bransonlab/mayank/PoseTF/data/jayMouse/valSplits.mat';
    nnsavefile = 'GTNNData_Jay20170416.mat';
    lkskeep = [];
    
    
    cprresfile = '/groups/branson/home/bransonk/tracking/code/APT/gt/TrainedCPRTrackers_Jay20170416.mat';
    
    finerootdatadir = '/nrs/branson/mayank/jay/fineResults';
    
    addsavefile = '/groups/branson/home/bransonk/tracking/code/APT/gt/PredLabelsAndCPR_Jay20170416.mat';
    
end


if strcmp(datatype,'roian'),
  predfns = {'pd_locs','mrf_locs'};
  prednames = {'Part detector','+ 2D pose'};
else
  predfns = {'pd_locs','mrf_locs','final_locs'};
  prednames = {'Part detector','+ 2D pose','+ 3D pose + time'};
end
npredfns = numel(predfns);

%% add in detection file locations

load(cprresfile);
nlblmovies = size(lbld.movieFilesAll,1);
nviews = size(lbld.movieFilesAll,2);
nlandmarks = size(lbld.labeledpos{1},1)/nviews;

hmapmatfiles = mydir(fullfile(nnrootdatadir,'*hmap.mat'));
detectmatfiles = strrep(hmapmatfiles,'_hmap','');
assert(all(cellfun(@exist,detectmatfiles)>0));

lbld.detectmatfiles = cell(1,nlblmovies);
lbld.hmapmatfiles = cell(1,nlblmovies);

% loop through all movies with predictions
for i = 1:numel(detectmatfiles),
  
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
  
  lbld.detectmatfiles{j} = detectmatfiles{i};
  lbld.hmapmatfiles{j} = hmapmatfiles{i};
  
end

%% load in fine detection results

if ~isempty(finerootdatadir),
  
  lbld.finematfiles = cell(1,nlblmovies);
  
  finematfiles = mydir(fullfile(finerootdatadir,'*.mat'));
  ishmap = ~cellfun(@isempty,regexp(finematfiles,'hmap\.mat','once'));
  assert(~any(ishmap));
  
  finepredfn = 'fine_locs';
  
  lbld.(finepredfn) = cell(size(lbld.labeledpos));
  for i = 1:numel(finematfiles),
    
    % load in predicted data
    dd = load(finematfiles{i});
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
    
    lbld.finematfiles{j} = finematfiles{i};
    
    lbld.(finepredfn){j} = nan(size(lbld.labeledpos{j}));
    for viewi = 1:nviews,
      % copy over
      off = (viewi-1)*nlandmarks;
      lbld.(finepredfn){j}(off+1:off+nlandmarks,:,:) = permute(dd{viewi}.(finepredfn),[2,3,1]);
    end
    
  end
end

%% pd, etc missing?

predfnsadd = {'pd_locs','mrf_locs','final_locs'};

if all(cellfun(@isempty,lbld.pd_locs)),
  
  for i = 1:nlblmovies,
    
    if isempty(lbld.detectmatfiles{i}),
      continue;
    end
    dd = load(lbld.detectmatfiles{i});
    dd = dd.R;
    
    for k = 1:numel(predfnsadd),
      predfn = predfnsadd{k};      
      lbld.(predfn){i} = nan(size(lbld.labeledpos{i}));
      for viewi = 1:nviews,
        % copy over
        off = (viewi-1)*nlandmarks;
        lbld.(predfn){i}(off+1:off+nlandmarks,:,:) = permute(dd{viewi}.(predfn),[2,3,1]);
      end
      
    end
    
  end
  
end

%% save

copyfile(cprresfile,addsavefile);
save('-append',addsavefile,'lbld');
