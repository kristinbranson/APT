%% paths

addpath /groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect/misc;
addpath /groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect/filehandling;

rootdatadir = '/groups/branson/bransonlab/mayank/stephenCV/results';
lblfile = '/groups/branson/bransonlab/mayank/PoseTF/headTracking/FlyHeadStephenRound1_Janelia_fixedmovies.lbl';
cvsplitfile = '/groups/branson/bransonlab/mayank/stephenCV/cvSplit.mat';
savefile = 'GTNNData_Stephen20170410.mat';

ddrootdir = '/groups/branson/bransonlab/mayank/stephenCV/';
lenddrootdir = numel(ddrootdir);
lblrootdir = '/groups/branson/bransonlab/mayank/PoseEstimationData/Stephen/';

gtmarker = '+';
predmarkers = 'osd';
gtcolor = [.75,0,.75];
predcolors = [
  .25,.25,1
  0,.75,.75
  0,.75,0
];
predfns = {'pd_locs','mrf_locs','final_locs'};
prednames = {'Part detector','+ 2D pose','+ 3D pose + time'};
npredfns = numel(predfns);
landmarkorder = [3,1,5,2,4,3];

figsavestr = '_Stephen20170410';
dosavefigs = true;

%% collect and load data

lbld = load(lblfile,'-mat');
lbld.animalids = regexp(lbld.movieFilesAll(:,1),'fly_?(\d+)[^\d]','tokens');
assert(~any(cellfun(@isempty,lbld.animalids)));
lbld.animalids = cellfun(@(x) str2double(x{end}),lbld.animalids);
assert(all(~isnan(lbld.animalids)));

cvd = load(cvsplitfile);
% 0-indexed, fix
for i = 1:numel(cvd.split),
  cvd.split{i} = cvd.split{i} + 1;
end

% make sure animal ids are separate per split
cvd.animalids = cell(size(cvd.split));
for i = 1:numel(cvd.split),
  cvd.animalids{i} = unique(lbld.animalids(cvd.split{i}));
end
for i = 1:numel(cvd.split),
  for j = i+1:numel(cvd.split),
    assert(isempty(intersect(cvd.animalids{i},cvd.animalids{j})));
  end
end


lbld0 = lbld;
cvd0 = cvd;

hmapmatfiles = mydir(fullfile(rootdatadir,'*hmap.mat'));
detectmatfiles = strrep(hmapmatfiles,'_hmap','');
nmovies = numel(detectmatfiles);

assert(all(cellfun(@exist,detectmatfiles)>0));

% remove duplicates in lbld
nlblmovies = size(lbld.movieFilesAll,1);
doremove = false(1,nlblmovies);
ischecked = false(1,nlblmovies);
replaced = 1:nlblmovies;
for i = 1:nlblmovies,
  if ischecked(i),
    continue;
  end
  
  moviename = lbld.movieFilesAll{i,2};
  ss = strsplit_jaaba(moviename,'/');
  mstr = fullfile(ss{end-2:end});
  idx = find(cellfun(@(x) endsWith(x,mstr),lbld.movieFilesAll(:,2)));

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
  assert(isempty(setxor(cvd.animalids{i},cvd0.animalids{i})));
end

% data size
dd = load(detectmatfiles{1});
dd = dd.R;
nviews = numel(dd);
nlandmarks = size(dd{1}.(predfns{1}),2);

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
  j = GetLabelMovieIdx(ddmoviefiles,lbld,ddrootdir);
  assert(~isempty(j));
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

%% save

save(savefile,'lbld','cvd','detectmatfiles','hmapmatfiles');

%% plot detection vs label for some examples

% which movie
i = 1;

% plot params
nplot = 5;
hfig = 1;

% index into lbld
j = dd2lblidx(i);

% which frames are labeled
islabeled = IsLabeled(lbld.labeledpos{j});
%islabeled = reshape(all(all(~isnan(lbld.labeledpos{j}),1),2),[1,size(lbld.labeledpos{j},3)]);
fslabeled = find(islabeled);
nlabeledcurr = numel(fslabeled);
if nlabeledcurr > nplot,
  fslabeled = sort(randsample(fslabeled,nplot));
end
nplotcurr = min(nlabeledcurr,nplot);

readframes = cell(1,nviews);
for viewi = 1:nviews,
  readframes{viewi} = get_readframe_fcn(lbld.movieFilesAll{j,viewi});
end

% plot
figure(hfig);
clf;
hax = createsubplots(nviews,nplotcurr,.01);
hax = reshape(hax,[nviews,nplotcurr]);

for fi = 1:nplotcurr,
  f = fslabeled(fi);
  for viewi = 1:nviews,
    im = readframes{viewi}(f);
    image(im,'Parent',hax(viewi,fi));
    axis(hax(viewi,fi),'image','off');
    hold(hax(viewi,fi),'on');
    for k = 1:npredfns,
      predfn = predfns{k};
      plot(hax(viewi,fi),lbld.(predfn){j}((viewi-1)*nlandmarks+landmarkorder,1,f),lbld.(predfn){j}((viewi-1)*nlandmarks+landmarkorder,2,f),...
        ['-',predmarkers(k)],'Color',predcolors(k,:));
    end
    plot(hax(viewi,fi),lbld.labeledpos{j}((viewi-1)*nlandmarks+landmarkorder,1,f),lbld.labeledpos{j}((viewi-1)*nlandmarks+landmarkorder,2,f),...
      ['-',gtmarker],'Color',gtcolor);    
  end
end

if dosavefigs,
  set(hfig,'Color','w','InvertHardCopy','off');
  SaveFigLotsOfWays(hfig,['TrackingExamples',figsavestr],{'pdf','fig','png'});
end

