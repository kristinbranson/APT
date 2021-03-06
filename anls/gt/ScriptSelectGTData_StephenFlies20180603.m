% set up paths

addpath ../user
addpath /groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect/misc/
addpath /groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect/filehandling/

lbldir = '/groups/huston/hustonlab/flp-chrimson_experiments/APT_projectFiles';
shfliesfile = '/groups/branson/bransonlab/apt/experiments/data/shflies20180518.csv';
rootvideodir = '/groups/huston/hustonlab/flp-chrimson_experiments';
trainingfliesfile = '/groups/branson/bransonlab/apt/experiments/data/training_flyids_20180601.txt';
bodyaxisfliesfile = '/groups/branson/bransonlab/apt/experiments/data/bodyaxis_flyids_20180601.txt';
excludefliesfile = '/groups/branson/bransonlab/apt/experiments/data/exclude_flyids_20180601.txt';
enrichedfliesfile = '/groups/branson/bransonlab/apt/experiments/data/enriched_flyids_20180601.txt';
traindatamatfile = '/groups/branson/bransonlab/apt/experiments/data/trnDataSH_20180503.mat';

%% number of frames

nframesselect = struct;
nframesselect.train_activation = 150;
nframesselect.train_noactivation = 100;
nframesselect.test_activation = 300;
nframesselect.test_noactivation = 200;
nframesselect.enriched_activation = 300;
nframesselect.enriched_noactivation = 0;
nframesselect.intra = 100;
nframesselect.inter = 200;

minnframes_train = 50;
minnframes_video = 1000;
minnframes_select_pervideo = 10;
minframedist = 10;

%% read info from fly file

trainingflyids = importdata(trainingfliesfile);
testflyids = importdata(bodyaxisfliesfile);
excludeflyids = importdata(excludefliesfile);
enrichedflyids = importdata(enrichedfliesfile);
td = load(traindatamatfile,'tMain');


assert(numel(trainingflyids) == numel(unique(trainingflyids)));
assert(numel(testflyids) == numel(unique(testflyids)));
assert(numel(excludeflyids) == numel(unique(excludeflyids)));
assert(isempty(intersect(trainingflyids,testflyids)));

allflyids = setdiff(union(trainingflyids,testflyids),excludeflyids);
enrichedflyids = setdiff(enrichedflyids,excludeflyids);

%% try to find all videos

% find lists of videos by looking for lbl files
flylblfiles = mydir(fullfile(lbldir,'fly*.lbl'));
lblmoviefiles = cell(1,numel(flylblfiles));
lblflyids = nan(1,numel(flylblfiles));
lblmoreinfo = cell(1,numel(flylblfiles));
lblnmovies = nan(numel(flylblfiles),2);

for i = 1:numel(flylblfiles),
  
  [~,n] = fileparts(flylblfiles{i});
  m = regexp(n,'^fly(?<fly>\d+)(?<moreinfo>([^\d].*)?)$','names','once');
  assert(~isempty(m));
  lblflyids(i) = str2double(m.fly);
  lblmoreinfo{i} = m.moreinfo;
  
end

for i = 1:numel(flylblfiles),
  
  ld = load(flylblfiles{i},'-mat','movieFilesAll');
  lblmoviefiles{i} = ld.movieFilesAll;
  lblnmovies(i,:) = size(ld.movieFilesAll);
  
end

fprintf('Ignoring the following flies found by searching for lbl files:\n');
idx = find(~ismember(lblflyids,allflyids));
for i = idx(:)',
  [~,n] = fileparts(flylblfiles{i});
  fprintf('%d (%s)\n',lblflyids(i),n);
end

fprintf('The following flies do not have lbl files, we will have to find their videos with a different method:\n');
fprintf('%d\n',setdiff(allflyids,lblflyids));

% find lists of videos by crawling directory structure
                                                                                  
[allvideofiles,info] = mydir(rootvideodir,'name','\.avi$','recursive',true);
allvideoflies = nan(1,numel(allvideofiles));
allvideoviews = nan(1,numel(allvideofiles));
for i = 1:numel(allvideofiles),
  [~,n] = fileparts(allvideofiles{i});
  m = regexp(allvideofiles{i},'[fF]ly_?(\d+)[^\d]','tokens');
  if isempty(m),
    fprintf('Could not parse fly for video %s\n',allvideofiles{i});
    continue;
  end
  flyidcurr = str2double(m{end});
  fprintf('%s -> fly %d\n',allvideofiles{i},flyidcurr);
  if strcmp(n(1:4),'C001'),
    allvideoviews(i) = 1;
  elseif strcmp(n(1:4),'C002'),
    allvideoviews(i) = 1;
  else
    fprintf('Could not parse view for video %s\n',allvideofiles{i});
    continue;
  end
  allvideoflies(i) = flyidcurr;
end

for i = find(isnan(allvideoflies(:)')),
  fprintf('Could not parse %s, ignoring\n',allvideofiles{i});
end

fprintf('Ignoring the following flies found by crawling directory structure:\n');
fprintf('%d\n',setdiff(allvideoflies(~isnan(allvideoflies)),allflyids));
fprintf('No videos found for the following flies by crawling the directory structure:\n');
fprintf('%d\n',setdiff(allflyids,allvideoflies(~isnan(allvideoflies))));

% find movies using train data mat file

badstrings = {'3ms','APT_projectFiles','[Cc]alib'};

flymoviefiles = cell(1,numel(allflyids));
for i = 1:numel(allflyids),
  
  ilbl = find(allflyids(i)==lblflyids);
  if isempty(ilbl),
    lblnmoviescurr = 0;
    lblvideoscurr = cell(0,2);
    lblvideosexist = zeros(0,2);
  else
    lblvideoscurr = lblmoviefiles{ilbl};
    lblvideoscurr = win2unixpath(lblvideoscurr,'/groups/huston/hustonlab');
    lblvideosexist = cellfun(@exist,lblvideoscurr);
    if ~all(lblvideosexist(:)),
      fprintf('Fly %d, %d / %d lbl file videos do not exist\n',allflyids(i),nnz(lblvideosexist==0),numel(lblvideoscurr));
    end
    lblnmoviescurr = nnz(lblvideosexist);
  end
  idir = find(allvideoflies==allflyids(i));
  
  itrain = find(td.tMain.flyID==allflyids(i));
  if isempty(itrain),
    trainnmoviescurr = 0;
    trainvideoscurr = cell(0,2);
    trainvideosexist = zeros(0,2);
  else
    trainvideoscurr = td.tMain.movFile_read(itrain,:);
    trainvideosexist = cellfun(@exist,trainvideoscurr);
    if ~all(trainvideosexist(:)),
      fprintf('Fly %d, %d / %d train videos do not exist\n',allflyids(i),nnz(trainvideosexist==0),numel(trainvideoscurr));
    end
    trainnmoviescurr = nnz(trainvideosexist);
  end
  allvideoscurr = union(union(lblvideoscurr(:),allvideofiles(idir)),trainvideoscurr(:));
  videoscurr = union(union(lblvideoscurr(lblvideosexist > 0),allvideofiles(idir)),trainvideoscurr(trainvideosexist>0));
  
  isbadvideo = false(size(videoscurr));
  for j = 1:numel(badstrings),
    m = regexp(videoscurr,badstrings{j},'once');
    isbadvideo = isbadvideo | ~cellfun(@isempty,m);
  end
  if nnz(isbadvideo) > 0,
%     fprintf('Removing %d videos with bad strings:\n',nnz(isbadvideo));
%     fprintf('  %s\n',videoscurr{isbadvideo});
    videoscurr(isbadvideo) = [];
  end
  
  if ~isempty(videoscurr),
    
    m = regexp(videoscurr,'^(.*)C(\d+)(H[^/]*)$','tokens','once');
    assert(all(~cellfun(@isempty,m)));
    m = cat(1,m{:});
    viewcurr = str2double(m(:,2));
    tokencurr = cellfun(@(x,y) [x,y], m(:,1),m(:,3),'Uni',0);
    view1idx = find(viewcurr==1);
    view2idx = find(viewcurr==2);
    %assert(nnz(viewcurr==1) == nnz(viewcurr==2));
    token1curr = tokencurr(view1idx);
    token2curr = strrep(tokencurr(view2idx),'C002','C001');
    pairidx = zeros(numel(view1idx),2);
    for j1 = 1:numel(view1idx),
      j2 = find(strcmp(token1curr{j1},token2curr));
      if isempty(j2),
        continue;
      end
      assert(numel(j2)==1);
      pairidx(j1,:) = [view1idx(j1),view2idx(j2)];
    end
    pairidx(any(pairidx==0,2),:) = [];    
    flymoviefiles{i} = videoscurr(pairidx);
  end
  
  %if numel(flymoviefiles{i}) == 0 ||  lblnmoviescurr ~= numel(idir) || numel(idir) ~= numel(flymoviefiles{i}),
    fprintf('%d: %d videos from lblfile, %d videos from directory search, %d videos from train file, %d total videos, %d matched up\n',allflyids(i),lblnmoviescurr,numel(idir),trainnmoviescurr,numel(videoscurr),numel(flymoviefiles{i}));
  %end
  
  ignoredvideoscurr = setdiff(allvideoscurr,flymoviefiles{i});
  if ~isempty(ignoredvideoscurr),
    fprintf('%d: Not including the following videos:\n',allflyids(i));
    fprintf('  %s\n',ignoredvideoscurr{:});
  end
end

% output summary of files considered
fprintf('Fly,N. videos\n');
for i = 1:numel(allflyids),
  fprintf('%d,%d\n',allflyids(i),size(flymoviefiles{i},1));
end

fid = 1;
fprintf(fid,'Fly,Video 1,Video 2\n');
for i = 1:numel(allflyids),
  for j = 1:size(flymoviefiles{i},1),
    fprintf(fid,'%d,%s,%s\n',allflyids(i),flymoviefiles{i}{j,:});
  end
end

nflies = numel(allflyids);
nmoviesperfly = cellfun(@(x) size(x,1),flymoviefiles);

%% count number of frames with training data per video
ntrainframestotal_perfly = zeros(nflies,1);
ntrainframes_perfly = cell(nflies,1);

for i = 1:nflies,
  flyid = allflyids(i);
  idx1curr = td.tMain.flyID == flyid;
  ntrainframestotal_perfly(i) = nnz(idx1curr);
  [~,idx2curr] = ismember(td.tMain.movFile_read(idx1curr,1),flymoviefiles{i}(:,1));
  assert(all(idx2curr>0));
  ntrainframes_perfly{i} = hist(idx2curr,1:size(flymoviefiles{i},1));
  assert(sum(ntrainframes_perfly{i})==ntrainframestotal_perfly(i));
end

%% read number of frames in each video

nframes_perfly = cell(nflies,1);
allmovie1s = cellfun(@(x) x(:,1),flymoviefiles,'Uni',0);
allmovie1s = cat(1,allmovie1s{:});
nframe1s = nan(size(allmovie1s));
parfor i = 1:numel(allmovie1s),
  [~,nframe1s(i)] = get_readframe_fcn(allmovie1s{i});
end

idx = 1;
for i = 1:nflies,
  nframes_perfly{i} = nan(1,nmoviesperfly(i));
  for j = 1:nmoviesperfly(i),
    nframes_perfly{i}(j) = nframe1s(idx);
    idx = idx+1;
  end
end

%% find activation times for each video

stimulus_on_off_perfly = cell(nflies,1);
nframes_activation_perfly = nan(nflies,1);
for i = 1:nflies,
  stimulus_on_off_perfly{i} = flyNum2stimFrames_SJH(allflyids(i));
  nframes_activation_perfly(i) = sum(stimulus_on_off_perfly{i}(:,2)-stimulus_on_off_perfly{i}(:,1));
end


%% which videos are selectable

videook_perfly = cellfun(@(x) x > minnframes_video,nframes_perfly,'Uni',0);
nmoviesperfly = cellfun(@(x) nnz(x), videook_perfly);

%% which flies are training flies

nunlabeledmoviesperfly = nan(nflies,1);
for i = 1:nflies,
  nunlabeledmoviesperfly(i) = nnz(ntrainframes_perfly{i} == 0 & videook_perfly{i});
end

trainflies = find(ntrainframestotal_perfly(:) >= minnframes_train & nmoviesperfly(:) > 0 & nunlabeledmoviesperfly > 0);
testflies = find(ntrainframestotal_perfly(:) == 0 & nmoviesperfly(:) > 0);
enrichedflies = find(ntrainframestotal_perfly(:) == 0 & nmoviesperfly(:) > 0 & ismember(allflyids,enrichedflyids));

%% select training frames

% seed random number generator
rng('default');
rng(1);
rnginfo = struct;
rnginfo.train = rng;

% select one video per fly
nfliesselect = min(numel(trainflies),...
  ceil( (nframesselect.train_activation+nframesselect.train_noactivation)/minnframes_select_pervideo) );

nframesselectperfly_activation = diff(round(linspace(0,nframesselect.train_activation,nfliesselect+1)));
nframesselectperfly_noactivation = diff(round(linspace(0,nframesselect.train_noactivation,nfliesselect+1)));
fliesselect = randsample(trainflies,nfliesselect);

frames2label = table;
frames2label.flyID = zeros(0);
frames2label.movFile = {};
frames2label.frm = zeros(0);
frames2label.type = {};

for i = 1:nfliesselect,
  
  flyi = fliesselect(i);
  flyid = allflyids(flyi);
  movieidx = find(videook_perfly{flyi} & (ntrainframes_perfly{flyi} == 0));
  moviei = randsample(movieidx,1);
  isactivation = false(1,nframes_perfly{flyi}(moviei));
  for j = 1:size(stimulus_on_off_perfly{flyi},1),
    isactivation(stimulus_on_off_perfly{flyi}(j,1):stimulus_on_off_perfly{flyi}(j,2)-1) = true;
  end
  framescurr1 = SelectRandomButSpacedFrames(isactivation,nframesselectperfly_activation(i),minframedist);
  canselect = ~isactivation;
  for j = 1:numel(framescurr1),
    canselect(max(1,framescurr1(j)-minframedist+1):min(numel(canselect),framescurr1(j)+minframedist-1)) = false;
  end
  framescurr2 = SelectRandomButSpacedFrames(canselect,nframesselectperfly_noactivation(i),minframedist);
  
  framescurr = sort([framescurr1,framescurr2]);
  for j = 1:numel(framescurr),
    if isactivation(framescurr(j)),
      labeltype = 'train_activation';
    else
      labeltype = 'train_noactivation';
    end
    frames2label = [frames2label;{flyid,flymoviefiles{flyi}(moviei,:),framescurr(j),labeltype}];
  end
  
end

%% select test frames

% seed random number generator
rng('default');
rng(2);
rnginfo.test = rng;

% select one video per fly
nfliesselect = min(numel(testflies),...
  ceil( (nframesselect.test_activation+nframesselect.test_noactivation)/minnframes_select_pervideo) );

nframesselectperfly_activation = diff(round(linspace(0,nframesselect.test_activation,nfliesselect+1)));
nframesselectperfly_noactivation = diff(round(linspace(0,nframesselect.test_noactivation,nfliesselect+1)));
fliesselect = randsample(testflies,nfliesselect);

for i = 1:nfliesselect,
  
  flyi = fliesselect(i);
  flyid = allflyids(flyi);
  movieidx = find(videook_perfly{flyi});
  moviei = randsample(movieidx,1);
  isactivation = false(1,nframes_perfly{flyi}(moviei));
  for j = 1:size(stimulus_on_off_perfly{flyi},1),
    isactivation(stimulus_on_off_perfly{flyi}(j,1):stimulus_on_off_perfly{flyi}(j,2)-1) = true;
  end
  framescurr1 = SelectRandomButSpacedFrames(isactivation,nframesselectperfly_activation(i),minframedist);
  canselect = ~isactivation;
  for j = 1:numel(framescurr1),
    canselect(max(1,framescurr1(j)-minframedist+1):min(numel(canselect),framescurr1(j)+minframedist-1)) = false;
  end
  framescurr2 = SelectRandomButSpacedFrames(canselect,nframesselectperfly_noactivation(i),minframedist);
  
  framescurr = sort([framescurr1,framescurr2]);
  for j = 1:numel(framescurr),
    if isactivation(framescurr(j)),
      labeltype = 'test_activation';
    else
      labeltype = 'test_noactivation';
    end
    frames2label = [frames2label;{flyid,flymoviefiles{flyi}(moviei,:),framescurr(j),labeltype}];
  end
  
end

%% select enriched frames

% seed random number generator
rng('default');
rng(3);
rnginfo.enriched = rng;

% select one video per fly
nfliesselect = min(numel(enrichedflies),...
  ceil( (nframesselect.enriched_activation+nframesselect.enriched_noactivation)/minnframes_select_pervideo) );

nframesselectperfly_activation = diff(round(linspace(0,nframesselect.enriched_activation,nfliesselect+1)));
nframesselectperfly_noactivation = diff(round(linspace(0,nframesselect.enriched_noactivation,nfliesselect+1)));
fliesselect = randsample(enrichedflies,nfliesselect);

selectedmovies = frames2label.movFile(:,1);

for i = 1:nfliesselect,
  
  flyi = fliesselect(i);
  flyid = allflyids(flyi);
  
  movieidx = find(videook_perfly{flyi} & ~ismember(flymoviefiles{flyi}(:,1)',selectedmovies));
  moviei = randsample(movieidx,1);
  isactivation = false(1,nframes_perfly{flyi}(moviei));
  for j = 1:size(stimulus_on_off_perfly{flyi},1),
    isactivation(stimulus_on_off_perfly{flyi}(j,1):stimulus_on_off_perfly{flyi}(j,2)-1) = true;
  end
  framescurr1 = SelectRandomButSpacedFrames(isactivation,nframesselectperfly_activation(i),minframedist);
  canselect = ~isactivation;
  for j = 1:numel(framescurr1),
    canselect(max(1,framescurr1(j)-minframedist+1):min(numel(canselect),framescurr1(j)+minframedist-1)) = false;
  end
  framescurr2 = SelectRandomButSpacedFrames(canselect,nframesselectperfly_noactivation(i),minframedist);
  
  framescurr = sort([framescurr1,framescurr2]);
  for j = 1:numel(framescurr),
    if isactivation(framescurr(j)),
      labeltype = 'enriched_activation';
    else
      labeltype = 'enriched_noactivation';
    end
    frames2label = [frames2label;{flyid,flymoviefiles{flyi}(moviei,:),framescurr(j),labeltype}];
  end
  
end

%% select frames already selected for intra-annotation 

% seed random number generator
rng('default');
rng(4);
rnginfo.intra = rng;

idxtest = ismember(frames2label.type,{'test_activation','test_noactivation'});
fliesselect_test = unique(frames2label.flyID(idxtest));
canselect = true(1,numel(fliesselect_test));
nframescurr = 0;
for i = 1:numel(fliesselect_test),
  j = randsample(find(canselect),1);
  canselect(j) = false;
  flyid = fliesselect_test(j);
  idxcurr = frames2label.flyID==flyid;
  dupframes2label = frames2label(idxcurr,:);
  for k = 1:size(dupframes2label,1),
    dupframes2label.type{k} = 'intra';
  end
  frames2label = [frames2label;dupframes2label];
  nframescurr = nframescurr + nnz(idxcurr);
  if nframescurr >= nframesselect.intra,
    break;
  end
end

%% select frames already selected for inter-annotation

% seed random number generator
rng('default');
rng(5);
rnginfo.inter = rng;

idxcurr = strcmp(frames2label.type,'intra');
inter_frames2label = frames2label(idxcurr,:);
for i = 1:size(inter_frames2label,1),
  inter_frames2label.type{i} = 'inter';
end

for i = nnz(~canselect)+1:numel(fliesselect_test),
  if nframescurr >= nframesselect.inter,
    break;
  end

  j = randsample(find(canselect),1);
  canselect(j) = false;
  flyid = fliesselect_test(j);
  idxcurr = frames2label.flyID==flyid;
  dupframes2label = frames2label(idxcurr,:);
  for k = 1:size(dupframes2label,1),
    dupframes2label.type{k} = 'inter';
  end
  inter_frames2label = [inter_frames2label;dupframes2label];
  nframescurr = nframescurr + nnz(idxcurr);
end

%% plot some info about frames to label

maxnframes = max(cellfun(@max,nframes_perfly));
isactivation = false(nflies,maxnframes);
for i = 1:nflies,
  
  for j = 1:size(stimulus_on_off_perfly{i},1),
    isactivation(i,stimulus_on_off_perfly{i}(j,1):stimulus_on_off_perfly{i}(j,2)-1) = true;
  end
end

hfig = 101;
figure(hfig);
clf;
% activationim = ones([nflies,maxnframes,3]);
% activationim(:,:,2) = max(.5,double(~isactivation));
% activationim(:,:,3) = max(.5,double(~isactivation));
% image(activationim);
hold on;
set(gca,'Color',[.7,.7,.7]);

for i = 1:nflies,
  
  flyid = allflyids(i);
  nmoviescurr = size(flymoviefiles{i},1);
  
  for j = 1:nmoviescurr,
    
    t = nframes_perfly{i}(j);
      
    y0 = flyid + (j-1)/nmoviescurr;
    y1 = y0+1/nmoviescurr;
    patch([1,1,t,t,1],[y0,y1,y1,y0,y0],[1,1,1],'LineStyle','none');

  end
  
  for j = 1:size(stimulus_on_off_perfly{i},1),
    patch(stimulus_on_off_perfly{i}(j,[1,1,2,2,1]),flyid-.5+[1,0,0,1,1],[1,.5,.5],'LineStyle','none');
  end
  
end

h = nan(1,3);
idxcurr = ismember(frames2label.type,{'train_activation','train_noactivation'});
h(1) = plot(frames2label.frm(idxcurr),frames2label.flyID(idxcurr),'kx');
idxcurr = ismember(frames2label.type,{'test_activation','test_noactivation'});
h(2) = plot(frames2label.frm(idxcurr),frames2label.flyID(idxcurr),'k+');
idxcurr = ismember(frames2label.type,{'enriched_activation','enriched_noactivation'});
h(3) = plot(frames2label.frm(idxcurr),frames2label.flyID(idxcurr),'kd');
idxcurr = ismember(frames2label.type,{'intra'});
h(4) = plot(frames2label.frm(idxcurr),frames2label.flyID(idxcurr),'c+');
h(5) = plot(inter_frames2label.frm,inter_frames2label.flyID,'mo');

set(hfig,'InvertHardCopy','off','Color','w');
set(gca,'XLim',[.5,maxnframes+.5],'YLim',[.5,max(allflyids)+.5]);
xlabel('Frame');
ylabel('Fly ID');
legend(h,{'Training','Test','Enriched','Test + Intra','Test + Inter'},'Location','SouthEast');

%% save to file

tMain = td.tMain;
save /groups/branson/bransonlab/apt/experiments/data/SelectedGTFrames_SJH_20180603.mat frames2label inter_frames2label ...
  rnginfo lbldir rootvideodir ...
  trainingfliesfile bodyaxisfliesfile excludefliesfile traindatamatfile ...
  nframesselect minnframes_train minnframes_video minnframes_select_pervideo minframedist ...
  allflyids trainingflyids testflyids excludeflyids tMain ...
  flymoviefiles nframes_perfly trainflies testflies enrichedflies;

%% 

[videonames,idx] = unique(frames2label.movFile(:,1));

fid = fopen('/groups/branson/bransonlab/apt/experiments/data/selectedfly_videolist_20180601.txt','w');
fprintf(fid,'Fly,Video1\n');
for i = 1:numel(videonames),
  fprintf(fid,'%d,%s\n',frames2label.flyID(idx(i)),videonames{i});
end
fclose(fid);