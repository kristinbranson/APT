
outDir = '/nrs/branson/mayank/jay/out/';
trkDir = '/nrs/branson/mayank/jay/tracks/';
dd = dir(fullfile(outDir,'*.h5'));

%%
params = singleViewTrackingDefaultParams;
parfor ndx = 1:numel(dd)
  curIn = fullfile(outDir,dd(ndx).name);
  curOut = fullfile(trkDir,dd(ndx).name);
  if ~exist(curOut,'file')
    trackSingleView(curOut,curIn,params);
  end
end



%%

L = load('/groups/branson/bransonlab/mayank/PoseTF/data/jayMouse/miceLabels_20170412.lbl','-mat');

%% 

redo = false;
outDir = '/nrs/branson/mayank/jay/out/';
trkDir = '/nrs/branson/mayank/jay/tracks/';
resDir = '/nrs/branson/mayank/jay/results/';
dd = dir(fullfile(outDir,'*_side.h5'));
movf = L.movieFilesAll(:,1);

lpos = L.labeledpos;
for ndx = 1:numel(dd)
  ename = dd(ndx).name(1:end-8);
  curIn1 = fullfile(outDir,[ename '_side.h5']);
  curOut1 = fullfile(trkDir,[ename '_side.h5']);
  if ~exist(curIn1,'file') || ~exist(curOut1,'file')
    fprintf('Input files dont exist for %d\n',ndx);
    continue;
  end
  
  curIn2 = fullfile(outDir,[ename '_front.h5']);
  curOut2 = fullfile(trkDir,[ename '_front.h5']);
  if ~exist(curIn2,'file') || ~exist(curOut2,'file')
    fprintf('Input files dont exist for %d\n',ndx);
    continue;
  end
  
  outfile = fullfile(resDir,[ename '.h5']);
  if exist(outfile,'file') && ~redo
    continue;
  end

  Q1 = struct;
  Q1.scores = permute(h5read(curIn1,'/scores'),[5,4,3,2,1]);
  Q1.locs = permute(h5read(curIn1,'/locs'),[4,3,2,1]);
  Q1.expdir = permute(h5read(curIn1,'/expname'),[4,3,2,1]);
  Q2 = struct;
  Q2.scores = permute(h5read(curIn2,'/scores'),[5,4,3,2,1]);
  Q2.locs = permute(h5read(curIn2,'/locs'),[4,3,2,1]);
  Q2.expdir = permute(h5read(curIn2,'/expname'),[4,3,2,1]);
  
  
%   Q1 = load(curIn);
  T1 = load(curOut1,'-mat');
  T2 = load(curOut2,'-mat');

  lblndx = find(strcmp(movf,Q1.expdir));
  if isempty(lblndx)
    fprintf('No label match found %d %d\n',fold,ndx)
    continue;
  end
  if numel(lblndx)>2
    fprintf('more than 2 matches found %d %d \n',fold,ndx);
  end

  if numel(lblndx)==2 
    pts1 = lpos{lblndx(1)};
    pts2 = lpos{lblndx(2)};
    fr1 = find(~isnan(pts1));
    fr2 = find(~isnan(pts2));
    if numel(fr1) > numel(fr2)
      lblndx = lblndx(1);
    else
      fprintf('Chosing the second label index for %d!!\n',ndx);
      lblndx = lblndx(2);
    end
  end
  
  curlabels = lpos{lblndx};
  R = struct;
  S = struct;
  S.R{1}.mrf_hmap = Q1.scores(:,:,:,:,1);
  R.R{1}.pd_locs = permute(Q1.locs(:,:,2,:),[1,2,4,3]);
  R.R{1}.mrf_locs = permute(Q1.locs(:,:,1,:),[1,2,4,3]);
  R.R{1}.final_locs = permute(T1.pTrk,[2,1,3]);
  R.R{1}.labels = permute(lpos{lblndx}(1,:,:),[3,1,2]);
  R.R{1}.movie = movf{lblndx};
  S.R{2}.mrf_hmap = Q2.scores(:,:,:,:,1);
  R.R{2}.pd_locs = permute(Q2.locs(:,:,2,:),[1,2,4,3]);
  R.R{2}.mrf_locs = permute(Q2.locs(:,:,1,:),[1,2,4,3]);
  R.R{2}.final_locs = permute(T2.pTrk,[2,1,3]);
  R.R{2}.labels = permute(lpos{lblndx}(2,:,:),[3,1,2]);
  R.R{2}.movie = movf{lblndx};
  
  npts = size(Q1.locs,2);
  T = size(Q1.locs,1);
  mrfs = nan(T,npts);
  fins = nan(T,npts);
  for tm = 1:T
    for curp = 1:npts
      xloc = max(1,round(Q1.locs(tm,curp,1,1)/4));
      yloc = max(1,round(Q1.locs(tm,curp,1,2)/4));
      mrfs(tm,curp) = Q1.scores(tm,yloc,xloc,curp,1);
      xloc = max(1,round(T1.pTrk(curp,tm,1)/4));
      yloc = max(1,round(T1.pTrk(curp,tm,2)/4));
      fins(tm,curp) = Q1.scores(tm,yloc,xloc,curp,1);
    end
  end
  R.R{1}.mrf_scores = mrfs;
  R.R{1}.final_scores = fins;
  
  mrfs = nan(T,npts);
  fins = nan(T,npts);
  for tm = 1:T
    for curp = 1:npts
      xloc = max(1,round(Q2.locs(tm,curp,1,1)/4));
      yloc = max(1,round(Q2.locs(tm,curp,1,2)/4));
      mrfs(tm,curp) = Q2.scores(tm,yloc,xloc,curp);
      xloc = max(1,round(T2.pTrk(curp,tm,1)/4));
      yloc = max(1,round(T2.pTrk(curp,tm,2)/4));
      fins(tm,curp) = Q2.scores(tm,yloc,xloc,curp);
    end
  end
  R.R{2}.mrf_scores = mrfs;
  R.R{2}.final_scores = fins;
  
  outfile = fullfile(resDir,[ename '.mat']);
  save(outfile,'-struct','R','-v7.3');
  outfile = fullfile(resDir,[ename '_hmap.mat']);
  save(outfile,'-struct','S','-v7.3');
  if size(R.R{1}.labels,1) ~= size(R.R{1}.pd_locs,1)
    fprintf('Frames dont match %d %d\n',fold,ndx)
  end

  if mod(ndx,20)==0, fprintf('.'); end
  
end

fprintf('\n');


%% test the tracking results

L = load('/groups/branson/bransonlab/mayank/PoseTF/data/jayMouse/miceLabels_20170412.lbl','-mat');

%% 

% M122_20140707_v002 62 is totally screwed.

outDir = '/nrs/branson/mayank/jayOut/';
trkDir = '/nrs/branson/mayank/jayTracks/';
resDir = '/nrs/branson/mayank/jayResults/';
dd = dir(fullfile(outDir,'*_side.h5'));
movf = L.movieFilesAll(:,1);
lpos = L.labeledpos;

f = figure(1);
for ndx = 2:20:numel(dd)
  ename = dd(ndx).name(1:end-8);
  curIn1 = fullfile(outDir,[ename '_side.h5']);
  curOut1 = fullfile(trkDir,[ename '_side.mat']);
  if ~exist(curIn1,'file') || ~exist(curOut1,'file')
    fprintf('Input files dont exist for %d\n',ndx);
    continue;
  end
  
  curIn2 = fullfile(outDir,[ename '_front.h5']);
  curOut2 = fullfile(trkDir,[ename '_front.mat']);
  if ~exist(curIn2,'file') || ~exist(curOut2,'file')
    fprintf('Input files dont exist for %d\n',ndx);
    continue;
  end
  
  outfile = fullfile(resDir,[ename '.mat']);
  T = load(outfile);

  [rfn,nframes,fid,hinfo] = get_readframe_fcn(T.R{1}.movie);
  
  for cframe = 1:30:nframes
    figure(f);
    imshow(rfn(cframe));
    hold on;
    scatter(T.R{1}.mrf_locs(cframe,:,1),T.R{1}.mrf_locs(cframe,:,2),500,'.');
    scatter(T.R{2}.mrf_locs(cframe,:,1),T.R{2}.mrf_locs(cframe,:,2),500,'.');
    hold off;
    pause(0.2);
  end
  
  if fid>0,fclose(fid); end
  
end



%% View the fine results

outDir = '/nrs/branson/mayank/jayOut/';
dd = dir(fullfile(outDir,'*_side_fine.h5'));

f = figure(1);
for ndx = 2:20:numel(dd)
  ename = dd(ndx).name(1:end-13);
  curIn1 = fullfile(outDir,[ename '_side_fine.h5']);
  if ~exist(curIn1,'file')
    fprintf('Input files dont exist for %d\n',ndx);
    continue;
  end
  
  curIn2 = fullfile(outDir,[ename '_front_fine.h5']);
  if ~exist(curIn2,'file') 
    fprintf('Input files dont exist for %d\n',ndx);
    continue;
  end
  
  gg = h5read(curIn1,'/locs');
  if ndims(gg)==4
    locs1 = permute(h5read(curIn1,'/locs'),[4,3,2,1]);
    locs2 = permute(h5read(curIn2,'/locs'),[4,3,2,1]);
  else
    locs1 = permute(h5read(curIn1,'/locs'),[3,2,1]);
    locs2 = permute(h5read(curIn2,'/locs'),[3,2,1]);
  end
  
  mname = h5read(curIn2,'/expname');
  [rfn,nframes,fid,hinfo] = get_readframe_fcn(mname);
  
  for cframe = 1:30:nframes
    figure(f);
    imshow(rfn(cframe));
    hold on;
    scatter(locs1(cframe,:,1),locs1(cframe,:,2),500,'.');
    scatter(locs2(cframe,:,1),locs2(cframe,:,2),500,'.');
    hold off;
    pause(0.2);
  end
  
  if fid>0,fclose(fid); end
  
end

%% add fine resolution results

outDir = '/nrs/branson/mayank/jay/out/';
resDir = '/nrs/branson/mayank/jay/results/';
fineDir = '/nrs/branson/mayank/jay/fineResults/';
dd = dir(fullfile(resDir,'*.mat'));
for ndx = 1:numel(dd)
  curInRes = fullfile(resDir,dd(ndx).name);
  curInFine1 = fullfile(outDir,[dd(ndx).name(1:end-4) '_side_fine.h5']);
  curInFine2 = fullfile(outDir,[dd(ndx).name(1:end-4) '_front_fine.h5']);
  curOut = fullfile(fineDir,dd(ndx).name);
  
  if ~exist(curInFine1,'file') || ~exist(curInFine2,'file') || ~exist(curInRes,'file') 
    fprintf('Input files dont exist for %d\n',ndx);
    continue;
  end
  
  Q = load(curInRes);
  f_locs1 = permute(h5read(curInFine1,'/locs'),[3,2,1]);
  f_locs2 = permute(h5read(curInFine2,'/locs'),[3,2,1]);
  Q.R{1}.fine_locs = f_locs1;
  Q.R{2}.fine_locs = f_locs2;
  
  save(curOut,'-struct','Q','-v7.3');
  if mod(ndx,20)==0, fprintf('.'); end
  
end

fprintf('\n');

