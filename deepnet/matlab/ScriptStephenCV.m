flynum = zeros(1,numel(lpos));

lbl_count = zeros(1,numel(lpos));
for ndx = 1:numel(lpos)
  [~,~,~,~,mtch] = regexp(movf{ndx,1},'fly_*(\d+)_');
  flynum(ndx) = str2double(mtch{1}{1});
  qq = lpos{ndx};
  nlabels = nnz(any(any(~isnan(qq),1),2));
  lbl_count(ndx) = nlabels;
end

%%

zz = unique(flynum);
nflies = numel(zz);
fly_lbl_count = zeros(1,nflies);
for ndx = 1:numel(zz)
  flymov = flynum == zz(ndx);
  fly_lbl_count(ndx) = sum(lbl_count(flymov));
  
end

%%  create softlinks

for ndx = 1:size(movf,1)
  if lbl_count(ndx) ==0, continue; end
  fparts = strsplit(movf{ndx,1},'/');
  startat = nan;
  for yy = numel(fparts)-2:-1:1
    if regexp(fparts{yy},'^fly_*\d+$')
      startat = yy;
      break;
    end
  end
  if isnan(startat), fprintf('could not find for %d %s\n',ndx,movf{ndx,1}); end
  
  for view = 1:2
    fparts = strsplit(movf{ndx,view},'/');
    newfolder = fullfile('/groups/branson/bransonlab/mayank/stephenCV/',fparts{startat:end-1});
    if ~exist(newfolder,'dir')
      mkdir(newfolder)
    end
    cmd = sprintf('ln -s %s %s',movf{ndx,view},newfolder);
    system(cmd);
  end
end

%%

Q = load('/groups/branson/bransonlab/mayank/PoseTF/headTracking/FlyHeadStephenRound1_Janelia_fixedmovies.lbl','-mat');
movf = Q.movieFilesAll;
lpos = Q.labeledpos;
for fold = 0:4
  basedir = '/groups/branson/bransonlab/mayank/stephenCV/';
  view1f = fullfile(basedir,sprintf('view1vids_fold_%d.txt',fold));
  view2f = fullfile(basedir,sprintf('view2vids_fold_%d.txt',fold));
  f1d = fopen(view1f,'r');
  f2d = fopen(view2f,'r');
  view1list = textscan(f1d,'%s\n');
  view2list = textscan(f2d,'%s\n');
  view1list = view1list{1};
  view2list = view2list{1};
  fclose(f1d);
  fclose(f2d);
  outdir = '/groups/branson/bransonlab/mayank/stephenCV/out';
  resdir = '/groups/branson/bransonlab/mayank/stephenCV/results';

  for ndx = 1:numel(view1list)
    ss = strsplit(view1list{ndx},'/');
    expname1 = [ss{end-5} '__' ss{end-2} '__' ss{end}(end-9:end-6) '_side.mat'];
    matf1 = fullfile(outdir,expname1);
    ss = strsplit(view2list{ndx},'/');
    expname2 = [ss{end-5} '__' ss{end-2} '__' ss{end}(end-9:end-6) '_front.mat'];
    matf2 = fullfile(outdir,expname2);
    trkf1 = [view1list{ndx}(1:end-4) '.trk'];
    trkf2 = [view2list{ndx}(1:end-4) '.trk'];

    if ~exist(matf1,'file') || ~exist(matf2,'file')
      fprintf('%d %d %s doesnt exist\n',fold,ndx,matf1);
      continue;
    end
    if ~exist(trkf1,'file') || ~exist(trkf2,'file')
      fprintf('%d %d %s doesnt exist\n',fold,ndx,trkf1);
      continue;
    end

    Q1 = load(matf1);
    Q2 = load(matf2);
    T1 = load([view1list{ndx}(1:end-4) '.trk'],'-mat');
    T2 = load([view2list{ndx}(1:end-4) '.trk'],'-mat');

    mstr = fullfile(ss{end-2:end});
    lblndx = find(endsWith(movf(:,2),mstr));
    if isempty(lblndx)
      fprintf('No label match found %d %d\n',fold,ndx)
      continue;
    end
    fprintf('%d %d %s\n',fold,ndx,mstr);
    if numel(lblndx)>2
      fprintf('more than 2 matches found %d %d \n',fold,ndx);
    end

    if numel(lblndx)==2 
      pts1 = lpos{lblndx(1)};
      pts2 = lpos{lblndx(2)};
      fr1 = find(~isnan(pts1));
      fr2 = find(~isnan(pts2));
      if ~isempty(fr1) && ~isempty(fr2)
        fprintf('Both matches have valid labels %d %d \n',fold,ndx);
        continue;
      end
      if isempty(fr2)
        lblndx = lblndx(1);
      end
      if isempty(fr1)
        fprintf('No labels for first\n');
        lblndx = lblndx(2);
      end

    end


    curlabels = lpos{lblndx};
    R = struct;
    S = struct;
    S.R{1}.mrf_hmap = Q1.scores;
    R.R{1}.pd_locs = squeeze(Q1.locs(:,:,2,:));
    R.R{1}.mrf_locs = squeeze(Q1.locs(:,:,1,:));
    R.R{1}.final_locs = permute(T1.pTrk,[3,1,2]);
    R.R{1}.labels = permute(lpos{lblndx}(1:5,:,:),[3,1,2]);
    R.R{1}.movie = view1list{ndx};
    S.R{2}.mrf_hmap = Q2.scores;
    R.R{2}.pd_locs = squeeze(Q2.locs(:,:,2,:));
    R.R{2}.mrf_locs = squeeze(Q2.locs(:,:,1,:));
    R.R{2}.final_locs = permute(T2.pTrk,[3,1,2]);
    R.R{2}.labels = permute(lpos{lblndx}(6:10,:,:),[3,1,2]);
    R.R{2}.movie = view2list{ndx};
    outfile = fullfile(resdir,[expname1(1:end-9) '.mat']);
    save(outfile,'-struct','R','-v7.3');
    outfile = fullfile(resdir,[expname1(1:end-9) '_hmap.mat']);
    save(outfile,'-struct','S','-v7.3');
    if size(R.R{2}.labels,1) ~= size(R.R{2}.pd_locs,1)
      fprintf('Frames dont match %d %d\n',fold,ndx)
    end

  end


end


%% add fine resolution results

outDir = '/groups/branson/home/kabram/bransonlab/stephenCV/out';
resDir = '/groups/branson/home/kabram/bransonlab/stephenCV/results';
fineDir = '/groups/branson/home/kabram/bransonlab/stephenCV/fineResults';
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

