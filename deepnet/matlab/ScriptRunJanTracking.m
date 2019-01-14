
outDir = '/nrs/branson/mayank/janOut/';
trkDir = '/nrs/branson/mayank/janTracks/';
dd = dir(fullfile(outDir,'*.mat'));

%%
params = singleViewTrackingDefaultParams;
parfor ndx = 1:numel(dd)
  curIn = fullfile(outDir,dd(ndx).name);
  curOut = fullfile(trkDir,dd(ndx).name);
  if ~exist(curOut,'file')
    trackSingleView(curOut,curIn,params);
  end
end

%% move files S to S0001 naming format

dd = dir(fullfile(outDir,'*.mat'));

doneNames = {};
ismatch = false(1,numel(dd));
for ndx = 1:numel(dd)
  curIn = fullfile(outDir,dd(ndx).name);
  curOut = fullfile(trkDir,dd(ndx).name);
  I = matfile(curIn);
  expname = I.expname;
  ss = strsplit(expname,'/');
  newname = ss{end}(1:end-4);
  newIn = fullfile(outDir,[newname '.mat']);
  newOut = fullfile(trkDir,[newname '.mat']);
  if any(strcmp(doneNames,newname))
    fprintf('%d %s Name already exists\n',ndx,newname);
    continue;
  end
  
  if ~strcmp(dd(ndx).name(1:end-4),newname)
    movefile(curIn,newIn);
    movefile(curOut,newOut);
  else
    ismatch(ndx) = true;
  end    
  doneNames{end+1} = newname;
  
  if mod(ndx,20)==0; fprintf('.'); end
  if mod(ndx,500)==0; fprintf('\n'); end
end




%%

L = load('/groups/branson/home/kabram/bransonlab/PoseTF/janLegTracking/160819_Dhyey_2_al_fixed.lbl','-mat');

%%
outDir = '/nrs/branson/mayank/janOut/';
trkDir = '/nrs/branson/mayank/janTracks/';
resDir = '/nrs/branson/mayank/janResults/';
dd = dir(fullfile(outDir,'*.mat'));
movf = L.movieFilesAll;

lpos = L.labeledpos;
for ndx = 1:numel(dd)
  curIn = fullfile(outDir,dd(ndx).name);
  curOut = fullfile(trkDir,dd(ndx).name);
  if ~exist(curIn,'file') || ~exist(curOut,'file')
    fprintf('Input files dont exist for %d\n',ndx);
    continue;
  end
  
  outfile = fullfile(resDir,[dd(ndx).name]);
  if exist(outfile,'file'),
    continue;
  end

  
  Q1 = load(curIn);
  T1 = load(curOut);
  mstr = [dd(ndx).name(1:end-3) 'avi'];

  lblndx = find(endsWith(movf,mstr));
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
  S.R{1}.mrf_hmap = Q1.scores;
  R.R{1}.pd_locs = squeeze(Q1.locs(:,:,2,:));
  R.R{1}.mrf_locs = squeeze(Q1.locs(:,:,1,:));
  R.R{1}.final_locs = permute(T1.pTrk,[2,1,3]);
  R.R{1}.labels = permute(lpos{lblndx}(4:7,:,:),[3,1,2]);
  R.R{1}.movie = movf{lblndx};
  
  npts = size(Q1.locs,2);
  T = size(Q1.locs,1);
  mrfs = nan(T,npts);
  fins = nan(T,npts);
  for tm = 1:T
    for curp = 1:npts
      xloc = round(Q1.locs(tm,curp,1,1)/4);
      yloc = round(Q1.locs(tm,curp,1,2)/4);
      mrfs(tm,curp) = Q1.scores(tm,yloc,xloc,curp);
      xloc = round(T1.pTrk(curp,tm,1)/4);
      yloc = round(T1.pTrk(curp,tm,2)/4);
      fins(tm,curp) = Q1.scores(tm,yloc,xloc,curp);
    end
  end
  R.R{1}.mrf_scores = mrfs;
  R.R{1}.final_scores = fins;
  outfile = fullfile(resDir,[dd(ndx).name]);
  save(outfile,'-struct','R','-v7.3');
  outfile = fullfile(resDir,[dd(ndx).name(1:end-4) '_hmap.mat']);
  save(outfile,'-struct','S','-v7.3');
  if size(R.R{1}.labels,1) ~= size(R.R{1}.pd_locs,1)
    fprintf('Frames dont match %d %d\n',fold,ndx)
  end

  if mod(ndx,20)==0, fprintf('.'); end
  
end

fprintf('\n');


%%

L = load('/groups/branson/home/kabram/bransonlab/PoseTF/janLegTracking/160819_Dhyey_2_al_fixed.lbl','-mat');

%%  view the results

outDir = '/nrs/branson/mayank/janOut/';
trkDir = '/nrs/branson/mayank/janTracks/';
resDir = '/nrs/branson/mayank/janResults/';
dd = dir(fullfile(outDir,'*.mat'));
movf = L.movieFilesAll;

lpos = L.labeledpos;

f = figure(1);
for ndx = 2:20:numel(dd)
  outfile = fullfile(resDir,[dd(ndx).name]);

  T = load(outfile);

  [rfn,nframes,fid,hinfo] = get_readframe_fcn(T.R{1}.movie);
  colors = jet(size(T.R{1}.final_locs,2));
  for cframe = 1:30:nframes
    figure(f);
    imshow(rfn(cframe));
    hold on;
    scatter(T.R{1}.final_locs(cframe,:,1),T.R{1}.final_locs(cframe,:,2),50,colors,'.');
    hold off;
    pause(0.2);
  end
  
  if fid>0,fclose(fid); end
  
end

%% add fine resolution results

outDir = '/nrs/branson/mayank/janOut/';
resDir = '/nrs/branson/mayank/janResults/';
fineDir = '/nrs/branson/mayank/janFineResults/';
dd = dir(fullfile(resDir,'*.mat'));
for ndx = 1:numel(dd)
  curInRes = fullfile(resDir,dd(ndx).name);
  curInFine = fullfile(outDir,[dd(ndx).name(1:end-4) '_fine.h5']);
  curOut = fullfile(fineDir,dd(ndx).name);
  
  if ~exist(curInFine,'file') || ~exist(curInRes,'file') 
    fprintf('Input files dont exist for %d\n',ndx);
    continue;
  end
  
  Q = load(curInRes);
  f_locs = permute(h5read(curInFine,'/locs'),[3,2,1]);
  Q.R{1}.fine_locs = f_locs;
  
  save(curOut,'-struct','Q','-v7.3');
  if mod(ndx,20)==0, fprintf('.'); end
  
end

fprintf('\n');

