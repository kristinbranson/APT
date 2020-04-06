% set up paths
addpath ..;
addpath ../misc;
addpath /groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect/misc/
addpath /groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect/filehandling/

labeleddatadirs = {
   '/tier2/hantman/Adam/Kristin/M118'
   '/tier2/hantman/Adam/Kristin/M119'
   '/tier2/hantman/Adam/Kristin/M122'
   '/tier2/hantman/Adam/Kristin/M127'
   '/tier2/hantman/Adam/Kristin/M130'
%  '/tier2/hantman/Adam/Kristin/M134/Final Labeller'
  '/groups/branson/home/bransonk/tracking/code/rcpr/data/M174LabeledTrackingData20150423'
  '/groups/branson/home/bransonk/tracking/code/rcpr/data/M173LabeledTrackingData20150531'
  '/tier2/hantman/Adam/Kristin/M147/Final Labeller'
};

winrootdir = 'Y:\';
linuxrootdir = '/tier2/hantman/';

%fixeddatadir = '/groups/branson/home/bransonk/tracking/code/rcpr/data/TrackingResults20150329/corrected';
%fixeddatafiles = {
%  'TrackingResults_M134C3VGATXChR2_20150205L_20150330.mat'
%  };
fixeddatadir = '';
fixeddatafiles = {};
nfixedvideos = inf;
autodatadir = '/groups/branson/home/bransonk/tracking/code/rcpr/data/TrackingResults20150329';


% labeleddatadirs = {
%   '/tier2/hantman/Adam/Kristin/M135/Final Labeller'
%   };

labeleddatares = {
  '.*a.*\.mat'
  '.*L.*\.mat'
  };
  
rootdir = '/tier2/hantman';

savedir = '/groups/branson/home/bransonk/tracking/code/rcpr/data';
savefile = fullfile(savedir,'M118_M119_M122_M127_M130_M173_M174_M147_20150615.mat');
%savefile = fullfile(savedir,'M135labeleddata.mat');
moviefilestr = 'movie_comb.avi';

%% put all the data in one mat file

inmatfiles = {};
for i = 1:numel(labeleddatadirs),
  for j = 1:numel(labeleddatares),
    inmatfilescurr = mydir(labeleddatadirs{i},'name',labeleddatares{j},'isdir',0);
    fprintf('Found %d training files in %s with re %s\n',numel(inmatfilescurr),labeleddatadirs{i},labeleddatares{j});
    inmatfiles = [inmatfiles,inmatfilescurr]; %#ok<AGROW>
  end
end

expdirs = {};
npts = [];

labels = struct;
labels.pts = [];
labels.ts = [];
labels.expidx = [];

for i = 1:numel(inmatfiles),
  
  d = load(inmatfiles{i});
  idxlabeledcurr = find(cellfun(@(x) nnz(~isnan(x)),d.labeledpos_perexp) > 0);
  if isempty(idxlabeledcurr),
    warning('No videos labeled in %s',inmatfiles{i});
    continue;
  end
  expdirscurr = d.expdirs(idxlabeledcurr);
%   if ismember('Y:\Jay\videos\M173VGATXChR2\20150415L\CTR\M173_20150415_v002',expdirscurr),
%     keyboard;
%   end
  assert(numel(unique(expdirscurr)) == numel(expdirscurr));
  assert(isempty(intersect(expdirscurr,expdirs)));
  expidxcurr = numel(expdirs)+1:numel(expdirs)+numel(expdirscurr);
  expdirs = [expdirs,expdirscurr];
  %expdirs = union(expdirs,expdirscurr);
  %[~,expidxcurr] = ismember(expdirscurr,expdirs);
  % x position for point p, frame t, video j is d.labeled_pos{j}(p,1,t)
  % y position for point p, frame t, video j is d.labeled_pos{j}(p,2,t)
  if isempty(npts),
    
    npts = size(d.labeledpos_perexp{idxlabeledcurr(1)},1);
    labels.pts = nan(npts,2,0);
    
  end

  for jj = 1:numel(idxlabeledcurr),
    j = idxlabeledcurr(jj);
    idxcurr = find(~any(any(isnan(d.labeledpos_perexp{j}),1),2));
    labels.pts = cat(3,labels.pts,d.labeledpos_perexp{j}(:,:,idxcurr));
    labels.ts = cat(2,labels.ts,idxcurr');
    labels.expidx = cat(2,labels.expidx,repmat(expidxcurr(jj),[1,numel(idxcurr)]));
  end
  
end

labels.expdirs = cell(1,numel(expdirs));

for i = 1:numel(expdirs),
  if ~isempty(regexp(expdirs{i},'^[A-Z]:\\','once')),
    labels.expdirs{i} = fullfile(rootdir,strrep(expdirs{i}(4:end),'\','/'));
  else
    labels.expdirs{i} = expdirs{i};
  end
end

%% load in fixed data

expi = 1;
fixedlabels = struct;
fixedlabels.expdirs = {};
fixedlabels.pts = zeros(2,2,0);
fixedlabels.auto_pts = zeros(2,2,0);
fixedlabels.ts = zeros(1,0);
fixedlabels.expidx = zeros(1,0);
fixedlabels.err = zeros(1,0);
if ~isempty(fixeddatafiles),
  inmatfiles = mydir(fullfile(fixeddatadir,'*.mat'));
else
  inmatfiles = cellfun(@(x) fullfile(fixeddatadir,x),fixeddatafiles,'Uni',0);
end
humanerr = [];
autoerr_in = [];
autoerr_out = [];
expierr = [];
terr = [];
for i = 1:numel(inmatfiles),
  
  fd = load(inmatfiles{i});
  [~,n] = myfileparts(inmatfiles{i});
  automatfile = fullfile(autodatadir,n);
  if ~exist(automatfile,'file'),
    warning('Could not find auto tracking results file %s, skipping\n',automatfile);
    continue;
  end
  ad = load(automatfile);
  
  if numel(nfixedvideos) >= i,
    nmoviescurr = min(numel(fd.moviefiles_all),nfixedvideos(i));
  else
    nmoviescurr = numel(fd.moviefiles_all);
  end
  
  for j = 1:nmoviescurr,
    
    moviefile = strrep(fd.moviefiles_all{j},winrootdir,linuxrootdir);
    moviefile = strrep(moviefile,'\','/');
    
    fixedlabels.expdirs{expi} = fileparts(moviefile);
    k = find(strcmp(ad.moviefiles_all,moviefile));
    assert(numel(k)==1);
    assert(j==k);
    [n,D] = size(ad.p_all{k});    
    
    autoerrcurr = sqrt(sum( (ad.p_all{k}(1:n,:)-fd.p_all{j}(1:n,:)).^2, 2));
    isfixed = autoerrcurr > 0;
    tsfixed = find(isfixed);
    
    if ~any(isfixed),
      continue;
    end
    
    npts = numel(tsfixed);
    ptscurr = reshape(fd.p_all{j}(isfixed,:)',[2,2,npts]);
    autoptscurr = reshape(ad.p_all{k}(isfixed,:)',[2,2,npts]);

    fixedlabels.pts(:,:,end+1:end+npts) = ptscurr;
    fixedlabels.ts(end+1:end+npts) = tsfixed;
    fixedlabels.expidx(end+1:end+npts) = expi;
    fixedlabels.err(end+1:end+npts) = autoerrcurr(isfixed);
    fixedlabels.auto_pts(:,:,end+1:end+npts) = autoptscurr;
    
    k = find(strcmp(fixedlabels.expdirs{expi},labels.expdirs));
    if isempty(k),
      
      autoerr_out = [autoerr_out;autoerrcurr];
      
    else
      
      expidx = find(labels.expidx==k);
      tslabeled = labels.ts(expidx);
      autoerr_in = [autoerr_in;autoerrcurr(tslabeled)];
      pcurr = reshape(labels.pts(:,:,expidx),[D,numel(expidx)])';
      humanerrcurr = sqrt(sum( (pcurr-fd.p_all{j}(tslabeled,:)).^2, 2));
      humanerr = [humanerr;humanerrcurr];
      
    end
    
    expi = expi + 1;
    
  end
  
end

if ~isempty(humanerr),
  edges = logspace(-3,log10(max([max(humanerr),max(autoerr_in),max(autoerr_out)])),101);
  centers = (edges(1:end-1)+edges(2:end))/2;
  humancounts = hist(humanerr,centers);
  autoincounts = hist(autoerr_in,centers);
  autooutcounts = hist(autoerr_out,centers);
  autoincounts_changed = hist(autoerr_in(autoerr_in>0),centers);
  autooutcounts_changed = hist(autoerr_out(autoerr_out>0),centers);
  humancounts_changed = hist(humanerr(autoerr_in>0),centers);
  
  clf;
  % plot(centers,[humancounts/sum(humancounts)
  %   autoincounts/sum(autoincounts)
  %   autooutcounts/sum(autooutcounts)
  %   humancounts_changed/sum(humancounts_changed)
  %   autoincounts_changed/sum(autoincounts_changed)
  %   autooutcounts_changed/sum(autooutcounts_changed)]','.-');
  plot(centers,...
    [humancounts_changed/sum(humancounts_changed)
    autoincounts_changed/sum(autoincounts_changed)
    autooutcounts_changed/sum(autooutcounts_changed)]','.-');
  set(gca,'XScale','log');
  ylim = get(gca,'YLim');
  set(gca,'XLim',[edges(1),edges(end)],'YLim',[-.01,ylim(2)]);
  %legend('Human','Tracker in, all','Tracker out, all','Tracker in, changed','Tracker out, changed');
  legend('Human','Tracker (in)','Tracker (out)');
  xlabel('Error');
  ylabel('Fraction of labeled points');
  
  fprintf('Human error: ');
  prctiles_compute = [50,75,95,99,100];
  for i = 1:numel(prctiles_compute),
    
    fprintf('%dth percentile = %f, ',prctiles_compute(i),prctile(humanerr(autoerr_in>0),prctiles_compute(i)));
    
  end
  fprintf('\n');
  
end
%%

labels.moviefilestr = moviefilestr;
fixedlabels.moviefilestr = moviefilestr;

save(savefile,'-struct','labels');
if ~isempty(fixedlabels.expdirs),
  save('-append',savefile,'fixedlabels');
end
