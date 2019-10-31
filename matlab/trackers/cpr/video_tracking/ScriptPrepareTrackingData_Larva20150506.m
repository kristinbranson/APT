% set up paths
addpath ..;
addpath ../misc;
addpath /groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect/misc/
addpath /groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect/filehandling/

inmatfiles = {'/groups/branson/bransonlab/projects/LarvalMuscles/15-04-29/94A04_D/labels/labelslarva_frames1.mat'};

savedir = '/groups/branson/home/bransonk/tracking/code/rcpr/data';
savefile = fullfile(savedir,'LarvaMuscles_94A04D_Frames1_20150429_20150506.mat');
%savefile = fullfile(savedir,'M135labeleddata.mat');
moviefilestr = 'Frames_1_00001.tif';

%% put all the data in one mat file

labels = struct;
labels.pts = [];
labels.ts = [];
labels.expidx = [];
labels.flies = [];
labels.expdirs = {};

for i = 1:numel(inmatfiles),

  expdir = fileparts(fileparts(inmatfiles{i}));
  labels.expdirs{i} = expdir;
  d = load(inmatfiles{i});
  
  [npts,ndims,nframes,nflies] = size(d.labeledpos);
  [ts,flies] = find(reshape(all(all(~isnan(d.labeledpos),1),2),[nframes,nflies]));
  if isempty(ts),
    warning('No videos labeled in %s',inmatfiles{i});
    continue;
  end
  if isempty(labels.pts),
    labels.pts = nan(npts,2,0);
  end

  for j = 1:numel(ts),
    t = ts(j);
    fly = flies(j);
    labels.pts(:,:,end+1) = d.labeledpos(:,:,t,fly);
    labels.ts(end+1) = t;
    labels.flies(end+1) = fly;
    labels.expidx(end+1) = i;
  end
  
end

%% save

labels.moviefilestr = moviefilestr;
save(savefile,'-struct','labels');
