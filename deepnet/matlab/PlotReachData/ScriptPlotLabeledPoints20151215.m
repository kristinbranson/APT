%% set up paths

JAABAcodedir = '/groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect';
labelfiles = {
  'Y:\Jay\videos\M186hm4dBPN\labeler\labels.lbl'
  'Y:\Jay\videos\M187hm4dBPN\labeler\labels.lbl'};
rootdatadir = '/tier2/hantman';

addpath ..;
addpath ../misc;
addpath(fullfile(JAABAcodedir,'filehandling'));
addpath(fullfile(JAABAcodedir,'misc'));

if isunix,
  labelfiles = win2unixpath(labelfiles,rootdatadir);
end
mousenames = regexp(labelfiles,'[/:\\](M[^/\\]+)[/\\]','tokens','once');
mousenames = [mousenames{:}];
nmice = numel(labelfiles);

%% load in labeled data

labeldata = [];
for i = 1:nmice,
  labeldata = structappend(labeldata,load(labelfiles{i},'-mat'));
  if isunix,
    labeldata(i).movieFilesAll = win2unixpath(labeldata(i).movieFilesAll,rootdatadir);
  end
  % extract date, etc. from exp name
  labeldata(i).expNames = cell(size(labeldata(i).movieInfoAll));
  labeldata(i).expInfo = [];
  for j = 1:numel(labeldata(i).movieInfoAll),
    [~,labeldata(i).expNames{j}] = fileparts(labeldata(i).movieInfoAll{j}.info.Path);
    info = parseExpName(labeldata(i).expNames{j});
    assert(~isempty(info),'Could not parse experiment name %s',labeldata(i).expNames{j});
    labeldata(i).expInfo = structappend(labeldata(i).expInfo,info);
  end
  
end

%% plot all labeled points

figure;
clf;
hax = createsubplots(nmice,1,.05);
days = cell(1,nmice);
muperday = cell(1,nmice);
Sperday = cell(1,nmice);
nperday = cell(1,nmice);
for mousei = 1:nmice,
  [days{mousei},muperday{mousei},Sperday{mousei},nperday{mousei}] = PlotLabeledReachesPerMouse(labeldata(mousei),'hax',hax(mousei),'mousename',mousenames{mousei});
end

%% plot outliers

noutliersplotperday = 3;
for mousei = 1:nmice,
  hfig = figure;
  set(hfig,'Units','pixels','Position',[10,10,800,500],'Name',sprintf('Outliers per day for mouse %s',mousenames{mousei}));
  clf;
  hax = createsubplots(numel(days{mousei}),noutliersplotperday+1,.05);
  hax = reshape(hax,[numel(days{mousei}),noutliersplotperday+1]);
  PlotLabeledReachOutliersPerMouse(labeldata(mousei),days{mousei},noutliersplotperday,...
    'hax',hax,'mousename',mousenames{mousei},...
    'muperday',muperday{mousei},'Sperday',Sperday{mousei});
end

%% plot variability in x1 and x2

hfig = figure;
hax = createsubplots(nmice,1,.05);
for mousei = 1:nmice,
  PlotLabeledReachPosXStdPerMouse(labeldata(mousei),Sperday{mousei},days{mousei},'hax',hax(mousei),...
    'mousename',mousenames{mousei});
end