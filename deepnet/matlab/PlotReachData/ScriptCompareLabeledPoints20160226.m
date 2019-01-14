%% set up paths

JAABAcodedir = '/groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect';
labelfiles = {
  '/tier2/hantman/Jay/videos/M186hm4dBPN/labeler/M186hm4dBPN_manualEndpoint1stCNO.lbl'
  '/tier2/hantman/Jay/videos/M186hm4dBPN/labeler/M186hm4dBPN_manualEndpoint2ndCNO.lbl'
  '/tier2/hantman/Jay/videos/M186hm4dBPN/labeler/M186hm4dBPN_manualEndpoint3rdCNO.lbl'
  '/tier2/hantman/Jay/videos/M186hm4dBPN/labeler/M186hm4dBPN_manualEndpoint4thCNO.lbl'
  '/tier2/hantman/Jay/videos/M186hm4dBPN/labeler/M186hm4dBPN_manualEndpoint5thCNO.lbl'
  '/tier2/hantman/Jay/videos/M187hm4dBPN/labeler/M187hm4dBPN_manualEndpoint1stCNO.lbl'
  '/tier2/hantman/Jay/videos/M187hm4dBPN/labeler/M187hm4dBPN_manualEndpoint2ndCNO.lbl'
  '/tier2/hantman/Jay/videos/M187hm4dBPN/labeler/M187hm4dBPN_manualEndpoint3rdCNO.lbl'
  '/tier2/hantman/Jay/videos/M187hm4dBPN/labeler/M187hm4dBPN_manualEndpoint4thCNO.lbl'
  '/tier2/hantman/Jay/videos/M187hm4dBPN/labeler/M187hm4dBPN_manualEndpoint5thCNO.lbl'
  };
rootdatadir = '/tier2/hantman';

CTRL = 1;
CNO = 2;
WASH = 3;
ntrialtypes = 3;
trialtypenames = {'Ctrl','CNO','Wash'};

addpath ..;
addpath ../misc;
addpath(fullfile(JAABAcodedir,'filehandling'));
addpath(fullfile(JAABAcodedir,'misc'));

if isunix && ~isempty(rootdatadir),
  labelfiles = win2unixpath(labelfiles,rootdatadir);
end

nlabelfiles = numel(labelfiles);

%% load in labeled data

labeldata = [];
fnscopy = {
  'movieFilesAll'
  'trxFilesAll'
  'labeledpos'
  'expNames'
  'expInfo'
  };
for i = 1:nlabelfiles,
  fprintf('Loading %s...\n',labelfiles{i});
  newlabeldata = load(labelfiles{i},'-mat');
  if isunix && ~isempty(rootdatadir),
    newlabeldata.movieFilesAll = win2unixpath(newlabeldata.movieFilesAll,rootdatadir);
  end
  
  % extract date, etc. from exp name
  newlabeldata.expNames = cell(size(newlabeldata.movieInfoAll));
  newlabeldata.expInfo = [];
  for j = 1:numel(newlabeldata.movieInfoAll),
    [~,newlabeldata.expNames{j}] = fileparts(newlabeldata.movieInfoAll{j}.info.Path);
    info = parseExpName(newlabeldata.expNames{j});
    assert(~isempty(info),'Could not parse experiment name %s',newlabeldata.expNames{j});
    newlabeldata.expInfo = structappend(newlabeldata.expInfo,info,1);
  end
  
  [micecurr,~,mouseidxcurr] = unique({newlabeldata.expInfo.mouse});
  for j = 1:numel(micecurr),
    fprintf('  %s: %d exps\n',micecurr{j},nnz(mouseidxcurr==j));
  end
    
  if isempty(labeldata),
    labeldata = newlabeldata;
  else
    
    % check for duplicates
    ism = ismember(newlabeldata.movieFilesAll,labeldata.movieFilesAll);
    assert(~any(ism));
    
    for j = 1:numel(fnscopy),
      labeldata.(fnscopy{j}) = [labeldata.(fnscopy{j});newlabeldata.(fnscopy{j})];
      if j > 1,
        assert(numel(labeldata.(fnscopy{j})) == numel(labeldata.(fnscopy{j-1})));
      end
    end
  end
 
end

[mice,~,mouseidx] = unique({labeldata.expInfo.mouse});
nmice = numel(mice);

labeldata0 = labeldata;

labeldata = repmat(labeldata0,[1,nmice]);
for mousei = 1:nmice,
  idxcurr = mouseidx == mousei;
  for j = 1:numel(fnscopy),
    labeldata(mousei).(fnscopy{j}) = labeldata0.(fnscopy{j})(idxcurr);
  end
end

% get trial type
for mousei = 1:nmice,
  
  % look at subdirectory name, see if it is before cno, has cno, or after
  % cno (wash)
  trialtype = regexp(labeldata(mousei).movieFilesAll,'/(?<cnonum>\d+)[^\d/]*/(?<datestr>\d{8})(?<cno>(CNO)?)/','once','names');
  assert(~any(cellfun(@isempty,trialtype)));
  trialtype = [trialtype{:}];
  [cnonums,~,dayidx] = unique({trialtype.cnonum});
  cnonums = str2double(cnonums);
  iscno = ~cellfun(@isempty,{trialtype.cno});
  [labeldata(mousei).expInfo.trialtype] = deal(0);
  [labeldata(mousei).expInfo(iscno).trialtype] = deal(CNO);
  datenum = [labeldata(mousei).expInfo.datenum];
  for datei = 1:numel(cnonums),
    idxcurr = dayidx == datei;
    if ~any(idxcurr&iscno),
      mintrialcno = inf;
      maxtrialcno = inf;
    else
      cnodatenums = [labeldata(mousei).expInfo(idxcurr&iscno).datenum];
      mindatenumcno = min(cnodatenums);
      maxdatenumcno = max(cnodatenums);
    end
    [labeldata(mousei).expInfo(idxcurr & datenum < mindatenumcno).trialtype] = deal(CTRL);
    [labeldata(mousei).expInfo(idxcurr & datenum > maxdatenumcno).trialtype] = deal(WASH);
  
    fprintf('%s, CNO %d:\n',mice{mousei},cnonums(datei));
    for j = 1:ntrialtypes,
      fprintf('  %s: %d exps\n',trialtypenames{j},nnz([labeldata(mousei).expInfo(idxcurr).trialtype]==j));
    end
    
    assert(all([labeldata(mousei).expInfo(idxcurr).trialtype]));
    
  end
  
end

% M186, CNO 1:
%   Ctrl: 30 exps
%   CNO: 30 exps
%   Wash: 30 exps
% M186, CNO 2:
%   Ctrl: 30 exps
%   CNO: 30 exps
%   Wash: 30 exps
% M186, CNO 3:
%   Ctrl: 30 exps
%   CNO: 30 exps
%   Wash: 30 exps
% M186, CNO 4:
%   Ctrl: 30 exps
%   CNO: 30 exps
%   Wash: 25 exps
% M186, CNO 5:
%   Ctrl: 30 exps
%   CNO: 30 exps
%   Wash: 30 exps
% M187, CNO 1:
%   Ctrl: 30 exps
%   CNO: 30 exps
%   Wash: 30 exps
% M187, CNO 2:
%   Ctrl: 30 exps
%   CNO: 30 exps
%   Wash: 30 exps
% M187, CNO 3:
%   Ctrl: 30 exps
%   CNO: 30 exps
%   Wash: 30 exps
% M187, CNO 4:
%   Ctrl: 30 exps
%   CNO: 30 exps
%   Wash: 30 exps
% M187, CNO 5:
%   Ctrl: 30 exps
%   CNO: 30 exps
%   Wash: 30 exps

%% plot all labeled points

days = cell(ntrialtypes,nmice);
muperday = cell(ntrialtypes,nmice);
Sperday = cell(ntrialtypes,nmice);
nperday = cell(ntrialtypes,nmice);
hfigs = nan(1,nmice);
for mousei = 1:nmice,  
  
  hfigs(mousei) = figure;
  set(hfigs(mousei),'Position',[10,10 1200 1000]);
  hax = createsubplots(ntrialtypes,1,.05);
  
  for trialtypei = 1:ntrialtypes,
    idxcurr = [labeldata(mousei).expInfo.trialtype] == trialtypei;
    labeldatacurr = labeldata(mousei);
    for j = 1:numel(fnscopy),
      labeldatacurr.(fnscopy{j}) = labeldata(mousei).(fnscopy{j})(idxcurr);
    end
    [days{trialtypei,mousei},muperday{trialtypei,mousei},Sperday{trialtypei,mousei},nperday{trialtypei,mousei}] = ...
      PlotLabeledReachesPerMouse(labeldatacurr,'hax',hax(trialtypei),...
      'mousename',sprintf('%s, %s',mice{mousei},trialtypenames{trialtypei}));
  end
  
  SaveFigLotsOfWays(hfigs(mousei),sprintf('GrabPositions_%s',mice{mousei}),{'pdf','png','fig'});
  
end


%% plot outliers

noutliersplotperday = 3;
for mousei = 1:nmice,
  for trialtypei = 1:ntrialtypes,

    idxcurr = [labeldata(mousei).expInfo.trialtype] == trialtypei;
    labeldatacurr = labeldata(mousei);
    for j = 1:numel(fnscopy),
      labeldatacurr.(fnscopy{j}) = labeldata(mousei).(fnscopy{j})(idxcurr);
    end

    
    hfig = figure;
    set(hfig,'Units','pixels','Position',[10,10,800,500],'Name',sprintf('Outliers per day for mouse %s, %s',mice{mousei},trialtypenames{trialtypei}));
    clf;
    hax = createsubplots(numel(days{trialtypei,mousei}),noutliersplotperday+1,.05);
    hax = reshape(hax,[numel(days{trialtypei,mousei}),noutliersplotperday+1]);
    PlotLabeledReachOutliersPerMouse(labeldatacurr,days{trialtypei,mousei},noutliersplotperday,...
      'hax',hax,'mousename',sprintf('%s, %s',mice{mousei},trialtypenames{trialtypei}),...
      'muperday',muperday{trialtypei,mousei},'Sperday',Sperday{trialtypei,mousei});
    
    SaveFigLotsOfWays(hfigs(mousei),sprintf('OutlierGrabPositions_%s_%s',mice{mousei},trialtypenames{trialtypei}),{'pdf','png','fig'});
    
  end
end

%% plot variability in x1 and x2

hfig = figure;
set(hfig,'Units','pixels','Position',[10 10 1600 900]);
hax = createsubplots(nmice,ntrialtypes,.05);
hax = reshape(hax,[nmice,ntrialtypes]);
for mousei = 1:nmice,
  for trialtypei = 1:ntrialtypes,
    
    idxcurr = [labeldata(mousei).expInfo.trialtype] == trialtypei;
    labeldatacurr = labeldata(mousei);
    for j = 1:numel(fnscopy),
      labeldatacurr.(fnscopy{j}) = labeldata(mousei).(fnscopy{j})(idxcurr);
    end

    PlotLabeledReachPosXStdPerMouse(labeldatacurr,Sperday{trialtypei,mousei},days{trialtypei,mousei},...
      'hax',hax(mousei,trialtypei),...
      'mousename',sprintf('%s, %s',mice{mousei},trialtypenames{trialtypei}));
  end
end

SaveFigLotsOfWays(hfig,'XPositionStd',{'pdf','png','fig'});
