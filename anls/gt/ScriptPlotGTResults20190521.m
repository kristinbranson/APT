%%


% this should be run from the APT directory

%gtfile = '/nrs/branson/mayank/apt_cache/alice_view0_time.mat';

allexptypes = {'FlyBubble','SHView0','SHView1','SH3D','RFView0','RFView1','RF3D','Roian','BSView0x','BSView1x','BSView2x','Larva','FlyBubbleMDNvsDLC'};
APT.setpath;
addpath anls/gt;

%savedir = '/groups/branson/bransonlab/apt/experiments/res/gt/20190523';
savedir = '/groups/branson/bransonlab/apt/experiments/res/gt/20200928';


if ~exist(savedir,'dir'),
  mkdir(savedir);
end
dosavefig = true;
exptypei = 1;
forcecompute = false;
%%
for exptypei = 1:numel(allexptypes),
exptype = allexptypes{exptypei};

%% parameters
resmatfile = fullfile(savedir,sprintf('PlotGTResults_%s.mat',exptype));

ScriptGTDataSetParameters;

%% load in data

if ~forcecompute && exist(resmatfile,'file'),
  load(resmatfile);
else
  
gtdata = GTLoad([gtfile_final(:);gtfile_trainsize(:)],true);

if isempty(gtfile_trainsize),
  gtdata_size = [];
else
  gtdata_size = GTLoad(gtfile_trainsize,false);
end
if isempty(gtfile_traintime),
  gtdata_time = [];
else
  gtdata_time = GTLoad(gtfile_traintime,false);
end
if ~isempty(annoterrfile),
  annoterrdata = load(annoterrfile);
end

net1 = find(ismember(nets,fieldnames(gtdata)),1);
npts = size(gtdata.(nets{net1}){end}.labels,2);
nlabels = size(gtdata.(nets{net1}){end}.labels,1);

%% plotting information

if ismember(exptype,{'BSView0x','BSView1x','BSView2x','FlyBubble','FlyBubbleMDNvsDLC','SH3D','RF3D','Roian'}),
  
  lObj = Labeler.stcLoadLblFile(lblfile);
  % try
  %   lObj = load(lblfile,'-mat');
  % catch
  %   tmploc = tempname;
  %   mkdir(tmploc);
  %   untar(lblfile,tmploc);
  %   lObj = load(fullfile(tmploc,'label_file.lbl'),'-mat');
  %   unix(sprintf('rm -r %s',tmploc));
  % end
  ptcolors = lObj.cfg.LabelPointsPlot.Colors;
  lObj.labeledpos = cellfun(@SparseLabelArray.full,lObj.labeledpos,'uni',0);
  
else
  lObj = StartAPT;
  lObj.projLoad(lblfile);
  ptcolors = lObj.LabelPointColors;
end

%% images for overlaying percentile errors 

if ~is3d,

% nets = {'openpose','leap','deeplabcut','unet','mdn'};
% nnets = numel(nets);
% minerr = 0;
% maxerr = 50;
% nbins = 100;
% binedges = linspace(minerr,maxerr,nbins+1);
% bincenters = (binedges(1:end-1)+binedges(2:end))/2;
% markers = {'o','s','d','^'};
%binedges(end) = inf;

switch exptype,
  case {'FlyBubble','FlyBubbleMDNvsDLC','Roian'},
    if isstruct(lObj),
      freezeInfo = lObj.cfg.PrevAxes.ModeInfo;
    else
      freezeInfo = lObj.prevAxesModeTargetSpec;
    end
    lpos = lObj.labeledpos{freezeInfo.iMov}(:,:,freezeInfo.frm,freezeInfo.iTgt);
    if freezeInfo.isrotated,
      lpos = [lpos,ones(size(lpos,1),1)]*freezeInfo.A;
      lpos = lpos(:,1:2);
    end
  case {'SHView0','SHView1','Larva','RFView0','RFView1'},
    lObj.setMFT(freezeInfo.iMov,freezeInfo.frm,freezeInfo.iTgt);
    vwi = str2double(exptype(end))+1;
    if isnan(vwi),
      vwi = 1;
    end
    freezeInfo.im = get(lObj.gdata.images_all(vwi),'CData');
    freezeInfo.xdata = get(lObj.gdata.images_all(vwi),'XData');
    freezeInfo.ydata = get(lObj.gdata.images_all(vwi),'YData');
    freezeInfo.isrotated = false;
    
    freezeInfo.axes_curr.XDir = get(lObj.gdata.axes_all(vwi),'XDir');
    freezeInfo.axes_curr.YDir = get(lObj.gdata.axes_all(vwi),'YDir');
    if ~isfield(freezeInfo.axes_curr,'XLim'),
      freezeInfo.axes_curr.XLim = get(lObj.gdata.axes_all(vwi),'XLim');
    end
    if ~isfield(freezeInfo.axes_curr,'YLim'),
      freezeInfo.axes_curr.YLim = get(lObj.gdata.axes_all(vwi),'YLim');
    end
    freezeInfo.axes_curr.CameraViewAngleMode = 'auto';
    npts = size(gtdata_size.mdn{1}.labels,2);
    lpos = lObj.labeledpos{freezeInfo.iMov}((vwi-1)*npts+(1:npts),:,freezeInfo.frm,freezeInfo.iTgt);
    
    delete(lObj.gdata.figs_all);
    clear lObj;
    lObj = Labeler.stcLoadLblFile(lblfile);

  case {'BSView0x','BSView1x','BSView2x'},
%     if ~isfield(freezeInfo,'i'),
%       i = find(gtimdata.ppdata.MD.frm == freezeInfo.frm & gtimdata.ppdata.MD.mov == freezeInfo.iMov & ...
%         gtimdata.ppdata.MD.iTgt == freezeInfo.iTgt);
%     end
    freezeInfo.im = gtimdata.ppdata.I{freezeInfo.i};
    freezeInfo.xdata = [1,size(freezeInfo.im,2)];
    freezeInfo.ydata = [1,size(freezeInfo.im,1)];
    freezeInfo.isrotated = false;
    freezeInfo.axes_curr.XDir = 'normal';
    freezeInfo.axes_curr.YDir = 'reverse';
    npts = size(gtdata_size.mdn{1}.labels,2);
    lpos = reshape(gtimdata.ppdata.pGT(freezeInfo.i,:),[npts,2]);
  otherwise
    error('Unknown exptype %s',exptype);
    
end

assert(all(~isnan(lpos(:))));

else
  lpos = [];
  
end

%% load in cpr data


if ~isempty(gtfile_cpr),
  gtdata = AddCPRGTData(gtdata,gtfile_cpr,lObj.labeledpos,vwi);
end

if ~isempty(gtfile_trainsize) && ~isempty(gtfile_trainsize_cpr),  
  gtdata_size = AddCPRGTData(gtdata_size,gtfile_trainsize_cpr,lObj.labeledpos,vwi);
end

if ~isempty(gtfile_traintime) && ~isempty(gtfile_traintime_cpr),
  gtdata_time = AddCPRGTData(gtdata_time,gtfile_trainsize_cpr,lObj.labeledpos,vwi);
end

%% save to results file

% which is the main gtdata file

% if ~isempty(gtdata_time),
%   gtdata = gtdata_time;
%   fns = setdiff(fieldnames(gtdata_size),fieldnames(gtdata_time));
%   for i = 1:numel(fns),
%     gtdata.(fns{i}) = gtdata_size.(fns{i});
%   end
% else
%   gtdata = gtdata_size;
% end
% fns = fieldnames(gtdata);
% for i = 1:numel(fns),
%   gtdata.(fns{i}) = gtdata.(fns{i})(end);
% end

save(resmatfile,'gtdata_size','gtdata_time','gtdata','lpos','freezeInfo','annoterrdata','ptcolors','gtimdata','conddata','labeltypes','datatypes','pttypes','lObj');

end

isnet = ismember(nets,fieldnames(gtdata));
if ~all(isnet),
  fprintf('No data available for the following networks:\n');
  fprintf('%s\n',legendnames{~isnet});
end
nets(~isnet) = []; %#ok<SAGROW>
colors(~isnet,:) = []; %#ok<SAGROW>
legendnames(~isnet) = []; %#ok<SAGROW>
nnets = numel(nets);

missingnets = setdiff(fieldnames(gtdata),nets);
if ~isempty(missingnets),
  fprintf('The following algorithms have results but will not be plotted:\n');
  fprintf('%s\n',missingnets{:});
end

%% compute kappa for OKS computation if there is annotation error data

if ~isempty(annoterrdata),
  dormoutliers = size(annoterrdata.intra{end}.pred,1) > 200;
  kappadistname = 'gamma2';
  
  ComputeOKSPrecisionTable(gtdata,...
    'nets',nets,'legendnames',legendnames,...
    'exptype',exptype,...
    'conddata',conddata,...
    'labeltypes',labeltypes,'datatypes',datatypes,...
    'savedir',savedir,...
    'dosavefig',dosavefig,...
    'pttypes',pttypes,...
    'annoterrdata',annoterrdata,...
    'kappadistname',kappadistname,...
    'dormoutliers',dormoutliers);

  kappadistname = 'gaussian';
  
  ComputeOKSPrecisionTable(gtdata,...
    'nets',nets,'legendnames',legendnames,...
    'exptype',exptype,...
    'conddata',conddata,...
    'labeltypes',labeltypes,'datatypes',datatypes,...
    'savedir',savedir,...
    'dosavefig',dosavefig,...
    'pttypes',pttypes,...
    'annoterrdata',annoterrdata,...
    'kappadistname',kappadistname,...
    'dormoutliers',dormoutliers);
  
end

%% compute average precision at various thresholds relative to the animal scale

threshmethod = 'prctileerr';
[pixelPrecisionTbls,~,APthresh] = ComputePixelPrecisionTable(gtdata,...
  'nets',nets,'legendnames',legendnames,...
  'exptype',exptype,...
  'conddata',conddata,...
  'labeltypes',labeltypes,'datatypes',datatypes,...
  'savedir',savedir,...
  'dosavefig',dosavefig,...
  'pttypes',pttypes,...
  'annoterrdata',annoterrdata,...
  'threshmethod',threshmethod);

if dosavefig,
  save('-append',resmatfile,'APthresh','pixelPrecisionTbls');
end

%% plot error percentiles per part type over training time

if doplotoverx && ~isempty(gtdata_time),

PlotPerLandmarkErrorPrctilesOverX('gtdata',gtdata_time,...
  'nets',nets,'legendnames',legendnames,...
  'colors',colors,...
  'exptype',exptype,...
  'conddata',conddata,...
  'labeltypes',labeltypes,'datatypes',datatypes,...
  'prcs',prcs,...
  'pttypes',pttypes,...
  'savedir',savedir,...
  'maxerr',maxerr,...
  'dosavefig',dosavefig,...
  'x','Training time (h)',...
  'savekey','TrainTime');
end


%% plot error percentiles for worst part over training time

if doplotoverx && ~isempty(gtdata_time),

for stati = 1,
  switch stati,
    case 1
      statname = 'Worst';
    case 2
      statname = 'Median';
    case 3
      statname = 'Best';
  end

  hfigs = PlotWorstLandmarkErrorOverX('gtdata',gtdata_time,...
    'statname',statname,...
    'nets',nets,'legendnames',legendnames,...
    'colors',colors,...
    'exptype',exptype,...
    'conddata',conddata,...
    'labeltypes',labeltypes,'datatypes',datatypes,...
    'prcs',prcs,...
    'savedir',savedir,...
    'maxerr',maxerr,...
    'dosavefig',dosavefig,...
    'x','Training time (h)',...
    'savekey','TrainTime');
  
end

end

%% plot ap over time

if doplotoverx && ~isempty(gtdata_time),

  hfig = PlotAPOverX('gtdata',gtdata_time,...
    'nets',nets,'legendnames',legendnames,...
    'colors',colors,...
    'exptype',exptype,...
    'conddata',conddata,...
    'labeltypes',labeltypes,'datatypes',datatypes,...
    'savedir',savedir,...
    'dosavefig',dosavefig,...
    'x','Training time (h)',...
    'savekey','TrainTime',...
    'pttypes',pttypes,...
    'annoterrdata',annoterrdata,...
    'kvals',APthresh,...
    'logscaley',true,...
    'LineWidth',1);
  
end


nlabeltypes = size(labeltypes,1);
ndatatypes = size(datatypes,1);

if doplotoverx && ~isempty(gtdata_time),
  
  
  if nlabeltypes > 1,
    hfig = PlotAPOverX_CompareConditions('gtdata',gtdata_time,...
      'nets',nets,'legendnames',legendnames,...
      'exptype',exptype,...
      'conddata',conddata,...
      'labeltypes',labeltypes,'datatypes',datatypes,...
      'savedir',savedir,...
      'dosavefig',dosavefig,...
      'x','Training time (h)',...
      'savekey','TrainTime',...
      'pttypes',pttypes,...
      'annoterrdata',annoterrdata,...
      'kvals',APthresh,...
      'logscaley',true,...
      'labelallonly',false,...
      'dataallonly',true);
  end
  
  if ndatatypes > 1,
    hfig = PlotAPOverX_CompareConditions('gtdata',gtdata_time,...
      'nets',nets,'legendnames',legendnames,...
      'exptype',exptype,...
      'conddata',conddata,...
      'labeltypes',labeltypes,'datatypes',datatypes,...
      'savedir',savedir,...
      'dosavefig',dosavefig,...
      'x','Training time (h)',...
      'savekey','TrainTime',...
      'pttypes',pttypes,...
      'annoterrdata',annoterrdata,...
      'kvals',APthresh,...
      'logscaley',true,...
      'labelallonly',true,...
      'dataallonly',false);
  end
  
end


%% plot error percentiles per part type over training set size

if doplotoverx && ~isempty(gtdata_size),
  PlotPerLandmarkErrorPrctilesOverX('gtdata',gtdata_size,...
    'nets',nets,'legendnames',legendnames,...
    'colors',colors,...
    'exptype',exptype,...
    'conddata',conddata,...
    'labeltypes',labeltypes,'datatypes',datatypes,...
    'prcs',prcs,...
    'pttypes',pttypes,...
    'savedir',savedir,...
    'maxerr',maxerr,...
    'dosavefig',dosavefig,...
    'x','N. training examples',...
    'savekey','TrainSetSize');
  
end

%% plot error percentiles for worst part over training set size

if doplotoverx && ~isempty(gtdata_size),
  
  for stati = 1,
    switch stati,
      case 1
        statname = 'Worst';
      case 2
        statname = 'Median';
      case 3
        statname = 'Best';
    end
    
    hfigs = PlotWorstLandmarkErrorOverX('gtdata',gtdata_size,...
      'statname',statname,...
      'nets',nets,'legendnames',legendnames,...
      'colors',colors,...
      'exptype',exptype,...
      'conddata',conddata,...
      'labeltypes',labeltypes,'datatypes',datatypes,...
      'prcs',prcs,...
      'savedir',savedir,...
      'maxerr',[],...
      'dosavefig',dosavefig,...
      'x','N. training examples',...
      'savekey','TrainSetSize');
    
    
%     if ~isempty(annoterrdata),
%       worsterr = max(sqrt(sum((annoterrdata.inter{end}.pred-annoterrdata.inter{end}.labels).^2,3)),[],2);
%       medianterworsterr = median(worsterr);
%     end
  
  end
  
end

%% plot AWP over training set size

threshmethod = 'prctileerr';
if doplotoverx && ~isempty(gtdata_size),

  hfig = PlotAPOverX('gtdata',gtdata_size,...
    'nets',nets,'legendnames',legendnames,...
    'colors',colors,...
    'exptype',exptype,...
    'conddata',conddata,...
    'labeltypes',labeltypes,'datatypes',datatypes,...
    'savedir',savedir,...
    'dosavefig',dosavefig,...
    'x','N. training examples',...
    'savekey','TrainSetSize',...
    'pttypes',pttypes,...
    'annoterrdata',annoterrdata,...
    'kvals',APthresh,...
    'logscaley',true,...
    'LineWidth',1);
  
end

%% plot AWP over training set size for different conditions

nlabeltypes = size(labeltypes,1);
ndatatypes = size(datatypes,1);

if doplotoverx && ~isempty(gtdata_size),
  
  
  if nlabeltypes > 1,
    hfig = PlotAPOverX_CompareConditions('gtdata',gtdata_size,...
      'nets',nets,'legendnames',legendnames,...
      'exptype',exptype,...
      'conddata',conddata,...
      'labeltypes',labeltypes,'datatypes',datatypes,...
      'savedir',savedir,...
      'dosavefig',dosavefig,...
      'x','N. training examples',...
      'savekey','TrainSetSize',...
      'pttypes',pttypes,...
      'annoterrdata',annoterrdata,...
      'kvals',APthresh,...
      'logscaley',true,...
      'labelallonly',false,...
      'dataallonly',true);
  end
  
  if ndatatypes > 1,
    hfig = PlotAPOverX_CompareConditions('gtdata',gtdata_size,...
      'nets',nets,'legendnames',legendnames,...
      'exptype',exptype,...
      'conddata',conddata,...
      'labeltypes',labeltypes,'datatypes',datatypes,...
      'savedir',savedir,...
      'dosavefig',dosavefig,...
      'x','N. training examples',...
      'savekey','TrainSetSize',...
      'pttypes',pttypes,...
      'annoterrdata',annoterrdata,...
      'kvals',APthresh,...
      'logscaley',true,...
      'labelallonly',true,...
      'dataallonly',false);
  end
  
end


%% overlay error percentiles per part for last entry

if ~is3d,

PlotOverlayedErrorPrctiles('freezeInfo',freezeInfo,...
  'lpos',lpos,...
  'gtdata',gtdata,...
  'nets',nets,'legendnames',legendnames,...
  'exptype',exptype,...
  'conddata',conddata,...
  'labeltypes',labeltypes,'datatypes',datatypes,...
  'prcs',prcs,...
  'savedir',savedir,...
  'dosavefig',dosavefig);

end
%% plot error percentiles per part type for last entry

PlotPerLandmarkErrorPrctiles('gtdata',gtdata,...
  'nets',nets,'legendnames',legendnames,...
  'colors',colors,...
  'exptype',exptype,...
  'conddata',conddata,...
  'labeltypes',labeltypes,'datatypes',datatypes,...
  'prcs',prcs,...
  'pttypes',pttypes,...
  'savedir',savedir,...
  'maxerr',maxerr,...
  'dosavefig',dosavefig);
  
%% plot error vs number of inliers, prctile vs error for worst, median, and best landmark

clear hfigs;

if ~isempty(annoterrdata),
  if isfield(annoterrdata,'inter'),
    cur_annoterrdata = annoterrdata.inter{end};
  else
    cur_annoterrdata = annoterrdata.intra{end};
  end
else
  cur_annoterrdata = [];
end

for stati = 1,
  switch stati,
    case 1
      statname = 'Worst';
    case 2
      statname = 'Median';
    case 3
      statname = 'Best';
  end
  
  
  PlotFracInliers('gtdata',gtdata,...
    'nets',nets,'legendnames',legendnames,...
    'colors',colors,...
    'exptype',exptype,...
    'conddata',conddata,...
    'labeltypes',labeltypes,'datatypes',datatypes,...
    'statname',statname,...
    'savedir',savedir,'dosavefig',dosavefig,...
    'maxerr',maxerr,...
    'prcs',prcs,...
    'maxprc',99.5,...
    'annoterrdata',annoterrdata,...
    'annoterrprc',[],...
    'plotannfracinliers',true,...
    'APthresh',APthresh);
  
  PlotFracInliers('gtdata',gtdata,...
    'nets',nets,'legendnames',legendnames,...
    'colors',colors,...
    'exptype',exptype,...
    'conddata',conddata,...
    'labeltypes',labeltypes,'datatypes',datatypes,...
    'statname',statname,...
    'savedir',savedir,'dosavefig',dosavefig,...
    'maxerr',maxerr,...
    'prcs',prcs,...
    'maxprc',99.5,...
    'annoterrdata',annoterrdata,...
    'annoterrprc',[],...
    'plotannfracinliers',true,...
    'APthresh',APthresh,...
    'xscale','log');

  
  if ~isempty(pttypes),
    npttypes = size(pttypes,1);
    pttypecolors = zeros(npttypes,3);
    for pti = 1:npttypes,
      pttypecolors(pti,:) = ptcolors(max(pttypes{pti,2}),:);
    end
  else
    pttypecolors = ptcolors;
  end
  for ndx = 1:nnets,
    PlotFracInliersPerLandmark('gtdata',gtdata,...
      'net',nets{ndx},'pttypes',pttypes,...
      'colors',pttypecolors,...
      'exptype',exptype,...
      'conddata',conddata,...
      'labeltypes',labeltypes,'datatypes',datatypes,...
      'statname',statname,...
      'savedir',savedir,'dosavefig',dosavefig,...
      'maxerr',maxerr,...
      'prcs',prcs,...
      'maxprc',99.5,...
      'plotannfracinliers',true,...
      'APthresh',APthresh);
  end
  
  PlotSortedWorstLandmarkError('gtdata',gtdata,...
    'nets',nets,'legendnames',legendnames,...
    'colors',colors,...
    'exptype',exptype,...
    'conddata',conddata,...
    'labeltypes',labeltypes,'datatypes',datatypes,...
    'statname',statname,...
    'savedir',savedir,'dosavefig',dosavefig,'maxerr',maxerr,...
    'prcs',prcs,...
    'maxprc',99.5,...
    'annoterrdata',cur_annoterrdata,...
    'annoterrprc',99);

  
end

%% plot error threshold vs precision for 
ndatatypes = size(datatypes,1);
nlabeltypes = size(labeltypes,1);

if nlabeltypes > 1
  for ndx = 1:nnets,
    PlotFracInliersPerGroup('gtdata',gtdata,...
      'net',nets{ndx},'pttypes',pttypes,...
      'exptype',exptype,...
      'conddata',conddata,...
      'labeltypes',labeltypes,'datatypes',datatypes,...
      'statname',statname,...
      'savedir',savedir,'dosavefig',dosavefig,...
      'maxerr',maxerr,...
      'prcs',prcs,...
      'maxprc',99.5,...
      'APthresh',APthresh,...
      'dataallonly',true);
  end
end

if ndatatypes > 1,
  for ndx = 1:nnets,

    PlotFracInliersPerGroup('gtdata',gtdata,...
      'net',nets{ndx},'pttypes',pttypes,...
      'exptype',exptype,...
      'conddata',conddata,...
      'labeltypes',labeltypes,'datatypes',datatypes,...
      'statname',statname,...
      'savedir',savedir,'dosavefig',dosavefig,...
      'maxerr',maxerr,...
      'prcs',prcs,...
      'maxprc',99.5,...
      'APthresh',APthresh,...
      'labelallonly',true);
    
  end
end

if ~isempty(annoterrdata) && nlabeltypes > 1,

PlotFracInliersPerGroup('gtdata',annoterrdata,...
  'net','intra','pttypes',pttypes,...
  'exptype',exptype,...
  'conddata',annoterrdata.intra{end},...
  'labeltypes',labeltypes,'datatypes',datatypes,...
  'statname',statname,...
  'savedir',savedir,'dosavefig',dosavefig,...
  'maxerr',maxerr,...
  'prcs',prcs,...
  'maxprc',99.5,...
  'APthresh',APthresh,...
  'dataallonly',true);

end

%% plot example predictions

if ~is3d,
  nexamples_random = 5;
  nexamples_disagree = 5;
  errnets = {'mdn_joint_fpn','deeplabcut'};
  hfigs = PlotExamplePredictions('gtdata',gtdata,...
    'gtimdata',gtimdata,...
    'lObj',lObj,...
    'nets',nets,'legendnames',legendnames,...
    'exptype',exptype,...
    'ptcolors',ptcolors,...
    'conddata',conddata,'labeltypes',labeltypes,'datatypes',datatypes,...
    'dosavefig',dosavefig,'savedir',savedir,...
    'nexamples_random',nexamples_random,...
    'nexamples_disagree',nexamples_disagree,...
    'doAlignCoordSystem',doAlignCoordSystem,...
    'errnets',errnets,...
    'MarkerSize',6,...
    'LineWidth',1.5);
end
%%
close all;

end
%% plot errors against each other for MDNvsDLC

if ismember(exptype,{'FlyBubbleMDNvsDLC'}),
  err = nan(size(gtdata.(nets{1}){end}.labels,1),nnets);
  for ndx = 1:nnets,
    err(:,ndx) = max(sqrt(sum((gtdata.(nets{ndx}){end}.labels-gtdata.(nets{ndx}){end}.pred).^2,3)),[],2);
  end
  hfig = figure;
  clf;
  maxerrcurr = max(err(:))*1.05;
  hold on;
  plot([0,maxerrcurr],[0,maxerrcurr],'c-');
  plot(err(:,1),err(:,2),'k.','MarkerFaceColor','k');
  axis equal;
  set(gca,'XLim',[0,maxerrcurr],'YLim',[0,maxerrcurr]);
  xlabel(sprintf('%s worst landmark error',legendnames{1}));
  ylabel(sprintf('%s worst landmark error',legendnames{2}));
  set(hfig,'Renderer','painters','Units','pixels','Position',[10,10,300,300]);
  saveas(hfig,fullfile(savedir,sprintf('%s_DLCErrorVsMDNError.svg',exptype)),'svg');
  
  preddist = max(sqrt(sum((gtdata.(nets{1}){end}.pred-gtdata.(nets{2}){end}.pred).^2,3)),[],2);
  maxerrcurr = max(max(err(:)),preddist(:))*1.05;
  hfig = figure;
  clf;
  hax = createsubplots(1,nnets,.05);
  for ndx = 1:nnets,
    axes(hax(ndx));
    hold on;
    plot([0,maxerrcurr],[0,maxerrcurr],'c-');
    plot(err(:,ndx),preddist,'k.','MarkerFaceColor','k');
    axis equal;
    set(gca,'XLim',[0,maxerrcurr],'YLim',[0,maxerrcurr]);
    xlabel(sprintf('%s worst landmark error',legendnames{ndx}));
    ylabel(sprintf('%s worst landmark error',legendnames{2}));
  set(hfig,'Renderer','painters','Units','pixels','Position',[10,10,300,300]);
  saveas(hfig,fullfile(savedir,sprintf('%s_DLCErrorVsMDNError.svg',exptype)),'svg');

  
  end

  
end

%% compare head angle

if ismember(exptype,{'FlyBubble'}),
  derivedstats = struct;
  for neti = 1:nnets,
    net = nets{neti};
    for ndx = 1:numel(gtdata.(net)),
      pts = permute(gtdata.(net){ndx}.pred,[3,1,4,2]);
      [~,~,~,~,headangle] = compute_HeadAngles(pts);
      legangles = compute_LegAngles(pts);
      abdomenangle = compute_AbdomenAngle(pts);
      derivedstats.(net){ndx}.headangle.pred = headangle(:);
      derivedstats.(net){ndx}.abdomenangle.pred = abdomenangle(:);
      fns = fieldnames(legangles);
      for i = 1:numel(fns),
        fn = fns{i};
        derivedstats.(net){ndx}.(fn).pred = legangles.(fn)(:);
      end
      pts = permute(gtdata.(net){ndx}.labels,[3,1,4,2]);
      [~,~,~,~,headangle] = compute_HeadAngles(pts);
      abdomenangle = compute_AbdomenAngle(pts);
      legangles = compute_LegAngles(pts);
      derivedstats.(net){ndx}.headangle.labels = headangle(:);
      derivedstats.(net){ndx}.abdomenangle.labels = abdomenangle(:);
      fns = fieldnames(legangles);
      for i = 1:numel(fns),
        fn = fns{i};
        derivedstats.(net){ndx}.(fn).labels = legangles.(fn)(:);
      end
    end
  end
  
  annotderivedstats = struct;
  annotfns = fieldnames(annoterrdata);
  for neti = 1:numel(annotfns),
    net = annotfns{neti};
    for ndx = 1:numel(annoterrdata.(net)),
      pts = permute(annoterrdata.(net){ndx}.pred,[3,1,4,2]);
      [~,~,~,~,headangle] = compute_HeadAngles(pts);
      abdomenangle = compute_AbdomenAngle(pts);
      legangles = compute_LegAngles(pts);
      annotderivedstats.(net){ndx}.headangle.pred = headangle(:);
      annotderivedstats.(net){ndx}.abdomenangle.pred = abdomenangle(:);
      fns = fieldnames(legangles);
      for i = 1:numel(fns),
        fn = fns{i};
        annotderivedstats.(net){ndx}.(fn).pred = legangles.(fn)(:);
      end
      pts = permute(annoterrdata.(net){ndx}.labels,[3,1,4,2]);
      [~,~,~,~,headangle] = compute_HeadAngles(pts);
      abdomenangle = compute_AbdomenAngle(pts);      
      legangles = compute_LegAngles(pts);
      annotderivedstats.(net){ndx}.headangle.labels = headangle(:);
      annotderivedstats.(net){ndx}.abdomenangle.labels = abdomenangle(:);
      fns = fieldnames(legangles);
      for i = 1:numel(fns),
        fn = fns{i};
        annotderivedstats.(net){ndx}.(fn).labels = legangles.(fn)(:);
      end
    end
  end
  
  anglefns = fieldnames(derivedstats.mdn{end});
  m = regexp(anglefns,'^left(.*)$','once','tokens');
  idxleft = find(~cellfun(@isempty,m));
  combinedanglefns = [m{idxleft}];
  m = regexp(anglefns,'^(left)|(right)','once');
  idxother = find(cellfun(@isempty,m));

  for i = 1:numel(combinedanglefns),
    fn = combinedanglefns{i};
    leftfn = ['left',fn];
    rightfn = ['right',fn];
    for neti = 1:nnets,
      net = nets{neti};
      for ndx = 1:numel(gtdata.(net)),
        derivedstats.(net){ndx}.(fn).pred = cat(1,derivedstats.(net){ndx}.(leftfn).pred,derivedstats.(net){ndx}.(rightfn).pred);
        derivedstats.(net){ndx}.(fn).labels = cat(1,derivedstats.(net){ndx}.(leftfn).labels,derivedstats.(net){ndx}.(rightfn).labels);
      end
    end
    for neti = 1:numel(annotfns),
      net = annotfns{neti};
      for ndx = 1:numel(annoterrdata.(net)),
        annotderivedstats.(net){ndx}.(fn).pred = cat(1,annotderivedstats.(net){ndx}.(leftfn).pred,annotderivedstats.(net){ndx}.(rightfn).pred);
        annotderivedstats.(net){ndx}.(fn).labels = cat(1,annotderivedstats.(net){ndx}.(leftfn).labels,annotderivedstats.(net){ndx}.(rightfn).labels);
      end
    end

  end
  
  for neti = 1:numel(annotfns),
    net = annotfns{neti};
    for ndx = 1:numel(annoterrdata.(net)),
      annotderivedstats.(net){ndx}.data_cond = annoterrdata.(net){ndx}.data_cond;
      annotderivedstats.(net){ndx}.label_cond = annoterrdata.(net){ndx}.label_cond;
    end
  end
  symmanglefns = [combinedanglefns,anglefns(idxother)'];
  
  for i = 1:numel(symmanglefns),
    if ismember(symmanglefns{i},{'headangle','abdomenangle'}),
      minerr = .25;
      maxerr = 30;
    else
      minerr = .5;
      maxerr = 45;
    end
    
    PlotFracInliers('gtdata',derivedstats,...
      'nets',nets,'legendnames',legendnames,...
      'colors',colors,...
      'exptype',exptype,...
      'conddata',conddata,...
      'labeltypes',labeltypes,'datatypes',datatypes,...
      'savedir',savedir,'dosavefig',dosavefig,...
      'prcs',prcs,...
      'maxprc',99.5,...
      'minerr',minerr,...
      'maxerr',maxerr,...
      'annoterrprc',[],...
      'plotannfracinliers',true,...
      'xscale','log',...
      'anglefn',symmanglefns{i},...
      'convert2deg',true,...
      'figpos',[10,10,1332,812],...
      'annoterrdata',annotderivedstats);
    %'annoterrdata',annoterrdata,...
  end
end
  
%% print data set info

strippedLblFile = struct;
strippedLblFile.SH = '/groups/branson/bransonlab/apt/experiments/data/sh_trn4992_gtcomplete_cacheddata_updated20190402_dlstripped.lbl';
strippedLblFile.FlyBubble = '/groups/branson/bransonlab/apt/experiments/data/multitarget_bubble_expandedbehavior_20180425_FxdErrs_OptoParams20181126_dlstripped.lbl';
strippedLblFile.RF = '/groups/branson/bransonlab/apt/experiments/data/romain_dlstripped.lbl';
strippedLblFile.BSView0x = '/groups/branson/bransonlab/apt/experiments/data/britton_dlstripped_0.lbl';
strippedLblFile.BSView1x = '/groups/branson/bransonlab/apt/experiments/data/britton_dlstripped_1.lbl';
strippedLblFile.BSView2x = '/groups/branson/bransonlab/apt/experiments/data/britton_dlstripped_2.lbl';
strippedLblFile.Roian = '/groups/branson/bransonlab/apt/experiments/data/roian_apt_dlstripped.lbl';
strippedLblFile.Larva = '/groups/branson/bransonlab/apt/experiments/data/larva_dlstripped_20190420.lbl';

gtResFile = struct;
gtResFile.SH = '/nrs/branson/mayank/apt_cache/stephen_view0_trainsize.mat';
gtResFile.FlyBubble = '/nrs/branson/mayank/apt_cache/alice_view0_trainsize.mat';

fns = fieldnames(strippedLblFile);
nTrain = struct;
nGT = struct;
for i = 1:numel(fns),
  fn = fns{i};
  ld = load(strippedLblFile.(fn),'-mat');
  nTrain.(fn) = size(ld.preProcData_I,1);
  fprintf('%s\t%d',fn,nTrain.(fn));
  if isfield(gtResFile,fn),
    gd = load(gtResFile.(fn));
    nGT.(fn) = size(gd.mdn{end}.pred,1);
    fprintf('\t%d',nGT.(fn));
  end
  fprintf('\n');
  
end
    

%% old stuff


% %% plot example predictions
% 
% rng(0);
% nexamples_random = 5;
% nexamples_disagree = 5;
% ms = 8;
% lw = 2;
% ndatapts = numel(conddata.data_cond);
% netsplot = find(isfield(gtdata_size,nets));
% landmarkColors = lObj.labelPointsPlotInfo.Colors;
% ndatatypes = size(datatypes);
% nlabeltypes = size(labeltypes);
% figpos = [10,10,1332,1468];
% 
% for datai = 1:ndatatypes,
%   for labeli = 1:nlabeltypes,
%     isselected = false(ndatapts,1);
%     idx = find(ismember(conddata.data_cond,datatypes{datai,2})&ismember(conddata.label_cond,labeltypes{labeli,2})&~isselected);
%     if isempty(idx),
%       continue;
%     end
%     hfigs(datai,labeli) = figure;
%     set(hfigs(datai,labeli),'Units','pixels','Position',figpos,'Renderer','painters');
%     
%     allpreds = nan([size(gtdata_size.(nets{netsplot(1)}){end}.pred),numel(netsplot)]);
%     
%     for netii = 1:numel(netsplot),
%       neti = netsplot(netii);
%       
%       predscurr = gtdata_size.(nets{neti}){end}.pred;
%       if ~isempty(strfind(nets{neti},'cpr')),
%         
%         tblP = gtimdata.tblPGT;
%         [ndatapts,nlandmarks,d] = size(predscurr);
%         % doing this manually, can't find a good function to do it
%         preds = gtdata_size.(nets{neti}){end}.pred;
%         for i = 1:ndatapts,
%           pAbs = permute(preds(i,:,:),[2,3,1]);
%           x = tblP.pTrx(i,1);
%           y = tblP.pTrx(i,2);
%           theta = tblP.thetaTrx(i);
%           T = [1,0,0
%             0,1,0
%             -x,-y,1];
%           R = [cos(theta+pi/2),-sin(theta+pi/2),0
%             sin(theta+pi/2),cos(theta+pi/2),0
%             0,0,1];
%           A = T*R;
%           tform = maketform('affine',A);
%           [pRel(:,1),pRel(:,2)] = ...
%             tformfwd(tform,pAbs(:,1),pAbs(:,2));
%           pRoi = lObj.preProcParams.TargetCrop.Radius+pRel;
%           predscurr(i,:,:) = pRoi;
%         end
%       end
%       allpreds(:,:,:,netii) = predscurr;
%     end
%     
%     if nexamples > numel(idx),
%       exampleidx = idx;
%       exampleinfo = repmat({'Rand'},[numel(exampleidx),1]);
% 
%     else
%       medpred = median(allpreds(idx,:,:,:),4);
%       disagreement = max(max(sqrt(sum( (allpreds(idx,:,:,:)-medpred).^2,3 )),[],4),[],2);
%       [sorteddisagreement,order] = sort(disagreement,1,'descend');
%       exampleidx_disagree = idx(order(1:nexamples_disagree));
%       isselected(exampleidx_disagree) = true;
%       idx = find(ismember(conddata.data_cond,datatypes{datai,2})&ismember(conddata.label_cond,labeltypes{labeli,2})&~isselected);
%       exampleidx_random = randsample(idx,nexamples_random);
%       isselected(exampleidx_random) = true;
%       exampleidx = [exampleidx_random;exampleidx_disagree];
%       exampleinfo = repmat({'Rand'},[numel(exampleidx_random),1]);
%       for i = 1:nexamples_disagree,
%         exampleinfo{end+1} = sprintf('%.1f',sorteddisagreement(i)); %#ok<SAGROW>
%       end
%       
%     end
%     hax = createsubplots(nexamples,numel(netsplot)+2,0);%[[.025,0],[.025,0]]);
%     hax = reshape(hax,[nexamples,numel(netsplot)+2]);
%     hax = hax';
%     labelscurr = gtdata_size.(nets{netsplot(1)}){end}.labels;
%     for exii = 1:numel(exampleidx),
%       exi = exampleidx(exii);
%       imagesc(gtimdata.ppdata.I{exi},'Parent',hax(1,exii));
%       axis(hax(1,exii),'image','off');
%       hold(hax(1,exii),'on');
%       for pti = 1:npts,
%         plot(hax(1,exii),labelscurr(exi,pti,1),labelscurr(exi,pti,2),'+','Color',landmarkColors(pti,:),'MarkerSize',ms,'LineWidth',lw);
%       end
%       text(1,size(gtimdata.ppdata.I{exi},1)/2,...
%         sprintf('%d, %s (Mov %d, Tgt %d, Frm %d)',exi,exampleinfo{exii},...
%         -gtimdata.tblPGT.mov(exi),gtimdata.tblPGT.iTgt(exi),gtimdata.tblPGT.frm(exi)),...
%         'Rotation',90,'HorizontalAlignment','center','VerticalAlignment','top','Parent',hax(1,exii),'FontSize',6);
%     end
%     exi = exampleidx(1);
%     text(size(gtimdata.ppdata.I{exi},2)/2,1,'Groundtruth','HorizontalAlignment','center','VerticalAlignment','top','Parent',hax(1,1));
%     
%     for exii = 1:numel(exampleidx),
%       exi = exampleidx(exii);
%       imagesc(gtimdata.ppdata.I{exi},'Parent',hax(2,exii));
%       axis(hax(2,exii),'image','off');
%       hold(hax(2,exii),'on');
%       for pti = 1:npts,
%         plot(hax(2,exii),labelscurr(exi,pti,1),labelscurr(exi,pti,2),'+','Color',landmarkColors(pti,:),'MarkerSize',ms,'LineWidth',lw);
%       end
%       for pti = 1:npts,
%         plot(hax(2,exii),squeeze(allpreds(exi,pti,1,:)),squeeze(allpreds(exi,pti,2,:)),'.','Color',ptcolors(pti,:),'MarkerSize',ms,'LineWidth',lw);
%       end
%     end
%     exi = exampleidx(1);
%     text(size(gtimdata.ppdata.I{exi},2)/2,1,'All','HorizontalAlignment','center','VerticalAlignment','top','Parent',hax(2,1));
% 
%     
%     for netii = 1:numel(netsplot),
%       neti = netsplot(netii);
%       
%       predscurr = allpreds(:,:,:,netii);
%       
%       for exii = 1:numel(exampleidx),
%         exi = exampleidx(exii);
%         imagesc(gtimdata.ppdata.I{exi},'Parent',hax(netii+2,exii));
%         axis(hax(netii+2,exii),'image','off');
%         hold(hax(netii+2,exii),'on');
%         for pti = 1:npts,
%           plot(hax(netii+2,exii),predscurr(exi,pti,1),predscurr(exi,pti,2),'+','Color',landmarkColors(pti,:),'MarkerSize',ms,'LineWidth',lw);
%         end
%         err = max(sqrt(sum((predscurr(exi,:,:)-labelscurr(exi,:,:)).^2,3)));
%         text(size(gtimdata.ppdata.I{exi},2)/2,size(gtimdata.ppdata.I{exi},1),num2str(err),'HorizontalAlignment','center','VerticalAlignment','bottom','Parent',hax(netii+2,exii));
%       end
%       text(size(gtimdata.ppdata.I{exi},2)/2,1,legendnames{neti},'HorizontalAlignment','center','VerticalAlignment','top','Parent',hax(netii+2,1));
%     end
%     colormap(hfigs(datai,labeli),'gray');
% 
%     set(hfigs(datai,labeli),'Name',sprintf('Examples: %s, %s, %s',exptype,datatypes{datai,1},labeltypes{labeli,1}));
%     savenames{datai,labeli} = fullfile(savedir,sprintf('%s_Examples_%s_%s.svg',exptype,datatypes{datai,1},labeltypes{labeli,1}));
%     saveas(hfigs(datai,labeli),savenames{datai,labeli},'svg');
%           
%   end
% end

% nbins = 60;
% binedges = linspace(minerr,maxerr,nbins+1);
% bincenters = (binedges(1:end-1)+binedges(2:end))/2;
% prcs = [50,75,90,95,97];%,99];
% markers = {'o','s','<','^','v','>'};
% nprcs = numel(prcs);
% 
% npttypes = size(pttypes,1);
% nlabeltypes = size(labeltypes,1);
% ndatatypes = size(datatypes,1);

% %% compute stats for error as a function of training set size
% 
% gtdata_size = load(gtfile_trainsize);
% if ~isempty(gtfile_trainsize_cpr),
%   gtdata_cpr = load(gtfile_trainsize_cpr);
% end
% newfns = setdiff(fieldnames(gtdata_cpr),fieldnames(gtdata_size));
% for i = 1:numel(newfns),
%   gtdata_size.(newfns{i}) = gtdata_cpr.(newfns{i});
% end
% 
% 
% %nets = fieldnames(gtdata);
% npts = size(gtdata_size.mdn{1}.labels,2);
% 
% n_models = numel(gtdata_size.(nets{1}));
% n_train = cellfun(@(x) x.model_timestamp,gtdata_size.mdn(2:end));
% 
% errfrac = nan([nbins,nnets,n_models-1,npttypes,nlabeltypes,ndatatypes]);
% errprctiles = nan([nprcs,nnets,n_models-1,npttypes,nlabeltypes,ndatatypes]);
% ndatapts = nan([nnets,n_models-1,npttypes,nlabeltypes,ndatatypes]);
% maxdist = nan([nprcs,nnets,n_models-1,nlabeltypes,ndatatypes]);
% mindist = nan([nprcs,nnets,n_models-1,nlabeltypes,ndatatypes]);
% mediandist = nan([nprcs,nnets,n_models-1,nlabeltypes,ndatatypes]);
% 
% for ndx = 1:nnets,
%   for mndx = 2:n_models
%     if numel(gtdata_size.(nets{ndx})) < mndx,
%       break;
%     end
%     cur_data = gtdata_size.(nets{ndx}){mndx};
%     preds = cur_data.pred;
%     labels = cur_data.labels;
%     assert(size(preds,3)==2);
%     assert(size(labels,3)==2);
%     iscpr = ~isempty(strfind(nets{ndx},'cpr'));
%     
%     for datai = 1:ndatatypes,
%       dtis = datatypes{datai,2};
%       for labeli = 1:nlabeltypes,
%         ltis = labeltypes{labeli,2};
%         idx = ismember(conddata.data_cond,dtis) & ismember(conddata.label_cond,ltis);
%         if iscpr && isshexp,
%           % special case for SH/cpr whose computed GT output only has
%           % 1149 rows instead of 1150 cause dumb
%           idx(4) = [];
%         end
%         
%         for typei = 1:npttypes,
%           ptis = pttypes{typei,2};
%           if all(idx)
%             dist = sqrt(sum( (preds(:,ptis,:)-labels(:,ptis,:)).^2,3)); 
%           else
%             dist = sqrt(sum( (preds(idx,ptis,:)-labels(idx,ptis,:)).^2,3));
%           end
%           counts = hist(dist(:),bincenters);
%           ndatapts(ndx,mndx-1,typei,labeli,datai) = sum(counts);
%           errfrac(:,ndx,mndx-1,typei,labeli,datai) = counts;
%           errprctiles(:,ndx,mndx-1,typei,labeli,datai) = prctile(dist(:),prcs);
%         end
%         dist = sqrt(sum( (preds(idx,:,:)-labels(idx,:,:)).^2,3));
%         maxdistcurr = max(dist,[],2);
%         mediandistcurr = median(dist,2);
%         mindistcurr = min(dist,[],2);
%         maxdist(:,ndx,mndx-1,labeli,datai) = prctile(maxdistcurr(:),prcs);
%         mediandist(:,ndx,mndx-1,labeli,datai) = prctile(mediandistcurr(:),prcs);
%         mindist(:,ndx,mndx-1,labeli,datai) = prctile(mindistcurr(:),prcs);
%         
%       end
%     end
% 
%   end
% end

% %% plot stats as a function of training set size - hist
% 
% 
% offs = linspace(-.3,.3,nnets);
% doff = offs(2)-offs(1);
% dx = [-1,1,1,-1,-1]*doff/2;
% dy = [0,0,1,1,0];
% 
% hfigs = gobjects(1,ndatatypes);
% 
% for datai = ndatatypes:-1:1,
% 
%   hfigs(datai) = figure;
%   clf;
%   set(hfigs(datai),'Position',[10,10,2526,150+1004/4*nlabeltypes]);
%   hax = createsubplots(nlabeltypes,npttypes,[[.025,.005];[.05,.005]]);
%   hax = reshape(hax,[nlabeltypes,npttypes]);
%   
%   for labeli = 1:nlabeltypes,
%   
%     for pti = 1:npttypes,
%   
%       axes(hax(labeli,pti));
%   
%       hold on;
% 
%       maxfrac = max(max(max(errfrac(:,:,:,pti,labeli,datai))));
%       
%       h = gobjects(1,nnets);
%       hprcs = gobjects(1,nprcs);
%       for ndx = 1:nnets,
%         for mndx = 2:n_models
%           offx = mndx-1 + offs(ndx);
%           if numel(gtdata.(nets{ndx})) < mndx,
%             patch(offx+dx,minerr+maxerr*dy,[.7,.7,.7],'LineStyle','none');
%             break;
%           end
%       
%           for bini = 1:nbins,
%             colorfrac = min(1,errfrac(bini,ndx,mndx-1,pti,labeli,datai)/maxfrac);
%             if colorfrac == 0,
%               continue;
%             end
%             patch(offx+dx,binedges(bini+dy),colors(ndx,:)*colorfrac + 1-colorfrac,'LineStyle','none');
%           end
%           for prci = 1:nprcs,
%             hprcs(prci) = plot(offx,min(maxerrplus,errprctiles(prci,ndx,mndx-1,pti,labeli,datai)),'w','MarkerFaceColor',colors(ndx,:),'Marker',markers{prci});
%           end
%         end
%         h(ndx) = patch(nan(size(dx)),nan(size(dx)),colors(ndx,:),'LineStyle','none');
%       end
%   
%       set(gca,'XTick',1:n_models-1,'XTickLabels',num2str(n_train(:)));
%   
%       if pti == 1 && labeli == 1,
%         legend([h,hprcs],[legendnames,arrayfun(@(x) sprintf('%d %%ile',x),prcs,'Uni',0)]);
%       end
%       if labeli == nlabeltypes,
%         xlabel('N. training examples');
%       end
%       if pti == 1,
%         ylabel(labeltypes{labeli,1});
%       end
%       if labeli == 1,
%         title(pttypes{pti,1},'FontWeight','normal');
%       end
%       set(gca,'XLim',[0,n_models],'YLim',[0,maxerrplus]);
%       drawnow;
%     end
%     set(hax(:,2:end),'YTickLabel',{});
%     set(hax(1:end-1,:),'XTickLabel',{});
%     
%   end
%   set(hfigs(datai),'Name',datatypes{datai,1});
%   set(hfigs(datai),'Renderer','painters');
%   break
%   %savefig(hfigs(datai),sprintf('FlyBubble_GTTrackingError_TrainingSetSize_%s.fig',datatypes{datai,1}),'compact');
%   saveas(hfigs(datai),sprintf('%s_GTTrackingError_TrainingSetSize_%s.svg',exptype,datatypes{datai,1}),'svg');
% 
% end

% %% plot stats as a function of training set size - prctiles
% 
% 
% offs = linspace(-.3,.3,nnets);
% doff = offs(2)-offs(1);
% dx = [-1,1,1,-1,-1]*doff/2;
% dy = [0,0,1,1,0];
% 
% hfigs = gobjects(1,ndatatypes);
% 
% for datai = ndatatypes:-1:1,
% 
%   hfigs(datai) = figure;
%   clf;
%   set(hfigs(datai),'Position',[10,10,2526/2,(150+1004/4*nlabeltypes)/2]);
%   hax = createsubplots(nlabeltypes,npttypes,[[.025,.005];[.1,.005]]);
%   hax = reshape(hax,[nlabeltypes,npttypes]);
% 
%   for nprcsplot = 1:nprcs,
%   
%   for labeli = 1:nlabeltypes,
%   
%     for pti = 1:npttypes,
%   
%       axes(hax(labeli,pti));
%       cla(hax(labeli,pti));
%   
%       hold on;
% 
%       maxfrac = max(max(max(errfrac(:,:,:,pti,labeli,datai))));
%       
%       h = gobjects(1,nnets);
%       hprcs = gobjects(1,nprcs);
%       for ndx = 1:nnets,
%         for mndx = 2:n_models
%           offx = mndx-1 + offs(ndx);
%           if numel(gtdata_size.(nets{ndx})) < mndx,
%             patch(offx+dx,minerr+maxerr*dy,[.7,.7,.7],'LineStyle','none');
%             break;
%           end
%           
%           
%           for prci = nprcsplot:-1:1,
%             colorfrac = colorfracprcs(prci);
%             tmp = [0,errprctiles(prci,ndx,mndx-1,pti,labeli,datai)];
%             patch(offx+dx,tmp(1+dy),colors(ndx,:)*colorfrac + 1-colorfrac,'EdgeColor',colors(ndx,:));
%           end
%       
%         end
%         if pti == 1 && labeli == 1,
%           h(ndx) = patch(nan(size(dx)),nan(size(dx)),colors(ndx,:),'LineStyle','none');
%         end
%       end
%   
%       set(gca,'XTick',1:n_models-1,'XTickLabels',num2str(n_train(:)));
%   
%       if pti == 1 && labeli == 1 && nprcsplot == nprcs,
%         hprcs = gobjects(1,nprcs);
%         for prci = 1:nprcs,
%           hprcs(prci) = patch(nan(size(dx)),nan(size(dx)),[1,1,1]-colorfracprcs(prci),'EdgeColor','k');
%         end
%         legend([h,hprcs],[legendnames,arrayfun(@(x) sprintf('%d %%ile',x),prcs,'Uni',0)]);
%       end
%       if labeli == nlabeltypes,
%         xlabel('N. training examples');
%       end
%       if pti == 1,
%         ylabel(labeltypes{labeli,1});
%       end
%       if labeli == 1,
%         title(pttypes{pti,1},'FontWeight','normal');
%       end
%       set(gca,'XLim',[0,n_models],'YLim',[0,maxerrplus]);
%       drawnow;
%     end
%     set(hax(:,2:end),'YTickLabel',{});
%     set(hax(1:end-1,:),'XTickLabel',{});
%     
%   end
%   set(hfigs(datai),'Name',datatypes{datai,1});
%   set(hfigs(datai),'Renderer','painters');
%   %keyboard;
%   %savefig(hfigs(datai),sprintf('FlyBubble_GTTrackingError_TrainingSetSize_%s.fig',datatypes{datai,1}),'compact');
%   saveas(hfigs(datai),fullfile(savedir,sprintf('%s_GTTrackingError_TrainingSetSize_Prctile%d_%s.svg',exptype,nprcsplot,datatypes{datai,1})),'svg');
%   end
%   %break;
% end

% %% plot worst error percentiles
% 
% clear hfigs;
% 
% for stati = 1:3,
%   switch stati,
%     case 1
%       statname = 'Worst';
%       statdist = maxdist;
%     case 2
%       statname = 'Median';
%       statdist = mediandist;
%     case 3
%       statname = 'Best';
%       statdist = mindist;
%   end
%   
%   
%   for datai = ndatatypes:-1:1,
%     
%     hfigs(datai,stati) = figure;
%     clf;
%     set(hfigs(datai,stati),'Position',[10,10,2526/2,(150+1004/4*nlabeltypes)/2]);
%     hax = createsubplots(nlabeltypes,nprcs,[[.025,.005];[.1,.005]]);
%     hax = reshape(hax,[nlabeltypes,nprcs]);
%     
%     for prci= 1:nprcs,
%       
%       minv = min(min(min(min(statdist(prci,:,:,:,:)))));
%       
%       for labeli = 1:nlabeltypes,
%         
%         axes(hax(labeli,prci));
%         cla(hax(labeli,prci));
%         
%         hold on;
%         h = gobjects(1,nnets);
%         for ndx = 1:nnets,
%           h(ndx) = plot(1:n_models-1,squeeze(statdist(prci,ndx,:,labeli,datai))','.-','LineWidth',2,'Color',colors(ndx,:),'MarkerSize',12);
%         end
%         set(gca,'XTick',1:n_models-1,'XTickLabels',num2str(n_train(:)));
%         if labeli == 1 && prci == 1,
%           legend(h,legendnames);
%         end
%         if labeli == nlabeltypes,
%           xlabel('N. training examples');
%         end
%         if prci == 1,
%           ylabel(labeltypes{labeli,1});
%         end
%         if labeli == 1,
%           title(sprintf('%dth %%ile, %s landmark',prcs(prci),statname),'FontWeight','normal');
%         end
%         set(gca,'XLim',[0,n_models],'YLim',[minv,maxerrplus]);
%         drawnow;
%         set(hax(labeli,prci),'YScale','log');
%       end
%       linkaxes(hax(:,prci));
%     end
%     set(hax(:,2:end),'YTickLabel',{});
%     set(hax(1:end-1,:),'XTickLabel',{});
%     
%     set(hfigs(datai,stati),'Name',sprintf('%s landmark, %s',lower(statname),datatypes{datai,1}));
%     set(hfigs(datai,stati),'Renderer','painters');
%     %keyboard;
%     %savefig(hfigs(datai),sprintf('FlyBubble_GTTrackingError_TrainingSetSize_%s.fig',datatypes{datai,1}),'compact');
%     saveas(hfigs(datai),fullfile(savedir,sprintf('%s_GTTrackingError_TrainingSetSize_%sLandmark_%s.svg',exptype,statname,datatypes{datai,1})),'svg');
%   end
% end

% %%
% gtdata_size = load(gtfile_traintime);
% 
% %nets = fieldnames(gtdata);
% npts = size(gtdata_size.mdn{1}.labels,2);
% 
% n_models = 0;
% for ndx = 1:nnets,
%   if ~isfield(gtdata_size,nets{ndx}),
%     continue;
%   end
%   n_models = max(n_models,numel(gtdata_size.(nets{ndx})));
% end
% train_time = nan(nnets,n_models-1);
% for ndx = 1:nnets,
% 
%   if ~isfield(gtdata_size,nets{ndx}),
%     continue;
%   end
%   ts = cellfun(@(x) x.model_timestamp,gtdata_size.(nets{ndx}));
%   train_time(ndx,1:numel(ts)-1) = (ts(2:end)-ts(1))/3600;
% end
% 
% errfrac = nan([nbins,nnets,n_models-1,npttypes,nlabeltypes,ndatatypes]);
% errprctiles = nan([nprcs,nnets,n_models-1,npttypes,nlabeltypes,ndatatypes]);
% ndatapts = nan([nnets,n_models-1,npttypes,nlabeltypes,ndatatypes]);
% 
% for ndx = 1:nnets,
%   if ~isfield(gtdata_size,nets{ndx}),
%     continue;
%   end
% 
%   for mndx = 2:n_models
%     
%     if numel(gtdata_size.(nets{ndx})) < mndx,
%       break;
%     end
%     if ~isfield(gtdata_size,nets{ndx}),
%       continue;
%     end
%     cur_data = gtdata_size.(nets{ndx}){mndx};
%     preds = cur_data.pred;
%     labels = cur_data.labels;
%     assert(size(preds,3)==2);
%     assert(size(labels,3)==2);
%    
%     for datai = 1:ndatatypes,
%       dtis = datatypes{datai,2};
%       for labeli = 1:nlabeltypes,
%         ltis = labeltypes{labeli,2};
%         idx = ismember(conddata.data_cond,dtis) & ismember(conddata.label_cond,ltis);
%         for typei = 1:npttypes,
%           ptis = pttypes{typei,2};
%           dist = sqrt(sum( (preds(idx,ptis,:)-labels(idx,ptis,:)).^2,3));
%           counts = hist(dist(:),bincenters);
%           ndatapts(ndx,mndx-1,typei,labeli,datai) = sum(counts);
%           errfrac(:,ndx,mndx-1,typei,labeli,datai) = counts ;
%           errprctiles(:,ndx,mndx-1,typei,labeli,datai) = prctile(dist(:),prcs);
%         end
%       end
%     end
% 
%   end
% end
% 
% %% plot stats as a function of training time
% 
% 
% max_train_time = max(train_time(:));
% %patchr = min(pdist(train_time(:)));
% patchr = max_train_time /100;
% dx = [-1,1,1,-1,-1]*patchr;
% dy = [0,0,1,1,0];
% 
% hfigs = gobjects(1,ndatatypes);
% 
% for datai = ndatatypes:-1:1,
% 
%   hfigs(datai) = figure;
%   clf;
%   
%   set(hfigs(datai),'Position',[10,10,2526,150+1004/4*nlabeltypes]);
%   %set(hfigs(datai),'Position',[10,10,2526,1154]);
%   hax = createsubplots(nlabeltypes,npttypes,[[.025,.005];[.05,.005]]);
%   hax = reshape(hax,[nlabeltypes,npttypes]);
%   
%   for labeli = 1:nlabeltypes,
%   
%     for pti = 1:npttypes,
%   
%       axes(hax(labeli,pti));
%   
%       hold on;
% 
%       maxfrac = max(max(max(errfrac(:,:,:,pti,labeli,datai))));
%       
%       h = gobjects(1,nnets);
%       hprcs = gobjects(1,nprcs);
%       for ndx = 1:nnets,
%         for mndx = 2:n_models
%           offx = train_time(ndx,mndx-1);
%           if numel(gtdata_size.(nets{ndx})) < mndx,
%             break;
%           end
%       
%           for bini = 1:nbins,
%             colorfrac = min(1,errfrac(bini,ndx,mndx-1,pti,labeli,datai)/maxfrac);
%             if colorfrac == 0,
%               continue;
%             end
%             patch(offx+dx,binedges(bini+dy),colors(ndx,:)*colorfrac + 1-colorfrac,'LineStyle','none');
%           end
% 
%         end
%         if labeli == 1 && pti == 1,
%           h(ndx) = patch(nan(size(dx)),nan(size(dx)),colors(ndx,:),'LineStyle','none');
%         end
%       end
%       for ndx = 1:nnets,
%         for mndx = 2:n_models
%           offx = train_time(ndx,mndx-1);
%           if numel(gtdata_size.(nets{ndx})) < mndx,
%             break;
%           end
%           for prci = 1:nprcs,
%             hprcs(prci) = plot(offx,min(maxerrplus,errprctiles(prci,ndx,mndx-1,pti,labeli,datai)),'w','MarkerFaceColor',colors(ndx,:),'Marker',markers{prci});
%           end
%         end
%       end
%   
%       %set(gca,'XTick',1:n_models-1,'XTickLabels',num2str(n_train(:)));
%   
%       if pti == 1 && labeli == 1,
%         legend([h,hprcs],[nets,arrayfun(@(x) sprintf('%d %%ile',x),prcs,'Uni',0)]);
%       end
%       if labeli == nlabeltypes,
%         xlabel('Training time (h)');
%       end
%       if pti == 1,
%         ylabel(labeltypes{labeli,1});
%       end
%       if labeli == 1,
%         title(pttypes{pti,1},'FontWeight','normal');
%       end
%       set(gca,'XLim',[min(train_time(:))-3*patchr,max(train_time(:))+3*patchr],'YLim',[0,maxerrplus]);
%       drawnow;
%     end
%     set(hax(:,2:end),'YTickLabel',{});
%     set(hax(1:end-1,:),'XTickLabel',{});
%     
%   end
%   set(hfigs(datai),'Name',datatypes{datai,1});
%   
%   set(hfigs(datai),'Renderer','painters');
%   %savefig(hfigs(datai),sprintf('FlyBubble_GTTrackingError_TrainingTime_%s.fig',datatypes{datai,1}),'compact');
%   saveas(hfigs(datai),sprintf('%s_GTTrackingError_TrainingTime_%s.svg',exptype,datatypes{datai,1}),'svg');
% end
% 
% %% 
% % for datai = 1:ndatatypes,
% %   set(hfigs(datai),'Renderer','painters');
% %   %savefig(hfigs(datai),sprintf('FlyBubble_GTTrackingError_TrainingTime_%s.fig',datatypes{datai,1}),'compact');
% %   saveas(hfigs(datai),sprintf('FlyBubble_GTTrackingError_TrainingTime_%s.svg',datatypes{datai,1}),'svg');
% % end
% 
% close all;
% 
% %% plot stats as a function of training time -prctiles only
% 
% 
% max_train_time = max(train_time(:));
% %patchr = min(pdist(train_time(:)));
% patchr = max_train_time /100;
% dx = [-1,1,1,-1,-1]*patchr;
% dy = [0,0,1,1,0];
% 
% hfigs = gobjects(1,ndatatypes);
% doplotnet = ~ismember(nets,{'cpropt','cprqck'});
% %doplotnet = ~ismember(nets,{'cprqck'});
% 
% for datai = ndatatypes:-1:1,
% 
%   hfigs(datai) = figure;
%   clf;
%   
%   set(hfigs(datai),'Position',[10,10,2526/2,(150+1004/4*nlabeltypes)/2]);
%   %set(hfigs(datai),'Position',[10,10,2526,1154]);
%   hax = createsubplots(nlabeltypes,npttypes,[[.025,.005];[.05,.005]]);
%   hax = reshape(hax,[nlabeltypes,npttypes]);
%   
%   for labeli = 1:nlabeltypes,
%   
%     for pti = 1:npttypes,
%   
%       axes(hax(labeli,pti));
%   
%       hold on;
% 
%       maxfrac = max(max(max(errfrac(:,:,:,pti,labeli,datai))));
%       
%       h = gobjects(1,nnets);
%       hprcs = gobjects(1,nprcs);
%       for ndx = 1:nnets,
%         if ~doplotnet(ndx),
%           continue;
%         end
%           
%         for mndx = 2:n_models
%           offx = train_time(ndx,mndx-1);
%           if numel(gtdata_size.(nets{ndx})) < mndx,
%             break;
%           end
%           
%           for prci = nprcs:-1:1,
%             colorfrac = colorfracprcs(prci);
%             tmp = [0,errprctiles(prci,ndx,mndx-1,pti,labeli,datai)];
%             patch(offx+dx,tmp(1+dy),colors(ndx,:)*colorfrac + 1-colorfrac,'EdgeColor',colors(ndx,:));
%           end
%       
%         end
%         if labeli == 1 && pti == 1,
%           h(ndx) = patch(nan(size(dx)),nan(size(dx)),colors(ndx,:),'LineStyle','none');
%         end
%       end
%       %set(gca,'XTick',1:n_models-1,'XTickLabels',num2str(n_train(:)));
%   
%       if pti == 1 && labeli == 1,
%         hprcs = gobjects(1,nprcs);
%         for prci = 1:nprcs,
%           hprcs(prci) = patch(nan(size(dx)),nan(size(dx)),[1,1,1]-colorfracprcs(prci),'EdgeColor','k');
%         end
%         legend([h(doplotnet),hprcs],[legendnames(doplotnet),arrayfun(@(x) sprintf('%d %%ile',x),prcs,'Uni',0)]);
%       end
%       if labeli == nlabeltypes,
%         xlabel('Training time (h)');
%       end
%       if pti == 1,
%         ylabel(labeltypes{labeli,1});
%       end
%       if labeli == 1,
%         title(pttypes{pti,1},'FontWeight','normal');
%       end
%       set(gca,'XLim',[min(train_time(:))-3*patchr,max(train_time(:))+3*patchr],'YLim',[0,maxerrplus]);
%       drawnow;
%     end
%     set(hax(:,2:end),'YTickLabel',{});
%     set(hax(1:end-1,:),'XTickLabel',{});
%     
%   end
%   set(hfigs(datai),'Name',datatypes{datai,1});
%   
%   set(hfigs(datai),'Renderer','painters');
%   %savefig(hfigs(datai),sprintf('FlyBubble_GTTrackingError_TrainingTime_%s.fig',datatypes{datai,1}),'compact');
%   break;
%   saveas(hfigs(datai),fullfile(savedir,sprintf('%s_GTTrackingError_TrainingTime_Prctiles_%s.svg',exptype,datatypes{datai,1})),'svg');
% end


% %% compute err again only for last entry
% 
% 
% gtdata = load(gtfile_trainsize);
% if ~isempty(gtfile_trainsize_cpr),
%   gtdata_cpr = load(gtfile_trainsize_cpr);
% end
% newfns = setdiff(fieldnames(gtdata_cpr),fieldnames(gtdata));
% for i = 1:numel(newfns),
%   gtdata.(newfns{i}) = gtdata_cpr.(newfns{i});
% end
% % 
% % gtdata = load(gtfile_trainsize);
% 
% %nets = fieldnames(gtdata);
% npts = size(gtdata.mdn{1}.labels,2);
% errfrac = nan([nbins,nnets,1,npts,nlabeltypes,ndatatypes]);
% errprctiles = nan([nprcs,nnets,1,npttypes,nlabeltypes,ndatatypes]);
% errprctilespts = nan([nprcs,nnets,1,npts,nlabeltypes,ndatatypes]);
% ndatapts = nan([nnets,1,npttypes,nlabeltypes,ndatatypes]);
% 
% maxdist = nan([nprcs,nnets,nlabeltypes,ndatatypes]);
% mindist = nan([nprcs,nnets,nlabeltypes,ndatatypes]);
% mediandist = nan([nprcs,nnets,nlabeltypes,ndatatypes]);
% 
% plottype = 'prctiles';
% prccolors = flipud(jet(100));
% prccolors = prccolors(round(linspace(1,100,nprcs)),:);
% 
% nerrthresh = 1000;
% errthresh = linspace(0,maxerr,nerrthresh);
% worstfracinliers = nan([nerrthresh,nnets,nlabeltypes,ndatatypes]);
% medianfracinliers = nan([nerrthresh,nnets,nlabeltypes,ndatatypes]);
% bestfracinliers = nan([nerrthresh,nnets,nlabeltypes,ndatatypes]);
% 
% for ndx = 1:nnets,
%   mndx = numel(gtdata.(nets{ndx}));
%   cur_data = gtdata.(nets{ndx}){mndx};
%   preds = cur_data.pred;
%   labels = cur_data.labels;
%   assert(size(preds,3)==2);
%   assert(size(labels,3)==2);
%   iscpr = ~isempty(strfind(nets{ndx},'cpr'));
% 
%   for datai = 1:ndatatypes,
%     dtis = datatypes{datai,2};
%     for labeli = 1:nlabeltypes,
%       ltis = labeltypes{labeli,2};
%       idx = ismember(conddata.data_cond,dtis) & ismember(conddata.label_cond,ltis);
%       if iscpr && isshexp,
%         % special case for SH/cpr whose computed GT output only has
%         % 1149 rows instead of 1150 cause dumb
%         idx(4) = [];
%       end
%       
%       for typei = 1:npttypes,
%         %ptis = typei;
%         ptis = pttypes{typei,2};
%         dist = sqrt(sum( (preds(idx,ptis,:)-labels(idx,ptis,:)).^2,3));
%         counts = hist(dist(:),bincenters);
%         ndatapts(ndx,1,typei,labeli,datai) = sum(counts);
%         errfrac(:,ndx,1,typei,labeli,datai) = counts;
%         errprctiles(:,ndx,1,typei,labeli,datai) = prctile(dist(:),prcs);
%       end
%       
%       for pti=1:npts
%         dist = sqrt(sum( (preds(idx,pti,:)-labels(idx,pti,:)).^2,3));
%         errprctilespts(:,ndx,1,pti,labeli,datai) = prctile(dist(:),prcs);        
%       end
%       
%       dist = sqrt(sum( (preds(idx,:,:)-labels(idx,:,:)).^2,3));
%       maxdistcurr = max(dist,[],2);
%       mediandistcurr = median(dist,2);
%       mindistcurr = min(dist,[],2);
%       maxdist(:,ndx,labeli,datai) = prctile(maxdistcurr(:),prcs);
%       mediandist(:,ndx,labeli,datai) = prctile(mediandistcurr(:),prcs);
%       mindist(:,ndx,labeli,datai) = prctile(mindistcurr(:),prcs);
%       % this is not optimal, but fast enough, not worth being smarter
%       ncurr = numel(maxdistcurr);
%       for i = 1:nerrthresh,
%         worstfracinliers(i,ndx,labeli,datai) = nnz(maxdistcurr <= errthresh(i)) / ncurr;
%         bestfracinliers(i,ndx,labeli,datai) = nnz(mindistcurr <= errthresh(i)) / ncurr;
%         medianfracinliers(i,ndx,labeli,datai) = nnz(mediandistcurr <= errthresh(i)) / ncurr;
%       end
%       
%     end
%   end
% end


%% compare algorithms per part

% offs = linspace(-.3,.3,nnets);
% doff = offs(2)-offs(1);
% dx = [-1,1,1,-1,-1]*doff/2;
% dy = [0,0,1,1,0];
% 
% hfigs = gobjects(1,ndatatypes);
% maxerrplus = maxerr*1.05;
% 
% for datai = ndatatypes:-1:1,
% 
%   hfigs(datai) = figure;
%   clf;
%   set(hfigs(datai),'Position',[10,10,150+(2080-150)/4*nlabeltypes,520]);
%   hax = createsubplots(1,nlabeltypes,[[.025,.005];[.05,.005]]);
%   hax = reshape(hax,[1,nlabeltypes]);
%   
%   for labeli = 1:nlabeltypes,
%     axes(hax(labeli));
%     hold on;
%   
%     h = gobjects(1,nnets);
%     for pti = 1:npttypes,
%       maxfrac = max(max(max(errfrac(:,:,:,pti,labeli,datai))));
%       
%       hprcs = gobjects(1,nprcs);
%       for ndx = 1:nnets,
%         %mndx = numel(gtdata.(nets{ndx}));
%         mndx = 2;
%         offx = pti + offs(ndx);
%         for bini = 1:nbins,
%           colorfrac = min(1,errfrac(bini,ndx,mndx-1,pti,labeli,datai)/maxfrac);
%           if colorfrac == 0,
%             continue;
%           end
%           patch(offx+dx,binedges(bini+dy),colors(ndx,:)*colorfrac + 1-colorfrac,'LineStyle','none');
%         end
%         for prci = 1:nprcs,
%           hprcs(prci) = plot(offx,min(maxerrplus,errprctiles(prci,ndx,mndx-1,pti,labeli,datai)),'w','MarkerFaceColor',colors(ndx,:),'Marker',markers{prci});
%         end
%         if pti == 1,
%           h(ndx) = patch(nan(size(dx)),nan(size(dx)),colors(ndx,:),'LineStyle','none');
%         end
%       end
%     end
%     
%     set(gca,'XTick',1:npttypes,'XTickLabels',pttypes,'XtickLabelRotation',45);
%   
%     if labeli == 1,
%       legend([h,hprcs],[nets,arrayfun(@(x) sprintf('%d %%ile',x),prcs,'Uni',0)]);
%     end
%     if labeli == 1,
%       ylabel('Error (px)');
%     end
%     title(labeltypes{labeli,1},'FontWeight','normal');
%     set(gca,'XLim',[0,npttypes+1],'YLim',[0,maxerrplus]);
%     drawnow;
%   end
%   set(hax(2:end),'YTickLabel',{});
%   
%   set(hfigs(datai),'Name',datatypes{datai,1});
%   set(hfigs(datai),'Renderer','painters');
%   %savefig(hfigs(datai),sprintf('FlyBubble_GTTrackingError_TrainingSetSize_%s.fig',datatypes{datai,1}),'compact');
%   saveas(hfigs(datai),sprintf('%s_GTTrackingError_FinalHist_%s.svg',exptype,datatypes{datai,1}),'svg');
% 
% end
