exptype = 'FlyBubbleAnnotator';
savedir = '/groups/branson/bransonlab/apt/experiments/res/gt/20190523';
cprdir = '/groups/branson/bransonlab/apt/experiments/res/cprgt20190407';

if ~exist(savedir,'dir'),
  mkdir(savedir);
end
dosavefig = false;

%% parameters

nets = {'openpose','leap','deeplabcut','unet','resnet_unet','mdn','cpropt','Alice','Austin'};
legendnames = {'OpenPose','LEAP','DeepLabCut','U-net','Res-U-net','MDN','CPR','Inter-annotator','Intra-annotator'};
nnets = numel(nets);
colors = [
  0         0.4470    0.7410
  0.4660    0.6740    0.1880
  0.8500    0.3250    0.0980
  0.9290    0.6940    0.1250
  0.6350    0.0780    0.1840
  0.4940    0.1840    0.5560
  0.3010    0.7450    0.9330
  .7        .7        .7
  .3        .3        .3
  ];
prcs = [50,75,90,95,97];

idxnet = [1 2 3 7 4 5 6 8 9];
nets = nets(idxnet);
colors = colors(idxnet,:);
legendnames = legendnames(idxnet);
vwi = 1;


gtfile_trainsize_cpr = fullfile(cprdir,'outputFINAL/alice_view0_trainsize_withcpr.mat');
gtfile_trainsize = '/nrs/branson/mayank/apt_cache/alice_view0_trainsize.mat';
annoterrfile = '/groups/branson/home/robiea/Projects_data/Labeler_APT/Austin_labelerprojects_expandedbehaviors/GT/RelabelingGT_20190529.mat';
condinfofile = '/nrs/branson/mayank/apt_cache/multitarget_bubble/multitarget_bubble_expandedbehavior_20180425_condinfo.mat';
gtimagefile = '/groups/branson/home/bransonk/tracking/code/APT/FlyBubbleGTData20190524.mat';

pttypes = {'head',[1,2,3]
  'body',[4,5,6,7]
  'middle leg joint 1',[8,10]
  'middle leg joint 2',[9,11]
  'front leg tarsi',[12,17]
  'middle leg tarsi',[13,16]
  'back leg tarsi',[14,15]};
labeltypes = {'moving',1
  'grooming',2
  'close',3
  'all',[1,2,3]};
datatypes = {'same fly',1
  'same genotype',2
  'different genotype',3
  'all',[1,2,3]};
lblfile = '/groups/branson/home/bransonk/tracking/code/APT/multitarget_bubble_expandedbehavior_20180425_FxdErrs_OptoParams20181126_mdn20190214_skeledges.lbl';
maxerr = 30;


%% images for overlaying percentile errors 

lObj = StartAPT;
lObj.projLoad(lblfile);

% nets = {'openpose','leap','deeplabcut','unet','mdn'};
% nnets = numel(nets);
% minerr = 0;
% maxerr = 50;
% nbins = 100;
% binedges = linspace(minerr,maxerr,nbins+1);
% bincenters = (binedges(1:end-1)+binedges(2:end))/2;
% markers = {'o','s','d','^'};
%binedges(end) = inf;

freezeSpec = lObj.kickAxesModeTargetSpec;
lpos = lObj.labeledpos{freezeSpec.iMov}(:,:,freezeSpec.frm,freezeSpec.iTgt);
if freezeSpec.isrotated,
  lpos = [lpos,ones(size(lpos,1),1)]*freezeSpec.A;
  lpos = lpos(:,1:2);
end
assert(all(~isnan(lpos(:))));
ptcolors = lObj.LabelPointColors;



%% load in data
gtimdata = load(gtimagefile);
conddata = load(condinfofile);

gtdata_size = load(gtfile_trainsize);

if ~isempty(gtfile_trainsize_cpr),  
  gtdata_size = AddCPRGTData(gtdata_size,gtfile_trainsize_cpr,lObj.labeledpos,vwi);
end


annoterrdata = load(annoterrfile);
annotators = fieldnames(annoterrdata);
gtidx = annoterrdata.(annotators{1}){1}.idx2allGT;
for i = 1:numel(annotators),
  for j = 1:numel(annoterrdata.(annotators{i})),
    gtidx = intersect(gtidx,annoterrdata.(annotators{i}){j}.idx2allGT);
  end
end
for i = 1:numel(annotators),
  for j = 1:numel(annoterrdata.(annotators{i})),
    [~,order] = ismember(gtidx,annoterrdata.(annotators{i}){j}.idx2allGT);
    annoterrdata.(annotators{i}){j}.pred = annoterrdata.(annotators{i}){j}.pred(order,:,:);
    annoterrdata.(annotators{i}){j}.labels = annoterrdata.(annotators{i}){j}.labels(order,:,:);
    annoterrdata.(annotators{i}){j}.mov = annoterrdata.(annotators{i}){j}.mov(order);
    annoterrdata.(annotators{i}){j}.frm = annoterrdata.(annotators{i}){j}.frm(order);
    annoterrdata.(annotators{i}){j}.iTgt = annoterrdata.(annotators{i}){j}.iTgt(order);
    annoterrdata.(annotators{i}){j}.idx2allGT = annoterrdata.(annotators{i}){j}.idx2allGT(order);
    annoterrdata.(annotators{i}){j}.data_cond = annoterrdata.(annotators{i}){j}.data_cond(order);
    annoterrdata.(annotators{i}){j}.label_cond = annoterrdata.(annotators{i}){j}.label_cond(order);
    assert(all(annoterrdata.(annotators{i}){j}.idx2allGT == gtidx));
  end

end

fns = fieldnames(gtdata_size);
for i = 1:numel(fns),
  for j = 1:numel(gtdata_size.(fns{i})),
    gtdata_size.(fns{i}){j}.labels = gtdata_size.(fns{i}){j}.labels(gtidx,:,:);
    gtdata_size.(fns{i}){j}.pred = gtdata_size.(fns{i}){j}.pred(gtidx,:,:);
  end
end
gtimdata.ppdata.I = gtimdata.ppdata.I(gtidx,:);
gtimdata.ppdata.MD = gtimdata.ppdata.MD(gtidx,:);
gtimdata.ppdata.pGT = gtimdata.ppdata.pGT(gtidx,:);
gtimdata.ppdata.bboxes = gtimdata.ppdata.bboxes(gtidx,:);
gtimdata.tblPGT = gtimdata.tblPGT(gtidx,:);

npts = size(gtdata_size.(nets{1}){end}.labels,2);
for i = 1:numel(annotators),
  gtdata_size.(annotators{i}) = annoterrdata.(annotators{i});
end
conddata.data_cond = conddata.data_cond(gtidx,:);
conddata.label_cond = conddata.label_cond(gtidx,:);

saveannoterrdata = struct;
saveannoterrdata.inter = annoterrdata.Alice;
saveannoterrdata.intra = annoterrdata.Austin;

save AnnotErrData20190614.mat -struct saveannoterrdata

%% overlay error percentiles per part for last entry

PlotOverlayedErrorPrctiles('freezeInfo',freezeInfo,...
  'lpos',lpos,...
  'gtdata',gtdata_size,...
  'nets',nets,'legendnames',legendnames,...
  'exptype',exptype,...
  'conddata',conddata,...
  'labeltypes',labeltypes,'datatypes',datatypes,...
  'prcs',prcs,...
  'savedir',savedir,...
  'dosavefig',dosavefig);

%% plot error percentiles per part type for last entry

PlotPerLandmarkErrorPrctiles('gtdata',gtdata_size,...
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

for stati = 1:3,
  switch stati,
    case 1
      statname = 'Worst';
    case 2
      statname = 'Median';
    case 3
      statname = 'Best';
  end

  PlotFracInliers('gtdata',gtdata_size,...
    'nets',nets,'legendnames',legendnames,...
    'colors',colors,...
    'exptype',exptype,...
    'conddata',conddata,...
    'labeltypes',labeltypes,'datatypes',datatypes,...
    'statname',statname,...
    'savedir',savedir,'dosavefig',dosavefig,...
    'maxerr',maxerr);
  
  PlotSortedWorstLandmarkError('gtdata',gtdata_size,...
    'nets',nets,'legendnames',legendnames,...
    'colors',colors,...
    'exptype',exptype,...
    'conddata',conddata,...
    'labeltypes',labeltypes,'datatypes',datatypes,...
    'statname',statname,...
    'savedir',savedir,'dosavefig',dosavefig,'maxerr',maxerr);

  
end

%% plot example predictions

nexamples_random = 5;
nexamples_disagree = 5;
hfigs = PlotExamplePredictions('gtdata',gtdata_size,...
  'gtimdata',gtimdata,...
  'lObj',lObj,...
  'nets',nets,'legendnames',legendnames,...
  'exptype',exptype,...
  'ptcolors',ptcolors,...
  'conddata',conddata,'labeltypes',labeltypes,'datatypes',datatypes,...
  'dosavefig',dosavefig,'savedir',savedir,...
  'nexamples_random',nexamples_random,...
  'nexamples_disagree',nexamples_disagree,...
  'errnets',annotators,...
  'figpos',[10,10,1576,1468]);