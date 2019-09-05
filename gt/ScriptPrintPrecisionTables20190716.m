%%

%gtfile = '/nrs/branson/mayank/apt_cache/alice_view0_time.mat';

allexptypes = {'FlyBubble','SHView0','SHView1','SH3D','RFView0','RFView1','RF3D','Roian','BSView0x','BSView1x','BSView2x','Larva','FlyBubbleMDNvsDLC'};

exptype = 'RFView1';
cprdir = '/groups/branson/bransonlab/apt/experiments/res/cprgt20190407';
codedir = fileparts(mfilename('fullpath'));
savedir = '/groups/branson/bransonlab/apt/experiments/res/gt/20190523';

if ~exist(savedir,'dir'),
  mkdir(savedir);
end
dosavefig = true;

Ptbls = cell(1,numel(allexptypes));

for exptypei = 1:numel(allexptypes),
exptype = allexptypes{exptypei};

disp(exptype);

%% parameters

gtimdata = [];

threshmethod = 'prctileerr';
ScriptGTDataSetParameters;

%% load in data

gtdata_size = load(gtfile_trainsize);
if isempty(gtfile_traintime),
  gtdata_time = [];
else
  gtdata_time = load(gtfile_traintime);
end
if isempty(annoterrfile),
  annoterrdata = [];
else
  annoterrdata = load(annoterrfile);
end

%% images for overlaying percentile errors 

try
  lObj = load(lblfile,'-mat');
catch
  tmploc = tempname;
  mkdir(tmploc);
  untar(lblfile,tmploc);
  lObj = load(fullfile(tmploc,'label_file.lbl'),'-mat');
  unix(sprintf('rm -r %s',tmploc));
end
ptcolors = lObj.cfg.LabelPointsPlot.Colors;
lObj.labeledpos = cellfun(@SparseLabelArray.full,lObj.labeledpos,'uni',0);

% nets = {'openpose','leap','deeplabcut','unet','mdn'};
% nnets = numel(nets);
% minerr = 0;
% maxerr = 50;
% nbins = 100;
% binedges = linspace(minerr,maxerr,nbins+1);
% bincenters = (binedges(1:end-1)+binedges(2:end))/2;
% markers = {'o','s','d','^'};
%binedges(end) = inf;


%% load in cpr data

if ~isempty(gtfile_trainsize_cpr),  
  gtdata_size = AddCPRGTData(gtdata_size,gtfile_trainsize_cpr,lObj.labeledpos,vwi);
end

if ~isempty(gtfile_traintime) && ~isempty(gtfile_traintime_cpr),
  gtdata_time = AddCPRGTData(gtdata_time,gtfile_trainsize_cpr,lObj.labeledpos,vwi);
end

%% which is the main gtdata file

if ~isempty(gtdata_time),
  gtdata = gtdata_time;
  fns = setdiff(fieldnames(gtdata_size),fieldnames(gtdata_time));
  for i = 1:numel(fns),
    gtdata.(fns{i}) = gtdata_size.(fns{i});
  end
else
  gtdata = gtdata_size;
end
fns = fieldnames(gtdata);
for i = 1:numel(fns),
  gtdata.(fns{i}) = gtdata.(fns{i})(end);
end

%% add in 3d reconstructions

%% compute kappa for OKS computation if there is annotation error data

if ~isempty(annoterrdata),

  dormoutliers = size(annoterrdata.intra{end}.pred,1) > 200;
  kappadistname = 'gamma2';
  ComputeOKSPrecisionTable(gtdata_size,...
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
  ComputeOKSPrecisionTable(gtdata_size,...
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

Ptbls{exptypei} = ComputePixelPrecisionTable(gtdata_size,...
  'nets',nets,'legendnames',legendnames,...
  'exptype',exptype,...
  'conddata',conddata,...
  'labeltypes',labeltypes,'datatypes',datatypes,...
  'savedir',savedir,...
  'dosavefig',dosavefig,...
  'pttypes',pttypes,...
  'annoterrdata',annoterrdata,...
  'threshmethod',threshmethod);

%%

end

%% print summary results

savename = fullfile(savedir,'appxsummary.tex');
PrintAllExpPrecisionTable(Ptbls,allexptypes,savename,'labelallonly',true,'dataallonly',true,'dorotateheader',false);

for exptypei = 1:numel(allexptypes),
  exptype = allexptypes{exptypei};
  [ndatatypes,nlabeltypes] = size(Ptbls{exptypei});
  if ndatatypes>1,
    savename = fullfile(savedir,sprintf('appx_%s_datatype_summary.tex',exptype));
    PrintAllExpPrecisionTable(Ptbls(exptypei),allexptypes(exptypei),savename,'dataallonly',false,'labelallonly',true,'dorotateheader',false);
  end
  if nlabeltypes>1,
    savename = fullfile(savedir,sprintf('appx_%s_labeltype_summary.tex',exptype));
    PrintAllExpPrecisionTable(Ptbls(exptypei),allexptypes(exptypei),savename,'dataallonly',true,'labelallonly',false,'dorotateheader',false);
  end
end