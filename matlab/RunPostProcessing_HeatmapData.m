function [postdata,ppobj] = RunPostProcessing_HeatmapData(hmdir,varargin)

ncores = feature('numcores');
if isdeployed,
  fprintf('Starting RunPostProcessing_HeatmapData, ncores = %d...\n',ncores);
end

savefile = '';
trxfile = '';
nframes = [];
lblfile = '';
expname = '';
expdir = '';
nviews = 1;
npts = [];
d = 2;
scales = 1;
heatmap_lowthresh = .1;
heatmap_highthresh = .5;
heatmap_nsamples = 25;
heatmap_sample_algorithm = 'gmm';
frames = [];

if ncores == 1,
  ncores = 0;
end

usegeometricerror = true;
viterbi_poslambda = .01;
viterbi_misscost = 5;
viterbi_dampen = .25;

kde_sigma_px = 5;
hmtype = 'jpg';

targets = [];
startframe = [];
endframe = [];

algorithms = {'maxdensity','median','viterbi'};

[paramsfile,rest] = myparse_nocheck(varargin,'paramsfile','');
if ~isempty(paramsfile),
  load(paramsfile);
end

if isdeployed,
  rest = DeployedChar2NumberArgs(rest,'nframes','nviews','npts','targets','startframe','endframe','scales',...
    'heatmap_lowthresh','heatmap_highthresh','usegeometricerror',...
    'viterbi_poslambda','viterbi_misscost','viterbi_dampen','ncores');
end

[trxfile,N,lblfile,expname,expdir,nviews,npts,targets,pts2run,startframe,endframe,scales,...
  heatmap_lowthresh,heatmap_highthresh,heatmap_nsamples,heatmap_sample_algorithm,...
  usegeometricerror,...
  viterbi_poslambda,viterbi_misscost,viterbi_dampen,savefile,algorithms,hmtype,...
  frames,ncores] = ...
  myparse(rest,'trxfile',trxfile,... 
  'nframes',nframes,...
  'lblfile',lblfile,'expname',expname,'expdir',expdir,...
  'nviews',nviews,'npts',npts,'targets',targets,...
  'pts2run',[],...
  'startframe',startframe,'endframe',endframe,... % raw/movie frames marking interval to consider (inclusive). if trx, then these may be modified based on traj availability etc
  'scales',scales,...
  'heatmap_lowthresh',heatmap_lowthresh,'heatmap_highthresh',heatmap_highthresh,...
  'heatmap_nsamples',heatmap_nsamples,'heatmap_sample_algorithm',heatmap_sample_algorithm,...
  'usegeometricerror',usegeometricerror,...
  'viterbi_poslambda',viterbi_poslambda,'viterbi_misscost',viterbi_misscost,...
  'viterbi_dampen',viterbi_dampen,...
  'savefile',savefile,'algorithms',algorithms,...
  'hmtype',hmtype,...
  'frames',frames,... % raw/movie frames to consider; must be within interval [startframe,endframe]; risky with trx
  'ncores',ncores);

if ~iscell(algorithms),
  algorithms = {algorithms};
end

if (viterbi_misscost <= 0),
  viterbi_misscost = inf;
end

%% figure out number of frames, trxfile from lblfile

if ~isempty(lblfile),

  ld = load(lblfile,'-mat');
  
  if isempty(expname),
    if ~isempty(expdir),
      [~,expname] = fileparts(expdir);
    else
      [~,fname] = fileparts(hmdir);
      expname = fname(1:end-5);
    end
  end
  
  nviews = ld.cfg.NumViews;
  npts = ld.cfg.NumLabelPoints;

  macrofns = fieldnames(ld.projMacros);
  moviefile = '';
  moviei = [];
  for i = 1:numel(ld.movieFilesAll),
    j = strfind(ld.movieFilesAll{i},expname);
    if ~isempty(j),
      moviefile = ld.movieFilesAll{i};
      for j = 1:numel(macrofns),
        moviefile = strrep(moviefile,['$',macrofns{j}],ld.projMacros.(macrofns{j}));
      end
      moviei = i;
      break;
    end
  end

  if ~isempty(moviei),
    N = ld.movieInfoAll{moviei}.nframes;
  end
  
  if isempty(expdir),
    expdir = fileparts(moviefile);
  end
  
  if isempty(trxfile) && isfield(ld,'trxFilesAll') && numel(ld.trxFilesAll) >= moviei,
    trxfile = ld.trxFilesAll{moviei};
    trxfile = strrep(trxfile,'$movdir',expdir);
  end
  
end

istrx = false;
ntargets = 1;
if ~isempty(trxfile),
  
  istrx = true;
  td = load(trxfile);
  ntargets = numel(td.trx);
  
  if isempty(N),
    N = max([td.trx.endframe]);
  end
  
end

if isempty(targets),
  targets = 1:ntargets;
end

if isempty(N),
  error('Could not figure out number of frames');
end
if isempty(npts),
  error('Could not figure out number of landmarks');
end

if istrx && nviews > 1,
  error('Not implemented');
end

startframe0 = startframe;
endframe0 = endframe;
N0 = N;

if numel(targets) > 1,
  allppobj = cell(1,numel(targets));
  allpostdata = cell(1,numel(targets));
end
  
for flyi = 1:numel(targets),
  fly = targets(flyi);

  if isempty(startframe0),
    if istrx,
      startframe = td.trx(fly).firstframe;
    else
      startframe = 1;
    end
  elseif istrx,
    assert(startframe >= td.trx(fly).firstframe);
  end
  if isempty(endframe0),
    if istrx,
      endframe = td.trx(fly).endframe;
    else
      endframe = N0;
    end
  elseif istrx,
    assert(endframe <= td.trx(fly).endframe);
  end
  N = endframe-startframe+1;
  
  % startframe, endframe, N now set for this fly. startframe and endframe
  % are real/raw/movie frame numbers

  % Generate trxcurr, cropped trx for this fly
  if istrx,
    trxcurr = td.trx(fly);
    fns = fieldnames(trxcurr);
    i0 = startframe-trxcurr.firstframe+1;
    i1 = startframe+N-1-trxcurr.firstframe+1;
    for i = 1:numel(fns),
      if numel(trxcurr.(fns{i})) == 1 || ischar(trxcurr.(fns{i})),
        continue;
      end
      l = trxcurr.nframes-numel(trxcurr.(fns{i})); % AL: guess this is just for dt
      trxcurr.(fns{i}) = trxcurr.(fns{i})(i0:i1-l); 
    end
    
    % cropped trx, metadata fields no longer correct
    trxcurr = rmfield(trxcurr,intersect(fns,{'off' 'firstframe' 'nframes' 'endframe'}));
    frm = (startframe:endframe)';
    tblMFT = table(frm);    
  else
    trxcurr = [];
  end
  
%   if istrx,
%     trxfirstframe = td.trx(fly).firstframe;
%   else
%     trxfirstframe = 1;
%   end
  
  readscorefuns = cell(npts,nviews);
  for viewi = 1:nviews,
    for pti = 1:npts,
      readscorefuns{pti,viewi} = get_readscore_fcn(hmdir,fly,pti,...
        'hmtype',hmtype,'firstframe',startframe);
    end
  end
  
  % make sure all heatmap files exist
  

  if size(scales,1) == 1,
    scales = repmat(scales,[nviews,1]);
  end
  if size(scales,2) == 1,
    scales = repmat(scales,[1,d]);
  end
  
  ppobj = PostProcess();
  if ~isempty(pts2run)
    ppobj.pts2run = pts2run;
  end
  ppobj.SetNCores(ncores);
  ppobj.SetUseGeometricError(usegeometricerror);
  ppobj.SetHeatmapData(readscorefuns,N,scales,trxcurr,frames-startframe+1);
  if istrx,
    ppobj.SetMFTInfo(tblMFT);
  end
  ppobj.SetKDESigma(kde_sigma_px);
  ppobj.SetHeatmapLowThresh(heatmap_lowthresh);
  ppobj.SetHeatmapHighThresh(heatmap_highthresh);
  ppobj.SetHeatmapNumSamples(heatmap_nsamples);
  ppobj.SetHeatmapSampleAlg(heatmap_sample_algorithm);
  ppobj.SetViterbiParams('poslambda',viterbi_poslambda,...
    'dampen',viterbi_dampen,'misscost',viterbi_misscost);
  
  assignin('base','ppobj',ppobj);
  fprintf(1,'assigned ppobj to base WS\n');
  
  for algi = 1:numel(algorithms),
    fprintf('Running %s...\n',algorithms{algi});
    starttime = tic;
    ppobj.SetAlgorithm(algorithms{algi});
    ppobj.run();
    fprintf('Time to run %s: %f\n',ppobj.GetPostDataAlgName(),toc(starttime));
  end
  
  postdata = ppobj.GetAllPostData();
  if numel(targets) > 1,
    allpostdata{flyi} = postdata;
    allppobj{flyi} = ppobj;
  end
  
end

if numel(targets) > 1,
  postdata = allpostdata;
  ppobj = allppobj;
end

if ~isempty(savefile),
  timestamp = now; %#ok<NASGU>
  save(savefile,'postdata','ppobj','hmdir','timestamp','targets');
end