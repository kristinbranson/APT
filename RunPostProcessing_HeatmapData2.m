function allppobj = RunPostProcessing_HeatmapData2(hmdir,varargin)
% Run single-view heatmap postproc

ncores = feature('numcores');
if isdeployed,
  fprintf('Starting RunPostProcessing_HeatmapData, ncores = %d...\n',ncores);
end

if ncores == 1,
  ncores = 0;
end

if isdeployed,
  varargin = DeployedChar2NumberArgs(varargin,...
    'lblfileImov',...
    'nviews','npts','targets','startframe','endframe','scales',...
    'heatmap_lowthresh','heatmap_highthresh','heatmap_nsamples','usegeometricerror',...
    'viterbi_poslambda','viterbi_misscost','viterbi_dampen','viterbi_grid_acradius','ncores');
end

d = 2;
[rootdir,lblfile,lblfilehmdirs,lblfileImov,...
  nviews,npts,nfrmsmov,trxfile,targets,pts2run,startframe,endframe,scales,...
  heatmap_lowthresh,heatmap_highthresh,heatmap_nsamples,heatmap_sample_algorithm,...
  usegeometricerror,kde_sigma_px,...
  viterbi_poslambda,viterbi_misscost,viterbi_dampen,viterbi_grid_acradius,...
  savefile,algorithms,hmtype,...
  frames,ncores,paramfiles] = ...
  myparse(varargin,...
  'rootdir','',... % (opt) root dir for lblfile, paramfiles, savefile
  'lblfile','',... % lblfile can be used to specify {nviews,npts,nfrmsmov,trxfile,hmdir}
  'lblfilehmdirs',[],... % specify when lblfile specified
  'lblfileImov',[],... % [1] movie index into lblfile/lblfilehmdirs %'hmdir',[],...
  'nviews',1,...
  'npts',[],...
  'nfrmsmov',[],...
  'trxfile','',... 
  'targets',[],...
  'pts2run',[],...
  'startframe',[],... % raw/movie frames marking interval to consider (inclusive). if trx, then these may be modified based on traj availability etc
  'endframe',[],...  % (cont) either start/endframe OR lblfile must be specified
  'scales',1,...
  'heatmap_lowthresh',.1,...
  'heatmap_highthresh',.5,...
  'heatmap_nsamples',25,...
  'heatmap_sample_algorithm','gmm',...
  'usegeometricerror',true,...
  'kde_sigma_px',5,...
  'viterbi_poslambda',.01,...
  'viterbi_misscost',5,...
  'viterbi_dampen',.25,...
  'viterbi_grid_acradius',12,...
  'savefile','',...
  'algorithms',{'viterbi_grid','maxdensity','median','viterbi'},...
  'hmtype','jpg',...
  'frames',[],... % raw/movie frames to consider; must be within interval [startframe,endframe]; risky with trx
  'ncores',ncores,...
  'paramfiles',[]... % files that can overlay/override these options
  );

if ~isempty(paramfiles)  
  if ischar(paramfiles)
    paramfiles = regexp(paramfiles,'#','split');
  end
  assert(iscellstr(paramfiles));
  nparamfiles = numel(paramfiles);
  for iparamfile=1:nparamfiles
    pfile = paramfiles{iparamfile};
    pfile = fullfile(rootdir,pfile);
    
    [~,~,pfileE] = fileparts(pfile);
    switch pfileE
      case '.mat'
        fprintf('Loading paramfile %s...\n',pfile);
        prmvars = load(pfile);
        vars = fieldnames(prmvars);
        for v=vars(:)',v=v{1};
          if iscell(prmvars.(v))
            matstr = sprintf('%s cell array',mat2str(size(prmvars.(v))));
          else
            matstr = mat2str(prmvars.(v));
          end
          fprintf('  %s <- %s\n',v,matstr);
          if exist(v,'var')==0
            warningNoTrace('paramfile variable ''%s'' is not in scope.',v);
          end
        end
        
        load(pfile);
      case '.m'
        fprintf('Applying patchfile %s...\n',pfile);

        pchs = readtxtfile(pfile,'discardemptylines',true);
        npch = numel(pchs);
        fprintf(1,'... read patch file %s. %d patches.\n',pfile,npch);
        for ipch=1:npch
          pch = pchs{ipch};
          pch = [pch ';']; %#ok<AGROW>
          tmp = strsplit(pch,'=');
          pchlhs = strtrim(tmp{1});
          fprintf(1,'  patch %d: %s\n',ipch,pch);
          fprintf(1,'  orig (%s): %s\n',pchlhs,evalc(pchlhs));
          eval(pch);
          fprintf(1,'  new (%s): %s\n',pchlhs,evalc(pchlhs));
        end
    end
  end
end

if ~iscell(algorithms),
  algorithms = {algorithms};
end

if (viterbi_misscost <= 0),
  viterbi_misscost = inf;
end

if ~isempty(lblfile)
  lblfile = fullfile(rootdir,lblfile);
  [nviews,npts,nfrmsmov,trxfile] = readStuffFromLbl(lblfile,lblfileImov);
  assert(~isempty(lblfilehmdirs));
  assert(nviews==1);
  hmdir = lblfilehmdirs{lblfileImov};
  fprintf('lblfile %s imov %d\n',lblfile,lblfileImov);
else
  assert(~isempty(npts));
  assert(~isempty(hmdir));
  if isempty(nfrmsmov)
    fprintf(1,'Finding first/last frames avail in hmdir: %s\n',hmdir);
    [startframe,endframe] = HeatmapReader.findFirstLastFrameHmapDir(hmdir);
  end
end

if isempty(startframe)
  startframe = 1;
end
if isempty(endframe)
  endframe = nfrmsmov;
end

fprintf('hmdir %s\n',hmdir);
fprintf('nviews %d npts %d startfr %d endfr %d\n',nviews,npts,startframe,endframe);
fprintf('trxfile %s\n',trxfile);

istrx = ~isempty(trxfile);
if istrx
  td = load(trxfile); % assumed fullpath
  ntargetstot = numel(td.trx);
else
  ntargetstot = 1;
end

if isempty(targets),
  targets = 1:ntargetstot;
end
assert(all(ismember(targets(:),(1:ntargetstot)')));
ntargets = numel(targets);

if isempty(pts2run)
  pts2run = 1:npts;
end

if size(scales,1) == 1,
  scales = repmat(scales,[nviews,1]);
end
if size(scales,2) == 1,
  scales = repmat(scales,[1,d]);
end

startframe0 = startframe;
endframe0 = endframe;
allppobj = cell(1,ntargets);
%allpostdata = cell(1,ntargets);
  
for flyi = 1:ntargets
  fly = targets(flyi);
  
  if istrx
    startframe = max(startframe0,td.trx(fly).firstframe);
    if startframe~=startframe0
      fprintf('Fly %d: startframe modified from %d->%d.\n',startframe0,startframe);
    end
    
    endframe = min(endframe0,td.trx(fly).endframe);
    if endframe~=endframe0
      fprintf('Fly %d: end modified from %d->%d.\n',endframe0,endframe);
    end    
  else
    startframe = startframe0;
    endframe = endframe0;
  end  
  N = endframe-startframe+1;
  
  % startframe, endframe, N now set for this fly. startframe and endframe
  % are real/raw/movie frame numbers

  % Generate trxcurr: cropped trx for this fly
  if istrx,
    trxcurr = td.trx(fly);
    fns = fieldnames(trxcurr);
    i0 = startframe-trxcurr.firstframe+1;
    i1 = startframe+N-1-trxcurr.firstframe+1;
    for i = 1:numel(fns),
      if numel(trxcurr.(fns{i}))==1 || ischar(trxcurr.(fns{i})),
        continue;
      end
      l = trxcurr.nframes-numel(trxcurr.(fns{i})); % AL: guess this is just for dt
      trxcurr.(fns{i}) = trxcurr.(fns{i})(i0:i1-l); 
    end
    
    % cropped trx, metadata fields no longer correct
    trxcurr = rmfield(trxcurr,...
                intersect(fns,{'off' 'firstframe' 'nframes' 'endframe'}));
    frm = (startframe:endframe)';
    tblMFT = table(frm);
  else
    trxcurr = [];
  end
  
  readscorefuns = cell(npts,nviews);
  for viewi = 1:nviews,
    for pti = 1:npts,
      readscorefuns{pti,viewi} = get_readscore_fcn(hmdir,fly,pti,...
        'hmtype',hmtype,'firstframe',startframe);
    end
  end
  
  ppobj = PostProcess();
  if ~isempty(pts2run)
    ppobj.pts2run = pts2run;
  end
  ppobj.SetNCores(ncores);
  ppobj.SetUseGeometricError(usegeometricerror);
  % if frames is empty, last arg will be empty
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
    'dampen',viterbi_dampen,'misscost',viterbi_misscost,...
    'grid_acradius',viterbi_grid_acradius);
  
  %assignin('base','ppobj',ppobj);
  %fprintf(1,'assigned ppobj to base WS\n');
  
  for algi = 1:numel(algorithms),
    fprintf('Running %s...\n',algorithms{algi});
    starttime = tic;
    ppobj.SetAlgorithm(algorithms{algi});
    ppobj.run();
    fprintf('Time to run %s: %f\n',ppobj.GetPostDataAlgName(),toc(starttime));
  end
  
  % Add 5-pt M.A.
  filt = [-3 12 17 12 -3]/35;
  ppobj.postdata.maxdensity_indep_ma5.x = ...
                      nan(size(ppobj.postdata.maxdensity_indep.x));
  for ipt=1:ppobj.npts
    for d=1:2
      ppobj.postdata.maxdensity_indep_ma5.x(:,ipt,d) = ...
        conv(ppobj.postdata.maxdensity_indep.x(:,ipt,d),filt,'same');
    end
  end
  fprintf('Ran 5 pt m.a.\n');
  
  %postdata = ppobj.GetAllPostData();
  %allpostdata{flyi} = postdata;
  allppobj{flyi} = ppobj;
end

if ~isempty(savefile),
  savefile = fullfile(rootdir,savefile);
  timestamp = now; %#ok<NASGU>
  save(savefile,'allppobj','hmdir','timestamp','targets');
end

function [nviews,npts,nfrms,trxfile,moviefile] = readStuffFromLbl(lblfile,imov)

ld = load(lblfile,'-mat');

nviews = ld.cfg.NumViews;
npts = ld.cfg.NumLabelPoints;
nfrms = ld.movieInfoAll{imov,1}.nframes;

assert(nviews==1); 
moviefile = ...
  FSPath.fullyLocalizeStandardizeChar(ld.movieFilesAll{imov,1},ld.projMacros);
trxfile = Labeler.trxFilesLocalize(ld.trxFilesAll{imov},moviefile);
