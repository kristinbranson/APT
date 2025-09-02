function [codestr,code_as_list] = trackCodeGenBase(totrackinfo, varargin)
% Generate the case command for tracking.  Returned command uses WSL paths.  This is a static method of DLBackEndClass.

% Serial mode: 
% - movtrk is [nmov] array
% - outtrk is [nmov] array
% - trxtrk is unsupplied, or [nmov] array
% - view is a *scalar* and *must be supplied*
% - croproi is unsupplied, or [xlo1 xhi1 ylo1 yhi1 xlo2 ... yhi_nmov] or row vec of [4*nmov]
% - model_file is unsupplied, or [1] cellstr, or [nmov] cellstr      

stages = totrackinfo.stages;
views = totrackinfo.views;
nstages = numel(stages);
nviews = numel(views);

% construct and concatenate multiple commands if tracking both
% multiple views and multiple movies
if nviews > 1 && totrackinfo.nmovies > 1 && ~totrackinfo.islistjob,
  code_as_list = cell(totrackinfo.nmovies,1);
  for i = 1:totrackinfo.nmovies,
    tticurr = totrackinfo.selectSubset('movie',i);
    tticurr.setJobid(totrackinfo.getJobid);
    [codestrcurr,code_as_list{i}] = APTInterf.trackCodeGenBase(tticurr,varargin{:});
    if i == 1,
      codestr = codestrcurr;
    else
      codestr = [codestr,' && ',codestrcurr]; %#ok<AGROW> 
    end
  end
  return;
end

% Process optional arguments
dmc = totrackinfo.trainDMC ;
[track_type,...
 aptroot,...
 ignore_local] = ...
  myparse(varargin,...
          'track_type','track',...  % track_type should be one of {'track', 'link', 'detect'}
          'aptroot',APT.Root,...
          'ignore_local',[]... % whether to remove local python modules from the path
          );

% aptintrf = APTInterf.aptInterfacePath(aptroot);
aptintrfNative = fullfile(aptroot, 'deepnet/APT_interface.py') ;  % this is a native path
aptintrf = wsl_path_from_native(aptintrfNative) ;  % The path to APT_interface.py, as a WSL path.


modelChainID = DeepModelChainOnDisk.getCheckSingle(dmc.getModelChainID());
nativeTrainConfig = DeepModelChainOnDisk.getCheckSingle(dmc.trainConfigLnx());  % native path
trainConfig = wsl_path_from_native(nativeTrainConfig) ;
nativeCacheRootDir = dmc.rootDir ;  % native path
cacheRootDir = wsl_path_from_native(nativeCacheRootDir) ;      

stage2models = cell(1,nstages);
for istage = 1:nstages,
  stage = stages(istage);
  % cell of length nviews or empty
  nativeModelPath = dmc.trainCurrModelSuffixlessLnx('stage',stage) ;  % native path
  modelPath = wsl_path_from_native(nativeModelPath) ;
  stage2models{istage} = modelPath ;
  assert(numel(stage2models{istage}) == nviews);
end

% one net type per stage
stage2netType = cell(1,nstages);
for istage = 1:nstages,
  stage = stages(istage);
  stage2netType{istage} = char(DeepModelChainOnDisk.getCheckSingle(dmc.getNetType('stage',stage)));
end

%netType = char(DeepModelChainOnDisk.getCheckSingle(fileinfo.netType)); % for 2-stage, this is the stage2 nettype
%netMode = fileinfo.netMode; % " netmode
% either char or [nviewx1] cellstr; or [nmov] in "serial mode" (see below)
%movtrk = fileinfo.movtrk; 
% save as movtrk, except for 2 stage, this will be [nviewx2] or [nmovx2]
%outtrk = fileinfo.outtrk; 
nativeConfigFile = totrackinfo.trackconfigfile;  % native path
configfile = wsl_path_from_native(nativeConfigFile) ;

% this should happen outside
%       if updateWinPaths2LnxContainer
%         fcnPathUpdate = @(x)DeepTracker.codeGenPathUpdateWin2LnxContainer(x,lnxContainerMntLoc);
%         aptintrf = fcnPathUpdate(aptintrf);
% 
%         movies2track = cellfun(fcnPathUpdate,movies2track,'uni',0);
%         outputtrkfiles = cellfun(fcnPathUpdate,outputtrkfiles,'uni',0);
%         if tftrx
%           trxtrk = cellfun(fcnPathUpdate,trxtrk,'uni',0);
%         end
%         if tfmodel
%           model_file = cellfun(fcnPathUpdate,model_file,'uni',0);
%         end
%         if tflog
%           log_file = fcnPathUpdate(log_file);
%         end
%         cacheRootDir = fcnPathUpdate(cacheRootDir);
%         errfile = fcnPathUpdate(errfile);
%         trainConfig = fcnPathUpdate(trainConfig);
%         configfile = fcnPathUpdate(configfile);
%       end      

% Get prefix the sets torch home dir
torchhome_native = APT.gettorchhomepath() ;
wsl_torch_home = wsl_path_from_native(torchhome_native) ;      
torchhomeprefix = sprintf('TORCH_HOME=%s', escape_string_for_bash(wsl_torch_home)) ;

code_as_list = { ...
  torchhomeprefix ...
  'python' escape_string_for_bash(aptintrf) ...
  escape_string_for_bash(trainConfig) ...
  '-name' modelChainID ...
  '-err_file' escape_string_for_bash(wsl_path_from_native(totrackinfo.errfile)) ...
  '-log_file' escape_string_for_bash(wsl_path_from_native(totrackinfo.logfile)) ...
  };
if dmc.isMultiStageTracker,
  code_as_list = [code_as_list {'-stage' 'multi'}];
end
if dmc.isMultiViewTracker,
  if nviews == 1,
    code_as_list = [code_as_list {'-view', num2str(views)}];
  end
end
code_as_list = [code_as_list {'-type', stage2netType{1}} ...
  {'-model_files'}, escape_cellstring_for_bash(wsl_path_from_native(stage2models{1}))];
if nstages > 1,
  assert(nstages==2);
  code_as_list = [code_as_list {'-type2', stage2netType{2}} ...
    {'-model_files2'}, escape_cellstring_for_bash(wsl_path_from_native(stage2models{2})) ...
    {'-name2'} totrackinfo.trainDMC.getModelChainID('stage',2)];
end

if ~isempty(ignore_local),
  code_as_list = [code_as_list, {'-ignore_local',num2str(ignore_local)}];
end
code_as_list = [code_as_list {'-cache' escape_string_for_bash(cacheRootDir)}];

code_as_list = [code_as_list {'track'}];

code_as_list = [code_as_list {'-config_file' escape_string_for_bash(configfile)}];

switch track_type
  case 'link'
    code_as_list = [code_as_list {'-track_type only_link'}]; 
  case 'detect'
    code_as_list = [code_as_list {'-track_type only_predict'}]; 
  case 'track'
    % do nothing
  otherwise
    error('track_type must be either ''track'', ''link'', or ''detect''') ;
end

[movidx,frm0,frm1,trxids,nextra] = totrackinfo.getIntervals();

% output is the final stage trk file
trkfiles = totrackinfo.getTrkFiles('stage',stages(end));

% convert to frms, trxids
if ~isempty(totrackinfo.listfile)
  code_as_list = [code_as_list {'-list_file' escape_string_for_bash(wsl_path_from_native(totrackinfo.listfile))}];
  code_as_list = [code_as_list {'-out'} escape_cellstring_for_bash(wsl_path_from_native(totrackinfo.listoutfiles))];
else
  tf = escape_cellstring_for_bash(wsl_path_from_native(trkfiles(movidx,:,:)));
  code_as_list = [code_as_list {'-out'} tf(:)'];
  if sum(nextra) > 0,
    warningNoTrace('Tracking %d already-tracked frames',sum(nextra));
  end
  nativeMovFiles = totrackinfo.getMovfiles('movie',movidx) ;  % native file paths
  movFiles = wsl_path_from_native(nativeMovFiles) ;
  code_as_list = [code_as_list {'-mov' space_out(escape_cellstring_for_bash(movFiles))}];
  if ~all(frm0==1 & frm1==-1),
    code_as_list = [code_as_list {'-start_frame' num2str(frm0(:)') '-end_frame' num2str(frm1(:)')}];
  end
  if totrackinfo.hasTrxfiles,
    nativeTrxFiles = totrackinfo.getTrxFiles('movie',movidx) ;
    trxFiles = wsl_path_from_native(nativeTrxFiles) ;
    code_as_list = [code_as_list {'-trx' space_out(escape_cellstring_for_bash(trxFiles))}];
  elseif nstages > 1,
    nativeTrxFiles = totrackinfo.getTrkFiles('stage',1) ;
    trxFiles = wsl_path_from_native(nativeTrxFiles) ;
    code_as_list = [code_as_list {'-trx' space_out(escape_cellstring_for_bash(trxFiles))}];
  end
%         if totrackinfo.hasTrxids,
%           for i = 1:numel(totrackinfo.getTrxids('movie',movidx)),
%             code = [code {'-trx_ids' num2str(trxids{i}(:)')}]; %#ok<AGROW>
  if ~all(cellfun(@isempty,trxids))
     for i = 1:numel(trxids)
        code_as_list = [code_as_list {'-trx_ids' num2str(trxids{i}(:)')}]; %#ok<AGROW>
     end
  end
end
if totrackinfo.hasCroprois && ~totrackinfo.islistjob,
  croproi = totrackinfo.getCroprois('movie',movidx);
  if iscell(croproi)
    croproi = cell2mat(croproi');
  end
  croproi = round(croproi);
  if ~isempty(croproi) && ~all(any(isnan(croproi),2),1),
    croproirowvec = croproi';
    croproirowvec = croproirowvec(:)'; % [xlovw1 xhivw1 ylovw1 yhivw1 xlovw2 ...] OR [xlomov1 xhimov1 ylomov1 yhimov1 xlomov2 ...] in serialmode
    code_as_list = [code_as_list {'-crop_loc' num2str(croproirowvec)}];
  end
end

codestr = space_out(code_as_list);
end  % function
