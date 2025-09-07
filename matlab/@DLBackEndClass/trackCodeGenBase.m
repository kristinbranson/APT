function command = trackCodeGenBase(totrackinfo, varargin)
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
  for i = 1:totrackinfo.nmovies,
    tticurr = totrackinfo.selectSubset('movie',i);
    tticurr.setJobid(totrackinfo.getJobid);
    commandForThisMovie = DLBackEndClass.trackCodeGenBase(tticurr,varargin{:});
    if i == 1,
      command = commandForThisMovie;
    else
      command = apt.ShellCommmand.cat(command, '&&', commandForThisMovie);
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
aptInterfaceDotPyNativePath = apt.MetaPath(fullfile(aptroot, 'deepnet/APT_interface.py'), 'native', 'source') ;  % this is a native path, as a char
aptInterfaceDotPyWslPath = aptInterfaceDotPyNativePath.asWsl() ;  % The path to APT_interface.py, as a WSL path.


modelChainID = DeepModelChainOnDisk.getCheckSingle(dmc.getModelChainID());
nativeTrainConfig = DeepModelChainOnDisk.getCheckSingle(dmc.trainConfigLnx());  % native path, as char
trainConfigFileNativePath = apt.MetaPath(nativeTrainConfig, 'native', 'cache');
trainConfigFileWslPath = trainConfigFileNativePath.asWsl();
nativeCacheRootDir = dmc.rootDir ;  % native path, as char
cacheRootDirNativePath = apt.MetaPath(nativeCacheRootDir, 'native', 'cache');
cacheRootDirWslPath = cacheRootDirNativePath.asWsl();      

stage2models = cell(1,nstages);
for istage = 1:nstages,
  stage = stages(istage);
  % cell of length nviews or empty
  nativeModelPath = dmc.trainCurrModelSuffixlessLnx('stage',stage) ;  % native path, as char
  modelPathNative = cellfun(@(x) apt.MetaPath(x, 'native', 'cache'), nativeModelPath, 'UniformOutput', false);
  modelPathWsl = cellfun(@(x) x.asWsl(), modelPathNative, 'UniformOutput', false);
  stage2models{istage} = modelPathWsl ;
  assert(numel(stage2models{istage}) == nviews);
end

% one net type per stage
stage2netType = cell(1,nstages);
for istage = 1:nstages,
  stage = stages(istage);
  stage2netType{istage} = char(DeepModelChainOnDisk.getCheckSingle(dmc.getNetType('stage',stage)));
end

nativeConfigFile = totrackinfo.trackconfigfile;  % native path, as char
configFileNativePath = apt.MetaPath(nativeConfigFile, 'native', 'cache');
configFileWslPath = configFileNativePath.asWsl();

% Get prefix the sets torch home dir
torchHomePathNativeRaw = APT.gettorchhomepath() ;
torchHomePathNative = apt.MetaPath(torchHomePathNativeRaw, 'native', 'cache');
torchHomePathWsl = torchHomePathNative.asWsl();      
torchHomePrefix = apt.ShellVariableAssignment('TORCH_HOME', torchHomePathWsl);

errFileNativePath = apt.MetaPath(totrackinfo.errfile, 'native', 'cache');
errFileWslPath = errFileNativePath.asWsl();
logFileNativePath = apt.MetaPath(totrackinfo.logfile, 'native', 'cache');
logFileWslPath = logFileNativePath.asWsl();

command0 = apt.ShellCommand(...
  { torchHomePrefix, ...
    'python', ...
    aptInterfaceDotPyWslPath, ...
    trainConfigFileWslPath, ...
    '-name', modelChainID, ...
    '-err_file', errFileWslPath, ...
    '-log_file', logFileWslPath, ...
    }, ...
  apt.PathLocale.wsl, ...
  apt.Platform.posix);
if dmc.isMultiStageTracker,
  command1 = command0.append('-stage', 'multi');
else
  command1 = command0;
end

if dmc.isMultiViewTracker,
  if nviews == 1,
    command2 = command1.append('-view', num2str(views));
  else
    command2 = command1;
  end
else
  command2 = command1;
end
command3 = command2.append('-type', stage2netType{1});
command4 = apt.ShellCommand.cat(command3.append('-model_files'), stage2models{1}{:});
if nstages > 1,
  assert(nstages==2);
  command5 = command4.append('-type2', stage2netType{2});
  command6 = apt.ShellCommand.cat(command5.append('-model_files2'), stage2models{2}{:});
  command7 = command6.append('-name2', totrackinfo.trainDMC.getModelChainID('stage',2));
else
  command7 = command4;
end

if ~isempty(ignore_local),
  command8 = command7.append('-ignore_local', num2str(ignore_local));
else
  command8 = command7;
end

command9 = command8.append('-cache', cacheRootDirWslPath);
command10 = command9.append('track');
command11 = command10.append('-config_file', configFileWslPath);

switch track_type
  case 'link'
    command12 = command11.append('-track_type', 'only_link'); 
  case 'detect'
    command12 = command11.append('-track_type', 'only_predict'); 
  case 'track'
    command12 = command11;
  otherwise
    error('track_type must be either ''track'', ''link'', or ''detect''') ;
end

[movidx,frm0,frm1,trxids,nextra] = totrackinfo.getIntervals();

% output is the final stage trk file
trkfiles = totrackinfo.getTrkFiles('stage',stages(end));

% convert to frms, trxids
if ~isempty(totrackinfo.listfile)
  listFileNativePath = apt.MetaPath(totrackinfo.listfile, 'native', 'cache');
  listFileWslPath = listFileNativePath.asWsl();
  listOutFilesNativePath = cellfun(@(x) apt.MetaPath(x, 'native', 'cache'), totrackinfo.listoutfiles, 'UniformOutput', false);
  listOutFilesWslPath = cellfun(@(x) x.asWsl(), listOutFilesNativePath, 'UniformOutput', false);
  command13a = command12.append('-list_file', listFileWslPath);
  command14a = apt.ShellCommand.cat(command13a.append('-out'), listOutFilesWslPath{:});
  command17 = command14a;  % The other branch is longer, so we skip some numbers here
else
  trkFilesSelectedNative = trkfiles(movidx,:,:);
  trkFilesSelectedNativePath = cellfun(@(x) apt.MetaPath(x, 'native', 'cache'), trkFilesSelectedNative(:), 'UniformOutput', false);
  trkFilesSelectedWslPath = cellfun(@(x) x.asWsl(), trkFilesSelectedNativePath, 'UniformOutput', false);
  command13b = apt.ShellCommand.cat(command12.append('-out'), trkFilesSelectedWslPath{:});
  if sum(nextra) > 0,
    warningNoTrace('Tracking %d already-tracked frames',sum(nextra));
  end
  nativeMovFiles = totrackinfo.getMovfiles('movie',movidx) ;  % native file paths, as chars
  movFilesNativePath = cellfun(@(x) apt.MetaPath(x, 'native', 'movie'), nativeMovFiles, 'UniformOutput', false);
  movFilesWslPath = cellfun(@(x) x.asWsl(), movFilesNativePath, 'UniformOutput', false);
  command14b = apt.ShellCommand.cat(command13b.append('-mov'), movFilesWslPath{:});
  if ~all(frm0==1 & frm1==-1),
    command15b = command14b.append('-start_frame', num2str(frm0(:)')).append('-end_frame', num2str(frm1(:)'));
  else
    command15b = command14b;
  end
  if totrackinfo.hasTrxfiles,
    nativeTrxFiles = totrackinfo.getTrxFiles('movie',movidx) ;
    trxFilesNativePath = cellfun(@(x) apt.MetaPath(x, 'native', 'cache'), nativeTrxFiles, 'UniformOutput', false);
    trxFilesWslPath = cellfun(@(x) x.asWsl(), trxFilesNativePath, 'UniformOutput', false);
    command16b = apt.ShellCommand.cat(command15b.append('-trx'), trxFilesWslPath{:});
  elseif nstages > 1,
    nativeTrxFiles = totrackinfo.getTrkFiles('stage',1) ;
    trxFilesNativePath = cellfun(@(x) apt.MetaPath(x, 'native', 'cache'), nativeTrxFiles, 'UniformOutput', false);
    trxFilesWslPath = cellfun(@(x) x.asWsl(), trxFilesNativePath, 'UniformOutput', false);
    command16b = apt.ShellCommand.cat(command15b.append('-trx'), trxFilesWslPath{:});
  else
    command16b = command15b;
  end
  if ~all(cellfun(@isempty,trxids))
    command17 = command16b;
    for i = 1:numel(trxids)
      command17 = command17.append('-trx_ids', num2str(trxids{i}(:)'));
    end
  else
    command17 = command16b;
  end
end  % if
% command17 should be the result of everything before here.

if totrackinfo.hasCroprois && ~totrackinfo.islistjob,
  croproi = totrackinfo.getCroprois('movie',movidx);
  if iscell(croproi)
    croproi = cell2mat(croproi');
  end
  croproi = round(croproi);
  if ~isempty(croproi) && ~all(any(isnan(croproi),2),1),
    croproirowvec = croproi';
    croproirowvec = croproirowvec(:)'; % [xlovw1 xhivw1 ylovw1 yhivw1 xlovw2 ...] OR [xlomov1 xhimov1 ylomov1 yhimov1 xlomov2 ...] in serialmode
    command18 = command17.append('-crop_loc', num2str(croproirowvec));
  else
    command18 = command17;
  end
else
  command18 = command17;
end

% At last, the command!
command = command18;

end  % function
