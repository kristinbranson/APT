function command = trainCodeGenBase(dmc, varargin)
% Generate a base command for training.  Returned command uses WSL paths (see
% DLBackEndClass documentation).  This is a static method of DLBackEndClass.
[nativeAptRootAsChar,...
 confparamsextra,...
 val_split,...
 ignore_local,...
 do_just_generate_db] = ...
myparse(varargin,...
        'nativeaptroot',APT.Root,...
        'confparamsextra',{},...
        'val_split',[],...
        'ignore_local',[],... % whether to remove local python modules from the path
        'do_just_generate_db', false ...
        );
%aptintrf = APTInterf.aptInterfacePath(aptroot);
aptInterfaceDotPyNativePath = apt.MetaPath(fullfile(nativeAptRootAsChar, 'deepnet/APT_interface.py'), 'native', 'source') ;  % this is a native path
aptInterfaceDotPyWslPath = aptInterfaceDotPyNativePath.asWsl() ;  % The path to APT_interface.py, as a WSL path.

modelChainID = DeepModelChainOnDisk.getCheckSingle(dmc.getModelChainID());
nativeTrainConfig = DeepModelChainOnDisk.getCheckSingle(dmc.trainConfigLnx());
trainConfigFileNativePath = apt.MetaPath(nativeTrainConfig, 'native', 'cache');
trainConfigFileWslPath = trainConfigFileNativePath.asWsl();
nativeCacheRootDir = apt.MetaPath(dmc.rootDir, 'native', 'cache') ;
% cacheRootDir = wsl_path_from_native(nativeCacheRootDir) ;
cacheRootDir = nativeCacheRootDir.asWsl() ;

nativeErrfile = DeepModelChainOnDisk.getCheckSingle(dmc.errfileLnx());
errFilePathNative = apt.MetaPath(nativeErrfile, 'native', 'cache');
errFilePathWsl = errFilePathNative.asWsl();
nativeLogFile = DeepModelChainOnDisk.getCheckSingle(dmc.trainLogLnx());
logFilePathNative = apt.MetaPath(nativeLogFile, 'native', 'cache');
logFilePathWsl = logFilePathNative.asWsl();
tfFollowsObjDet = dmc.getFollowsObjDet();
stages = unique(dmc.getStages());
views = unique(dmc.getViews());
nstages = numel(stages);
nviews = numel(views);
% one net type per stage
stage2netType = cell(1,nstages);
for istage = 1:nstages,
  stage = stages(istage);
  stage2netType{istage} = char(DeepModelChainOnDisk.getCheckSingle(dmc.getNetType('stage',stage)));
end
nativeTrainLocFile = DeepModelChainOnDisk.getCheckSingle(dmc.trainLocLnx());
trainLocFilePathNative = apt.MetaPath(nativeTrainLocFile, 'native', 'cache');
trainLocFilePathWsl = trainLocFilePathNative.asWsl();
stage2prevModels = cell(1,nstages);
for istage = 1:nstages,
  stage = stages(istage);
  % cell of length nviews or empty
  stage2prevModels{istage} = dmc.getPrevModels('stage',stage); 
  assert(isempty(stage2prevModels{istage}) || numel(stage2prevModels{istage}) == nviews);
end
% trainType has to be unique - only one parameter to APT_interface to
% specify this
trainType = DeepModelChainOnDisk.getCheckSingle(dmc.getTrainType);

% MK 20220128 -- db_format should come from params_deeptrack_net.yaml
%       confParams = { ... %        'is_multi' 'True' ...    'max_n_animals' num2str(maxNanimals) ...
%         'db_format' [confparamsfilequote 'coco' confparamsfilequote] ... % when the job is submitted the double quote need to escaped. This is tested fro cluster. Not sure for AWS etc. MK 20210226
%         confparamsextra{:} ...
%         };
confParams = confparamsextra;
%filequote = '"';

% Get prefix the sets torch home dir
torchHomePathNativeRaw = APT.gettorchhomepath() ;
torchHomePathNative = apt.MetaPath(torchHomePathNativeRaw, 'native', 'cache');
torchHomePathWsl = torchHomePathNative.asWsl();      
%torchHomePrefix = sprintf('TORCH_HOME=%s', escape_string_for_bash(torchHomePathWsl)) ;
torchHomePrefix = apt.ShellVariableAssignment('TORCH_HOME', torchHomePathWsl) ;

command0 = apt.ShellCommand(...
  { torchHomePrefix, ...
    'python', ...
    aptInterfaceDotPyWslPath, ...
    trainConfigFileWslPath, ...
    '-name', modelChainID, ...
    '-err_file', errFilePathWsl, ...
    '-log_file', logFilePathWsl, ...
    '-json_trn_file', trainLocFilePathWsl, ...
    }, ...
  apt.PathLocale.wsl, ...
  apt.Platform.posix);

if dmc.isMultiStageTracker,
  if nstages > 1,
    stageflag = 'multi';
  elseif stage == 1,
    stageflag = 'first';
  elseif stage == 2,
    stageflag = 'second';
  else
    error('Stage must be 1 or 2');
  end
  command1 = command0.append('-stage', stageflag);
else
  command1 = command0;
end

if dmc.isMultiViewTracker,
  if nviews == 1,
    command2 = command1.append('-view', num2str(views+1));
  else
    command2 = command1;
  end
else
  command2 = command1;
end

% conf params
command3 = command2.append('-conf_params', confParams{:});

% only training stage 2 in this job
if tfFollowsObjDet(1),
  command4 = command3.append('use_bbox_trx', 'True');
else
  command4 = command3;
end

% type for the first stage trained in this job
command5 = command4.append('-type', stage2netType{1});
if ~isempty(stage2prevModels{1}{1})
  % MK 202300310. Stage2prevmodels is
  % repmat({repmat({''},[1,nviews]),[1,nstages]) for single animal
  % projects when no model is present. So  instead of checking
  % stage2prevModels{1}, I'm checking stage2prevModels{1}{1}. Not
  % tested for multi-animal. If it errors fix accordingly. Check line
  % 869 in DeepModelChainOnDisk.m
  command6 = command5.append('-model_files', stage2prevModels{1});
else
  command6 = command5;
end

% conf params for the second stage trained in this job
if nstages > 1,
  assert(nstages==2);
  command7 = command6.append('-conf_params2');
  if tfFollowsObjDet(2),
    command8 = command7.append('use_bbox_trx', 'True');
  else
    command8 = command7;
  end
  command9 = command8.append('-type2', stage2netType{2});
  if ~isempty(stage2prevModels{2}{1})
    % check the comment for model_files
    command10 = apt.ShellCommmand.cat(command9.append('-model_files2'), stage2prevModels{2});
  else
    command10 = command9;
  end
else
  command10 = command6;
end

if ~isempty(ignore_local),
  command11 = command10.append('-ignore_local', num2str(ignore_local));
else
  command11 = command10;
end

command12 = command11.append('-cache', cacheRootDir);
command13 = command12.append('train', '-use_cache');

if trainType == DLTrainType.Restart,
  command14 = command13.append('-continue', '-skip_db');
else
  command14 = command13;
end

if do_just_generate_db ,
  command15 = command14.append('-only_db');
else
  command15 = command14;
end

dosplit = ~isempty(val_split);
if dosplit
  command16 = command15.append('-val_split', num2str(val_split));
else
  command16 = command15;
end      

command = command16;

end  % function
