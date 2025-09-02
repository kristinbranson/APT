function codestr = trainCodeGenBase(dmc, varargin)
% Generate a base command for training.  Returned command uses WSL paths (see
% DLBackEndClass documentation).  This is a static method of DLBackEndClass.
[aptroot,confparamsextra,...
  val_split,...
  ignore_local,...
  do_just_generate_db] = ...
  myparse( ...
  varargin,...
  'aptroot',APT.Root,...
  'confparamsextra',{},...
  'val_split',[],...
  'ignore_local',[],... % whether to remove local python modules from the path
  'do_just_generate_db', false ...
  );
%aptintrf = APTInterf.aptInterfacePath(aptroot);
aptintrfNative = fullfile(aptroot, 'deepnet/APT_interface.py') ;  % this is a native path
aptintrf = wsl_path_from_native(aptintrfNative) ;  % The path to APT_interface.py, as a WSL path.


modelChainID = DeepModelChainOnDisk.getCheckSingle(dmc.getModelChainID());
nativeTrainConfig = DeepModelChainOnDisk.getCheckSingle(dmc.trainConfigLnx());
trainConfig = wsl_path_from_native(nativeTrainConfig) ;
nativeCacheRootDir = dmc.rootDir ;
cacheRootDir = wsl_path_from_native(nativeCacheRootDir) ;
nativeErrfile = DeepModelChainOnDisk.getCheckSingle(dmc.errfileLnx());
errFile = wsl_path_from_native(nativeErrfile) ;
nativeLogFile = DeepModelChainOnDisk.getCheckSingle(dmc.trainLogLnx());
logFile = wsl_path_from_native(nativeLogFile) ;
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
trainLocFile = wsl_path_from_native(nativeTrainLocFile) ;
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
torchhome_native = APT.gettorchhomepath() ;
wsl_torch_home = wsl_path_from_native(torchhome_native) ;      
torchhomeprefix = sprintf('TORCH_HOME=%s', escape_string_for_bash(wsl_torch_home)) ;

code_as_list = { ...
  torchhomeprefix ...
  'python' ...
  escape_string_for_bash(aptintrf) ...
  escape_string_for_bash(trainConfig) ...
  '-name' modelChainID ...
  '-err_file' escape_string_for_bash(errFile) ... 
  '-log_file' escape_string_for_bash(logFile) ... 
  '-json_trn_file' escape_string_for_bash(trainLocFile) ...
  };

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
  code_as_list = [code_as_list {'-stage', stageflag}];
end

if dmc.isMultiViewTracker,
  if nviews == 1,
    code_as_list = [code_as_list {'-view', num2str(views+1)}];
  end
end

% conf params
code_as_list = [code_as_list {'-conf_params'} confParams];

% only training stage 2 in this job
if tfFollowsObjDet(1),
  code_as_list = [code_as_list {'use_bbox_trx' 'True'}];
end

% type for the first stage trained in this job
code_as_list = [code_as_list,{'-type',stage2netType{1}}];
if ~isempty(stage2prevModels{1}{1})
  % MK 202300310. Stage2prevmodels is
  % repmat({repmat({''},[1,nviews]),[1,nstages]) for single animal
  % projects when no model is present. So  instead of checking
  % stage2prevModels{1}, I'm checking stage2prevModels{1}{1}. Not
  % tested for multi-animal. If it errors fix accordingly. Check line
  % 869 in DeepModelChainOnDisk.m
  code_as_list = [code_as_list {'-model_files'} escape_cellstring_for_bash(stage2prevModels{1})];
end

% conf params for the second stage trained in this job
if nstages > 1,
  assert(nstages==2);
  code_as_list = [code_as_list,{'-conf_params2'}];
  if tfFollowsObjDet(2),
    code_as_list = [code_as_list {'use_bbox_trx' 'True'}];
  end
  code_as_list = [code_as_list,{'-type2',stage2netType{2}}];
  if ~isempty(stage2prevModels{2}{1})
    % check the comment for model_files
    code_as_list = [code_as_list {'-model_files2'} escape_cellstring_for_bash(stage2prevModels{2})];
  end
end

if ~isempty(ignore_local),
  code_as_list = [code_as_list, {'-ignore_local',num2str(ignore_local)}];
end

code_as_list = [code_as_list {'-cache' escape_string_for_bash(cacheRootDir)}];
code_as_list = [code_as_list {'train' '-use_cache'}];

if trainType == DLTrainType.Restart,
  code_as_list = [code_as_list {'-continue -skip_db'}];
end

if do_just_generate_db ,
  code_as_list = [code_as_list {'-only_db'}];
end

dosplit = ~isempty(val_split);
if dosplit
  code_as_list = [code_as_list {'-val_split' num2str(val_split)}];
end      

codestr = space_out(code_as_list);

end  % function
