classdef APTInterf
  % CodeGen methods for APT_interface.py

  properties (Constant)
    pymodule = 'APT_interface.py';
    pymoduleparentdir = 'deepnet';
  end
  
  methods (Static)

    function codestr = trainCodeGenBase(dmc,varargin)
      
      [aptroot,filesep0,confparamsextra,...
        torchhome,val_split,...
        ignore_local] = ...
        myparse(varargin,...
        'aptroot',APT.Root,...
        'filesep','/',...
        'confparamsextra',{},...
        'torchhome',APT.torchhome, ...
        'val_split',[],...
        'ignore_local',[]... % whether to remove local python modules from the path
        );
      aptintrf = APTInterf.aptInterfacePath(aptroot);

      modelChainID = DeepModelChainOnDisk.getCheckSingle(dmc.getModelChainID());
      nativeTrainConfig = DeepModelChainOnDisk.getCheckSingle(dmc.trainConfigLnx());
      trainConfig = linux_path(nativeTrainConfig) ;
      nativeCacheRootDir = dmc.getRootDir();
      cacheRootDir = linux_path(nativeCacheRootDir) ;
      nativeErrfile = DeepModelChainOnDisk.getCheckSingle(dmc.errfileLnx());
      errFile = linux_path(nativeErrfile) ;
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
      trainLocFile = linux_path(nativeTrainLocFile) ;
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
      
      code = { ...
        APTInterf.getTorchHomeCode(torchhome) ...
        'python' ...
        escape_string_for_bash(aptintrf) ...
        escape_string_for_bash(trainConfig) ...
        '-name' modelChainID ...
        '-err_file' escape_string_for_bash(errFile) ... 
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
        code = [code {'-stage', stageflag}];
      end

      if dmc.isMultiViewTracker,
        if nviews == 1,
          code = [code {'-view', num2str(views+1)}];
        end
      end

      % conf params
      code = [code {'-conf_params'} confParams];

      % only training stage 2 in this job
      if tfFollowsObjDet(1),
        code = [code {'use_bbox_trx' 'True'}];
      end

      % type for the first stage trained in this job
      code = [code,{'-type',stage2netType{1}}];
      if ~isempty(stage2prevModels{1}{1})
        % MK 202300310. Stage2prevmodels is
        % repmat({repmat({''},[1,nviews]),[1,nstages]) for single animal
        % projects when no model is present. So  instead of checking
        % stage2prevModels{1}, I'm checking stage2prevModels{1}{1}. Not
        % tested for multi-animal. If it errors fix accordingly. Check line
        % 869 in DeepModelChainOnDisk.m
        code = [code {'-model_files'} escape_cellstring_for_bash(stage2prevModels{1})];
      end

      % conf params for the second stage trained in this job
      if nstages > 1,
        assert(nstages==2);
        code = [code,{'-conf_params2'}];
        if tfFollowsObjDet(2),
          code = [code {'use_bbox_trx' 'True'}];
        end
        code = [code,{'-type2',stage2netType{2}}];
        if ~isempty(stage2prevModels{2}{1})
          % check the comment for model_files
          code = [code {'-model_files2'} escape_cellstring_for_bash(stage2prevModels{2})];
        end
      end

      if ~isempty(ignore_local),
        code = [code, {'-ignore_local',num2str(ignore_local)}];
      end
      
      code = [code {'-cache' escape_string_for_bash(cacheRootDir)}];
      code = [code {'train' '-use_cache'}];

      if trainType == DLTrainType.Restart,
        code = [code {'-continue -skip_db'}];
      end

      dosplit = ~isempty(val_split);
      if dosplit
        code = [code {'-val_split' num2str(val_split)}];
      end      

      codestr = space_out(code);

    end

    function result = aptInterfacePath(aptroot)
      aptintrf = fullfile(aptroot, APTInterf.pymoduleparentdir, APTInterf.pymodule) ;  % this is a native path
      result = linux_path(aptintrf) ;
    end

    function [codestr,code] = trackCodeGenBase(totrackinfo,varargin)
      
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
        code = cell(totrackinfo.nmovies,1);
        for i = 1:totrackinfo.nmovies,
          tticurr = totrackinfo.selectSubset('movie',i);
          tticurr.setJobid(totrackinfo.getJobid);
          [codestrcurr,code{i}] = APTInterf.trackCodeGenBase(tticurr,varargin{:});
          if i == 1,
            codestr = codestrcurr;
          else
            codestr = [codestr,' && ',codestrcurr]; %#ok<AGROW> 
          end
        end
        return;
      end

      [filequote,frm0,frm1,...
        listfile,trxids,trxtrk,...
        croproi,track_type,...
        aptroot,filesep0,...
        torchhome,ignore_local] = ...
        myparse(varargin,...
        'filequote','"',...
        'frm0',[],'frm1',[],...
        'listfile','',...
        'trxids',[],...
        'trxtrk',{},...
        'croproi',[],...
        'track_type','track',...
        'aptroot',APT.Root,...
        'filesep','/',...
        'torchhome',APT.torchhome, ...
        'ignore_local',[]... % whether to remove local python modules from the path
        );

      dmc = totrackinfo.trainDMC;

      aptintrf = APTInterf.aptInterfacePath(aptroot);

      modelChainID = DeepModelChainOnDisk.getCheckSingle(dmc.getModelChainID());
      nativeTrainConfig = DeepModelChainOnDisk.getCheckSingle(dmc.trainConfigLnx());  % native path
      trainConfig = linux_path(nativeTrainConfig) ;
      nativeCacheRootDir = dmc.getRootDir();  % native path
      cacheRootDir = linux_path(nativeCacheRootDir) ;      

      stage2models = cell(1,nstages);
      for istage = 1:nstages,
        stage = stages(istage);
        % cell of length nviews or empty
        nativeModelPath = dmc.trainCurrModelSuffixlessLnx('stage',stage) ;  % native path
        modelPath = linux_path(nativeModelPath) ;
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
      configfile = linux_path(nativeConfigFile) ;

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

      code = { ...
        APTInterf.getTorchHomeCode(torchhome) ...
        'python' escape_string_for_bash(aptintrf) ...
        escape_string_for_bash(trainConfig) ...
        '-name' modelChainID ...
        '-err_file' escape_string_for_bash(linux_path(totrackinfo.errfile)) ...
        };
      if dmc.isMultiStageTracker,
        code = [code {'-stage' 'multi'}];
      end
      if dmc.isMultiViewTracker,
        if nviews == 1,
          code = [code {'-view', num2str(views)}];
        end
      end
      code = [code {'-type', stage2netType{1}} ...
        {'-model_files'}, escape_cellstring_for_bash(linux_path(stage2models{1}))];
      if nstages > 1,
        assert(nstages==2);
        code = [code {'-type2', stage2netType{2}} ...
          {'-model_files2'}, escape_cellstring_for_bash(linux_path(stage2models{2}))];
      end

      if ~isempty(ignore_local),
        code = [code, {'-ignore_local',num2str(ignore_local)}];
      end
      code = [code {'-cache' escape_string_for_bash(cacheRootDir)}];

      code = [code {'track'}];

      code = [code {'-config_file' escape_string_for_bash(configfile)}];
      
      switch track_type
        case 'link'
          code = [code {'-track_type only_link'}]; 
        case 'detect'
          code = [code {'-track_type only_predict'}]; 

      end

      [movidx,frm0,frm1,trxids,nextra] = totrackinfo.getIntervals();

      % output is the final stage trk file
      trkfiles = totrackinfo.getTrkfiles('stage',stages(end));

      % convert to frms, trxids
      if ~isempty(totrackinfo.listfile)
        code = [code {'-list_file' escape_string_for_bash(totrackinfo.listfile)}];
        code = [code {'-out'} escape_cellstring_for_bash(linux_path(totrackinfo.listoutfiles))];
      else
        tf = escape_cellstring_for_bash(linux_path(trkfiles(movidx,:,:)));
        code = [code {'-out'} tf(:)'];
        if sum(nextra) > 0,
          warning('Tracking contiguous intervals, tracking %d extra frames',sum(nextra));
        end
        nativeMovFiles = totrackinfo.getMovfiles('movie',movidx) ;  % native file paths
        movFiles = linux_path(nativeMovFiles) ;
        code = [code {'-mov' space_out(escape_cellstring_for_bash(movFiles))}];
        if ~all(frm0==1 & frm1==-1),
          code = [code {'-start_frame' num2str(frm0(:)') '-end_frame' num2str(frm1(:)')}];
        end
        if totrackinfo.hasTrxfiles,
          nativeTrxFiles = totrackinfo.getTrxfiles('movie',movidx) ;
          trxFiles = linux_path(nativeTrxFiles) ;
          code = [code {'-trx' space_out(escape_cellstring_for_bash(trxFiles))}];
        elseif nstages > 1,
          nativeTrxFiles = totrackinfo.getTrxfiles('stage',1) ;
          trxFiles = linux_path(nativeTrxFiles) ;
          code = [code {'-trx' space_out(escape_cellstring_for_bash(trxFiles))}];
        end
%         if totrackinfo.hasTrxids,
%           for i = 1:numel(totrackinfo.getTrxids('movie',movidx)),
%             code = [code {'-trx_ids' num2str(trxids{i}(:)')}]; %#ok<AGROW>
        if ~all(cellfun(@isempty,trxids))
           for i = 1:numel(trxids)
              code = [code {'-trx_ids' num2str(trxids{i}(:)')}]; %#ok<AGROW>
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
          code = [code {'-crop_loc' num2str(croproirowvec)}];
        end
      end

      
      codestr = space_out(code);
    end

    function basecmd = trainCodeGen(fileinfo,varargin)
      warning('Obsolete code');
      isMA = fileinfo.netType{1}.isMultiAnimal; % this means is bottom-up multianimal
      isNewStyle = isMA || ...
        (fileinfo.netMode{1}~=DLNetMode.singleAnimal && fileinfo.netMode{1}~=DLNetMode.multiAnimalTDPoseTrx);
      
      if isNewStyle
        basecmd = APTInterf.maTrainCodeGenTrnPack(fileinfo,varargin{:});
      else
        basecmd = APTInterf.regTrainCodeGen(fileinfo,varargin{:});
      end
    end
    
    function [codestr,code] = maTrainCodeGenTrnPack(fileinfo,varargin)
      % Wrapper for matrainCodeGen, assumes standard trnpack structure.
      % Reads some files from trnpack
      % 
      % netType/netMode: for TopDown trackers, these are currently ALWAYS
      % stage2. pass stage1 in the varargin. Yes, this is a little weird if
      % maTopDownStage=='first'.

      warning('Obsolete code');
      
      [maTopDown,maTopDownStage,maTopDownStage1NetType,...
        maTopDownStage1NetMode,leftovers] = ...
        myparse_nocheck(varargin,...
        'maTopDown',false, ...
        'maTopDownStage',[], ... % '-stage' flag: {'multi','first','second'}
        'maTopDownStage1NetType',[], ...
        'maTopDownStage1NetMode',[] ...
        );

      netMode = fileinfo.netMode;
      
      %[trnpack,dllblID] = fileparts(dlconfigfile);
      %trnjson = fullfile(trnpack,'loc.json');
%       dllbljson = fullfile(trainlocfile);
%       dlj = readtxtfile(dllbljson);
%       dlj = jsondecode(dlj{1});
%       
%       assert(maTopDown == ~isscalar(dlj.TrackerData));
      
      if maTopDown
        isObjDet = any(cellfun(@(x) x.isObjDet,netMode));
        [codestr,code] = APTInterf.matdTrainCodeGen(fileinfo,isObjDet,maTopDownStage1NetType,maTopDownStage,...
          leftovers{:});
      else
        assert(all(cellfun(@(x) x==DLNetMode.multiAnimalBU,netMode)));
        [codestr,code] = APTInterf.mabuTrainCodeGen(fileinfo,leftovers{:});
      end
    end
    
    function [codestr,code] = mabuTrainCodeGen(fileinfo,varargin)

      warning('Obsolete code');

      % Simplified relative to trainCodeGen

      modelChainID = fileinfo.modelChainID;
      dlconfig = fileinfo.dlconfig;
      cache = fileinfo.cache;
      errfile = fileinfo.errfile;
      netType = char(DeepModelChainOnDisk.getCheckSingle(fileinfo.netType));
      trnjson = fileinfo.trainlocfile;
      
      [deepnetroot,fs,filequote,confparamsextra,confparamsfilequote,...
        prev_model,torchhome,val_split,augOnly,augOut,...
        clsfyIsClsfy,clsfyOut,ignore_local] = ...
        myparse_nocheck(varargin,...
        'deepnetroot',APT.getpathdl,...
        'filesep','/',...
        'filequote','\"',... % quote char used to protect filenames/paths.
        ... % *IMPORTANT*: Default is escaped double-quote \" => caller
        ... % is expected to wrap in enclosing regular double-quotes " !!
        'confparamsextra',{},...
        'confparamsfilequote','\"', ...
        'prev_model',[],...
        'torchhome',APT.torchhome, ...
        'val_split',[],...
        'augOnly',false, ...
        'augOut','', ... % used only if augOnly==true        
        'clsfyIsClsfy',false,... % if true, classify not train
        'clsfyOut',[], ...        
        'ignore_local',[] ...
        );
      
      aptintrf = [deepnetroot fs 'APT_interface.py'];

      % MK 20220128 -- db_format should come from params_deeptrack_net.yaml
%       confParams = { ... %        'is_multi' 'True' ...    'max_n_animals' num2str(maxNanimals) ...
%         'db_format' [confparamsfilequote 'coco' confparamsfilequote] ... % when the job is submitted the double quote need to escaped. This is tested fro cluster. Not sure for AWS etc. MK 20210226
%         confparamsextra{:} ...
%         };
      confParams = confparamsextra;
      
      code = cat(2,{ ...
        APTInterf.getTorchHomeCode(torchhome) ...
        'python' ...
        [filequote aptintrf filequote] ...
        dlconfig ...
        '-name' modelChainID ...
        '-err_file' [filequote errfile filequote] ... 
        '-json_trn_file' trnjson ...
        '-conf_params'}, ...
        confParams, ...
        {'-type' netType}); ...
      if ~isempty(ignore_local),
        code = [code, {'-ignore_local',num2str(ignore_local)}];
      end
      if ~isempty(prev_model)
        code = [code {'-model_files'}];
        for i = 1:numel(prev_model),
          code = [code, {[filequote prev_model{i} filequote]}]; %#ok<AGROW> 
        end
      end
      
      code = [code {'-cache' [filequote cache filequote]}];
      if clsfyIsClsfy
        assert(~isempty(clsfyOut));
        code = [code {'classify' '-out' clsfyOut}];
      else
        code = [code {'train' '-use_cache'}];
      end
      if augOnly
        code = [code ...
          {'-only_aug' ...
          '-aug_out' augOut}];
      end
      dosplit = ~isempty(val_split) && ~clsfyIsClsfy;
      if dosplit
        code = [code {'-val_split' num2str(val_split)}];
      end      

      codestr = String.cellstr2DelimList(code,' ');
    end

    function torchhomecmd = getTorchHomeCode(native_torch_home)
      torch_home = linux_path(native_torch_home) ;      
      torchhomecmd = ['TORCH_HOME=' escape_string_for_bash(torch_home)] ;
    end
            
    function [codestr,code] = matdTrainCodeGen(fileinfo,...
        isObjDet,netTypeStg1,stage,varargin)
      
      warning('Obsolete code');

      modelChainID = fileinfo.modelChainID;
      dlconfigfile = fileinfo.dlconfig;
      cache = fileinfo.cache;
      errfile = fileinfo.errfile;
      trnjson = fileinfo.trainlocfile;
      netTypeStg2 = char(fileinfo.netType{end});

      [deepnetroot,fs,filequote,confparamsfilequote,...
        prev_model,prev_model2,torchhome,augOnly,augOut,ignore_local] = ...
        myparse_nocheck(varargin,...
        'deepnetroot',APT.getpathdl,...
        'filesep','/',...
        'filequote','\"',... % quote char used to protect filenames/paths.
        ... % *IMPORTANT*: Default is escaped double-quote \" => caller
        ... % is expected to wra% TODO: ht paramsp in enclosing regular double-quotes " !!
        'confparamsfilequote','\"', ...
        'prev_model',[],...
        'prev_model2',[],...
        'torchhome',APT.torchhome, ...
        'augOnly',false,...
        'augOut','', ... % used only if augOnly==true
        'ignore_local',[]...
        );
      
      % currently this only works with a single view
      if ~isempty(prev_model),
        prev_model = DeepModelChainOnDisk.getCheckSingle(prev_model);
      end
      if ~isempty(prev_model2),
        prev_model2 = DeepModelChainOnDisk.getCheckSingle(prev_model2);
      end
      if ~isempty(netTypeStg1),
        netTypeStg1 = char(DeepModelChainOnDisk.getCheckSingle(netTypeStg1));
      end

      aptintrf = [deepnetroot fs 'APT_interface.py'];
      
      STAGEFLAGS = {'multi' 'first' 'second'};
      
      code = { ...
        APTInterf.getTorchHomeCode(torchhome) ...
        'python' ...
        [filequote aptintrf filequote] ...
        dlconfigfile ...
        '-name' modelChainID ...
        '-err_file' [filequote errfile filequote] ...
        '-json_trn_file' trnjson ...
        '-stage' STAGEFLAGS{stage+1} ...
        };
      if ~isempty(ignore_local),
        code = [code, {'-ignore_local',num2str(ignore_local)}];
      end
              
      % set -conf_params, -type
      if stage==0 || stage==1 % inc stg1/detect
        code = [code ...
          {
          ... %'-conf_params' ...
          ... %'db_format' [confparamsfilequote 'coco' confparamsfilequote] ... % when the job is submitted the double quote need to escaped. This is tested fro cluster. Not sure for AWS etc. MK 20210226
          ... %'mmdetect_net' [confparamsfilequote 'frcnn' confparamsfilequote] ...
          '-type' netTypeStg1 ...
          } ];
      end
      
      % if nec, set -conf_params2, -type2
      if stage==2
        % single stage 2 
        
        tfaddstg2 = true;
        confparamsstg2 = '-conf_params';
        typestg2 = '-type';
        if ~isempty(prev_model)
          code = [code {'-model_files' [filequote prev_model filequote]}];
        end
      elseif stage==0
        tfaddstg2 = true;
        confparamsstg2 = '-conf_params2';
        typestg2 = '-type2';
        if ~isempty(prev_model)
          code = [code {'-model_files' [filequote prev_model filequote]}];
        end
        if ~isempty(prev_model2)
          code = [code {'-model_files2' [filequote prev_model2 filequote]}];
        end
        
      else
        tfaddstg2 = false;
        if ~isempty(prev_model)
          code = [code {'-model_files' [filequote prev_model filequote]}];
        end

      end
      if tfaddstg2
        confparams2 = { ...
          confparamsstg2 ...
          ... %'db_format' [confparamsfilequote 'tfrecord' confparamsfilequote] ...
          };
        if isObjDet
          confparams2 = [confparams2 {'use_bbox_trx' 'True'}];
        end
        confparams2 = [confparams2 {typestg2 netTypeStg2}];
        code = [code confparams2];
      end
      
      code = [code ...
        { ...
        '-cache' [filequote cache filequote] ... % String.escapeSpaces(cache),...
        'train' ...
        '-use_cache' ...
        } ];
      if augOnly
        code = [code ...
          {'-only_aug' ...
          '-aug_out' augOut}];
      end
      
      codestr = String.cellstr2DelimList(code,' ');
    end
    
    function codestr = regTrainCodeGen(fileinfo,varargin)
      % "reg" => regular, from pre-MA APT
      
      modelChainID = fileinfo.modelChainID;
      dlconfigfile = fileinfo.dlconfig;
      cache = fileinfo.cache;
      errfile = fileinfo.errfile;
      netType = char(DeepModelChainOnDisk.getCheckSingle(fileinfo.netType));
      trnjson = fileinfo.trainlocfile;

      [view,deepnetroot,splitfile,classify_val,classify_val_out,val_split,...
        trainType,fs,prev_model,filequote,augOnly,confparamsfilequote,augOut,ignore_local] = myparse(varargin,...
        'view',[],... % (opt) 1-based view index. If supplied, train only that view. If not, all views trained serially
        'deepnetroot',APT.getpathdl,...
        'split_file',[],...
        'classify_val',false,... % if true, split_file must be spec'd
        'classify_val_out',[],... % etc
        'val_split',[],...
        'trainType',DLTrainType.New,...
        'filesep','/',...
        'prev_model',[],...
        'filequote','\"',... % quote char used to protect filenames/paths.
        'augOnly',false,...
        'confparamsfilequote','\"',... % this is used in other train code functions, adding here to remove warning
        'augOut','', ... % used only if augOnly==true
        'ignore_local',[] ...
        ... % *IMPORTANT*: Default is escaped double-quote \" => caller
        ... % is expected to wrap in enclosing regular double-quotes " !!
        );
      torchhome = APT.torchhome;
      
      tfview = ~isempty(view);
%       [trnpack,dllblID] = fileparts(dlconfigfile);
%       trnjson = fullfile(trnpack,'loc.json');
%       dllbljson = fullfile(trnpack,[dllblID '.json']);
%       dlj = readtxtfile(dllbljson);
%       dlj = jsondecode(dlj{1});
  
      aptintrf = [deepnetroot fs 'APT_interface.py'];
      
      switch trainType
        case DLTrainType.New
          continueflags = '';
        case DLTrainType.Restart
          continueflags = '-continue -skip_db';
        case DLTrainType.RestartAug
          continueflags = '-continue';
        otherwise
          assert(false);
      end
      
      dosplit = ~isempty(val_split);
      if dosplit
        splitfileargs = sprintf('-val_split %d',val_split);
        if classify_val
          splitfileargs = [splitfileargs sprintf(' -classify_val -classify_val_out %s',classify_val_out)];
        end
      else
        splitfileargs = '';
      end
      
      code = { ...
        APTInterf.getTorchHomeCode(torchhome) ...
        'python' ...
        [filequote aptintrf filequote] ...
        '-name' ...
        modelChainID ...
        };
      if ~isempty(ignore_local),
        code = [code, {'-ignore_local',num2str(ignore_local)}];
      end
      if tfview
        code = [code {'-view' num2str(view)}];
      end
      code = [code { ...
        '-cache' ...
        [filequote cache filequote] ... % String.escapeSpaces(cache),...
        '-err_file' ...
        [filequote errfile filequote] ...
        '-json_trn_file' ...
        [filequote trnjson filequote]}
        ]; ... % String.escapeSpaces(errfile),...
      unique_stages = unique(fileinfo.stage);

      for istage = 1:numel(unique_stages),
        stage = unique_stages(istage);
        isfirstview = true;
        idx1 = find(fileinfo.stage==stage);
        for i = idx1(:)',
          if numel(prev_model) >= i && ~isempty(prev_model{i}),
            if isfirstview,
              if istage == 1,
                code{end+1} = '-model_files'; %#ok<AGROW> 
              else
                code{end+1} = sprintf('-model_files%d',istage); %#ok<AGROW> 
              end
              isfirstview = false;
            end
            code{end+1} = [filequote prev_model{i} filequote]; %#ok<AGROW> 
          end
        end
      end
      code = [code ...
        {'-type' ...
        netType ...
        [filequote dlconfigfile filequote] ... % String.escapeSpaces(dllbl),...
        'train' ...
        '-use_cache' ...
        continueflags ...
        splitfileargs} ];
      if augOnly
        code = [code ...
          {'-only_aug' ...
          '-aug_out' augOut}];
      end
      codestr = String.cellstr2DelimList(code,' ');
    end
        
    function splitTrainValCodeGenCmdfile(codefname,classifyOutmat,...
          trnID,dlconfig,cache,errfile,netType,netMode,valSplit,varargin)
      % write cmdfile to disk for train/val of given split
      % 
      % if not successful, throws
        
      [fh,msg] = fopen(codefname,'w');
      if isequal(fh,-1)
        error('Filed to open %s: %s',codefname,msg);
      end
      
      argsTrain = [varargin {'filequote' '"' 'val_split' valSplit}];
      argsVal = [varargin ...
        {'filequote' '"' 'clsfyIsClsfy' true 'clsfyOut' classifyOutmat }];
      
      s1 = APTInterf.trainCodeGen(trnID,dlconfig,cache,errfile,...
        netType,netMode,argsTrain{:});
      s2 = APTInterf.trainCodeGen(trnID,dlconfig,cache,errfile,...
        netType,netMode,argsVal{:});
      
      fprintf(fh,'#!/usr/bin/env bash\n');
      fprintf(fh,'\n%s\n',s1);
      fprintf(fh,'\n%s\n',s2);
      fclose(fh);
    end
  end
end
