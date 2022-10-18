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
      aptintrf = APTInterf.aptInterfacePath(aptroot,filesep0);

      modelChainID = DeepModelChainOnDisk.getCheckSingle(dmc.getModelChainID());
      trainConfig = DeepModelChainOnDisk.getCheckSingle(dmc.trainConfigLnx());
      cacheRootDir = dmc.getRootDir();
      errfile = DeepModelChainOnDisk.getCheckSingle(dmc.errfileLnx());
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
      trainLocFile = DeepModelChainOnDisk.getCheckSingle(dmc.trainLocLnx());
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
      filequote = '"';
      
      code = { ...
        APTInterf.getTorchHomeCode(torchhome,filequote) ...
        'python' ...
        [filequote aptintrf filequote] ...
        [filequote trainConfig filequote] ...
        '-name' modelChainID ...
        '-err_file' [filequote errfile filequote] ... 
        '-json_trn_file' [filequote trainLocFile filequote]...
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
      if ~isempty(stage2prevModels{1}),
        code = [code {'-model_files'} String.quoteCellStr(stage2prevModels{1},filequote)];
      end

      % conf params for the second stage trained in this job
      if nstages > 1,
        assert(nstages==2);
        code = [code,{'-conf_params2'}];
        if tfFollowsObjDet(2),
          code = [code {'use_bbox_trx' 'True'}];
        end
        code = [code,{'-type2',stage2netType{2}}];
        if ~isempty(stage2prevModels{2}),
          code = [code {'-model_files2'} String.quoteCellStr(stage2prevModels{2},filequote)];
        end
      end

      if ~isempty(ignore_local),
        code = [code, {'-ignore_local',num2str(ignore_local)}];
      end
      
      code = [code {'-cache' [filequote cacheRootDir filequote]}];
      code = [code {'train' '-use_cache'}];

      if trainType == DLTrainType.Restart,
        code = [code {'-continue -skip_db'}];
      end

      dosplit = ~isempty(val_split);
      if dosplit
        code = [code {'-val_split' num2str(val_split)}];
      end      

      codestr = String.cellstr2DelimList(code,' ');

    end

    function aptintrf = aptInterfacePath(aptroot,filesep0)
      aptintrf = [aptroot filesep0 APTInterf.pymoduleparentdir filesep0 APTInterf.pymodule];
    end

    function [codestr,code] = trackCodeGenBaseNew(totrackinfo,varargin)
      
      % Serial mode: 
      % - movtrk is [nmov] array
      % - outtrk is [nmov] array
      % - trxtrk is unsupplied, or [nmov] array
      % - view is a *scalar* and *must be supplied*
      % - croproi is unsupplied, or [xlo1 xhi1 ylo1 yhi1 xlo2 ... yhi_nmov] or row vec of [4*nmov]
      % - model_file is unsupplied, or [1] cellstr, or [nmov] cellstr      

      [filequote,frm0,frm1,...
        listfile,trxids,trxtrk,...
        croproi,do_linking,...
        aptroot,filesep0,...
        torchhome,ignore_local] = ...
        myparse(varargin,...
        'filequote','"',...
        'frm0',[],'frm1',[],...
        'listfile','',...
        'trxids',[],...
        'trxtrk',{},...
        'croproi',[],...
        'do_linking',true,...
        'aptroot',APT.Root,...
        'filesep','/',...
        'torchhome',APT.torchhome, ...
        'ignore_local',[]... % whether to remove local python modules from the path
        );

      dmc = totrackinfo.trainDMC;

      aptintrf = APTInterf.aptInterfacePath(aptroot,filesep0);

      modelChainID = DeepModelChainOnDisk.getCheckSingle(dmc.getModelChainID());
      trainConfig = DeepModelChainOnDisk.getCheckSingle(dmc.trainConfigLnx());
      cacheRootDir = dmc.getRootDir();

      stages = totrackinfo.stages;
      views = totrackinfo.views;
      nstages = numel(stages);
      nviews = numel(views);
      stage2models = cell(1,nstages);
      for istage = 1:nstages,
        stage = stages(istage);
        % cell of length nviews or empty
        stage2models{istage} = dmc.trainCurrModelSuffixlessLnx('stage',stage);
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
      configfile = totrackinfo.trackconfigfile;

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
        APTInterf.getTorchHomeCode(torchhome,filequote) ...
        'python' [filequote aptintrf filequote] ...
        [filequote trainConfig filequote] ...
        '-name' modelChainID ...
        '-err_file' [filequote totrackinfo.errfile filequote] ...
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
        {'-model_files'}, String.quoteCellStr(stage2models{1},filequote)];
      if nstages > 1,
        assert(nstages==2);
        code = [code {'-type2', stage2netType{2}} ...
          {'-model_files2'}, String.quoteCellStr(stage2models{2},filequote)];
      end

      if ~isempty(ignore_local),
        code = [code, {'-ignore_local',num2str(ignore_local)}];
      end
      code = [code {'-cache' [filequote cacheRootDir filequote]}];

      code = [code {'track'}];

      code = [code {'-config_file' [filequote configfile filequote]}];

      if ~do_linking
        code = [code {'-track_type only_predict'}]; % i think this might be in the track config file?
      end

      [movidx,frm0,frm1,trxids,nextra] = totrackinfo.getIntervals();

      % output is the final stage trk file
      trkfiles = totrackinfo.getTrkfiles('stage',stages(end));
      code = [code {'-out'} String.quoteCellStr(trkfiles(movidx,:,:),filequote)];

      % convert to frms, trxids
      if ~isempty(totrackinfo.listfile),
        code = [code {'-list_file' [filequote totrackinfo.listfile filequote]}];
      else
        if sum(nextra) > 0,
          warning('Tracking contiguous intervals, tracking %d extra frames',sum(nextra));
        end
        code = [code {'-mov' DeepTracker.cellstr2SpaceDelimWithQuote(totrackinfo.movfiles(movidx,:),filequote)}];
        if ~all(frm0==1 & frm1==-1),
          code = [code {'-start_frame' num2str(frm0(:)') '-end_frame' num2str(frm1(:)')}];
        end
        if totrackinfo.hasTrxfiles,
          code = [code {'-trx' DeepTracker.cellstr2SpaceDelimWithQuote(totrackinfo.trxfiles(movidx,:),filequote)}];
        elseif nstages > 1,
          code = [code {'-trx' DeepTracker.cellstr2SpaceDelimWithQuote(totrackinfo.getTrkfiles('stage',1),filequote)}];
        end
        if totrackinfo.hasTrxids,
          for i = 1:numel(totrackinfo.trxids(movidx)),
            code = [code {'-trx_ids' num2str(trxids{i}(:)')}]; %#ok<AGROW>
          end
        end
      end
      if totrackinfo.hasCroprois,
        croproi = round(totrackinfo.croprois);
        if ~isempty(croproi) && ~all(any(isnan(croproi),2),1),
          croproirowvec = croproi';
          croproirowvec = croproirowvec(:)'; % [xlovw1 xhivw1 ylovw1 yhivw1 xlovw2 ...] OR [xlomov1 xhimov1 ylomov1 yhimov1 xlomov2 ...] in serialmode
          code = [code {'-crop_loc' num2str(croproirowvec)}];
        end
      end

      
      codestr = String.cellstr2DelimList(code,' ');
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
        APTInterf.getTorchHomeCode(torchhome,filequote) ...
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

    function torchhomecmd = getTorchHomeCode(torchhome,filequote)

      if ispc,
        torchhomecmd = '';
      else
        torchhomecmd = ['TORCH_HOME=' filequote torchhome filequote];
      end

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
        APTInterf.getTorchHomeCode(torchhome,filequote) ...
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
        APTInterf.getTorchHomeCode(torchhome,filequote) ...
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
    

    function [codestr,code] = trackCodeGenBase(fileinfo,...
        frm0,frm1,... % (opt) can be empty. these should prob be in optional P-Vs
        varargin)
      
      % Serial mode: 
      % - movtrk is [nmov] array
      % - outtrk is [nmov] array
      % - trxtrk is unsupplied, or [nmov] array
      % - view is a *scalar* and *must be supplied*
      % - croproi is unsupplied, or [xlo1 xhi1 ylo1 yhi1 xlo2 ... yhi_nmov] or row vec of [4*nmov]
      % - model_file is unsupplied, or [1] cellstr, or [nmov] cellstr      

      modelChainID = fileinfo.modelChainID;
      dlconfig = DeepModelChainOnDisk.getCheckSingle(fileinfo.dlconfig);
      errfile = fileinfo.errfile;
      netType = char(DeepModelChainOnDisk.getCheckSingle(fileinfo.netType)); % for 2-stage, this is the stage2 nettype
      netMode = fileinfo.netMode; % " netmode
      % either char or [nviewx1] cellstr; or [nmov] in "serial mode" (see below)
      movtrk = fileinfo.movtrk; 
      % save as movtrk, except for 2 stage, this will be [nviewx2] or [nmovx2]
      outtrk = fileinfo.outtrk; 
      configfile = fileinfo.configfile;
      
      [listfile,cache,trxtrk,trxids,view,croproi,hmaps,deepnetroot,model_file,log_file,...
        updateWinPaths2LnxContainer,lnxContainerMntLoc,fs,filequote,...
        confparamsfilequote,tfserialmode,...
        do_linking,ignore_local] = ...
        myparse_nocheck(varargin,...
        'listfile','',...
        'cache',[],... % (opt) cachedir
        'trxtrk','',... % (opt) trxfile for movtrk to be tracked 
        'trxids',[],... % (opt) 1-based index into trx structure in trxtrk. empty=>all trx
        'view',[],... % (opt) 1-based view index. If supplied, track only that view. If not, all views tracked serially 
        'croproi',[],... % (opt) 1-based [xlo xhi ylo yhi] roi (inclusive). can be [nview x 4] for multiview
        'hmaps',false,...% (opt) if true, generate heatmaps
        'deepnetroot',APT.getpathdl,...
        'model_file',[], ... % can be [nview] cellstr
        'log_file',[],... (opt)
        'updateWinPaths2LnxContainer',ispc, ... % if true, all paths will be massaged from win->lnx for use in container 
        'lnxContainerMntLoc','/mnt',... % used when updateWinPaths2LnxContainer==true
        'filesep','/',...
        'filequote','\"',... % quote char used to protect filenames/paths.
                        ... % *IMPORTANT*: Default is escaped double-quote \" => caller
                        ... % is expected to wrap in enclosing regular double-quotes " !!
        'confparamsfilequote','\"', ...
        'serialmode',false, ...  % see serialmode above
        'do_linking',true, ... % Track id over ride json conf setting
        'ignore_local',[] ...
        );
      
     
      tflistfile = ~isempty(listfile);
      tffrm = ~tflistfile && ~isempty(frm0) && ~isempty(frm1);
      if tffrm, % ignore frm if it doesn't limit things
        if all(frm0 == 1) && all(isinf(frm1)),
          tffrm = false;
        end
      end
      tfcache = ~isempty(cache);
      tftrx = ~tflistfile && ~isempty(trxtrk);
      tftrxids = ~tflistfile && ~isempty(trxids);
      tfview = ~isempty(view);
      tfcrop = ~isempty(croproi) && ~all(any(isnan(croproi),2),1);
      tflog = ~isempty(log_file);
      tf2stg = netMode.isTwoStage;
      nstage = 1+double(tf2stg);
      tfmodel = ~isempty(model_file) && ~tf2stg; % -model_file arg not working in Py for 2stg yet
      
      torchhome = APT.torchhome;
                
      movtrk = cellstr(movtrk);
      outtrk = cellstr(outtrk);
      if tftrx
        trxtrk = cellstr(trxtrk);
      end
      if tfmodel
        model_file = cellstr(model_file);
      end
      
      if tfserialmode
        nmovserialmode = numel(movtrk);
        szassert(outtrk,[nmovserialmode nstage]);
        if tftrx
          assert(numel(trxtrk)==nmovserialmode);
        end
        assert(isscalar(view),'A scalar view must be specified for serial-mode.');
        if tfcrop
          szassert(croproi,[nmovserialmode 4]);
        end
        if tfmodel
          if isscalar(model_file)
            model_file = repmat(model_file,numel(view),1);
          else
            assert(numel(model_file)==numel(view));
          end
        end        
      else
        if tfview % view specified. track a single movie
          nview = 1;
          assert(isscalar(view));
          if tftrx
            assert(isscalar(trxtrk));
          end
        else
          nview = numel(movtrk);
          if nview>1
            assert(~tftrx && ~tftrxids,'Trx not supported for multiple views.');
          end
        end
        assert(nview==numel(movtrk));
        szassert(outtrk,[nview nstage]);
        if tfmodel
          assert(numel(model_file)==nview);
        end
        if tfcrop
          szassert(croproi,[nview 4]);
        end      
      end
      
      assert(~(tftrx && tfcrop));
      aptintrf = [deepnetroot fs 'APT_interface.py'];
      
      if updateWinPaths2LnxContainer
        fcnPathUpdate = @(x)DeepTracker.codeGenPathUpdateWin2LnxContainer(x,lnxContainerMntLoc);
        aptintrf = fcnPathUpdate(aptintrf);

        movtrk = cellfun(fcnPathUpdate,movtrk,'uni',0);
        outtrk = cellfun(fcnPathUpdate,outtrk,'uni',0);
        if tftrx
          trxtrk = cellfun(fcnPathUpdate,trxtrk,'uni',0);
        end
        if tfmodel
          model_file = cellfun(fcnPathUpdate,model_file,'uni',0);
        end
        if tflog
          log_file = fcnPathUpdate(log_file);
        end
        if tfcache
          cache = fcnPathUpdate(cache);
        end
        errfile = fcnPathUpdate(errfile);
        dlconfig = fcnPathUpdate(dlconfig);
        configfile = fcnPathUpdate(configfile);
      end      

      code = { ...
        APTInterf.getTorchHomeCode(torchhome,filequote) ...
        'python' [filequote aptintrf filequote] ...
        '-name' modelChainID ...
        };

      if tfview
        code = [code {'-view' num2str(view)}]; % view: 1-based for APT_interface
      end
      if ~isempty(ignore_local),
        code = [code, {'-ignore_local',num2str(ignore_local)}];
      end
      if tfcache
        code = [code {'-cache' [filequote cache filequote]}];
      end
      code = [code {'-err_file' [filequote errfile filequote]}];
      if tfmodel
        code = [code {'-model_files' ...
          DeepTracker.cellstr2SpaceDelimWithQuote(model_file,filequote)}];
      end
      if tflog
        code = [code {'-log_file' [filequote log_file filequote]}];
      end
                      
      if tf2stg
        %szassert(outtrk,[1 nstage],...
        %  'Multiview or serial multimovie unsupported for two-stage trackers.');
        if fileinfo.netmodeStage1.isObjDet
          use_bbox_trx_val = 'True';
        else
          use_bbox_trx_val = 'False';
        end
        code = [code {'-stage' 'multi' ...
                      '-type' char(fileinfo.nettypeStage1) ...
                      '-type2' char(netType) ...
                      [filequote dlconfig filequote] ...
                      'track' ...
                      '-out' DeepTracker.cellstr2SpaceDelimWithQuote(outtrk(:,2),filequote) }];        
      else
        code = [code {'-type' char(netType) ...
                      [filequote dlconfig filequote] ...
                      'track' ...
                      '-out' DeepTracker.cellstr2SpaceDelimWithQuote(outtrk,filequote) }];
      end
      code = [code {'-config_file' [filequote configfile filequote]}];

      if ~do_linking
        code = [code {'-track_type only_predict'}];
      end

      if tflistfile
        code = [code {'-list_file' [filequote listfile filequote]}];
      else
        code = [code {'-mov' DeepTracker.cellstr2SpaceDelimWithQuote(movtrk,filequote)}];
      end
      if tffrm
        frm0(isnan(frm0)) = 1;
        frm1(isinf(frm1)|isnan(frm1)) = -1;
        frm0 = round(frm0); % fractional frm0/1 errs in APT_interface due to argparse type=int
        frm1 = round(frm1); % just round silently for now        
        sfrm0 = sprintf('%d ',frm0); sfrm0 = sfrm0(1:end-1);
        sfrm1 = sprintf('%d ',frm1); sfrm1 = sfrm1(1:end-1);
        code = [code {'-start_frame' sfrm0 '-end_frame' sfrm1}];
      end
      if tftrx
        code = [code {'-trx' DeepTracker.cellstr2SpaceDelimWithQuote(trxtrk,filequote)}];
        if tftrxids
          if ~iscell(trxids),
            trxids = {trxids};
          end
          for i = 1:numel(trxids),
            trxidstr = sprintf('%d ',trxids{i});
            trxidstr = trxidstr(1:end-1);
            code = [code {'-trx_ids' trxidstr}]; %#ok<AGROW>
          end
        end
      elseif tf2stg
        code = [code {'-trx' DeepTracker.cellstr2SpaceDelimWithQuote(outtrk(:,1),filequote)}];
      end
      if tfcrop
        croproi = round(croproi);
        croproirowvec = croproi';
        croproirowvec = croproirowvec(:)'; % [xlovw1 xhivw1 ylovw1 yhivw1 xlovw2 ...] OR [xlomov1 xhimov1 ylomov1 yhimov1 xlomov2 ...] in serialmode
        roistr = mat2str(croproirowvec);
        roistr = roistr(2:end-1);
        code = [code {'-crop_loc' roistr}];
      end
      if hmaps
        code = [code {'-hmaps'}];
      end      
      codestr = String.cellstr2DelimList(code,' ');
    end
    
  end
  
end
