classdef DeepModelChainOnDisk < matlab.mixin.Copyable
  % DMCOD understands the filesystem structure of a deep model. This same
  % structure is used on both remote and local filesystems.
  %
  % DMCOD also now handles tracking output: eg trkfiles and associated
  % log/errfiles/etc; gt results files etc. These are a bit conceptually
  % different but they live underneath the cache/modelchaindir at runtime.
  %
  % DMCOD does know whether the model is on a local or remote filesystem 
  % via the .reader property. The .reader object is a delegate that knows 
  % how to actually read the (possibly remote) filesystem. This works fine 
  % for now future design unclear.
  
  properties (Constant)
    configFileExt = '.json'; % switching this to output json file in train/track commands
    gen_strippedlblfile = false; % try disabling the stripped lbl file generation!!
    trainPackName = 'trnpack.json';
    trainLocName = 'loc.json';
    trainingImagesName = 'deepnet_training_samples.mat';

    props_numeric = {'jobidx','stage','view','splitIdx','iterFinal','iterCurr','nLabels'};
    props_cell = {'netType','netMode','trainType','modelChainID','trainID','restartTS','trainConfigNameOverride','trkTaskKeyword','prev_models'};
    props_bool = {'tfFollowsObjDet'};
  end

  properties
    % All properties will be cells/arrays, with an entry for each separate
    % training / tracking.
    % We will keep track of which job, stage, and view the
    % training/tracking corresponds to, and allow indexing by these.
    % We will always return cells/arrays. 
    jobidx = [];
    stage = [];
    view = [];
    splitIdx = [];

    % root/parent "Models" dir
    % All files will be within a directory specified with the argument 
    % `-cache` and stored in `args.cache`. In MATLAB, this is stored in the
    % `DeepModelChainOnDisk` property `rootDir`. For local or JRC backends,
    % this is set to the Labeler object's `DLCacheDir` which mirrors
    % `Labeler.projTempDir`. Within the Labeler code, this is set to a
    % randomly named directory within `APT.getdlcacheroot`. By default, on
    % Linux, this is the directory `.apt` within the user's home directory.
    % On Windows, it is the directory `.apt` within:
    % ```
    % userDir = winqueryreg('HKEY_CURRENT_USER',...
    %                       ['Software\Microsoft\Windows\CurrentVersion\' ...
    %                       'Explorer\Shell Folders'],'Personal');
    % ```
    % (I'm not sure what this returns, will need to check on Windows...)
    % This can be set to a different location by adding the line
    % ```
    % dltemproot,/my/directory/.apt
    % ```
    % to the file `APT/Manifest.txt`.
    % When the `DeepModelChainOnDisk` for a train/track job is defined,
    % e.g. within `DeepTracker.trnSpawn, `rootDir` is set. We
    % should probably encapsulate this code so that it is not copy-pasted
    % all over given that it is done very similarly everywhere.   
    % Example: `/groups/branson/home/bransonk/.apt/tp76715886_6c90_4126_a9f4_0c3d31206ee5`
    % This will be the same for all stages/views.
    localRootDir_ = '' ;
    remoteRootDir_ = '' ;
    % Underscore means "protected by convention"
    %rootDir = '';

    % All data associated with a given Labeler project will be stored in a
    % subdirectory of the cache directory, stored in
    % `DeepModelChainOnDisk.dirProjLnx`. The name of this directory is based
    % on the initial project lbl file's name and is stored in
    % `Labeler.projname` and `DeepModelChainOnDisk.projID`. Note that if you
    % rename the lbl file, or choose "Save as" within the GUI, this project
    % name is not updated. I'm not sure why we need this subdirectory, I
    % think there can only be one project directory per cache directory? I
    % propose removing this. This directory is only known to the MATLAB
    % code. When creating a train/track job's DeepModelChainOnDisk, the
    % Labeler.projname is passed to the `DeepModelChainOnDisk` as `projID`.
    % Example: `/groups/branson/home/bransonk/.apt/tp76715886_6c90_4126_a9f4_0c3d31206ee5/four_points_180806`
    % This will be the same for all stages/views.
    projID = '';

    netType = {}; % cell of DLNetTypes
    netMode = {};
    modelChainID = {}; % unique ID for a training model for (projname,view). 
                       % A model can be trained once, but also train-augmented.
    trainID = {}; % a single modelID may be trained multiple times due to 
                  % train-augmentation, so a single modelID may have multiple
                  % trainID associated with it. Each (modelChainID,trainID) pair
                  % has a unique associated stripped lbl.            
    restartTS = {}; % Training for each (modelChainID,trainID) can be killed and
                    % restarted arbitrarily many times. This timestamp uniquely
                    % identifies a restart
    trainType = {}; % cell of DLTrainType
    
    isMultiViewTracker = []; % whether this tracker is part of a multi-view tracker
    isMultiStageTracker = []; % whether this tracker is part of a multi-stage tracker
    tfFollowsObjDet = []; % whether the previous stage was an object detection stage
    % if provided, overrides .lblStrippedName. used for each running splits
    % wherein a single stripped lbl is used in multiple runs
    trainConfigNameOverride = {}; 
    
    iterFinal = []; % final expected iteration    
    iterCurr = []; % last completed iteration, corresponds to actual model file used
      % iterCurr is 0x0 (if there are zero models) or 1 x nModels.  Elements can
      % be nan, which means no training has been done on that model.
    nLabels = []; % number of labels used to train
    
    %reader % scalar DeepModelChainReader. used to update the itercurr; 
      % knows how to read the (possibly remote) filesys etc
      
    filesep ='/'; % file separator

    trkTaskKeyword = {}; % arbitrary tracking task keyword; used for tracking output files
    trkTSstr = '';% timestamp for tracking
    prev_models = []; % prev model to start training from
    isRemote_ = false  
      % True iff the "current" version of the model is on a remote AWS filesystem.  
      % Underscore means "protected by convention"
  end

  properties (Dependent)
    n
    nviews
    njobs
    nstages
    rootDir
    localRootDir
    remoteRootDir
  end

  methods
    function n = get.n(obj)
      n = numel(obj.view);
    end
    function v = get.nviews(obj)
      v = numel(unique(obj.view));
    end
    function v = get.nstages(obj)
      v = numel(unique(obj.stage));
    end
    function v = get.njobs(obj)
      v = numel(unique(obj.jobidx));
    end
    function v = get.rootDir(obj)
      if obj.isRemote_ ,
        v = obj.remoteRootDir_ ;
      else
        v = obj.localRootDir_ ;        
      end
    end
    function set.rootDir(obj,v)
      if obj.isRemote_ ,
        obj.remoteRootDir_ = v ;
      else
        obj.localRootDir_ = v ;
      end
    end    
    function v = get.remoteRootDir(obj)
      v = obj.remoteRootDir_ ;
    end
    function v = get.localRootDir(obj)
      v = obj.localRootDir_ ;
    end
    function [v,idx] = getJobs(obj,varargin)
      idx = obj.select(varargin{:});
      v = obj.jobidx(idx);
    end
    function [v,idx] = getStages(obj,varargin)
      idx = obj.select(varargin{:});
      v = obj.stage(idx);
    end
    function [v,idx] = getViews(obj,varargin)
      idx = obj.select(varargin{:});
      v = obj.view(idx);
    end
    function [v,idx] = getSplits(obj,varargin)
      idx = obj.select(varargin{:});
      v = obj.splitIdx(idx);
    end
    function v = isRemote(obj)
      v = obj.isRemote_ ;
    end
    function idx = select(obj,varargin)
      idx = DeepModelChainOnDisk.selectHelper(obj,varargin{:});
    end
    function info = getIdentifiers(obj)
      info = struct;
      info.jobidx = obj.jobidx;
      info.view = obj.view;
      info.stage = obj.stage;
      info.splitIdx = obj.splitIdx;
    end
    function [ijob,ivw,istage,isplit] = ind2sub(obj,idx)
      ijob = obj.jobidx(idx);
      ivw = obj.view(idx);
      istage = obj.stage(idx);
      isplit = obj.splitIdx(idx);
    end
    function v = isSplit(obj,varargin)
      idx = obj.select(varargin{:});
      v = obj.splitIdx(idx) > 0;
    end

    % dirProjLnx should be the same for all jobs, views, stages
    function v = dirProjLnx(obj)
      v = [obj.rootDir obj.filesep obj.projID];
    end

    function [v,idx] = getModelChainID(obj,varargin)
      idx = obj.select(varargin{:});
      v = obj.modelChainID(idx);
    end
    function [v,idx] = getTrainID(obj,varargin)
      idx = obj.select(varargin{:});
      v = obj.trainID(idx);
    end
    function [v,idx] = getView(obj,varargin)
      idx = obj.select(varargin{:});
      v = obj.view(idx);
    end
    function [v,idx] = getNetType(obj,varargin)
      idx = obj.select(varargin{:});
      v = obj.netType(idx);
    end
    function [v,idx] = getNetMode(obj,varargin)
      idx = obj.select(varargin{:});
      v = obj.netMode(idx);
    end
    function [v,idx] = getTrainType(obj,varargin)
      idx = obj.select(varargin{:});
      v = obj.trainType(idx);
    end
    function setTrainType(obj,v,varargin)
      idx = obj.select(varargin{:});
      ncurr = numel(idx);
      obj.trnType(idx) = DeepModelChainOnDisk.toCellArray(v,ncurr);
    end
    function [v,idx] = getTrkTaskKeyword(obj,varargin)
      idx = obj.select(varargin{:});
      v = obj.trkTaskKeyword(idx);
    end
    function v = getTrkTSstr(obj)
      v = obj.trkTSstr;
    end
    function setTrkTaskKeyword(obj,v,varargin)
      idx = obj.select(varargin{:});
      ncurr = numel(idx);
      obj.trkTaskKeyword(idx) = DeepModelChainOnDisk.toCellArray(v,ncurr);
    end
    function setTrkTSstr(obj,v),
      assert(ischar(v));
      obj.trkTSstr = v;
    end
    function [v,idx] = getIterCurr(obj,varargin)
      idx = obj.select(varargin{:});
      v = obj.iterCurr(idx);
    end
    function [v,idx] = getIterFinal(obj,varargin)
      idx = obj.select(varargin{:});
      v = obj.iterFinal(idx);
    end
    function [v,idx] = getNLabels(obj,varargin)
      idx = obj.select(varargin{:});
      v = obj.nLabels(idx);
    end
    function [v,idx] = getFollowsObjDet(obj,varargin)
      idx = obj.select(varargin{:});
      v = obj.tfFollowsObjDet(idx);
    end
    function v = getRootDir(obj)
      v = obj.rootDir ;
    end
    function setRootDir(obj,v)
      obj.rootDir = v ;
    end
    function v = getFileSep(obj)
      v = obj.filesep;
    end
    function setFileSep(obj,v)
      obj.filesep = v;
    end
    function checkFileSep(obj)
      if isempty(obj.filesep),
        obj.filesep = '/';
      end
    end
    function setPrevModels(obj,prev_models,varargin)
      idx = obj.select(varargin{:});
      assert(numel(prev_models) == numel(idx));
      obj.prev_models(idx) = prev_models;
    end
    function v = job2view(obj,ijob)
      v = unique(obj.view(obj.jobidx==ijob));
    end
    function v = job2stage(obj,ijob)
      v = unique(obj.stage(obj.jobidx==ijob));
    end
    function resetIsMultiViewTracker(obj)
      obj.isMultiViewTracker = numel(unique(obj.view))>1;
    end
    function resetIsMultiStageTracker(obj)
      obj.isMultiStageTracker = numel(unique(obj.stage))>1;
    end
    function idx = setFollowsObjDet(obj,isObjDet,varargin)
      idx = obj.select(varargin{:});
      obj.tfFollowsObjDet(idx) = isObjDet;
    end
    function resetFollowsObjDet(obj)
      stages = obj.getStages();
      ustages = unique(stages);
      obj.setFollowsObjDet(false,'stage',1);
      for stagei = 2:numel(ustages),
        stagecurr = ustages(stagei);
        stageprev = stagecurr - 1;
        idxcurr = obj.select('stage',stagecurr);
        for ii = 1:numel(idxcurr),
          icurr = idxcurr(ii);
          [~,vwcurr,~,~] = obj.ind2sub(icurr);
          netModePrev = DeepModelChainOnDisk.getCheckSingle(obj.getNetMode('stage',stageprev,'view',vwcurr));
          if isempty(netModePrev),
            isObjDet = false;
          else
            isObjDet = netModePrev.isObjDet;
          end
          obj.setFollowsObjDet(isObjDet,icurr);
        end
      end
    end
    % dirNetLnx can depend on netType, so return a cell
    function [v,idx] = dirNetLnx(obj,varargin)
      idx = obj.select(varargin{:});
      v = cell(1,numel(idx));
      for ii = 1:numel(idx),
        icurr = idx(ii);
        v{ii} = [obj.rootDir obj.filesep obj.projID obj.filesep char(obj.netType{icurr})];
      end
    end
    function [v,idx] = getNetDescriptor(obj,varargin)
      idx = obj.select(varargin{:});
      v = cell(1,numel(idx));
      for ii = 1:numel(idx),
        icurr = idx(ii);
        v{ii} = sprintf('%s_view%d',char(obj.netType{icurr}),obj.view(icurr));
      end
    end
    function [v,idx] = dirViewLnx(obj,varargin)
      idx = obj.select(varargin{:});
      v = cell(1,numel(idx));
      for ii = 1:numel(idx),
        icurr = idx(ii);
        v{ii} = [obj.rootDir obj.filesep obj.projID obj.filesep char(obj.netType{icurr}) obj.filesep sprintf('view_%d',obj.view(icurr))];
      end
    end
    function [v,idx] = dirModelChainLnx(obj,varargin)
      [dirViewLnxs,idx] = obj.dirViewLnx(varargin{:});
      v = cell(1,numel(idx));
      for ii = 1:numel(idx),
        icurr = idx(ii);
        v{ii} = [dirViewLnxs{ii} obj.filesep obj.modelChainID{icurr}];
      end
    end
    function [v,idx] = dirTrkOutLnx(obj,varargin)
      [dirModelChainLnxs,idx] = obj.dirModelChainLnx(varargin{:});
      v = cell(1,numel(dirModelChainLnxs));
      for icurr = 1:numel(dirModelChainLnxs),
        v{icurr} = [dirModelChainLnxs{icurr} obj.filesep 'trk'];
      end
    end 
    function v = dirAptRootLnx(obj)
      v = [obj.rootDir obj.filesep 'APT'];
    end
    function [v,idx] = trainConfigLnx(obj,varargin)
      [trainConfigNames,idx] = obj.trainConfigName(varargin{:});
      v = cell(1,numel(idx));
      for icurr = 1:numel(idx),
        v{icurr} = [obj.dirProjLnx obj.filesep trainConfigNames{icurr} obj.configFileExt];
      end
    end
    function [v,idx] = trainConfigName(obj,varargin)
      idx = obj.select(varargin{:});
      v = cell(1,numel(idx));
      for ii = 1:numel(idx),
        icurr = idx(ii);
        if numel(obj.trainConfigNameOverride) >= icurr && ~isempty(obj.trainConfigNameOverride{icurr}),
          v{ii} = obj.trainConfigNameOverride{icurr};
        else
          v{ii} = sprintf('%s_%s',obj.modelChainID{icurr},obj.trainID{icurr});
        end
      end
    end
    function [v,idx] = lblStrippedLnx(obj,varargin)
      warning('OBSOLETE: Reference to stripped lbl file. We are trying to remove these. Let Kristin know how you got here!');
      [trainConfigNames,idx] = obj.trainConfigName(varargin{:});
      v = cell(1,numel(idx));
      for icurr = 1:numel(idx),
        v{icurr} = [obj.dirProjLnx obj.filesep trainConfigNames{icurr} '.lbl'];
      end
    end
    % full path to json config for this train session
    function [v,idx] = trainJsonLnx(obj,varargin)
      [trainConfigNames,idx] = obj.trainConfigName(varargin{:});
      v = cell(1,numel(idx));
      for icurr = 1:numel(idx),
        v{icurr} = [obj.dirProjLnx obj.filesep trainConfigNames{icurr} '.json'];
      end
    end
     % full path to training annotations - unused
    function v = trainPackLnx(obj)
      v = [obj.dirProjLnx obj.filesep obj.trainPackName];      
    end
     % full path to training annotations - used - unique
    function v = trainLocLnx(obj)
      v = [obj.dirProjLnx obj.filesep obj.trainLocName];
    end

    function [v,idx] = trainCmdfileLnx(obj,varargin)
      [v,idx] = obj.trainCmdfileName(varargin{:});
      for icurr = 1:numel(v),
        v{icurr} = [obj.dirProjLnx obj.filesep v{icurr}];
      end
    end

    function [v,idx] = trainCmdfileName(obj,varargin)
      idx = obj.select(varargin{:});
      v = cell(1,numel(idx));
      isMultiViewJob = obj.isMultiViewJob(idx);
      netModeName = obj.netModeName(idx);
      for ii = 1:numel(idx),
        icurr = idx(ii);
        if isMultiViewJob(ii), % this job is for multiple views
          viewstr = '';
        else
          viewstr = sprintf('view%d',obj.view(icurr));
        end
        v{icurr} = sprintf('%s%s_%s_%s',obj.modelChainID{icurr},viewstr,obj.trainID{icurr},netModeName{ii});
        if obj.isSplit,
          v{icurr} = [v{icurr},'.sh'];
        else
          v{icurr} = [v{icurr},'.cmd'];
        end
      end
    end
    function [v,idx] = splitfileLnx(obj,varargin)
      [v,idx] = obj.splitfileName(varargin{:});
      for icurr = 1:numel(v),
        v{icurr} = [obj.dirProjLnx obj.filesep v{icurr}];
      end
    end
    function [v,idx] = splitfileName(obj,varargin)
      idx = obj.select(varargin{:});
      % not sure how multiview works yet
      if obj.isSplit
        v = cell(1,numel(idx));
        for ii = 1:numel(idx),
          icurr = idx(ii);
          v{icurr} = sprintf('%s_view%d_split.json',obj.modelChainID{icurr},obj.view(icurr));
        end
      else
        v = repmat({'__NOSPLIT__'},[1,numel(idx)]);
      end
    end
    function [v,idx] = valresultsLnx(obj,varargin)
      [valresultsName,idx] = obj.valresultsName(varargin{:});
      [dirTrkOutLnx] = obj.dirTrkOutLnx(idx);
      v = cell(1,numel(idx));
      for icurr = 1:numel(idx),
        v{icurr} = [dirTrkOutLnx{icurr} obj.filesep valresultsName{icurr}];
      end
    end

    function [v,idx] = valresultsBaseLnx(obj,varargin)
      [valresultsNameBase,idx] = obj.valresultsNameBase(varargin{:});
      [dirTrkOutLnx] = obj.dirTrkOutLnx(idx);
      v = cell(1,numel(idx));
      for icurr = 1:numel(idx),
        v{icurr} = [dirTrkOutLnx{icurr} obj.filesep valresultsNameBase{icurr}];
      end
    end    
    function [v,idx] = valresultsName(obj,varargin)
      idx = obj.select(varargin{:});
      v = cell(1,numel(idx));
      for ii = 1:numel(idx),
        icurr = idx(ii);
        v{ii} = sprintf('%s_%d.mat',obj.trainID{icurr},obj.view(icurr));
      end
    end
    function [v,idx] = valresultsNameBase(obj,varargin)
      idx = obj.select(varargin{:});
      v = obj.trainID(idx);
    end 
    function [v,idx] = errfileLnx(obj,varargin)
      [errfileName,idx] = obj.errfileName(varargin{:});
      v = cell(1,numel(idx));
      for icurr = 1:numel(idx),
        v{icurr} = [obj.dirProjLnx obj.filesep errfileName{icurr}];
      end
    end
    function [v,idx] = errfileName(obj,varargin)
      idx = obj.select(varargin{:});
      v = cell(1,numel(idx));
      isMultiViewJob = obj.isMultiViewJob(idx);
      for ii = 1:numel(idx),
        icurr = idx(ii);
        if isMultiViewJob(ii), % this job is for multiple views
          viewstr = '';
        else
          viewstr = sprintf('view%d',obj.view(icurr));
        end
        netModeName = obj.netModeName(icurr);
        v{icurr} = sprintf('%s%s_%s_%s.err',obj.modelChainID{icurr},viewstr,obj.trainID{icurr},netModeName{1});
      end
    end
    function v = isMultiStageJob(obj,varargin)
      idx = obj.select(varargin{:});
      v = false(1,obj.n);
      for ijob = 1:obj.njobs,
        stagecurr = obj.job2stage(ijob);
        v(obj.jobidx==ijob) = numel(stagecurr) > 1;
      end
      v = v(idx);
    end
    function v = isMultiViewJob(obj,varargin)
      idx = obj.select(varargin{:});
      v = false(1,obj.n);
      for ijob = 1:obj.njobs,
        viewcurr = obj.job2view(ijob);
        v(obj.jobidx==ijob) = numel(viewcurr) > 1;
      end
      v = v(idx);
    end

    function [netmodestr,idx] = netModeName(obj,varargin)
      idx = obj.select(varargin{:});
      netmodestr = cell(1,numel(idx));
      isMultiStageJob = obj.isMultiStageJob(idx);
      for i = 1:numel(idx),
        if isMultiStageJob(i),
          netmodestr{i} = 'multistage';
        else
          netmodestr{i} = obj.netMode{idx(i)}.shortCode;
        end
      end
    end
    function [v,idx] = trainLogLnx(obj,varargin)
      [trainLogName,idx] = obj.trainLogName(varargin{:});
      v = cell(1,numel(trainLogName));
      for icurr = 1:numel(trainLogName),
        v{icurr} = [obj.dirProjLnx obj.filesep trainLogName{icurr}];
      end
    end
    function [v,idx] = trainLogName(obj,varargin)
      idx = obj.select(varargin{:});
      v = cell(1,numel(idx));
      isMultiViewJob = obj.isMultiViewJob(idx);
      for ii = 1:numel(idx),
        icurr = idx(ii);
        if isMultiViewJob(ii), % this job is for multiple views
          viewstr = '';
        else
          viewstr = sprintf('view%d',obj.view(icurr));
        end
        if isequal(obj.trainType{icurr},DLTrainType.Restart),
          restartstr = obj.restartTS{icurr};
        else
          restartstr = '';
        end
        v{ii} = sprintf('%s%s_%s_%s_%s%s.log',obj.modelChainID{icurr},viewstr,...
          obj.trainID{icurr},DeepModelChainOnDisk.getCheckSingle(obj.netModeName(icurr)),...
          lower(char(obj.trainType{icurr})),restartstr);
      end
    end
    function [v,idx] = trkName(obj,varargin)
      [idx] = obj.select(varargin{:});
      v = cell(1,numel(idx));
      for ii = 1:numel(idx),
        icurr = idx(ii);
        v{ii} = sprintf('%s_%s_vw%d_%s',obj.trkTaskKeyword{icurr},obj.modelChainID, ...
          obj.view(icurr),obj.trkTSstr);
      end
    end    
    
    function [v,idx] = trkExtLnx(obj,ext,varargin)
      [v,idx] = obj.trkExtName(ext,varargin{:});
      [dirTrkOutLnx] = obj.dirTrkOutLnx(idx);
      for icurr = 1:numel(idx),
        v{icurr} = [dirTrkOutLnx{icurr} obj.filesep v{icurr}];
      end
    end
    function [v,idx] = trkExtName(obj,ext,varargin)
      [v,idx] = obj.trkName(varargin{:});
      v = cellfun(@(x) [x,ext],v,'Uni',0);
%       v = sprintf('%s_%s_vw%d_%s.log',obj.trkTaskKeyword,obj.modelChainID, ...
%         obj.view,obj.trkTSstr);
    end
    function [v,idx] = trkLogLnx(obj,varargin)
      [v,idx] = obj.trkExtLnx('.log',varargin{:});
    end
    function [v,idx] = trkLogName(obj,varargin)
      [v,idx] = obj.trkExtName('.log',varargin{:});
%       v = sprintf('%s_%s_vw%d_%s.log',obj.trkTaskKeyword,obj.modelChainID, ...
%         obj.view,obj.trkTSstr);
    end
    function [v,idx] = trkErrfileLnx(obj,varargin)
      [v,idx] = obj.trkExtLnx('.err',varargin{:});
    end
    function [v,idx] = trkErrName(obj,varargin)
      [v,idx] = obj.trkExtName('.err',varargin{:});
%       v = sprintf('%s_%s_vw%d_%s.err',obj.trkTaskKeyword,obj.modelChainID, ...
%         obj.view,obj.trkTSstr);
    end
    function [v,idx] = trkCmdfileLnx(obj,varargin)
      [v,idx] = obj.trkExtLnx('.cmd',varargin{:});
    end
    function [v,idx] = trkCmdfileName(obj,varargin)
      [v,idx] = obj.trkExtName('.cmd',varargin{:});
    end  
    function [v,idx] = trkSnapshotLnx(obj,varargin)
      [trkSnapshotName,idx] = obj.trkSnapshotName(varargin{:});
      [dirTrkOutLnx] = obj.dirTrkOutLnx(idx);
      v = cell(1,numel(idx));
      for icurr = 1:numel(idx),
        v{icurr} = [dirTrkOutLnx{icurr} obj.filesep trkSnapshotName{icurr}];
      end
    end
    function [v,idx] = trkSnapshotName(obj,varargin)
      idx = obj.select(varargin{:});
      v = cell(1,numel(idx));
      for ii = 1:numel(idx),
        icurr = idx(ii);
        v{ii} = sprintf('%s_%s_vw%d_%s.aptsnapshot',obj.trkTaskKeyword{icurr},obj.modelChainID{icurr}, ...
          obj.view{icurr},obj.trkTSstr);
      end
    end
    function [v,idx] = gtOutfilePartLnx(obj,varargin)
      [v,idx] = obj.gtOutfileLnx(varargin{:});
      v = cellfun(@(x) [x,'.part'],v,'Uni',0);
    end
    function [v,idx] = gtOutfileLnx(obj,varargin)
      [gtOutfileName,idx] = obj.gtOutfileName(varargin{:});
      [dirTrkOutLnx] = obj.dirTrkOutLnx(idx);
      v = cell(1,numel(idx));
      for icurr = 1:numel(idx),
        v{icurr} = [dirTrkOutLnx{icurr} obj.filesep gtOutfileName{icurr}];
      end
    end
    function [v,idx] = gtOutfileName(obj,varargin)
      idx = obj.select(varargin{:});
      v = cell(1,numel(idx));
      for ii = 1:numel(idx),
        icurr = idx(ii);
        v{ii} = sprintf('gtcls_vw%d_%s.mat',obj.view(icurr),obj.trkTSstr);
      end
    end
    function [v,idx] = killTokenLnx(obj,varargin)
      [killTokenName,idx] = obj.killTokenName(varargin{:});
      isMultiViewJob = obj.isMultiViewJob(idx);
      if ~all(isMultiViewJob),
        dirModelChainLnx = obj.dirModelChainLnx(varargin{:});
      end
      v = cell(1,numel(idx));
      for i = 1:numel(idx),
        if isMultiViewJob(i),
          v{i} = [obj.dirProjLnx obj.filesep killTokenName{i}];
        else
          v{i} = [dirModelChainLnx{i} obj.filesep killTokenName{i}];
        end
      end
    end    
    function [v,idx] = killTokenName(obj,varargin)
      idx = obj.select(varargin{:});
      v = cell(1,numel(idx));
      for ii = 1:numel(idx),
        icurr = idx(ii);
        if isequal(obj.trainType{icurr},DLTrainType.Restart),
          restartstr = obj.restartTS{icurr};
        else
          restartstr = '';
        end
        v{icurr} = sprintf('%s_%s%s.KILLED',obj.trainID{icurr},lower(char(obj.trainType{icurr})),restartstr);
      end
    end  
    function [v,idx] = trainDataLnx(obj,varargin)
      [dirModelChainLnx,idx] = obj.dirModelChainLnx(varargin{:});
      v = cell(1,numel(idx));
      for icurr = 1:numel(idx),
        v{icurr} = [dirModelChainLnx{icurr} obj.filesep 'traindata.json'];
      end
    end
    function v = trainContainerName(obj,varargin)
      idx = obj.select(varargin{:});
      DeepModelChainOnDisk.getCheckSingle(obj.getJobs(idx));
      isMultiViewJob = obj.isMultiViewJob(idx);
      if any(isMultiViewJob),
        viewstr = '';
      else
        view1 = DeepModelChainOnDisk.getCheckSingle(obj.getView(idx));
        viewstr = sprintf('_view%d',view1);
      end
      modelChainID1 = DeepModelChainOnDisk.getCheckSingle(obj.getModelChainID(idx));
      trainID1 = DeepModelChainOnDisk.getCheckSingle(obj.getTrainID(idx));
      netModeName1 = DeepModelChainOnDisk.getCheckSingle(obj.netModeName(idx));
      v = ['train_' modelChainID1 '_' trainID1 '_' netModeName1 viewstr];
    end
    function [v,idx] = trainFinalModelLnx(obj,varargin)
      [dirModelChainLnx,idx] = obj.dirModelChainLnx(varargin{:});
      [trainFinalModelName] = obj.trainFinalModelName(idx);
      v = cell(1,numel(idx));
      for icurr = 1:numel(idx),
        v{icurr} = [dirModelChainLnx{icurr} obj.filesep trainFinalModelName{icurr}];
      end
    end
    function [v,idx] = trainCompleteArtifacts(obj,varargin)
      [trainFinalModelLnx,idx] = obj.trainFinalModelLnx(varargin{:});
      v = cell(1,numel(idx));
      for icurr = 1:numel(idx),
        v{icurr} = {trainFinalModelLnx{icurr}}; %#ok<CCAT1> 
      end
    end
    function [v,idx] = trainCurrModelLnx(obj,varargin)
      [dirModelChainLnx,idx] = obj.dirModelChainLnx(varargin{:});
      [trainCurrModelName] = obj.trainCurrModelName(idx);
      v = cell(1,numel(idx));
      for icurr = 1:numel(idx),
        v{icurr} = [dirModelChainLnx{icurr} obj.filesep trainCurrModelName{icurr}];
      end
    end
    function [v,idx] = trainCurrModelSuffixlessLnx(obj,varargin)
      [v,idx] = obj.trainCurrModelLnx(varargin{:});
      v = regexprep(v,'\.index$','');
    end
    function [v,idx] = trainFinalModelName(obj,varargin)
      idx = obj.select(varargin{:});
      v = cell(1,numel(idx));
      for ii = 1:numel(idx),
        icurr = idx(ii);
        pat = obj.netType{icurr}.mdlNamePat;
        v{ii} = sprintf(pat,obj.iterFinal(icurr));
      end
    end    
    function [v,idx] = trainCurrModelName(obj,varargin)
      idx = obj.select(varargin{:});
      v = cell(1,numel(idx));
      for ii = 1:numel(idx),
        icurr = idx(ii);
        pat = obj.netType{icurr}.mdlNamePat;
        v{ii} = sprintf(pat,obj.iterCurr(icurr));
      end
    end
    function [v,idx] = trainImagesNameLnx(obj,varargin)
      [dirModelChainLnx,idx] = obj.dirModelChainLnx(varargin{:});
      v = cell(1,numel(idx));
      for icurr = 1:numel(idx),
        v{icurr} = [dirModelChainLnx{icurr} obj.filesep obj.trainingImagesName];
      end
    end
    function [v,idx] = trainModelGlob(obj,varargin)
      idx = obj.select(varargin{:});
      v = cell(1,numel(idx));
      for ii = 1:numel(idx),
        icurr = idx(ii);
        v{ii} = obj.netType{icurr}.mdlGlobPat;
      end
    end
    function [v,idx] = aptRepoSnapshotLnx(obj,varargin)
      [aptRepoSnapshotName,idx] = obj.aptRepoSnapshotName(varargin{:});
      v = cell(1,numel(idx));
      for icurr = 1:numel(idx),
        v{icurr} = [obj.dirProjLnx obj.filesep aptRepoSnapshotName{icurr}];
      end
    end
    function [v,idx] = aptRepoSnapshotName(obj,varargin)
      idx = obj.select(varargin{:});
      v = cell(1,numel(idx));
      for ii = 1:numel(idx),
        icurr = idx(ii);
        v{ii} = sprintf('%s_%s.aptsnapshot',obj.modelChainID{icurr},obj.trainID{icurr});
      end
    end
    function v = dockerImgPath(obj,backend) %#ok<INUSL> 
      % todo: this should depend on what type of tracker
      v = backend.dockerimgroot;
      if ~isempty(backend.dockerimgtag)
        v = [v ':' backend.dockerimgtag];
      end
    end
    function [v,idx] = getPrevModels(obj,varargin)
      idx = obj.select(varargin{:});
      v = obj.prev_models(idx);
    end
  end
  methods (Access=protected)
    function obj2 = copyElement(obj)
      obj2 = copyElement@matlab.mixin.Copyable(obj);
%       if ~isempty(obj.reader)
%         obj2.reader = copy(obj.reader);
%       end
    end
  end
  
  methods
    function obj = DeepModelChainOnDisk(varargin)

      % allow to call with no inputs, but then all responsibility for
      % properly setting variables is on outside code.
      if nargin == 0,
        return;
      end
        
      nmodels = [];
      for iprop=1:2:numel(varargin)
        prop = varargin{iprop};
        if strcmp(prop,'nmodels'),
          nmodels = varargin{iprop+1};
        else
          obj.(varargin{iprop}) = varargin{iprop+1};
        end
      end

      obj.autoFix(nmodels);
    end
    
    function autoFix(obj,nmodels)
      if nargin < 2 || isempty(nmodels),
        nmodels = max([numel(obj.view),numel(obj.jobidx),numel(obj.stage),numel(obj.splitIdx)]);
      end
      if isempty(obj.view),
        obj.view = zeros(1,nmodels);
      end
      if isempty(obj.jobidx),
        obj.jobidx = ones(1,nmodels);
      end
      if isempty(obj.stage),
        obj.stage = ones(1,nmodels);
      end
      if isempty(obj.splitIdx),
        obj.splitIdx = zeros(1,nmodels);
      end
      obj.jobidx = obj.jobidx(:)';
      obj.view = obj.view(:)';
      obj.stage = obj.stage(:)';
      obj.splitIdx = obj.splitIdx(:)';
      assert(nmodels==numel(obj.jobidx));
      assert(nmodels==numel(obj.stage));
      assert(nmodels==numel(obj.view));
      assert(nmodels==numel(obj.splitIdx));
      
      % put in more checking, or robustness to skimping on parameters
      assert(~isempty(obj.modelChainID));
      obj.modelChainID = DeepModelChainOnDisk.toCellArray(obj.modelChainID,nmodels);
      assert( (numel(obj.modelChainID)==nmodels) && all(cellfun(@ischar,obj.modelChainID)) );
      assert(~isempty(obj.trainID));
      obj.trainID = DeepModelChainOnDisk.toCellArray(obj.trainID,nmodels);
      assert(~isempty(obj.projID));
      assert(~isempty(obj.rootDir));
      assert(~isempty(obj.netType));
      obj.netType = DeepModelChainOnDisk.toCellArray(obj.netType,nmodels);
      assert(~isempty(obj.netMode));
      obj.netMode = DeepModelChainOnDisk.toCellArray(obj.netMode,nmodels);
      assert(~isempty(obj.trainType));
      obj.trainType = DeepModelChainOnDisk.toCellArray(obj.trainType,nmodels);
      if isempty(obj.trkTaskKeyword),
        obj.trkTaskKeyword = repmat({''},[1,nmodels]);
      else
        obj.trkTaskKeyword = DeepModelChainOnDisk.toCellArray(obj.trkTaskKeyword,nmodels);
      end
      if isempty(obj.restartTS),
        obj.restartTS = repmat({''},[1,nmodels]);
      else
        obj.restartTS = DeepModelChainOnDisk.toCellArray(obj.restartTS,nmodels);
      end
      if isempty(obj.trainConfigNameOverride),
        obj.trainConfigNameOverride = repmat({''},[1,nmodels]);
      else
        obj.trainConfigNameOverride = DeepModelChainOnDisk.toCellArray(obj.trainConfigNameOverride,nmodels,true);
      end
      if isempty(obj.prev_models),
        obj.prev_models = repmat({''},[1,nmodels]);
      else
        obj.prev_models = DeepModelChainOnDisk.toCellArray(obj.prev_models,nmodels,true);
      end
      
      for i = 1:numel(obj.netType),
        if ischar(obj.netType{i}),
          obj.netType{i} = DLNetType.(obj.netType{i});
        end
      end
      for i = 1:numel(obj.netMode),
        if ischar(obj.netMode{i}),
          obj.netMode{i} = DLNetMode.(obj.netMode{i});
        end
      end
      if isempty(obj.isMultiViewTracker),
        obj.resetIsMultiViewTracker();
      end
      if isempty(obj.isMultiStageTracker),
        obj.resetIsMultiStageTracker();
      end
      if isempty(obj.tfFollowsObjDet),
        obj.tfFollowsObjDet = false(1,nmodels);
        obj.resetFollowsObjDet();
      end
      
      if isempty(obj.iterFinal),
        obj.iterFinal = nan([1,nmodels]);
      end
      if isempty(obj.iterCurr),
        obj.iterCurr = nan([1,nmodels]);
      end
      if isempty(obj.nLabels),
        obj.nLabels = nan([1,nmodels]);
      elseif isscalar(obj.nLabels),
        obj.nLabels = repmat(obj.nLabels,[1,nmodels]);
      end
      obj.checkFileSep();
      if isempty(obj.isRemote_),
        obj.isRemote_ = false ;
      end
%       if isempty(obj.reader),
%         obj.reader = DeepModelChainReaderLocal();
%       end
    end  % function

    function tf = isPostRefactor202208(obj)
      nmodels = max([numel(obj.view),numel(obj.jobidx),numel(obj.stage),numel(obj.splitIdx)]);
      tf = nmodels==numel(obj.jobidx) && nmodels==numel(obj.stage) && ...
          nmodels==numel(obj.view) && nmodels==numel(obj.splitIdx) && ...
          nmodels==numel(obj.modelChainID) && ...
          nmodels==numel(obj.trainID) && ...
          nmodels==numel(obj.projID) && ...
          nmodels==numel(obj.netType) && ...
          nmodels==numel(obj.netMode);
      %tf = tf && isequal(numel(unique(obj.view)) > 1,obj.isMultiView);
      %tf = tf && isequal(numel(unique(obj.stage)) > 1,obj.isMultiStage);
    end

    function merge(obj,dmc)
      assert(isequaln(obj.projID,dmc.projID));
      tocheck = {'rootDir','trkTSstr'};
      for i = 1:numel(tocheck),
        prop = tocheck{i};
        if ~isequal(obj.(prop),dmc.(prop)),
          warning('Differing values for %s, using %s',prop,obj.(prop));
        end
      end
      tocat = obj.props_numeric;
      for i = 1:numel(tocat),
        prop = tocat{i};
        if len(obj.(prop)) < obj.n,
          v = nan(1,obj.n);
          v(1:numel(obj.(prop))) = obj.(prop);
        else
          v = obj.(prop);
        end
        obj.(prop) = [v,dmc.(prop)];
      end
      tocat = obj.props_cell;
      for i = 1:numel(tocat),
        prop = tocat{i};
        if len(obj.(prop)) < obj.n,
          v = cell(1,obj.n);
          v(1:numel(obj.(prop))) = obj.(prop);
        else
          v = obj.(prop);
        end
        obj.(prop) = [v,dmc.(prop)];
      end
      tocat = obj.props_bool;
      for i = 1:numel(tocat),
        prop = tocat{i};
        if len(obj.(prop)) < obj.n,
          v = false(1,obj.n);
          v(1:numel(obj.(prop))) = obj.(prop);
        else
          v = obj.(prop);
        end
        obj.(prop) = [v,dmc.(prop)];
      end
      obj.resetFollowsObjDet();
      obj.resetIsMultiViewTracker();
      obj.resetIsMultiStageTracker();
    end

    function dmc = selectSubset(obj,varargin)
      idx = obj.select(varargin{:});
      dmc = obj.copy();
      props = [obj.props_numeric,obj.props_cell,obj.props_bool];
      for i = 1:numel(props),
        prop = props{i};
        ncurr = numel(obj.(prop));
        if any(idx > ncurr),
          warning('Property %s is missing elements',prop);
        end
        dmc.(prop) = obj.(prop)(idx(idx <= ncurr));
      end
    end

    function dmcs = splitByJob(obj)
      unique_jobs = unique(obj.jobidx);
      isfirst = true;
      for ijob = unique_jobs(:)',
        idx = obj.select('jobidx',ijob);
        dmccurr = obj.selectSubset(idx);
        if isfirst,
          dmcs = dmccurr;
          isfirst = false;
        else
          dmcs(end+1) = dmccurr; %#ok<AGROW> 
        end
      end
    end


    function [dmc1,dmc2] = split(obj,varargin)
      idx1 = obj.select(varargin{:});
      idx2 = true(1,obj.n);
      idx2(idx1) = false;
      idx2 = find(idx2);
      dmc1 = obj.selectSubset(idx1);
      dmc2 = obj.selectSubset(idx2);
    end

    function obj2 = copyAndDetach(obj)
      obj2 = copy(obj);
      obj2.prepareBg();
    end    

    function prepareBg(obj)
      % 'Detach' a DMC for use in bg processes
      % Typically you would deepcopy the DMC before calling this
      for i=1:numel(obj)
%         if ~isempty(obj(i).reader)
%           obj(i).reader.prepareBg();
%         end
      end
    end

    function [fileinfo,idx] = trainFileInfo(obj,varargin) 
      idx = obj.select(varargin{:});
      fileinfo = struct() ;
      fileinfo.modelChainID = obj.modelChainID(idx);
      fileinfo.trainID = obj.trainID(idx);
      fileinfo.dlconfig = obj.trainConfigLnx;
      fileinfo.trainlocfile = obj.trainLocLnx;
      fileinfo.cache = obj.rootDir;
      fileinfo.errfile = obj.errfileLnx(idx);
      fileinfo.netType = obj.netType(idx);
      fileinfo.netMode = obj.netMode(idx);
      fileinfo.netModeName = obj.netModeName(idx);
      fileinfo.view = obj.view(idx);
      fileinfo.jobidx = obj.jobidx(idx);
      fileinfo.stage = obj.stage(idx);
      fileinfo.splitidx = obj.splitIdx(idx);
      fileinfo.selectfun = @(idx1) DeepModelChainOnDisk.selectHelper(fileinfo,idx);
    end

    function [fileinfo,idx] = trainFileInfoSingle(obj,varargin)
      [fileinfo,idx] = obj.trainFileInfo(varargin{:});
      fileinfo.modelChainID = DeepModelChainOnDisk.getCheckSingle(fileinfo.modelChainID);
      fileinfo.trainID = DeepModelChainOnDisk.getCheckSingle(fileinfo.trainID);
      fileinfo.dlconfig = DeepModelChainOnDisk.getCheckSingle(fileinfo.dlconfig);
      % fileinfo.trainlocfile is already a char
      % fileinfo.cache is already a char
      fileinfo.errfile = DeepModelChainOnDisk.getCheckSingle(fileinfo.errfile);
      % fileinfo.netType is a cell still
      % fileinfo.netMode is a cell still
      fileinfo.netModeName = DeepModelChainOnDisk.getCheckSingle(fileinfo.netModeName);
      % fileinfo.view may be a vector
      fileinfo.jobidx = DeepModelChainOnDisk.getCheckSingle(fileinfo.jobidx);
      % fileinfo.stage may be a vector
      % fileinfo.splitidx may be a vector
    end

    function [fileinfo,idx] = trackFileInfo(obj,varargin)
      % TODO update and test
      [fileinfo,idx] = obj.trainFileInfo(varargin{:});
      fileinfo.errfile = obj.trkErrfileLnx(idx);
      fileinfo.logfile = obj.trkLogfileLnx(idx);
      fileinfo.configfile = obj.trkConfigLnx(idx);
    end

%     function lsProjDir(obj,varargin)
%       idx = obj.select(varargin{:});
%       obj.reader.lsProjDir(obj,idx);
%     end
% 
%     function lsModelChainDir(obj)
%       obj.reader.lsModelChainDir(obj);
%     end
% 
%     function lsTrkDir(obj)
%       obj.reader.lsTrkDir(obj);
%     end

    function [g,idx] = modelGlobsLnx(obj,varargin)
      % filesys paths/globs of important parts/stuff to keep
      
      [dmcl,idx] = obj.dirModelChainLnx(varargin{:});
      g = cell(1,numel(idx));
      for ii = 1:numel(idx),
        i = idx(ii);
        netty = obj.netType{i};
        gnetspecific = netty.getModelGlobs(obj.iterCurr(i));
        gnetspecific = cellfun(@(x)[dmcl{ii} obj.filesep x],gnetspecific,'uni',0);
      
        g{ii} = [{ ...
          [obj.dirProjLnx obj.filesep sprintf('%s_%s*',obj.modelChainID{i},obj.trainID{i})]; ... % lbl
          [dmcl{ii} obj.filesep sprintf('%s*',obj.trainID{i})]; ... % toks, logs, errs
          };...
          gnetspecific(:)];
      end
    end

    function [mdlFiles,idx] = findModelGlobsLocal(obj,varargin)
      % Return all key/to-be-saved model files
      %
      % mdlFiles{i}: column cellvec full paths

      idx = obj.select(varargin{:});
      mdlFiles = cell(1,numel(idx));
      for ii = 1:numel(idx),
        i = idx(ii);
        mdlFiles{ii} = cell(0,1);
        globs = obj.modelGlobsLnx(i); 
        assert(numel(globs)==1);
        globs = globs{1};
        for j = 1:numel(globs),
          g = globs{j};
          if contains(g,'*')
            gP = fileparts(g);
            dd = dir(g);
            mdlFilesNew = {dd.name}';
            mdlFilesNew = cellfun(@(x) fullfile(gP,x),mdlFilesNew,'uni',0);
            mdlFiles{ii} = [mdlFiles{i}; mdlFilesNew]; 
          elseif exist(g,'file')>0
            mdlFiles{ii}{end+1,1} = g; 
          end
        end
      end      
    end

    function modelFilesDst = copyModelFiles(obj,newRootDir,debug)
      if nargin < 3,
        debug = false;
      end
      modelFiles = obj.findModelGlobsLocal();
      modelFiles = cat(1,modelFiles{:});
      modelFiles = unique(modelFiles);
      modelFilesDst = strrep(modelFiles,obj.getRootDir(),newRootDir);
      % nothing to do
      if isequal(obj.getRootDir(),newRootDir), 
        return
      end
      if obj.isRemote_
        warningNoTrace('Remote model detected. This will not be migrated.');
        return
      end
      backend = [] ;  % we know we don't need a backend, b/c isRemote is false
      tfsucc = obj.updateCurrInfo(backend);
      if ~all(tfsucc),
        for i = find(~tfsucc(:)'),
          warningNoTrace('Failed to update model iteration count for for net type %s.',...
            char(obj.netType{i}));
        end
      end
      for mndx = 1:numel(modelFiles)
        copyfileensuredir(modelFiles{mndx},modelFilesDst{mndx}); % throws
        if debug,
          fprintf(1,'%s -> %s\n',modelFiles{mndx},modelFilesDst{mndx});
        end
      end
    end
    
    function tfSuccess = updateCurrInfo(obj, backend, varargin)
      % Update .iterCurr by probing filesys
      
      assert(isscalar(obj));
      % will update for all
      maxiter = obj.getMostRecentModel_(backend, varargin{:});
      obj.iterCurr = maxiter;
      tfSuccess = (maxiter >= 0) ;
      
      if any(maxiter>obj.iterFinal),
        warningNoTrace('Current model iteration exceeds specified maximum/target iteration: %s.',...
           DeepTracker.printIter(maxiter,obj.iterFinal));
      end
    end

    function [maxiter,idx] = getMostRecentModel_(obj, backend, varargin)
      if obj.isRemote_ ,
        % maxiter is nan if something bad happened or if DNE
        % TODO allow polling for multiple models at once
        aws = backend.awsec2 ;  % Should probably refactor to do directly using backend methods
        [dirModelChainLnx,idx] = obj.dirModelChainLnx(varargin{:});
        fspollargs = {};
        for i = 1:numel(idx),
          fspollargs = [fspollargs,{'mostrecentmodel' dirModelChainLnx{i}}]; %#ok<AGROW>
        end
        [tfsucc,res] = aws.remoteCallFSPoll(fspollargs);
        if tfsucc
          maxiter = str2double(res(1:numel(idx))); % includes 'DNE'->nan
        else
          maxiter = nan(1,numel(idx));
        end        
      else
        [modelglob,idx] = obj.trainModelGlob(varargin{:});
        [dirModelChainLnx] = obj.dirModelChainLnx(idx);

        maxiter = nan(1,numel(idx));
        for i = 1:numel(idx),
          modelfiles= mydir(fullfile(dirModelChainLnx{i},modelglob{i}));
          if isempty(modelfiles),
            continue;
          end
          for j = 1:numel(modelfiles),
            iter = DeepModelChainOnDisk.getModelFileIter(modelfiles{j});
            if ~isempty(iter),
              maxiter(i) = max(maxiter(i),iter);
            end
          end
        end
      end
    end

    function tf = canTrack(obj)
      % For each known model, whether or not it is in a state that allows tracking.
      % We consider a model to be in a state that allows tracking if it has
      % undergone at least one iteration of training.  Note also that a model that
      % has been trained for nan iterations is not considered to be in a state that
      % allows tracking.
      tf = (obj.iterCurr >= 0) ;
    end
    
    % read nLabels from config file
    function readNLabels(obj)
      if strcmp(obj.configFileExt,'.json'),
        trainLocLnx = obj.trainLocLnx();
        [un,~,idx] = unique(trainLocLnx);
        nLabels1 = nan(1,obj.n);
        for i = 1:numel(un),
          assert(exist(un{i},'file') > 0);
          nLabels1(idx==i) = TrnPack.readNLabels(un{i});
        end
        obj.setNLabels(nLabels1);
      else
        lblStrippedLnx = obj.lblStrippedLnx();
        [un,~,idx] = unique(lblStrippedLnx);
        nLabels1 = nan(1,obj.n);
        for i = 1:numel(un),
          assert(exist(un{i},'file') > 0);
          s = load(obj.lblStrippedLnx,'preProcData_MD_frm','-mat');
          nLabels1(idx==i) = size(s.preProcData_MD_frm,1);
        end
        obj.setNLabels(nLabels1);
      end
    end

    function setNLabels(obj,nLabels,varargin)
      idx = obj.select(varargin{:});
      obj.nLabels(idx) = nLabels;
    end

     function setRestartTS(obj,restartTS)
      if ischar(restartTS),
        obj.restartTS = repmat({restartTS},[1,obj.n]);
      elseif numel(restartTS) == 1,
        obj.restartTS = repmat(restartTS,[1,obj.n]);
      else
        assert(numel(restartTS)==obj.n);
        obj.restartTS = restartTS;
      end
    end

    % whether training has actually started
    function tf = isPartiallyTrained(obj)      
      tf = ~isempty(obj.iterCurr) & ~isnan(obj.iterCurr);
    end
    
    function mirrorToBackend(obj, backend, mode)
      % mode should be 'tracking' or 'training'.
      if ~exist('mode', 'var') || isempty(mode) ,
        mode = 'tracking' ;
      end
      if ~backend.isFilesystemLocal() ,
        if ~obj.isRemote_ ,
          obj.mirrorToRemoteAws_(backend, mode) ;
        end
      end
    end

    function mirrorToRemoteAws_(obj, backend, mode)
      % Take a local DMC and mirror/upload it to the AWS instance aws; 
      % update .rootDir, .reader appropriately to point to model on remote 
      % disk.
      %
      % In practice for the client, this action updates the "latest model"
      % to point to the remote aws instance.
      %
      % PostConditions: 
      % - remote cachedir mirrors this model for key model files; "extra"
      % remote files not removed; identities of existing files not
      % confirmed but naming/immutability of DL artifacts makes this seem
      % safe
      % - .rootDir updated to remote cacheloc
      % - .reader update to AWS reader
      
      assert(isscalar(obj));
      assert(isequal(backend.type, DLBackEnd.AWS), 'Backend must be AWS in order to mirror/upload.');      

      succ = obj.updateCurrInfo(backend);
      if strcmp(mode, 'tracking') && any(~succ) ,
        dmclfail = obj.dirModelChainLnx(find(~succ));
        fstr = sprintf('%s ',dmclfail{:});
        error('Failed to determine latest model iteration in %s.',fstr);
      end
      if isnan(obj.iterCurr) ,
        fprintf('Currently, there is no trained model.\n');
      else
        fprintf('Current model iteration is %s.\n',mat2str(obj.iterCurr));
      end
     
      backend.checkConnection();  % throws error if backend is not connected
      
%       % Upload the already-existing model files
%       mdlFilesWrapper = obj.findModelGlobsLocal();
%       mdlFiles = cat(1,mdlFilesWrapper{:});
%       %pat = obj.rootDir;  % e.g. /home/taylora/.apt/tpb9c364f9_46b1_4ac0_9095_11ddd5c2493c
%       %pat = regexprep(pat,'\\','\\\\');
%       %mdlFilesRemote = regexprep(mdlFiles,pat,DLBackEndClass.RemoteAWSCacheDir);  % DLBackEndClass.RemoteAWSCacheDir is typically '/home/ubuntu/cacheDL'
%       mdlFilesRemoteRaw = cellfun(@(path)(FSPath.replacePrefix(path, obj.rootDir, DLBackEndClass.RemoteAWSCacheDir)), mdlFiles, 'UniformOutput', false) ;
%       mdlFilesRemote = FSPath.standardPath(mdlFilesRemoteRaw);  % transform to linux-style path
%       nMdlFiles = numel(mdlFiles);
%       networkTypeAsString = charArrayFromCharArrayStringOrCellstring(obj.netType) ;
%       fprintf(1,'Upload/mirror %d model files for net %s.\n',nMdlFiles,networkTypeAsString);
%       descstr = sprintf('Model file: %s',networkTypeAsString);
%       for i=1:nMdlFiles
%         src = mdlFiles{i};
%         info = dir(src);
%         leafname = info.name ;
%         filesz = round(info.bytes/2^10);
%         dst = mdlFilesRemote{i};
%         % We just use scpUploadOrVerify which does not confirm the identity
%         % of file if it already exists. These model files should be
%         % immutable once created and their naming (underneath timestamped
%         % modelchainIDs etc) should be pretty/totally unique. 
%         %
%         % Only situation that might cause problems are augmentedtrains but
%         % let's not worry about that for now.
%         backend.scpUploadOrVerify(src, ...
%                                   dst, ...
%                                   sprintf('%s (%s), %d KB',descstr,leafname,filesz), ...
%                                   'destRelative', false) ;  % throws
%       end

      % To support training on AWS, and the fact that a DeepModelChainOnDisk has
      % only a single boolean to represent whether it's local or remote, we're just
      % going to upload everything under fullfile(obj.rootDir, obj.projID) to the
      % backend.  -- ALT, 2024-06-25
      localProjectPath = fullfile(obj.rootDir, obj.projID) ;
      remoteProjectPath = linux_fullfile(DLBackEndClass.RemoteAWSCacheDir, obj.projID) ;  % ensure linux-style path
      backend.rsyncUpload(localProjectPath, remoteProjectPath) ;

%       localDisFromPathIndex = enumerate_all_files_and_folders(localProjectPath) ; % nx1 "dis" struct array, same fields as result of dir()
%       localPathFromPathIndex = path_from_dir_struct(localDisFromPathIndex) ;
%       rawRemotePaths = cellfun(@(path)(FSPath.replacePrefix(path, obj.rootDir, DLBackEndClass.RemoteAWSCacheDir)), ...
%                                localPathFromPathIndex, ...
%                                'UniformOutput', false) ;
%       remotePaths = FSPath.standardPath(rawRemotePaths) ;  % transform to linux-style paths
%       pathCount = numel(localDisFromPathIndex) ;
%       networkTypeAsString = charArrayFromCharArrayStringOrCellstring(obj.netType) ;
%       fprintf('Going to upload %d project files for net %s.\n',pathCount,networkTypeAsString);
%       descstr = sprintf('Model file: %s',networkTypeAsString);
%       for i = 1:pathCount ,
%         localDis = localDisFromPathIndex(i) ;        
%         localPath = localPathFromPathIndex{i} ;  % already computed above, might as well use
%         leafname = localDis.name ;
%         fileSizeInMibibytes = round(localDis.bytes/2^10) ;
%         remotePath = remotePaths{i} ;
%         if localDis.isdir ,
%           backend.mkdir(remotePath) ;
%         else
%           % We just use scpUploadOrVerify which does not confirm the identity
%           % of file if it already exists. These model files should be
%           % immutable once created and their naming (underneath timestamped
%           % modelchainIDs etc) should be pretty/totally unique. 
%           %
%           % Only situation that might cause problems are augmentedtrains but
%           % let's not worry about that for now.
%           backend.scpUploadOrVerify(localPath, ...
%                                     remotePath, ...
%                                     sprintf('%s (%s), %d KB',descstr,leafname,fileSizeInMibibytes), ...
%                                     'destRelative', false) ;  % throws
%         end
%       end

      % if we made it here, upload successful
      
      obj.remoteRootDir_ = DLBackEndClass.RemoteAWSCacheDir ;
      %obj.reader = backend.getDmcReader();
      obj.isRemote_ = true ;
    end
    
    function mirrorFromBackend(obj, backend)
      % If the model chain is remote, download it
      if obj.isRemote_ ,
        obj.mirrorFromRemoteAws_(backend) ;
      end
    end

    function mirrorFromRemoteAws_(obj, backend)
      % Inverse of mirror2remoteAws. Download/mirror model from remote AWS
      % instance to local cache.
      %
      % update .rootDir, .reader appropriately to point to model in local
      % cache.
      %
      % In practice for the client, this action updates the "latest model"
      % to point to the local cache.
      
      assert(isscalar(obj));      
      assert(isequal(backend.type, DLBackEnd.AWS), 'Backend must be AWS in order to mirror/download.');      
      
      cacheDirLocal = obj.localRootDir_ ;
      aws = backend.awsec2;  % Should probably refactor do this directly using backend methods
      [tfexist,tfrunning] = aws.inspectInstance();
      if ~tfexist,
        error('AWS EC2 instance %s could not be found.',aws.instanceID);
      end
      if ~tfrunning,
        [tfsucc,~,warningstr] = aws.startInstance();
        if ~tfsucc,
          error('Could not start AWS EC2 instance %s: %s',aws.instanceID,warningstr);
        end
      end      
      %aws.checkInstanceRunning(); % harderrs if instance isn't running
     
      succ = obj.updateCurrInfo(backend) ;
      if any(~succ),
        dirModelChainLnx = obj.dirModelChainLnx(find(~succ));
        fstr = sprintf('%s ',dirModelChainLnx{:});
        error('Failed to determine latest model iteration in %s.',...
          fstr);
      end
      fprintf('Current model iteration is %s.\n',mat2str(obj.iterCurr));
     
      modelGlobsLnx = obj.modelGlobsLnx();
      for j = 1:obj.n,
        mdlFilesRemote = aws.remoteGlob(modelGlobsLnx{j});
        cacheDirLocalEscd = regexprep(cacheDirLocal,'\\','\\\\');
        mdlFilesLcl = regexprep(mdlFilesRemote,obj.rootDir,cacheDirLocalEscd);
        nMdlFiles = numel(mdlFilesRemote);
        netstr = char(obj.netType{j}); 
        fprintf(1,'Download/mirror %d model files for net %s.\n',nMdlFiles,netstr);
        for i=1:nMdlFiles
          fsrc = mdlFilesRemote{i};
          fdst = mdlFilesLcl{i};
          % See comment in mirror2RemoteAws regarding not confirming ID of
          % files-that-already-exist
          aws.scpDownloadOrVerifyEnsureDir(fsrc,fdst,...
            'sysCmdArgs',{'dispcmd' true 'failbehavior' 'err'}); % throws
        end
      end
      
      % if we made it here, download successful
      
      %obj.rootDir = cacheDirLocal;
      %obj.reader = DeepModelChainReaderLocal();
      obj.isRemote_ = false ;
    end
    
    function [tf,tpdir,idx] = trnPackExists(obj,varargin)
      % Training package exists
      [trainLocLnx,idx] = obj.trainLocLnx(varargin{:});
      tpdir = obj.dirProjLnx;
      tf = exist(tpdir,'dir')>0 & cellfun(@(x) exist(x,'file')>0,trainLocLnx);
    end

  end
  
  
  methods (Static)
    
    function iter = getModelFileIter(filename)
      
      iter = regexp(filename,'deepnet-(\d+)','once','tokens');
      if isempty(iter),
        iter = [];
        return;
      end
      iter = str2double(iter{1});
      
    end
    function view = getModelFileView(filename)

      view = regexp(filename,'view_(\d+)','once','tokens');
      if isempty(view),
        view = 0;
        return;
      end
      view = str2double(view{1});

    end
    function mcId = modelChainIDForSplit(mcIdBase,isplit)
      mcId = sprintf('%s_splt_%03d',mcIdBase,isplit);
    end

    function idx = selectHelper(info,varargin)
      if numel(varargin) == 1,
        idx = varargin{:};
        return;
      end
      if isempty(varargin),
        idx = 1:numel(info.view);
        return;
      end
      [jobidx1,view1,stage1,splitidx1] = myparse(varargin,'jobidx','all','view','all','stage','all','splitidx','all'); 
      idx = true(1,numel(info.view));
      if ~isequal(jobidx1,'all'), 
        idx = idx & jobidx1 == info.jobidx; 
      end
      if ~isequal(view1,'all'), 
        idx = idx & view1 == info.view; 
      end
      if ~isequal(stage1,'all'), 
        idx = idx & stage1 == info.stage; 
      end
      if ~isequal(splitidx1,'all'), 
        idx = idx & splitidx1 == info.splitIdx;
      end
      idx = find(idx);
    end

    function obj = modernize(dmcs,varargin)

      if isempty(dmcs)
        obj = dmcs;
        return
      end

      netmodes = myparse(varargin,'netmode',[]);

      % is this post-refactor from 202208?
      isrefactored = numel(dmcs) == 1 && dmcs.isPostRefactor202208();

      if ~isrefactored,
        % is this multi-view, multi-stage?
        view = [dmcs.view];
        isMultiView = numel(unique(view)) > 1;
        netMode = {};
        for i = 1:numel(dmcs),
          if isempty(dmcs(i).netMode),
            netMode = netmodes(i);
          elseif iscell(dmcs(i).netMode),
            netMode = [netMode,cellfun(@char,dmcs(i).netMode,'Uni',0)]; %#ok<AGROW> 
          elseif ischar(dmcs(i).netMode),
            netMode{end+1} = dmcs(i).netMode; %#ok<AGROW> 
          elseif numel(dmcs(i).netMode) > 1,
            netMode = [netMode,arrayfun(@char,dmcs(i).netMode,'Uni',0)]; %#ok<AGROW> 
          else
            netMode{end+1} = char(dmcs(i).netMode); %#ok<AGROW> 
          end
        end
        isMultiStage = numel(unique(netMode))>1;
        % can't be both
        assert(~(isMultiView&&isMultiStage));
        if isMultiView,
          nmodels = numel(view);
          stage = ones(1,nmodels);
        else
          nmodels = numel(netMode);
          stage = 1:nmodels;
        end
        jobidx = zeros(1,nmodels);
        splitIdx = zeros(1,nmodels);
        modelChainID = cell(1,nmodels);
        trainID = cell(1,nmodels);
        restartTS = repmat({''},[1,nmodels]);
        trainType = cell(1,nmodels);
        netType = cell(1,nmodels);
        netMode = cell(1,nmodels);
        trainConfigNameOverride = cell(1,nmodels);
        iterCurr = nan(1,nmodels);
        iterFinal = nan(1,nmodels);
        nLabels = nan(1,nmodels);
        prev_models = cell(1,nmodels);
        trkTaskKeyword = cell(1,nmodels);

        j = 0;
        for i = 1:numel(dmcs),
          ncurr = numel(dmcs(i).view);
          jobidx(j+1:j+ncurr) = i;
          if ~isnan(dmcs(i).splitIdx),
            splitIdx(j+1:j+ncurr) = dmcs(i).splitIdx;
          end
          modelChainID(j+1:j+ncurr) = DeepModelChainOnDisk.toCellArray(dmcs(i).modelChainID,ncurr,true);
          trainID(j+1:j+ncurr) = DeepModelChainOnDisk.toCellArray(dmcs(i).trainID,ncurr,true);
          trainType(j+1:j+ncurr) = DeepModelChainOnDisk.toCellArray(dmcs(i).trainType,ncurr);
          netType(j+1:j+ncurr) = DeepModelChainOnDisk.toCellArray(dmcs(i).netType,ncurr);
          if isempty(dmcs(i).netMode),
            netmodecurr = netmodes(i);
          else
            netmodecurr = dmcs(i).netMode;
          end
          netMode(j+1:j+ncurr) = DeepModelChainOnDisk.toCellArray(netmodecurr,ncurr);
          if isempty(dmcs(i).trkTaskKeyword),
            trkTaskKeyword(j+1:j+ncurr) = DeepModelChainOnDisk.toCellArray('',ncurr);
          else
            try
              trkTaskKeyword(j+1:j+ncurr) = DeepModelChainOnDisk.toCellArray(dmcs(i).trkTaskKeyword,ncurr);
            catch
              warning('Do not know how to modernize trkTaskKeyword setting to be empty strings.');
              trkTaskKeyword(j+1:j+ncurr) = repmat({''},[1,ncurr]);
            end
          end
          if ~isempty(dmcs(i).restartTS),
            restartTS(j+1:j+ncurr) = DeepModelChainOnDisk.toCellArray(dmcs(i).restartTS,ncurr,true);
          end
          if ~isempty(dmcs(i).trainConfigNameOverride),
            trainConfigNameOverride(j+1:j+ncurr) = DeepModelChainOnDisk.toCellArray(dmcs(i).trainConfigNameOverride,ncurr,true);
          end
          if ~isempty(dmcs(i).iterCurr),
            iterCurr(j+1:j+ncurr) = dmcs(i).iterCurr;
          end
          if ~isempty(dmcs(i).iterFinal),
            iterFinal(j+1:j+ncurr) = dmcs(i).iterFinal;
          end
          if ~isempty(dmcs(i).nLabels),
            nLabels(j+1:j+ncurr) = dmcs(i).nLabels;
          end
          if ~isempty(dmcs(i).prev_models),
            prev_models(j+1:j+ncurr) = DeepModelChainOnDisk.toCellArray(dmcs(i).prev_models,ncurr,true);
          end
          j = j + numel(dmcs(i).view);
        end
        obj = DeepModelChainOnDisk(...
          'view',view,...
          'stage',stage,...
          'jobidx',jobidx,...
          'splitIdx',splitIdx,...
          'modelChainID',modelChainID,...
          'trainID',trainID,...
          'rootDir',dmcs(1).rootDir,...
          'projID',dmcs(1).projID,...
          'netType',netType,...
          'netMode',netMode,...
          'restartTS',restartTS,...
          'trainType',trainType,...
          'trainConfigNameOverride',trainConfigNameOverride,...
          'iterFinal',iterFinal,...
          'iterCurr',iterCurr,...
          'nLabels',nLabels,...
          'prev_models',prev_models,...
          'filesep',dmcs(1).filesep,...
          'trkTaskKeyword',trkTaskKeyword,...
          'trkTSstr',dmcs(1).trkTSstr...
          );
        obj.resetFollowsObjDet();
        obj.resetIsMultiViewTracker();
        obj.resetIsMultiStageTracker();
      else
        assert(numel(dmcs)==1);
        obj = dmcs;
      end
    end
    function info = TrackerInfo(dmc)
      if isempty(dmc),
        info.nmodels = 0;
        info.isTrainStarted = false;
        info.isTrainRestarted = false;
        info.trainStartTS = [];
        info.iterCurr = 0;
        info.iterFinal = nan;
        info.nLabels = 0;
      else
        info.nmodels = dmc.n;
        info.isTrainStarted = true;
        info.isTrainRestarted = strcmp(dmc.trainType,'Restart');
        info.trainStartTS = datenum(dmc.modelChainID,'yyyymmddTHHMMSS');  %#ok<DATNM> 
        assert(all(~isnan(info.trainStartTS)));
        info.iterCurr = dmc.iterCurr;
        if isempty(dmc.iterCurr),
          info.iterCurr = zeros(1,dmc.n);
        else
          info.iterCurr = dmc.iterCurr;
        end
        if isempty(dmc.iterFinal),
          info.iterFinal = zeros(1,dmc.n);
        else
          info.iterFinal = dmc.iterFinal;
        end
        if isempty(dmc.nLabels),
          info.nLabels = nan(1,dmc.n);
        else
          info.nLabels = dmc.nLabels;
        end
      end

    end

    function result = getCheckSingle(s)
      % Checks that all elements of s are the same, in some class-appropriate sense,
      % and returns the common element.  If s is a cell array, unwraps the common
      % element before returning it.
      if isempty(s),
        error('input is empty');
      end
      class_tochar = {'DLNetType','DLNetMode','DLTrainType'};
      if iscell(s),
        if ischar(s{1}),
          assert(numel(unique(s))==1);
        else
          t = class(s{1});
          if ismember(t,class_tochar),
            schar = cellfun(@char,s,'Uni',0);
            assert(numel(unique(schar))==1);
          end
        end
        result = s{1};
      elseif ischar(s) || numel(s) == 1,
        % nothing to do
        result =s ;
      else
        if isnumeric(s),
          assert(numel(unique(s))==1);
        else
          t = class(s);
          if ismember(t,class_tochar),
            schar = arrayfun(@char,s,'Uni',0);
            assert(numel(unique(schar))==1);
          end
        end
        result = s(1);
      end
    end

    function v = toCellArray(v,ncurr,ischarcellarray)

      if nargin < 3,
        ischarcellarray = false;
      end
      
      % sometimes there are cells of cells for some reason?
      while iscell(v) && numel(v) == 1,
        v = v{1};
      end
      if ischarcellarray && isequal(v,[]),
        v = '';
      end
      if iscell(v),
        if numel(v) == 1,
          v = repmat(v,[1,ncurr]);
        else
          assert(numel(v)==ncurr);
        end
      elseif ischarcellarray && isequal(v,[]),
        v = repmat({''},[1,ncurr]);
      elseif ischar(v) || numel(v) == 1,
        v = repmat({v},[1,ncurr]);
      elseif numel(v) == ncurr,
        vin = v;
        v = cell(1,ncurr);
        for i = 1:ncurr,
          v{i} = vin(i);
        end
      else
        error('Could not convert to cell array of length %d',ncurr);
      end

    end

  end
end
    
  
  
