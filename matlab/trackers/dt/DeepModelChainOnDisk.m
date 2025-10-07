classdef DeepModelChainOnDisk < matlab.mixin.Copyable  % matlab.mixin.Copyable is a subclass of handle
  % DMCOD understands the filesystem structure of a deep model. This same
  % structure is used on both remote and local filesystems.
  %
  % DMCOD also now handles tracking output: eg trkfiles and associated
  % log/errfiles/etc; gt results files etc. These are a bit conceptually
  % different but they live underneath the cache/modelchaindir at runtime.

  % As far as I can tell, all the fields of DMCOD contain only value objects in
  % normal usage.  Thus obj.copy() makes an entirely independent copy of obj, with
  % no need to implement a custom copyElement() method.  -- ALT, 2024-12-20
  
  % Note that this is copyable, and we want it to stay that way.  So think
  % before adding properties to it.  -- ALT, 2025-01-07

  properties (Constant)
    configFileExt = '.json'; % switching this to output json file in train/track commands
    gen_strippedlblfile = false; % try disabling the stripped lbl file generation!!
    trainPackName = 'trnpack.json';
    trainLocName = 'loc.json';
    trainingImagesName = 'deepnet_training_samples.mat';

    % Lists of the properties of different kinds that should all have shape 1 x obj.n
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
    localRootDir_ = '' ;  % e.g. /groups/branson/home/bransonk/.apt/tp76715886_6c90_4126_a9f4_0c3d31206ee5
    %remoteRootDir_ = '' ;  % e.g. /home/ubuntu/cacheDL
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
    trainID = {}; % a single modelChainID may be trained multiple times due to 
                  % train-augmentation, so a single modelID may have multiple
                  % trainIDs associated with it. Each (modelChainID,trainID) pair
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
      
    %filesep ='/'; % file separator

    trkTaskKeyword = {}; % arbitrary tracking task keyword; used for tracking output files
    trkTSstr = '';% timestamp for tracking
    prev_models = []; % prev model to start training from
    % isRemote_ = false  
    %   % True iff the "current" version of the model is on a remote AWS filesystem.  
    %   % Underscore means "protected by convention"
  end

  properties (Dependent)
    n  % The number of models.  Most properties of obj should be 1 x n
    nviews
    njobs
    nstages
    rootDir  % The (native) root dir of the DMCoD, as a char array
    %localRootDir
    %remoteRootDir
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
      v = obj.localRootDir_ ;
    end

    function set.rootDir(obj,v)
      obj.localRootDir_ = v ;
    end    

    % function v = get.remoteRootDir(obj)
    %   v = obj.remoteRootDir_ ;
    % end

    % function v = get.localRootDir(obj)
    %   v = obj.localRootDir_ ;
    % end

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

    % function v = isRemote(obj)
    %   v = obj.isRemote_ ;
    % end

    function idx = select(obj,varargin)
      idx = DeepModelChainOnDisk.selectHelper(obj,varargin{:});
    end

    function result = getIdentifiers(obj)
      result = struct();
      result.jobidx = obj.jobidx;
      result.view = obj.view;
      result.stage = obj.stage;
      result.splitIdx = obj.splitIdx;
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
      v = [obj.rootDir '/' obj.projID];
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
      obj.trainType(idx) = DeepModelChainOnDisk.toCellArray(v,ncurr);
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

    % function v = getRootDir(obj)
    %   v = obj.rootDir ;
    % end
    % function setRootDir(obj,v)
    %   obj.rootDir = v ;
    % end

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
        v{ii} = [obj.rootDir '/' obj.projID '/' char(obj.netType{icurr})];
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
        v{ii} = [obj.rootDir '/' obj.projID '/' char(obj.netType{icurr}) '/' sprintf('view_%d',obj.view(icurr))];
      end
    end
    function [v,idx] = dirModelChainLnx(obj,varargin)
      [dirViewLnxs,idx] = obj.dirViewLnx(varargin{:});
      v = cell(1,numel(idx));
      for ii = 1:numel(idx),
        icurr = idx(ii);
        v{ii} = [dirViewLnxs{ii} '/' obj.modelChainID{icurr}];
      end
    end
    function [v,idx] = dirTrkOutLnx(obj,varargin)
      [dirModelChainLnxs,idx] = obj.dirModelChainLnx(varargin{:});
      v = cell(1,numel(dirModelChainLnxs));
      for icurr = 1:numel(dirModelChainLnxs),
        v{icurr} = [dirModelChainLnxs{icurr} '/' 'trk'];
      end
    end 
    function v = dirAptRootLnx(obj)
      v = [obj.rootDir '/' 'APT'];
    end
    function [v,idx] = trainConfigLnx(obj,varargin)
      [trainConfigNames,idx] = obj.trainConfigName(varargin{:});
      v = cell(1,numel(idx));
      for icurr = 1:numel(idx),
        v{icurr} = [obj.dirProjLnx '/' trainConfigNames{icurr} obj.configFileExt];
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
        v{icurr} = [obj.dirProjLnx '/' trainConfigNames{icurr} '.lbl'];
      end
    end
    % full path to json config for this train session
    function [v,idx] = trainJsonLnx(obj,varargin)
      [trainConfigNames,idx] = obj.trainConfigName(varargin{:});
      v = cell(1,numel(idx));
      for icurr = 1:numel(idx),
        v{icurr} = [obj.dirProjLnx '/' trainConfigNames{icurr} '.json'];
      end
    end
     % full path to training annotations - unused
    function v = trainPackLnx(obj)
      v = [obj.dirProjLnx '/' obj.trainPackName];      
    end
     % full path to training annotations - used - unique
    function v = trainLocLnx(obj)
      v = [obj.dirProjLnx '/' obj.trainLocName];
    end

    function [v,idx] = trainCmdfileLnx(obj,varargin)
      [v,idx] = obj.trainCmdfileName(varargin{:});
      for icurr = 1:numel(v),
        v{icurr} = [obj.dirProjLnx '/' v{icurr}];
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
        v{icurr} = [obj.dirProjLnx '/' v{icurr}];
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
        v{icurr} = [dirTrkOutLnx{icurr} '/' valresultsName{icurr}];
      end
    end

    function [v,idx] = valresultsBaseLnx(obj,varargin)
      [valresultsNameBase,idx] = obj.valresultsNameBase(varargin{:});
      [dirTrkOutLnx] = obj.dirTrkOutLnx(idx);
      v = cell(1,numel(idx));
      for icurr = 1:numel(idx),
        v{icurr} = [dirTrkOutLnx{icurr} '/' valresultsNameBase{icurr}];
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
        v{icurr} = [obj.dirProjLnx '/' errfileName{icurr}];
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
        v{icurr} = [obj.dirProjLnx '/' trainLogName{icurr}];
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
        v{icurr} = [dirTrkOutLnx{icurr} '/' v{icurr}];
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
        v{icurr} = [dirTrkOutLnx{icurr} '/' trkSnapshotName{icurr}];
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
        v{icurr} = [dirTrkOutLnx{icurr} '/' gtOutfileName{icurr}];
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
    % function [v,idx] = killTokenLnx(obj,varargin)
    %   [killTokenName,idx] = obj.killTokenName(varargin{:});
    %   isMultiViewJob = obj.isMultiViewJob(idx);
    %   if ~all(isMultiViewJob),
    %     dirModelChainLnx = obj.dirModelChainLnx(varargin{:});
    %   end
    %   v = cell(1,numel(idx));
    %   for i = 1:numel(idx),
    %     if isMultiViewJob(i),
    %       v{i} = [obj.dirProjLnx '/' killTokenName{i}];
    %     else
    %       v{i} = [dirModelChainLnx{i} '/' killTokenName{i}];
    %     end
    %   end
    % end    
    % function [v,idx] = killTokenName(obj,varargin)
    %   idx = obj.select(varargin{:});
    %   v = cell(1,numel(idx));
    %   for ii = 1:numel(idx),
    %     icurr = idx(ii);
    %     if isequal(obj.trainType{icurr},DLTrainType.Restart),
    %       restartstr = obj.restartTS{icurr};
    %     else
    %       restartstr = '';
    %     end
    %     v{icurr} = sprintf('%s_%s%s.KILLED',obj.trainID{icurr},lower(char(obj.trainType{icurr})),restartstr);
    %   end
    % end  
    function [v,idx] = trainDataLnx(obj,varargin)
      [dirModelChainLnx,idx] = obj.dirModelChainLnx(varargin{:});
      v = cell(1,numel(idx));
      for icurr = 1:numel(idx),
        v{icurr} = [dirModelChainLnx{icurr} '/' 'traindata.json'];
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
        v{icurr} = [dirModelChainLnx{icurr} '/' trainFinalModelName{icurr}];
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
        v{icurr} = [dirModelChainLnx{icurr} '/' trainCurrModelName{icurr}];
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
        v{icurr} = [dirModelChainLnx{icurr} '/' obj.trainingImagesName];
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
        v{icurr} = [obj.dirProjLnx '/' aptRepoSnapshotName{icurr}];
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

%     function v = dockerImgPath(obj,backend) %#ok<INUSL> 
%       % todo: this should depend on what type of tracker
%       v = backend.dockerimgroot;
%       if ~isempty(backend.dockerimgtag)
%         v = [v ':' backend.dockerimgtag];
%       end
%     end

    function [v,idx] = getPrevModels(obj,varargin)
      idx = obj.select(varargin{:});
      v = obj.prev_models(idx);
    end

    function obj = DeepModelChainOnDisk(varargin)
      % allow to call with no inputs, but then all responsibility for
      % properly setting variables is on outside code.
      if nargin == 0,
        return
      end
        
      nmodels = [];
      for iprop=1:2:numel(varargin)
        prop = varargin{iprop} ;
        value = varargin{iprop+1} ;
        if strcmp(prop,'nmodels'),
          nmodels = value;
        else
          obj.(prop) = value;
        end
      end

      obj.autoFix_(nmodels);
    end
    
    function autoFix_(obj,nmodels)
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
      %obj.checkFileSep();
      % if isempty(obj.isRemote_),
      %   obj.isRemote_ = false ;
      % end
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
      % Merge irreconcilable properties by warning, then keeping ours
      tocheck = {'rootDir','trkTSstr'};
      for i = 1:numel(tocheck),
        prop = tocheck{i};
        if ~isequal(obj.(prop),dmc.(prop)),
          warning('Differing values for %s, using %s',prop,obj.(prop));
        end
      end
      % Merge numeric properties
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
      % Merge cell array properties
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
      % Merge boolean properties
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
      % Recompute a bunch of things based on the now-merged properties
      obj.resetFollowsObjDet();
      obj.resetIsMultiViewTracker();
      obj.resetIsMultiStageTracker();
    end  % function

    function dmc = selectSubset(obj,varargin)
      % Return a DMC with a subset of the models in obj.
      % The returned DMC is completely indepdent of obj, i.e.
      % they have no shared structure.
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

    % function obj2 = copyAndDetach(obj)
    %   obj2 = copy(obj);
    %   %obj2.prepareBg();
    % end    

%     function prepareBg(obj)
%       % 'Detach' a DMC for use in bg processes
%       % Typically you would deepcopy the DMC before calling this
%       for i=1:numel(obj)
% %         if ~isempty(obj(i).reader)
% %           obj(i).reader.prepareBg();
% %         end
%       end
%     end

    function [fileinfo,idx] = trainFileInfo(obj,varargin) 
      % This method is only called by trainFileInfoSingle() and trackFileInfo(),
      % neither of which ever gets called by anyone. --ALT, 2024-10-01
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
      % This method does not seem to ever get called by anyone. --ALT, 2024-10-01
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
      % This method does not seem to ever get called by anyone. --ALT, 2024-10-01
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
        gnetspecific = cellfun(@(x)[dmcl{ii} '/' x],gnetspecific,'uni',0);
      
        g{ii} = [{ ...
          [obj.dirProjLnx '/' sprintf('%s_%s*',obj.modelChainID{i},obj.trainID{i})]; ... % lbl
          [dmcl{ii} '/' sprintf('%s*',obj.trainID{i})]; ... % toks, logs, errs
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
    
    function tf = canTrack(obj)
      % For each known model, whether or not it is in a state that allows tracking.
      % We consider a model to be in a state that allows tracking if it has
      % undergone at least one iteration of training.  Note also that a model that
      % has been trained for nan iterations is not considered to be in a state that
      % allows tracking.
      tf = (obj.iterCurr > 0) ;
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
        
    function [tf,tpdir] = trnPackExists(obj,varargin)
      % Training package exists
      trainLocLnx = obj.trainLocLnx();  % old-style string
      tpdir = obj.dirProjLnx;
      tf = logical(exist(tpdir,'dir')) && logical(exist(trainLocLnx,'file')) ;
    end

    % function result = getTorchHome(obj)
    %   if obj.isRemote_ ,
    %     result = linux_fullfile(obj.remoteRootDir_, 'torch') ;
    %   else
    %     result = fullfile(APT.getdotaptdirpath(), 'torch') ;
    %   end
    % end

  end  % methods
  
  
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
      % Returns a numeric array suitable for indexing into dmc.view, dmc.stage, etc.
      % The returned array is generally a numeric array of indices.
      % Example: selectHelper(dmc, 'view', 1) => (indices of all models in dmc with view==1)
      % Example: selectHelper(dmc, 'stage', 2) => (indices of all models in dmc with stage==1)
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
    end  % function

    function result = modernize(dmcs,varargin)
      % Return a DeepModelChainOnDisk handle array, similar to dmcs, but patched
      % up to conform to how a DeepModelChainOnDisk should be in the current version
      % of APT.  If dmcs is modern enough, result will be an alias for dmcs.  But if
      % dmcs is oldish, result will be a new, independent DeepModelChainOnDisk.

      if isempty(dmcs)
        result = dmcs;
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
        result = DeepModelChainOnDisk(...
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
          'trkTaskKeyword',trkTaskKeyword,...
          'trkTSstr',dmcs(1).trkTSstr...
          );
        result.resetFollowsObjDet();
        result.resetIsMultiViewTracker();
        result.resetIsMultiStageTracker();
      else
        assert(numel(dmcs)==1);
        result = dmcs;
      end
    end  % function

    function info = trackerInfo(dmc)
      info = struct() ;
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
    end  % function

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
    end  % function

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

    end  % function

  end  % methods (Static)
end  % classdef
