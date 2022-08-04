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

    props_numeric = {'jobidx','stage','view','iterFinal','iterCurr','nLabels'};
    props_cell = {'netType','netMode','modelChainID','trainID','restartTS','trainConfigNameOverride'};
    props_bool = {'isMultiView','isMultiStage'};

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
    % e.g. within `DeepTracker.trnSpawnBsubDocker`, `rootDir` is set. We
    % should probably encapsulate this code so that it is not copy-pasted
    % all over given that it is done very similarly everywhere.   
    % Example: `/groups/branson/home/bransonk/.apt/tp76715886_6c90_4126_a9f4_0c3d31206ee5`
    % This will be the same for all stages/views.
    rootDir = '';

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
    
    isMultiView = []; % whether this was trained with one call to APT_interface for all views
    isMultiStage = []; % whether this was trained with one call to APT_interface for all stages
    % if provided, overrides .lblStrippedName. used for each running splits
    % wherein a single stripped lbl is used in multiple runs
    trainConfigNameOverride = {}; 
    
    iterFinal = []; % final expected iteration    
    iterCurr = []; % last completed iteration, corresponds to actual model file used
    nLabels = []; % number of labels used to train
    
    reader % scalar DeepModelChainReader. used to update the itercurr; 
      % knows how to read the (possibly remote) filesys etc
      
    filesep ='/'; % file separator

    trkTaskKeyword = ''; % arbitrary tracking task keyword; used for tracking output files
    trkTSstr = '';% timestamp for tracking
    prev_models = []; % prev model to start training from
  end
  properties (Dependent)
    n
    nviews
    njobs
    nstages
    isRemote
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
      v = obj.splitidx(idx);
    end
    function v = get.isRemote(obj)
      v = obj.reader.getModelIsRemote();
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
      v = obj.splitidx(idx) > 0;
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
    function v = getRootDir(obj)
      v = obj.rootDir;
    end
    function setRootDir(obj,v)
      obj.rootDir = v;
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
    function v = job2views(obj,ijob)
      v = unique(obj.view(obj.jobidx==ijob));
    end
    function v = job2stages(obj,ijob)
      v = unique(obj.stage(obj.jobidx==ijob));
    end
    function autoSetIsMultiView(obj)
      v = false(1,obj.n);
      for ijob = 1:obj.njobs,
        viewcurr = obj.job2view(ijob);
        v(obj.jobidx==ijob) = numel(viewcurr) > 1;
      end
      obj.isMultiView = v;
    end
    function autoSetIsMultiStage(obj)
      v = false(1,obj.n);
      for ijob = 1:obj.njobs,
        stagecurr = obj.job2stage(ijob);
        v(obj.jobidx==ijob) = numel(stagecurr) > 1;
      end
      obj.isMultiStage = v;
    end
    % dirNetLnx can depend on netType, so return a cell
    function [v,idx] = dirNetLnx(obj,varargin)
      idx = obj.select(varargin{:});
      v = cell(1,numel(idx));
      for ii = 1:numel(idx),
        i = idx(ii);
        v{ii} = [obj.rootDir obj.filesep obj.projID obj.filesep char(obj.netType{i})];
      end
    end
    function [v,idx] = getNetDescriptor(obj,varargin)
      idx = obj.select(varargin{:});
      v = cell(1,numel(idx));
      for ii = 1:numel(idx),
        i = idx(ii);
        v{ii} = sprintf('%s_view%d',char(obj.netType{i}),obj.view(i));
      end
    end
    function [v,idx] = dirViewLnx(obj,varargin)
      idx = obj.select(varargin{:});
      v = cell(1,numel(idx));
      for ii = 1:numel(idx),
        i = idx(ii);
        v{ii} = [obj.rootDir obj.filesep obj.projID obj.filesep char(obj.netType{i}) obj.filesep sprintf('view_%d',obj.view(i))];
      end
    end
    function [v,idx] = dirModelChainLnx(obj,varargin)
      [dirViewLnxs,idx] = obj.dirViewLnx(varargin{:});
      v = cell(1,numel(idx));
      for ii = 1:numel(idx),
        i = idx(ii);
        v{ii} = [dirViewLnxs{ii} obj.filesep obj.modelChainID{i}];
      end
    end
    function [v,idx] = dirTrkOutLnx(obj,varargin)
      [dirModelChainLnxs,idx] = obj.dirModelChainLnx(varargin{:});
      v = cell(1,numel(dirModelChainLnxs));
      for i = 1:numel(dirModelChainLnxs),
        v{i} = [dirModelChainLnxs{i} obj.filesep 'trk'];
      end
    end 
    function v = dirAptRootLnx(obj)
      v = [obj.rootDir obj.filesep 'APT'];
    end
    function [v,idx] = trainConfigLnx(obj,varargin)
      [trainConfigNames,idx] = obj.trainConfigName(varargin{:});
      v = cell(1,numel(idx));
      for i = 1:numel(idx),
        v{i} = [obj.dirProjLnx obj.filesep trainConfigNames{i} obj.configFileExt];
      end
    end
    function [v,idx] = trainConfigName(obj,varargin)
      idx = obj.select(varargin{:});
      v = cell(1,numel(idx));
      for ii = 1:numel(idx),
        i = idx(ii);
        if numel(obj.trainConfigNameOverride) >= i && ~isempty(obj.trainConfigNameOverride{i}),
          v{ii} = obj.trainConfigNameOverride{i};
        else
          v{ii} = sprintf('%s_%s',obj.modelChainID{i},obj.trainID{i});
        end
      end
    end
    function [v,idx] = lblStrippedLnx(obj,varargin)
      warning('OBSOLETE: Reference to stripped lbl file. We are trying to remove these. Let Kristin know how you got here!');
      [trainConfigNames,idx] = obj.trainConfigName(varargin{:});
      v = cell(1,numel(idx));
      for i = 1:numel(idx),
        v{i} = [obj.dirProjLnx obj.filesep trainConfigNames{i} '.lbl'];
      end
    end
    % full path to json config for this train session
    function [v,idx] = trainJsonLnx(obj,varargin)
      [trainConfigNames,idx] = obj.trainConfigName(varargin{:});
      v = cell(1,numel(idx));
      for i = 1:numel(idx),
        v{i} = [obj.dirProjLnx obj.filesep trainConfigNames{i} '.json'];
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

    function [v,idx] = cmdfileLnx(obj,varargin)
      [v,idx] = obj.cmdfileName(varargin{:});
      for i = 1:numel(v),
        v{i} = [obj.dirProjLnx obj.filesep v{i}];
      end
    end

    function [v,idx] = cmdfileName(obj,varargin)
      idx = obj.select(varargin{:});
      v = cell(1,numel(idx));
      for ii = 1:numel(idx),
        i = idx(ii);
        if obj.isMultiView(i),
          viewstr = '';
        else
          viewstr = sprintf('view%d',obj.view(i));
        end
        v{i} = sprintf('%s%s_%s_%s',obj.modelChainID{i},viewstr,obj.trainID{i},obj.netMode{i}.shortCode);
        if obj.isSplit(i)
          v{i} = [v{i},'.sh'];
        else
          v{i} = [v{i},'.cmd'];
        end
      end
    end
    function [v,idx] = splitfileLnx(obj,varargin)
      [v,idx] = obj.splitfileName(varargin{:});
      for i = 1:numel(v),
        v{i} = [obj.dirProjLnx obj.filesep v{i}];
      end
    end
    function [v,idx] = splitfileName(obj,varargin)
      idx = obj.select(varargin{:});
      % not sure how multiview works yet
      if obj.isSplit
        v = cell(1,numel(idx));
        for ii = 1:numel(idx),
          i = idx(ii);
          v{i} = sprintf('%s_view%d_split.json',obj.modelChainID{i},obj.view(i));
        end
      else
        v = repmat({'__NOSPLIT__'},[1,numel(idx)]);
      end
    end
    function [v,idx] = valresultsLnx(obj,varargin)
      [valresultsName,idx] = obj.valresultsName(varargin{:});
      [dirTrkOutLnx] = obj.dirTrkOutLnx(idx);
      v = cell(1,numel(idx));
      for i = 1:numel(idx),
        v{i} = [dirTrkOutLnx{i} obj.filesep valresultsName{i}];
      end
    end

    function [v,idx] = valresultsBaseLnx(obj,varargin)
      [valresultsNameBase,idx] = obj.valresultsNameBase(varargin{:});
      [dirTrkOutLnx] = obj.dirTrkOutLnx(idx);
      v = cell(1,numel(idx));
      for i = 1:numel(idx),
        v{i} = [dirTrkOutLnx{i} obj.filesep valresultsNameBase{i}];
      end
    end    
    function [v,idx] = valresultsName(obj,varargin)
      idx = obj.select(varargin{:});
      v = cell(1,numel(idx));
      for ii = 1:numel(idx),
        i = idx(ii);
        v{ii} = sprintf('%s_%d.mat',obj.trainID{i},obj.view(i));
      end
    end
    function [v,idx] = valresultsNameBase(obj,varargin)
      idx = obj.select(varargin{:});
      v = obj.trainID(idx);
    end 
    function [v,idx] = errfileLnx(obj,varargin)
      [errfileName,idx] = obj.errfileName(varargin{:});
      v = cell(1,numel(idx));
      for i = 1:numel(idx),
        v{i} = [obj.dirProjLnx obj.filesep errfileName{i}];
      end
    end
    function [v,idx] = errfileName(obj,varargin)
      idx = obj.select(varargin{:});
      v = cell(1,numel(idx));
      for ii = 1:numel(idx),
        i = idx(ii);
        if obj.isMultiView(i),
          viewstr = '';
        else
          viewstr = sprintf('view%d',obj.view(i));
        end
        v{i} = sprintf('%s_%s_%s.err',obj.modelChainID{i},viewstr,obj.trainID{i},obj.netMode{i}.shortCode);
      end
    end
    function [v,idx] = trainLogLnx(obj,varargin)
      [trainLogName,idx] = obj.trainLogName(varargin{:});
      v = cell(1,numel(trainLogName));
      for i = 1:numel(trainLogName),
        v{i} = [obj.dirProjLnx obj.filesep trainLogName{i}];
      end
    end
    function [v,idx] = trainLogName(obj,varargin)
      idx = obj.select(varargin{:});
      v = cell(1,numel(idx));
      for ii = 1:numel(idx),
        i = idx(ii);
        if ~obj.isMultiView(i),
          viewstr = '';
        else
          viewstr = sprintf('view%d',obj.view(i));
        end
        if isequal(obj.trainType{i},DLTrainType.Restart),
          restartstr = obj.restartTS{i};
        else
          restartstr = '';
        end
        v{ii} = sprintf('%s_%s_%s_%s%s.log',obj.modelChainID{i},viewstr,...
          obj.trainID{i},obj.netMode{i}.shortCode,...
          lower(char(obj.trnType{i})),restartstr);
      end
    end
    function [v,idx] = trkName(obj,varargin)
      [idx] = obj.select(varargin{:});
      v = cell(1,numel(idx));
      for ii = 1:numel(idx),
        i = idx(ii);
        v{ii} = sprintf('%s_%s_vw%d_%s',obj.trkTaskKeyword,obj.modelChainID, ...
          obj.view(i),obj.trkTSstr);
      end
    end    
    
    function [v,idx] = trkExtLnx(obj,ext,varargin)
      [v,idx] = obj.trkExtName(ext,varargin{:});
      [dirTrkOutLnx] = obj.dirTrkOutLnx(idx);
      for i = 1:numel(idx),
        v{i} = [dirTrkOutLnx{i} obj.filesep v{i}];
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
      for i = 1:numel(idx),
        v{i} = [dirTrkOutLnx{i} obj.filesep trkSnapshotName{i}];
      end
    end
    function [v,idx] = trkSnapshotName(obj,varargin)
      idx = obj.select(varargin{:});
      v = cell(1,numel(idx));
      for ii = 1:numel(idx),
        i = idx(ii);
        v{ii} = sprintf('%s_%s_vw%d_%s.aptsnapshot',obj.trkTaskKeyword,obj.modelChainID{i}, ...
          obj.view{i},obj.trkTSstr);
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
      for i = 1:numel(idx),
        v{i} = [dirTrkOutLnx{i} obj.filesep gtOutfileName{i}];
      end
    end
    function [v,idx] = gtOutfileName(obj,varargin)
      idx = obj.select(varargin{:});
      v = cell(1,numel(idx));
      for ii = 1:numel(idx),
        i = idx(ii);
        v{ii} = sprintf('gtcls_vw%d_%s.mat',obj.view(i),obj.trkTSstr);
      end
    end
    function [v,idx] = killTokenLnx(obj,varargin)
      [killTokenName,idx] = obj.killTokenName(varargin{:});
      if any(obj.isMultiView(idx)),
        dirModelChainLnx = obj.dirModelChainLnx(varargin{:});
      end
      v = cell(1,numel(idx));
      for ii = 1:numel(idx),
        i = idx(ii);
        if obj.isMultiView(i),
          v{ii} = [obj.dirProjLnx obj.filesep killTokenName{ii}];
        else
          v{ii} = [dirModelChainLnx{ii} obj.filesep killTokenName{ii}];
        end
      end
    end    
    function [v,idx] = killTokenName(obj,varargin)
      idx = obj.select(varargin{:});
      v = cell(1,numel(idx));
      for ii = 1:numel(idx),
        i = idx(ii);
        if isequal(obj.trainType{i},DLTrainType.Restart),
          restartstr = obj.restartTS{i};
        else
          restartstr = '';
        end
        v{i} = sprintf('%s_%s%s.KILLED',obj.trainID{i},lower(char(obj.trainType{i})),restartstr);
      end
    end  
    function [v,idx] = trainDataLnx(obj,varargin)
      [dirModelChainLnx,idx] = obj.dirModelChainLnx(varargin{:});
      v = cell(1,numel(idx));
      for i = 1:numel(idx),
        v{i} = [dirModelChainLnx{i} obj.filesep 'traindata.json'];
      end
    end
    function [v,idx] = trainFinalModelLnx(obj,varargin)
      [dirModelChainLnx,idx] = obj.dirModelChainLnx(varargin{:});
      [trainFinalModelName] = obj.trainFinalModelName(idx);
      v = cell(1,numel(idx));
      for i = 1:numel(idx),
        v{i} = [dirModelChainLnx{i} obj.filesep trainFinalModelName{i}];
      end
    end
    function [v,idx] = trainCompleteArtifacts(obj,varargin)
      [trainFinalModelLnx,idx] = obj.trainFinalModeLnx(varargin{:});
      v = cell(1,numel(idx));
      for i = 1:numel(idx),
        v{i} = {trainFinalModelLnx{i}}; %#ok<CCAT1> 
      end
    end
    function [v,idx] = trainCurrModelLnx(obj,varargin)
      [dirModelChainLnx,idx] = obj.dirModelChainLnx(varargin{:});
      [trainCurrModelName] = obj.trainCurrModelName(idx);
      v = cell(1,numel(idx));
      for i = 1:numel(idx),
        v = [dirModelChainLnx{i} obj.filesep trainCurrModelName{i}];
      end
    end
    function [v,idx] = trainFinalModelName(obj,varargin)
      idx = obj.select(varargin{:});
      pat = obj.netType.mdlNamePat;
      v = cell(1,numel(idx));
      for ii = 1:numel(idx),
        i = idx(ii);
        v{ii} = sprintf(pat,obj.iterFinal(i));
      end
    end    
    function [v,idx] = trainCurrModelName(obj,varargin)
      idx = obj.select(varargin{:});
      pat = DLNetType.(obj.netType).mdlNamePat;
      v = cell(1,numel(idx));
      for ii = 1:numel(idx),
        i = idx(ii);
        v{ii} = sprintf(pat,obj.iterCurr(i));
      end
    end
    function [v,idx] = trainImagesNameLnx(obj,varargin)
      [dirModelChainLnx,idx] = obj.dirModelChainLnx(varargin{:});
      v = cell(1,numel(idx));
      for i = 1:numel(idx),
        v{i} = [dirModelChainLnx{i} obj.filesep obj.trainingImagesName];
      end
    end
    function [v,idx] = trainModelGlob(obj,varargin)
      idx = obj.select(varargin{:});
      v = cell(1,numel(idx));
      for ii = 1:numel(idx),
        i = idx(ii);
        v{ii} = obj.netType{i}.mdlGlobPat;
      end
    end
    function [v,idx] = aptRepoSnapshotLnx(obj,varargin)
      [aptRepoSnapshotName,idx] = obj.aptRepoSnapshotName(varargin{:});
      v = cell(1,numel(idx));
      for i = 1:numel(idx),
        v{i} = [obj.dirProjLnx obj.filesep aptRepoSnapshotName{i}];
      end
    end
    function [v,idx] = aptRepoSnapshotName(obj,varargin)
      idx = obj.select(varargin{:});
      v = cell(1,numel(idx));
      for ii = 1:numel(idx),
        i = idx(ii);
        v{ii} = sprintf('%s_%s.aptsnapshot',obj.modelChainID{i},obj.trainID{i});
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
      if ~isempty(obj.reader)
        obj2.reader = copy(obj.reader);
      end
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
      if isempty(nmodels),
        nmodels = max(numel(obj.view),numel(obj.jobidx),numel(obj.stage),numel(obj.splitIdx));
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
      if ischar(obj.modelChainID),
        obj.modelChainID = repmat({obj.modelChainID},[1,nmodels]);
      end
      assert(~isempty(obj.trainID));
      if ischar(obj.trainID),
        obj.trainID = repmat({obj.trainID},[1,nmodels]);
      end
      assert(~isempty(obj.projID));
      if ischar(obj.projID),
        obj.projID = repmat({obj.projID},[1,nmodels]);
      end
      assert(~isempty(obj.rootDir));
      if ischar(obj.projID),
        obj.rootDir = repmat({obj.rootDir},[1,nmodels]);
      end
      assert(~isempty(obj.netType));
      if numel(obj.netType) == 1,
        obj.netType = repmat({obj.netType},[1,nmodels]);
      end
      assert(~isempty(obj.netMode));
      if numel(obj.netMode) == 1,
        obj.netMode = repmat({obj.netMode},[1,nmodels]);
      end
      % not required when training
      if ischar(obj.trkTaskKeyword),
        obj.trkTaskKeyword = repmat({obj.trkTaskKeyword},[1,nmodels]);
      end
      % not required when training
      if ischar(obj.trkTSstr),
        obj.trkTSstr = repmat({obj.trkTSstr},[1,nmodels]);
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
      if isempty(obj.isMultiView),
        obj.autoSetIsMultiView();
      end
      if isempty(obj.isMultiStage),
        obj.autoSetIsMultiStage();
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

    end
    function obj = merge(obj,dmc)
      assert(isequaln(obj.projID,dmc.projID));
      tocheck = {'rootDir','trkTaskKeyword','trkTSstr'};
      for i = 1:numel(tocheck),
        prop = tocheck{i};
        if ~isequal(obj.(prop),dmc.(prop)),
          warning('Differing values for %s, using %s',prop,obj.(prop));
        end
      end
      tocheck = {'doSplit','splitIdx'};
      for i = 1:numel(tocheck),
        prop = tocheck{i};
        if ~isequaln(obj.(prop),dmc.(prop)),
          warning('Differing values for %s, using %d',prop,obj.(prop));
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
        if ~isempty(obj(i).reader)
          obj(i).reader.prepareBg();
        end
      end
    end
    function fileinfo = trainFileInfo(obj,varargin) 
      idx = obj.select(varargin{:});
      if ~any(idx),
        fileinfo = [];
        return;
      end
      for ii = 1:numel(idx),
        i = idx(ii);
        fileinfocurr = struct(...
          'modelchainID',obj.modelChainID{i},...
          'trnID',obj.trainID{i},...
          'dlconfig',obj.trainConfigLnx(i),...
          'trainlocfile',obj.trainLocLnx(i),...
          'cache',obj.rootDir,...
          'errfile',obj.errfileLnx(i),...
          'nettype',obj.netType{i},...
          'netmode',obj.netMode{i});
        fileinfo(ii) = fileinfocurr; %#ok<AGROW> 
      end
    end
%     % OBSOLETE
%     function printall(obj)
%       mc = metaclass(obj);
%       props = mc.PropertyList;
%       tf = [props.Dependent];
%       propnames = {props(tf).Name}';
% %       tf = cellfun(@(x)~isempty(regexp(x,'lnx$','once')),propnames);
% %       propnames = propnames(tf);
% 
%       nobj = numel(obj);
%       for iobj=1:nobj
%         if nobj>1
%           fprintf('### obj %d ###\n',iobj);
%         end
%         for iprop=1:numel(propnames)
%           p = propnames{iprop};
%           fprintf('%s: %s\n',p,obj(iobj).(p));
%         end
%       end
%     end
    function lsProjDir(obj,varargin)
      idx = obj.select(varargin{:});
      obj.reader.lsProjDir(obj,idx);
    end
    function lsModelChainDir(obj)
      obj.reader.lsModelChainDir(obj);
    end
    function lsTrkDir(obj)
      obj.reader.lsTrkDir(obj);
    end
    function [g,idx] = modelGlobsLnx(obj,varargin)
      % filesys paths/globs of important parts/stuff to keep
      
      [dmcl,idx] = obj.dirModelChainLnx(varargin{:});
      g = cell(1,numel(idx));
      for ii = 1:numel(idx),
        i = idx(ii);
        netty = obj.netType{i};
        gnetspecific = netty.getModelGlobs(obj.iterCurr(i));
        gnetspecific = cellfun(@(x)[dmcl{ii} obj.filesep x],gnetspecific,'uni',0);
      
        g{i} = [{ ...
          [obj.dirProjLnx obj.filesep sprintf('%s_%s*',obj.modelChainID(i),obj.trainID(i))]; ... % lbl
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
        globs = globs{1};
        for j = 1:numel(globs),
          g = globs{j};
          if contains(g,'*')
            gP = fileparts(g);
            dd = dir(g);
            mdlFilesNew = {dd.name}';
            mdlFilesNew = cellfun(@(x) fullfile(gP,x),mdlFilesNew,'uni',0);
            mdlFiles{i} = [mdlFiles{i}; mdlFilesNew]; 
          elseif exist(g,'file')>0
            mdlFiles{i}{end+1,1} = g; 
          end
        end
      end      
    end
    
    function tfSuccess = updateCurrInfo(obj)
      % Update .iterCurr by probing filesys
      
      assert(isscalar(obj));
      % will update for all
      maxiter = obj.reader.getMostRecentModel(obj);
      obj.iterCurr = maxiter;
      tfSuccess = maxiter >= 0;
      
      if maxiter>obj.iterFinal
        warningNoTrace('Current model iteration (%d) exceeds specified maximum/target iteration (%d).',...
          maxiter,obj.iterFinal);
      end
    end
    
    % read nLabels from the stripped lbl file
    function readNLabels(obj)
      if strcmp(obj.configFileExt,'.json'),
        trainLocLnx = obj.trainLocLnx();
        [un,~,idx] = unique(trainLocLnx);
        nLabels1 = nan(1,obj.n);
        for i = 1:numel(un),
          assert(exist(un{i},'file') > 0);
          nLabels1(idx==i) = TrnPack.readNLabels(un{i});
        end
        obj.nLabels = nLabels1;
      else
        lblStrippedLnx = obj.lblStrippedLnx();
        [un,~,idx] = unique(lblStrippedLnx);
        nLabels1 = nan(1,obj.n);
        for i = 1:numel(un),
          assert(exist(un{i},'file') > 0);
          s = load(obj.lblStrippedLnx,'preProcData_MD_frm','-mat');
          nLabels1(idx==i) = size(s.preProcData_MD_frm,1);
        end
        obj.nLabels = nLabels1;
      end
    end
    
    % whether training has actually started
    function tf = isPartiallyTrained(obj)      
      tf = ~isempty(obj.iterCurr);      
    end
    
    function mirror2remoteAws(obj,aws)
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
      assert(~obj.isRemote,'Model must be local in order to mirror/upload.');      

      succ = obj.updateCurrInfo;
      if ~all(succ),
        dmclfail = obj.dirModelChainLnx(find(~succ));
        fstr = sprintf('%s ',dmclfail{:});
        error('Failed to determine latest model iteration in %s.',fstr);
      end
      fprintf('Current model iteration is %s.\n',mat2str(obj.iterCurr));
     
      aws.checkInstanceRunning(); % harderrs if instance isn't running
      
      mdlFiles = obj.findModelGlobsLocal();
      mdlFiles = cat(1,mdlFiles{:});
      pat = obj.rootDir;
      pat = regexprep(pat,'\\','\\\\');
      mdlFilesRemote = regexprep(mdlFiles,pat,DLBackEndClass.RemoteAWSCacheDir);
      mdlFilesRemote = FSPath.standardPath(mdlFilesRemote);
      nMdlFiles = numel(mdlFiles);
      netstr = char(obj.netType); 
      fprintf(1,'Upload/mirror %d model files for net %s.\n',nMdlFiles,netstr);
      descstr = sprintf('Model file: %s',netstr);
      for i=1:nMdlFiles
        src = mdlFiles{i};
        info = dir(src);
        filesz = info.bytes/2^10;
        dst = mdlFilesRemote{i};
        % We just use scpUploadOrVerify which does not confirm the identity
        % of file if it already exists. These model files should be
        % immutable once created and their naming (underneath timestamped
        % modelchainIDs etc) should be pretty/totally unique. 
        %
        % Only situation that might cause problems are augmentedtrains but
        % let's not worry about that for now.
        aws.scpUploadOrVerify(src,dst,sprintf('%s (%s), %d KB',descstr,info.name,round(filesz)),'destRelative',false); % throws
      end
      
      % if we made it here, upload successful
      
      obj.rootDir = DLBackEndClass.RemoteAWSCacheDir;
      obj.reader = DeepModelChainReaderAWS(aws);
    end
    
    function mirrorFromRemoteAws(obj,cacheDirLocal)
      % Inverse of mirror2remoteAws. Download/mirror model from remote AWS
      % instance to local cache.
      %
      % update .rootDir, .reader appropriately to point to model in local
      % cache.
      %
      % In practice for the client, this action updates the "latest model"
      % to point to the local cache.
      
      assert(isscalar(obj));      
      assert(obj.isRemote,'Model must be remote in order to mirror/download.');      
      
      aws = obj.reader.awsec2;
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
     
      succ = obj.updateCurrInfo;
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
        netstr = char(obj.netType(j)); 
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
      
      obj.rootDir = cacheDirLocal;
      obj.reader = DeepModelChainReaderLocal();
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
    function mcId = modelChainIDForSplit(mcIdBase,isplit)
      mcId = sprintf('%s_splt_%03d',mcIdBase,isplit);
    end

    function idx = selectHelper(info,varargin)
      if numel(varargin) == 1,
        idx = varargin{:};
        return;
      end
      [jobidx1,view1,stage1,splitidx1] = myparse(varargin,'jobidx','all','view','all','stage','all','splitidx','all'); 
      idx = false(1,numel(info.view));
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

  end
end
    
  
  