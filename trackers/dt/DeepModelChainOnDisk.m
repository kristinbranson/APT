classdef DeepModelChainOnDisk < matlab.mixin.Copyable
  % DMCOD understands the filesystem structure of a deep model. This same
  % structure is used both remotely and locally.
  
  properties
    rootDir % root/parent "Models" dir
    projID 
    netType % scalar DLNetType
    view % 0-based
    modelChainID % unique ID for a training model for (projname,view). 
                 % A model can be trained once, but also train-augmented.
    trainID % a single modelID may be trained multiple times due to 
            % train-augmentation, so a single modelID may have multiple
            % trainID associated with it. Each (modelChainID,trainID) pair
            % has a unique associated stripped lbl.            
    restartTS % Training for each (modelChainID,trainID) can be killed and
              % restarted arbitrarily many times. This timestamp uniquely
              % identifies a restart
    trainType % scalar DLTrainType
    isMultiView = false; % whether this was trained with one call to APT_interface for all views
    iterFinal % final expected iteration    
    iterCurr % last completed iteration, corresponds to actual model file used
    nLabels % number of labels used to train
        
    aptRootUser % (optional) external/user APT code checkout root
  end
  properties (NonCopyable)
    backEndClass % scalar DLBackEndClass. This should be a copy, as it is a
      % snapshot of the backend used to generate the DMCOD. 
      % Conceptually, this prop is sometimes nec to specify the 
      % location/machine/server of the filesystem holding the DMCOD (eg for 
      % aws, or in general when the DMCOD lives on a remote filesys)
  end
  properties (Dependent)
    dirProjLnx
    dirNetLnx
    dirViewLnx  
    dirModelChainLnx
    dirTrkOutLnx
    dirAptRootLnx % loc of APT checkout (JRC)
    
    lblStrippedLnx % full path to stripped lbl file for this train session
    lblStrippedName % short filename 
    cmdfileLnx
    cmdfileName
    errfileLnx 
    errfileName
    trainLogLnx
    trainLogName
    viewName
    killTokenLnx
    killTokenName
    trainDataLnx    
    trainFinalModelLnx
    trainFinalModelName
    trainCurrModelLnx
    trainCurrModelName
    aptRepoSnapshotLnx
    aptRepoSnapshotName
    
    trainModelGlob
  end
  methods
    function v = get.dirProjLnx(obj)
      v = [obj.rootDir '/' obj.projID];
    end
    function v = get.dirNetLnx(obj)
      v = [obj.rootDir '/' obj.projID '/' char(obj.netType)];
    end
    function v = get.viewName(obj)
      v = sprintf('view_%d',obj.view(1));
    end
    function v = get.dirViewLnx(obj)
      v = [obj.rootDir '/' obj.projID '/' char(obj.netType) '/' sprintf('view_%d',obj.view)];
    end
    function v = get.dirModelChainLnx(obj)
      v = [obj.rootDir '/' obj.projID '/' char(obj.netType) '/' sprintf('view_%d',obj.view) '/' obj.modelChainID];
    end
    function v = get.dirTrkOutLnx(obj)
      v = [obj.rootDir '/' obj.projID '/' char(obj.netType) '/' sprintf('view_%d',obj.view) '/' obj.modelChainID '/' 'trk'];
    end 
    function v = get.dirAptRootLnx(obj)
      v = [obj.rootDir '/APT'];
    end 
    function v = get.lblStrippedLnx(obj)      
      v = [obj.dirProjLnx '/' obj.lblStrippedName];      
    end
    function v = get.lblStrippedName(obj)
      v = sprintf('%s_%s.lbl',obj.modelChainID,obj.trainID);
    end
    function v = get.cmdfileLnx(obj)      
      v = [obj.dirProjLnx '/' obj.cmdfileName];      
    end
    function v = get.cmdfileName(obj)
      if obj.isMultiView
        v = sprintf('%s_%s.cmd',obj.modelChainID,obj.trainID);
      else
        v = sprintf('%sview%d_%s.cmd',obj.modelChainID,obj.view,obj.trainID);
      end
    end    
    function v = get.errfileLnx(obj)      
      v = [obj.dirProjLnx '/' obj.errfileName];      
    end
    function v = get.errfileName(obj)
      if obj.isMultiView,
        v = sprintf('%s_%s.err',obj.modelChainID,obj.trainID);
      else
        v = sprintf('%sview%d_%s.err',obj.modelChainID,obj.view,obj.trainID);
      end
    end
    function v = get.trainLogLnx(obj)
      v = [obj.dirProjLnx '/' obj.trainLogName];
    end
    function v = get.trainLogName(obj)
      switch obj.trainType
        case DLTrainType.Restart
          if obj.isMultiView,
            v = sprintf('%s_%s_%s%s.log',obj.modelChainID,obj.trainID,lower(char(obj.trainType)),obj.restartTS);
          else
            v = sprintf('%sview%d_%s_%s%s.log',obj.modelChainID,obj.view,obj.trainID,lower(char(obj.trainType)),obj.restartTS);
          end
        otherwise
          if obj.isMultiView,
            v = sprintf('%s_%s_%s.log',obj.modelChainID,obj.trainID,lower(char(obj.trainType)));
          else
            v = sprintf('%sview%d_%s_%s.log',obj.modelChainID,obj.view,obj.trainID,lower(char(obj.trainType)));
          end
      end
    end    
    function v = get.killTokenLnx(obj)
      if obj.isMultiView,
        v = [obj.dirProjLnx '/' obj.killTokenName];
      else
        v = [obj.dirModelChainLnx '/' obj.killTokenName];
      end
    end    
    function v = get.killTokenName(obj)
      switch obj.trainType
        case DLTrainType.Restart
          v = sprintf('%s_%s%s.KILLED',obj.trainID,lower(char(obj.trainType)),obj.restartTS);
        otherwise
          v = sprintf('%s_%s.KILLED',obj.trainID,lower(char(obj.trainType)));
      end
    end  
    function v = get.trainDataLnx(obj)
      v = [obj.dirModelChainLnx '/traindata.json'];
    end
    function v = get.trainFinalModelLnx(obj)
      v = [obj.dirModelChainLnx '/' obj.trainFinalModelName];
    end
    function v = get.trainCurrModelLnx(obj)
      v = [obj.dirModelChainLnx '/' obj.trainCurrModelName];
    end
    function v = get.trainFinalModelName(obj)
      switch obj.netType
        case {DLNetType.openpose DLNetType.leap}
          v = sprintf('deepnet-%d',obj.iterFinal);
        otherwise
          v = sprintf('deepnet-%d.index',obj.iterFinal);
      end
    end    
    function v = get.trainCurrModelName(obj)
      switch obj.netType
        case {DLNetType.openpose DLNetType.leap}
          v = sprintf('deepnet-%d',obj.iterCurr);
        otherwise
          v = sprintf('deepnet-%d.index',obj.iterCurr);
      end
    end
    function v = get.trainModelGlob(obj)
      switch obj.netType
        case {DLNetType.openpose DLNetType.leap}
          v = 'deepnet-*';
        otherwise
          v = 'deepnet-*.index';
      end
    end
    function v = get.aptRepoSnapshotLnx(obj)
      v = [obj.dirProjLnx '/' obj.aptRepoSnapshotName];
    end
    function v = get.aptRepoSnapshotName(obj)
      v = sprintf('%s_%s.aptsnapshot',obj.modelChainID,obj.trainID);
    end      
  end
  methods
    function obj = DeepModelChainOnDisk(varargin)
      for iprop=1:2:numel(varargin)
        obj.(varargin{iprop}) = varargin{iprop+1};
      end
    end
    function printall(obj)
      mc = metaclass(obj);
      props = mc.PropertyList;
      tf = [props.Dependent];
      propnames = {props(tf).Name}';
%       tf = cellfun(@(x)~isempty(regexp(x,'lnx$','once')),propnames);
%       propnames = propnames(tf);

      nobj = numel(obj);
      for iobj=1:nobj
        if nobj>1
          fprintf('### obj %d ###\n',iobj);
        end
        for iprop=1:numel(propnames)
          p = propnames{iprop};
          fprintf('%s: %s\n',p,obj(iobj).(p));
        end
      end
    end
    function g = keepGlobsLnx(obj)
      % filesys paths/globs of important parts/stuff to keep
      
      g = { ...
        [obj.dirProjLnx '/' sprintf('%s_%s*',obj.modelChainID,obj.trainID)]; ... % lbl
        [obj.dirModelChainLnx '/' sprintf('%s*',obj.trainID)]; ... % toks, logs, errs
        [obj.dirModelChainLnx '/' sprintf('deepnet-%d.*',obj.iterCurr)]; ... % latest iter 
        [obj.dirModelChainLnx '/' 'deepnet_ckpt']; ... 
        [obj.dirModelChainLnx '/' 'splitdata.json']; ...
        [obj.dirModelChainLnx '/' 'traindata*']; ...
        };
    end
    
    function tfSuccess = updateCurrInfo(obj,beCurr)
      % Update current model iteration/info by probing filesys
      %
      % beCurr: the current DLBackEndClass. This may differ from
      % obj(i).backEndClass if the user has switched backends etc. 
      % In this case we may be in a bind as the filesystem holding obj may
      % no longer be accessible.
      
      assert(isscalar(obj));      
      
      beTrained = obj.backEndClass;
            
      switch beTrained.type
        case DLBackEnd.AWS
          if beCurr.type==DLBackEnd.AWS && ...
             beTrained.awsec2.isSameInstance(beCurr.awsec2)
           
           getMostRecentModelMeth = 'getMostRecentModelAWS';
           getMostRecentModelMethArgs = {beTrained.awsec2};
          else
            
            % problem, user has switched backends. AWSec2 instance that obj
            % was trained on may be Stopped, Terminated, etc.
              
          end
        case {DLBackEnd.Bsub DLBackEnd.Docker DLBackEnd.Unknown}
          % assume obj is on local filesystem
          
          % none
      end
      
      
      [getMostRecentModelMeth,getMostRecentModelMethArgs] = ...
        myparse(varargin,...
        'getMostRecentModelMeth','getMostRecentModelLocalLnx',...
        'getMostRecentModelMethArgs',{}...
        );
      
      
    switch be.type
        case DLBackEnd.AWS
          aws = be.awsec2;
          args = {'getMostRecentModelMeth' 'getMostRecentModelAWS' ...
                  'getMostRecentModelMethArgs' {aws}};
        otherwise
          % Assume locally-accessible cache. Win may cause a beef
          args = {};
      end
      

      
      maxiter = feval(getMostRecentModelMeth,obj,getMostRecentModelMethArgs{:});
      obj.iterCurr = maxiter;
      tfSuccess = ~isnan(maxiter);
    end
    
    function maxiter = getMostRecentModelLocalLnx(obj,varargin)
      maxiter = nan;
      %filepath = '';
      
      modelglob = obj.trainModelGlob;
      modelfiles = mydir(fullfile(obj.dirModelChainLnx,modelglob));
      if isempty(modelfiles),
        return;
      end
      
      maxiter = -1;
      for i = 1:numel(modelfiles),
        iter = DeepModelChainOnDisk.getModelFileIter(modelfiles{i});
        if iter > maxiter,
          maxiter = iter;
          %filepath = modelfiles{i};
        end
      end
    end
    
    function maxiter = getMostRecentModelAWS(obj,aws)
      % maxiter is nan if something bad happened or if DNE
      
      fspollargs = {'mostrecentmodel' obj.dirModelChainLnx};
      [tfsucc,res] = aws.remoteCallFSPoll(fspollargs);
      if tfsucc
        maxiter = str2double(res{1}); % includes 'DNE'->nan
      else
        maxiter = nan;
      end
    end
    
    % if nLabels not set, try to read it from the stripped lbl file
    function readNLabels(obj)
      if ~isempty(obj.lblStrippedLnx)
        s = load(obj.lblStrippedLnx,'preProcData_MD_frm','-mat');
        obj.nLabels = size(s.preProcData_MD_frm,1);
      end
    end
    
    % whether training has actually started
    function tf = isPartiallyTrained(obj)
      
      tf = ~isempty(obj.iterCurr);
      
    end
       
  end
  
  methods (Access = protected)
    function cpObj = copyElement(obj)
      % overloaded to deep copy .backEndClass
      cpObj = copyElement@matlab.mixin.Copyable(obj);
      if ~isempty(obj.backEndClass)
        assert(isa(obj.backEndClass,'DLBackEndClass') && isscalar(obj.backEndClass));
        cpObj.backEndClass = obj.backEndClass.copy();
      end
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
    
  end
end
    
  
  