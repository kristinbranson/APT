classdef DeepModelChainOnDisk < handle & matlab.mixin.Copyable
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
    iterFinal % final expected iteration    
    %backEnd % back-end info (bsub, docker, aws)
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
    errfileLnx 
    errfileName
    trainLogLnx
    trainLogName
    killTokenLnx
    killTokenName
    trainDataLnx    
    trainFinalIndexLnx
    trainFinalIndexName
    aptRepoSnapshotLnx
    aptRepoSnapshotName
  end
  methods
    function v = get.dirProjLnx(obj)
      v = [obj.rootDir '/' obj.projID];
    end
    function v = get.dirNetLnx(obj)
      v = [obj.rootDir '/' obj.projID '/' char(obj.netType)];
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
    function v = get.errfileLnx(obj)      
      v = [obj.dirProjLnx '/' obj.errfileName];      
    end
    function v = get.errfileName(obj)
      v = sprintf('%s_%s.err',obj.modelChainID,obj.trainID);
    end
    function v = get.trainLogLnx(obj)
      v = [obj.dirProjLnx '/' obj.trainLogName];
    end
    function v = get.trainLogName(obj)
      switch obj.trainType
        case DLTrainType.Restart
          v = sprintf('%s_%s_%s%s.log',obj.modelChainID,obj.trainID,lower(char(obj.trainType)),obj.restartTS);
        otherwise
          v = sprintf('%s_%s_%s.log',obj.modelChainID,obj.trainID,lower(char(obj.trainType)));
      end
    end    
    function v = get.killTokenLnx(obj)
      v = [obj.dirModelChainLnx '/' obj.killTokenName];
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
    function v = get.trainFinalIndexLnx(obj)
      v = [obj.dirModelChainLnx '/' obj.trainFinalIndexName];
    end
    function v = get.trainFinalIndexName(obj)
      v = sprintf('deepnet-%d.index',obj.iterFinal);
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
        [obj.dirProjLnx '/' sprintf('%s_%s*',obj.modelChainID,obj.trainID)]; ... % lbl, err, logs
        [obj.dirModelChainLnx '/' sprintf('%s*',obj.trainID)]; ... % toks
        [obj.dirModelChainLnx '/' sprintf('deepnet-%d.*',obj.iterFinal)]; ... % final iter stuff
        [obj.dirModelChainLnx '/' 'deepnet_ckpt']; ... 
        [obj.dirModelChainLnx '/' 'splitdata.json']; ...
        [obj.dirModelChainLnx '/' 'traindata*']; ...
        };
    end
  end
end
    
  
  