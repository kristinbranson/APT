classdef DLCacheProj < handle
  % DLCacheProj understands the DL cache directory structure, names+locs of
  % artifacts, and so on; for a single project.
  
  properties
    dir = '/home/ubuntu/cache'; % the root cache directory
    projname % eg 'multitarget_bubble'    
    view = 1; % 1-based idx
    
    % Views currently tracked independently so (projname,view) identify a 
    % trackable "project"
    
    modelID   % unique ID for a training model for (projname,view). 
            % A model can be trained once, but also train-augmented.
    trainID % a single modelID may be trained multiple times due to 
            % train-augmentation, so a single modelID may have multiple
            % trainID associated with it.
    trainType % scalar DLTrainType
    
    trainFinalIdx % final training iteration
  end
  
  properties (Dependent)
    lblstrippedlnx % full path to stripped lbl file for this train session
    lblstrippedname % short filename for 
    trainloglnx % full path to logfile for a certain train session 
    trainlogname % 
    killtoklnx
    killtokname
    projdirlnx % full path cache subdir for this proj
    projdirname % 
    modeldirlnx % cache subdir for this model
    traindatalnx % fullpath to traindata.json for this model/train session    
    trainfinalindexlnx
    %trainindexname
    trainerrfilelnx
  end
  
  methods
    function v = get.lblstrippedlnx(obj)      
      v = [obj.projdirlnx '/' obj.lblstrippedname];      
    end
    function v = get.lblstrippedname(obj)
      v = obj.lblstrippednameStc(obj.modelID,obj.trainID);
    end
    function v = get.trainloglnx(obj)
      v = [obj.projdirlnx '/' obj.trainlogname];
    end    
    function v = get.trainlogname(obj)
      v = sprintf('%s_%s_%s.log',obj.modelID,obj.trainID,...
        lower(char(obj.trainType)));
    end    
    function v = get.killtoklnx(obj)
      v = [obj.projdirlnx '/' obj.killtokname];
    end    
    function v = get.killtokname(obj)
      v = sprintf('%s_%s_%s.KILLED',obj.modelID,obj.trainID,...
        lower(char(obj.trainType)));
    end    
    function v = get.projdirlnx(obj)
      v = [obj.dir '/' obj.projdirname];
    end
    function v = get.projdirname(obj)
      v = sprintf('%s_view%d',obj.projname,obj.view-1);
    end
    function v = get.modeldirlnx(obj)
      v = [obj.projdirlnx '/' obj.modelID];
    end
    function v = get.traindatalnx(obj)
      v = [obj.modeldirlnx '/' 'traindata.json'];
    end  
    function v = get.trainfinalindexlnx(obj)
      v = [obj.modeldirlnx '/' obj.gettrainindexname(obj.trainFinalIdx)];
    end
    function v = gettrainindexname(obj,idx)
      v = sprintf('deepnet-%d.index',idx);
    end
    function v = get.trainerrfilelnx(obj)
      v = ['/home/ubuntu/' sprintf('%s.err',obj.modelID)];
    end
  end  
  methods (Static)
    function v = lblstrippednameStc(modelID,trainID)
      v = sprintf('%s_%s.lbl',modelID,trainID);
    end
  end
  
  methods  
    function obj = DLCacheProj(varargin)
      for i=1:2:nargin
        p = varargin{i};
        v = varargin{i+1};
        obj.(p) = v;
      end
    end
    function printall(obj)
      mc = metaclass(obj);
      props = mc.PropertyList;
      propnames = {props.Name}';
      tf = cellfun(@(x)~isempty(regexp(x,'lnx$','once')),propnames);
      propnames = propnames(tf);
      for i=1:numel(propnames)
        p = propnames{i};
        fprintf('%s: %s\n',p,obj.(p));
      end
    end
  end
end