classdef DeepModelChainReaderAWS < DeepModelChainReader
  
  properties 
    awsec2
  end
  
  methods (Access=protected)
    function obj2 = copyElement(obj)
      obj2 = copyElement@DeepModelChainReader(obj);
      if ~isempty(obj.awsec2)
        obj2.awsec2 = copy(obj.awsec2);
      end
    end
  end
  
  methods
    
    function obj = DeepModelChainReaderAWS(aws)
      assert(isa(aws,'AWSec2'));
      assert(aws.isSpecified);
      obj.awsec2 = aws; % awsec2 is specified and so .instanceID is immutable
    end
    
    function prepareBg(obj)
      % Typically you would deepcopy the reader before calling this
      assert(isscalar(obj));
      if ~isempty(obj.awsec2)
        obj.awsec2.clearStatusFuns();
      end
    end
    
    function  tf = getModelIsRemote(obj) %#ok<MANU> 
      tf = true;
    end
    
    function [maxiter,idx] = getMostRecentModel(obj,dmc,varargin)
      % maxiter is nan if something bad happened or if DNE
      
      % TODO allow polling for multiple models at once
      aws = obj.awsec2;
      [dirModelChainLnx,idx] = dmc.dirModelChainLnx(varargin{:});
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
    end
    
    function lsProjDir(obj,dmc)
      obj.awsec2.remoteLS(dmc.dirProjLnx);
    end
    function lsModelChainDir(obj,dmc)
      for i = 1:dmc.n,
        dir = dmc.dirModelChainLnx(i);
        dir = dir{1};
        obj.awsec2.remoteLS(dir);
      end
    end
    function lsTrkDir(obj,dmc)
       for i = 1:dmc.n,
        dir = dmc.dirTrkOutLnx(i);
        dir = dir{1};
        obj.awsec2.remoteLS(dir);
       end
    end
  end
end