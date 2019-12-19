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
    
    function  tf = getModelIsRemote(obj)
      tf = true;
    end
    
    function maxiter = getMostRecentModel(obj,dmc)
      % maxiter is nan if something bad happened or if DNE
      
      aws = obj.awsec2;
      fspollargs = {'mostrecentmodel' dmc.dirModelChainLnx};
      [tfsucc,res] = aws.remoteCallFSPoll(fspollargs);
      if tfsucc
        maxiter = str2double(res{1}); % includes 'DNE'->nan
      else
        maxiter = nan;
      end
    end
    
    function lsProjDir(obj,dmc)
      obj.awsec2.remoteLS(dmc.dirProjLnx);
    end
    function lsModelChainDir(obj,dmc)
      obj.awsec2.remoteLS(dmc.dirModelChainLnx);
    end
    function lsTrkDir(obj,dmc)
      obj.awsec2.remoteLS(dmc.dirTrkOutLnx);
    end
  end
end