classdef DeepModelChainReaderAWS < DeepModelChainReader
  
  properties 
    awsec2
  end
  
  methods
    
    function obj = DeepModelChainReaderAWS(aws)
      assert(isa(aws,'AWSec2'));
      assert(aws.isSpecified);
      obj.awsec2 = aws; % awsec2 is specified and so .instanceID is immutable
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
    
  end
end