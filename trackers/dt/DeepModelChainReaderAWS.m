classdef DeepModelChainReaderAWS < DeepModelChainReader
  
  properties 
    awsec2
  end
  
  methods
    
    function obj = DeepModelChainReaderAWS(backEnd)      
      assert(backEnd.type==DLBackEnd.AWS);
      assert(backEnd.awsec2.isSpecified);
      obj.awsec2 = backEnd.awsec2; % awsec2 is specified and so .instanceID is immutable
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
    
  end
end