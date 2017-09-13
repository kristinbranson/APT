classdef TargetSet < handle
  % A TargetSet represents a set of targets for a movie in an APT project. 
  % An explicit list of target indices can be generated given a labeler 
  % instance and movie index.
  
  enumeration
    AllTgts ('All targets'); % means "all targets in movie, without regard to frame"
    CurrTgt ('Current target') % means "current labeler target (index), which lives in the current movie; even if the movie in consideration is not the current movie"
  end
  
  properties
    prettyString
  end
  
  methods
    
    function obj = TargetSet(ps)
      obj.prettyString = ps;
    end
    
    function iTgts = getTargets(obj,labelerObj,iMov)
      % Get targets given movie
      %
      % iMov: scalar movie index (in general different from
      % labelerObj.currMovie)
      %
      % iTgts: vector of target indices. 
      
      assert(isscalar(iMov));
      switch obj
        case TargetSet.AllTgts
          % Multiview: assume first view is representative; trx elements
          % are supposed to match across views 
          tfaf = labelerObj.trxFilesAllFull{iMov,1}; % XXX GT MERGE
          nfrm = labelerObj.movieInfoAll{iMov,1}.nframes; % XXX GT MERGE
          trx = labelerObj.getTrx(tfaf,nfrm);
          ntrx = numel(trx);
          
          iTgts = 1:ntrx;          
          % iTgts represents all targest present in the movie, without
          % regard to frame. Not all targets are necessarily live for all
          % frames.
        case TargetSet.CurrTgt
          iTgts = labelerObj.currTarget; 
          % The current target is returned irrespective of iMov; iTgts may
          % not even apply to iMov.
        otherwise
          assert(false);
      end
    end
    
  end
end
    