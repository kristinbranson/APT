classdef TargetSetVariable < TargetSet
  properties
    prettyString
    getTargetIndicesHook % fcn with sig iTgts = fcn(labeler,iMov)
  end
  methods
    function obj = TargetSetVariable(ps,fcn)
      obj.prettyString = ps;
      obj.getTargetIndicesHook = fcn;
    end
    function str = getPrettyString(obj,lObj)
      str = obj.prettyString;
    end
    function iTgts = getTargetIndices(obj,lObj,iMov)
      if ~lObj.hasMovie
        iTgts = zeros(1,0);
      else
        iTgts = obj.getTargetIndicesHook(lObj,iMov);
        iTgts = iTgts(:)';
      end
    end
  end
  
  properties (Constant) % canned/enumerated vals
    AllTgts = TargetSetVariable('All targets',@lclAllTargetsFcn);
    CurrTgt = TargetSetVariable('Current target',@(lo,mov)lo.currTarget);
  end  
end

function iTgts = lclAllTargetsFcn(lObj,iMov)
% Multiview: assume first view is representative; trx elements
% are supposed to match across views
tfaf = lObj.trxFilesAllFull{iMov,1}; % XXX GT MERGE
nfrm = lObj.movieInfoAll{iMov,1}.nframes; % XXX GT MERGE
trx = lObj.getTrx(tfaf,nfrm);
ntrx = numel(trx);

iTgts = 1:ntrx;
% iTgts represents all targest present in the movie, without
% regard to frame. Not all targets are necessarily live for all
% frames.
end