classdef TargetSetVariable < TargetSet
  properties
    prettyString
    % fcn with sig iTgts = fcn(labeler,mIdx). 
    getTargetIndicesHook 
  end
  methods
    function obj = TargetSetVariable(ps,fcn)
      obj.prettyString = ps;
      obj.getTargetIndicesHook = fcn;
    end
    function str = getPrettyString(obj,lObj)
      str = obj.prettyString;
    end
    function iTgts = getTargetIndices(obj,lObj,mIdx)
      assert(isscalar(mIdx));
      if ~lObj.hasMovie
        iTgts = zeros(1,0);
      else
        iTgts = obj.getTargetIndicesHook(lObj,mIdx);
        iTgts = iTgts(:)';
      end
    end
  end
  
  properties (Constant) % canned/enumerated vals
    AllTgts = TargetSetVariable('All targets',@lclAllTargetsFcn);
    CurrTgt = TargetSetVariable('Current target',@lclCurrTargetFcn);
  end  
end

function iTgts = lclAllTargetsFcn(lObj,mIdx)
% Multiview: assume first view is representative; trx elements
% are supposed to match across views

assert(isscalar(mIdx));
nTrx = lObj.getnTrxMovIdx(mIdx);
iTgts = 1:nTrx;
% iTgts represents all targest present for mIdx, without regard to frame. 
% Not all targets are necessarily live for all frames.
end

function iTgts = lclCurrTargetFcn(lObj,mIdx)
assert(isscalar(mIdx));
ntrx = lObj.getnTrxMovIdx(mIdx);
iTgts = lObj.currTarget;
if iTgts>ntrx || iTgts==0
  iTgts = zeros(1,0);
end
end