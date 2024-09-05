classdef TargetSetVariable < TargetSet
  properties
    prettyString
    prettyCompactString
    % fcn with sig iTgts = fcn(labeler,mIdx). 
    getTargetIndicesHook 
  end
  methods
    function obj = TargetSetVariable(ps,pcs,fcn)
      obj.prettyString = ps;
      obj.prettyCompactString = pcs;
      obj.getTargetIndicesHook = fcn;
    end
    function str = getPrettyString(obj,labelerObj)
      assert(isstruct(labelerObj), 'labelerObj, despite the name, must be a struct') ;      
      str = obj.prettyString;
    end
    function str = getPrettyCompactString(obj,labelerObj)
      assert(isstruct(labelerObj), 'labelerObj, despite the name, must be a struct') ;      
      str = obj.prettyCompactString;
    end
    function iTgts = getTargetIndices(obj,lObj,mIdx)
      % mIdx: [n] vector of MovieIndices
      % iTgts: [n] cell array. iTgts{i} contains a vector of 1-based target
      %   indices for i'th mov      
      if ~lObj.hasMovie
        iTgts = cell(size(mIdx));
      else
        iTgts = obj.getTargetIndicesHook(lObj,mIdx);        
      end
    end
  end
  
  properties (Constant) % canned/enumerated vals
    AllTgts = TargetSetVariable('All targets','All targ',@lclAllTargetsFcn);
    CurrTgt = TargetSetVariable('Current target','Cur targ',@lclCurrTargetFcn);
  end  
end

function iTgts = lclAllTargetsFcn(lObj,mIdx)
% Multiview: assume first view is representative; trx elements
% are supposed to match across views

if lObj.maIsMA
  iTgts = arrayfun(@(x)Labels.uniqueTgts(lObj.getLabelsMovIdx(x)),mIdx,'uni',0);
else
  nTrxArr = lObj.getnTrxMovIdx(mIdx);
  iTgts = arrayfun(@(x)(1:x),nTrxArr,'uni',0);
  % iTgts represents all targets present for each el of mIdx, without regard 
  % to frame. Not all targets are necessarily live for all frames.
end
end

function iTgts = lclCurrTargetFcn(lObj,mIdx)
assert(isscalar(mIdx));
ntrx = lObj.getnTrxMovIdx(mIdx);
iTgts = lObj.currTarget;
if iTgts>ntrx || iTgts==0
  iTgts = zeros(1,0);
end
iTgts = {iTgts};
end