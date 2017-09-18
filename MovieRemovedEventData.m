classdef MovieRemovedEventData < event.EventData
  properties
    % containers.Map. iMovOrig2New(iMovOrig) gives iMovNew, the new/updated 
    % (after movie removal) movie index for iMovOrig. If movie iMovOrig is 
    % no longer present, iMovNew will equal 0.
    iMovOrig2New 
  end
  methods
    function obj = MovieRemovedEventData(mIdx,nMovOrigReg,...
                                         nMovOrigGT)
      % mIdx: scalar MovieIndex
      % nMovOrigReg: original number of regular movies
      % nMovOrigGT: " GT movies
      
      assert(isscalar(mIdx) && isa(mIdx,'MovieIndex'));
      assert(nMovOrigReg>0);
      assert(nMovOrigGT>0);
      
      mIdx = int32(mIdx);
      origIdxs = [1:nMovOrigReg -1:-1:-nMovOrigGT];
      if mIdx>0
        newIdxs = [1:mIdx-1 0 mIdx:nMovOrigReg-1 ...
                  -1:-1:-nMovOrigGT];
      else
        newIdxs = [1:nMovOrigReg ...
                   -1:-1:mIdx+1 0 mIdx:-1:-nMovOrigGT+1];
      end
      obj.iMovOrig2New = containers.Map(origIdxs,newIdxs);
    end
  end
end